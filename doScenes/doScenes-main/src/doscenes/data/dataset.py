from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset

try:
    from pyquaternion import Quaternion
except Exception:
    Quaternion = None


@dataclass
class DoScenesRecord:
    scene_number: int
    scene_name: str
    scene_token: str
    instruction: str
    instruction_type: str
    annotator_file: str


def load_paths(path_file: str = "paths.txt") -> tuple[str, str]:
    config: dict[str, str] = {}
    with open(path_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            config[key.strip()] = val.strip()

    if "NUSCENES_ROOT" not in config:
        raise ValueError("Missing NUSCENES_ROOT in paths.txt")
    if "DOSCENES_ANNOTATIONS" not in config:
        raise ValueError("Missing DOSCENES_ANNOTATIONS in paths.txt")

    return config["NUSCENES_ROOT"], config["DOSCENES_ANNOTATIONS"]


def split_indices_by_scene_hash(window_index: list[tuple[int, int]], scene_names: list[str], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")

    groups: dict[str, list[int]] = {}
    for global_idx, (record_idx, _) in enumerate(window_index):
        scene_name = scene_names[record_idx]
        groups.setdefault(scene_name, []).append(global_idx)

    scene_keys = sorted(groups)
    ordered = sorted(scene_keys, key=lambda x: hashlib.md5(f"{x}-{seed}".encode("utf-8")).hexdigest())
    val_count = max(1, int(len(ordered) * val_ratio))
    val_scene = set(ordered[:val_count])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for scene, indices in groups.items():
        if scene in val_scene:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)
    return train_idx, val_idx


def _scene_name_from_number(scene_number: int) -> str:
    return f"scene-{int(scene_number):04d}"


def _yaw_from_wxyz(rotation_wxyz: list[float]) -> float:
    if Quaternion is None:
        return 0.0
    q = Quaternion(rotation_wxyz)
    w, x, y, z = q.elements
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class DoScenesDataset(Dataset):
    def __init__(
        self,
        nusc: Any,
        annotations: str,
        hist_sec: float,
        fut_sec: float,
        sample_freq: float,
        relative_coords: bool,
        align_heading: bool,
        coord_scale: float,
        window_stride: int,
        keep_empty_instruction: bool = True,
    ) -> None:
        self.nusc = nusc
        self.hist_frames = int(hist_sec * sample_freq) + 1
        self.fut_frames = int(fut_sec * sample_freq)
        self.relative_coords = relative_coords
        self.align_heading = align_heading
        self.coord_scale = coord_scale
        self.window_stride = window_stride
        self.keep_empty_instruction = keep_empty_instruction

        self.scene_by_name = {scene["name"]: scene for scene in self.nusc.scene}
        self.records = self._load_records(annotations)
        self._scene_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.window_index = self._build_window_index()

    def _load_records(self, annotations: str) -> list[DoScenesRecord]:
        ann_path = Path(annotations)
        files = sorted(ann_path.glob("*.csv")) if ann_path.is_dir() else [ann_path]
        if not files:
            raise FileNotFoundError("No annotation csv files found")

        records: list[DoScenesRecord] = []
        for csv_path in files:
            df = pd.read_csv(csv_path)
            rename_map = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
            df = df.rename(columns=rename_map)
            for row in df.itertuples(index=False):
                try:
                    scene_number = int(getattr(row, "scene_number"))
                except Exception:
                    continue
                raw_instruction = getattr(row, "instruction", "")
                instruction = "" if pd.isna(raw_instruction) else str(raw_instruction).strip()
                if (not instruction) and (not self.keep_empty_instruction):
                    continue
                scene_name = _scene_name_from_number(scene_number)
                scene = self.scene_by_name.get(scene_name)
                if scene is None:
                    continue
                records.append(
                    DoScenesRecord(
                        scene_number=scene_number,
                        scene_name=scene_name,
                        scene_token=scene["token"],
                        instruction=instruction,
                        instruction_type=str(getattr(row, "instruction_type", "")).strip(),
                        annotator_file=csv_path.name,
                    )
                )
        if not records:
            raise ValueError("No valid annotation records loaded")
        return records

    def _extract_scene(self, scene_token: str) -> tuple[torch.Tensor, torch.Tensor]:
        if scene_token in self._scene_cache:
            return self._scene_cache[scene_token]

        scene = self.nusc.get("scene", scene_token)
        token = scene["first_sample_token"]
        xy: list[list[float]] = []
        yaws: list[float] = []
        while token:
            sample = self.nusc.get("sample", token)
            sd_token = sample["data"].get("LIDAR_TOP") or sample["data"].get("CAM_FRONT")
            sd = self.nusc.get("sample_data", sd_token)
            ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
            xy.append([float(ego_pose["translation"][0]), float(ego_pose["translation"][1])])
            yaws.append(_yaw_from_wxyz(ego_pose["rotation"]))
            token = sample["next"]

        xy_t = torch.tensor(xy, dtype=torch.float32)
        yaw_t = torch.tensor(yaws, dtype=torch.float32)
        self._scene_cache[scene_token] = (xy_t, yaw_t)
        return xy_t, yaw_t

    def _build_window_index(self) -> list[tuple[int, int]]:
        total_required = self.hist_frames + self.fut_frames
        index: list[tuple[int, int]] = []
        for ridx, record in enumerate(self.records):
            xy, _ = self._extract_scene(record.scene_token)
            seq_len = len(xy)
            if seq_len < total_required:
                index.append((ridx, 0))
                continue
            max_start = seq_len - total_required
            for start in range(0, max_start + 1, self.window_stride):
                index.append((ridx, start))
            if max_start % self.window_stride != 0:
                index.append((ridx, max_start))
        if not index:
            raise ValueError("No valid window samples built")
        return index

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ridx, start = self.window_index[idx]
        record = self.records[ridx]
        xy, yaws = self._extract_scene(record.scene_token)

        total_required = self.hist_frames + self.fut_frames
        if len(xy) < total_required:
            pad = total_required - len(xy)
            xy = torch.cat([xy, xy[-1:].repeat(pad, 1)], dim=0)
            yaws = torch.cat([yaws, yaws[-1:].repeat(pad)], dim=0)
            start = 0

        hs = start
        he = hs + self.hist_frames
        fe = he + self.fut_frames

        history_xy = xy[hs:he].clone()
        future_xy = xy[he:fe].clone()
        heading = float(yaws[he - 1].item())
        origin = history_xy[-1].clone()

        if self.relative_coords:
            history_xy -= origin
            future_xy -= origin
            if self.align_heading:
                cos_y = math.cos(-heading)
                sin_y = math.sin(-heading)
                rot = torch.tensor([[cos_y, -sin_y], [sin_y, cos_y]], dtype=torch.float32)
                history_xy = history_xy @ rot.T
                future_xy = future_xy @ rot.T

        history_xy /= self.coord_scale
        future_xy /= self.coord_scale

        instruction_hash = hashlib.md5(record.instruction.encode("utf-8")).hexdigest()[:8]
        sample_id = f"{record.scene_name}_{hs}_{instruction_hash}"

        return {
            "sample_id": sample_id,
            "scene_token": record.scene_token,
            "instruction": record.instruction,
            "has_instruction": bool(record.instruction),
            "history_xy": history_xy,
            "future_xy_gt": future_xy,
            "origin": origin,
            "heading": heading,
            "coord_scale": self.coord_scale,
            "relative_coords": self.relative_coords,
            "align_heading": self.align_heading,
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [x["sample_id"] for x in batch],
        "scene_token": [x["scene_token"] for x in batch],
        "instruction": [x["instruction"] for x in batch],
        "has_instruction": torch.tensor([1 if x["has_instruction"] else 0 for x in batch], dtype=torch.long),
        "history_xy": torch.stack([x["history_xy"] for x in batch], dim=0),
        "future_xy_gt": torch.stack([x["future_xy_gt"] for x in batch], dim=0),
        "origin": torch.stack([x["origin"] for x in batch], dim=0),
        "heading": torch.tensor([x["heading"] for x in batch], dtype=torch.float32),
        "coord_scale": float(batch[0]["coord_scale"]),
        "relative_coords": bool(batch[0]["relative_coords"]),
        "align_heading": bool(batch[0]["align_heading"]),
    }


def build_nuscenes_dataset(path_file: str, **kwargs: Any) -> DoScenesDataset:
    nusc_root, annotations = load_paths(path_file)
    nusc = NuScenes(version="v1.0-trainval", dataroot=nusc_root, verbose=False)
    return DoScenesDataset(nusc=nusc, annotations=annotations, **kwargs)
