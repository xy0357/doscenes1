from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_annotation_frames(annotation_dir: Path) -> pd.DataFrame:
    files = sorted(annotation_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv found in {annotation_dir}")

    frames = []
    for p in files:
        df = pd.read_csv(p)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    annotation_dir = root / "Annotations"
    out_dir = root / "artifacts" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotation_frames(annotation_dir)

    if "instruction" not in df.columns:
        raise ValueError("Column 'instruction' not found in annotation csv files")

    raw_rows = len(df)
    empty_instruction_mask = df["instruction"].isna() | (df["instruction"].astype(str).str.strip() == "")
    empty_instruction_rows = int(empty_instruction_mask.sum())
    non_empty_instruction_rows = int(raw_rows - empty_instruction_rows)

    scene_col = "scene_number" if "scene_number" in df.columns else None
    unique_scene_count = int(df[scene_col].nunique()) if scene_col else None

    profile = {
        "raw_rows": raw_rows,
        "empty_instruction_rows": empty_instruction_rows,
        "non_empty_instruction_rows": non_empty_instruction_rows,
        "empty_instruction_ratio": (empty_instruction_rows / raw_rows) if raw_rows else 0.0,
        "unique_scene_count": unique_scene_count,
    }

    (out_dir / "data_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    if scene_col:
        scene_stats = (
            df.groupby(scene_col, dropna=False)
            .size()
            .reset_index(name="row_count")
            .sort_values("row_count", ascending=False)
        )
        scene_stats.to_csv(out_dir / "data_profile_scene_counts.csv", index=False, encoding="utf-8")

    source_stats = (
        df.assign(is_empty_instruction=empty_instruction_mask)
        .groupby("source_file", dropna=False)["is_empty_instruction"]
        .agg(["count", "sum"])
        .reset_index()
        .rename(columns={"count": "rows", "sum": "empty_instruction_rows"})
    )
    source_stats["empty_instruction_ratio"] = source_stats["empty_instruction_rows"] / source_stats["rows"]
    source_stats.to_csv(out_dir / "data_profile_source_counts.csv", index=False, encoding="utf-8")

    print("Saved:")
    print(out_dir / "data_profile.json")
    if scene_col:
        print(out_dir / "data_profile_scene_counts.csv")
    print(out_dir / "data_profile_source_counts.csv")


if __name__ == "__main__":
    main()
