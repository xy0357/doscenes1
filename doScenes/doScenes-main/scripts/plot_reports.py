from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def plot_train_report(train_report: dict, out_png: Path) -> None:
    history = train_report.get("history", [])
    if not history:
        raise ValueError("train_report history is empty")

    epochs = [int(x["epoch"]) for x in history]
    train_ade = [float(x["train_ade"]) for x in history]
    val_ade = [float(x["val_ade"]) for x in history]
    gap_ade = [float(x["gap_ade"]) for x in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_ade, label="train_ADE", marker="o")
    axes[0].plot(epochs, val_ade, label="val_ADE", marker="o")
    axes[0].set_title("ADE Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ADE")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, gap_ade, label="gap_ADE (val-train)", marker="o", color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Overfit Gap")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Gap ADE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_delta_report(delta_report: dict, out_png: Path) -> None:
    baseline = delta_report.get("baseline", {})
    instruction = delta_report.get("instruction", {})

    labels = ["ADE", "FDE"]
    base_vals = [float(baseline.get("ade", 0.0)), float(baseline.get("fde", 0.0))]
    inst_vals = [float(instruction.get("ade", 0.0)), float(instruction.get("fde", 0.0))]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width / 2 for i in x], base_vals, width=width, label="baseline")
    ax.bar([i + width / 2 for i in x], inst_vals, width=width, label="instruction")
    ax.set_xticks(list(x), labels)
    ax.set_title(
        f"Delta ADE={float(delta_report.get('delta_ade', 0.0)):.4f}, "
        f"Delta FDE={float(delta_report.get('delta_fde', 0.0)):.4f}"
    )
    ax.set_ylabel("Error (lower is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp_dir = root / "artifacts" / "experiments"
    vis_dir = root / "artifacts" / "visualizations"

    train_report_path = exp_dir / "train_report.json"
    delta_report_path = exp_dir / "delta_report.json"

    if train_report_path.exists():
        train_report = load_json(train_report_path)
        out1 = vis_dir / "train_ade_gap.png"
        plot_train_report(train_report, out1)
        print(f"Saved {out1}")
    else:
        print(f"Skip train plot, not found: {train_report_path}")

    if delta_report_path.exists():
        delta_report = load_json(delta_report_path)
        out2 = vis_dir / "delta_baseline_vs_instruction.png"
        plot_delta_report(delta_report, out2)
        print(f"Saved {out2}")
    else:
        print(f"Skip delta plot, not found: {delta_report_path}")


if __name__ == "__main__":
    main()
