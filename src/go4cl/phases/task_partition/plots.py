"""The four task-partition figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASK_COLOR = {"A": "#0072B2", "B": "#E69F00", "C": "#009E73"}
PROTOCOL_STYLE = {
    "ab_then_c": ("-", "AB joint → C"),
    "abc_joint": ("--", "ABC joint"),
    "ab_continued": (":", "AB joint continued"),
}
PROTOCOL_ORDER = ("ab_then_c", "abc_joint", "ab_continued")
_MIDDLE_PREFERENCE = (10_000, 5000, 20_000, 1000)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            xs.append(int(row["step"]))
            ys.append(float(value))
    return xs, ys


def plot_accuracy_curves(curves: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for protocol in PROTOCOL_ORDER:
        rows = curves["protocols"].get(protocol, [])
        style, protocol_label = PROTOCOL_STYLE[protocol]
        for task in ("A", "B", "C"):
            xs, ys = _finite_series(rows, f"{task}_test_acc")
            if not xs:
                continue
            ax.plot(
                xs,
                ys,
                linestyle=style,
                color=TASK_COLOR[task],
                linewidth=1.8,
                label=f"{protocol_label} / {task}",
            )
    switch = curves.get("switch_step")
    if isinstance(switch, int):
        ax.axvline(switch, color="#444444", linewidth=1.0, linestyle="-.")
        ax.text(
            switch,
            0.98,
            " AB→C switch",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#444444",
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("test accuracy")
    ax.set_title("Task-partition test accuracy")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_final_bars(summary: dict[str, Any], path: Path) -> None:
    finals = summary["final_test_accuracy"]
    labels = [PROTOCOL_STYLE[name][1] for name in PROTOCOL_ORDER]
    x = np.arange(len(PROTOCOL_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for index, task in enumerate(("A", "B", "C")):
        values = [float(finals[name][task]) for name in PROTOCOL_ORDER]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=task,
            color=TASK_COLOR[task],
        )
    ax.axhline(0.90, color="#666666", linewidth=0.8, linestyle="--", label="0.90")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Final A/B/C accuracy")
    ax.legend(frameon=False, ncol=4)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _matrix_array(record: dict[str, Any]) -> np.ndarray:
    matrix = record["counterfactual"]["matrix"]
    array = np.asarray(matrix, dtype=float)
    return array


def select_counterfactual_panels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """AB pretrain, one C midpoint, and C final."""
    by_role = {str(record.get("role")): record for record in records}
    chosen: list[dict[str, Any]] = []
    if "ab_pretrain" in by_role:
        chosen.append(by_role["ab_pretrain"])
    final = by_role.get("c_final")
    final_step = None if final is None else final.get("routing_step")
    mids = [
        record
        for record in records
        if record.get("routing_step") not in (None, 0, final_step)
        and str(record.get("role", "")).startswith("c_")
    ]
    middle = None
    for prefer in _MIDDLE_PREFERENCE:
        for record in mids:
            if int(record["routing_step"]) == prefer:
                middle = record
                break
        if middle is not None:
            break
    if middle is None and mids and final_step is not None:
        target = float(final_step) / 2.0
        middle = min(mids, key=lambda record: abs(float(record["routing_step"]) - target))
    if middle is not None:
        chosen.append(middle)
    if final is not None:
        chosen.append(final)
    return chosen


def plot_counterfactual(records: list[dict[str, Any]], path: Path) -> None:
    panels = select_counterfactual_panels(records)
    if not panels:
        raise RuntimeError("no counterfactual records to plot")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0), squeeze=False)
    names = panels[0]["counterfactual"]["task_names"]
    for ax, record in zip(axes[0], panels, strict=True):
        matrix = _matrix_array(record)
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(names)), [f"label {name}" for name in names])
        ax.set_yticks(range(len(names)), [f"TASK_{name}" for name in names])
        ax.set_title(str(record.get("title") or record.get("tag")))
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text = "" if not np.isfinite(value) else f"{value:.2f}"
                color = "white" if np.isfinite(value) and value < 0.55 else "black"
                ax.text(col, row, text, ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Task-token counterfactual  M[u, v]", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_routing(records: list[dict[str, Any]], path: Path) -> None:
    points = [
        record
        for record in records
        if isinstance(record.get("routing_step"), int) and "routing" in record
    ]
    points.sort(key=lambda record: int(record["routing_step"]))
    # One point per continuation step. Later aliases (first-stable copy) lose.
    dedup: dict[int, dict[str, Any]] = {}
    for record in points:
        dedup.setdefault(int(record["routing_step"]), record)
    ordered = [dedup[step] for step in sorted(dedup)]
    if not ordered:
        raise RuntimeError("no routing records to plot")
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True)
    for task in ("A", "B", "C"):
        xs = [int(record["routing_step"]) for record in ordered]
        task_mass = [float(record["routing"][task]["task_mass"]) for record in ordered]
        operand_mass = [float(record["routing"][task]["operand_mass"]) for record in ordered]
        axes[0].plot(xs, task_mass, marker="o", color=TASK_COLOR[task], label=task)
        axes[1].plot(xs, operand_mass, marker="o", color=TASK_COLOR[task], label=task)
    axes[0].set_ylabel("attention to task token")
    axes[1].set_ylabel("attention to correct operands")
    axes[1].set_xlabel("C-continuation step (0 = AB checkpoint)")
    axes[0].set_title("Query routing during AB → C")
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_figures(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    curves = _load(run_dir / "curves.json")
    summary = _load(run_dir / "summary.json")
    index = _load(run_dir / "mechanism_index.json")
    records = [_load(run_dir / item["file"]) for item in index]
    outputs = {
        "accuracy_curves.png": lambda path: plot_accuracy_curves(curves, path),
        "final_accuracy_comparison.png": lambda path: plot_final_bars(summary, path),
        "task_token_counterfactual.png": lambda path: plot_counterfactual(records, path),
        "routing_dynamics.png": lambda path: plot_routing(records, path),
    }
    written: list[Path] = []
    for name, drawer in outputs.items():
        path = figure_dir / name
        drawer(path)
        written.append(path)
    return written
