#!/usr/bin/env python3
"""Plot reproducible real-OFT rollout timing and GPU-monitor artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


FILE_RE = re.compile(r"(?P<variant>baseline|optimized)_n(?P<num_envs>\d+)_gpu\.json$")


def load_monitor(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_result(path: Path) -> dict | None:
    log_path = path.with_name(path.name.replace("_gpu.json", ".log"))
    if not log_path.is_file():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def collect(results_dir: Path) -> dict[int, dict[str, tuple[dict, dict | None]]]:
    grouped: dict[int, dict[str, tuple[dict, dict | None]]] = {}
    for path in results_dir.glob("*_n*_gpu.json"):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        num_envs = int(match.group("num_envs"))
        grouped.setdefault(num_envs, {})[match.group("variant")] = (
            load_monitor(path),
            load_result(path),
        )
    return grouped


def plot_gpu_curves(grouped: dict, output: Path) -> None:
    complete = [
        n
        for n, variants in sorted(grouped.items())
        if {"baseline", "optimized"} <= set(variants)
    ]
    if not complete:
        return
    fig, axes = plt.subplots(len(complete), 2, figsize=(12, 3.3 * len(complete)), squeeze=False)
    colors = {"baseline": "#777777", "optimized": "#1976d2"}
    labels = {"baseline": "Baseline (pre-optimization)", "optimized": "Optimized"}
    for row, num_envs in enumerate(complete):
        for variant in ("baseline", "optimized"):
            monitor, _ = grouped[num_envs][variant]
            samples = monitor.get("samples", [])
            if not samples:
                continue
            elapsed = [sample["elapsed_s"] for sample in samples]
            util = [sample["gpu_util_percent"] for sample in samples]
            memory_gib = [sample["gpu_memory_mib"] / 1024 for sample in samples]
            axes[row, 0].plot(elapsed, util, color=colors[variant], label=labels[variant])
            axes[row, 1].plot(elapsed, memory_gib, color=colors[variant], label=labels[variant])
        axes[row, 0].set_title(f"N={num_envs}: GPU utilization")
        axes[row, 0].set_ylabel("utilization (%)")
        axes[row, 0].set_ylim(0, 100)
        axes[row, 1].set_title(f"N={num_envs}: GPU memory")
        axes[row, 1].set_ylabel("memory (GiB)")
        for column in range(2):
            axes[row, column].set_xlabel("measured rollout time (s)")
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend(loc="best")
    fig.suptitle("Real OFT rollout: GPU monitoring", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(grouped: dict, output_csv: Path, output_plot: Path) -> None:
    rows = []
    for num_envs, variants in sorted(grouped.items()):
        for variant, (_, result) in variants.items():
            if result is None or "end_to_end_rollout_epoch_s" not in result:
                continue
            elapsed = result["end_to_end_rollout_epoch_s"]
            rows.append(
                {
                    "variant": variant,
                    "num_envs": num_envs,
                    "epoch_s": elapsed,
                    "physical_steps_per_s": num_envs * result["action_chunks"] * 25 / elapsed,
                    "gpu_peak_gib": result["gpu_memory_peak_mib"] / 1024,
                    "gpu_util_mean_percent": result.get("gpu_util_mean_percent", ""),
                }
            )
    if not rows:
        return
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    colors = {"baseline": "#777777", "optimized": "#1976d2"}
    titles_and_keys = [
        ("Full rollout time", "epoch_s", "time (s)"),
        ("Physical-control throughput", "physical_steps_per_s", "steps/s"),
        ("Peak GPU memory", "gpu_peak_gib", "GiB"),
    ]
    for axis, (title, key, ylabel) in zip(axes, titles_and_keys):
        for variant in ("baseline", "optimized"):
            subset = [row for row in rows if row["variant"] == variant]
            if subset:
                axis.plot(
                    [row["num_envs"] for row in subset],
                    [row[key] for row in subset],
                    "o-",
                    color=colors[variant],
                    label="Baseline (pre-optimization)" if variant == "baseline" else "Optimized",
                )
        axis.set_title(title)
        axis.set_xlabel("envs / EnvWorker")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    fig.suptitle("Real OFT rollout summary (excluding construction and model load)", y=1.03)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = collect(args.results_dir)
    plot_gpu_curves(grouped, args.output_dir / "gpu_utils_baseline_vs_optimized.png")
    write_summary(
        grouped,
        args.output_dir / "rollout_epoch_summary.csv",
        args.output_dir / "rollout_epoch_comparison.png",
    )


if __name__ == "__main__":
    main()
