"""Aggregate all completed conditions into portable JSON/CSV/Markdown."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from .protocol import SIZE_SPECS, VARIANTS, load_json, save_json


def _rows(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output_root / "runs").glob("*/*/*/result.json")):
        payload = load_json(path)
        metrics = payload["metrics"]
        rows.append(
            {
                "size": payload["size"],
                "train_dataset": payload["train_dataset"],
                "variant": payload["variant"],
                "method": payload["variant_label"],
                "wikitext_ppl": metrics.get("wikitext", {}).get("score"),
                "piqa_acc_norm": metrics.get("piqa", {}).get("score"),
                "trainable_params": payload["parameters"]["trainable_params"],
                "total_params": payload["parameters"]["total_params"],
                "adapter_rank": payload["adapter_rank"],
                "selected_experts": payload["selected_experts"],
                "num_groups": payload["num_groups"],
                "result_path": str(path),
            }
        )
    return rows


def _fmt(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "--"
    if percent:
        return f"{100 * float(value):.2f}%"
    return f"{float(value):.2f}"


def render(output_root: Path) -> dict[str, Path]:
    rows = _rows(output_root)
    report_dir = output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(report_dir / "results.json", {"schema_version": 1, "rows": rows})

    columns = (
        "size", "train_dataset", "variant", "method", "wikitext_ppl",
        "piqa_acc_norm", "trainable_params", "total_params", "adapter_rank",
        "selected_experts", "num_groups", "result_path",
    )
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    (report_dir / "results.csv").write_text(stream.getvalue(), encoding="utf-8")

    lines = [
        "# SPREAD router/LoRA comparison",
        "",
        "All adaptation rows are teacher-free. WikiText reports perplexity (lower is better); "
        "PIQA reports normalized accuracy (higher is better). Frozen baselines are listed under "
        "`none` because they do not consume a training dataset.",
        "",
    ]
    for size in SIZE_SPECS:
        selected = [row for row in rows if row["size"] == size]
        if not selected:
            continue
        lines.extend(
            [
                f"## {size.upper()}",
                "",
                "| Train data | Method | WikiText PPL ↓ | PIQA AccNorm ↑ | Trainable | Total | C→K |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        order = {key: index for index, key in enumerate(VARIANTS)}
        for row in sorted(selected, key=lambda item: (item["train_dataset"], order[item["variant"]])):
            lines.append(
                "| {train_dataset} | {method} | {ppl} | {piqa} | {trainable:,} | "
                "{total:,} | {selected_experts}→{num_groups} |".format(
                    **row,
                    ppl=_fmt(row["wikitext_ppl"]),
                    piqa=_fmt(row["piqa_acc_norm"], percent=True),
                    trainable=row["trainable_params"],
                    total=row["total_params"],
                )
            )
        lines.append("")
    (report_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": report_dir / "results.json",
        "csv": report_dir / "results.csv",
        "markdown": report_dir / "results.md",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in render(args.output_root.resolve()).values():
        print(path)
