#!/usr/bin/env python3
"""Build a Model C label inventory and deployment-readiness support report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml_new.model_c.labels import (  # noqa: E402
    HOP_S,
    REPORT_ONLY_TECHNIQUES,
    SELECTED_USER_FACING_TECHNIQUES,
    TECHNIQUES,
    read_gtsinger_entries,
)


def _counter(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return dict(Counter((row.get(key) or "missing") for row in rows))


def _nested_counter(rows: list[dict[str, str]], *keys: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        cursor = out
        for key in keys[:-1]:
            cursor = cursor.setdefault(row.get(key) or "missing", {})
        leaf = row.get(keys[-1]) or "missing"
        cursor[leaf] = int(cursor.get(leaf, 0)) + 1
    return out


def _path_missing(value: str | None, root: Path) -> bool:
    if not value:
        return True
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return not path.exists()


def _entry_duration(entry: dict[str, Any]) -> float:
    starts = entry.get("ph_start", [])
    ends = entry.get("ph_end", [])
    total = 0.0
    for idx, end_value in enumerate(ends):
        if idx >= len(starts):
            continue
        try:
            total += max(0.0, float(end_value) - float(starts[idx]))
        except (TypeError, ValueError):
            continue
    return total


def _positive_duration_for_entry(entry: dict[str, Any], technique: str) -> float:
    values = entry.get(technique, [])
    starts = entry.get("ph_start", [])
    ends = entry.get("ph_end", [])
    total = 0.0
    for idx, value in enumerate(values):
        if idx >= len(starts) or idx >= len(ends):
            continue
        if str(value).strip() != "1":
            continue
        try:
            total += max(0.0, float(ends[idx]) - float(starts[idx]))
        except (TypeError, ValueError):
            continue
    return total


def _json_label_support(rows: list[dict[str, str]], root: Path) -> dict[str, Any]:
    support: dict[str, Any] = {
        tech: {
            "positive_duration_s": 0.0,
            "negative_duration_s": 0.0,
            "positive_files": 0,
            "negative_files": 0,
            "by_split": defaultdict(lambda: {"positive_duration_s": 0.0, "negative_duration_s": 0.0, "files": 0}),
            "by_singer": defaultdict(lambda: {"positive_duration_s": 0.0, "negative_duration_s": 0.0, "files": 0}),
        }
        for tech in TECHNIQUES
    }
    seen_json: set[Path] = set()
    missing_json = 0
    parse_errors: list[dict[str, str]] = []
    for row in rows:
        json_value = row.get("json_path")
        if not json_value:
            missing_json += 1
            continue
        json_path = Path(json_value)
        if not json_path.is_absolute():
            json_path = root / json_path
        if json_path in seen_json:
            continue
        seen_json.add(json_path)
        if not json_path.exists():
            missing_json += 1
            continue
        try:
            entries = read_gtsinger_entries(json_path)
        except Exception as exc:  # pragma: no cover - exact parser errors are data dependent
            parse_errors.append({"json_path": str(json_path), "error": str(exc)})
            continue
        duration = sum(_entry_duration(entry) for entry in entries)
        split = row.get("split") or "missing"
        singer = row.get("singer_id") or "missing"
        for tech in TECHNIQUES:
            pos = sum(_positive_duration_for_entry(entry, tech) for entry in entries)
            neg = max(0.0, duration - pos)
            support[tech]["positive_duration_s"] += pos
            support[tech]["negative_duration_s"] += neg
            support[tech]["positive_files"] += int(pos > 0.0)
            support[tech]["negative_files"] += int(neg > 0.0)
            support[tech]["by_split"][split]["positive_duration_s"] += pos
            support[tech]["by_split"][split]["negative_duration_s"] += neg
            support[tech]["by_split"][split]["files"] += 1
            support[tech]["by_singer"][singer]["positive_duration_s"] += pos
            support[tech]["by_singer"][singer]["negative_duration_s"] += neg
            support[tech]["by_singer"][singer]["files"] += 1
    clean: dict[str, Any] = {}
    for tech, value in support.items():
        pos = float(value["positive_duration_s"])
        neg = float(value["negative_duration_s"])
        total = pos + neg
        clean[tech] = {
            "role": "selected_user_facing" if tech in SELECTED_USER_FACING_TECHNIQUES else "report_only_or_hidden",
            "positive_duration_s": round(pos, 3),
            "negative_duration_s": round(neg, 3),
            "positive_files": int(value["positive_files"]),
            "negative_files": int(value["negative_files"]),
            "positive_rate_by_duration": round(pos / total, 6) if total else None,
            "majority_baseline_by_duration": round(max(pos, neg) / total, 6) if total else None,
            "by_split": {
                key: {
                    "positive_duration_s": round(float(item["positive_duration_s"]), 3),
                    "negative_duration_s": round(float(item["negative_duration_s"]), 3),
                    "files": int(item["files"]),
                }
                for key, item in sorted(value["by_split"].items())
            },
            "by_singer": {
                key: {
                    "positive_duration_s": round(float(item["positive_duration_s"]), 3),
                    "negative_duration_s": round(float(item["negative_duration_s"]), 3),
                    "files": int(item["files"]),
                }
                for key, item in sorted(value["by_singer"].items())
            },
        }
    return {
        "techniques": clean,
        "json_files_seen": len(seen_json),
        "missing_json_rows": missing_json,
        "parse_errors": parse_errors[:20],
    }


def build_inventory(manifest: Path, *, root: Path | None = None) -> dict[str, Any]:
    root = root or _ROOT
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    n_frames = []
    for row in rows:
        try:
            n_frames.append(int(float(row.get("n_frames") or 0)))
        except (TypeError, ValueError):
            n_frames.append(0)
    missing = {
        "npz_path": sum(_path_missing(row.get("npz_path"), root) for row in rows),
        "audio_path": sum(_path_missing(row.get("audio_path"), root) for row in rows),
        "json_path": sum(_path_missing(row.get("json_path"), root) for row in rows),
        "nanopitch_npz": sum(_path_missing(row.get("nanopitch_npz"), root) for row in rows),
    }
    return {
        "schema_version": "label_inventory.v1",
        "manifest": str(manifest),
        "rows": len(rows),
        "duration_hours_from_manifest": round(float(sum(n_frames) * HOP_S / 3600.0), 3),
        "selected_user_facing_labels": list(SELECTED_USER_FACING_TECHNIQUES),
        "report_only_labels": list(REPORT_ONLY_TECHNIQUES),
        "counts": {
            "dataset": _counter(rows, "dataset"),
            "split": _counter(rows, "split"),
            "singer": _counter(rows, "singer_id"),
            "source_technique": _counter(rows, "source_technique"),
            "supervision": _counter(rows, "supervision"),
            "dataset_by_split": _nested_counter(rows, "dataset", "split"),
            "split_by_source_technique": _nested_counter(rows, "split", "source_technique"),
        },
        "missing_files": missing,
        "label_support": _json_label_support(rows, root),
        "deployment_gate": {
            "f1_min": 0.85,
            "precision_min": 0.90,
            "recall_min": 0.80,
            "false_positive_rate_max": 0.05,
            "dominant_prediction_share_max": 0.60,
            "split_requirement": "test_alto1 + test_tenor1 + test_vocalset held out",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("ml_new/model_c/manifests/model_c.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/label_inventory/model_c_label_inventory.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_inventory(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
