#!/usr/bin/env python3
"""
Convert unified radiology tester JSON progress files into a CSV summary.

Each row in the CSV corresponds to a unique `base_id` (image name) and contains
the classification value reported for every test type listed in the
`metadata.test_types` section of the JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Dict, List, Optional

CATEGORY_ORDER = ["normal", "meningioma", "glioma", "pituitary"]


def load_json(json_path: Path) -> Dict:
    """Load JSON data from disk."""
    with json_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_rows(data: Dict) -> Dict[str, Dict[str, Optional[int]]]:
    """
    Build a nested dict keyed by base_id with inner dicts keyed by test_type.

    Returns:
        Dict mapping base_id -> {test_type: classification}
    """
    metadata = data.get("metadata", {})
    test_types: List[str] = metadata.get("test_types", [])
    # Fall back to discovering unique test types from results if metadata missing
    if not test_types:
        test_types = sorted(
            {entry.get("test_type") for entry in data.get("results", []) if entry.get("test_type")}
        )

    rows: Dict[str, Dict[str, Optional[int]]] = {}
    for entry in data.get("results", []):
        base_id = entry.get("base_id")
        test_type = entry.get("test_type")
        classification = entry.get("classification")
        if not base_id or not test_type:
            continue

        if base_id not in rows:
            rows[base_id] = {test: None for test in test_types}

        # Keep the latest classification seen for that test_type/base_id
        rows[base_id][test_type] = classification

    return rows


def write_csv(rows: Dict[str, Dict[str, Optional[int]]], test_types: List[str], output_path: Path) -> None:
    """Write the aggregated rows into a CSV file."""
    fieldnames = ["image_name"] + test_types
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for base_id in sorted(rows.keys(), key=_sort_key_for_base_id):
            row = {"image_name": base_id}
            row.update({test: rows[base_id].get(test) for test in test_types})
            writer.writerow(row)


def _sort_key_for_base_id(base_id: str) -> tuple:
    """Return a sort key enforcing Normal -> Meningioma -> Glioma -> Pituitary ordering."""
    prefix = base_id.split("_", 1)[0].lower()
    category_idx = CATEGORY_ORDER.index(prefix) if prefix in CATEGORY_ORDER else len(CATEGORY_ORDER)
    match = re.search(r"(\d+)$", base_id)
    numeric_part = int(match.group(1)) if match else float("inf")
    return (category_idx, prefix, numeric_part, base_id)


def convert(json_path: Path, csv_path: Optional[Path] = None) -> Path:
    """High-level helper: load JSON, aggregate rows, and write CSV."""
    if csv_path is None:
        csv_path = json_path.with_suffix(".csv")

    data = load_json(json_path)
    metadata = data.get("metadata", {})
    test_types: List[str] = metadata.get("test_types", [])
    if not test_types:
        test_types = sorted(
            {entry.get("test_type") for entry in data.get("results", []) if entry.get("test_type")}
        )

    rows = build_rows(data)
    write_csv(rows, test_types, csv_path)
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert unified progress JSON to CSV.")
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to the unified progress JSON file (e.g., OpenRouter_*_progress.json).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to the JSON path with .csv extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = convert(args.json_path, args.output)
    print(f"Wrote CSV to {output_path}")


if __name__ == "__main__":
    main()

