from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_csv_has_header(path: Path, fieldnames: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def append_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> int:
    count = 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_csv_dicts(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict] = []
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
        return rows

