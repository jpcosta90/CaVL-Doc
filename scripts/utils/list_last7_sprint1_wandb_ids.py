#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List


def _parse_created_at(value: str) -> datetime:
    raw = (value or "").strip()
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def load_last_runs(csv_path: str, limit: int) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if (row.get("state") or "").strip().lower() == "finished"]

    rows.sort(key=lambda row: _parse_created_at(row.get("created_at", "")), reverse=True)
    return rows[:limit]


def save_csv(rows: List[Dict[str, str]], output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "run_name", "best_eer", "created_at"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row.get("run_id", ""),
                    "run_name": row.get("run_name", ""),
                    "best_eer": row.get("best_eer", ""),
                    "created_at": row.get("created_at", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lista as últimas N runs finalizadas do Sprint 1 com run_id, run_name e best_eer."
    )
    parser.add_argument(
        "--runs-csv",
        default="analysis/sprint1_report/sprint1_runs_raw.csv",
        help="CSV consolidado das runs do Sprint 1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=7,
        help="Quantidade de runs mais recentes para listar",
    )
    parser.add_argument(
        "--output-csv",
        default="analysis/sprint1_report/sprint1_last7_valid_ids.csv",
        help="CSV de saída com os ids válidos",
    )
    args = parser.parse_args()

    rows = load_last_runs(csv_path=args.runs_csv, limit=args.limit)

    if not rows:
        raise SystemExit("Nenhuma run finalizada encontrada no CSV informado.")

    print("run_id,run_name,best_eer")
    for row in rows:
        print(f"{row.get('run_id','')},{row.get('run_name','')},{row.get('best_eer','')}")

    save_csv(rows, args.output_csv)
    print(f"\nCSV salvo em: {args.output_csv}")


if __name__ == "__main__":
    main()
