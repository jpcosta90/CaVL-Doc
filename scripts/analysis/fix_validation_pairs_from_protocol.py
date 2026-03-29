#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _safe_int(value, default=-1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corrige labels de validation_pairs.csv com base no protocol oficial (split/protocolo)."
    )
    parser.add_argument(
        "--validation-csv",
        default="data/LA-CDIP/validation_pairs.csv",
        help="CSV local de validação",
    )
    parser.add_argument(
        "--protocol-csv",
        default="/mnt/data/la-cdip/protocol.csv",
        help="CSV oficial protocol",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="Split number no protocol.csv",
    )
    parser.add_argument(
        "--protocol-mode",
        default="zsl_split",
        choices=["zsl_split", "gzsl_split"],
        help="Modo no protocol.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="data/LA-CDIP/validation_pairs.corrected_split0.csv",
        help="Arquivo de saída corrigido",
    )
    parser.add_argument(
        "--report-csv",
        default="analysis/pair_mismatch_report/lacdip_zsl_split_split0/label_fix_report.csv",
        help="Relatório de linhas alteradas",
    )
    args = parser.parse_args()

    validation_path = Path(args.validation_csv)
    protocol_path = Path(args.protocol_csv)
    output_path = Path(args.output_csv)
    report_path = Path(args.report_csv)

    if not validation_path.is_absolute():
        validation_path = Path.cwd() / validation_path
    if not protocol_path.is_absolute():
        protocol_path = Path.cwd() / protocol_path
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path

    if not validation_path.exists():
        raise FileNotFoundError(f"validation csv não encontrado: {validation_path}")
    if not protocol_path.exists():
        raise FileNotFoundError(f"protocol csv não encontrado: {protocol_path}")

    val = pd.read_csv(validation_path)
    proto = pd.read_csv(protocol_path)

    required_val = {"file_a_name", "file_b_name", "is_equal"}
    required_proto = {"split_mode", "split_number", "file_a_name", "file_b_name", "is_equal"}
    if not required_val.issubset(set(val.columns)):
        raise RuntimeError(f"validation csv sem colunas obrigatórias: {required_val - set(val.columns)}")
    if not required_proto.issubset(set(proto.columns)):
        raise RuntimeError(f"protocol csv sem colunas obrigatórias: {required_proto - set(proto.columns)}")

    proto = proto[(proto["split_mode"] == args.protocol_mode) & (proto["split_number"] == args.split)].copy()

    exact_map: dict[tuple[str, str], int] = {}
    rev_map: dict[tuple[str, str], int] = {}
    for _, row in proto.iterrows():
        a = str(row["file_a_name"])
        b = str(row["file_b_name"])
        y = _safe_int(row["is_equal"], -1)
        exact_map[(a, b)] = y
        rev_map[(b, a)] = y

    fixed = val.copy()
    changes = []
    unmatched = []

    for idx, row in fixed.iterrows():
        a = str(row["file_a_name"])
        b = str(row["file_b_name"])
        old = _safe_int(row["is_equal"], -1)

        if (a, b) in exact_map:
            new = exact_map[(a, b)]
            match_mode = "exact"
        elif (a, b) in rev_map:
            new = rev_map[(a, b)]
            match_mode = "reverse"
        else:
            unmatched.append(idx)
            continue

        if old != new:
            fixed.at[idx, "is_equal"] = new
            changes.append(
                {
                    "row_idx": int(idx),
                    "match_mode": match_mode,
                    "file_a_name": a,
                    "file_b_name": b,
                    "old_label": old,
                    "new_label": new,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(output_path, index=False)
    pd.DataFrame(changes).to_csv(report_path, index=False)

    print("✅ Correção concluída")
    print(f"validation original: {validation_path}")
    print(f"validation corrigido: {output_path}")
    print(f"report mudanças: {report_path}")
    print(f"linhas alteradas: {len(changes)}")
    print(f"linhas sem match no protocol: {len(unmatched)}")


if __name__ == "__main__":
    main()
