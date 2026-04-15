#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _parse_csv_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def prepare_protocol_split(
    data_root: Path,
    output_dir: Path,
    val_split_idx: int,
    protocol: str,
    exclude_train_splits: List[int],
) -> None:
    if protocol != "zsl":
        raise ValueError("prepare_protocol_split suporta somente protocolo 'zsl' no momento.")

    splits_csv = data_root / "splits.csv"
    protocol_csv = data_root / "protocol.csv"
    if not splits_csv.exists() or not protocol_csv.exists():
        raise FileNotFoundError(f"Esperado splits.csv e protocol.csv em: {data_root}")

    split_meta = pd.read_csv(splits_csv)
    protocol_df = pd.read_csv(protocol_csv)

    id_to_path = pd.Series(split_meta.doc_path.values, index=split_meta.doc_id).to_dict()
    id_to_class = pd.Series(split_meta.class_name.values, index=split_meta.doc_id).to_dict()

    protocol_mode = f"{protocol}_split"
    zdf = protocol_df[protocol_df["split_mode"] == protocol_mode].copy()
    all_splits = sorted(zdf["split_number"].dropna().astype(int).unique().tolist())

    train_source_splits = [s for s in all_splits if s != val_split_idx and s not in exclude_train_splits]
    if not train_source_splits:
        raise RuntimeError(
            f"Sem splits de treino para val_split={val_split_idx}. Excluídos={exclude_train_splits}."
        )

    train_df = zdf[zdf["split_number"].astype(int).isin(train_source_splits)].copy()
    val_df = zdf[zdf["split_number"].astype(int) == int(val_split_idx)].copy()

    for df in (train_df, val_df):
        df["file_a_path"] = df["file_a_name"].map(id_to_path)
        df["file_b_path"] = df["file_b_name"].map(id_to_path)
        df["class_a_name"] = df["file_a_name"].map(id_to_class)
        df["class_b_name"] = df["file_b_name"].map(id_to_class)
        df.dropna(subset=["file_a_path", "file_b_path", "class_a_name", "class_b_name"], inplace=True)

    cols = ["file_a_path", "file_b_path", "is_equal", "class_a_name", "class_b_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df[cols].to_csv(output_dir / "train_pairs.csv", index=False)
    val_df[cols].to_csv(output_dir / "validation_pairs.csv", index=False)

    print(
        f"[OK] protocol split pronto | val={val_split_idx} | "
        f"train_splits={train_source_splits} | train_pairs={len(train_df)} | val_pairs={len(val_df)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara train/validation pairs a partir do protocol.csv")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--val-split-idx", type=int, required=True)
    parser.add_argument("--protocol", default="zsl", choices=["zsl"])
    parser.add_argument(
        "--exclude-train-splits",
        default="",
        help="Splits excluídos do treino (CSV). Ex.: 5",
    )
    args = parser.parse_args()

    prepare_protocol_split(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        val_split_idx=args.val_split_idx,
        protocol=args.protocol,
        exclude_train_splits=_parse_csv_int_list(args.exclude_train_splits),
    )


if __name__ == "__main__":
    main()
