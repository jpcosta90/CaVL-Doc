#!/usr/bin/env python3
"""
Gera relatório HTML com pares divergentes entre:
- validation_pairs.csv local do projeto
- protocol.csv oficial do dataset (split/protocolo específicos)

Inclui imagens lado a lado (miniaturas) e classificação dos casos:
- label_mismatch: mesmo par, labels diferentes
- only_in_dataset: existe só no validation local
- only_in_official: existe só no protocol oficial

Uso (LA split 0):
  python scripts/analysis/generate_pair_mismatch_html.py --dataset lacdip --split 0

Uso (RVL split 0):
  python scripts/analysis/generate_pair_mismatch_html.py --dataset rvlcdip --split 0
"""

from __future__ import annotations

import argparse
import hashlib
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow não encontrado. Instale com: pip install pillow") from exc


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    validation_csv: Path
    data_root: Path
    base_image_dir: Path


DATASETS: dict[str, DatasetConfig] = {
    "lacdip": DatasetConfig(
        key="lacdip",
        display_name="LA-CDIP",
        validation_csv=Path("data/LA-CDIP/validation_pairs.csv"),
        data_root=Path("/mnt/data/la-cdip"),
        base_image_dir=Path("/mnt/data/la-cdip/data"),
    ),
    "rvlcdip": DatasetConfig(
        key="rvlcdip",
        display_name="RVL-CDIP",
        validation_csv=Path("data/RVL-CDIP/validation_pairs.csv"),
        data_root=Path("/mnt/data/zs_rvl_cdip"),
        base_image_dir=Path("/mnt/data/zs_rvl_cdip/data"),
    ),
}


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _pair_key(a_name: str, b_name: str) -> tuple[str, str]:
    return tuple(sorted((str(a_name), str(b_name))))


def _ensure_name_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "file_a_name" not in out.columns and "file_a_path" in out.columns:
        out["file_a_name"] = out["file_a_path"].astype(str).map(lambda p: Path(p).name)
    if "file_b_name" not in out.columns and "file_b_path" in out.columns:
        out["file_b_name"] = out["file_b_path"].astype(str).map(lambda p: Path(p).name)
    return out


def _ensure_path_cols(df: pd.DataFrame, splits_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    id_to_path = dict(zip(splits_df["doc_id"], splits_df["doc_path"]))
    if "file_a_path" not in out.columns and "file_a_name" in out.columns:
        out["file_a_path"] = out["file_a_name"].map(id_to_path)
    if "file_b_path" not in out.columns and "file_b_name" in out.columns:
        out["file_b_path"] = out["file_b_name"].map(id_to_path)
    return out


def _build_map(df: pd.DataFrame, source: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in df.iterrows():
        a_name = str(row.get("file_a_name", ""))
        b_name = str(row.get("file_b_name", ""))
        if not a_name or not b_name:
            continue
        key = _pair_key(a_name, b_name)
        result[key] = {
            "source": source,
            "file_a_name": a_name,
            "file_b_name": b_name,
            "file_a_path": row.get("file_a_path"),
            "file_b_path": row.get("file_b_path"),
            "label": _safe_int(row.get("is_equal"), default=-1),
        }
    return result


def _abs_image_path(base_image_dir: Path, rel_or_abs: Any) -> Path | None:
    if rel_or_abs is None or (isinstance(rel_or_abs, float) and pd.isna(rel_or_abs)):
        return None
    raw = str(rel_or_abs)
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    return base_image_dir / p


def _make_thumb(img_path: Path | None, thumbs_dir: Path, max_size: tuple[int, int]) -> str:
    if img_path is None:
        return ""
    key = hashlib.sha1(str(img_path).encode("utf-8")).hexdigest()[:16]
    thumb_path = thumbs_dir / f"{key}.jpg"
    if thumb_path.exists():
        return str(thumb_path.name)
    if not img_path.exists():
        return ""
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size)
            img.save(thumb_path, format="JPEG", quality=90)
        return str(thumb_path.name)
    except Exception:
        return ""


def _render_html(
    output_html: Path,
    title: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    thumbs_rel_dir: str,
) -> None:
    css = """
body { font-family: Arial, sans-serif; margin: 24px; background:#fafafa; color:#222; }
h1 { margin-bottom: 4px; }
.meta { color:#555; margin-bottom: 16px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 16px; }
.card { background:white; border:1px solid #ddd; border-radius:10px; padding:12px; }
.kind { font-weight: bold; margin-bottom: 8px; }
.imgs { display:flex; gap:10px; margin-bottom:10px; }
.imgbox { width:48%; border:1px solid #eee; border-radius:8px; padding:8px; background:#fcfcfc; }
.imgbox img { width:100%; height:auto; border-radius:6px; }
.kv { font-size: 13px; line-height:1.35; }
.warn { color:#a00; font-weight:bold; }
.ok { color:#0a6; font-weight:bold; }
"""
    parts = [
        "<html><head><meta charset='utf-8'><title>{}</title><style>{}</style></head><body>".format(
            html.escape(title), css
        ),
        f"<h1>{html.escape(title)}</h1>",
        (
            "<div class='meta'>"
            f"total={summary['total']} | "
            f"label_mismatch={summary['label_mismatch']} | "
            f"only_in_dataset={summary['only_in_dataset']} | "
            f"only_in_official={summary['only_in_official']}"
            "</div>"
        ),
        "<div class='cards'>",
    ]

    for row in rows:
        kind = row["kind"]
        ours_label = row.get("dataset_label")
        off_label = row.get("official_label")

        img_a = row.get("thumb_a")
        img_b = row.get("thumb_b")

        img_a_html = (
            f"<img src='{html.escape(thumbs_rel_dir + '/' + img_a)}' alt='A'>" if img_a else "<div class='warn'>imagem A indisponível</div>"
        )
        img_b_html = (
            f"<img src='{html.escape(thumbs_rel_dir + '/' + img_b)}' alt='B'>" if img_b else "<div class='warn'>imagem B indisponível</div>"
        )

        parts.append("<div class='card'>")
        parts.append(f"<div class='kind'>{html.escape(kind)}</div>")
        parts.append("<div class='imgs'>")
        parts.append(f"<div class='imgbox'><div class='kv'><b>A</b>: {html.escape(str(row['file_a_name']))}</div>{img_a_html}</div>")
        parts.append(f"<div class='imgbox'><div class='kv'><b>B</b>: {html.escape(str(row['file_b_name']))}</div>{img_b_html}</div>")
        parts.append("</div>")

        parts.append("<div class='kv'>")
        parts.append(f"dataset_label=<b>{html.escape(str(ours_label))}</b> | official_label=<b>{html.escape(str(off_label))}</b><br>")
        parts.append(f"dataset_paths: {html.escape(str(row.get('dataset_paths', '')))}<br>")
        parts.append(f"official_paths: {html.escape(str(row.get('official_paths', '')))}")
        parts.append("</div>")

        parts.append("</div>")

    parts.append("</div></body></html>")

    output_html.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera HTML visual de divergências entre validation local e protocolo oficial")
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), default="lacdip")
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--protocol", default="zsl_split", choices=["zsl_split", "gzsl_split"])
    parser.add_argument("--validation-csv", default=None, help="Override do validation_pairs.csv")
    parser.add_argument("--data-root", default=None, help="Override do data root oficial (onde estão splits.csv e protocol.csv)")
    parser.add_argument("--base-image-dir", default=None, help="Override do diretório base de imagens")
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--thumb-width", type=int, default=420)
    parser.add_argument("--thumb-height", type=int, default=420)
    parser.add_argument("--output-dir", default="analysis/pair_mismatch_report")
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    repo_root = Path(__file__).resolve().parents[2]

    validation_csv = Path(args.validation_csv) if args.validation_csv else (repo_root / cfg.validation_csv)
    data_root = Path(args.data_root) if args.data_root else cfg.data_root
    base_image_dir = Path(args.base_image_dir) if args.base_image_dir else cfg.base_image_dir

    splits_csv = data_root / "splits.csv"
    protocol_csv = data_root / "protocol.csv"

    if not validation_csv.exists():
        raise FileNotFoundError(f"validation_pairs.csv não encontrado: {validation_csv}")
    if not splits_csv.exists() or not protocol_csv.exists():
        raise FileNotFoundError(f"splits/protocol não encontrados em: {data_root}")

    validation = pd.read_csv(validation_csv)
    splits = pd.read_csv(splits_csv)
    protocol = pd.read_csv(protocol_csv)

    validation = _ensure_name_cols(validation)
    validation = _ensure_path_cols(validation, splits)

    protocol_filtered = protocol[(protocol["split_mode"] == args.protocol) & (protocol["split_number"] == args.split)].copy()
    protocol_filtered = _ensure_name_cols(protocol_filtered)
    protocol_filtered = _ensure_path_cols(protocol_filtered, splits)

    ds_map = _build_map(validation, source="dataset")
    off_map = _build_map(protocol_filtered, source="official")

    keys_all = sorted(set(ds_map.keys()) | set(off_map.keys()))
    rows: list[dict[str, Any]] = []

    for key in keys_all:
        d = ds_map.get(key)
        o = off_map.get(key)

        if d and o:
            if d["label"] == o["label"]:
                continue
            kind = "label_mismatch"
            file_a_name = d["file_a_name"]
            file_b_name = d["file_b_name"]
            file_a_path = d.get("file_a_path") or o.get("file_a_path")
            file_b_path = d.get("file_b_path") or o.get("file_b_path")
            dataset_label = d["label"]
            official_label = o["label"]
        elif d and not o:
            kind = "only_in_dataset"
            file_a_name = d["file_a_name"]
            file_b_name = d["file_b_name"]
            file_a_path = d.get("file_a_path")
            file_b_path = d.get("file_b_path")
            dataset_label = d["label"]
            official_label = None
        else:
            kind = "only_in_official"
            file_a_name = o["file_a_name"]
            file_b_name = o["file_b_name"]
            file_a_path = o.get("file_a_path")
            file_b_path = o.get("file_b_path")
            dataset_label = None
            official_label = o["label"]

        rows.append(
            {
                "kind": kind,
                "file_a_name": file_a_name,
                "file_b_name": file_b_name,
                "file_a_path": file_a_path,
                "file_b_path": file_b_path,
                "dataset_label": dataset_label,
                "official_label": official_label,
                "dataset_paths": f"{d.get('file_a_path') if d else None} | {d.get('file_b_path') if d else None}",
                "official_paths": f"{o.get('file_a_path') if o else None} | {o.get('file_b_path') if o else None}",
            }
        )

    # prioridade: label mismatch primeiro
    priority = {"label_mismatch": 0, "only_in_dataset": 1, "only_in_official": 2}
    rows.sort(key=lambda r: (priority.get(r["kind"], 99), str(r["file_a_name"]), str(r["file_b_name"])))
    if args.max_cases > 0:
        rows = rows[: args.max_cases]

    out_dir = Path(args.output_dir) / f"{args.dataset}_{args.protocol}_split{args.split}"
    thumbs_dir = out_dir / "thumbs"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    thumb_size = (args.thumb_width, args.thumb_height)

    for row in rows:
        abs_a = _abs_image_path(base_image_dir, row.get("file_a_path"))
        abs_b = _abs_image_path(base_image_dir, row.get("file_b_path"))
        row["thumb_a"] = _make_thumb(abs_a, thumbs_dir, thumb_size)
        row["thumb_b"] = _make_thumb(abs_b, thumbs_dir, thumb_size)

    summary = {
        "total": len(rows),
        "label_mismatch": sum(1 for r in rows if r["kind"] == "label_mismatch"),
        "only_in_dataset": sum(1 for r in rows if r["kind"] == "only_in_dataset"),
        "only_in_official": sum(1 for r in rows if r["kind"] == "only_in_official"),
    }

    title = f"{cfg.display_name} | {args.protocol} split={args.split} | divergências dataset vs oficial"
    output_html = out_dir / "mismatch_report.html"
    _render_html(output_html, title, summary, rows, thumbs_rel_dir="thumbs")

    # CSV auxiliar com todos os casos renderizados
    pd.DataFrame(rows).to_csv(out_dir / "mismatch_cases.csv", index=False)

    print("\n✅ Relatório gerado")
    print(f"HTML: {output_html}")
    print(f"CSV : {out_dir / 'mismatch_cases.csv'}")
    print(f"Resumo: {summary}")


if __name__ == "__main__":
    main()
