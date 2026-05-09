#!/usr/bin/env python3
"""
Relatório HTML comparando imagens originais do split 5 com variantes augmentadas.

Uso:
    python scripts/analysis/generate_synthetic_split5_report.py \
        --base-image-dir /mnt/data/la-cdip/data \
        --output results/synthetic_split5_report.html
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import random
from pathlib import Path

from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SPLIT5_CSV    = WORKSPACE_ROOT / "data" / "generated_splits" / "eval_test_split5" / "validation_pairs.csv"
SYNTHETIC_DIR = WORKSPACE_ROOT / "data" / "generated_splits" / "eval_test_split5_synthetic"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _img_b64(path: Path, max_width: int = 280) -> str | None:
    try:
        img = Image.open(path).convert("RGB")
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"  [WARN] Não conseguiu abrir {path}: {e}")
        return None


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _collect_originals(rows: list[dict]) -> dict[str, str]:
    """Returns {rel_path: class_name}."""
    seen: dict[str, str] = {}
    for r in rows:
        for pk, ck in [("file_a_path", "class_a_name"), ("file_b_path", "class_b_name")]:
            if r[pk] not in seen:
                seen[r[pk]] = r[ck]
    return seen


def _build_aug_map(orig_by_path: dict, aug_images_dir: Path) -> dict[str, list[Path]]:
    aug_map: dict[str, list[Path]] = {}
    for rel, cls in orig_by_path.items():
        stem = Path(rel).stem
        variants = sorted(aug_images_dir.glob(f"{cls}/{stem}_aug*.tif"))
        aug_map[rel] = variants
    return aug_map


# ---------------------------------------------------------------------------
# HTML building
# ---------------------------------------------------------------------------

def _img_tag(b64: str, label: str, border: str = "#6c757d") -> str:
    return f"""
      <div style="text-align:center;flex:1;min-width:0">
        <img src="data:image/jpeg;base64,{b64}"
             style="max-width:100%;border:2px solid {border};border-radius:4px"/>
        <div style="font-size:11px;color:#aaa;margin-top:4px">{label}</div>
      </div>"""


def _aug_gallery_section(samples, base_image_dir, aug_images_dir, aug_map) -> str:
    cards = []
    for rel, cls in samples:
        orig_path = base_image_dir / rel
        b64_orig = _img_b64(orig_path)
        if b64_orig is None:
            continue
        variants = aug_map.get(rel, [])
        b64_v0 = _img_b64(variants[0]) if len(variants) > 0 else None
        b64_v1 = _img_b64(variants[1]) if len(variants) > 1 else None

        imgs_html = _img_tag(b64_orig, "Original", "#6c757d")
        if b64_v0:
            imgs_html += _img_tag(b64_v0, "Variante 1", "#0d6efd")
        if b64_v1:
            imgs_html += _img_tag(b64_v1, "Variante 2", "#6610f2")

        stem = Path(rel).stem
        cards.append(f"""
      <div style="background:#1e1e1e;border-radius:8px;padding:12px;margin-bottom:16px">
        <div style="font-size:11px;color:#888;margin-bottom:8px">
          <b style="color:#ccc">{cls}</b> · {stem}
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start">
          {imgs_html}
        </div>
      </div>""")

    return f"""
  <h2 style="color:#e0e0e0;border-bottom:1px solid #333;padding-bottom:8px">
    Galeria de Augmentation — Original vs Variantes
  </h2>
  <p style="color:#999;font-size:13px">
    Cada linha mostra a imagem original (bordas cinza), variante 1 (azul) e variante 2 (roxo).
    As transformações simulam artefatos reais de digitalização de documentos.
  </p>
  {''.join(cards)}"""


def _pair_section(pair_rows, aug_images_dir, title: str, color: str, label: str) -> str:
    cards = []
    for r in pair_rows:
        p_a = aug_images_dir / r["file_a_path"]
        p_b = aug_images_dir / r["file_b_path"]
        b64_a = _img_b64(p_a)
        b64_b = _img_b64(p_b)
        if not b64_a or not b64_b:
            continue

        cls_a = r["class_a_name"]
        cls_b = r["class_b_name"]
        same = cls_a == cls_b
        pair_label = f"{Path(r['file_a_path']).stem} × {Path(r['file_b_path']).stem}"

        cards.append(f"""
      <div style="background:#1e1e1e;border-radius:8px;padding:12px;
                  border-left:3px solid {color};margin-bottom:12px">
        <div style="font-size:11px;color:#888;margin-bottom:8px">
          {'<b style="color:#ccc">' + cls_a + '</b>' if same else
           '<b style="color:#ccc">' + cls_a + '</b> <span style="color:#666">vs</span> <b style="color:#ccc">' + cls_b + '</b>'}
          · {pair_label}
        </div>
        <div style="display:flex;gap:10px;align-items:flex-start">
          {_img_tag(b64_a, "A", color)}
          <div style="display:flex;align-items:center;padding:0 4px;color:{color};font-size:20px;flex-shrink:0">
            {'=' if same else '≠'}
          </div>
          {_img_tag(b64_b, "B", color)}
        </div>
      </div>""")

    return f"""
  <h2 style="color:#e0e0e0;border-bottom:1px solid #333;padding-bottom:8px;margin-top:40px">
    {title}
  </h2>
  <p style="color:#999;font-size:13px">{label}</p>
  {''.join(cards)}"""


def _pipeline_section() -> str:
    steps = [
        ("Rotação", "±5°", "0.80", "Simula inclinação do scanner ou da página."),
        ("Perspectiva", "shift 0.5–2%", "0.50", "Warp projetivo leve — documento não perfeitamente plano."),
        ("Brilho/Contraste", "±15%", "0.90", "Variações de exposição e calibração do scanner."),
        ("Blur Gaussiano", "σ = 0.2–1.0", "0.70", "Desfoque de foco ou movimento da câmera."),
        ("Amarelamento", "força 2–8%", "0.40", "Papel envelhecido — reduz canal B, eleva R/G. Apenas RGB."),
        ("Ruído Gaussiano", "σ = 1–6", "0.80", "Ruído eletrônico do sensor de digitalização."),
        ("Salt & Pepper", "0.05–0.3%", "0.50", "Pixels defeituosos do scanner ou compressão lossy."),
    ]
    rows_html = "".join(
        f"""<tr>
          <td style="padding:6px 12px;color:#ccc;font-weight:500">{name}</td>
          <td style="padding:6px 12px;color:#aaa;font-family:monospace">{params}</td>
          <td style="padding:6px 12px">
            <div style="background:#0d6efd;height:8px;border-radius:4px;width:{int(float(prob)*100)}%"></div>
            <span style="color:#888;font-size:11px">{int(float(prob)*100)}%</span>
          </td>
          <td style="padding:6px 12px;color:#888;font-size:12px">{desc}</td>
        </tr>"""
        for name, params, prob, desc in steps
    )
    return f"""
  <h2 style="color:#e0e0e0;border-bottom:1px solid #333;padding-bottom:8px;margin-top:40px">
    Pipeline de Augmentation
  </h2>
  <p style="color:#999;font-size:13px">
    Cada imagem passa pelas transformações abaixo (aplicadas em sequência com probabilidades independentes).
    O seed garante reprodutibilidade — cada variante tem seed único baseado na posição e índice da variante.
  </p>
  <table style="width:100%;border-collapse:collapse;background:#1e1e1e;border-radius:8px;overflow:hidden">
    <thead>
      <tr style="background:#2a2a2a;color:#888;font-size:12px;text-transform:uppercase">
        <th style="padding:8px 12px;text-align:left">Transformação</th>
        <th style="padding:8px 12px;text-align:left">Parâmetros</th>
        <th style="padding:8px 12px;text-align:left">Prob.</th>
        <th style="padding:8px 12px;text-align:left">Motivação</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>"""


def build_html(
    stats: dict,
    aug_gallery: str,
    genuine_section: str,
    impostor_section: str,
    pipeline_section: str,
) -> str:
    cards_html = "".join(
        f"""<div style="background:#1e1e1e;border-radius:8px;padding:20px;text-align:center;flex:1">
          <div style="font-size:28px;font-weight:700;color:{color}">{value:,}</div>
          <div style="font-size:12px;color:#888;margin-top:4px">{label}</div>
        </div>"""
        for value, label, color in [
            (stats["n_classes"],   "Classes",           "#0d6efd"),
            (stats["n_orig"],      "Imagens originais", "#6c757d"),
            (stats["n_aug"],       "Imagens geradas",   "#6610f2"),
            (stats["n_genuine"],   "Pares genuínos",    "#198754"),
            (stats["n_impostor"],  "Pares impostores",  "#dc3545"),
            (stats["total_pairs"], "Total de pares",    "#ffc107"),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Split 5 Sintético — Relatório de Augmentation</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #121212; color: #e0e0e0; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 20px 60px; }}
    h1 {{ font-size: 24px; font-weight: 700; }}
    h2 {{ font-size: 18px; font-weight: 600; margin-top: 0; }}
  </style>
</head>
<body>
<div class="container">

  <div style="margin-bottom:32px;padding-bottom:20px;border-bottom:1px solid #333">
    <h1>Split 5 Sintético — Augmentation de Scan</h1>
    <p style="color:#888;margin-top:6px;font-size:13px">
      Conjunto de teste sintético gerado a partir do split 5 original (LA-CDIP).
      Cada imagem original gera 2 variantes com transformações que simulam artefatos reais de digitalização.
      Pares genuínos comparam duas variantes da mesma imagem; pares impostores comparam variantes de classes distintas.
    </p>
  </div>

  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:40px">
    {cards_html}
  </div>

  {aug_gallery}
  {genuine_section}
  {impostor_section}
  {pipeline_section}

</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Gera relatório HTML do split 5 sintético.")
    p.add_argument("--base-image-dir", required=True,
                   help="Diretório base das imagens LA-CDIP originais")
    p.add_argument("--synthetic-dir",  default=str(SYNTHETIC_DIR),
                   help="Diretório do split 5 sintético gerado")
    p.add_argument("--split5-csv",     default=str(SPLIT5_CSV),
                   help="CSV original do split 5")
    p.add_argument("--output",         default=str(WORKSPACE_ROOT / "results" / "synthetic_split5_report.html"),
                   help="Caminho do HTML de saída")
    p.add_argument("--n-gallery",      type=int, default=15,
                   help="Número de imagens na galeria de augmentation")
    p.add_argument("--n-pairs",        type=int, default=8,
                   help="Número de pares de exemplo (genuínos e impostores)")
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    base_image_dir = Path(args.base_image_dir)
    synthetic_dir  = Path(args.synthetic_dir)
    aug_images_dir = synthetic_dir / "images"
    synthetic_csv  = synthetic_dir / "validation_pairs.csv"

    rng = random.Random(args.seed)

    print("Lendo CSVs...")
    orig_rows     = _load_csv(Path(args.split5_csv))
    synthetic_rows = _load_csv(synthetic_csv)

    orig_by_path = _collect_originals(orig_rows)
    aug_map      = _build_aug_map(orig_by_path, aug_images_dir)

    n_orig    = len(orig_by_path)
    n_aug     = sum(len(v) for v in aug_map.values())
    n_genuine = sum(1 for r in synthetic_rows if r["is_equal"] == "1")
    n_impost  = sum(1 for r in synthetic_rows if r["is_equal"] == "0")
    n_classes = len(set(orig_by_path.values()))

    print(f"  {n_orig} imagens originais | {n_aug} variantes geradas | {len(synthetic_rows)} pares")

    # Galeria: amostra aleatória de imagens com original + 2 variantes
    eligible = [
        (rel, cls) for rel, cls in orig_by_path.items()
        if len(aug_map.get(rel, [])) >= 2 and (base_image_dir / rel).exists()
    ]
    rng.shuffle(eligible)
    gallery_samples = eligible[:args.n_gallery]
    print(f"  {len(gallery_samples)} imagens para galeria")

    # Pares de exemplo
    genuine_rows = [r for r in synthetic_rows if r["is_equal"] == "1"]
    impostor_rows = [r for r in synthetic_rows if r["is_equal"] == "0"]
    rng.shuffle(genuine_rows)
    rng.shuffle(impostor_rows)

    def _pick_pairs(rows, n):
        picked = []
        for r in rows:
            if (aug_images_dir / r["file_a_path"]).exists() and (aug_images_dir / r["file_b_path"]).exists():
                picked.append(r)
            if len(picked) >= n:
                break
        return picked

    genuine_samples  = _pick_pairs(genuine_rows,  args.n_pairs)
    impostor_samples = _pick_pairs(impostor_rows, args.n_pairs)

    print("Renderizando galeria...")
    aug_gallery = _aug_gallery_section(gallery_samples, base_image_dir, aug_images_dir, aug_map)

    print("Renderizando pares genuínos...")
    genuine_sec = _pair_section(
        genuine_samples, aug_images_dir,
        title="Pares Genuínos (mesma imagem, variantes diferentes)",
        color="#198754",
        label="Ambas as imagens são variantes augmentadas do mesmo documento original. "
              "O modelo deve reconhecê-las como idênticas apesar das transformações."
    )

    print("Renderizando pares impostores...")
    impostor_sec = _pair_section(
        impostor_samples, aug_images_dir,
        title="Pares Impostores (classes distintas)",
        color="#dc3545",
        label="Cada imagem é de um documento diferente. "
              "O modelo deve distingui-las mesmo com aparência visual similar devido ao augmentation."
    )

    pipeline_sec = _pipeline_section()

    print("Montando HTML...")
    html = build_html(
        stats={
            "n_classes":   n_classes,
            "n_orig":      n_orig,
            "n_aug":       n_aug,
            "n_genuine":   n_genuine,
            "n_impostor":  n_impost,
            "total_pairs": len(synthetic_rows),
        },
        aug_gallery=aug_gallery,
        genuine_section=genuine_sec,
        impostor_section=impostor_sec,
        pipeline_section=pipeline_sec,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"\n✅ Relatório salvo em: {out_path}")


if __name__ == "__main__":
    main()
