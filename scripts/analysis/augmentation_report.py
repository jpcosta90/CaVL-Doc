#!/usr/bin/env python3
"""
Gera um relatório HTML visual das variações de augmentação aplicadas às imagens do split 3.

Uso:
  python scripts/analysis/augmentation_report.py \
    --split-csv data/generated_splits/zsl_split_3/train_pairs.csv \
    --base-image-dir /mnt/data/la-cdip_backup/data \
    --output augmentation_report.html \
    --n-samples 4
"""
import argparse
import base64
import io
import random
import csv
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ---------------------------------------------------------------------------
# Augmentation ops — versão aprimorada para split 3
# ---------------------------------------------------------------------------

def _rotate(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-15.0, 15.0)
    bg = int(np.array(img.convert("L")).mean())
    return img.rotate(angle, expand=False, fillcolor=bg, resample=Image.BICUBIC)


def _perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    shift = rng.uniform(0.02, 0.07)
    dx, dy = int(w * shift), int(h * shift)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (rng.randint(0, dx),     rng.randint(0, dy)),
        (w - rng.randint(0, dx), rng.randint(0, dy)),
        (w - rng.randint(0, dx), h - rng.randint(0, dy)),
        (rng.randint(0, dx),     h - rng.randint(0, dy)),
    ]
    coeffs = _perspective_coeffs(dst, src)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def _perspective_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    return list(np.linalg.solve(A, B).flatten())


def _brightness_contrast(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    return ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 1.30))


def _blur(img: Image.Image, rng: random.Random) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.8)))


def _sharpen(img: Image.Image, rng: random.Random) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(rng.uniform(1.5, 3.0))


def _gaussian_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    std = rng.uniform(3.0, 12.0)
    noise = np.random.RandomState(rng.randint(0, 99999)).randn(*arr.shape) * std
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), mode=img.mode)


def _jpeg_compression(img: Image.Image, rng: random.Random) -> Image.Image:
    quality = rng.randint(30, 70)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    result = Image.open(buf).copy()
    return result.convert(img.mode)


def _resolution_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    factor = rng.uniform(0.4, 0.7)
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _crop_pad(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    margin = rng.uniform(0.02, 0.08)
    dx, dy = int(w * margin), int(h * margin)
    bg = int(np.array(img.convert("L")).mean())
    cropped = img.crop((dx, dy, w - dx, h - dy))
    result = Image.new(img.mode, (w, h), bg)
    result.paste(cropped, (dx, dy))
    return result


def _yellowing(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    s = rng.uniform(0.05, 0.15)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - s * 2), 0, 255)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + s),     0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 + s * 0.5), 0, 255)
    result = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    return result.convert(img.mode)


def _salt_pepper(img: Image.Image, rng: random.Random) -> Image.Image:
    arr  = np.array(img)
    prob = rng.uniform(0.001, 0.006)
    mask = np.random.RandomState(rng.randint(0, 99999)).random(arr.shape[:2])
    arr[mask < prob / 2] = 0
    arr[(mask >= prob / 2) & (mask < prob)] = 255
    return Image.fromarray(arr.astype(np.uint8), mode=img.mode)


# Lista de todas as ops com nome e descrição
OPS: List[Tuple[str, str, Callable]] = [
    ("Rotação",            "±15° com preenchimento da cor de fundo",           _rotate),
    ("Perspectiva",        "Distorção de perspectiva 2–7%",                    _perspective),
    ("Brilho/Contraste",   "Variação ±25% de brilho e contraste",              _brightness_contrast),
    ("Blur",               "Gaussian blur 0.3–1.8px",                          _blur),
    ("Nitidez",            "Over-sharpening 1.5–3×",                           _sharpen),
    ("Ruído Gaussiano",    "Std 3–12, simula sensor noise",                    _gaussian_noise),
    ("Compressão JPEG",    "Quality 30–70, simula artefatos de scan",          _jpeg_compression),
    ("Resolução baixa",    "Downscale 40–70% + upscale, simula scan de baixa resolução", _resolution_jitter),
    ("Crop/Padding",       "Recorte com margem 2–8% e preenchimento de fundo", _crop_pad),
    ("Amarelamento",       "Simulação de envelhecimento do papel",             _yellowing),
    ("Salt & Pepper",      "Ruído impulsivo 0.1–0.6%",                        _salt_pepper),
]


def augment_combined(img: Image.Image, seed: int) -> Image.Image:
    """Pipeline completo de augmentação para o treino."""
    rng = random.Random(seed)
    probs = [0.8, 0.6, 0.9, 0.7, 0.4, 0.8, 0.5, 0.5, 0.5, 0.4, 0.5]
    for (_, _, fn), prob in zip(OPS, probs):
        if rng.random() < prob:
            try:
                img = fn(img, rng)
            except Exception:
                pass
    return img


# ---------------------------------------------------------------------------
# Helpers HTML
# ---------------------------------------------------------------------------

def _img_to_b64(img: Image.Image, max_w: int = 280) -> str:
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _thumb(b64: str, label: str, sublabel: str = "") -> str:
    sub = f"<div class='sublabel'>{sublabel}</div>" if sublabel else ""
    return (
        f"<div class='thumb'>"
        f"<img src='data:image/jpeg;base64,{b64}'>"
        f"<div class='label'>{label}</div>{sub}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-csv",      default="data/generated_splits/zsl_split_3/train_pairs.csv")
    p.add_argument("--base-image-dir", default="/mnt/data/la-cdip_backup/data")
    p.add_argument("--output",         default="augmentation_report.html")
    p.add_argument("--n-samples",      type=int, default=4)
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    root      = Path(__file__).resolve().parents[2]
    csv_path  = (root / args.split_csv).resolve()
    base_dir  = Path(args.base_image_dir).resolve()
    out_path  = (root / args.output).resolve()

    rng = random.Random(args.seed)

    # Seleciona imagens de classes variadas
    by_class: dict = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            cls = row["class_a_name"]
            by_class.setdefault(cls, set()).add(row["file_a_path"])

    classes = sorted(by_class.keys())
    rng.shuffle(classes)
    samples = []
    for cls in classes:
        if len(samples) >= args.n_samples:
            break
        imgs = sorted(by_class[cls])
        path = base_dir / rng.choice(imgs)
        if path.exists():
            try:
                img = Image.open(path)
                img.load()
                samples.append((cls, path, img))
            except Exception:
                continue

    if not samples:
        print("Nenhuma imagem encontrada. Verifique --base-image-dir.")
        return

    print(f"Amostras selecionadas: {len(samples)} imagens")

    # ── Seção 1: cada op individualmente ────────────────────────────────────
    section_ops = ""
    for op_name, op_desc, op_fn in OPS:
        thumbs = ""
        for cls, path, img in samples:
            orig_b64 = _img_to_b64(img)
            try:
                aug = op_fn(img.copy(), rng)
                aug_b64 = _img_to_b64(aug)
            except Exception as e:
                aug_b64 = orig_b64
            thumbs += (
                f"<div class='pair'>"
                f"{_thumb(orig_b64, 'Original', cls[:28])}"
                f"<div class='arrow'>→</div>"
                f"{_thumb(aug_b64, op_name)}"
                f"</div>"
            )
        section_ops += (
            f"<div class='op-block'>"
            f"<div class='op-header'><span class='op-name'>{op_name}</span>"
            f"<span class='op-desc'>{op_desc}</span></div>"
            f"<div class='pairs-row'>{thumbs}</div>"
            f"</div>"
        )

    # ── Seção 2: variantes combinadas ───────────────────────────────────────
    section_combined = ""
    n_variants = 6
    for cls, path, img in samples:
        orig_b64 = _img_to_b64(img)
        thumbs = _thumb(orig_b64, "Original", cls[:28])
        for v in range(n_variants):
            seed = rng.randint(0, 999999)
            aug  = augment_combined(img.copy(), seed)
            thumbs += _thumb(_img_to_b64(aug), f"Variante {v + 1}")
        section_combined += f"<div class='combined-row'>{thumbs}</div>"

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Relatório de Augmentação — Split 3</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f0f2f5; color: #1a1a2e; font-size: 13px; }}
  .page {{ max-width: 1400px; margin: 0 auto; padding: 28px 20px; }}
  h1 {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; }}
  .subtitle {{ color: #6c757d; font-size: .88rem; margin-bottom: 28px; }}
  h2 {{ font-size: 1.05rem; font-weight: 600; margin: 28px 0 14px;
        padding-bottom: 6px; border-bottom: 2px solid #dee2e6; }}

  /* Op individual */
  .op-block {{ background: white; border-radius: 10px; padding: 16px 20px;
               margin-bottom: 14px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
  .op-header {{ display: flex; align-items: baseline; gap: 10px; margin-bottom: 12px; }}
  .op-name {{ font-weight: 700; font-size: .9rem; color: #2c3e50; }}
  .op-desc {{ font-size: .78rem; color: #6c757d; }}
  .pairs-row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
  .pair {{ display: flex; align-items: center; gap: 6px; }}
  .arrow {{ color: #adb5bd; font-size: 1.2rem; }}

  /* Variantes combinadas */
  .combined-row {{ display: flex; flex-wrap: wrap; gap: 10px; background: white;
                   border-radius: 10px; padding: 16px 20px; margin-bottom: 14px;
                   box-shadow: 0 1px 4px rgba(0,0,0,.06); }}

  /* Thumb */
  .thumb {{ display: flex; flex-direction: column; align-items: center; gap: 4px; }}
  .thumb img {{ border-radius: 6px; border: 1px solid #dee2e6;
                max-width: 200px; height: auto; display: block; }}
  .label {{ font-size: .72rem; font-weight: 600; color: #495057; text-align: center; }}
  .sublabel {{ font-size: .65rem; color: #adb5bd; text-align: center; max-width: 200px;
               overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin-bottom: 24px; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 14px 18px;
                box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  .stat-card .num {{ font-size: 1.6rem; font-weight: 700; color: #3498db; }}
  .stat-card .lbl {{ font-size: .78rem; color: #6c757d; margin-top: 2px; }}
</style>
</head>
<body>
<div class="page">
  <h1>Relatório de Augmentação — Split 3 (Modelo Definitivo)</h1>
  <p class="subtitle">
    {len(samples)} imagens amostradas · {len(OPS)} operações individuais · {n_variants} variantes combinadas por imagem
  </p>

  <div class="stats">
    <div class="stat-card"><div class="num">{len(OPS)}</div><div class="lbl">Operações de augmentação</div></div>
    <div class="stat-card"><div class="num">{n_variants}</div><div class="lbl">Variantes por imagem (treino)</div></div>
    <div class="stat-card"><div class="num">120</div><div class="lbl">Classes de treino (Split 3)</div></div>
    <div class="stat-card"><div class="num">24</div><div class="lbl">Classes novel (validação ZSL)</div></div>
    <div class="stat-card"><div class="num">1748</div><div class="lbl">Imagens originais de treino</div></div>
    <div class="stat-card"><div class="num">{1748 * n_variants:,}</div><div class="lbl">Imagens augmentadas totais</div></div>
  </div>

  <h2>🔬 Operações Individuais de Augmentação</h2>
  {section_ops}

  <h2>🎲 Variantes Combinadas (Pipeline Completo de Treino)</h2>
  {section_combined}
</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"✅ Relatório salvo em: {out_path}  ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
