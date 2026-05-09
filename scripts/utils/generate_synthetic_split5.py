#!/usr/bin/env python3
"""
Geração do conjunto de teste sintético para o split 5 (eval_test_split5_synthetic).

Estratégia:
  - Cada imagem original do split 5 gera N variantes com augmentation de scan realista
    (rotação, blur, ruído, perspectiva, brilho/contraste, amarelamento)
  - Pares genuínos: variante_1 vs variante_2 da mesma imagem
  - Pares impostores: variante de imagem_A (classe A) vs variante de imagem_B (classe B)
  - Resultado: mesmo volume de pares que o split 5 original → comparação direta

Uso:
    python scripts/utils/generate_synthetic_split5.py \
        --base-image-dir /mnt/data/la-cdip/data \
        --output-dir data/generated_splits/eval_test_split5_synthetic
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SPLIT5_CSV     = WORKSPACE_ROOT / "data" / "generated_splits" / "eval_test_split5" / "validation_pairs.csv"

# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def _rotate(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-5.0, 5.0)
    if abs(angle) < 0.3:
        return img
    bg = int(img.convert("L").getextrema()[1] * 0.95)
    return img.rotate(angle, expand=False, fillcolor=bg, resample=Image.BICUBIC)


def _perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    """Distorção de perspectiva leve — simula documento não perfeitamente plano."""
    w, h = img.size
    shift = rng.uniform(0.005, 0.02)
    dx = int(w * shift)
    dy = int(h * shift)
    # Mapeamento: 4 cantos originais → 4 cantos levemente deslocados
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (rng.randint(0, dx),      rng.randint(0, dy)),
        (w - rng.randint(0, dx),  rng.randint(0, dy)),
        (w - rng.randint(0, dx),  h - rng.randint(0, dy)),
        (rng.randint(0, dx),      h - rng.randint(0, dy)),
    ]
    # Coeficientes da transformação projetiva
    coeffs = _perspective_coeffs(dst, src)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def _perspective_coeffs(pa: List[Tuple[int,int]], pb: List[Tuple[int,int]]) -> List[float]:
    """Calcula coeficientes de transformação projetiva de pa → pb."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    res = np.linalg.solve(A, B)
    return list(np.array(res).flatten())


def _blur(img: Image.Image, rng: random.Random) -> Image.Image:
    sigma = rng.uniform(0.2, 1.0)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def _brightness_contrast(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    return img


def _gaussian_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    sigma = rng.uniform(1.0, 6.0)
    noise = rng.gauss(0, sigma) * np.random.randn(*arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode=img.mode)


def _yellowing(img: Image.Image, rng: random.Random) -> Image.Image:
    """Leve amarelamento — simula papel envelhecido."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    # Reduz canal azul levemente, eleva vermelho/verde ligeiramente
    strength = rng.uniform(0.02, 0.08)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - strength * 2), 0, 255)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + strength),     0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 + strength * 0.5), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _salt_pepper(img: Image.Image, rng: random.Random) -> Image.Image:
    """Artefatos de scan — pixels brancos/pretos aleatórios."""
    arr   = np.array(img)
    prob  = rng.uniform(0.0005, 0.003)
    mask  = np.random.random(arr.shape[:2])
    arr[mask < prob / 2]               = 0
    arr[(mask >= prob / 2) & (mask < prob)] = 255
    return Image.fromarray(arr.astype(np.uint8), mode=img.mode)


def augment(img: Image.Image, seed: int) -> Image.Image:
    """Aplica pipeline de augmentation realista de scan."""
    rng = random.Random(seed)
    # Garante RGB para yellowing; preserva modo original se for L (grayscale)
    original_mode = img.mode
    if original_mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Ordem das transformações: geométricas → fotométricas → ruído
    ops = [
        (_rotate,             0.8),
        (_perspective,        0.5),
        (_brightness_contrast, 0.9),
        (_blur,               0.7),
    ]
    if original_mode == "RGB":
        ops.append((_yellowing, 0.4))
    ops.append((_gaussian_noise, 0.8))
    ops.append((_salt_pepper,    0.5))

    for fn, prob in ops:
        if rng.random() < prob:
            img = fn(img, rng)

    # Restaura modo original se era grayscale
    if original_mode == "L" and img.mode != "L":
        img = img.convert("L")

    return img


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def load_split5_csv(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def collect_images_by_class(rows: list) -> dict:
    """Retorna {class_name: [rel_path, ...]}."""
    by_class: dict = {}
    seen = set()
    for r in rows:
        for path_key, class_key in [("file_a_path", "class_a_name"),
                                     ("file_b_path", "class_b_name")]:
            rel = r[path_key]
            cls = r[class_key]
            if rel not in seen:
                seen.add(rel)
                by_class.setdefault(cls, []).append(rel)
    return by_class


def generate_augmented_images(
    by_class: dict,
    base_image_dir: Path,
    aug_dir: Path,
    n_variants: int,
    seed_base: int,
) -> dict:
    """
    Para cada imagem original, gera n_variants augmented versions.
    Retorna {rel_path: [aug_rel_path_v0, aug_rel_path_v1, ...]}.
    """
    aug_map: dict = {}
    total = sum(len(v) for v in by_class.values())
    done  = 0

    for cls, rel_paths in by_class.items():
        cls_dir = aug_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for rel in rel_paths:
            src = base_image_dir / rel
            if not src.exists():
                print(f"  [WARN] Imagem não encontrada: {src}")
                aug_map[rel] = []
                done += 1
                continue

            img  = Image.open(src)
            stem = Path(rel).stem
            variants = []
            for v in range(n_variants):
                out_name = f"{stem}_aug{v}.tif"
                out_path = cls_dir / out_name
                if not out_path.exists():
                    aug_img = augment(img, seed=seed_base + done * 100 + v)
                    aug_img.save(out_path, format="TIFF", compression="lzw")
                variants.append(f"{cls}/{out_name}")

            aug_map[rel] = variants
            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{total}] imagens processadas...")

    return aug_map


def build_pairs(
    by_class: dict,
    aug_map: dict,
    n_genuine_per_img: int,
    rng: random.Random,
) -> list:
    """
    Cria pares genuínos e impostores a partir das variantes augmentadas.
    Genuíno:   variante_i vs variante_j da mesma imagem (i ≠ j).
    Impostor:  variante de img_A (classe A) vs variante de img_B (classe B ≠ A).
    """
    genuine = []
    impostor = []

    # Pares genuínos
    for cls, rel_paths in by_class.items():
        for rel in rel_paths:
            variants = aug_map.get(rel, [])
            if len(variants) < 2:
                continue
            # Todas as combinações de pares (v_i, v_j) com i < j
            from itertools import combinations
            for v_a, v_b in combinations(variants, 2):
                genuine.append({
                    "file_a_path":  v_a,
                    "file_b_path":  v_b,
                    "is_equal":     1,
                    "class_a_name": cls,
                    "class_b_name": cls,
                })

    # Pares impostores — mesmo número que os genuínos
    classes = list(by_class.keys())
    target_impostors = len(genuine)
    attempts = 0
    impostor_set = set()

    while len(impostor) < target_impostors and attempts < target_impostors * 20:
        cls_a = rng.choice(classes)
        cls_b = rng.choice(classes)
        if cls_a == cls_b:
            attempts += 1
            continue
        imgs_a = [r for r in by_class[cls_a] if len(aug_map.get(r, [])) >= 1]
        imgs_b = [r for r in by_class[cls_b] if len(aug_map.get(r, [])) >= 1]
        if not imgs_a or not imgs_b:
            attempts += 1
            continue
        rel_a = rng.choice(imgs_a)
        rel_b = rng.choice(imgs_b)
        v_a   = rng.choice(aug_map[rel_a])
        v_b   = rng.choice(aug_map[rel_b])
        key   = tuple(sorted([v_a, v_b]))
        if key in impostor_set:
            attempts += 1
            continue
        impostor_set.add(key)
        impostor.append({
            "file_a_path":  v_a,
            "file_b_path":  v_b,
            "is_equal":     0,
            "class_a_name": cls_a,
            "class_b_name": cls_b,
        })

    # Embaralha e intercala
    rng.shuffle(genuine)
    rng.shuffle(impostor)
    pairs = []
    for g, i in zip(genuine, impostor):
        pairs.append(g)
        pairs.append(i)
    return pairs


def write_csv(pairs: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file_a_path", "file_b_path", "is_equal", "class_a_name", "class_b_name"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Gera split 5 sintético via augmentation de scan.")
    p.add_argument("--base-image-dir", required=True,
                   help="Diretório base das imagens LA-CDIP (onde estão as pastas de classes)")
    p.add_argument("--split5-csv",     default=str(SPLIT5_CSV),
                   help="CSV do split 5 original")
    p.add_argument("--output-dir",     default=str(WORKSPACE_ROOT / "data" / "generated_splits" / "eval_test_split5_synthetic"),
                   help="Diretório de saída para imagens augmentadas e CSV")
    p.add_argument("--n-variants",     type=int, default=2,
                   help="Variantes augmentadas por imagem (mín. 2 para pares genuínos)")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--dry-run",        action="store_true",
                   help="Apenas mostra estatísticas sem gerar arquivos")
    args = p.parse_args()

    base_image_dir = Path(args.base_image_dir)
    output_dir     = Path(args.output_dir)
    aug_dir        = output_dir / "images"
    out_csv        = output_dir / "validation_pairs.csv"
    rng            = random.Random(args.seed)

    print(f"Lendo split 5 de: {args.split5_csv}")
    rows     = load_split5_csv(Path(args.split5_csv))
    by_class = collect_images_by_class(rows)

    n_images  = sum(len(v) for v in by_class.values())
    n_classes = len(by_class)
    print(f"  {n_images} imagens únicas em {n_classes} classes")
    print(f"  Variantes por imagem: {args.n_variants}")
    print(f"  Total de imagens a gerar: {n_images * args.n_variants}")

    if args.dry_run:
        print("\n[DRY RUN] Nenhum arquivo gerado.")
        return

    print("\nGerando imagens augmentadas...")
    aug_map = generate_augmented_images(
        by_class=by_class,
        base_image_dir=base_image_dir,
        aug_dir=aug_dir,
        n_variants=args.n_variants,
        seed_base=args.seed,
    )

    print("\nCriando pares...")
    pairs = build_pairs(by_class, aug_map, n_genuine_per_img=1, rng=rng)

    genuine   = sum(1 for p in pairs if p["is_equal"] == 1)
    impostor  = sum(1 for p in pairs if p["is_equal"] == 0)
    print(f"  Pares genuínos:   {genuine}")
    print(f"  Pares impostores: {impostor}")
    print(f"  Total:            {len(pairs)}")

    write_csv(pairs, out_csv)
    print(f"\n✅ CSV salvo em: {out_csv}")
    print(f"   Imagens em:   {aug_dir}")


if __name__ == "__main__":
    main()
