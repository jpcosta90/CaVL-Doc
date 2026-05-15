#!/usr/bin/env python3
"""
Prepara os dados de treino e validação para o modelo ArcDoc final.

Estratégia:
  - Treino:    pares de treino de TODOS os splits fornecidos, combinados.
  - Validação: pares de validação do split de referência (padrão: split 0),
               cujas classes nunca aparecem no treino (protocolo ZSL).
  - Augmentation offline: cada imagem gera N variantes com pipeline de scan
    realista (rotação, perspectiva, blur, ruído, yellowing, salt & pepper).
  - Pares genuínos:   variante_i vs variante_j da mesma imagem.
  - Pares impostores: variante de classe_A vs variante de classe_B.

Uso:
  python scripts/utils/prepare_arcdoc_training.py \\
      --splits-dirs data/generated_splits/sprint3_zsl_val_0_train_excl_5 \\
                    data/generated_splits/sprint3_zsl_val_1_train_excl_5 \\
                    data/generated_splits/sprint3_zsl_val_2_train_excl_5 \\
                    data/generated_splits/sprint3_zsl_val_3_train_excl_5 \\
      --val-split-dir data/generated_splits/sprint3_zsl_val_0_train_excl_5 \\
      --base-image-dir /mnt/data/la-cdip/data \\
      --output-dir data/generated_splits/arcdoc_final \\
      --n-variants 3
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Augmentation pipeline (reutilizado de generate_synthetic_split5.py)
# ---------------------------------------------------------------------------

def _rotate(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-5.0, 5.0)
    if abs(angle) < 0.3:
        return img
    bg = int(img.convert("L").getextrema()[1] * 0.95)
    return img.rotate(angle, expand=False, fillcolor=bg, resample=Image.BICUBIC)


def _perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    shift = rng.uniform(0.005, 0.02)
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


def _perspective_coeffs(pa, pb) -> List[float]:
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.array(matrix, dtype=float)
    B = np.array(pb).reshape(8)
    res = np.linalg.solve(A, B)
    return list(res.flatten())


def _blur(img: Image.Image, rng: random.Random) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0)))


def _brightness_contrast(img: Image.Image, rng: random.Random) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    return ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))


def _gaussian_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = rng.gauss(0, rng.uniform(1.0, 6.0)) * np.random.randn(*arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), mode=img.mode)


def _yellowing(img: Image.Image, rng: random.Random) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32)
    s = rng.uniform(0.02, 0.08)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - s * 2), 0, 255)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + s),     0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 + s * 0.5), 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _salt_pepper(img: Image.Image, rng: random.Random) -> Image.Image:
    arr  = np.array(img)
    prob = rng.uniform(0.0005, 0.003)
    mask = np.random.random(arr.shape[:2])
    arr[mask < prob / 2] = 0
    arr[(mask >= prob / 2) & (mask < prob)] = 255
    return Image.fromarray(arr.astype(np.uint8), mode=img.mode)


def augment(img: Image.Image, seed: int) -> Image.Image:
    rng          = random.Random(seed)
    original_mode = img.mode
    if original_mode not in ("RGB", "L"):
        img = img.convert("RGB")

    ops = [
        (_rotate,              0.8),
        (_perspective,         0.5),
        (_brightness_contrast, 0.9),
        (_blur,                0.7),
        (_gaussian_noise,      0.8),
        (_salt_pepper,         0.5),
    ]
    if original_mode == "RGB":
        ops.insert(4, (_yellowing, 0.4))

    for fn, prob in ops:
        if rng.random() < prob:
            img = fn(img, rng)

    if original_mode == "L" and img.mode != "L":
        img = img.convert("L")
    return img


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(pairs: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file_a_path", "file_b_path", "is_equal", "class_a_name", "class_b_name"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pairs)


def _collect_images_by_class(rows: List[dict]) -> Dict[str, List[str]]:
    by_class: Dict[str, List[str]] = {}
    seen: set = set()
    for r in rows:
        for path_key, class_key in [("file_a_path", "class_a_name"),
                                     ("file_b_path", "class_b_name")]:
            rel = r[path_key]
            cls = r[class_key]
            if rel not in seen:
                seen.add(rel)
                by_class.setdefault(cls, []).append(rel)
    return by_class


# ---------------------------------------------------------------------------
# Augmentation offline
# ---------------------------------------------------------------------------

def generate_augmented_images(
    by_class: Dict[str, List[str]],
    base_image_dir: Path,
    aug_dir: Path,
    n_variants: int,
    seed_base: int,
    label: str = "",
) -> Dict[str, List[str]]:
    """Gera N variantes augmentadas por imagem. Retorna {rel_path: [aug_rel_path, ...]}."""
    aug_map: Dict[str, List[str]] = {}
    total = sum(len(v) for v in by_class.values())
    done  = 0

    for cls, rel_paths in by_class.items():
        cls_dir = aug_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for rel in rel_paths:
            src = base_image_dir / rel
            if not src.exists():
                print(f"  [WARN] Não encontrada: {src}")
                aug_map[rel] = []
                done += 1
                continue

            img      = Image.open(src)
            stem     = Path(rel).stem
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
            if done % 100 == 0:
                print(f"  {label}[{done}/{total}] imagens augmentadas...")

    print(f"  {label}{total} imagens → {total * n_variants} variantes geradas.")
    return aug_map


# ---------------------------------------------------------------------------
# Par building
# ---------------------------------------------------------------------------

def build_pairs(
    by_class: Dict[str, List[str]],
    aug_map: Dict[str, List[str]],
    rng: random.Random,
) -> List[dict]:
    genuine:  List[dict] = []
    impostor: List[dict] = []
    classes = list(by_class.keys())

    for cls, rel_paths in by_class.items():
        for rel in rel_paths:
            variants = aug_map.get(rel, [])
            for v_a, v_b in combinations(variants, 2):
                genuine.append({
                    "file_a_path":  v_a,
                    "file_b_path":  v_b,
                    "is_equal":     1,
                    "class_a_name": cls,
                    "class_b_name": cls,
                })

    target   = len(genuine)
    seen_set = set()
    attempts = 0
    while len(impostor) < target and attempts < target * 20:
        cls_a = rng.choice(classes)
        cls_b = rng.choice(classes)
        if cls_a == cls_b:
            attempts += 1
            continue
        imgs_a = [r for r in by_class[cls_a] if aug_map.get(r)]
        imgs_b = [r for r in by_class[cls_b] if aug_map.get(r)]
        if not imgs_a or not imgs_b:
            attempts += 1
            continue
        v_a = rng.choice(aug_map[rng.choice(imgs_a)])
        v_b = rng.choice(aug_map[rng.choice(imgs_b)])
        key = tuple(sorted([v_a, v_b]))
        if key in seen_set:
            attempts += 1
            continue
        seen_set.add(key)
        impostor.append({
            "file_a_path":  v_a,
            "file_b_path":  v_b,
            "is_equal":     0,
            "class_a_name": cls_a,
            "class_b_name": cls_b,
        })

    rng.shuffle(genuine)
    rng.shuffle(impostor)
    pairs = []
    for g, i in zip(genuine, impostor):
        pairs.extend([g, i])
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepara dados de treino/validação augmentados para o ArcDoc final."
    )
    p.add_argument("--splits-dirs",    nargs="+", required=True,
                   help="Diretórios dos splits de treino (train_pairs.csv em cada um)")
    p.add_argument("--val-split-dir",  required=True,
                   help="Diretório do split de referência para validação (validation_pairs.csv)")
    p.add_argument("--base-image-dir", required=True,
                   help="Diretório base das imagens originais")
    p.add_argument("--output-dir",
                   default=str(WORKSPACE_ROOT / "data" / "generated_splits" / "arcdoc_final"),
                   help="Diretório de saída")
    p.add_argument("--n-variants",    type=int, default=3,
                   help="Variantes augmentadas por imagem (mín. 2 para pares genuínos)")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--dry-run",       action="store_true",
                   help="Mostra estatísticas sem gerar arquivos")
    args = p.parse_args()

    if args.n_variants < 2:
        print("--n-variants deve ser >= 2 para criar pares genuínos.")
        sys.exit(1)

    base_image_dir = Path(args.base_image_dir)
    output_dir     = Path(args.output_dir)
    train_aug_dir  = output_dir / "images_train"
    val_aug_dir    = output_dir / "images_val"
    rng            = random.Random(args.seed)

    # ── Treino: combina todos os splits ──────────────────────────────────────
    print("Lendo pares de treino...")
    all_train_rows: List[dict] = []
    for split_dir in args.splits_dirs:
        csv_path = Path(split_dir) / "train_pairs.csv"
        if not csv_path.exists():
            print(f"  [WARN] Não encontrado: {csv_path}")
            continue
        rows = _read_csv(csv_path)
        print(f"  {csv_path.parent.name}: {len(rows)} pares")
        all_train_rows.extend(rows)

    train_by_class = _collect_images_by_class(all_train_rows)
    print(f"  Total treino: {len(all_train_rows)} pares | "
          f"{sum(len(v) for v in train_by_class.values())} imagens únicas | "
          f"{len(train_by_class)} classes")

    # ── Validação: split de referência ───────────────────────────────────────
    val_csv_path = Path(args.val_split_dir) / "validation_pairs.csv"
    if not val_csv_path.exists():
        print(f"ERRO: validation_pairs.csv não encontrado em {args.val_split_dir}")
        sys.exit(1)

    val_rows = _read_csv(val_csv_path)
    val_by_class = _collect_images_by_class(val_rows)
    print(f"\n  Validação: {len(val_rows)} pares | "
          f"{sum(len(v) for v in val_by_class.values())} imagens únicas | "
          f"{len(val_by_class)} classes")

    # Remove do treino qualquer par que contenha classes de validação
    val_classes = set(val_by_class.keys())
    overlap = val_classes & set(train_by_class.keys())
    if overlap:
        before = len(all_train_rows)
        all_train_rows = [
            r for r in all_train_rows
            if r["class_a_name"] not in val_classes and r["class_b_name"] not in val_classes
        ]
        # Recalcula by_class sem as classes filtradas
        train_by_class = _collect_images_by_class(all_train_rows)
        print(f"  ⚠️  {len(overlap)} classes de validação removidas do treino.")
        print(f"     Pares filtrados: {before} → {len(all_train_rows)}")
        print(f"     Classes de treino restantes: {len(train_by_class)}")
    else:
        print(f"  ✅ Nenhuma sobreposição de classes treino/validação.")

    if args.dry_run:
        n_train_imgs = sum(len(v) for v in train_by_class.values())
        n_val_imgs   = sum(len(v) for v in val_by_class.values())
        print(f"\n[DRY RUN]")
        print(f"  Imagens de treino a augmentar: {n_train_imgs} × {args.n_variants} = {n_train_imgs * args.n_variants}")
        print(f"  Imagens de val a augmentar:    {n_val_imgs}   × {args.n_variants} = {n_val_imgs * args.n_variants}")
        return

    # ── Gera augmentation de treino ──────────────────────────────────────────
    print(f"\nGerando variantes de treino ({args.n_variants} por imagem)...")
    train_aug_map = generate_augmented_images(
        by_class=train_by_class,
        base_image_dir=base_image_dir,
        aug_dir=train_aug_dir,
        n_variants=args.n_variants,
        seed_base=args.seed,
        label="[TREINO] ",
    )

    print(f"\nCriando pares de treino augmentados...")
    train_pairs = build_pairs(train_by_class, train_aug_map, rng)
    train_out   = output_dir / "train_pairs.csv"
    _write_csv(train_pairs, train_out)
    genuine_tr  = sum(1 for r in train_pairs if r["is_equal"] == 1)
    print(f"  Genuínos: {genuine_tr}  |  Impostores: {len(train_pairs) - genuine_tr}  |  Total: {len(train_pairs)}")
    print(f"  CSV: {train_out}")

    # ── Gera augmentation de validação ───────────────────────────────────────
    print(f"\nGerando variantes de validação ({args.n_variants} por imagem)...")
    val_aug_map = generate_augmented_images(
        by_class=val_by_class,
        base_image_dir=base_image_dir,
        aug_dir=val_aug_dir,
        n_variants=args.n_variants,
        seed_base=args.seed + 99999,
        label="[VAL] ",
    )

    print(f"\nCriando pares de validação augmentados...")
    val_pairs = build_pairs(val_by_class, val_aug_map, random.Random(args.seed + 1))
    val_out   = output_dir / "validation_pairs.csv"
    _write_csv(val_pairs, val_out)
    genuine_val = sum(1 for r in val_pairs if r["is_equal"] == 1)
    print(f"  Genuínos: {genuine_val}  |  Impostores: {len(val_pairs) - genuine_val}  |  Total: {len(val_pairs)}")
    print(f"  CSV: {val_out}")

    print(f"\n✅ Dados ArcDoc prontos em: {output_dir}")
    print(f"   Imagens treino: {train_aug_dir}")
    print(f"   Imagens val:    {val_aug_dir}")


if __name__ == "__main__":
    main()
