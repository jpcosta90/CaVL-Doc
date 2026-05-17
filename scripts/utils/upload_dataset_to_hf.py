#!/usr/bin/env python3
"""
Upload do dataset final_split3 para o Hugging Face Hub.

Faz upload das imagens augmentadas (images_train/ e images_val/) e dos CSVs
de pares para um repositório de dataset no HF.

Uso:
    python scripts/utils/upload_dataset_to_hf.py

    # Com opções explícitas:
    python scripts/utils/upload_dataset_to_hf.py \
        --dataset-dir data/generated_splits/final_split3 \
        --repo-id Jpcosta90/cavl-doc-lacdip-split3 \
        --private
"""

import argparse
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = WORKSPACE_ROOT / "data" / "generated_splits" / "final_split3"
DEFAULT_REPO_ID = "Jpcosta90/cavl-doc-lacdip-split3"


def _build_dataset_card(dataset_dir: Path, repo_id: str) -> str:
    n_train = sum(1 for _ in (dataset_dir / "images_train").rglob("*.tif"))
    n_val   = sum(1 for _ in (dataset_dir / "images_val").rglob("*.tif"))
    return f"""\
---
license: other
task_categories:
  - image-to-image
  - feature-extraction
language:
  - en
tags:
  - document-understanding
  - document-retrieval
  - metric-learning
  - zero-shot-learning
  - la-cdip
  - cavl-doc
size_categories:
  - 10K<n<100K
---

# CaVL-Doc — LA-CDIP Final Split 3 (Augmented)

Dataset de treino e validação para o modelo definitivo CaVL-Doc, baseado no
**Split 3** do protocolo ZSL (Zero-Shot Learning) sobre o LA-CDIP.

## Estrutura

| Conjunto | Imagens | Pares |
|---|---|---|
| Treino (`images_train/`) | {n_train:,} variantes augmentadas | 34.960 |
| Validação (`images_val/`) | {n_val:,} variantes augmentadas | 10.500 |

- **120 classes** para treino · **24 classes novel** para validação (sem sobreposição)
- Cada imagem original gera **5 variantes** com o pipeline de augmentação offline
- Augmentações: rotação ±15°, perspectiva 2–7%, brilho/contraste ±25%, blur,
  nitidez, ruído gaussiano, compressão JPEG, resolução baixa, crop/padding,
  amarelamento, salt & pepper

## Arquivos

```
train_pairs.csv        — pares de treino (file_a, file_b, is_equal, class)
validation_pairs.csv   — pares de validação ZSL (24 classes novel)
images_train/          — imagens augmentadas de treino organizadas por classe
images_val/            — imagens augmentadas de validação organizadas por classe
```

## Como baixar

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    local_dir="data/generated_splits/final_split3",
)
```

## Origem

Gerado a partir do LA-CDIP (Tobacco Document Library) com o script
`scripts/utils/prepare_arcdoc_training.py` do repositório CaVL-Doc.
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Upload do dataset final_split3 para o HF Hub.")
    p.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR),
                   help=f"Diretório do dataset (default: {DEFAULT_DATASET_DIR})")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"ID do repositório HF dataset (default: {DEFAULT_REPO_ID})")
    p.add_argument("--private", action="store_true", default=True,
                   help="Repositório privado (default: True)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"❌ Dataset não encontrado: {dataset_dir}")
        sys.exit(1)

    train_csv = dataset_dir / "train_pairs.csv"
    val_csv   = dataset_dir / "validation_pairs.csv"
    img_train = dataset_dir / "images_train"
    img_val   = dataset_dir / "images_val"

    for path in [train_csv, val_csv, img_train, img_val]:
        if not path.exists():
            print(f"❌ Esperado mas não encontrado: {path}")
            sys.exit(1)

    n_train = sum(1 for _ in img_train.rglob("*.tif"))
    n_val   = sum(1 for _ in img_val.rglob("*.tif"))
    size_gb = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file()) / 1e9

    print(f"{'='*60}")
    print(f"  Dataset    : {dataset_dir}")
    print(f"  Repo HF    : {args.repo_id}")
    print(f"  Imagens    : {n_train:,} treino + {n_val:,} val")
    print(f"  Tamanho    : {size_gb:.1f} GB")
    print(f"  Privado    : {args.private}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY-RUN] Nenhum arquivo enviado.")
        return

    from huggingface_hub import HfApi, create_repo
    import os

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or True
    api = HfApi(token=token)

    print("\nCriando repositório de dataset (se não existir)...")
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    print("Gerando README.md no diretório do dataset...")
    readme_path = dataset_dir / "README.md"
    readme_path.write_text(_build_dataset_card(dataset_dir, args.repo_id), encoding="utf-8")

    n_train = sum(1 for _ in img_train.rglob("*.tif"))
    n_val   = sum(1 for _ in img_val.rglob("*.tif"))
    print(f"\nEnviando dataset completo ({n_train + n_val:,} arquivos via upload_large_folder)...")
    print("  (inclui CSVs, images_train/ e images_val/)")
    api.upload_large_folder(
        folder_path=str(dataset_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        print_report=True,
        print_report_every=30,
    )

    print(f"\n✅ Dataset enviado: https://huggingface.co/datasets/{args.repo_id}")
    print(f"\nPara baixar em outro servidor:")
    print(f"  from huggingface_hub import snapshot_download")
    print(f"  snapshot_download(repo_id='{args.repo_id}', repo_type='dataset',")
    print(f"                    local_dir='data/generated_splits/final_split3')")


if __name__ == "__main__":
    main()
