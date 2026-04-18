#!/usr/bin/env python3
"""
Upload do melhor checkpoint CaVL-Doc para o Hugging Face Hub.

Busca automaticamente o melhor run subcenter_cosface no W&B (menor EER na fase2_profON)
ou aceita um caminho direto via --checkpoint-path.

Uso básico (auto-select via W&B):
    python scripts/utils/upload_to_hf_hub.py \
        --repo-id seu-usuario/cavl-doc-lacdip \
        --wandb-project CaVL-Doc_LA-CDIP_Sprint3_Staged5x5

Uso com checkpoint explícito:
    python scripts/utils/upload_to_hf_hub.py \
        --repo-id seu-usuario/cavl-doc-lacdip \
        --checkpoint-path /mnt/large/checkpoints/<run>/best_siam.pt \
        --eer 0.0312
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Auto-select por leitura local dos checkpoints
# ---------------------------------------------------------------------------

def _find_best_checkpoint_local(
    checkpoint_root: Path,
    loss_filter: str = "subcenter_cosface",
    stage_filter: str = "Sprint3",
    phase_filter: str = "fase2_profON",
) -> tuple[Path, float, str]:
    """
    Escaneia checkpoint_root procurando por best_siam.pt cujo diretório
    contenha loss_filter, stage_filter e phase_filter (se fornecidos).
    Lê o EER diretamente do arquivo — sem depender de W&B.
    """
    candidates = []

    for best_pt in checkpoint_root.rglob("best_siam.pt"):
        run_name = best_pt.parent.name
        if loss_filter.lower() not in run_name.lower():
            continue
        if stage_filter and stage_filter not in run_name:
            continue
        if phase_filter and phase_filter not in run_name:
            continue

        try:
            ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
            eer = ckpt.get("metrics", {}).get("eer")
            if eer is None:
                continue
            candidates.append((float(eer), best_pt, run_name))
        except Exception as e:
            print(f"  [SKIP] {run_name} — erro ao ler checkpoint: {e}")
            continue

    if not candidates:
        filters = f"loss='{loss_filter}'"
        if stage_filter:
            filters += f", stage='{stage_filter}'"
        raise FileNotFoundError(
            f"Nenhum best_siam.pt encontrado em '{checkpoint_root}' com {filters}."
        )

    candidates.sort(key=lambda x: x[0])

    print(f"\n  Encontrados {len(candidates)} checkpoint(s) com '{loss_filter}':")
    for eer, path, name in candidates[:5]:
        print(f"    EER={eer*100:.2f}%  {name}")
    if len(candidates) > 5:
        print(f"    ... e mais {len(candidates)-5}")

    best_eer, best_ckpt, best_run_name = candidates[0]
    return best_ckpt, best_eer, best_run_name


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------

def _build_model_card(
    run_name: str,
    eer: float,
    config: dict,
    loss_type: str,
    dataset: str,
) -> str:
    proj_out = config.get("projection_output_dim", config.get("proj_out", "?"))
    cut_layer = config.get("cut_layer", "?")
    pooler = config.get("pooler_type", "attention")
    head = config.get("head_type", "mlp")
    margin = config.get("margin", "?")
    scale = config.get("scale", "?")
    k = config.get("num_sub_centers", config.get("num_sub_centers", "?"))

    return f"""\
---
language: pt
license: mit
tags:
  - document-understanding
  - document-retrieval
  - metric-learning
  - siamese-network
  - internvl
  - cavl-doc
datasets:
  - {dataset}
metrics:
  - eer
model-index:
  - name: CaVL-Doc ({loss_type})
    results:
      - task:
          type: document-retrieval
        dataset:
          name: {dataset}
          type: {dataset.lower()}
        metrics:
          - type: eer
            value: {eer:.4f}
---

# CaVL-Doc — {loss_type} ({dataset})

**CaVL-Doc** é um modelo de embeddings para recuperação de documentos visuais,
treinado com aprendizado métrico supervisionado e seleção de exemplos difíceis via
Reinforcement Learning (Professor Network).

## Arquitetura

| Componente | Valor |
|---|---|
| Backbone | InternVL3-2B (`OpenGVLab/InternVL3-2B`) |
| Cut layer | {cut_layer} |
| Pooler | {pooler} |
| Head | {head} |
| Embedding dim | {proj_out} |
| Loss | {loss_type} (m={margin}, s={scale}, k={k}) |

## Performance

| Dataset | EER |
|---|---|
| {dataset} | **{eer*100:.2f}%** |

Run de origem: `{run_name}`

## Instalação

```bash
pip install cavl-doc huggingface_hub
```

## Uso

```python
import torch
from huggingface_hub import hf_hub_download
from cavl_doc.models.backbone_loader import load_model
from cavl_doc.utils.checkpointing import load_trained_siamese

device = "cuda" if torch.cuda.is_available() else "cpu"

# Baixa os pesos fine-tuned (backbone carregado automaticamente do HF Hub)
ckpt_path = hf_hub_download(repo_id="{run_name.split('/')[0] if '/' in run_name else 'seu-usuario'}/cavl-doc-lacdip", filename="best_siam.pt")
backbone, _, tokenizer, _, _ = load_model("InternVL3-2B")
model = load_trained_siamese(ckpt_path, backbone, tokenizer, device)
model.eval()

# Inferência
from PIL import Image
from cavl_doc.data.transforms import load_and_preprocess_image  # ajuste conforme seu pipeline

img = load_and_preprocess_image("documento.png")
with torch.no_grad():
    embedding = model(images=img)  # [1, {proj_out}]
```

## Treinamento

Treinado com o pipeline CaVL-Doc Sprint 3:
- **Fase 1** (10 épocas): treino do student sem professor, `pool = batch = 8`
- **Fase 2** (5 épocas): professor ativo com shadow warmup de 1 época,
  `pool = 8`, `batch = 4`, seleção por hard mining via policy gradient

## Citação

```bibtex
@misc{{cavldoc2026,
  title  = {{CaVL-Doc: Curriculum and Active-learning Vision-Language for Document Retrieval}},
  author = {{Costa, João Paulo}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{run_name}}}
}}
```
"""


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload(
    repo_id: str,
    checkpoint_path: Path,
    eer: float,
    run_name: str,
    loss_type: str,
    dataset: str,
    private: bool,
    dry_run: bool,
) -> None:
    from huggingface_hub import HfApi, create_repo

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("config", {})

    print(f"\n{'='*60}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Run        : {run_name}")
    print(f"  EER        : {eer*100:.2f}%")
    print(f"  Repo HF    : {repo_id}")
    print(f"  Privado    : {private}")
    print(f"{'='*60}\n")

    if dry_run:
        print("[DRY-RUN] Nenhum arquivo enviado.")
        return

    import os
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or True
    api = HfApi(token=token)

    print("Criando repositório (se não existir)...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    except Exception as e:
        print(f"  Aviso: não foi possível criar o repo automaticamente ({e}).")
        print("  Assumindo que o repositório já existe e prosseguindo com o upload...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Pesos — copia só o necessário para inferência (sem optimizer/professor)
        inference_ckpt = {
            "epoch":      ckpt.get("epoch"),
            "metrics":    ckpt.get("metrics"),
            "config":     config,
            "siam_pool":  ckpt["siam_pool"],
            "siam_head":  ckpt["siam_head"],
        }
        if "backbone_trainable" in ckpt:
            inference_ckpt["backbone_trainable"] = ckpt["backbone_trainable"]

        weights_path = tmp / "best_siam.pt"
        torch.save(inference_ckpt, weights_path)
        print(f"Peso salvo localmente: {weights_path} ({weights_path.stat().st_size / 1e6:.1f} MB)")

        # 2. Config JSON
        config_path = tmp / "cavl_config.json"
        config_path.write_text(json.dumps(config, indent=2, default=str))

        # 3. Model card
        readme_path = tmp / "README.md"
        readme_path.write_text(
            _build_model_card(run_name, eer, config, loss_type, dataset)
        )

        # 4. Upload
        print("\nEnviando arquivos para o HF Hub...")
        for fpath in [weights_path, config_path, readme_path]:
            print(f"  -> {fpath.name}")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fpath.name,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )

    print(f"\n✅ Upload concluído: https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload do melhor checkpoint CaVL-Doc para o HF Hub.")

    p.add_argument("--repo-id", required=True,
                   help="ID do repositório HF (ex: seu-usuario/cavl-doc-lacdip)")

    # Checkpoint manual ou auto-select via W&B
    p.add_argument("--checkpoint-path", default=None,
                   help="Caminho direto para best_siam.pt (ignora busca W&B).")
    p.add_argument("--eer", type=float, default=None,
                   help="EER do checkpoint (obrigatório se --checkpoint-path for usado).")

    # Auto-select W&B
    p.add_argument("--loss-filter", default="subcenter_cosface",
                   help="Filtro de loss (substring do nome do diretório de checkpoint).")
    p.add_argument("--stage-filter", default="Sprint3",
                   help="Filtro de stage (substring do nome do run). Default: 'Sprint3'.")
    p.add_argument("--phase-filter", default="fase2_profON",
                   help="Filtro de fase (substring do nome do run). Default: 'fase2_profON'.")
    p.add_argument("--checkpoint-root", default=None,
                   help="Raiz dos checkpoints (default: /mnt/large/checkpoints ou ./checkpoints).")

    p.add_argument("--dataset", default="LA-CDIP")
    p.add_argument("--private", action="store_true", default=False,
                   help="Cria o repositório como privado.")
    p.add_argument("--dry-run", action="store_true",
                   help="Simula sem enviar nenhum arquivo.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.checkpoint_path:
        # Caminho explícito
        ckpt_path = Path(args.checkpoint_path)
        if not ckpt_path.exists():
            print(f"❌ Checkpoint não encontrado: {ckpt_path}")
            sys.exit(1)
        if args.eer is None:
            print("❌ --eer é obrigatório quando --checkpoint-path é fornecido.")
            sys.exit(1)
        eer = args.eer
        run_name = ckpt_path.parent.name
    else:
        # Auto-select por leitura local dos checkpoints
        if args.checkpoint_root:
            ckpt_root = Path(args.checkpoint_root)
        elif Path("/mnt/large/checkpoints").exists():
            ckpt_root = Path("/mnt/large/checkpoints")
        else:
            ckpt_root = WORKSPACE_ROOT / "checkpoints"

        stage_info = f" / stage='{args.stage_filter}'" if args.stage_filter else ""
        phase_info = f" / phase='{args.phase_filter}'" if args.phase_filter else ""
        print(f"Buscando melhor checkpoint '{args.loss_filter}'{stage_info}{phase_info} em {ckpt_root}...")
        ckpt_path, eer, run_name = _find_best_checkpoint_local(
            checkpoint_root=ckpt_root,
            loss_filter=args.loss_filter,
            stage_filter=args.stage_filter,
            phase_filter=args.phase_filter,
        )
        print(f"\n  ✅ Melhor run : {run_name}")
        print(f"     EER        : {eer*100:.2f}%")
        print(f"     Checkpoint : {ckpt_path}")

    upload(
        repo_id=args.repo_id,
        checkpoint_path=ckpt_path,
        eer=eer,
        run_name=run_name,
        loss_type=args.loss_filter,
        dataset=args.dataset,
        private=args.private,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
