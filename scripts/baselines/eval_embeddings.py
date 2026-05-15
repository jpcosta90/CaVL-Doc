#!/usr/bin/env python3
"""
Baseline de similaridade de documentos via embeddings.

Métodos disponíveis (--method):

  jina-v4          jinaai/jina-embeddings-v4 — modelo multimodal treinado para
                   retrieval de documentos; embeds cada imagem separadamente e
                   calcula cosseno entre os vetores.

  internvl3-in     InternVL3-2B — embeddings dos tokens de entrada (camada 0 do
                   LM), mean-pool sobre todos os tokens visuais/textuais.

  internvl3-out    InternVL3-2B — embeddings de saída (hidden state na
                   cut_layer=27), mean-pool; mesma representação usada no treino
                   do CaVL-Doc.

  pixel-cosine     Distância de cosseno entre vetores de pixels (imagens
                   redimensionadas para 448×448, flatten em float32).

  mm-embed         nvidia/MM-Embed — modelo multimodal da NVIDIA baseado em
                   NV-Embed-v2; recebe imagem + texto vazio e retorna embedding
                   normalizado.

  pixel-l2         Distância L2 normalizada entre vetores de pixels — convertida
                   para similaridade via 1 / (1 + d).

Uso:
  python scripts/baselines/eval_embeddings.py \\
      --method jina-v4 \\
      --data-root /mnt/data/la-cdip \\
      --base-image-dir /mnt/data/la-cdip/data \\
      --splits all \\
      --gpu-id 0

  python scripts/baselines/eval_embeddings.py \\
      --method pixel-cosine \\
      --data-root /mnt/data/la-cdip \\
      --base-image-dir /mnt/data/la-cdip/data \\
      --splits all \\
      --no-wandb
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PREP_SCRIPT    = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_protocol_split.py"

WANDB_ENTITY  = "jpcosta1990-university-of-brasilia"
WANDB_PROJECT = "CaVL-Doc_LA-CDIP_Embedding_Baseline"

EMBEDDING_PROMPT = "<image> Analyze this document"
CUT_LAYER        = 27
MODEL_NAME       = "InternVL3-2B"
PROJ_OUT_DIM     = 1536

PIXEL_SIZE = 448  # resize target for pixel baselines


# ---------------------------------------------------------------------------
# Cache cleanup
# ---------------------------------------------------------------------------

def _delete_model_cache(model_id: str) -> None:
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        deleted_mb = 0.0
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                for revision in repo.revisions:
                    for f in revision.files:
                        p = Path(f.file_path)
                        if p.exists():
                            deleted_mb += p.stat().st_size / 1024 / 1024
                            p.unlink()
                snapshot_dir = Path(repo.repo_path)
                if snapshot_dir.exists():
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
        print(f"  Cache removido: {model_id} (~{deleted_mb:.0f} MB)")
    except Exception as e:
        print(f"  ⚠️  Falha ao remover cache de {model_id}: {e}")


# ---------------------------------------------------------------------------
# Split preparation
# ---------------------------------------------------------------------------

def _prepare_split(data_root: str, split_idx: int) -> Path:
    split_dir = WORKSPACE_ROOT / "data" / "generated_splits" / f"eval_split{split_idx}"
    val_csv   = split_dir / "validation_pairs.csv"
    if val_csv.exists():
        return val_csv
    cmd = [
        sys.executable, str(PREP_SCRIPT),
        "--data-root",     data_root,
        "--output-dir",    str(split_dir),
        "--val-split-idx", str(split_idx),
        "--protocol",      "zsl",
    ]
    print(f"[PREP] split {split_idx}")
    subprocess.run(cmd, check=True)
    return val_csv


# ---------------------------------------------------------------------------
# Pixel baseline
# ---------------------------------------------------------------------------

def _img_to_vector(img: Image.Image) -> np.ndarray:
    img_r = img.resize((PIXEL_SIZE, PIXEL_SIZE)).convert("RGB")
    return np.array(img_r, dtype=np.float32).flatten()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pixel_scores(val_csv: Path, base_image_dir: str,
                  metric: str, limit: Optional[int]) -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_csv(val_csv)
    if limit:
        df = df.head(limit)
    rows = []
    for i, row in df.iterrows():
        try:
            va = _img_to_vector(Image.open(Path(base_image_dir) / row["file_a_path"]).convert("RGB"))
            vb = _img_to_vector(Image.open(Path(base_image_dir) / row["file_b_path"]).convert("RGB"))
            if metric == "cosine":
                score = _cosine(va, vb)
            else:  # l2 → similarity
                d = float(np.linalg.norm(va - vb))
                score = 1.0 / (1.0 + d / 1e4)  # normalise scale
            rows.append({"file_a_path": row["file_a_path"],
                         "file_b_path": row["file_b_path"],
                         "is_equal":    int(row["is_equal"]),
                         "similarity_score": score})
            if len(rows) % 100 == 0:
                print(f"  [{len(rows)}/{len(df)}]")
        except Exception as e:
            print(f"  [ERR] par {i}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Jina Embeddings v4
# ---------------------------------------------------------------------------

class _JinaV4Embedder:
    MODEL_ID = "jinaai/jina-embeddings-v4"

    def __init__(self, device: str):
        from transformers import AutoModel
        print(f"Carregando {self.MODEL_ID}...")
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to(device).eval()
        self.device = device

    def embed(self, img: Image.Image) -> np.ndarray:
        import contextlib, io
        # Jina abre uma barra tqdm interna por chamada; suprimimos via redirect_stderr.
        with contextlib.redirect_stderr(io.StringIO()):
            if hasattr(self.model, "encode_image"):
                vec = self.model.encode_image([img], task="retrieval")
            else:
                vec = self.model.encode([img], task="retrieval.passage", truncate_dim=None)
        return vec[0].cpu().float().numpy()

    def scores(self, val_csv: Path, base_image_dir: str,
               limit: Optional[int]) -> "pd.DataFrame":
        import pandas as pd
        from tqdm import tqdm
        df = pd.read_csv(val_csv)
        if limit:
            df = df.head(limit)
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  jina-v4", ncols=90):
            try:
                img_a = Image.open(Path(base_image_dir) / row["file_a_path"]).convert("RGB")
                img_b = Image.open(Path(base_image_dir) / row["file_b_path"]).convert("RGB")
                ea, eb = self.embed(img_a), self.embed(img_b)
                score  = _cosine(ea, eb)
                rows.append({"file_a_path": row["file_a_path"],
                             "file_b_path": row["file_b_path"],
                             "is_equal":    int(row["is_equal"]),
                             "similarity_score": score})
            except Exception as e:
                tqdm.write(f"  [ERR] {row['file_a_path']}: {e}")
        return pd.DataFrame(rows)

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
        _delete_model_cache(self.MODEL_ID)


# ---------------------------------------------------------------------------
# InternVL3 embedding extractor
# ---------------------------------------------------------------------------

class _InternVL3Embedder:
    MODEL_ID = f"OpenGVLab/{MODEL_NAME}"

    def __init__(self, device: str, layer: str):
        """layer: 'input' (layer 0) or 'output' (cut_layer 27)"""
        from cavl_doc.models.backbone_loader import load_model, warm_up_model
        print(f"Carregando {self.MODEL_ID} (layer={layer})...")
        backbone, _, tokenizer, _, _ = load_model(
            model_name=MODEL_NAME,
            adapter_path=None,
            load_in_4bit=False,
            projection_output_dim=PROJ_OUT_DIM,
        )
        backbone.requires_grad_(False)
        warm_up_model(backbone, tokenizer)
        self.backbone  = backbone
        self.tokenizer = tokenizer
        self.device    = device
        self.layer     = layer

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding
        from cavl_doc.data.transforms import dynamic_preprocess
        from torchvision import transforms

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=6)
        pixel_values = torch.stack([tfm(t) for t in tiles]).to(torch.bfloat16)

        out = prepare_inputs_for_multimodal_embedding(
            self.backbone, self.tokenizer, pixel_values, EMBEDDING_PROMPT
        )
        input_ids    = out["input_ids"].to(self.device)
        pixel_values = out["pixel_values"].to(self.device, dtype=torch.bfloat16)
        image_flags  = out["image_flags"].to(self.device)

        result = self.backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_flags=image_flags,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = result.hidden_states

        if self.layer == "input":
            # Layer 0: embedding layer output (before transformer blocks)
            h = hidden[0]
        else:
            # Output: cut_layer (same as training)
            lm  = self.backbone.language_model.model
            idx = CUT_LAYER + 1 if len(hidden) == (len(lm.layers) + 1) else CUT_LAYER
            h   = hidden[idx]

        # Mean-pool over sequence length → [hidden_dim]
        vec = h.mean(dim=1).squeeze(0).float().cpu().numpy()
        return vec

    def scores(self, val_csv: Path, base_image_dir: str,
               limit: Optional[int]) -> "pd.DataFrame":
        import pandas as pd
        df = pd.read_csv(val_csv)
        if limit:
            df = df.head(limit)
        rows = []
        for i, row in df.iterrows():
            try:
                img_a  = Image.open(Path(base_image_dir) / row["file_a_path"]).convert("RGB")
                img_b  = Image.open(Path(base_image_dir) / row["file_b_path"]).convert("RGB")
                ea, eb = self.embed(img_a), self.embed(img_b)
                score  = _cosine(ea, eb)
                rows.append({"file_a_path": row["file_a_path"],
                             "file_b_path": row["file_b_path"],
                             "is_equal":    int(row["is_equal"]),
                             "similarity_score": score})
                if len(rows) % 50 == 0:
                    print(f"  [{len(rows)}/{len(df)}] score={score:.4f}")
            except Exception as e:
                print(f"  [ERR] par {i}: {e}")
        return pd.DataFrame(rows)

    def cleanup(self):
        del self.backbone
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# NVIDIA MM-Embed
# ---------------------------------------------------------------------------

class _MMEmbedEmbedder:
    MODEL_ID = "nvidia/MM-Embed"

    def __init__(self, device: str):
        import importlib, types
        from huggingface_hub import snapshot_download
        print(f"Carregando {self.MODEL_ID}...")

        model_path = snapshot_download(self.MODEL_ID)

        # Carrega o módulo customizado diretamente, sem passar pelo AutoModel.register
        # que conflita com LlavaNextConfig já registrado em versões novas do transformers.
        spec = importlib.util.spec_from_file_location(
            "modeling_nvmmembed",
            f"{model_path}/modeling_nvmmembed.py",
        )
        mod = types.ModuleType(spec.name)

        # Monkey-patch AutoModel.register para ignorar conflitos de registro
        from transformers import AutoModel as _AutoModel
        _orig_register = _AutoModel.register.__func__
        def _register_noop(cls, config_class, model_class, exist_ok=False):
            _orig_register(cls, config_class, model_class, exist_ok=True)
        _AutoModel.register = classmethod(_register_noop)

        spec.loader.exec_module(mod)

        _AutoModel.register = classmethod(_orig_register)  # restaura

        NVMMEmbedModel = mod.NVMMEmbedModel
        self.model = NVMMEmbedModel.from_pretrained(model_path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, img: Image.Image) -> np.ndarray:
        result = self.model.encode(
            [{"txt": "", "img": img}], max_length=4096
        )
        vec = result["hidden_states"][0]
        if isinstance(vec, torch.Tensor):
            return vec.cpu().float().numpy()
        return np.array(vec, dtype=np.float32)

    def scores(self, val_csv: Path, base_image_dir: str,
               limit: Optional[int]) -> "pd.DataFrame":
        import pandas as pd
        from tqdm import tqdm
        df = pd.read_csv(val_csv)
        if limit:
            df = df.head(limit)
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="  mm-embed", ncols=90):
            try:
                img_a = Image.open(Path(base_image_dir) / row["file_a_path"]).convert("RGB")
                img_b = Image.open(Path(base_image_dir) / row["file_b_path"]).convert("RGB")
                ea, eb = self.embed(img_a), self.embed(img_b)
                score  = _cosine(ea, eb)
                rows.append({"file_a_path": row["file_a_path"],
                             "file_b_path": row["file_b_path"],
                             "is_equal":    int(row["is_equal"]),
                             "similarity_score": score})
            except Exception as e:
                tqdm.write(f"  [ERR] {row['file_a_path']}: {e}")
        return pd.DataFrame(rows)

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()
        _delete_model_cache(self.MODEL_ID)


# ---------------------------------------------------------------------------
# EER
# ---------------------------------------------------------------------------

def _compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2), float(thr[idx])


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def _log_wandb(method: str, split_label: str, eer: float, thr: float,
               n_pairs: int, wandb_entity: str, wandb_project: str) -> None:
    try:
        import wandb
        run_name = f"Emb_{method}_split{split_label}"
        run = wandb.init(
            entity=wandb_entity, project=wandb_project, name=run_name,
            config={"method": method, "split": split_label, "n_pairs": n_pairs},
            reinit=True,
        )
        wandb.log({"test/eer": eer, "test/threshold": thr, "test/n_pairs": n_pairs})
        run.finish()
        print(f"  W&B logged: {run_name}")
    except Exception as e:
        print(f"  ⚠️  W&B log falhou: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_splits(s: str) -> List[int]:
    if s.lower() == "all":
        return [0, 1, 2, 3, 4, 5]
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline de embeddings para documentos.")
    p.add_argument("--method", required=True,
                   choices=["jina-v4", "internvl3-in", "internvl3-out",
                            "mm-embed", "pixel-cosine", "pixel-l2"],
                   help="Método de embedding a avaliar")
    p.add_argument("--data-root",      default=None,
                   help="Raiz do dataset LA-CDIP (necessário para preparar splits)")
    p.add_argument("--base-image-dir", required=True,
                   help="Diretório base das imagens LA-CDIP")
    p.add_argument("--splits",         default="5",
                   help="Splits a avaliar: '5', '0,1,2,3,4,5' ou 'all' (default: 5)")
    p.add_argument("--output-dir",     default=None,
                   help="Diretório para salvar CSVs (default: results/emb_baseline/<method>)")
    p.add_argument("--gpu-id",         type=int, default=None)
    p.add_argument("--wandb-entity",   default=WANDB_ENTITY)
    p.add_argument("--wandb-project",  default=WANDB_PROJECT)
    p.add_argument("--no-wandb",       action="store_true")
    p.add_argument("--limit",          type=int, default=None,
                   help="Limitar a N pares por split (para teste rápido)")
    p.add_argument("--eval-csv",             default=None,
                   help="CSV custom de pares (sobrescreve a resolução automática para split 5)")
    p.add_argument("--split-label",          default=None,
                   help="Label do split para W&B e arquivos de saída (ex: '5_synthetic')")
    p.add_argument("--splits-dir",           default=None,
                   help="Diretório base com splits pré-construídos (ex: data/generated_splits). "
                        "Quando fornecido, ignora --data-root e a lógica de preparação automática.")
    p.add_argument("--split-name-template",  default="RVL-CDIP_zsl_split_{idx}",
                   help="Template do subdiretório de cada split, relativo a --splits-dir "
                        "(default: RVL-CDIP_zsl_split_{idx})")
    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Método: {args.method}")

    splits = _parse_splits(args.splits)
    print(f"Splits: {splits}")

    out_dir = Path(args.output_dir) if args.output_dir else \
              WORKSPACE_ROOT / "results" / "emb_baseline" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build embedder / pixel method
    is_pixel = args.method.startswith("pixel")
    embedder = None
    if not is_pixel:
        if args.method == "jina-v4":
            embedder = _JinaV4Embedder(device)
        elif args.method == "internvl3-in":
            embedder = _InternVL3Embedder(device, layer="input")
        elif args.method == "internvl3-out":
            embedder = _InternVL3Embedder(device, layer="output")
        elif args.method == "mm-embed":
            embedder = _MMEmbedEmbedder(device)

    import pandas as pd
    summary_rows = []

    for split_idx in splits:
        print(f"\n{'='*60}")
        print(f"SPLIT {split_idx}")
        print(f"{'='*60}")

        # Prepare CSV
        if args.splits_dir:
            split_name = args.split_name_template.replace("{idx}", str(split_idx))
            val_csv    = Path(args.splits_dir) / split_name / "validation_pairs.csv"
            split_label = str(split_idx)
            if not val_csv.exists():
                print(f"  ⚠️  CSV não encontrado: {val_csv}. Pulando.")
                continue
        elif args.eval_csv and split_idx == 5:
            val_csv     = Path(args.eval_csv)
            split_label = args.split_label or "5_custom"
        elif split_idx == 5:
            val_csv = WORKSPACE_ROOT / "data" / "generated_splits" / \
                      "eval_test_split5" / "validation_pairs.csv"
            if not val_csv.exists():
                if not args.data_root:
                    print("  ⚠️  Split 5 CSV não encontrado e --data-root não fornecido.")
                    continue
                val_csv = _prepare_split(args.data_root, split_idx)
            split_label = str(split_idx)
        else:
            if not args.data_root:
                print(f"  ⚠️  --data-root necessário para split {split_idx}. Pulando.")
                continue
            val_csv     = _prepare_split(args.data_root, split_idx)
            split_label = str(split_idx)
        print(f"  CSV: {val_csv}")

        t0 = time.time()
        if is_pixel:
            metric = "cosine" if args.method == "pixel-cosine" else "l2"
            df_res = _pixel_scores(val_csv, args.base_image_dir, metric, args.limit)
        else:
            df_res = embedder.scores(val_csv, args.base_image_dir, args.limit)
        elapsed = time.time() - t0

        if len(df_res) < 10:
            print(f"  ⚠️  Pares insuficientes ({len(df_res)}). Pulando.")
            continue

        scores = df_res["similarity_score"].values
        labels = df_res["is_equal"].values
        eer, thr = _compute_eer(scores, labels)

        df_res["split"] = split_label
        pairs_csv = out_dir / f"split{split_label}_pairs.csv"
        df_res.to_csv(pairs_csv, index=False)

        summary_rows.append({
            "split":        split_label,
            "n_pairs":      len(df_res),
            "eer":          eer,
            "eer_pct":      round(eer * 100, 2),
            "threshold":    thr,
            "elapsed_min":  round(elapsed / 60, 1),
        })

        print(f"  EER={eer*100:.2f}%  threshold={thr:.4f}  ({elapsed/60:.1f} min)")
        print(f"  Resultados: {pairs_csv}")

        if not args.no_wandb:
            _log_wandb(args.method, split_label, eer, thr, len(df_res),
                       args.wandb_entity, args.wandb_project)

    # Summary
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        summary_csv = out_dir / "summary.csv"
        df_sum.to_csv(summary_csv, index=False)

        eers = df_sum["eer"].values
        print(f"\n{'='*60}")
        print(f"RESUMO — {args.method}")
        print(f"{'='*60}")
        print(df_sum[["split", "n_pairs", "eer_pct", "threshold"]].to_string(index=False))
        print(f"\n  Média:   {eers.mean()*100:.2f}%")
        print(f"  Std:     {eers.std()*100:.2f} pp")
        print(f"  Mediana: {np.median(eers)*100:.2f}%")
        print(f"\n  Sumário: {summary_csv}")

        if not args.no_wandb:
            try:
                import wandb
                agg = wandb.init(
                    entity=args.wandb_entity, project=args.wandb_project,
                    name=f"Emb_{args.method}_agg",
                    config={"method": args.method, "splits": splits, "type": "aggregate"},
                    reinit=True,
                )
                wandb.log({
                    "agg/eer_mean":   float(eers.mean()),
                    "agg/eer_std":    float(eers.std()),
                    "agg/eer_median": float(np.median(eers)),
                    "agg/n_splits":   len(eers),
                })
                agg.finish()
            except Exception as e:
                print(f"  ⚠️  W&B agg log falhou: {e}")

    # Cleanup model from GPU and disk
    if embedder is not None:
        embedder.cleanup()


if __name__ == "__main__":
    main()
