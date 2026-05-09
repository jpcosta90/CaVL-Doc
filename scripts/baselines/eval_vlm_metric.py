#!/usr/bin/env python3
"""
Baseline de similaridade via métrica numérica direta (sem justificativa textual).

Variante de eval_vlm_prompt.py onde o VLM responde apenas com um número inteiro
(0-100), sem JSON nem justificativa. Mais rápido e sem erros de parse.

Diferenças em relação a eval_vlm_prompt.py:
  - Prompt pede apenas o número, sem JSON/justificativa
  - --models aceita lista de modelos (ex: internvl3-2b,qwen3vl-2b)
  - W&B project separado: CaVL-Doc_LA-CDIP_VLM_Metric
  - Por padrão roda todos os splits (all)

Uso:
  python scripts/baselines/eval_vlm_metric.py \\
      --models internvl3-2b,qwen3vl-2b \\
      --base-image-dir /mnt/data/la-cdip/data \\
      --splits all \\
      --gpu-id 0
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Garante que downloads HF (XetHub) usem /tmp e não TMPDIR customizado cheio
if Path(tempfile.gettempdir()).stat().st_dev != Path("/tmp").stat().st_dev or \
        tempfile.gettempdir() != "/tmp":
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = "/tmp"

import transformers
transformers.logging.set_verbosity_error()

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

WANDB_ENTITY  = "jpcosta1990-university-of-brasilia"
WANDB_PROJECT = "CaVL-Doc_LA-CDIP_VLM_Metric"

SIMILARITY_PROMPT = """\
Image-1: <image>
Image-2: <image>

You are an AI assistant specialized in document analysis. Your task is to compare two company documents and assess their **visual similarity** based on their layout structure.

**Instructions:**
Analyze the two provided document images and measure their **visual similarity** based on:
- **Shapes and Elements:** Compare the presence of graphical components, tables, sections, headers, and any other visual elements.
- **Layout Consistency:** Evaluate the spatial arrangement of text blocks, margins, and alignments.
- **Content Type:** Ensure that both documents contain similar types of content (e.g., tables, forms, paragraphs), regardless of specific wording.

**Similarity Scoring:**
Assign a **similarity score** between **0 and 100**, where:
- **90-100** → **Nearly identical**: Documents have almost no visual differences.
- **70-89** → **Highly similar**: Documents share the same structure with minor variations (e.g., small alignment changes).
- **50-69** → **Moderately similar**: Key components remain, but there are noticeable structural differences.
- **30-49** → **Weak similarity**: Some elements are shared, but the overall layout is significantly different.
- **0-29** → **Completely different**: The documents do not share a recognizable visual structure.

**Output Format:**
Respond with **only** a single integer between 0 and 100. No text, no explanation, no JSON — just the number."""

MODEL_REGISTRY = {
    # ── InternVL3 ──
    "internvl3-2b":    "OpenGVLab/InternVL3-2B",          # ~6 GB BF16
    "internvl3-8b":    "OpenGVLab/InternVL3-8B",          # ~16 GB BF16 / ~8 GB 4-bit
    "internvl3-14b":   "OpenGVLab/InternVL3-14B",         # ~28 GB BF16 download / ~7 GB 4-bit (requer >28 GB livre em disco)

    # ── Qwen3-VL ──
    "qwen3vl-2b":      "Qwen/Qwen3-VL-2B-Instruct",       # ~5 GB BF16
    "qwen3vl-4b":      "Qwen/Qwen3-VL-4B-Instruct",       # ~9 GB BF16
    "qwen3vl-8b":      "Qwen/Qwen3-VL-8B-Instruct",       # ~16 GB BF16 / ~8 GB 4-bit

    # ── Requer transformers >= 5.5 (.venv_vlm5) ──
    "gemma4-e2b":      "google/gemma-4-E2B-it",           # ~5 GB BF16
    "gemma4-e4b":      "google/gemma-4-E4B-it",           # ~9 GB BF16

    # Modelos removidos (incompatíveis com RTX 4060 Ti / ambiente atual):
    # "minicpm-v45": openbmb/MiniCPM-V-4_5 — código trust_remote_code incompatível com BnB
    #   int4/int8; em BF16 puro (~16 GB) excede a VRAM (16 GB) sem margem para ativações.
    # "ministral-3b/8b": mistralai/Ministral-3-*B-Instruct-2512 — FP8 static com
    #   dequantize=False; exige kernels Hopper (sm_90+/H100). RTX 4060 Ti (sm_89) carrega
    #   os pesos mas ignora os scale factors de ativação → saída incoerente.
    # "pixtral-12b": mistralai/Pixtral-12B-2409 — sem AutoProcessor HF; requer
    #   mistral_common/vllm nativos.
}

# Modelos que requerem transformers >= 5.5 — usam .venv_vlm5 automaticamente
MODELS_VLM5 = {"gemma4-e2b", "gemma4-e4b"}
VENV_VLM5 = Path(__file__).resolve().parents[2] / ".venv_vlm5" / "bin" / "python"


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def _parse_score(text: str) -> Optional[float]:
    """Extract a single numeric score (0-100) from the model response."""
    m = re.search(r'\b(\d{1,3}(?:\.\d+)?)\b', text.strip())
    if m:
        val = float(m.group(1))
        if 0.0 <= val <= 100.0:
            return val
    return None


# ---------------------------------------------------------------------------
# Cache cleanup
# ---------------------------------------------------------------------------

def _delete_model_cache(model_id: str) -> None:
    """Delete the HuggingFace cached files for model_id to free disk space."""
    import shutil
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
                            size_mb = p.stat().st_size / 1024 / 1024
                            p.unlink()
                            deleted_mb += size_mb
                # Remove empty snapshot dirs
                snapshot_dir = Path(repo.repo_path)
                if snapshot_dir.exists():
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
                    deleted_mb += 0  # dirs already counted via files
        print(f"  Cache removido: {model_id} (~{deleted_mb:.0f} MB liberados)")
    except Exception as e:
        print(f"  ⚠️  Falha ao remover cache de {model_id}: {e}")


def _get_bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------

class _InternVLAdapter:
    """Adapter for InternVL3 family."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device,
            quantization_config=_get_bnb_config() if load_in_4bit else None,
        ).eval()

    def infer(self, img_a: Image.Image, img_b: Image.Image) -> str:
        from cavl_doc.data.transforms import dynamic_preprocess

        def _process(img: Image.Image) -> torch.Tensor:
            # max_num=4 (vs 6) to fit 14B 4-bit within 16 GB VRAM during inference
            tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=4)
            from torchvision import transforms
            tfm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            return torch.stack([tfm(t) for t in tiles])

        pv_a = _process(img_a)
        pv_b = _process(img_b)
        pixel_values = torch.cat([pv_a, pv_b], dim=0).to(torch.bfloat16).to(next(self.model.parameters()).device)
        num_patches = [len(pv_a), len(pv_b)]

        question = SIMILARITY_PROMPT
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config={"max_new_tokens": 256, "do_sample": False},
            num_patches_list=num_patches,
        )
        return response


class _QwenVLAdapter:
    """Adapter for Qwen3-VL family (uses AutoModelForImageTextToText for forward compat)."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=_get_bnb_config() if load_in_4bit else None,
        ).eval()

    def infer(self, img_a: Image.Image, img_b: Image.Image) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("Instale qwen_vl_utils: pip install qwen-vl-utils")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_a},
                {"type": "image", "image": img_b},
                {"type": "text",  "text": SIMILARITY_PROMPT.replace("Image-1: <image>\nImage-2: <image>\n\n", "")},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        out_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed  = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


class _MiniCPMAdapter:
    """Adapter for MiniCPM-V-4_5."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # MiniCPM custom ops são incompatíveis com BnB int4/int8; usa device_map="auto"
        # para CPU offload quando não cabe em VRAM (~16 GB BF16).
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

    def infer(self, img_a: Image.Image, img_b: Image.Image) -> str:
        prompt = SIMILARITY_PROMPT.replace("<image>", "").strip()
        msgs = [{"role": "user", "content": [img_a, img_b, prompt]}]
        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            do_sample=False,
        )
        return response


class _PixtralAdapter:
    """Adapter for Pixtral-12B (Mistral VLM) via transformers."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=_get_bnb_config() if load_in_4bit else None,
        ).eval()

    def infer(self, img_a: Image.Image, img_b: Image.Image) -> str:
        prompt_text = SIMILARITY_PROMPT.replace("Image-1: <image>\nImage-2: <image>\n\n", "")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=text, images=[img_a, img_b],
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        out_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = out_ids[0][inputs.input_ids.shape[-1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


class _Gemma4Adapter:
    """Adapter for Gemma 4 vision models (requer transformers >= 5.5)."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        from transformers import BitsAndBytesConfig
        # Vision tower deve ficar em BF16; NF4 no encoder silencia o processamento de imagens.
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector",
                                   "language_model.embed_tokens", "lm_head"],
        ) if load_in_4bit else None
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=quant_cfg,
        ).eval()

    def infer(self, img_a: Image.Image, img_b: Image.Image) -> str:
        prompt_text = SIMILARITY_PROMPT.replace("Image-1: <image>\nImage-2: <image>\n\n", "")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }]
        # Gemma4 processor expects PIL images passed separately (not embedded in content dict)
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=text, images=[img_a, img_b],
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        out_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = out_ids[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


class _MistralVLAdapter:
    """Adapter for Ministral-3 multimodal (requer transformers >= 5.5)."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoProcessor, Mistral3ForConditionalGeneration, AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Remove FP8 quantization tag from config so transformers doesn't treat this as a
        # pre-quantized checkpoint (transformers 5.x bug: hasattr returns True even when None).
        config.__dict__.pop("quantization_config", None)

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, fix_mistral_regex=True
        )
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_id,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=_get_bnb_config() if load_in_4bit else None,
        ).eval()

    def infer(self, img_a: Image.Image, img_b: Image.Image) -> str:
        prompt_text = SIMILARITY_PROMPT.replace("Image-1: <image>\nImage-2: <image>\n\n", "")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=text, images=[img_a, img_b],
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        out_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = out_ids[0][inputs.input_ids.shape[-1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)


def _build_adapter(model_key: str, device: str, load_in_4bit: bool = False):
    model_id = MODEL_REGISTRY[model_key]
    print(f"Carregando {model_id} {'(4-bit)' if load_in_4bit else '(bfloat16)'}...")
    kw = {"load_in_4bit": load_in_4bit}
    if model_key.startswith("internvl"):
        return _InternVLAdapter(model_id, device, **kw), model_id
    elif model_key.startswith("qwen"):
        return _QwenVLAdapter(model_id, device, **kw), model_id
    elif model_key.startswith("minicpm"):
        return _MiniCPMAdapter(model_id, device, **kw), model_id
    elif model_key.startswith("pixtral"):
        return _PixtralAdapter(model_id, device, **kw), model_id
    elif model_key.startswith("gemma4"):
        return _Gemma4Adapter(model_id, device, **kw), model_id
    elif model_key.startswith("ministral"):
        return _MistralVLAdapter(model_id, device, **kw), model_id
    else:
        raise ValueError(f"Adapter não implementado para {model_key}. "
                         f"Opções: {list(MODEL_REGISTRY)}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _load_image(path: str, base_dir: str) -> Image.Image:
    full = Path(base_dir) / path
    return Image.open(full).convert("RGB")


def _prepare_split(data_root: str, split_idx: int) -> Path:
    """Prepare validation CSV for a given split index via prepare_protocol_split.py."""
    split_dir = WORKSPACE_ROOT / "data" / "generated_splits" / f"eval_split{split_idx}"
    val_csv   = split_dir / "validation_pairs.csv"
    if val_csv.exists():
        return val_csv
    prep = WORKSPACE_ROOT / "scripts" / "utils" / "prepare_protocol_split.py"
    cmd = [
        sys.executable, str(prep),
        "--data-root",     data_root,
        "--output-dir",    str(split_dir),
        "--val-split-idx", str(split_idx),
        "--protocol",      "zsl",
    ]
    print(f"[PREP] split {split_idx}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return val_csv


def _run_baseline(adapter, val_csv: Path, base_image_dir: str,
                  limit: Optional[int] = None,
                  max_parse_errors: int = 20):
    """Run inference on all pairs. Returns a DataFrame with scores and labels."""
    import pandas as pd
    df = pd.read_csv(val_csv)
    if limit:
        df = df.head(limit)

    rows, parse_errors = [], 0
    pbar = tqdm(df.iterrows(), total=len(df), desc="  Pares", unit="par",
                dynamic_ncols=True, leave=True)
    for i, row in pbar:
        try:
            img_a    = _load_image(row["file_a_path"], base_image_dir)
            img_b    = _load_image(row["file_b_path"], base_image_dir)
            response = adapter.infer(img_a, img_b)
            score    = _parse_score(response)

            if score is None:
                parse_errors += 1
                pbar.write(f"  [WARN] Par {i}: score não parseado. Resposta: {response[:120]}")
                if parse_errors >= max_parse_errors:
                    pbar.write("  Muitos erros de parse. Encerrando split.")
                    break
                continue

            rows.append({
                "file_a_path":       row["file_a_path"],
                "file_b_path":       row["file_b_path"],
                "is_equal":          int(row["is_equal"]),
                "similarity_score":  score,
            })

            pbar.set_postfix(score=f"{score:.0f}", label=int(row["is_equal"]),
                             erros=parse_errors, refresh=False)

        except Exception as e:
            pbar.write(f"  [ERR] Par {i}: {e}")

    return pd.DataFrame(rows)


def _compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thr[idx])


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def _log_wandb(model_key: str, model_id: str, split_label: str,
               eer: float, threshold: float, n_pairs: int,
               wandb_entity: str, wandb_project: str) -> None:
    try:
        import wandb
        run_name = f"VLM_{model_key}_split{split_label}"
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=run_name,
            config={"model_key": model_key, "model_id": model_id,
                    "split": split_label, "n_pairs": n_pairs},
            reinit=True,
        )
        wandb.log({"test/eer": eer, "test/threshold": threshold, "test/n_pairs": n_pairs})
        run.finish()
        print(f"  W&B logged: {run_name}")
    except Exception as e:
        print(f"  ⚠️  W&B log falhou: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_splits(splits_arg: str) -> List[int]:
    if splits_arg.lower() == "all":
        return [0, 1, 2, 3, 4, 5]
    return [int(s.strip()) for s in splits_arg.split(",") if s.strip()]


DEFAULT_MODELS_2B = ["internvl3-2b", "qwen3vl-2b"]


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline VLM via métrica numérica — múltiplos modelos e splits.")
    p.add_argument("--models",         default=",".join(DEFAULT_MODELS_2B),
                   help=f"Modelos a avaliar, separados por vírgula (default: {','.join(DEFAULT_MODELS_2B)}). "
                        f"Disponíveis: {list(MODEL_REGISTRY)}")
    p.add_argument("--data-root",      default=None,
                   help="Raiz do dataset LA-CDIP (necessário para preparar splits 0-4)")
    p.add_argument("--base-image-dir", required=True,
                   help="Diretório base das imagens LA-CDIP")
    p.add_argument("--splits",         default="all",
                   help="Splits a avaliar: '5', '0,1,2,3,4,5' ou 'all' (default: all)")
    p.add_argument("--output-dir",     default=None,
                   help="Diretório para salvar CSVs de resultados (default: results/vlm_metric)")
    p.add_argument("--gpu-id",         type=int, default=None)
    p.add_argument("--wandb-entity",   default=WANDB_ENTITY)
    p.add_argument("--wandb-project",  default=WANDB_PROJECT)
    p.add_argument("--no-wandb",       action="store_true")
    p.add_argument("--load-in-4bit",   action="store_true",
                   help="Carregar modelo em 4-bit (NF4) para reduzir VRAM/RAM")
    p.add_argument("--limit",          type=int, default=None,
                   help="Limitar a N pares por split (para teste rápido)")
    p.add_argument("--eval-csv",       default=None,
                   help="CSV custom de pares (sobrescreve a resolução automática para split 5)")
    p.add_argument("--split-label",    default=None,
                   help="Label do split para W&B e arquivos de saída (ex: '5_synthetic')")
    args = p.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    for mk in model_keys:
        if mk not in MODEL_REGISTRY:
            p.error(f"Modelo desconhecido: '{mk}'. Disponíveis: {list(MODEL_REGISTRY)}")

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    splits = _parse_splits(args.splits)
    print(f"Splits a avaliar: {splits}")
    print(f"Modelos: {model_keys}")

    import pandas as pd
    all_summary = []

    for model_key in model_keys:
        print(f"\n{'='*60}")
        print(f"MODELO: {model_key}")
        print(f"{'='*60}")

        # Relança no .venv_vlm5 se necessário
        if model_key in MODELS_VLM5 and sys.executable != str(VENV_VLM5):
            if not VENV_VLM5.exists():
                print(f"[ERRO] {model_key} requer .venv_vlm5. Pulando.")
                continue
            print(f"[INFO] {model_key} requer transformers >= 5.5 → relançando em .venv_vlm5")
            os.execv(str(VENV_VLM5), [str(VENV_VLM5)] + sys.argv)

        out_dir = Path(args.output_dir) if args.output_dir else \
                  WORKSPACE_ROOT / "results" / "vlm_metric" / model_key
        out_dir.mkdir(parents=True, exist_ok=True)

        adapter, model_id = _build_adapter(model_key, device, load_in_4bit=args.load_in_4bit)
        summary_rows = []

        for split_idx in splits:
            print(f"\n{'='*60}")
            print(f"SPLIT {split_idx}")
            print(f"{'='*60}")

            # Prepare CSV
            if args.eval_csv and split_idx == 5:
                val_csv = Path(args.eval_csv)
            elif split_idx == 5:
                val_csv = WORKSPACE_ROOT / "data" / "generated_splits" / "eval_test_split5" / "validation_pairs.csv"
                if not val_csv.exists():
                    if not args.data_root:
                        print(f"  ⚠️  Split 5 CSV não encontrado e --data-root não fornecido. Pulando.")
                        continue
                    val_csv = _prepare_split(args.data_root, split_idx)
            else:
                if not args.data_root:
                    print(f"  ⚠️  --data-root necessário para preparar split {split_idx}. Pulando.")
                    continue
                val_csv = _prepare_split(args.data_root, split_idx)

            split_label = args.split_label if (args.eval_csv and split_idx == 5) else str(split_idx)
            print(f"  CSV: {val_csv}")

            t0 = time.time()
            df_results = _run_baseline(adapter, val_csv, args.base_image_dir, limit=args.limit)
            elapsed = time.time() - t0

            if len(df_results) < 10:
                print(f"  ⚠️  Pares insuficientes ({len(df_results)}). Pulando EER.")
                continue

            scores = df_results["similarity_score"].values
            labels = df_results["is_equal"].values
            eer, thr = _compute_eer(scores, labels)

            df_results["split"] = split_label
            pairs_csv = out_dir / f"split{split_label}_pairs.csv"
            df_results.to_csv(pairs_csv, index=False)

            summary_rows.append({
                "split":     split_label,
                "n_pairs":   len(df_results),
                "eer":       eer,
                "eer_pct":   round(eer * 100, 2),
                "threshold": thr,
                "elapsed_min": round(elapsed / 60, 1),
            })

            print(f"  EER={eer*100:.2f}%  threshold={thr:.1f}  ({elapsed/60:.1f} min)")
            print(f"  Resultados salvos: {pairs_csv}")

            if not args.no_wandb:
                _log_wandb(model_key, model_id, split_label, eer, thr,
                           len(df_results), args.wandb_entity, args.wandb_project)

        # Summary por modelo
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            summary_csv = out_dir / "summary.csv"
            df_summary.to_csv(summary_csv, index=False)

            print(f"\n{'='*60}")
            print(f"RESUMO — {model_key}")
            print(f"{'='*60}")
            print(df_summary[["split", "n_pairs", "eer_pct", "threshold"]].to_string(index=False))
            eers = df_summary["eer"].values
            print(f"\n  Média EER:   {eers.mean()*100:.2f}%")
            print(f"  Std EER:     {eers.std()*100:.2f} pp")
            print(f"  Mediana EER: {np.median(eers)*100:.2f}%")
            print(f"\n  Sumário salvo: {summary_csv}")

            if not args.no_wandb:
                try:
                    import wandb
                    agg = wandb.init(
                        entity=args.wandb_entity, project=args.wandb_project,
                        name=f"VLM_{model_key}_agg",
                        config={"model_key": model_key, "model_id": model_id,
                                "splits": splits, "type": "aggregate"},
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

            all_summary.append({"model": model_key, **{k: v for k, v in df_summary[["eer_pct"]].mean().items()}})

        # Cleanup do modelo antes de carregar o próximo
        del adapter
        torch.cuda.empty_cache()
        _delete_model_cache(model_id)

    if len(model_keys) > 1 and all_summary:
        print(f"\n{'='*60}")
        print("RESUMO GERAL")
        print(f"{'='*60}")
        print(pd.DataFrame(all_summary).to_string(index=False))


if __name__ == "__main__":
    main()
