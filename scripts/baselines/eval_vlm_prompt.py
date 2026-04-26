#!/usr/bin/env python3
"""
Baseline de similaridade de documentos via prompt para LLMs multimodais.

Envia dois documentos + prompt de comparação para um VLM open-source e usa
o similarity_score (0-100) retornado como métrica para calcular EER no split 5.

Modelos suportados (--model):
  internvl3-2b    OpenGVLab/InternVL3-2B
  internvl3-8b    OpenGVLab/InternVL3-8B
  qwen25vl-3b     Qwen/Qwen2.5-VL-3B-Instruct
  qwen25vl-7b     Qwen/Qwen2.5-VL-7B-Instruct
  qwen3vl-7b      Qwen/Qwen3-VL-7B-Instruct
  minicpm-v26     openbmb/MiniCPM-V-2_6
  pixtral-12b     mistralai/Pixtral-12B-2409

Uso:
  python scripts/evaluation/eval_vlm_baseline.py \\
      --model internvl3-2b \\
      --val-csv data/generated_splits/eval_test_split5/validation_pairs.csv \\
      --base-image-dir /mnt/data/la-cdip/data \\
      --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

WANDB_ENTITY  = "jpcosta1990-university-of-brasilia"
WANDB_PROJECT = "CaVL-Doc_LA-CDIP_VLM_Baseline"

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
Respond **only** with a JSON object structured as follows:
```json
{
    "similarity_score": <value between 0 and 100>,
    "category": "<one of: Nearly Identical, Highly Similar, Moderately Similar, Weak Similarity, Completely Different>",
    "justification": "Briefly explain the key visual similarities or differences detected."
}
```"""

MODEL_REGISTRY = {
    # InternVL3 — 2B e 14B confirmados funcionando; 9B não funciona
    "internvl3-2b":   "OpenGVLab/InternVL3-2B",
    "internvl3-14b":  "OpenGVLab/InternVL3-14B",
    # Qwen-VL
    "qwen25vl-3b":    "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen25vl-7b":    "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3vl-7b":     "Qwen/Qwen3-VL-7B-Instruct",
    # MiniCPM
    "minicpm-v26":    "openbmb/MiniCPM-V-2_6",
    # Mistral — pixtral a testar; ministral-3b confirmado multimodal
    "pixtral-12b":    "mistralai/Pixtral-12B-2409",
    "ministral-3b":   "mistralai/Ministral-3-3B-Instruct-2512",
    # Gemma 4 — E4B (~4B params eficientes, cabe em 16GB); 31B não cabe
    "gemma4-e4b":     "google/gemma-4-E4B-it",
}


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

def _parse_score(text: str) -> Optional[float]:
    """Extract similarity_score from model JSON response."""
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```[a-z]*", "", text).strip().strip("`")
        data = json.loads(clean)
        return float(data["similarity_score"])
    except Exception:
        pass
    # Fallback: regex
    m = re.search(r'"similarity_score"\s*:\s*(\d+(?:\.\d+)?)', text)
    if m:
        return float(m.group(1))
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
            tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=6)
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
    """Adapter for Qwen2.5-VL and Qwen3-VL families."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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
    """Adapter for MiniCPM-V 2.6."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device,
            quantization_config=_get_bnb_config() if load_in_4bit else None,
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


class _MistralVLAdapter:
    """Adapter for Mistral vision models (Ministral-3, Pixtral) via transformers."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            quantization_config=_get_bnb_config() if load_in_4bit else None,
        ).eval()

    def infer(self, img_a: Image.Image, img_b: Image.Image) -> str:
        prompt_text = SIMILARITY_PROMPT.replace("<image>", "[IMG]")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt_text.replace("[IMG]", "").strip()},
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
    """Adapter for Gemma 4 vision models."""

    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
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
                {"type": "image", "image": img_a},
                {"type": "image", "image": img_b},
                {"type": "text",  "text": prompt_text},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        out_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = out_ids[0][inputs["input_ids"].shape[-1]:]
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
    elif model_key.startswith("ministral") or model_key.startswith("pixtral"):
        return _MistralVLAdapter(model_id, device, **kw), model_id
    elif model_key.startswith("gemma4"):
        return _Gemma4Adapter(model_id, device, **kw), model_id
    else:
        raise ValueError(f"Adapter não implementado para {model_key}. "
                         f"Opções: {list(MODEL_REGISTRY)}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _load_image(path: str, base_dir: str) -> Image.Image:
    full = Path(base_dir) / path
    return Image.open(full).convert("RGB")


def _run_baseline(adapter, val_csv: Path, base_image_dir: str,
                  max_parse_errors: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on all pairs. Returns (scores, labels)."""
    import pandas as pd
    df = pd.read_csv(val_csv)
    scores, labels = [], []
    parse_errors = 0

    for i, row in df.iterrows():
        try:
            img_a = _load_image(row["file_a_path"], base_image_dir)
            img_b = _load_image(row["file_b_path"], base_image_dir)
            response = adapter.infer(img_a, img_b)
            score    = _parse_score(response)

            if score is None:
                parse_errors += 1
                print(f"  [WARN] Par {i}: não foi possível parsear score. Resposta: {response[:120]}")
                if parse_errors >= max_parse_errors:
                    print("  Muitos erros de parse. Encerrando.")
                    break
                continue

            scores.append(score)
            labels.append(int(row["is_equal"]))

            if i % 50 == 0:
                print(f"  [{i}/{len(df)}] score={score:.1f} label={int(row['is_equal'])}")

        except Exception as e:
            print(f"  [ERR] Par {i}: {e}")

    return np.array(scores, dtype=float), np.array(labels, dtype=int)


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

def _log_wandb(model_key: str, model_id: str, eer: float, threshold: float,
               n_pairs: int, wandb_entity: str, wandb_project: str) -> None:
    try:
        import wandb
        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=f"VLM_{model_key}_split5",
            config={"model_key": model_key, "model_id": model_id,
                    "test_split": 5, "n_pairs": n_pairs},
            reinit=True,
        )
        wandb.log({"test/eer": eer, "test/threshold": threshold, "test/n_pairs": n_pairs})
        run.finish()
        print(f"  W&B logged: VLM_{model_key}_split5")
    except Exception as e:
        print(f"  ⚠️  W&B log falhou: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Baseline VLM via prompt para split 5.")
    p.add_argument("--model",          required=True, choices=list(MODEL_REGISTRY),
                   help=f"Modelo a avaliar: {list(MODEL_REGISTRY)}")
    p.add_argument("--val-csv",        default=None,
                   help="CSV de pares de validação (default: split 5 gerado)")
    p.add_argument("--base-image-dir", required=True,
                   help="Diretório base das imagens LA-CDIP")
    p.add_argument("--gpu-id",         type=int, default=None)
    p.add_argument("--wandb-entity",   default=WANDB_ENTITY)
    p.add_argument("--wandb-project",  default=WANDB_PROJECT)
    p.add_argument("--no-wandb",       action="store_true")
    p.add_argument("--load-in-4bit",   action="store_true",
                   help="Carregar modelo em 4-bit (NF4) para reduzir uso de VRAM/RAM")
    p.add_argument("--limit",          type=int, default=None,
                   help="Limitar a N pares (para teste rápido)")
    args = p.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Val CSV
    if args.val_csv:
        val_csv = Path(args.val_csv)
    else:
        val_csv = WORKSPACE_ROOT / "data" / "generated_splits" / "eval_test_split5" / "validation_pairs.csv"
    if not val_csv.exists():
        print(f"CSV não encontrado: {val_csv}")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(val_csv)
    if args.limit:
        df = df.head(args.limit)
        tmp = val_csv.parent / "_tmp_limited.csv"
        df.to_csv(tmp, index=False)
        val_csv = tmp

    print(f"Val CSV: {val_csv}  ({len(df)} pares)")

    # Load model
    adapter, model_id = _build_adapter(args.model, device, load_in_4bit=args.load_in_4bit)

    # Run
    print(f"\nRodando inferência com {args.model}...")
    t0 = time.time()
    scores, labels = _run_baseline(adapter, val_csv, args.base_image_dir)
    elapsed = time.time() - t0

    if len(scores) < 10:
        print("Pares insuficientes para calcular EER.")
        sys.exit(1)

    eer, thr = _compute_eer(scores, labels)
    print(f"\n{'='*60}")
    print(f"Modelo:    {args.model} ({model_id})")
    print(f"Pares:     {len(scores)}")
    print(f"EER:       {eer*100:.2f}%")
    print(f"Threshold: {thr:.1f}")
    print(f"Tempo:     {elapsed/60:.1f} min")
    print(f"{'='*60}")

    if not args.no_wandb:
        _log_wandb(args.model, model_id, eer, thr, len(scores),
                   args.wandb_entity, args.wandb_project)

    # Free GPU and delete HF cache to recover disk space
    del adapter
    torch.cuda.empty_cache()
    _delete_model_cache(model_id)


if __name__ == "__main__":
    main()
