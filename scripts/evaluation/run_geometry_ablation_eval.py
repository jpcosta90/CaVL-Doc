#!/usr/bin/env python3
import os
import sys
import glob
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import json

# ==============================================================================
# CONFIGURAÇÃO DE AMBIENTE E IMPORTS
# ==============================================================================
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

import cavl_doc
from cavl_doc.models.backbone_loader import load_model, warm_up_model
from cavl_doc.models.modeling_cavl import build_cavl_model
from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.trainers.rl_trainer import validate_siam_on_loader, build_loss
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

CHECKPOINT_ROOT = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"

# Configurações Defaults
DEFAULT_DATASET = "RVL-CDIP"
SPLITS = [0, 1, 2, 3] # RVL defaults

# Iterators
MARGINS = [0.15, 0.45, 0.75]
SUBCENTERS = [1, 3, 5]
SIGMAS = [0.0125, 0.05, 0.1]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def find_checkpoint(patterns):
    if isinstance(patterns, str): patterns = [patterns]
    
    candidates = []
    for pat in patterns:
        search_path = os.path.join(CHECKPOINT_ROOT, pat)
        candidates.extend(glob.glob(search_path))
    
    if not candidates: return None
    candidates = sorted(candidates) # Sort to get latest
    latest_run = candidates[-1]
    
    best_ckpt = os.path.join(latest_run, "best_siam.pt")
    last_ckpt = os.path.join(latest_run, "last_checkpoint.pt")
    if os.path.exists(best_ckpt): return best_ckpt
    elif os.path.exists(last_ckpt): return last_ckpt
    return None

def load_trained_model(ckpt_path, device, tokenizer, backbone):
    print(f"Carregando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    if not config and 'args' in ckpt: config = vars(ckpt['args'])
    
    if not config:
        ckpt_dir = os.path.dirname(ckpt_path)
        json_path = os.path.join(ckpt_dir, "training_config.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f: config = json.load(f)
            
    cut_layer = config.get('cut_layer', 27)
    proj_dim = config.get('projection_output_dim', 1536)

    def _encode_fn(backbone, images, cut_layer=cut_layer, **kwargs):
        input_ids_list = []
        pixel_values_list = []
        image_flags_list = []
        if isinstance(images, torch.Tensor): images_list = [images[i] for i in range(images.shape[0])] if images.dim() == 5 else [images]
        else: images_list = images

        for img in images_list:
            out = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, img, "<image> Analyze this document")
            input_ids_list.append(out['input_ids'][0])
            pixel_values_list.append(out['pixel_values'])
            image_flags_list.append(out['image_flags']) 

        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded_input_ids = []
        padded_attention_mask = []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            p_ids = torch.cat([ids, torch.full((pad_len,), pad_id, device=ids.device, dtype=ids.dtype)])
            p_mask = torch.cat([torch.ones_like(ids), torch.zeros((pad_len,), device=ids.device, dtype=ids.dtype)])
            padded_input_ids.append(p_ids)
            padded_attention_mask.append(p_mask)

        batch_input_ids = torch.stack(padded_input_ids).to(device)
        batch_attention_mask = torch.stack(padded_attention_mask).to(device)
        batch_pixel_values = torch.cat(pixel_values_list, dim=0).to(device, dtype=torch.bfloat16)
        batch_image_flags = torch.cat(image_flags_list, dim=0).to(device)

        out = backbone(input_ids=batch_input_ids, attention_mask=batch_attention_mask, pixel_values=batch_pixel_values, image_flags=batch_image_flags, output_hidden_states=True, return_dict=True)
        return out.hidden_states[cut_layer + 1], None

    siam = build_cavl_model(backbone=backbone, cut_layer=cut_layer, encode_fn=_encode_fn, pool_dim=1536, proj_hidden=4096, proj_out=proj_dim, set_trainable=False, tokenizer=tokenizer, pooler_type=config.get('pooler_type', 'attention'), head_type=config.get('head_type', 'mlp'), num_queries=config.get('num_queries', 1))
    
    if 'siam_pool' in ckpt:
        siam.pool.load_state_dict(ckpt['siam_pool'])
        siam.head.load_state_dict(ckpt['siam_head'])
        if 'backbone_trainable' in ckpt and ckpt['backbone_trainable']: siam.backbone.load_state_dict(ckpt['backbone_trainable'], strict=False)
    else:
        try: siam.load_state_dict(ckpt['model_state_dict'], strict=False)
        except: pass
    siam.to(device); siam.eval()
    return siam, config

def rl_full_collate_fn(batch):
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    return [item['image_a'] for item in batch], [item['image_b'] for item in batch], labels, torch.zeros(len(batch), dtype=torch.long), torch.zeros(len(batch), dtype=torch.long)

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--best-margin", type=float, default=0.45) # Used to check if we are evaluating Stage 2/3
    parser.add_argument("--best-k", type=int, default=1)
    args = parser.parse_args()

    config_rvl = {
        "output_csv": "results/RVL-CDIP_geometry_ablation_eval.csv",
        "splits": [0, 1, 2, 3],
        "data_root": "/mnt/data/zs_rvl_cdip/data",
        "val_path_fmt": "data/generated_splits/RVL-CDIP_zsl_split_{split}/validation_pairs.csv",
        "num_classes": 16
    }
    # LA-CDIP placeholder if needed
    
    cfg = config_rvl
    output_csv = os.path.join(WORKSPACE_ROOT, cfg["output_csv"])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Backbone
    backbone, processor, tokenizer, _, _ = load_model("InternVL3-2B", load_in_4bit=False, projection_output_dim=1536)
    backbone.requires_grad_(False)
    img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == img_context_id: tokenizer.pad_token_id = tokenizer.eos_token_id
    warm_up_model(backbone, processor)

    results = []
    if os.path.exists(output_csv):
        try: results = pd.read_csv(output_csv).to_dict('records')
        except: pass

    for split in cfg["splits"]:
        val_csv = os.path.join(WORKSPACE_ROOT, cfg["val_path_fmt"].format(split=split))
        if not os.path.exists(val_csv): continue
        val_dataset = DocumentPairDataset(val_csv, cfg["data_root"], 448, 12, 'cpu')
        val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=0, collate_fn=rl_full_collate_fn)

        # Helper to run eval
        def run_single_eval(name, margin, k, std, pattern):
            if any(r['split'] == split and r['name'] == name and r['margin'] == margin and r['k'] == k and r['std'] == str(std) for r in results):
                return
            
            ckpt_path = find_checkpoint(pattern)
            if not ckpt_path: return

            try:
                siam, c = load_trained_model(ckpt_path, device, tokenizer, backbone)
                # Build correct loss for validation metric (though EER is metric-agnostic usually, criterion helps log loss)
                # Assuming simple contrastive/cosine logic for eval or matching loss_type
                # Using 'cosface' as generic holder or reading from config?
                # Using build_loss with generic args
                criterion = build_loss("cosface", num_classes=cfg["num_classes"], in_features=c['projection_output_dim']).to(device)
                
                _, eer, _, _ = validate_siam_on_loader(siam, val_loader, device, criterion)
                print(f"✅ {name} (m={margin}, k={k}, std={std}) S{split}: {eer*100:.2f}%")
                
                results.append({
                    "dataset": args.dataset, "split": split, "name": name, 
                    "margin": margin, "k": k, "std": str(std), 
                    "eer": eer * 100
                })
                pd.DataFrame(results).to_csv(output_csv, index=False)
            except Exception as e:
                print(f"❌ Error {name}: {e}")

        # 1. Margins
        for m in MARGINS:
            # TRY GEO FIRST, THEN ARCH (fallback for m=0.45/default params)
            # Arch Ablation uses pattern: RVL_ABLATION_ARCH_Q{queries}_S{split}_cosface_*
            # Assuming Q=AutoConfig['queries'] (usually 4) matches.
            # We hardcode Q=4 here just to match what was likely run in Architecture Ablation if we fallback.
            # But safer is to assume user wants us to find matching m.
            
            geo_pat = f"RVL_ABLATION_GEO_cosface_m{m}_k1_S{split}_*"
            arch_pat = f"RVL_ABLATION_ARCH_Q4_S{split}_cosface_*" if m == 0.45 else "IMPOSSIBLE_PATTERN"
            
            # Note: Architecture ablation implicitly uses margin=0.45 (common args). 
            # So if m=0.45, we look for Q4 architecture run.
            
            run_single_eval("margin_variation", m, 1, "None", [geo_pat, arch_pat])

        # 2. Subcenters (Assume they were run with args.best_margin)
        for k in SUBCENTERS:
            loss = "subcenter_cosface" if k > 1 else "cosface"
            pat = f"RVL_ABLATION_GEO_{loss}_m{args.best_margin}_k{k}_S{split}_*"
            # If K=1, it might overlap with margin run above, but that's okay, logic handles dedupe via keys
            run_single_eval("subcenter_variation", args.best_margin, k, "None", pat)

        # 3. Elasticity (Assume run with best_margin and best_k=1)
        for s in SIGMAS:
            pat = f"RVL_ABLATION_GEO_elastic_cosface_m{args.best_margin}_k1_std{s}_S{split}_*"
            run_single_eval("elasticity_variation", args.best_margin, 1, s, pat)

    print(f"Done. Results in {output_csv}")

if __name__ == "__main__":
    main()
