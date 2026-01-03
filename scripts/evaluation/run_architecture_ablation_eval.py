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
print(f"Using cavl_doc from: {os.path.dirname(cavl_doc.__file__)}")

from cavl_doc.models.backbone_loader import load_model, warm_up_model
from cavl_doc.models.modeling_cavl import build_cavl_model
from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.trainers.rl_trainer import validate_siam_on_loader, build_loss
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

# ==============================================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ==============================================================================
CHECKPOINT_ROOT = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"

# Configurações para RVL-CDIP (Architecture Ablation)
DATASETS = {
    "LA-CDIP": {
        "output_csv": "results/LA-CDIP_architecture_ablation_eval.csv",
        "splits": [1, 2, 3, 4, 5],
        "data_root": "/mnt/data/la-cdip/data",
        "val_path_fmt": "data/generated_splits/zsl_split_{split}/validation_pairs.csv",
        "ckpt_pattern_fmt": "ABLATION_ARCH_Q{queries}_S{split}_{loss}_*",
        "num_classes": 10
    },
    "RVL-CDIP": {
        "output_csv": "results/RVL-CDIP_architecture_ablation_eval.csv",
        "splits": [0, 1, 2, 3],
        "data_root": "/mnt/data/zs_rvl_cdip/data",
        "val_path_fmt": "data/generated_splits/RVL-CDIP_zsl_split_{split}/validation_pairs.csv",
        # Pattern derived from run_architecture_ablation_rvl.py: RVL_ABLATION_ARCH_Q{queries}_S{split}_{loss}_*
        "ckpt_pattern_fmt": "RVL_ABLATION_ARCH_Q{queries}_S{split}_{loss}_*",
        "num_classes": 16
    }
}

QUERIES = [1, 8]
LOSSES = ["triplet", "cosface"]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def find_checkpoint(queries, split, loss, pattern_fmt):
    """Localiza o checkpoint da ablação de arquitetura."""
    pattern = pattern_fmt.format(queries=queries, split=split, loss=loss)
    search_path = os.path.join(CHECKPOINT_ROOT, pattern)
    candidates = sorted(glob.glob(search_path))
    
    if not candidates:
        return None
    
    latest_run = candidates[-1]
    
    # Prioridade: Best -> Last
    best_ckpt = os.path.join(latest_run, "best_siam.pt")
    last_ckpt = os.path.join(latest_run, "last_checkpoint.pt")
    
    if os.path.exists(best_ckpt):
        return best_ckpt
    elif os.path.exists(last_ckpt):
        print(f"⚠️  best_siam.pt não encontrado. Usando last_checkpoint.pt para {os.path.basename(latest_run)}")
        return last_ckpt
    
    return None

def load_trained_model(ckpt_path, device, tokenizer, backbone):
    """Carrega o modelo do checkpoint."""
    print(f"Carregando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    config = ckpt.get('config', {})
    if not config and 'args' in ckpt:
        config = vars(ckpt['args'])
    
    if not config:
        ckpt_dir = os.path.dirname(ckpt_path)
        json_path = os.path.join(ckpt_dir, "training_config.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                config = json.load(f)
    
    if not config:
        raise ValueError("Não foi possível encontrar a configuração do modelo.")

    cut_layer = config.get('cut_layer', 27)
    proj_dim = config.get('projection_output_dim', 1536)

    # --- Encode Fn (Cópia Local) ---
    def _encode_fn(backbone, images, cut_layer=cut_layer, **kwargs):
        input_ids_list = []
        pixel_values_list = []
        image_flags_list = []
        
        if isinstance(images, torch.Tensor):
            if images.dim() == 5: images_list = [images[i] for i in range(images.shape[0])]
            else: images_list = [images]
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

        out = backbone(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            pixel_values=batch_pixel_values,
            image_flags=batch_image_flags,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = out.hidden_states
        lm = backbone.language_model.model
        idx = cut_layer + 1 if len(hidden_states) == (len(lm.layers) + 1) else cut_layer
        return hidden_states[idx], None

    siam = build_cavl_model(
        backbone=backbone, 
        cut_layer=cut_layer, 
        encode_fn=_encode_fn,
        pool_dim=1536,
        proj_hidden=4096,
        proj_out=proj_dim,
        set_trainable=False,
        tokenizer=tokenizer,
        pooler_type=config.get('pooler_type', 'attention'),
        head_type=config.get('head_type', 'mlp'),
        num_queries=config.get('num_queries', 1)
    )
    
    if 'siam_pool' in ckpt:
        siam.pool.load_state_dict(ckpt['siam_pool'])
        siam.head.load_state_dict(ckpt['siam_head'])
        if 'backbone_trainable' in ckpt and ckpt['backbone_trainable']:
            siam.backbone.load_state_dict(ckpt['backbone_trainable'], strict=False)
    else:
        print("⚠️ Chaves 'siam_pool'/'siam_head' não encontradas. Tentando carregar state_dict completo...")
        try:
            siam.load_state_dict(ckpt['model_state_dict'], strict=False)
        except:
             print("❌ Falha ao carregar pesos.")
        
    siam.to(device)
    siam.eval()
    return siam, config

def rl_full_collate_fn(batch):
    img_a_list = [item['image_a'] for item in batch]
    img_b_list = [item['image_b'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    class_a = torch.zeros(len(batch), dtype=torch.long)
    class_b = torch.zeros(len(batch), dtype=torch.long)
    return img_a_list, img_b_list, labels, class_a, class_b

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Avaliação de Architecture Ablation")
    parser.add_argument("--dataset", type=str, default="RVL-CDIP", choices=["RVL-CDIP", "LA-CDIP"], help="Dataset para avaliação")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    cfg = DATASETS[dataset_name]
    output_csv = os.path.join(WORKSPACE_ROOT, cfg["output_csv"])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Iniciando Avaliação Architecture Ablation - {dataset_name}")
    print(f"Device: {device}")
    
    # 1. Carrega Backbone
    print("Carregando Backbone InternVL3-2B...")
    backbone, processor, tokenizer, _, _ = load_model(
        model_name="InternVL3-2B",
        adapter_path=None,
        load_in_4bit=False, 
        projection_output_dim=1536
    )
    backbone.requires_grad_(False)
    
    img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == img_context_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    print("🔥 Warm-up...")
    warm_up_model(backbone, processor)

    results = []
    if os.path.exists(output_csv):
        try: results = pd.read_csv(output_csv).to_dict('records')
        except: pass
        
    splits = cfg["splits"]
    
    for split in splits:
        val_csv = os.path.join(WORKSPACE_ROOT, cfg["val_path_fmt"].format(split=split))
        if not os.path.exists(val_csv):
            print(f"⚠️  Validação não encontrada: {val_csv}")
            continue
            
        print(f"\nCarregando Validação Split {split}: {val_csv}")
        val_dataset = DocumentPairDataset(val_csv, cfg["data_root"], 448, 12, 'cpu')
        val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=0, collate_fn=rl_full_collate_fn)
        
        for q in QUERIES:
            for loss in LOSSES:
                # Deduplicação
                if any(r['split'] == split and r['queries'] == q and r['loss'] == loss and r.get('status') == 'success' for r in results):
                    print(f"⏩ Q={q} | {loss} | Split {split} (Já existe)")
                    continue

                ckpt_path = find_checkpoint(q, split, loss, cfg["ckpt_pattern_fmt"])
                if not ckpt_path:
                    print(f"❌ Checkpoint não encontrado: Q={q} {loss} S{split}")
                    continue
                    
                print(f"--- Avaliando: Q={q} | {loss} | S{split} ---")
                try:
                    siam, config = load_trained_model(ckpt_path, device, tokenizer, backbone)
                    criterion = build_loss(loss, num_classes=cfg["num_classes"], in_features=config['projection_output_dim']).to(device)
                    
                    mean_loss, eer, thr, r1 = validate_siam_on_loader(siam, val_loader, device, criterion)
                    print(f"✅ EER: {eer*100:.2f}%")
                    
                    results.append({
                        "dataset": dataset_name,
                        "split": split,
                        "queries": q,
                        "loss": loss,
                        "eer": eer * 100,
                        "status": "success"
                    })
                    pd.DataFrame(results).to_csv(output_csv, index=False)
                    
                except Exception as e:
                    print(f"❌ Erro: {e}")
                    # import traceback
                    # traceback.print_exc()

    print(f"\n🎉 Concluído! CSV: {output_csv}")

if __name__ == "__main__":
    main()
