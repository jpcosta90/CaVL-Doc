#!/usr/bin/env python3
import os
import sys
import glob
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import traceback

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import cavl_doc
print(f"Using cavl_doc from: {os.path.dirname(cavl_doc.__file__)}")

from cavl_doc.models.backbone_loader import load_model, warm_up_model
from cavl_doc.models.modeling_cavl import build_cavl_model
from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.trainers.rl_trainer import validate_siam_on_loader, build_loss
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
WORKSPACE_ROOT = "/home/joaopaulo/Projects/CaVL-Doc"
CHECKPOINT_ROOT = "/mnt/large/checkpoints" if os.path.exists("/mnt/large") else "checkpoints"

LOSSES_TO_TEST = [
    "contrastive",
    "triplet",
    "arcface",
    "cosface",
    # "expface",
    # "circle",
    # "subcenter_arcface"
]

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def find_best_checkpoint(dataset_name, protocol, split_idx, loss_type):
    """Encontra o checkpoint best_siam.pt para a combinação dada."""
    search_dataset_name = dataset_name
    if dataset_name == "LA-CDIP":
        search_dataset_name = "LACDIP"

    base_name = f"{search_dataset_name}_{protocol}_S{split_idx}_{loss_type}"
    search_pattern = os.path.join(CHECKPOINT_ROOT, f"{base_name}_*")
    runs = sorted(glob.glob(search_pattern))
    print(f"Procurando por: {search_pattern}")
    if not runs:
        return None
    
    # Pega o run mais recente
    latest_run = runs[-1]
    ckpt_path = os.path.join(latest_run, "best_siam.pt")
    
    if os.path.exists(ckpt_path):
        return ckpt_path
    return None

def load_trained_model(ckpt_path, device, tokenizer, backbone):
    """Carrega o modelo a partir do checkpoint."""
    print(f"Carregando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    
    # Extrai configs
    ckpt_max_num = config.get('max_num_image_tokens', 12) 
    print(f" -> Config do Checkpoint: max_num={ckpt_max_num}, cut_layer={config['cut_layer']}")

    # ==========================================================================
    # ENCODE FN (Mesma lógica do rl_trainer.py)
    # ==========================================================================
    def _encode_fn(backbone, images, cut_layer=config['cut_layer'], **kwargs):
        # images: Tensor [B, N, C, H, W] ou List[Tensor [Ni, C, H, W]]
        
        input_ids_list = []
        pixel_values_list = []
        image_flags_list = []
        
        # Normaliza entrada para lista
        if isinstance(images, torch.Tensor):
            if images.dim() == 5:
                images_list = [images[i] for i in range(images.shape[0])]
            else:
                images_list = [images]
        else:
            images_list = images

        # Prepara inputs individualmente
        for img in images_list:
            out = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, img, "<image> Analyze this document")
            input_ids_list.append(out['input_ids'][0]) 
            pixel_values_list.append(out['pixel_values']) 
            image_flags_list.append(out['image_flags']) 

        # Padding manual dos input_ids
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

        # Batch final
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
        cut_layer=config['cut_layer'], 
        encode_fn=_encode_fn,
        pool_dim=1536,
        proj_hidden=4096,
        proj_out=config['projection_output_dim'],
        set_trainable=False, # Eval mode
        tokenizer=tokenizer,
        pooler_type=config.get('pooler_type', 'attention'),
        head_type=config.get('head_type', 'mlp'),
        num_queries=config.get('num_queries', 1)
    )
    
    siam.pool.load_state_dict(ckpt['siam_pool'])
    siam.head.load_state_dict(ckpt['siam_head'])
    
    if 'backbone_trainable' in ckpt and ckpt['backbone_trainable']:
        siam.backbone.load_state_dict(ckpt['backbone_trainable'], strict=False)
        
    siam.to(device)
    siam.eval()
    
    return siam, config

def rl_full_collate_fn(batch):
    img_a_list = [item['image_a'] for item in batch]
    img_b_list = [item['image_b'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    if 'class_a' in batch[0]:
        class_a = torch.tensor([item['class_a'] for item in batch], dtype=torch.long)
        class_b = torch.tensor([item['class_b'] for item in batch], dtype=torch.long)
    else:
        class_a = torch.zeros(len(batch), dtype=torch.long)
        class_b = torch.zeros(len(batch), dtype=torch.long)
        
    return img_a_list, img_b_list, labels, class_a, class_b

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["LA-CDIP", "RVL-CDIP"])
    parser.add_argument("--protocol", type=str, default="zsl")
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--output-csv", type=str, default="full_eval_results.csv")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Carrega Backbone
    print("Carregando Backbone InternVL3-2B...")
    backbone, processor, tokenizer, _, _ = load_model(
        model_name="InternVL3-2B",
        adapter_path=None,
        load_in_4bit=False, 
        projection_output_dim=1536
    )
    backbone.requires_grad_(False)

    # --- FIX: Ensure pad_token_id does not collide with <IMG_CONTEXT> ---
    img_context_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == img_context_id:
        print(f"⚠️  [FIX] Pad ID collision detected ({tokenizer.pad_token_id}). Switching pad_id to EOS token.")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # --------------------------------------------------------------------
    
    # IMPORTANTE: Warm up do modelo para inicializar buffers lazy
    print("🔥 Executando Warm-up...")
    warm_up_model(backbone, processor)

    # Carrega resultados existentes para evitar reprocessamento (Resume)
    if os.path.exists(args.output_csv):
        print(f"📂 Carregando resultados existentes de: {args.output_csv}")
        try:
            results = pd.read_csv(args.output_csv).to_dict('records')
        except Exception as e:
            print(f"⚠️ Erro ao ler CSV existente: {e}. Começando do zero.")
            results = []
    else:
        results = []

    for split_idx in args.splits:
        # Caminho dos dados
        if args.dataset == "RVL-CDIP":
             generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/RVL-CDIP_{args.protocol}_split_{split_idx}")
        else:
             generated_data_dir = os.path.join(WORKSPACE_ROOT, f"data/generated_splits/{args.protocol}_split_{split_idx}")
             
        val_csv = os.path.join(generated_data_dir, "validation_pairs.csv")
        base_image_dir = "/mnt/data/zs_rvl_cdip/data" if args.dataset == "RVL-CDIP" else "/mnt/data/la-cdip/data"

        if not os.path.exists(val_csv):
            print(f"⚠️  Arquivo de validação não encontrado: {val_csv}. Pulando Split {split_idx}.")
            continue
            
        print(f"\nCarregando Dataset de Validação Completo: {val_csv}")
        # Carrega com max_num=12 para segurança do dataset
        val_dataset = DocumentPairDataset(
            csv_path=val_csv, 
            base_dir=base_image_dir,
            input_size=448, 
            max_num=12, 
            device='cpu'
        )
        val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=0, collate_fn=rl_full_collate_fn)
        
        for loss_type in LOSSES_TO_TEST:
            # Verifica se já foi processado com sucesso
            already_processed = any(
                r['dataset'] == args.dataset and 
                r['split'] == split_idx and 
                r['loss'] == loss_type and 
                r.get('status') == 'success'
                for r in results
            )
            
            if already_processed:
                print(f"⏩ Pulando Split {split_idx} | Loss: {loss_type} (Já existe no CSV)")
                continue

            print(f"\n--- Avaliando Split {split_idx} | Loss: {loss_type} ---")
            
            ckpt_path = find_best_checkpoint(args.dataset, args.protocol, split_idx, loss_type)
            print(f"Checkpoint: {ckpt_path}")
            
            if not ckpt_path:
                print(f"❌ Checkpoint não encontrado.")
                results.append({
                    "dataset": args.dataset,
                    "split": split_idx,
                    "loss": loss_type,
                    "eer": None,
                    "loss_val": None,
                    "status": "missing_checkpoint"
                })
                continue
                
            try:
                # Carrega modelo e roda a validação
                siam, config = load_trained_model(ckpt_path, device, tokenizer, backbone)
                
                criterion = build_loss(loss_type, num_classes=10, in_features=config['projection_output_dim']).to(device)
                
                mean_loss, eer, thr, r1 = validate_siam_on_loader(siam, val_loader, device, criterion)
                
                print(f"✅ Resultado: EER = {eer*100:.2f}%")
                
                results.append({
                    "dataset": args.dataset,
                    "split": split_idx,
                    "loss": loss_type,
                    "eer": eer * 100,
                    "loss_val": mean_loss,
                    "status": "success"
                })
                
                pd.DataFrame(results).to_csv(args.output_csv, index=False)
                
            except Exception as e:
                # import traceback
                # traceback.print_exc()
                print(f"❌ Erro ao avaliar: {e}")
                results.append({
                    "dataset": args.dataset,
                    "split": split_idx,
                    "loss": loss_type,
                    "eer": None,
                    "loss_val": None,
                    "status": f"error: {str(e)}"
                })
                torch.cuda.empty_cache()

    print(f"\n🎉 Avaliação concluída! Resultados salvos em {args.output_csv}")

if __name__ == "__main__":
    main()