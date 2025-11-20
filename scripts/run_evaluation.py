# Em: scripts/run_evaluation.py
# (Substitua o seu arquivo por este código)

import os
import torch
import pandas as pd
import argparse
import hashlib
import numpy as np
from sklearn.metrics import roc_curve
import sys
import torch
import torch.nn as nn 
from peft import PeftModel 
# ==========================================================
# 1. IMPORTAÇÕES
# ==========================================================
from src.models.lvlm_handler import load_model, warm_up_model
from src.data_loaders.documentpairs import DocumentPairDataset
from src.metrics.evaluation import run_meanpooling_embedding_comparison
from src.metrics.baseline_metrics import run_pixel_comparison
from src.utils.tracker import ExperimentTracker
from src.utils.visualization import plot_density

import logging

# ==========================================================
# 2. FUNÇÕES AUXILIARES
# ==========================================================
def count_total_parameters(model) -> float:
    # ... (A sua função count_total_parameters não muda) ...
    if isinstance(model, PeftModel):
        return sum(p.numel() for p in model.parameters()) / 1_000_000
    elif isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters()) / 1_000_000
    return 0.0

# ==========================================================
# 3. FUNÇÃO PRINCIPAL
# ==========================================================
def main(args):
    # --- ETAPA 1: SETUP ---
    print("--- 1. Configurando o Experimento ---")
    
    prompt_to_use = "N/A" 
    
    if args.evaluation_method == 'embedding':
        if args.evolution_log:
            print(f"    -> Procurando pelo melhor prompt no arquivo de log: '{args.evolution_log}'")
            try:
                optimization_df = pd.read_csv(args.evolution_log)
                best_row = optimization_df.loc[optimization_df['eer_score'].idxmin()]
                prompt_to_use = best_row['prompt_text']
                print(f"    -> Melhor prompt encontrado (EER {best_row['eer_score']:.4f}): '{prompt_to_use[:90]}...'")
            except Exception as e:
                print(f"❌ ERRO: Não foi possível ler o melhor prompt de '{args.evolution_log}'. Erro: {e}")
                sys.exit(1)
        else:
            prompt_to_use = args.prompt

    dataset_base_name = os.path.basename(os.path.dirname(args.pairs_csv))
    output_csv_path = args.output_csv if args.output_csv else f"results/{dataset_base_name}_master_results.csv"
    
    if args.method_name:
        method_name = args.method_name
    elif 'pixel' in args.evaluation_method:
        method_name = args.evaluation_method + '_baseline'
    else: # evaluation_method == 'embedding'
        prompt_hash = hashlib.sha1(prompt_to_use.encode('utf-8')).hexdigest()[:6]
        
        if args.checkpoint_path:
            checkpoint_folder_name = os.path.basename(args.checkpoint_path.rstrip('/'))
            
            # [MODIFICADO] Lógica de nomenclatura para incluir novos checkpoints
            # Tenta extrair do nome do checkpoint de RL
            parts = checkpoint_folder_name.split('_')
            if checkpoint_folder_name.startswith('LA-CDIP') and len(parts) >= 6: 
                # LA-CDIP_InternVL3-2B_RL_Full_Head_2000_20251108-194952
                cp_dataset_name = parts[0]
                cp_model_slug = parts[1]
                cp_type = parts[3] # "Head"
                cp_train_size = parts[4] 
                cp_timestamp = parts[5]
                
                method_name = (
                    f"{cp_dataset_name}_{cp_model_slug}_{cp_type}_{cp_train_size}_"
                    f"{cp_timestamp}-{prompt_hash}_{args.metric}"
                )
            # Tenta extrair do nome do checkpoint antigo (LoRA/Connector)
            elif len(parts) >= 5: 
                cp_dataset_name = parts[0]
                cp_model_slug = parts[1] 
                cp_layers = parts[2] 
                cp_train_size = parts[3] 
                cp_timestamp = parts[4] 
                method_name = (
                    f"{cp_dataset_name}_{cp_model_slug}_{cp_layers}_{cp_train_size}_"
                    f"{cp_timestamp}-{prompt_hash}_{args.metric}"
                )
            else: # Fallback
                print(f"⚠️ Aviso: Nome do checkpoint '{checkpoint_folder_name}' não segue padrões conhecidos. Usando nome genérico.")
                model_id = checkpoint_folder_name 
                method_name = f"{model_id}_{prompt_hash}_{args.metric}"
        else: # Modelo base
            model_id = args.model_name
            method_name = f"{model_id}_{prompt_hash}_{args.metric}"

    print(f"    - Nome do Método: {method_name}")
    print(f"    - Arquivo de Resultados: {output_csv_path}")

    # ... (A sua verificação de 'force' não muda) ...
    if not args.force:
        tracker_check = ExperimentTracker(file_path=output_csv_path)
        existing_methods = tracker_check.results['method_name'].tolist() if 'method_name' in tracker_check.results.columns else []
        if method_name in existing_methods:
             existing_eer = tracker_check.results[tracker_check.results['method_name'] == method_name]['eer'].iloc[0]
             print(f"\n⚠️  AVISO: O experimento '{method_name}' já foi executado. EER: {existing_eer:.4f}. Use --force.")
             sys.exit()

    # --- 2. CARREGAMENTO E COLETA DE METADADOS ---
    print("\n--- 2. Carregando Configuração e Coletando Metadados ---")
    dataset = DocumentPairDataset(csv_path=args.pairs_csv, base_dir=args.base_image_dir)
    
    if args.evaluation_method == 'embedding':
        # VVV --- [MODIFICAÇÃO IMPORTANTE] --- VVV
        model, processor, tokenizer, connector, head = load_model(
            model_name=args.model_name,
            adapter_path=args.checkpoint_path,
            load_in_4bit=True, # Carrega em 4-bit para avaliação
            projection_output_dim=args.projection_output_dim # Passa a dimensão
        )
        # ^^^ --- [FIM DA MODIFICAÇÃO] --- ^^^
        
        warm_up_model(model, processor) 
        
        # [MODIFICADO] A contagem de parâmetros agora inclui a cabeça (head)
        params_in_millions = count_total_parameters(model)
        if head is not None:
             params_in_millions += count_total_parameters(head)
        if connector is not None:
             params_in_millions += count_total_parameters(connector)


        model_id = os.path.basename(args.checkpoint_path) if args.checkpoint_path else args.model_name
        arch = 'Fine-Tuned (Head)' if head is not None else ('Fine-Tuned (LoRA)' if isinstance(model, PeftModel) else 'LLM Base')
    else: # Baselines
        model, tokenizer, head = None, None, None # Adiciona head=None
        params_in_millions = 0.1
        model_id = 'N/A'
        arch = 'Baseline'
        
    print(f"    - Arquitetura: {arch}")
    print(f"    - Parâmetros Totais (M): {params_in_millions:.2f}")

    # --- 3. EXECUÇÃO DA AVALIAÇÃO ---
    print("\n--- 3. Executando Avaliação ---")
    if args.evaluation_method == 'embedding':
        # VVV --- [MODIFICAÇÃO IMPORTANTE] --- VVV
        results_df = run_meanpooling_embedding_comparison(
            dataset=dataset, model=model, tokenizer=tokenizer,
            prompt=prompt_to_use, 
            metric_type=args.metric,
            student_head=head  # <-- PASSA A CABEÇA CARREGADA
        )
        # ^^^ --- [FIM DA MODIFICAÇÃO] --- ^^^
    else: 
        metric_type = 'cosine' if 'cosine' in args.evaluation_method else 'euclidean'
        results_df = run_pixel_comparison(dataset=dataset, metric_type=metric_type)

    # --- 4. CÁLCULO DE EER E REGISTRO ---
    print("\n--- 4. Calculando EER e Registrando Resultado ---")
    tracker = ExperimentTracker(file_path=output_csv_path) 
    
    labels = results_df['is_equal'].values
    scores = results_df['metric_score'].values
    
    is_distance_metric = (args.metric == 'euclidean') or ('pixel' in args.evaluation_method)
    if is_distance_metric:
        scores = -scores

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer_score = (fpr[eer_index] + fnr[eer_index]) / 2
    
    metric_for_log = 'cosine' if 'cosine' in args.evaluation_method else ('euclidean' if 'euclidean' in args.evaluation_method else args.metric)

    tracker.log(
        method_name=method_name,
        eer=eer_score,
        model=model_id, 
        arch=arch,
        params=params_in_millions,
        metric=metric_for_log,
        dataset=dataset_base_name,
        partition=os.path.basename(args.pairs_csv)
    )

    # --- 5. VISUALIZAÇÃO (OPCIONAL) ---
    if args.plot:
        print("\n--- 5. Gerando Gráfico de Densidade ---")
        eer_threshold = thresholds[eer_index]
        if is_distance_metric:
            eer_threshold = -eer_threshold
        
        plot_density(
            results_df=results_df, 
            eer_score=eer_score, 
            eer_threshold=eer_threshold, 
            method_name=method_name,
            dataset_name=dataset_base_name,
            metric_type=metric_for_log
        )

    print(f"\n✅ Experimento '{method_name}' concluído.")
    print("\nVisão Geral dos Resultados Consolidados:")
    tracker.display()

# ==========================================================
# 4. PONTO DE ENTRADA DO SCRIPT
# ==========================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Executa um experimento de avaliação e registra o resultado.")
    
    prompt_group = parser.add_mutually_exclusive_group(required=False)
    prompt_group.add_argument('--prompt', type=str, 
                               help='O prompt a ser avaliado diretamente.')
    prompt_group.add_argument('--evolution-log', type=str, 
                               help='Caminho para o log da otimização para usar o melhor prompt encontrado.')
    
    parser.add_argument('--evaluation-method', type=str, default='embedding', 
                        choices=['embedding', 'pixel_cosine', 'pixel_euclidean'], 
                        help="Método de avaliação.")
    parser.add_argument('--method-name', type=str, help='(Opcional) Sobrescreve o nome do método.')
    parser.add_argument('--pairs-csv', type=str, required=True, help='Caminho para o arquivo .csv de pares.')
    parser.add_argument('--base-image-dir', type=str, required=True, help='Diretório raiz das imagens.')
    parser.add_argument('--model-name', type=str, default='InternVL3-2B', help='Nome do modelo base.')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='(Opcional) Caminho para a pasta do checkpoint (LoRA ou Head).')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'], help='Métrica para o método embedding.')
    parser.add_argument('--output-csv', type=str, help='(Opcional) Sobrescreve o caminho do log mestre.')
    parser.add_argument('--plot', action='store_true', help='Gera e salva um gráfico de densidade.')
    parser.add_argument('--force', action='store_true', help='Força a re-execução do experimento.')
    
    # VVV --- [ARGUMENTO NOVO ADICIONADO] --- VVV
    parser.add_argument(
        '--projection-output-dim', 
        type=int, 
        default=512,
        help='Dimensão de saída do ProjectionHead (necessário se carregar um checkpoint de Head).'
    )
    # ^^^ --- [FIM DO ARGUMENTO NOVO] --- ^^^
    
    args = parser.parse_args()
    
    if args.evaluation_method == 'embedding' and not args.prompt and not args.evolution_log:
         # [MODIFICADO] A validação foi ajustada, pois um checkpoint *sem* um prompt agora é válido
         if not args.checkpoint_path:
            parser.error("Para o método 'embedding' sem um checkpoint, você deve fornecer --prompt ou --evolution-log.")
         elif not args.prompt:
            print("Aviso: Rodando com checkpoint mas sem prompt explícito. Usando prompt padrão (se houver) ou vazio.")
            # Permite que o script continue, assumindo que o prompt será o que você pediu
            # (que será passado pela linha de comando)

    main(args)