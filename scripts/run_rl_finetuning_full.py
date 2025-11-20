# Em: scripts/run_rl_finetuning_full.py
# (Use este código para o seu script)

import argparse
import os
import json
import torch
import warnings
from datetime import datetime
import sys
import random 
import logging

# Suprimir UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Importações de módulos do seu projeto
from src.data_loaders.documentpairs import DocumentPairDataset
from src.models.lvlm_handler import load_model, warm_up_model
from src.utils.helpers import setup_experiment_dir
from torch.utils.data import Subset 

# --- [NOVAS IMPORTAÇÕES] ---
from src.models.professor import ProfessorNetwork
from src.models.heads import ProjectionHead
# [MODIFICADO] Importa o NOVO trainer "full"
from src.finetuning.rl_full_trainer import run_rl_full_loop 
# ---------------------------

logger = logging.getLogger(__name__)

def main(args):
    # --- 1. Configurar Ambiente e Saída ---
    print("--- 1. Configurando o experimento (RL Full/Head-Only Training) ---")
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # <<< MUDANÇA 1: Usa o novo 'args.dataset_name' >>>
    experiment_name = f"{args.dataset_name}_{args.model_name}_RL_Full_Head_{args.training_sample_size}_{timestamp}"
    output_dir = setup_experiment_dir("checkpoints", experiment_name)
    
    training_config = {
        # <<< MUDANÇA 2: Salva o dataset_name no config >>>
        'dataset_name': args.dataset_name,
        'model_name': args.model_name,
        'load_in_4bit': args.load_in_4bit,
        'pairs_csv': args.pairs_csv,
        'base_image_dir': args.base_image_dir,
        'training_sample_size': args.training_sample_size,
        'epochs': args.epochs,
        'student_lr': args.student_lr, # (Learning rate da Cabeça)
        'professor_lr': args.professor_lr,
        'candidate_pool_size': args.candidate_pool_size,
        'student_batch_size': args.student_batch_size,
        'projection_output_dim': args.projection_output_dim,
        'max_num_image_tokens': args.max_num_image_tokens,
        'timestamp': timestamp,
        'output_dir': output_dir
    }
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(training_config, f, indent=4)
    print(f"   -> Os checkpoints e logs serão salvos em: {output_dir}")

    # --- 2. Carregar Dados e Modelos ---
    print("\n--- 2. Carregando dados e modelos ---")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"   -> Carregando Modelo Base (Congelado) (4-bit: {args.load_in_4bit})...")
    base_model, processor, tokenizer, _, _ = load_model(
        model_name=args.model_name,
        adapter_path=None,   
        load_in_4bit=args.load_in_4bit,     
        projection_output_dim=args.projection_output_dim
    )
    
    base_model.requires_grad_(False)
    print(f"   -> Modelo base '{args.model_name}' carregado e congelado.")

    # [NOVO] Instanciamos a "Cabeça" (Student) treinável
    
    # TODO: Confirme esta dimensão de entrada
    # (É a dimensão do hidden_state do LLM, ex: 1536 para InternVL3-2B?)
    LLM_HIDDEN_DIM = 1536 
    
    student_head = ProjectionHead(
        input_dim=LLM_HIDDEN_DIM, 
        output_dim=args.projection_output_dim
    ).to(DEVICE)
    student_head.train() 
    print("   -> Camada 'Student' (ProjectionHead) instanciada e pronta para treino.")
    
    # [NOVO] Instanciamos o Professor
    print("   -> Instanciando Modelo Professor (Professor)...")
    professor_model = ProfessorNetwork(input_dim=1).to(DEVICE)
    professor_model.train()
    
    warm_up_model(base_model, processor)

    # Carregar dataset
    dataset = DocumentPairDataset(
        csv_path=args.pairs_csv, 
        base_dir=args.base_image_dir,
        input_size=448, 
        max_num=args.max_num_image_tokens,
        device='cpu' 
    )
    print(f"   -> Dataset completo carregado de '{args.pairs_csv}' com {len(dataset)} amostras.")

    if args.training_sample_size > 0 and args.training_sample_size < len(dataset):
        indices = random.sample(range(len(dataset)), args.training_sample_size)
        train_subset = Subset(dataset, indices)
    else:
        train_subset = dataset

    # --- 3. Iniciando o ciclo de treinamento de RL ---
    print("\n--- 3. Iniciando o ciclo de treinamento do Currículo de RL (Full/Head-Only) ---")
    
    # [MODIFICADO] Chamada para o NOVO trainer e função
    run_rl_full_loop(
        base_model=base_model,
        student_head=student_head,
        professor_model=professor_model,
        tokenizer=tokenizer, 
        dataset=train_subset,
        epochs=args.epochs,
        student_lr=args.student_lr,
        professor_lr=args.professor_lr,
        device=DEVICE,
        output_dir=output_dir,
        candidate_pool_size=args.candidate_pool_size,
        student_batch_size=args.student_batch_size,
        max_num_image_tokens=args.max_num_image_tokens
    )
    
    logger.info(f"\nTreinamento de RL finalizado. Resultados salvos em: {output_dir}")
    print("\n✅ Treinamento de Currículo de RL (Full/Head-Only) concluído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para fine-tuning (Full/Head-Only) com Currículo de RL.")
    
    parser.add_argument("--model-name", type=str, default="InternVL3-2B")
    
    # <<< MUDANÇA 3: Adiciona o novo argumento com o default "LA-CDIP" >>>
    parser.add_argument("--dataset-name", type=str, default="LA-CDIP", help="Nome do dataset (usado no nome do checkpoint).")
    
    parser.add_argument("--pairs-csv", type=str, required=True)
    parser.add_argument("--base-image-dir", type=str, required=True)
    parser.add_argument("--projection-output-dim", type=int, default=512)
    parser.add_argument("--max-num_image_tokens", type=int, default=4)
    parser.add_argument("--training-sample-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    
    parser.add_argument("--load-in-4bit", action='store_true', default=True, help="Carrega o modelo base em 4-bit (congelado).")

    parser.add_argument("--student-lr", type=float, default=1e-4, help="Taxa de aprendizado do Aluno (ProjectionHead).")
    parser.add_argument("--professor-lr", type=float, default=1e-4, help="Taxa de aprendizado do Professor (agente RL).")
    
    parser.add_argument("--candidate-pool-size", type=int, default=64, help="K: Quantos pares o Professor avalia por vez.")
    parser.add_argument("--student-batch-size", type=int, default=16, help="B: Quantos pares (selecionados pelo Prof) o Aluno treina.")
    
    args = parser.parse_args()
    
    if args.student_batch_size > args.candidate_pool_size:
        raise ValueError("O student_batch_size (B) não pode ser maior que o candidate_pool_size (K).")
        
    main(args)