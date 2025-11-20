# Em: src/finetuning/rl_trainer.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import os
import logging
from tqdm import tqdm # [Opcional] Para uma barra de progresso
from typing import Dict, Any
import glob

# Importações do projeto
# O 'ProfessorNetwork' está em 'src/models/professor.py'
from src.models.professor import ProfessorNetwork 
# A 'ContrastiveLoss' está em 'src/finetuning/losses.py'
from src.finetuning.losses import ContrastiveLoss 

# Imports necessários para a função
from src.models.heads import ProjectionHead


logger = logging.getLogger(__name__)

# [MODIFICADO] A assinatura da função mudou para aceitar os 3 componentes do Aluno
def run_rl_curriculum_loop(
    base_model,
    student_connector,
    student_head,
    professor_model,
    dataset,
    epochs,
    student_lr,
    professor_lr,
    device,
    output_dir,
    candidate_pool_size, # K
    student_batch_size  # B
):
    """
    Orquestra o loop de treinamento Professor-Aluno.
    """
    print(f"Iniciando loop de RL: Pool de Candidatos (K)={candidate_pool_size}, Batch do Aluno (B)={student_batch_size}")
    
    # --- 1. Configurar Modelos e Otimizadores ---
    
    # Mover todos os componentes do modelo para o device
    base_model.to(device)
    student_connector.to(device)
    student_head.to(device)
    professor_model.to(device)
    
    # [MODIFICADO] Otimizador do Aluno (Student)
    # Criado APENAS com os parâmetros das camadas que queremos treinar
    trainable_student_params = list(student_connector.parameters()) + list(student_head.parameters())
    
    if not trainable_student_params:
        logger.error("ERRO: Nenhum parâmetro treinável encontrado no student_connector ou student_head.")
        return
        
    student_optimizer = optim.Adam(trainable_student_params, lr=student_lr)
    
    # Otimizador do Professor (Professor)
    professor_optimizer = optim.Adam(professor_model.parameters(), lr=professor_lr)
    
    # --- 2. Critério de Perda (O "Ambiente") ---
    
    # Usamos a 'ContrastiveLoss' que agora tem o método .forward_individual
    student_criterion = ContrastiveLoss(margin=1.0).to(device)
        
    # --- 3. DataLoader ---
    
    # O DataLoader nos dará "Pools de Candidatos" (tamanho K)
    data_loader = DataLoader(
        dataset, 
        batch_size=candidate_pool_size, 
        shuffle=True, 
        num_workers=4, # Ajuste conforme sua máquina
        pin_memory=True
    )

    print("Iniciando o loop de co-treinamento...")
    for epoch in range(epochs):
        print(f"\n--- Época {epoch + 1}/{epochs} ---")
        
        # [Opcional] Barra de progresso
        pbar = tqdm(data_loader, desc=f"Época {epoch+1}", unit="batch")
        
        for i, candidate_batch in enumerate(pbar):
            # Mover dados do pool para o device
            # [MODIFICADO] O dataset está no 'cpu', movemos o batch para 'device'
            img_a = candidate_batch['image_a'].to(device, non_blocking=True)
            img_b = candidate_batch['image_b'].to(device, non_blocking=True)
            labels = candidate_batch['label'].to(device, non_blocking=True)
            
            # --- [NOVO] O Forward Pass completo do Aluno (Student) ---
            def student_forward_pass(img_a_batch, img_b_batch):
                # Coloca as camadas treináveis em modo de treino
                student_connector.train()
                student_head.train()
                # O base_model permanece em .eval() (congelado)
                
                # [IMPORTANTE] Precisamos saber como o InternVL extrai features
                # Vamos assumir que é .extract_features()
                # TODO: Confirme o nome deste método
                try:
                    features_a = base_model.extract_features(img_a_batch)
                    features_b = base_model.extract_features(img_b_batch)
                except AttributeError:
                    logger.error("ERRO: 'base_model' não tem '.extract_features()'.")
                    logger.error("Precisa ser atualizado para o método correto (ex: 'model.vision_tower(...)')")
                    return
                except Exception as e:
                    logger.error(f"Erro ao extrair features do base_model: {e}")
                    return

                # Passa as features pela "ponte"
                connected_a = student_connector(features_a)
                connected_b = student_connector(features_b)
                
                # Passa pela "cabeça" para obter os embeddings finais
                emb_a = student_head(connected_a)
                emb_b = student_head(connected_b)
                
                return emb_a, emb_b

            # --- 1. Turno do Professor (Seleção de Amostras) ---
            
            # Coloca o Professor em modo de treino
            professor_model.train()
            
            # Geramos o "Estado" (State)
            with torch.no_grad():
                # [MODIFICADO] Usamos a função helper para obter embeddings
                # As camadas do aluno são colocadas em .eval() internamente
                student_connector.eval()
                student_head.eval()
                
                # Re-usa a lógica de extração (agora sem gradiente)
                features_a = base_model.extract_features(img_a)
                features_b = base_model.extract_features(img_b)
                connected_a = student_connector(features_a)
                connected_b = student_connector(features_b)
                emb_a = student_head(connected_a)
                emb_b = student_head(connected_b)
                
                # [FUNCIONA AGORA] Esta é a linha que implementamos em losses.py
                state_losses = student_criterion.forward_individual(emb_a, emb_b, labels)
                
                # Normalizar as perdas para estabilidade (ex: entre 0 e 1)
                state_losses_norm = (state_losses - state_losses.min()) / (state_losses.max() - state_losses.min() + 1e-6)
                state_input = state_losses_norm.unsqueeze(-1) # Shape [K, 1]

            # Obter Ação (a) - O Professor gera "logits" (pontuações)
            action_logits = professor_model(state_input).squeeze(-1) # Shape [K]
            
            # Amostrar o "Batch do Aluno" (B)
            prob_dist = Categorical(logits=action_logits)
            selected_indices = prob_dist.sample((student_batch_size,))
            selected_log_probs = prob_dist.log_prob(selected_indices) # Shape [B]
            
            # --- 2. Turno do Aluno (Treinamento) ---
            
            # 2a. Criar o batch do Aluno (B) usando os índices selecionados
            student_img_a = img_a[selected_indices]
            student_img_b = img_b[selected_indices]
            student_labels = labels[selected_indices]
            
            # 2b. Treinar o Aluno
            student_optimizer.zero_grad()
            
            # [MODIFICADO] Executa o forward pass completo do Aluno
            student_emb_a, student_emb_b = student_forward_pass(student_img_a, student_img_b)
            
            # Calcular a perda do Aluno *apenas* no batch selecionado
            # (agora usa o .forward() padrão da perda, que calcula a média)
            student_loss = student_criterion(student_emb_a, student_emb_b, student_labels)
            
            student_loss.backward()
            student_optimizer.step()
            
            # --- 3. Turno do Professor (Atualização com REINFORCE) ---
            
            # 3a. Obter a Recompensa (r)
            reward = student_loss.item() 
            
            # 3b. Calcular a Perda do Professor (REINFORCE)
            professor_optimizer.zero_grad()
            # Perda = -E[log_prob(a) * R]
            professor_loss = - (selected_log_probs * reward).mean() # Média sobre o batch B
            
            professor_loss.backward()
            professor_optimizer.step()

            # [Opcional] Atualiza a barra de progresso
            pbar.set_postfix({
                'Aluno_Loss': f"{student_loss.item():.4f}", 
                'Prof_Loss': f"{professor_loss.item():.4f}",
                'Reward': f"{reward:.4f}"
            })
        
        # Fim do loop de batch (pbar)
        pbar.close()
        
        # --- 4. Salvar Checkpoints no final da época ---
        print(f"Fim da Época {epoch+1}. Salvando checkpoints...")
        
        # [MODIFICADO] Salva o state_dict do Aluno (Connector + Head)
        student_checkpoint = {
            'new_connector_state_dict': student_connector.state_dict(),
            'projection_head_state_dict': student_head.state_dict()
        }
        # Este é o mesmo formato que o seu 'load_model' espera!
        torch.save(student_checkpoint, os.path.join(output_dir, f"training_checkpoint_epoch_{epoch+1}.pt"))
        
        # Salva o Professor
        torch.save(professor_model.state_dict(), os.path.join(output_dir, f"professor_model_epoch_{epoch+1}.pt"))

    print("\n✅ Treinamento de Currículo de RL concluído.")

# Função Helper para o Passo 3 (Carregamento Otimizado)
def _find_latest_epoch_file(checkpoint_path: str, pattern: str) -> str:
    """Encontra o arquivo de checkpoint com a maior época."""
    files = glob.glob(os.path.join(checkpoint_path, pattern))
    if not files: return None
    files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
    return files[0]

def _load_rl_models_for_ag(
    base_model,
    tokenizer,
    checkpoint_path: str, 
    projection_output_dim: int, 
    device: str
) -> Dict[str, Any]:
    """
    Carrega os modelos de RL (Student Head e Professor) e os emparelha com o base_model existente.
    [CRÍTICO]: Reutiliza o base_model para economizar VRAM.
    """
    # Define LLM_INPUT_DIM (confirmado por você como 1536)
    LLM_INPUT_DIM = 1536

    # 1. Carregar Student Head (Aluno)
    student_head = ProjectionHead(input_dim=LLM_INPUT_DIM, output_dim=projection_output_dim)
    head_ckpt_path = _find_latest_epoch_file(checkpoint_path, "student_head_epoch_*.pt")
    
    if head_ckpt_path is None:
        raise FileNotFoundError("Checkpoint do Student Head não encontrado.")
        
    student_head.load_state_dict(torch.load(head_ckpt_path, map_location='cpu'))
    student_head.to(device).eval()

    # 2. Carregar Professor
    professor_model = ProfessorNetwork(input_dim=1).to(device)
    prof_ckpt_path = _find_latest_epoch_file(checkpoint_path, "professor_model_epoch_*.pt")
    
    if prof_ckpt_path is None:
        raise FileNotFoundError("Checkpoint do Professor (RL) não encontrado.")
        
    professor_model.load_state_dict(torch.load(prof_ckpt_path, map_location='cpu'))
    professor_model.to(device).eval()

    # 3. Criar o Dicionário de Modelos (Otimizado para VRAM)
    return {
        "base_model": base_model, # <--- REUTILIZAÇÃO DE VRAM
        "student_head": student_head,
        "professor_model": professor_model,
        "tokenizer": tokenizer,
        "criterion": ContrastiveLoss(margin=1.0).to(device),
        "device": device
    }