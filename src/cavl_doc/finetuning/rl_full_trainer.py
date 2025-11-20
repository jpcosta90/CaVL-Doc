# Em: src/finetuning/rl_full_trainer.py
# (Substitua o arquivo inteiro por este)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import os
import logging
from tqdm import tqdm 
import csv # [NOVO] Importa o módulo CSV

# Importações do projeto
from src.models.professor import ProfessorNetwork 
from src.finetuning.losses import ContrastiveLoss 

try:
    from src.utils.embedding_utils import prepare_inputs_for_multimodal_embedding
except ImportError:
    from src.metrics.evaluation import prepare_inputs_for_multimodal_embedding
    logging.warning("Importando 'prepare_inputs_for_multimodal_embedding' de 'metrics/evaluation.py'. Considere movê-lo para 'src/utils'.")


logger = logging.getLogger(__name__)

EMBEDDING_PROMPT = "<image> Analyze this document"

def rl_full_collate_fn(batch):
    img_a_list = [item['image_a'] for item in batch]
    img_b_list = [item['image_b'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    return img_a_list, img_b_list, labels


def run_rl_full_loop(
    base_model,
    student_head,
    professor_model,
    tokenizer,
    dataset,
    epochs,
    student_lr,
    professor_lr,
    device,
    output_dir,
    candidate_pool_size, # K
    student_batch_size,  # B
    max_num_image_tokens
):
    
    print(f"Iniciando loop de RL (Full/Head-Only): Pool (K)={candidate_pool_size}, Batch (B)={student_batch_size}")
    
    # --- 1. Configurar Modelos e Otimizadores ---
    student_head.to(device)
    professor_model.to(device)
    
    trainable_student_params = list(student_head.parameters())
    student_optimizer = optim.Adam(trainable_student_params, lr=student_lr)
    
    professor_optimizer = optim.Adam(professor_model.parameters(), lr=professor_lr)
    
    # --- 2. Critério de Perda (O "Ambiente") ---
    student_criterion = ContrastiveLoss(margin=1.0).to(device)
        
    # --- 3. DataLoader ---
    data_loader = DataLoader(
        dataset, 
        batch_size=candidate_pool_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=rl_full_collate_fn
    )

    # --- [NOVO] Configurar o arquivo de Log CSV ---
    log_file_path = os.path.join(output_dir, "training_log.csv")
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'batch', 'aluno_loss', 'prof_loss', 'reward'])
    print(f"  -> Salvando log de treino em: {log_file_path}")
    # -----------------------------------------------

    print("Iniciando o loop de co-treinamento...")
    global_batch_step = 0
    for epoch in range(epochs):
        print(f"\n--- Época {epoch + 1}/{epochs} ---")
        pbar = tqdm(data_loader, desc=f"Época {epoch+1}", unit="batch")
        
        def student_forward_pass(pixel_values_list_a, pixel_values_list_b, student_head_mode_train=True):
            # ... (O resto desta função interna não muda) ...
            if student_head_mode_train:
                student_head.train()
            else:
                student_head.eval()

            embeddings = []
            for pixel_values_list in [pixel_values_list_a, pixel_values_list_b]:
                collated_embeddings = []
                for pv_sample in pixel_values_list:
                    inputs = prepare_inputs_for_multimodal_embedding(
                        base_model, tokenizer, pv_sample.to(device), EMBEDDING_PROMPT
                    )
                    outputs = base_model(
                        input_ids=inputs['input_ids'].to(device),
                        attention_mask=inputs['attention_mask'].to(device),
                        pixel_values=inputs['pixel_values'].to(device, dtype=torch.bfloat16),
                        image_flags=inputs['image_flags'].to(device),
                        output_hidden_states=True,
                        return_dict=True
                    )
                    pooled_output = outputs.hidden_states[-1].mean(dim=1).to(torch.float32)
                    final_embedding = student_head(pooled_output)
                    collated_embeddings.append(final_embedding)
                embeddings.append(torch.cat(collated_embeddings, dim=0))

            return embeddings[0], embeddings[1]

        # Loop de Treinamento
        for i, (img_a_list, img_b_list, labels) in enumerate(pbar):
            labels = labels.to(device).float() 

            # --- 1. Turno do Professor ---
            professor_model.train()
            with torch.no_grad():
                emb_a, emb_b = student_forward_pass(img_a_list, img_b_list, student_head_mode_train=False)
                state_losses = student_criterion.forward_individual(emb_a, emb_b, labels)
                state_losses_norm = (state_losses - state_losses.min()) / (state_losses.max() - state_losses.min() + 1e-6)
                state_input = state_losses_norm.unsqueeze(-1) 

            action_logits = professor_model(state_input).squeeze(-1) 
            prob_dist = Categorical(logits=action_logits)
            selected_indices = prob_dist.sample((student_batch_size,))
            selected_log_probs = prob_dist.log_prob(selected_indices) 
            
            # --- 2. Turno do Aluno ---
            student_img_a = [img_a_list[i] for i in selected_indices]
            student_img_b = [img_b_list[i] for i in selected_indices]
            student_labels = labels[selected_indices]
            
            student_optimizer.zero_grad()
            student_emb_a, student_emb_b = student_forward_pass(student_img_a, student_img_b, student_head_mode_train=True)
            student_loss = student_criterion(student_emb_a, student_emb_b, student_labels)
            
            student_loss.backward()
            student_optimizer.step()
            
            # --- 3. Turno do Professor ---
            reward = student_loss.item() 
            professor_optimizer.zero_grad()
            professor_loss = - (selected_log_probs * reward).mean()
            
            professor_loss.backward()
            professor_optimizer.step()

            # [NOVO] Escreve as métricas no CSV
            log_writer.writerow([
                epoch + 1, 
                global_batch_step, 
                f"{student_loss.item():.4f}", 
                f"{professor_loss.item():.4f}", 
                f"{reward:.4f}"
            ])
            global_batch_step += 1
            # ---------------------------------

            pbar.set_postfix({
                'Aluno_Loss': f"{student_loss.item():.4f}", 
                'Prof_Loss': f"{professor_loss.item():.4f}",
                'Reward': f"{reward:.4f}"
            })
        
        pbar.close()
        
        # --- 4. Salvar Checkpoints ---
        print(f"Fim da Época {epoch+1}. Salvando checkpoints...")
        torch.save(
            student_head.state_dict(), 
            os.path.join(output_dir, f"student_head_epoch_{epoch+1}.pt")
        )
        torch.save(
            professor_model.state_dict(), 
            os.path.join(output_dir, f"professor_model_epoch_{epoch+1}.pt")
        )

    # [NOVO] Fecha o arquivo de log
    log_file.close()
    print("\n✅ Treinamento de Currículo de RL (Full/Head-Only) concluído.")