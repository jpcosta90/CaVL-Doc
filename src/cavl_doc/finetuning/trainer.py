import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm 
import os
from peft import get_peft_model, LoraConfig, TaskType
import sys
import random
from torch.utils.data import DataLoader, Subset
import logging
import json
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.amp # Deve estar aqui

# Importações de outros módulos do seu projeto
from src.utils.image_processing import load_image
from src.models.lvlm_handler import prepare_inputs_for_multimodal_embedding, get_lora_target_modules_vision_encoder, get_full_finetune_target_blocks_vision_encoder
from src.finetuning.losses import ContrastiveLoss

# Importar a ContrastiveLoss
from src.finetuning.losses import ContrastiveLoss
# Importar ProjectionHead
from src.models.heads import ProjectionHead

def run_training_loop(
    model, 
    tokenizer, 
    dataset, 
    prompt: str,
    output_dir: str,
    num_vision_layers_to_train: int = 0,
    num_epochs: int = 5, 
    learning_rate: float = 1e-5,
    samples_per_epoch: int = 2000
):
    """
    Executa o ciclo de fine-tuning (Aprendizado Contrastivo) com QLoRA,
    baseado na lógica validada do notebook.
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. Preparar o Modelo com QLoRA ---
    print("--- 1. Preparando modelo para fine-tuning com QLoRA... ---")
    
    vision_layer_names = get_lora_target_modules_vision_encoder(model, num_vision_layers_to_train)
    target_modules = ["q_proj", "v_proj"] + vision_layer_names
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=learning_rate)
    criterion = ContrastiveLoss()
    
    peft_model.to(DEVICE)
    peft_model.train()

    # --- 2. Loop de Épocas ---
    print(f"\n🚀 Iniciando Treinamento por {num_epochs} épocas...")
    
    # <<< NOVO: Barra de progresso para as épocas >>>
    epochs_progress_bar = tqdm(range(num_epochs), desc="Treinamento de Épocas", leave=True, file=sys.stdout)

    for epoch in epochs_progress_bar: # Usa a nova barra de épocas
        total_loss = 0
        num_samples = min(samples_per_epoch, len(dataset.df))
        feedback_df = dataset.df.sample(n=num_samples, replace=False)
        
        # O print da época foi movido para a descrição da barra de épocas
        # print(f"\nÉpoca {epoch+1}/{num_epochs}: Treinando com {len(feedback_df)} amostras...") # <<< REMOVER ESTA LINHA

        # Usa tqdm para uma barra de progresso mais informativa para as amostras
        # <<< CORREÇÕES APLICADAS AQUI >>>
        samples_progress_bar = tqdm(feedback_df.iterrows(), total=len(feedback_df), 
                                    desc=f"Época {epoch+1}/{num_epochs} (Amostras)",
                                    leave=False, # Não deixa a barra de progresso na tela após o loop da época
                                    file=sys.stdout # Garante que está usando a saída padrão
                                   )

        for index, row in samples_progress_bar: # Usa a nova barra de amostras
            optimizer.zero_grad()
            
            path_a = os.path.join(dataset.base_dir, row["file_a_path"])
            path_b = os.path.join(dataset.base_dir, row["file_b_path"])
            label = torch.tensor([float(row["is_equal"])], device=DEVICE)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Processa a imagem A
                pixel_values1 = load_image(path_a, max_num=10, use_thumbnail=False).to(DEVICE, dtype=torch.bfloat16)
                inputs1 = prepare_inputs_for_multimodal_embedding(peft_model, tokenizer, pixel_values1, prompt)
                
                outputs1 = peft_model.base_model(
                    input_ids=inputs1['input_ids'],
                    attention_mask=inputs1['attention_mask'],
                    pixel_values=inputs1['pixel_values'],
                    image_flags=inputs1['image_flags'],
                    output_hidden_states=True,
                    return_dict=True
                )
                embedding1 = outputs1.hidden_states[-1].mean(dim=1)

                # Processa a imagem B
                pixel_values2 = load_image(path_b, max_num=10, use_thumbnail=False).to(DEVICE, dtype=torch.bfloat16)
                inputs2 = prepare_inputs_for_multimodal_embedding(peft_model, tokenizer, pixel_values2, prompt)
                
                outputs2 = peft_model.base_model(
                    input_ids=inputs2['input_ids'],
                    attention_mask=inputs2['attention_mask'],
                    pixel_values=inputs2['pixel_values'],
                    image_flags=inputs2['image_flags'],
                    output_hidden_states=True,
                    return_dict=True
                )
                embedding2 = outputs2.hidden_states[-1].mean(dim=1)
                
                loss = criterion(embedding1, embedding2, label)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Atualiza a barra de progresso das amostras
            samples_progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = total_loss / len(feedback_df)
        
        # <<< Atualiza a barra de progresso das épocas e imprime a perda média da época >>>
        epochs_progress_bar.set_postfix(avg_loss=f"{avg_loss:.6f}")
        # print(f"Fim da Época {epoch+1}: Perda Média = {avg_loss:.6f}", flush=True) # <<< REMOVER ESTA LINHA

    # --- 3. Salvar os Adaptadores Treinados ---
    print(f"\n✅ Treinamento concluído!")
    print(f"Salvando adaptadores LoRA treinados em: {output_dir}")
    peft_model.save_pretrained(output_dir)
    print("Adaptadores salvos com sucesso.")

    return peft_model

# Obter o logger
logger = logging.getLogger(__name__)

# --- Função Collate SIMPLIFICADA (Apenas Visual) ---
# (Certifique-se que esta função também esteja no seu arquivo)
def collate_fn_for_finetune_visual_only(batch_list, model_device):
    """
    Função collate_fn para DataLoader.
    Processa um batch de itens do dataset e os prepara como tensores.
    Ignora texto, foca apenas nas imagens e no label.
    """
    item = batch_list[0] # batch_size=1
    pixel_values1 = item["image_a"]
    pixel_values2 = item["image_b"]
    label = torch.tensor([item["label"]], device=model_device, dtype=torch.float)
    return pixel_values1, pixel_values2, label


# --- Função Principal do Loop de Treinamento (COM QUICK FIX PARA SHAPE) ---
def run_training_loop_full_finetune(
    model,
    dataset: Dataset, # Pode ser o Dataset completo ou um Subset
    epochs: int,
    learning_rate: float,
    device: str,
    output_dir: str,
    projection_output_dim: int = 512
    # REMOVEMOS: tokenizer, train_vision_layers, prompt_template
):
    logger.info("--- 3. Iniciando o ciclo de treinamento (Estratégia: Treinar Ponte + Head) ---")

    # --- 1. Definir Arquitetura e Congelar Modelo Base ---
    VISION_OUTPUT_DIM = 1024 # Saída do model.vision_model
    LLM_INPUT_DIM = 1536     # Entrada do language_model (e saída da nossa ponte)

    logger.info("   -> Congelando o modelo base (ViT + LLM)...")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # --- 2. Criar Camadas Treináveis ---
    logger.info(f"   -> Criando nova ponte treinável: {VISION_OUTPUT_DIM} -> {LLM_INPUT_DIM}")
    new_connector = nn.Linear(VISION_OUTPUT_DIM, LLM_INPUT_DIM).to(device).to(torch.bfloat16)
    new_connector.train()

    logger.info(f"   -> Criando novo projection head treinável: {LLM_INPUT_DIM} -> {projection_output_dim}")
    projection_head = ProjectionHead(input_dim=LLM_INPUT_DIM,
                                     output_dim=projection_output_dim).to(device).to(torch.bfloat16)
    projection_head.train()

    # --- 3. Otimizador ---
    params_to_optimize = list(new_connector.parameters())
    params_to_optimize += list(projection_head.parameters())

    logger.info("   -> Configurando o Otimizador...")
    optimizer = optim.AdamW(params_to_optimize, lr=learning_rate)
    criterion = ContrastiveLoss(margin=1.0)

    # --- 4. DataLoader ---
    train_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: collate_fn_for_finetune_visual_only(x, device)
    )

    # --- 5. Loop de Treinamento ---
    for epoch in tqdm(range(epochs), desc="Épocas"):
        new_connector.train()
        projection_head.train()
        model.eval() # Garante que o modelo base fique em eval

        total_loss = 0
        warned_about_shape = False # Para evitar spam de warnings

        samples_progress_bar = tqdm(train_dataloader, desc=f"Loop de Treinamento (Época {epoch+1}/{epochs})")
        # O collate_fn agora retorna (pixel_values1, pixel_values2, label)
        for i, (pixel_values1, pixel_values2, label) in enumerate(samples_progress_bar):
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

                # --- Imagem 1 ---
                with torch.no_grad():
                    # VVV --- QUICK FIX: Take only the first image if multiple are loaded --- VVV
                    single_pixel_values1 = pixel_values1
                    if pixel_values1.shape[0] > 1:
                        if not warned_about_shape: # Mostra o aviso apenas uma vez por época
                             logger.warning(f"Dataset retornou shape {pixel_values1.shape} para img A (esperado [1,...]), usando apenas o primeiro item.", exc_info=False)
                             warned_about_shape = True # Marca que já avisou
                        single_pixel_values1 = pixel_values1[0:1, :, :, :]
                    # ^^^ --- END QUICK FIX --- ^^^

                    vision_outputs_1 = model.vision_model(pixel_values=single_pixel_values1)[0] # Shape [1, N, 1024]
                    pooled_features_1 = vision_outputs_1[:, 0, :] # Shape [1, 1024]

                llm_embedding_1 = new_connector(pooled_features_1) # Shape [1, 1536]

                # --- Imagem 2 (repetir fix) ---
                with torch.no_grad():
                    # VVV --- QUICK FIX --- VVV
                    single_pixel_values2 = pixel_values2
                    if pixel_values2.shape[0] > 1:
                        # Não precisa avisar de novo para a imagem B
                        single_pixel_values2 = pixel_values2[0:1, :, :, :]
                    # ^^^ --- END QUICK FIX --- ^^^

                    vision_outputs_2 = model.vision_model(pixel_values=single_pixel_values2)[0]
                    pooled_features_2 = vision_outputs_2[:, 0, :]

                llm_embedding_2 = new_connector(pooled_features_2) # Shape [1, 1536]

                # --- Projection e Loss ---
                # Entradas devem ser [1, 1536], saídas [1, 512]
                projected_embedding_1 = projection_head(llm_embedding_1)
                projected_embedding_2 = projection_head(llm_embedding_2)

                # Loss agora recebe [1, 512], [1, 512], [1] - Deve funcionar!
                loss = criterion(projected_embedding_1, projected_embedding_2, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            samples_progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Época {epoch+1}/{epochs}, Perda Média: {avg_loss:.4f}")

        # --- Salvar Checkpoint ---
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_output_dir, exist_ok=True)

        torch.save(
            {
                'model_state_dict': {}, # Modelo base não treinado
                'new_connector_state_dict': new_connector.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            },
            os.path.join(epoch_output_dir, "training_checkpoint.pt")
        )
        logger.info(f"Checkpoint da Época {epoch+1} salvo em '{epoch_output_dir}'")

    logger.info("Treinamento concluído!")