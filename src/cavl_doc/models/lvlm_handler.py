import torch
import pandas as pd
from typing import List, Optional, Dict
from torch.utils.data import Dataset, DataLoader
import os
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
import torch.nn as nn
from peft import PeftModel
from src.models.misc import NewConnector # Ou onde quer que NewConnector esteja definida
from src.models.heads import ProjectionHead
import glob

import logging
from typing import Tuple, Optional # Importar Optional para tipos mais claros

# Importar o ProjectionHead
from src.models.heads import ProjectionHead

def _get_quantization_config() -> BitsAndBytesConfig:
    """Retorna a configuração padrão de quantização BitsAndBytes."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

logger = logging.getLogger(__name__)

def warm_up_model(model, tokenizer):
    """
    Realiza uma inferência com uma única imagem para "aquecer" o modelo
    InternVL usando seu método específico `model.chat()`.

    Versão corrigida para lidar com a saída de `model.chat` que retorna
    apenas um valor (a string da resposta).

    Args:
        model: O modelo InternVL carregado.
        tokenizer: O tokenizer correspondente ao modelo.
    """
    print("Aquecendo o modelo com uma única imagem de teste...")

    try:
        # 1. Criar um tensor 'pixel_values' falso para uma única imagem
        pixel_values = torch.randn(1, 3, 448, 448, dtype=model.dtype, device=model.device)

        # 2. Definir a lista de patches para uma imagem
        num_patches_list = [pixel_values.size(0)]

        # 3. Criar um prompt e configuração simples
        question = "<image>\nDescreva esta imagem em poucas palavras."
        generation_config = dict(max_new_tokens=10, do_sample=False)

        # 4. Executar a inferência com `model.chat`
        with torch.no_grad():
            # CORREÇÃO: Atribuir a saída a uma única variável 'response'
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )

        print(f"Saída de teste gerada (será descartada): '{response.strip()}'")
        print("Aquecimento do modelo concluído com sucesso.")

    except Exception as e:
        print(f"Ocorreu um erro durante o aquecimento do modelo: {e}")
        print("O modelo pode não estar totalmente inicializado. Verifique a mensagem de erro.")

def prepare_inputs_for_multimodal_embedding(model, tokenizer, pixel_values, question, num_patches=None):
    device = model.device

    if '<image>' not in question:
        question = '<image>\n' + question

    if num_patches is None:
        num_patches = pixel_values.shape[0]  # número de blocos/imagens

    # Criar espaço de input para os vit_embeds
    image_tokens = '<img>' + ('<IMG_CONTEXT>' * model.num_image_token * num_patches) + '</img>'
    question = question.replace('<image>', image_tokens, 1)

    inputs = tokenizer(question, return_tensors='pt').to(device)
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'pixel_values': pixel_values.to(device),
        'image_flags': torch.ones(pixel_values.shape[0], dtype=torch.long).to(device)
    }

class SBR:
    """
    Implementação do Subclass Brain Registry (SBR).

    Gerencia um registro de prompts e seus resultados de fitness (EER)
    usando um DataFrame do Pandas, seguindo a taxonomia do SuperBrain.
    """
    def __init__(self):
        """Inicializa um SBR vazio e um contador de IDs."""
        self.columns = ['prompt_id', 'prompt_text', 'eer_score', 'generation', 'parent_ids']
        self.registry = pd.DataFrame(columns=self.columns)
        self.next_prompt_id = 0
        print("Subclass Brain Registry (SBR) inicializado.")

    def add_entry(self, prompt_text: str, eer_score: float, generation: int = 0, parent_ids: Optional[List[int]] = None):
        """
        Adiciona uma nova entrada (prompt e seu resultado) ao registro.
        """
        if parent_ids is None:
            parent_ids = []

        new_entry = {
            'prompt_id': self.next_prompt_id,
            'prompt_text': prompt_text,
            'eer_score': eer_score,
            'generation': generation,
            'parent_ids': parent_ids
        }
        
        # Usar pd.concat para adicionar a nova linha
        self.registry = pd.concat([self.registry, pd.DataFrame([new_entry])], ignore_index=True)
        print(f"Adicionada entrada ID {self.next_prompt_id} com EER: {eer_score:.4f}")
        self.next_prompt_id += 1

    def get_best_prompts(self, n: int = 5) -> pd.DataFrame:
        """Retorna os 'n' melhores prompts (menor EER)."""
        return self.registry.sort_values(by='eer_score', ascending=True).head(n)

    def save(self, file_path: str = 'sbr_registry.csv'):
        """Salva o registro atual em um arquivo CSV."""
        self.registry.to_csv(file_path, index=False)
        print(f"SBR salvo em '{file_path}'")

    def load(self, file_path: str = 'sbr_registry.csv'):
        """Carrega um registro de um arquivo CSV."""
        try:
            self.registry = pd.read_csv(file_path)
            # Garante que a coluna de parent_ids seja lida como lista
            self.registry['parent_ids'] = self.registry['parent_ids'].apply(eval)
            self.next_prompt_id = self.registry['prompt_id'].max() + 1
            print(f"SBR carregado de '{file_path}'. Próximo ID: {self.next_prompt_id}")
        except FileNotFoundError:
            print(f"Arquivo '{file_path}' não encontrado. Iniciando com SBR vazio.")

    def __repr__(self):
        """Representação em string do objeto."""
        return f"<SBR com {len(self.registry)} entradas>"

    def display(self, n: int = 10):
        """Exibe as últimas 'n' entradas do registro."""
        print("--- Estado Atual do SBR ---")
        if self.registry.empty:
            print("O registro está vazio.")
        else:
            # Exibe as últimas n entradas ou todas se forem menos que n
            display_df = self.registry.tail(n).reset_index(drop=True)
            print(display_df.to_string())
        print("--------------------------")

def get_best_prompt_overall(sbr_df: pd.DataFrame) -> Optional[Dict]:
    """
    Encontra e retorna as informações do melhor prompt de um DataFrame SBR.
    
    NOVA REGRA: Em caso de empate no 'eer_score', o prompt com o menor
    comprimento de texto (mais curto) será escolhido como o melhor.

    Args:
        sbr_df (pd.DataFrame): O DataFrame do Subclass Brain Registry.

    Returns:
        Optional[Dict]: Um dicionário com as informações do melhor prompt.
    """
    if sbr_df.empty:
        print("SBR DataFrame está vazio. Nenhum prompt para retornar.")
        return None
    
    # 1. Encontrar o menor EER score no registro
    min_eer_score = sbr_df['eer_score'].min()
    
    # 2. Filtrar todas as linhas que têm esse score mínimo
    best_prompts_df = sbr_df[sbr_df['eer_score'] == min_eer_score]
    
    # 3. Verificar se há um empate (mais de uma linha com o melhor score)
    if len(best_prompts_df) > 1:
        print(f"Empate detectado: {len(best_prompts_df)} prompts com EER de {min_eer_score:.4f}. Aplicando regra de desempate (prompt mais curto)...")
        
        # 4. Regra de desempate: encontrar o prompt mais curto entre os melhores
        # idxmin() aqui selecionará o índice do prompt com o menor comprimento
        best_prompt_index = best_prompts_df['prompt_text'].str.len().idxmin()
        best_prompt_series = best_prompts_df.loc[best_prompt_index]
        
    else:
        # Se não houver empate, apenas pegue a única linha
        best_prompt_series = best_prompts_df.iloc[0]
        
    return best_prompt_series.to_dict()

# --- 1. Dataset and DataLoader ---
# This class is now perfect, as it loads one document (sequence of tokens) at a time.
class DocumentTokenDataset(Dataset):
    def __init__(self, file_paths, embeddings_dir):
        self.file_paths = file_paths
        self.embeddings_dir = embeddings_dir

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        full_path = os.path.join(self.embeddings_dir, self.file_paths[idx])
        return torch.load(full_path, weights_only=True)


def load_my_finetuned_model(base_model_name='OpenGVLab/InternVL3-2B', adapter_path='../../models/internvl_rl_qlora_adapters'):
    """
    Carrega o modelo base original e anexa os adaptadores LoRA treinados.
    """
    print(f"--- 1. Carregando o Modelo Base: {base_model_name} ---")
    
    # Configurações para carregar o modelo base em 4-bit (exatamente como antes)
    quant_config = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.bfloat16
    }
    
    # Carrega o modelo base
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **quant_config
    )
    
    print(f"--- 2. Carregando e Anexando Adaptadores de: {adapter_path} ---")
    
    # Carrega os adaptadores e os anexa ao modelo base
    finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # É uma boa prática "fundir" os adaptadores para inferência mais rápida, se possível
    # finetuned_model = finetuned_model.merge_and_unload() # Opcional, mas recomendado para velocidade

    # O tokenizer e o processor continuam sendo os do modelo base
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, use_fast=False)
    
    print("\n✅ Modelo fine-tuned carregado com sucesso!")
    
    return finetuned_model.eval(), processor, tokenizer

def load_internvl(model_name: str = 'InternVL3-2B'):
    """
    Carrega um modelo InternVL, seu processador e tokenizer correspondentes.

    Args:
        model_version (str): A versão do modelo a ser carregada ('2B' ou '14B').

    Returns:
        tuple: Uma tupla contendo o (modelo, processador, tokenizer) carregados.
    """
    lab_name = 'OpenGVLab'

    print(f"\n--- Carregando o modelo: {model_name} ---")
    
    model_path = f"{lab_name}/{model_name}"
    
    quant_config = {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.bfloat16
    }
    
    # Carregar o processador (que contém o image_processor e o tokenizer)
    processor = AutoProcessor.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    # Carregar o modelo
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        **quant_config
    ).eval()
    
    # --- NOVO: Acessar o tokenizer a partir do processador ---
    # Não é necessário carregar separadamente, o que é mais rápido e eficiente.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    print(f"--- Modelo, Processador e Tokenizer para {model_name} carregados com sucesso. ---")
    
    # Retorna todos os três objetos
    return model, processor, tokenizer

# ==============================================================================
# FUNÇÃO DE GERAÇÃO DE EMBEDDING (ATUALIZADA)
# ==============================================================================
@torch.no_grad()
def generate_refined_embedding(raw_tokens, attention_encoder):
    """
    Gera o embedding final passando os tokens brutos pelo AttentionEncoder treinado.
    VERSÃO ATUALIZADA: A lógica de batching é feita aqui, fora do modelo.
    """
    device = next(attention_encoder.parameters()).device
    raw_tokens = raw_tokens.to(device)
    
    # 1. Adicionar a dimensão de batch para o modelo
    # O modelo Transformer espera um input 3D: (Batch, Seq_Len, Features).
    # Nossos 'raw_tokens' são 2D (Seq_Len, Features), então criamos um lote de tamanho 1.
    batched_tokens = raw_tokens.unsqueeze(0)
    
    # 2. Refinar os tokens com o AttentionEncoder
    refined_batched_tokens = attention_encoder(batched_tokens)
    
    # 3. Remover a dimensão de batch para o pooling
    refined_tokens = refined_batched_tokens.squeeze(0)
    
    # 4. Fazer o mean pooling nos tokens REFINADOS para criar o embedding final
    embedding = refined_tokens.mean(dim=0)
    
    return embedding.cpu()

def _get_quantization_config() -> BitsAndBytesConfig:
    """
    Retorna uma configuração padrão de quantização para QLoRA em 4-bit.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

logger = logging.getLogger(__name__)

# (A função _get_quantization_config não muda)
def _get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def load_model(
    model_name: str = 'InternVL3-2B',
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
    projection_output_dim: int = 512
) -> Tuple[AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Optional[nn.Module], Optional[nn.Module]]:
    
    model_path = f"OpenGVLab/{model_name}"
    final_model: AutoModelForCausalLM = None
    new_connector_loaded: Optional[nn.Module] = None
    projection_head_loaded: Optional[nn.Module] = None

    # Determinar tipo de checkpoint
    is_lora_checkpoint = False
    is_connector_head_checkpoint = False
    is_rl_head_checkpoint = False 
    rl_head_ckpt_path = None # [NOVO] O caminho será determinado
    effective_load_in_4bit = load_in_4bit

    if adapter_path and os.path.isdir(adapter_path):
        lora_config_path = os.path.join(adapter_path, 'adapter_config.json')
        connector_head_ckpt_path = os.path.join(adapter_path, 'training_checkpoint.pt')
        
        # VVV --- [LÓGICA DE BUSCA MODIFICADA] --- VVV
        # Procura por *qualquer* checkpoint de cabeça de aluno (student_head)
        epoch_files = glob.glob(os.path.join(adapter_path, 'student_head_epoch_*.pt'))
        
        if os.path.exists(lora_config_path):
            is_lora_checkpoint = True
            logger.info(f"Detectado checkpoint LoRA em: {adapter_path}")
        
        elif os.path.exists(connector_head_ckpt_path):
            is_connector_head_checkpoint = True
            logger.info(f"Detectado checkpoint Connector/Head (Antigo) em: {adapter_path}")
            if load_in_4bit:
                logger.warning("Checkpoints Connector/Head requerem bfloat16. Ignorando load_in_4bit=True.")
            effective_load_in_4bit = False 
        
        elif len(epoch_files) > 0:
            is_rl_head_checkpoint = True
            
            # Pega a época mais recente (ex: 1, 2, 3 -> pega 3)
            epoch_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
            rl_head_ckpt_path = epoch_files[0] # Pega o checkpoint da época mais alta
            
            logger.info(f"Detectado checkpoint RL-Head (Aluno). Usando a época mais recente: {os.path.basename(rl_head_ckpt_path)}")
        # ^^^ --- [FIM DA LÓGICA MODIFICADA] --- ^^^
        
        else:
            logger.warning(f"Diretório de checkpoint '{adapter_path}' não contém 'adapter_config.json', 'training_checkpoint.pt' ou 'student_head_epoch_*.pt'. Carregando apenas modelo base.")
    
    elif adapter_path:
        logger.warning(f"Caminho do checkpoint '{adapter_path}' não encontrado. Carregando apenas modelo base.")

    # --- 1. Carregar o Modelo Base ---
    # (O restante desta seção não muda)
    print(f"--- 1. Carregando o Modelo Base: {model_path} ---")
    quantization_config_obj = None
    model_dtype = torch.bfloat16 
    if effective_load_in_4bit:
        print("    -> Configurando para carregar em 4-bit (QLoRA).")
        quantization_config_obj = _get_quantization_config()
    else:
        print("    -> Configurando para carregar em bfloat16 (precisão total).")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=model_dtype,
            device_map="auto", 
            quantization_config=quantization_config_obj
        )
        base_model.eval()
    except Exception as e:
        logger.error(f"Falha ao carregar modelo base {model_path}: {e}", exc_info=True)
        raise e

    # --- 2. Processar Checkpoint (se aplicável) ---
    if is_lora_checkpoint:
        # ... (lógica do LoRA não muda) ...
        print(f"--- 2. Anexando Adaptadores LoRA de: {adapter_path} ---")
        try:
            final_model = PeftModel.from_pretrained(base_model, adapter_path)
            final_model.eval()
            print("    -> Adaptadores LoRA aplicados com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar adaptadores LoRA: {e}", exc_info=True)
            final_model = base_model

    elif is_connector_head_checkpoint:
        # ... (lógica do Connector/Head antigo não muda) ...
        print(f"--- 2. Carregando Camadas Connector/Head (Antigo) de: {adapter_path} ---")
        try:
            final_model = base_model
            final_model.eval() 
            ckpt_path = os.path.join(adapter_path, 'training_checkpoint.pt')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            VISION_OUTPUT_DIM = 1024
            LLM_INPUT_DIM = 1536 
            if 'new_connector_state_dict' in checkpoint:
                new_connector_loaded = NewConnector(VISION_OUTPUT_DIM, LLM_INPUT_DIM)
                new_connector_loaded.load_state_dict(checkpoint['new_connector_state_dict'])
                new_connector_loaded = new_connector_loaded.to(final_model.device).to(model_dtype).eval()
                print("    -> Camada NewConnector (Antiga) carregada com sucesso.")
            if 'projection_head_state_dict' in checkpoint:
                projection_head_loaded = ProjectionHead(input_dim=LLM_INPUT_DIM, output_dim=projection_output_dim)
                projection_head_loaded.load_state_dict(checkpoint['projection_head_state_dict'])
                projection_head_loaded = projection_head_loaded.to(final_model.device).to(model_dtype).eval()
                print("    -> Camada ProjectionHead (Antiga) carregada com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar camadas Connector/Head (Antigo): {e}", exc_info=True)
            final_model = base_model
            new_connector_loaded = None
            projection_head_loaded = None

    # VVV --- [LÓGICA MODIFICADA] --- VVV
    elif is_rl_head_checkpoint:
        print(f"--- 2. Carregando Camada RL-Head (Aluno) de: {os.path.basename(rl_head_ckpt_path)} ---")
        try:
            final_model = base_model
            final_model.eval() 
            
            # [MODIFICADO] Usa o caminho do arquivo encontrado
            checkpoint = torch.load(rl_head_ckpt_path, map_location='cpu')

            LLM_INPUT_DIM = 1536 

            projection_head_loaded = ProjectionHead(input_dim=LLM_INPUT_DIM, output_dim=projection_output_dim)
            projection_head_loaded.load_state_dict(checkpoint)
            
            projection_head_loaded = projection_head_loaded.to(final_model.device).to(model_dtype).eval()
            print("    -> Camada RL-Head (Aluno) carregada com sucesso.")

        except Exception as e:
            logger.error(f"Erro ao carregar camada RL-Head: {e}", exc_info=True)
            projection_head_loaded = None
    # ^^^ --- [FIM DA LÓGICA MODIFICADA] --- ^^^
            
    else:
        final_model = base_model

    # --- 3. Carregar Processor e Tokenizer ---
    # (Não muda)
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        print("✅ Processador e Tokenizer carregados.")
    except Exception as e:
        logger.error(f"Falha ao carregar processor/tokenizer de {model_path}: {e}", exc_info=True)
        raise e

    print("✅ Carregamento concluído.")
    return final_model, processor, tokenizer, new_connector_loaded, projection_head_loaded

def get_lora_target_modules_vision_encoder(model_to_inspect, num_layers_to_train: int) -> List[str]:
    """
    Inspeciona um modelo e retorna os nomes dos submódulos Linear (q_proj, v_proj, fc1, fc2)
    das 'N' últimas camadas do encoder de visão.

    Args:
        model_to_inspect: O modelo a ser analisado.
        num_layers_to_train (int): O número de BLOCOS do encoder a treinar, a partir do final.
                                   Use -1 para treinar todas as camadas lineares do encoder.

    Returns:
        list: Uma lista de strings com os nomes das subcamadas lineares a serem treinadas.
    """
    target_submodules = ["qkv", "proj", "fc1", "fc2"] # Nomes comuns dentro dos blocos ViT/InternVL
    # Para InternVL pode ser qkv ou q_proj/v_proj/k_proj separadamente, e proj ou o_proj
    # E mlp.fc1, mlp.fc2
    
    layer_names_to_train = []
    
    try:
        # Tenta encontrar o encoder de visão (ajuste o caminho se necessário para InternVL)
        # O caminho correto pode ser model_to_inspect.vision_tower.vision_model.encoder.layers
        if hasattr(model_to_inspect, 'vision_tower') and \
           hasattr(model_to_inspect.vision_tower, 'vision_model') and \
           hasattr(model_to_inspect.vision_tower.vision_model, 'encoder') and \
           hasattr(model_to_inspect.vision_tower.vision_model.encoder, 'layers'):
           
            vision_encoder_layers = model_to_inspect.vision_tower.vision_model.encoder.layers
            base_path = "vision_tower.vision_model.encoder.layers"
        elif hasattr(model_to_inspect, 'vision_model') and \
             hasattr(model_to_inspect.vision_model, 'encoder') and \
             hasattr(model_to_inspect.vision_model.encoder, 'layers'):
             
            vision_encoder_layers = model_to_inspect.vision_model.encoder.layers
            base_path = "vision_model.encoder.layers"
        else:
             raise AttributeError("Estrutura do vision encoder não encontrada como esperado.")

        total_layers = len(vision_encoder_layers)
        
        print("="*60)
        print("🔎 Análise da Arquitetura do Modelo para Fine-Tuning")
        print("="*60)
        print(f"   - Total de BLOCOS no vision encoder: {total_layers}")

        if num_layers_to_train == 0:
            print("   -> Nenhuma camada do vision encoder será treinada (num_layers_to_train=0).")
            return []
            
        if num_layers_to_train == -1 or num_layers_to_train > total_layers:
            num_to_consider = total_layers
            print(f"   -> Alvo: TODAS as camadas lineares suportadas nos {total_layers} blocos do vision encoder.")
        else:
            num_to_consider = num_layers_to_train
            print(f"   -> Alvo: Camadas lineares suportadas nos últimos {num_to_consider} blocos do vision encoder.")

        # Calcula o índice do primeiro bloco a ser considerado
        start_index = total_layers - num_to_consider
        
        # Itera sobre os blocos selecionados e encontra os submódulos lineares
        for i in range(start_index, total_layers):
            layer_prefix = f"{base_path}.{i}"
            # Itera sobre os submódulos dentro do bloco atual
            for sub_name, sub_module in vision_encoder_layers[i].named_modules():
                # Verifica se o submódulo é uma camada linear (ou Linear4bit)
                # E se o nome do submódulo contém um dos alvos (qkv, proj, fc1, fc2)
                # Adicione 'Linear4bit' se estiver usando bitsandbytes
                if isinstance(sub_module, (torch.nn.Linear, torch.nn.modules.linear.Linear)) or "Linear4bit" in str(type(sub_module)):
                    if any(target in sub_name for target in target_submodules):
                         # Constrói o nome completo do submódulo
                         full_sub_name = f"{layer_prefix}.{sub_name}"
                         layer_names_to_train.append(full_sub_name)
        
        if layer_names_to_train:
            print("   - Submódulos lineares alvo encontrados para LoRA:")
            for name in layer_names_to_train[:5]: # Mostra apenas os 5 primeiros
                print(f"     - {name}")
            if len(layer_names_to_train) > 5:
                print(f"     - ... e mais {len(layer_names_to_train) - 5} outros.")
        else:
             print("   -> AVISO: Nenhum submódulo linear suportado foi encontrado nos blocos selecionados.")
            
        return layer_names_to_train

    except AttributeError as e:
        print(f"❌ ERRO: Não foi possível encontrar a estrutura esperada do vision encoder: {e}")
        return []
    

def get_full_finetune_target_blocks_vision_encoder(model_to_inspect, num_layers_to_train: int) -> List[str]:
    """
    Inspeciona um modelo e retorna os nomes *completos* dos BLOCOS (InternVisionEncoderLayer)
    do vision encoder que devem ser treinados diretamente (sem LoRA).

    Args:
        model_to_inspect: O modelo a ser analisado (pode ser o modelo base carregado).
        num_layers_to_train (int): O número de BLOCOS do encoder de visão a serem considerados,
                                   a partir do final, para o fine-tuning direto.
                                   Use -1 para considerar todos os blocos do encoder.

    Returns:
        list: Uma lista de strings com os nomes *completos* dos BLOCOS
              (ex: 'vision_model.encoder.layers.22') a serem treinados.
    """
    layer_names_to_train = []
    
    try:
        vision_encoder_layers = None
        base_path = ""

        if hasattr(model_to_inspect, 'vision_model') and \
           hasattr(model_to_inspect.vision_model, 'encoder') and \
           hasattr(model_to_inspect.vision_model.encoder, 'layers'):
           
            vision_encoder_layers = model_to_inspect.vision_model.encoder.layers
            base_path = "vision_model.encoder.layers"
        elif hasattr(model_to_inspect, 'base_model') and \
             hasattr(model_to_inspect.base_model, 'vision_model') and \
             hasattr(model_to_inspect.base_model.vision_model, 'encoder') and \
             hasattr(model_to_inspect.base_model.vision_model.encoder, 'layers'):
            
            vision_encoder_layers = model_to_inspect.base_model.vision_model.encoder.layers
            base_path = "base_model.vision_model.encoder.layers"
        else:
            raise AttributeError("Não foi possível encontrar a estrutura do vision encoder no modelo fornecido.")

        total_layers = len(vision_encoder_layers)
        
        print("="*60)
        print("🔎 Análise da Arquitetura do Modelo para Fine-Tuning Direto (Blocos de Camadas)")
        print("="*60)
        print(f"   - Total de BLOCOS (InternVisionEncoderLayer) no vision encoder: {total_layers}")

        if num_layers_to_train == 0:
            print("   -> Nenhuma camada do vision encoder será treinada (num_layers_to_train=0).")
            return []
            
        if num_layers_to_train == -1 or num_layers_to_train > total_layers:
            num_to_consider = total_layers
            print(f"   -> Alvo: TODOS os {total_layers} blocos do vision encoder para fine-tuning.")
        else:
            num_to_consider = num_layers_to_train
            print(f"   -> Alvo: Os últimos {num_to_consider} blocos do vision encoder para fine-tuning.")

        start_index = total_layers - num_to_consider
        
        for i in range(start_index, total_layers):
            block_name = f"{base_path}.{i}"
            layer_names_to_train.append(block_name)
        
        if layer_names_to_train:
            print(f"   - Total de {len(layer_names_to_train)} BLOCOS identificados para treinamento:")
            for name in layer_names_to_train[:5]: 
                print(f"     - {name}")
            if len(layer_names_to_train) > 5:
                print(f"     - ... e mais {len(layer_names_to_train) - 5} outros.")
        else:
             print("   -> AVISO: Nenhum bloco do vision encoder foi encontrado para treinamento com as configurações selecionadas.")
            
        return layer_names_to_train

    except AttributeError as e:
        print(f"❌ ERRO: Problema ao inspecionar a arquitetura do vision encoder: {e}")
        return []
    
def load_full_fine_tuned_model(
    model_name: str,
    full_fine_tune_checkpoint_path: str,
    projection_output_dim: int,
    device: str
) -> Tuple[AutoModelForCausalLM, AutoProcessor, AutoTokenizer, ProjectionHead]:
    """
    Carrega um modelo InternVL que foi fine-tuned diretamente nos pesos
    (Full Fine-Tuning) e seu ProjectionHead associado.

    Args:
        model_name (str): O nome do modelo base (ex: 'InternVL3-2B').
        full_fine_tune_checkpoint_path (str): Caminho completo para o arquivo
                                              'training_checkpoint.pt' do checkpoint de full fine-tune.
        projection_output_dim (int): Dimensão de saída do ProjectionHead.
        device (str): Dispositivo para carregar o modelo ('cuda' ou 'cpu').

    Returns:
        tuple: Uma tupla contendo o (modelo, processador, tokenizer, projection_head)
               prontos para uso.
    Raises:
        ValueError: Se full_fine_tune_checkpoint_path ou projection_output_dim não forem válidos.
        FileNotFoundError: Se o arquivo de checkpoint não for encontrado.
    """
    model_path = f"OpenGVLab/{model_name}"
    logger.info(f"--- 1. Carregando Modelo Base para Full Fine-Tuning: {model_path} ---")

    if not full_fine_tune_checkpoint_path or not os.path.exists(full_fine_tune_checkpoint_path):
        raise FileNotFoundError(f"Arquivo de checkpoint de Full Fine-Tuning não encontrado ou inválido: '{full_fine_tune_checkpoint_path}'")
    
    if projection_output_dim is None:
        raise ValueError("projection_output_dim é obrigatório para carregar um checkpoint de Full Fine-Tuning.")

    # Carregar o modelo base em bfloat16 para aplicar os pesos do checkpoint
    model_dtype = torch.bfloat16
    logger.info(f"  -> Carregando modelo base em bfloat16 de precisão total para aplicar pesos fine-tuned.")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=model_dtype,
        device_map="auto",
        quantization_config=None # Sem quantização para full fine-tune
    ).to(device) # Mover para o dispositivo especificado

    logger.info(f"--- 2. Aplicando pesos de Full Fine-Tuning de: {full_fine_tune_checkpoint_path} ---")
    try:
        checkpoint = torch.load(full_fine_tune_checkpoint_path, map_location=device)
        
        # Aplica o state_dict do modelo (partes treinadas do Vision Encoder)
        base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("  -> Pesos do Vision Encoder atualizados com sucesso.")
        
        # Instancia e carrega o ProjectionHead
        projection_head = ProjectionHead(input_dim=base_model.config.hidden_size,
                                         output_dim=projection_output_dim).to(device)
        projection_head = projection_head.to(model_dtype) # Garante que o dtype seja consistente
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        logger.info(f"  -> ProjectionHead carregado com sucesso (Input: {base_model.config.hidden_size}, Output: {projection_output_dim}).")
        
        projection_head.eval() # Colocar o head em modo de avaliação
    except Exception as e:
        logger.error(f"❌ Erro ao carregar Full Fine-Tuning checkpoint: {e}")
        raise # Levantar o erro para indicar falha no carregamento do FT
    
    # Carrega o processor e o tokenizer, que são sempre do modelo base
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # Definir pad_token para o tokenizer se não estiver definido
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model.eval() # Definir para modo de avaliação por padrão
    logger.info("✅ Modelo Full Fine-Tuned carregado e pronto para inferência.")
    
    return base_model, processor, tokenizer, projection_head