import os
import csv
import random
import itertools

# --- DEFINIÇÃO DOS SPLITS ZERO-SHOT (Baseado na Imagem) ---
ALL_16_CLASSES = [
    'letter', 'form', 'email', 'handwritten', 'advertisement', 
    'scientific_report', 'scientific_publication', 'specification', 'resume', 
    'file_folder', 'news_article', 'budget', 'invoice', 'memo', 
    'presentation', 'questionnaire'
]

SPLITS_DEFINITION = {
    'A_unseen': ['email', 'form', 'handwritten', 'letter'],
    'B_unseen': ['advertisement', 'scientific_publication', 'scientific_report', 'specification'],
    'C_unseen': ['budget', 'file_folder', 'invoice', 'news_article'],
    'D_unseen': ['memo', 'presentation', 'questionnaire', 'resume']
}

SPLITS_DEFINITION['A_seen'] = [cls for cls in ALL_16_CLASSES if cls not in SPLITS_DEFINITION['A_unseen']]
SPLITS_DEFINITION['B_seen'] = [cls for cls in ALL_16_CLASSES if cls not in SPLITS_DEFINITION['B_unseen']]
SPLITS_DEFINITION['C_seen'] = [cls for cls in ALL_16_CLASSES if cls not in SPLITS_DEFINITION['C_unseen']]
SPLITS_DEFINITION['D_seen'] = [cls for cls in ALL_16_CLASSES if cls not in SPLITS_DEFINITION['D_unseen']]


def create_rvl_cdip_pairs(base_dataset_path, subset_folder, output_csv, num_comparisons, allowed_classes=None, seed=None):
    """
    Gera um arquivo CSV com uma AMOSTRA ALEATÓRIA de pares de imagens
    para um subconjunto do dataset RVL-CDIP, com distribuição uniforme,
    USANDO APENAS AS CLASSES PERMITIDAS.

    Args:
        base_dataset_path (str): O caminho para o diretório raiz do dataset.
        subset_folder (str): O nome da subpasta ('train' ou 'val').
        output_csv (str): O nome do arquivo CSV de saída.
        num_comparisons (int): O número total de comparações a serem amostradas.
        allowed_classes (list, optional): Lista de nomes de classes para incluir.
                                          Se None, usa todas as classes.
        seed (int, optional): Semente para o gerador de números aleatórios.
    """
    
    # --- ★ MUDANÇA (1): Configura a semente no início da função ---
    if seed is not None:
        print(f"Usando semente de aleatoriedade: {seed}")
        random.seed(seed)
    
    full_subset_path = os.path.join(base_dataset_path, subset_folder)

    classes_found_all = [d for d in os.listdir(full_subset_path) if os.path.isdir(os.path.join(full_subset_path, d))]
    
    if not classes_found_all:
        print(f"Nenhuma subpasta de classe encontrada em {full_subset_path}. Verifique o caminho.")
        return

    if allowed_classes:
        classes_to_process = [cls for cls in classes_found_all if cls in allowed_classes]
    else:
        print("Aviso: 'allowed_classes' não foi definido. Usando todas as classes encontradas (não-zero-shot).")
        classes_to_process = classes_found_all
        
    if not classes_to_process:
        print(f"Nenhuma classe válida (de 'allowed_classes') encontrada em {full_subset_path}.")
        return

    print(f"Mapeando arquivos de imagem para o conjunto '{subset_folder}'...")
    print(f"Classes a processar: {classes_to_process}")
    
    image_files_by_class = {cls: [] for cls in classes_to_process}

    for cls in classes_to_process:
        class_path = os.path.join(full_subset_path, cls)
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_files_by_class[cls].append(filename)
    
    image_files_by_class = {k: v for k, v in image_files_by_class.items() if v}
    classes = list(image_files_by_class.keys())
    
    if not classes:
        print(f"Nenhuma imagem encontrada para as classes permitidas em '{subset_folder}'.")
        return
        
    print(f"Encontradas {len(classes)} classes com imagens no conjunto '{subset_folder}'.")
    
    all_positive_pairs = []
    all_negative_pairs = []

    print("Gerando lista de todos os pares positivos possíveis...")
    for class_name, files in image_files_by_class.items():
        if len(files) >= 2:
            for file_a, file_b in itertools.combinations(files, 2):
                all_positive_pairs.append({
                    'file_a_name': file_a,
                    'file_b_name': file_b,
                    'class_a_name': class_name,
                    'class_b_name': class_name,
                    'file_a_path': os.path.join(subset_folder, class_name, file_a),
                    'file_a_path': os.path.join(subset_folder, class_name, file_a),
                    'file_b_path': os.path.join(subset_folder, class_name, file_b),
                    'is_equal': 1
                })

    print("Gerando lista de todos os pares negativos possíveis...")
    for class_a_name, class_b_name in itertools.combinations(classes, 2):
        files_a = image_files_by_class[class_a_name]
        files_b = image_files_by_class[class_b_name]
        for file_a, file_b in itertools.product(files_a, files_b):
            all_negative_pairs.append({
                'file_a_name': file_a,
                'file_b_name': file_b,
                'class_a_name': class_a_name,
                'class_b_name': class_b_name,
                'file_a_path': os.path.join(subset_folder, class_a_name, file_a),
                'file_b_path': os.path.join(subset_folder, class_b_name, file_b),
                'is_equal': 0
            })

    num_positive_target = num_comparisons // 2
    num_negative_target = num_comparisons - num_positive_target

    print(f"Total de pares positivos possíveis: {len(all_positive_pairs)}")
    print(f"Total de pares negativos possíveis: {len(all_negative_pairs)}")
    
    # O random.seed() no início da função garante que estes shuffles sejam reprodutíveis
    random.shuffle(all_positive_pairs)
    random.shuffle(all_negative_pairs)
    
    sampled_positives = all_positive_pairs[:min(num_positive_target, len(all_positive_pairs))]
    sampled_negatives = all_negative_pairs[:min(num_negative_target, len(all_negative_pairs))]
    
    final_pairs = sampled_positives + sampled_negatives
    random.shuffle(final_pairs) # Embaralha a lista final

    print(f"Total de {len(final_pairs)} pares amostrados para '{subset_folder}'. Escrevendo no arquivo CSV...")

    headers = [
        'file_a_name', 'file_b_name', 'class_a_name', 'class_b_name',
        'file_a_path', 'file_b_path', 'is_equal'
    ]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(final_pairs)
        
    print(f"Arquivo CSV '{output_csv}' criado com sucesso!")


if __name__ == '__main__':
    # --- CONFIGURAÇÃO ---
    base_dataset_location = '/mnt/data/rvl-cdip-small-200' 
    
    # --- ★ MUDANÇA (2): Define uma semente global para seus experimentos ---
    GLOBAL_SEED = 42
    
    # --- EXEMPLO: Gerando os arquivos para o SPLIT A ---
    
    # 1. Gerar para o conjunto de TREINO (SPLIT A - SEEN)
    print("\n--- Gerando CSV para o SPLIT A (TREINO - SEEN) ---")
    create_rvl_cdip_pairs(
        base_dataset_path=base_dataset_location,
        subset_folder='train',
        output_csv='../../data/RVL-CDIP/train_pairs.csv',
        num_comparisons=9000,
        allowed_classes=SPLITS_DEFINITION['A_seen'],
        seed=GLOBAL_SEED  # <-- ★ MUDANÇA (3): Passa a semente
    )
    
    # 2. Gerar para o conjunto de VALIDAÇÃO (SPLIT A - UNSEEN)
    print("\n--- Gerando CSV para o SPLIT A (VALIDAÇÃO - UNSEEN) ---")
    create_rvl_cdip_pairs(
        base_dataset_path=base_dataset_location,
        subset_folder='val',
        output_csv='../../data/RVL-CDIP/validation_pairs.csv',
        num_comparisons=1000,
        allowed_classes=SPLITS_DEFINITION['A_unseen'],
        seed=GLOBAL_SEED  # <-- ★ MUDANÇA (4): Passa a semente
    )