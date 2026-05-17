---
license: other
task_categories:
  - image-to-image
  - feature-extraction
language:
  - en
tags:
  - document-understanding
  - document-retrieval
  - metric-learning
  - zero-shot-learning
  - la-cdip
  - cavl-doc
size_categories:
  - 10K<n<100K
---

# CaVL-Doc — LA-CDIP Final Split 3 (Augmented)

Dataset de treino e validação para o modelo definitivo CaVL-Doc, baseado no
**Split 3** do protocolo ZSL (Zero-Shot Learning) sobre o LA-CDIP.

## Estrutura

| Conjunto | Imagens | Pares |
|---|---|---|
| Treino (`images_train/`) | 8,740 variantes augmentadas | 34.960 |
| Validação (`images_val/`) | 2,625 variantes augmentadas | 10.500 |

- **120 classes** para treino · **24 classes novel** para validação (sem sobreposição)
- Cada imagem original gera **5 variantes** com o pipeline de augmentação offline
- Augmentações: rotação ±15°, perspectiva 2–7%, brilho/contraste ±25%, blur,
  nitidez, ruído gaussiano, compressão JPEG, resolução baixa, crop/padding,
  amarelamento, salt & pepper

## Arquivos

```
train_pairs.csv        — pares de treino (file_a, file_b, is_equal, class)
validation_pairs.csv   — pares de validação ZSL (24 classes novel)
images_train/          — imagens augmentadas de treino organizadas por classe
images_val/            — imagens augmentadas de validação organizadas por classe
```

## Como baixar

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="Jpcosta90/cavl-doc-lacdip-split3",
    repo_type="dataset",
    local_dir="data/generated_splits/final_split3",
)
```

## Origem

Gerado a partir do LA-CDIP (Tobacco Document Library) com o script
`scripts/utils/prepare_arcdoc_training.py` do repositório CaVL-Doc.
