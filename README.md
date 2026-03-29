# CaVL-Doc: Comparative Aligned Vision-Language Document Embeddings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![InternVL3](https://img.shields.io/badge/Backbone-InternVL3--2B-blue)](https://github.com/OpenGVLab/InternVL)
[![Dataset](https://img.shields.io/badge/Research-Document--Comparison-orange)](#)

**CaVL-Doc** is a specialized framework for learning **Comparative Aligned Vision-Language Document Embeddings**. Unlike standard document VQA, CaVL-Doc optimizes Large Vision-Language Models (LVLMs) to produce unified, low-dimensional representations optimized for **similarity matching**, **retrieval**, and **zero-shot classification**.

---

## 🎯 Project Focus & Value Proposition

The core objective of CaVL-Doc is to transform raw document images into a high-quality **Metric Space** where:
*   **Semantic Similarity**: Visually different but semantically related documents (e.g., two different invoices) are pulled together.
*   **Discrimination**: Visually similar but unrelated documents are pushed apart.
*   **Zero-Shot capability**: Documents can be categorized without re-training, simply by comparing their embeddings to class prototypes.

---

## 🏗️ Architecture & Strategy

CaVL-Doc utilizes a **Hybrid Curriculum-Reinforcement Learning** strategy to fine-tune a projection head attached to a frozen LVLM backbone.

![Architecture Overview](docs/assets/architecture_with_teacher.png)

### 1. The Core Components
*   **Backbone (Frozen)**: InternVL3-2B provides basic multimodal tokenization.
*   **Student (CaVL Head)**: A trainable projection layer (Attention/MLP) that maps multimodal tokens into a metric embedding.
*   **Teacher (RL Policy)**: A reinforcement learning agent (PPO-based) that selects the most informative training pairs for the Student.

### 2. Multi-Phase Curriculum
The training progresses through three distinct optimization phases:
1.  **Phase 1 (Alignment)**: Geometric initialization using Contrastive Loss.
2.  **Phase 2 (Angular Margin)**: Refinement using ArcFace/ExpFace to enforce strict angular boundaries.
3.  **Phase 3 (Hard Adaptation)**: Final tuning with stochastic margins to handle edge cases.

---

## 🚀 Research & Optimization Pipeline

The project follows a rigorous, two-stage optimization process for model hyperparameter discovery.

| Stage | Name | Description | Tools |
| :--- | :--- | :--- | :--- |
| **Stage 1** | **Coarse Sweep** | Broad exploration of hyperparameters (LR, Margin, Scale) using WandB Bayesian search. | `scripts/optimization/coarse_search/` |
| **Stage 2** | **Fine Search** | Automated analysis of Stage 1 results, pruning inert parameters for high-fidelity refinement. | `scripts/optimization/fine_search/` |

---

## 📂 Repository Structure

The repository is organized following the functional phases of the research pipeline:

```
.
├── src/cavl_doc/             # Core Library: Models, Losses, Trainers
├── scripts/                  # Executable Pipelines
│   ├── optimization/         # Stage 1 (Coarse) and Stage 2 (Fine) Sweeps
│   ├── training/             # Main Curriculum-RL Training loops
│   ├── evaluation/           # Metrics (EER), benchmarks and visualization
│   └── utils/                # Data preparation (prepare_splits.py) and maintenance
├── data/                     # Data Manifest and local split pointers
├── analysis/                 # Research notebooks and detailed reports
├── results/                  # Final metrics, EER plots, and sweep summaries
└── docs/                     # Documentation assets and paper drafts
```

---

## 🛠️ Getting Started

### Installation
```bash
git clone https://github.com/jpcosta90/CaVL-Doc.git 
cd CaVL-Doc
pip install -e .
```

### Data Preparation
CaVL-Doc assumes datasets are stored in `/mnt/data/`. To prepare a specific split:
```bash
python scripts/utils/prepare_splits.py --data-root /mnt/data/la-cdip --protocol zsl --split-idx 1
```

### Running a Sweep
```bash
# Launch Stage 1 (Coarse)
wandb sweep scripts/optimization/coarse_search/sweep_config.yaml
```

---

## 📊 Results & Artifacts

## Results

### Performance vs. Parameters (LA-CDIP Dataset)

Master results and performance benchmarks are tracked in the `results/` directory. 
*   **EER Plots**: Available in `results/plots/`.
*   **Sweep Analysis**: Automated reports generated in `results/sweeps/`.

---

## 📜 Citation & License

If you use this work in your research, please cite our paper:

```bibtex
@article{Costa2025CaVLDOC,
  title   = {CaVL-Doc: Comparative Aligned Vision-Language Document Embeddings for Zero-Shot Classification},
  author  = {Joao Paulo Vieira Costa and Co-authors},
  journal = {Journal or Conference Name},
  year    = {2025},
  volume  = {XX},
  pages   = {XX--XX}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.