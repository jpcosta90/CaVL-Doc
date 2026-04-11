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

## 🚀 Research Construction Flow (Documented Procedure)

The repository now follows the research procedure that was actually executed in this project, in chronological order.

| Phase | What was done | Main artifacts | Current status |
| :--- | :--- | :--- | :--- |
| **1** | Experimental protocol definition (datasets, splits, metrics, seeds) | Split generation + protocol docs | ✅ Done |
| **2** | Base training/evaluation pipeline stabilization | Core training scripts + checkpoints + W&B tracking | ✅ Done |
| **3** | **LA-CDIP Coarse Sweep** (broad hyperparameter exploration) | `analysis/sweep_report/dataframes/lacdip_coarse_all.csv` | ✅ Done |
| **3B** | **RVL-CDIP Sweep cycle (parallel)** coarse + fine | `analysis/sweep_report/dataframes/rvlcdip_coarse_all.csv`, `analysis/sweep_report/dataframes/rvlcdip_fine_all.csv` | ✅ Done (low conversion to final gains) |
| **4** | **LA-CDIP Fine Search** (high-fidelity refinement) | `scripts/optimization/coarse_search/configs/lacdip/fine_search/runs_raw.csv` | ✅ Done |
| **5** | **Sprint 1 (LA-CDIP):** compare best config of each loss | W&B project `CaVL-Doc_LA-CDIP_Sprint1_Top5Validation` | ✅ Done |
| **6** | **Sprint 2:** small teacher-network sweep on best losses + contrastive | Teacher sweep runs (time-bounded) | ⬜ Planned |
| **7** | **Sprint 3:** ablation + final LA-CDIP results | `teacher off/on × 5 splits × 2 losses` | ⬜ Planned |
| **8** | **Sprint 4:** transfer learning to RVL-CDIP + batch-size sweep | `with/without transfer × 5 batch sizes`, 5 epochs | ⬜ Planned |
| **9** | **Sprint 5:** final zero-shot RVL-CDIP results | `1 final config × 5 splits` | ⬜ Planned |
| **10** | **Sprint 6:** final write-up and submission closure | Final tables, plots, discussion, conclusion | ⬜ Planned |

### Phase Dependencies (Operational)

1. Coarse Sweep (LA-CDIP)  
2. Fine Search (LA-CDIP)  
3. Sprint 1: best-loss comparison (LA-CDIP)  
4. Sprint 2: teacher sweep (small network, time-bounded)  
5. Sprint 3: teacher ablation + final LA-CDIP  
6. Sprint 4: transfer + batch sweep (RVL zero-shot)  
7. Sprint 5: final RVL splits  
8. Sprint 6: submission closure

For the detailed timeline (start/end dates and runtime accounting), see `docs/cronograma_fases_pesquisa.md`.

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

Master results and performance benchmarks are tracked in the `results/` directory.

Current evidence sources:
* **LA-CDIP Sweeps**: coarse/fine artifacts in `analysis/sweep_report/dataframes/`.
* **Sprint 1 Comparison**: loss-level comparison artifacts in `analysis/sprint1_report/`.
* **RVL-CDIP Sweeps**: available and fully logged in W&B/local artifacts; useful as negative/diagnostic evidence even when conversion to final gains is limited.
* **Plots**: available in `results/plots/`.

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