# Figure Generation Scripts

Scripts to regenerate the paper figures saved to `docs/assets/`.

| Script | Output | Paper Figure |
|--------|--------|--------------|
| `generate_ablation_loss.py`    | `ablation_loss.pdf/png`    | Fig. 4 — Loss ablation on LA-CDIP |
| `generate_comparison_lacdip.py`| `comparison_lacdip.pdf/png`| Fig. 5 — Method comparison bar chart |
| `generate_per_split_eer.py`    | `per_split_eer.pdf/png`    | Fig. 6 — Per-split EER line chart |

Figures 1–3 (teaser, architecture, metric space) are in `analysis/paper_figures.ipynb`.

Run from repo root with the project venv active:

```bash
source .venv/bin/activate
python scripts/figures/generate_ablation_loss.py
python scripts/figures/generate_comparison_lacdip.py
python scripts/figures/generate_per_split_eer.py
```
