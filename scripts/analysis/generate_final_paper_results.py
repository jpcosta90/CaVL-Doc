#!/usr/bin/env python3
"""
Gera paper_results_final.html — versão limpa para o paper.

Seções:
  1. Sprint3b — ArcDoc (UnB): ablação de losses (treino, subset de validação)
  2. Full Eval — Sprint3b: mesmas métricas nos CSVs completos (comparável com baselines)
  3. Baselines — Embedding (incluindo Jina-v4 finetuned)
  4. Baselines — VLM (métrica numérica)

Uso:
    python scripts/analysis/generate_final_paper_results.py
    python scripts/analysis/generate_final_paper_results.py --output results/paper_results_final.html
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

# Reutiliza tudo do script principal
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from generate_paper_results_html import (
    ENTITY, PROJECTS, CSS,
    _fetch_runs, _build_exp3, _build_full_eval,
    _build_baselines_embedding, _build_baselines_vlm,
    _table_html, _chart_html,
    _fmt_eer,
)

WORKSPACE_ROOT  = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT  = WORKSPACE_ROOT / "results" / "paper_results_final.html"


def build_final_html(
    exp3b_t1, exp3b_t2, exp3b_t3, exp3b_c1, exp3b_c2, exp3b_c3,
    exp3b_partial, exp3b_missing,
    full_eval_t1, full_eval_t2, full_eval_t3,
    full_eval_c1, full_eval_c2, full_eval_c3,
    full_eval_partial, full_eval_missing,
    emb_cv, emb_chart_cv,
    vlm_cv, vlm_chart_cv,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def partial_badge(is_p):
        return '<span class="partial-badge">⚠ Parcial</span>' if is_p else ""

    def missing_note(m):
        return f'<p class="missing-note">Splits pendentes: {m}</p>' if m else ""

    sections = []

    # --- Sprint3b (treino, subset) ---
    sections.append(f"""
<div class="section">
  <h2>1. Sprint3b — ArcDoc (UnB) {partial_badge(exp3b_partial)}</h2>
  <p class="desc">
    Treinamento em dois estágios com professor RL ativo. Loss Sub-Center (s=32, k=3),
    inicialização aleatória (source-init-mode=none). EER reportado sobre subset de validação
    balanceado (≈557 pares, 24 classes unseen por split).
  </p>
  {missing_note(exp3b_missing)}
  <h3>Estágio 1 — Pré-treinamento (10 épocas)</h3>
  {_chart_html(exp3b_c1)}
  {_table_html(exp3b_t1)}
  <h3>Estágio 2 — Efeito do Professor</h3>
  {_chart_html(exp3b_c2)}
  {_table_html(exp3b_t2)}
  <h3>Melhor EER Acumulado (Estágio 1 + 2)</h3>
  {_chart_html(exp3b_c3)}
  {_table_html(exp3b_t3)}
</div>""")

    # --- Full Eval ---
    sections.append(f"""
<div class="section">
  <h2>2. Avaliação Completa — Sprint3b (Full Eval) {partial_badge(full_eval_partial)}</h2>
  <p class="desc">
    Mesmas métricas do Sprint3b mas calculadas nos CSVs de validação completos
    (sem subset). Números diretamente comparáveis com os baselines de embedding e VLM.
  </p>
  {missing_note(full_eval_missing)}
  <h3>Estágio 1 — Fase 1 (pares completos)</h3>
  {_chart_html(full_eval_c1)}
  {_table_html(full_eval_t1)}
  <h3>Estágio 2 — Efeito do Professor (pares completos)</h3>
  {_chart_html(full_eval_c2)}
  {_table_html(full_eval_t2)}
  <h3>Melhor EER Acumulado — Full Eval</h3>
  {_chart_html(full_eval_c3)}
  {_table_html(full_eval_t3)}
</div>""")

    # --- Baselines Embedding ---
    sections.append(f"""
<div class="section">
  <h2>3. Baselines — Similaridade por Embedding</h2>
  <p class="desc">
    Pixel bruto, Jina-v4 (unadapted), InternVL3-2B e Jina-v4 finetuned (LoRA r=48, InfoNCE).
    Validação cruzada splits 0–4.
  </p>
  {_chart_html(emb_chart_cv)}
  {_table_html(emb_cv)}
</div>""")

    # --- Baselines VLM ---
    sections.append(f"""
<div class="section">
  <h2>4. Baselines — VLM com Métrica Numérica</h2>
  <p class="desc">
    Modelos VLM solicitados a retornar similaridade 0–100. Splits 0–4.
  </p>
  {_chart_html(vlm_chart_cv)}
  {_table_html(vlm_cv)}
</div>""")

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <title>ArcDoc — Resultados para o Paper</title>
  <style>{CSS}</style>
</head>
<body>
  <h1>ArcDoc — Resultados para o Paper</h1>
  <p class="subtitle">Gerado em {now} · W&B: {ENTITY}</p>
  {''.join(sections)}
  <footer>generate_final_paper_results.py · CaVL-Doc</footer>
</body>
</html>"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Buscando runs no W&B...")
    runs3b         = _fetch_runs(PROJECTS["exp3b"])
    runs_full_eval = _fetch_runs(PROJECTS["full_eval"])
    runs_emb       = _fetch_runs(PROJECTS["emb_baseline"])
    runs_jina_lora = _fetch_runs(PROJECTS["jina_lora"])
    runs_vlm       = _fetch_runs(PROJECTS["vlm_metric"])

    print("Construindo seções...")
    exp3b_t1, exp3b_t2, exp3b_t3, exp3b_c1, exp3b_c2, exp3b_c3, exp3b_partial, exp3b_missing = \
        _build_exp3(runs3b, run_prefix="Sprint3b_")
    full_eval_t1, full_eval_t2, full_eval_t3, full_eval_c1, full_eval_c2, full_eval_c3, full_eval_partial, full_eval_missing = \
        _build_full_eval(runs_full_eval)
    emb_cv, _, emb_chart_cv, _ = _build_baselines_embedding(runs_emb, extra_runs=runs_jina_lora)
    vlm_cv, _, vlm_chart_cv, _ = _build_baselines_vlm(runs_vlm, "Baselines VLM")

    html = build_final_html(
        exp3b_t1, exp3b_t2, exp3b_t3, exp3b_c1, exp3b_c2, exp3b_c3,
        exp3b_partial, exp3b_missing,
        full_eval_t1, full_eval_t2, full_eval_t3,
        full_eval_c1, full_eval_c2, full_eval_c3,
        full_eval_partial, full_eval_missing,
        emb_cv, emb_chart_cv,
        vlm_cv, vlm_chart_cv,
    )

    output.write_text(html, encoding="utf-8")
    print(f"\n✅ Relatório salvo em: {output}")


if __name__ == "__main__":
    main()
