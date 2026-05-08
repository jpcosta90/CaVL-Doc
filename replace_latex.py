import re

with open('docs/paper/sn-article.tex', 'r', encoding='utf-8') as f:
    text = f.read()

new_section = r"""\section{Results and Discussion}\label{sec:results}

In this section, we present the empirical evaluation of the CaVL-Doc framework. The experimental results validate our architectural choices, analyze the impact of the active hard-negative mining (Teacher agent), and benchmark our method against several embedding and Visual Language Model (VLM) baselines. Following our evaluation protocol, the results are reported as the average Equal Error Rate (EER) of splits 0 to 4 (cross-validation), excluding the reserved split 5 unless otherwise specified.

\subsection{Hyperparameter Optimization}
The hyperparameter search was conducted in two stages on the LA-CDIP dataset. The initial \textbf{Coarse Search} explored a wide configuration space utilizing 50 steps per epoch, where Triplet and Circle loss yielded the lowest initial errors. This was followed by a \textbf{Fine Search} (100 steps per epoch) refining the top candidates. After the fine search, Sub-Center ArcFace and Sub-Center CosFace emerged as the tightest bounds, both achieving a hyperparameter search EER of 0.51\%. These optimized configurations were employed for the controlled experiments in the subsequent phases.

\subsection{Ablation Studies}

\subsubsection{Objective Function Analysis}
In the first phase, we evaluated the performance of various metric learning objective functions under controlled conditions (10 epochs, 140 steps per epoch) without active hard-negative mining. 

\begin{table}[htbp]
    \centering
    \caption{Phase 1: Comparison of Objective Functions on LA-CDIP without hard-negative mining. Lower EER is better.}
    \label{tab:loss-functions}
    \begin{tabular}{lc}
    \toprule
    \textbf{Loss Function} & \textbf{Avg. EER (\%)} \\
    \midrule
    \textbf{Sub-Center CosFace} & \textbf{1.83} \\
    Sub-Center ArcFace & 2.03 \\
    Triplet & 2.70 \\
    Circle & 3.28 \\
    Contrastive & 4.25 \\
    \bottomrule
    \end{tabular}
\end{table}

As shown in Table \ref{tab:loss-functions}, Sub-Center CosFace (1.83\%) and Sub-Center ArcFace (2.03\%) achieved the strongest results in the initial phase, demonstrating that angular margin-based losses perform favorably on isolating layout identities compared to simple distance-based formulation like Triplet or Contrastive.

\subsubsection{Impact of the Teacher Agent (Hard Negative Mining)}
We investigated the impact of the RL-based Teacher agent, which actively mines hard negatives. We compared models reaching the second stage (Teacher ON) versus standard ongoing training (Teacher OFF).

\begin{table}[htbp]
    \centering
    \caption{Phase 2: Impact of the Teacher Agent (Hard Negative Mining) on LA-CDIP. The EER (\%) is compared between runs with the Teacher ON (Com Min.) and Teacher OFF (Sem Min.).}
    \label{tab:teacher-ablation}
    \begin{tabular}{lccc}
    \toprule
    \multirow{2}{*}{\textbf{Loss Function}} & \multicolumn{2}{c}{\textbf{Avg. EER (\%)}} & \multirow{2}{*}{\textbf{$\Delta$ (pp)}} \\
    \cmidrule{2-3}
     & \textbf{Teacher ON} & \textbf{Teacher OFF} & \\
    \midrule
    Sub-Center ArcFace & 1.09 & \textbf{0.95} & -0.14 \\
    Sub-Center CosFace & 1.37 & \textbf{1.36} & -0.01 \\
    Triplet & \textbf{1.45} & 1.63 & +0.18 \\
    Contrastive & \textbf{2.20} & 2.34 & +0.14 \\
    \bottomrule
    \end{tabular}
\end{table}

The outcomes depicted in Table \ref{tab:teacher-ablation} showcase a dichotomous behavior. The active hard mining successfully supported the distance-based losses (Triplet and Contrastive), enhancing their discriminative capacity by +0.18 pp and +0.14 pp, respectively. Conversely, inserting actively selected hard negatives harmed the geometry of angular margin-based losses (Sub-Center ArcFace and CosFace). The global optimum across all stages was achieved by Sub-Center ArcFace trained without the Teacher agent (0.82\% cumulative EER), serving as our primary configuration going forward.

\subsection{Generalization and Domain Transfer}
To test the transferability of the learned embeddings, we validated the model trained on LA-CDIP on the visually distinct RVL-CDIP dataset.

\begin{table}[htbp]
    \centering
    \caption{Domain Transfer (LA-CDIP $\rightarrow$ RVL-CDIP). Compares Transfer EER against Direct Training EER on the RVL-CDIP target splits.}
    \label{tab:domain-transfer}
    \begin{tabular}{lccc}
    \toprule
    \textbf{Loss Function} & \textbf{Transfer EER (\%)} & \textbf{Direct EER (\%)} & \textbf{$\Delta$ Transfer (pp)} \\
    \midrule
    Average Sub-Center ArcFace & 25.56 & 23.89 & -1.67 \\
    Average Sub-Center CosFace & 25.54 & 24.35 & -1.19 \\
    Average Triplet & 26.43 & 25.27 & -1.16 \\
    Average Contrastive & 27.62 & 27.74 & +0.12 \\
    \bottomrule
    \end{tabular}
\end{table}

Noticeably, directly training on the RVL-CDIP domain typically outperformed transferring weights from LA-CDIP (negative $\Delta$), reflecting the substantial semantic shift between layout-driven features (LA-CDIP) and content-driven features (RVL-CDIP).

\subsection{Comparison with Baselines and SOTA}
Finally, we compared our best configurations against unadapted embedding baselines and zero-shot VLM models outputting scalar similarities.

\begin{table}[htbp]
    \centering
    \caption{Embedding Similarity Baselines (Average EER on splits 0-4). Compares raw and unadapted embeddings.}
    \label{tab:baselines-embedding}
    \begin{tabular}{lc}
    \toprule
    \textbf{Method} & \textbf{Avg. EER (\%)} \\
    \midrule
    Jina-v4 & 3.08 \\
    InternVL3-out & 4.78 \\
    InternVL3-in & 7.62 \\
    Pixel-L2 / Pixel-Cosine & 31.20 / 32.30 \\
    \bottomrule
    \end{tabular}
\end{table}

\begin{table}[htbp]
    \centering
    \caption{VLM Baseline Models scoring numerical metrics (Average EER on splits 0-4).}
    \label{tab:baselines-vlm}
    \begin{tabular}{lc}
    \toprule
    \textbf{Method} & \textbf{Avg. EER (\%)} \\
    \midrule
    \textbf{Ours (Sub-Center ArcFace Best)} & \textbf{0.82} \\
    Qwen-2.5-VL-8B & 1.12 \\
    InternVL-3-14B & 1.30 \\
    InternVL-3-8B & 2.13 \\
    Qwen-2.5-VL-4B & 2.67 \\
    InternVL-3-2B (Base) & 29.99 \\
    Gemma-4-E2B / E4B & $\sim$49.80 \\
    \bottomrule
    \end{tabular}
\end{table}

Our adapted model achieved an EER of 0.82\%, significantly outperforming its native unadapted base model (InternVL-3-2B series), which hovered at 29.99\% EER. Furthermore, our 2B parameter configuration managed to outperform models an order of magnitude larger operating in a zero-shot numerical capacity, such as Qwen-2.5-VL-8B (1.12\%) and InternVL-3-14B (1.30\%), affirming the efficiency of establishing a designated metric learning objective over raw VLM numerical predictions.
"""

start_idx = text.find(r"\section{Results and Discussion}")
if start_idx != -1:
    text = text[:start_idx] + new_section
    with open('docs/paper/sn-article.tex', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Replaced section successfully.")
else:
    print("Could not find section.")

