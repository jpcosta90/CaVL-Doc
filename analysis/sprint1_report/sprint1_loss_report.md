# Sprint 1 — Relatório de Comparação de Losses (LA-CDIP)

## Escopo
- Projeto W&B: `CaVL-Doc_LA-CDIP_Sprint1_Top5Validation`
- Entity: `jpcosta1990-university-of-brasilia`
- Runs coletados: `15` (com EER válido: `15`)

## Melhor com professor (últimas 5 épocas)
- Loss: `subcenter_cosface` | mean best EER: `0.016210` | runs: `3`

## Tabela — Loss x Variante
| loss_type         | professor_last5_label   |   runs |   unique_seeds |   best_eer_min |   best_eer_mean |   best_eer_std |   best_eer_median |   best_eer_max |
|:------------------|:------------------------|-------:|---------------:|---------------:|----------------:|---------------:|------------------:|---------------:|
| subcenter_cosface | on                      |      3 |              3 |       0.014366 |        0.01621  |       0.001807 |          0.016283 |       0.017979 |
| subcenter_arcface | on                      |      3 |              3 |       0.016133 |        0.01733  |       0.001944 |          0.016283 |       0.019573 |
| triplet           | on                      |      3 |              3 |       0.014366 |        0.024547 |       0.009884 |          0.02517  |       0.034104 |
| circle            | on                      |      3 |              3 |       0.030558 |        0.03896  |       0.007482 |          0.041419 |       0.044902 |
| contrastive       | on                      |      3 |              3 |       0.035822 |        0.041274 |       0.004807 |          0.043099 |       0.044902 |

## Comparação pareada por seed (OFF - ON)
- Interpretação: `delta_off_minus_on > 0` favorece professor ON; `< 0` favorece OFF.
_Sem dados._

## Próximos passos sugeridos
- Rodar Sprint 1 sem professor em todas as épocas para a melhor loss observada.
- Fazer sweep dedicado da rede professor mantendo fixa a melhor loss do aluno.
- Repetir com 3 seeds fixas para confirmar estabilidade da decisão final.
