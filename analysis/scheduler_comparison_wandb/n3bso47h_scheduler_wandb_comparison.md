# Comparação de schedulers com histórico real do W&B

## Fonte
- Run: `n3bso47h`
- Métrica analisada: `val/eer`
- Épocas observadas: `4`

## Configuração da simulação
- LR inicial: `1e-05`
- Plateau factor: `0.5`
- Plateau patience: `1`

## Resumo
- Melhor métrica observada: `0.035822` na época `1`
- Cosine termina em LR `0`
- Plateau termina em LR `5e-06` com `1` reduções

## Leitura prática
- `cosine` reduz a LR de forma previsível, mesmo se a validação oscilar.
- `plateau` reage diretamente ao histórico: segura LR enquanto houver melhoria e reduz ao estagnar.
- Se a run real melhora cedo e depois piora, `plateau` costuma ser mais coerente para preservar o ponto bom.
