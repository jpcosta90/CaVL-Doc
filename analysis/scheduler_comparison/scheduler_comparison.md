# Comparação hipotética de schedulers

## Configuração
- Learning rate inicial: `1e-05`
- ReduceLROnPlateau: factor=`0.5` | patience=`1`
- Épocas simuladas: `10`

## Leitura rápida
- Cosine termina em `0`.
- Plateau termina em `1.25e-06` com `3` reduções.

## Interpretação
- `cosine` faz uma queda suave e previsível ao longo das épocas.
- `plateau` mantém a LR até a métrica estagnar e só então reduz por fator multiplicativo.
- Em cenários com melhoria inicial seguida de piora, `plateau` reage mais diretamente ao sinal de validação.

## Próxima decisão prática
- Se a validação oscila e piora cedo, `plateau` tende a ser mais adaptativo.
- Se você quer uma agenda determinística sem depender da métrica, `cosine` é mais previsível.
