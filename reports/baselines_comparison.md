# Baselines de Logistic Regression — comparativa

Dos modelos LR para aislar la contribución del bloque de texto vs el bloque tabular. Ambos:
- Entrenados sobre 80% del train (split estratificado, seed=42)
- Evaluados sobre 20% interno (held-out test del loader queda intocable)
- `sample_weight` asimétrico: `used` pesa 3.0, `new` pesa 1.0 (ratio 3:1 FP:FN)
- `solver=liblinear`, `C=1.0`, `max_iter=1000`

## Resultados (validación interna, n=18.000)

| Métrica | LR text-only | LR full | Δ (full − text) |
|---------|-------------:|--------:|----------------:|
| **Accuracy** | 0.7680 | **0.8621** | **+9.41 pp** |
| **F_0.5 (new)** | 0.8356 | **0.9081** | +7.25 pp |
| Precision (new) | 0.9091 | 0.9429 | +3.38 pp |
| Recall (new) | 0.6312 | 0.7911 | +15.99 pp |
| Recall (used) | 0.9268 | 0.9444 | +1.76 pp |
| F1 macro | 0.7661 | 0.8620 | +9.59 pp |
| Expected cost (3:1) | 0.2998 | 0.1894 | −0.110 |
| Tiempo de fit | 4.8s | 29.0s | +24s |

### Confusion matrices

**LR text-only** — TN=7.720, FP=610, FN=3.566, TP=6.104
**LR full**     — TN=7.867, FP=463, FN=2.020, TP=7.650

## Lectura clave

1. **El LR full ya pasa el umbral del enunciado (0.8621 ≥ 0.86)**. Esto significa que el problema es razonablemente lineal en el espacio de features que diseñamos, y que LightGBM va a aportar mejora marginal por encima — no una mejora dramática.

2. **El delta de +9.4 pp entre text-only y full confirma que las features tabulares aportan señal sustancial**. No es solo TF-IDF; `listing_type_id`, `sold_quantity`, `has_warranty`, etc. están haciendo trabajo real.

3. **Precision (new) = 0.9429 con `sample_weight` 3:1**: el modelo está calibrado al lado conservador. Cuando dice "new", acierta el 94% de las veces. Es exactamente lo que el argumento de costo asimétrico pedía.

4. **Recall (used) = 0.9444 (LR full)**: visto como Trust & Safety, detectamos el 94.4% de los items realmente usados. Equivalente a especificidad de la clase 'new'.

5. **Expected cost (3:1) baja de 0.30 a 0.19** del baseline solo-texto al baseline full. Esto va en la misma dirección que F_0.5 — los dos baselines no están en conflicto sobre cuál es mejor, lo cual da confianza al threshold tuning posterior.

## Top-15 n-grams por clase (LR text-only)

**Hacia 'used'** (coeficientes positivos):
- `'usad'`, `'como '`, `' usad'`, `' como'`, `'usado'`, `'ntigu'`, `'antig'`, `'estad'` → palabras del español argentino que delatan: "usado", "como nuevo", "antiguo", "buen estado".

**Hacia 'new'** (coeficientes negativos):
- `' x '`, `' p/'`, `'ml '`, `' mint'`, `'cc '`, `'mm '`, `'0w '`, `'0ml'` → especificaciones técnicas: "10x", "para", "300ml", "150cc", "20mm", "0w" (aceites). Productos nuevos típicamente listan especs métricas; productos usados los describen subjetivamente.

Esto es un patrón muy interpretable y va al reporte como sanity check de que el modelo aprendió señal del producto, no ruido espurio.

## Implicancia para LightGBM

- **Objetivo razonable**: superar 0.88 accuracy. Anything above eso valida que el GBDT explota interacciones no lineales entre features.
- **Si LightGBM con tuning serio se queda en ~0.86–0.87**, el incremento es marginal y la ablación de `listing_type_id` será aún más informativa.
- Las features que ya identificó la LR (signo del coeficiente) van a alinearse con la feature_importance del LightGBM. Si discrepan, vale revisar.
