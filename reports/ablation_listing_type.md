# Ablacion: con vs sin `listing_type_id`

Re-entrenamiento del LightGBM tuneado (mismos hiperparametros, mismo split,
misma semilla) sobre dos variantes:
- **A**: full features (incluyendo listing_type_id)
- **B**: full features menos listing_type_id

## Resultados

| Metrica | A (full) | B (sin listing_type_id) | Delta (A - B) |
|---------|---------:|------------------------:|--------------:|
| Accuracy | 0.8999 | 0.8566 | +4.328 pp |
| F_0.5 (new) | 0.9351 | 0.9064 | +2.872 pp |
| Recall (used) | 0.9580 | 0.9478 | +1.020 pp |
| Expected cost (3:1) | 0.1390 | 0.1917 | -0.0527 |
| n_features tras preprocessor | 20272 | 20265 | 7 |
| best_iter LightGBM | 1866 | 2000 | — |

## Confusion matrices

**A (full)**:
- TN=7980
- FP=350
- FN=1452
- TP=8218

**B (sin listing_type_id)**:
- TN=7895
- FP=435
- FN=2146
- TP=7524

## Veredicto

**Caida intermedia (4.33pp)**: el modelo depende moderadamente de listing_type_id. La feature aporta senal pero el modelo no es solo un proxy del tier; el resto de features hace trabajo.

## Como interpretarlo en la entrevista

`listing_type_id` es la feature dominante segun el EDA (spread de %used desde
1.93% en `gold_special` hasta 92.59% en `free`). NO es leakage tecnico (lo
elige el seller al listar, no depende de la etiqueta).

Pero el riesgo era que el modelo se apoyara casi exclusivamente en este tier
y aprendiera "vendedor pago listing premium -> producto nuevo" en vez de la
senal real del producto.

Esta tabla cuantifica esa dependencia con un experimento ablation y permite
afirmar con numeros si el modelo es defendible en produccion.
