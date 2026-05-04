# Item Condition Classifier — Reporte final

**Autor**: Sergio Iván Villamizar Delgado
**Generado**: 2026-05-03
**Repo**: https://github.com/sivd860806/TL_Item_Condition_Classifier

---

## 1. Dataset overview

- 100.000 listings de Mercado Libre Argentina (`MLA_100k.jsonlines`).
- Cargado vía `build_dataset()` de `new_or_used.py` (regla del enunciado).
- Split del loader: 90.000 train + 10.000 test (held-out).
- Balance de clases: train **53.72% new** / **46.28% used**;
  test **54.06% / 45.94%**.
- Sin distribution shift de clase entre train y test (χ² p=0.53).

## 2. Hallazgos clave del EDA

(Detalle completo en [`eda_findings.md`](eda_findings.md))

- **El leakage típico no aplica**: 0 hits con `id == "ITEM_CONDITION"` en `attributes` (verificado con match estricto sobre las 90k filas).
- **Leakage parcial real**: `permalink` contiene "usado" en 516 filas y "nuevo" en 3.231 filas. **Drop obligatorio**.
- **Feature dominante identificada**: `listing_type_id` muestra spread de %used desde 1.93% (`gold_special`) hasta 92.59% (`free`). Es señal real (no leakage técnico) pero el riesgo de proxy se cuantifica en la ablación (sección 6).
- **`sold_quantity` con dirección inversa a la intuición común**: new mean=4.25 vs used mean=0.10 (re-listeo masivo de productos nuevos).
- **`title` con señal léxica fuerte**: 1.624 hits used-leaning ("usado", "permuto", "como nuevo", "antiguo") y 557 hits new-leaning ("sellado", "en caja", "0km").

## 3. Decisiones de features

(Tabla completa campo-por-campo en [`feature_catalog.md`](feature_catalog.md))

Resumen de decisiones críticas:
- **Drop por leakage o no generalización**: `id`, `permalink`, `thumbnail`, `secure_thumbnail`, `seller_id`, `geolocation`, `location`.
- **Multi-hot top-5 de `tags`** (incluye señales fuertes como `free_relist` → 96.1% used).
- **TF-IDF char_wb (3,5) max_features=20.000** sobre `title` — tolera mayúsculas/acentos/typos.
- **Derivaciones temporales**: `listing_duration_days`, `listing_age_days`, `time_since_update_days`, todas relativas a `ref_date = max(last_updated)` del train (sin leakage temporal).
- **Numéricas escaladas solo para LR**, no para GBDT (invariante a escala).

Total: ~80 features densas + hasta 20.000 features sparse de TF-IDF.

## 4. Modelo elegido y rationale

| Modelo | Accuracy (val) | F_0.5(new) | Tiempo fit |
|--------|---------------:|-----------:|-----------:|
| LR text-only (TF-IDF puro) | 0.7680 | 0.8356 | 4.8s |
| LR full (tabular + TF-IDF) | 0.8621 | 0.9081 | 29.0s |
| **LightGBM tuneado (Optuna)** | **0.8999** | **0.9351** | ~40min total |

Modelo elegido: **LightGBM tuneado**. Aporta +3.78pp accuracy sobre LR full
con un costo total de ~40 minutos (50 trials Optuna, cap 30 min).

XGBoost se intentó como cuarto comparativo pero la combinación de matriz sparse
de ~20.000 features + early_stopping_rounds + sample_weight asimétrico generó
tiempos de entrenamiento prohibitivos (>40 min para un solo fit). Se descartó
como decisión consciente: el costo no justifica el incremento marginal esperado.

## 5. Hiperparámetros finales

Encontrados por Optuna TPE (50 trials, timeout 1800s, optimizando F_0.5(new)):

```
learning_rate     : 0.030710573677773714
num_leaves        : 230
max_depth         : 10
min_child_samples : 29
feature_fraction  : 0.6624074561769746
bagging_fraction  : 0.662397808134481
reg_alpha         : 3.3323645788192616e-08
reg_lambda        : 0.6245760287469893

best_iteration   : 1866
sample_weight    : 3.0 para 'used', 1.0 para 'new' (ratio 3:1)
seed             : 42
```

## 6. Ablación estrella: con vs sin `listing_type_id`

(Detalle en [`ablation_listing_type.md`](ablation_listing_type.md))

| Métrica | A (full) | B (sin listing_type_id) | Δ |
|---------|---------:|------------------------:|---:|
| Accuracy | 0.8999 | 0.8566 | +4.328pp |
| F_0.5(new) | 0.9351 | 0.9064 | +2.872pp |
| Recall(used) | 0.9580 | 0.9478 | +1.020pp |
| Expected cost (3:1) | 0.1390 | 0.1917 | -0.0527 |

**Lectura**: el modelo SIN `listing_type_id` retiene **95.2%** de su accuracy original. La feature aporta señal complementaria pero NO es el único pilar — el modelo aprende patrones reales del producto más allá del pricing tier de Meli. Defendible en producción.

## 7. Métrica secundaria y argumento de negocio

**Métrica elegida**: F_0.5 sobre clase `new` (β=0.5 pondera Precision al doble que Recall).

**Argumento**: en el marketplace de Mercado Libre, los dos tipos de error tienen impacto asimétrico:

- **Falso Positivo** (predijo `new`, era `used`) → comprador engañado; pérdida de confianza, devolución, posible fraude. **Evento no recuperable**.
- **Falso Negativo** (predijo `used`, era `new`) → seller perjudicado en visibilidad; ranking subóptimo. **Evento recuperable**.

Por esto: cuando el modelo afirma "new", debe estar muy seguro → maximizar Precision sobre `new`. Esto es matemáticamente equivalente a maximizar Recall sobre `used` (formulación natural en lenguaje de Trust & Safety).

**Cross-check**: Expected cost con ratio 3:1 (FP:FN). Cota inferior conservadora del costo real (devoluciones + soporte + churn de comprador en una orden de magnitud sobre la visibilidad perdida del seller). El ratio se inyecta también como `sample_weight` durante el fit (used=3.0, new=1.0).

## 8. Threshold operacional

(Detalle en `threshold_sweep.json` + figuras `f05_vs_threshold.png`, `pr_curve_used.png`, `roc_curve_new.png`, `confusion_matrix_optimal.png`)

Barrido de thresholds [0.30, 0.70] step 0.01 sobre validación interna.

**Threshold óptimo elegido por F_0.5(new)**: **0.560** (no el default 0.5).

| Threshold | Accuracy | F_0.5(new) | Precision(new) | Recall(used) | Expected cost |
|-----------|---------:|-----------:|---------------:|-------------:|--------------:|
| 0.50 (default) | 0.8999 | 0.9351 | 0.9592 | 0.9580 | 0.1390 |
| **0.56 (óptimo)** | **0.8940** | **0.9355** | 0.9653 | 0.9653 | 0.1381 |

## 9. Métricas finales sobre held-out X_test

**Aplicado UNA SOLA VEZ sobre el X_test del loader** (n=10.000 listings nunca vistos durante tuning).

| Métrica | Valor |
|---------|------:|
| **Accuracy** | **0.8910** |
| F_0.5(new) | 0.9335 |
| Precision(new) | 0.9637 |
| Recall(new) | 0.8296 |
| Recall(used) | 0.9632 |
| F1 macro | 0.8910 |
| Expected cost (3:1) | 0.1428 |
| Threshold operativo | 0.560 |

**Confusion matrix**:
- TN (used→used): 4425
- FP (used→new, daño confianza): 169
- FN (new→used, daño visibilidad): 921
- TP (new→new): 4485

**¿Pasa el threshold del enunciado (≥0.86)?** **SÍ** (0.8910 vs 0.86).

## 10. Análisis de errores

(Detalle en [`error_analysis_summary.md`](error_analysis_summary.md) + CSV `errors_top100.csv`)

Top-50 FP (más confiados en "new" siendo "used") + Top-50 FN (más confiados en "used" siendo "new") con sus campos: `title`, `category_id`, `listing_type_id`, `price`, `n_pictures`, `has_warranty`, `proba_new`. Patrones agregados por `listing_type_id` y por `category_id` documentados en el resumen.

## 11. Limitaciones

- **No se procesaron imágenes** (`pictures`): requiere modelo de visión y rompe el presupuesto de 8h.
- **TF-IDF char_wb sobre español argentino específicamente**: el modelo puede degradarse sobre listings de otros sites LATAM (MLM, MLB, MLC) con vocabulario distinto.
- **`listing_type_id` aporta +4.33 pp** de accuracy (sección 6); modelos que NO tengan acceso a ese feature en producción tendrán performance ~0.86 en lugar de 0.90.
- **Sin calibración explícita de probabilidades**: el threshold tuning sustituye en parte; si se requiere uso probabilístico se recomienda Platt scaling.
- **Validación interna 80/20 (no k-fold) durante tuning**: decisión pragmática de tiempo. K-fold rigoroso es next step.
- **XGBoost descartado** por tiempos de entrenamiento prohibitivos sobre matriz sparse de 20k features.

## 12. What I would do with more time

1. **Embeddings multilingües sobre `title`** (e5-multilingual o LaBSE) y blending con LightGBM.
2. **K-fold k=5 estratificado por `condition` y `category_id`** durante tuning.
3. **Calibración Platt** sobre validación, para que las probabilidades tengan interpretación frecuentista.
4. **Análisis temporal**: entrenar con primer 50% del archivo y evaluar con segundo 50%, prueba real de robustez en producción.
5. **Detección de fraud signals adicionales**: precio vs precio mediano de la categoría (z-score), seller con tasa anormal de listings nuevos.
6. **Modelo de visión** sobre la primera picture y blending con el GBDT.
7. **Servicio de inferencia con FastAPI + Pydantic** y SHAP en la respuesta para auditoría.

---

## Reproducibilidad

```bash
uv sync   # instalar deps
make all  # corre EDA, baselines, train, ablation, threshold, errors, eval final
```

Random seed fijo: 42. Splits deterministas. Todas las decisiones de hiperparámetros y threshold persistidas en `models/lgbm_best.joblib` y `reports/*.json`.

---

*Reporte generado automáticamente por `src/eval/run_test.py` citando los outputs de las fases P1.0 a P1.8.*
