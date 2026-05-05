# Item Condition Classifier — MercadoLibre (MLA)

> Clasificador binario `new` / `used` sobre listings de Mercado Libre
> Argentina, como entrega de la **Parte 1** del Technical Assessment para
> el equipo LATAM E-commerce Scraping (Fintech).

**Resultado**: accuracy **0.8910** sobre el held-out X_test (n=10.000) del
loader `build_dataset()`, vs el **0.86 requerido** por el enunciado.
Margen de **+3.1pp**.

---

## Headline numbers

| Métrica | Valor en held-out X_test |
|---|---:|
| **Accuracy** | **0.8910** |
| F_0.5 sobre `new` (métrica secundaria) | 0.9335 |
| Precision sobre `new` | 0.9637 |
| Recall sobre `new` | 0.8296 |
| Recall sobre `used` (= 1 - FP rate) | 0.9632 |
| Expected cost (FP:FN = 3:1) | 0.1428 |
| Threshold operativo | **0.560** (no el default 0.5) |
| Confusion matrix | TN=4425 / FP=169 / FN=921 / TP=4485 |

---

## Stack

- **Python 3.11+**, sklearn pipelines, LightGBM (modelo principal)
- **Optuna** TPE para hyperparameter tuning (50 trials, cap 30 min)
- **TF-IDF char_wb** sobre `title` (tolerante a typos del español argentino)
- **Pandas** para feature engineering, **matplotlib** para figuras
- **Pytest** para tests de regresión sobre features y métricas
- **uv** para gestión de deps + `pyproject.toml`

---

## Quickstart

```bash
# 1. Instalar dependencias (recomendado: uv, fallback: pip)
uv sync
# o: pip install -e .

# 2. Verificar que el dataset esté presente
ls -lh MLA_100k.jsonlines    # ~332 MB, viene en el zip del assessment

# 3. Correr la pipeline completa (~45 min total con Optuna)
make all

# 4. Ver el reporte final
cat reports/REPORT.md
```

`make help` lista todos los targets individuales si querés correr fases
sueltas (`make eda`, `make baselines`, `make train`, `make ablation`,
`make threshold`, `make eval`).

---

## Pipeline

| Fase | Script | Output |
|------|--------|--------|
| **P1.1** EDA + auditoría de leakage | `src/eda/runner.py` | `reports/eda_findings.{md,json}` + 3 figuras |
| **P1.2** Catálogo de features | (manual, doc) | `reports/feature_catalog.md` |
| **P1.3** Baselines LR (text-only + full) | `src/models/lr_baseline.py` | `reports/baselines_comparison.md` + JSONs |
| **P1.4** LightGBM + Optuna 50 trials | `src/models/lightgbm_model.py` | `models/lgbm_best.joblib` + `reports/lgbm_metrics.json` + `optuna_study.json` |
| **P1.5** XGBoost (descartado: tiempos prohibitivos) | `src/models/xgboost_model.py` | (decisión documentada en REPORT.md §4) |
| **P1.6** Ablación con/sin `listing_type_id` | `src/experiments/ablation_listing_type.py` | `reports/ablation_listing_type.{md,json}` |
| **P1.7** Threshold tuning | `src/eval/threshold.py` | `reports/threshold_sweep.json` + 4 figuras |
| **P1.8** Análisis de errores | `src/eval/error_analysis.py` | `reports/error_analysis_summary.md` + `errors_top100.csv` |
| **P1.9** Eval final + REPORT | `src/eval/run_test.py` | `reports/final_test_metrics.json` + `reports/REPORT.md` |

---

## Resultados de modelado

Sobre validación interna (80/20 split del X_train, seed=42):

| Modelo | Accuracy | F_0.5(new) | Tiempo fit |
|--------|---------:|-----------:|-----------:|
| LR text-only (TF-IDF puro) | 0.7680 | 0.8356 | 4.8s |
| LR full (tabular + TF-IDF) | 0.8621 | 0.9081 | 29.0s |
| **LightGBM tuneado (Optuna)** | **0.8999** | **0.9351** | ~40 min total |

XGBoost se intentó como cuarto comparativo, pero la combinación de matriz
sparse de ~20.000 features + early_stopping_rounds + sample_weight asimétrico
generó tiempos prohibitivos (>40 min para un solo fit). Se descartó como
**decisión consciente documentada** en REPORT.md §4 — el costo no justifica
el incremento marginal esperado.

---

## Decisiones clave

| Decisión | Valor | Razón resumida |
|----------|-------|----------------|
| Modelo principal | LightGBM | Rápido, eficiente en memoria, manejo nativo de missing |
| Tuning | Optuna TPE 50 trials, 30 min cap | Más eficiente que GridSearchCV en mismo wall-clock |
| Validación interna | 80/20 split (no k-fold) | Pragmático por presupuesto de 8h; k-fold es next step |
| Eval final | UNA SOLA VEZ sobre X_test del loader | Sin filtración del test durante tuning |
| Métrica primaria | accuracy ≥ 0.86 | Requisito del enunciado |
| **Métrica secundaria** | **F_0.5 sobre `new`** | Costo asimétrico (ver abajo) |
| Ratio de costo | **3:1 (FP:FN)** | Cota inferior conservadora; inyectado vía `sample_weight` Y reportado como `expected_cost` |
| Threshold operativo | **0.560** (no 0.5) | Tuneado sobre validación, maximiza F_0.5(new) |
| Random seed | 42 | Reproducibilidad |

---

## Argumento de la métrica secundaria

**F_0.5 sobre clase `new`** (β=0.5 → Precision pesa el doble que Recall).

Los dos errores tienen costo asimétrico para el negocio:

- **Falso Positivo** (predijo `new`, era `used`) → comprador engañado;
  pérdida de confianza, devolución, posible fraude. **Evento no
  recuperable**.
- **Falso Negativo** (predijo `used`, era `new`) → seller perjudicado en
  visibilidad / ranking. **Evento recuperable**.

Por esto: cuando el modelo afirma "new", debe estar muy seguro →
maximizar Precision sobre `new`. Es matemáticamente equivalente a
maximizar Recall sobre `used`, formulación natural en lenguaje de
Trust & Safety.

**Cross-check**: Expected cost con ratio 3:1 (FP:FN). El ratio se inyecta
como `sample_weight` durante el fit (used=3.0, new=1.0) **y** se usa para
el threshold tuning post-fit. Las dos palancas se complementan: el
sample_weight afecta el modelo subyacente, el threshold ajusta la decisión
final. Detalle en REPORT.md §7.

---

## Ablación destacada: con vs sin `listing_type_id`

El EDA reveló que `listing_type_id` es feature dominante: spread brutal de
%used desde **1.93%** en `gold_special` hasta **92.59%** en `free`.
**No es leakage técnico** (el seller elige el tier al listar), pero el
riesgo era que el modelo se apoyara casi exclusivamente en este tier.

| Métrica | A (full) | B (sin listing_type_id) | Δ |
|---------|---------:|------------------------:|---:|
| Accuracy | 0.8999 | 0.8566 | +4.328pp |
| F_0.5(new) | 0.9351 | 0.9064 | +2.872pp |
| Recall(used) | 0.9580 | 0.9478 | +1.020pp |
| Expected cost (3:1) | 0.1390 | 0.1917 | -0.0527 |

**Lectura**: el modelo retiene **95.2%** de su accuracy sin esa feature.
Sigue pasando el umbral 0.86. **El modelo aprendió señal real del
producto, no es un proxy del pricing tier de Meli.** Defendible en
producción.

Detalle: [`reports/ablation_listing_type.md`](reports/ablation_listing_type.md)

---

## Hallazgos clave del EDA

(detalle completo en [`reports/eda_findings.md`](reports/eda_findings.md))

- **Leakage típico no aplica**: 0 hits con `id == "ITEM_CONDITION"` en
  `attributes` (verificación global con match estricto). Una versión
  inicial del filtro usaba `'condici' in name` y producía 1.405 falsos
  positivos por capturar "Aire **acondici**onado" — documentado como
  prueba del rigor del check.
- **Leakage parcial real**: `permalink` contiene 'usado' en 516 filas y
  'nuevo' en 3.231 filas → **drop obligatorio**.
- **`title` con señal léxica fuerte**: 1.624 hits used-leaning ("usado",
  "permuto", "como nuevo", "antiguo") y 557 hits new-leaning ("sellado",
  "en caja", "0km").
- **`sold_quantity` con dirección inversa a la intuición común**: new
  mean=4.25 vs used mean=0.10 (re-listeo masivo de productos nuevos por
  sellers de escala).
- **Sin distribution shift de clase entre train y test** (χ² p=0.53).
- **Top-15 campos por nulos**: 8 columnas con >97% nulos → drop o
  derivación de flags.

---

## Reportes detallados (output de `make all`)

- [`reports/REPORT.md`](reports/REPORT.md) — el documento final del
  entregable, agregando los outputs de las fases P1.1 a P1.9.
- [`reports/eda_findings.md`](reports/eda_findings.md) — hallazgos de EDA
  con leakage audit + tablas de %used por feature dominante.
- [`reports/feature_catalog.md`](reports/feature_catalog.md) — decisión
  campo-por-campo (48 campos) sobre qué entra al modelo, cómo, y por qué.
- [`reports/baselines_comparison.md`](reports/baselines_comparison.md) —
  LR text-only vs LR full, contribución relativa del bloque tabular.
- [`reports/ablation_listing_type.md`](reports/ablation_listing_type.md) —
  el experimento de defensa central.
- [`reports/error_analysis_summary.md`](reports/error_analysis_summary.md) —
  top-50 FP + top-50 FN con patrones agregados por categoría.
- [`reports/figures/`](reports/figures/) — confusion matrix, PR curve, ROC
  curve, threshold sweep, distribución de errores.

---

## Limitaciones

- **Sin imágenes**: el campo `pictures` no se procesa (requiere modelo de
  visión). Documentado en REPORT.md §11.
- **Validación interna 80/20 (no k-fold)** durante tuning. Decisión
  pragmática de tiempo. K-fold k=5 estratificado es next step.
- **TF-IDF char_wb sobre español argentino específicamente**: el modelo
  puede degradarse sobre listings de otros sites LATAM (MLM, MLB, MLC).
- **Sin calibración explícita de probabilidades**: el threshold tuning
  sustituye en parte; si se requiere uso probabilístico downstream, Platt
  scaling sería el siguiente paso.
- **XGBoost descartado** por tiempos de entrenamiento prohibitivos sobre
  matriz sparse de 20k features.

---

## What I'd do with more time

(en orden de ROI)

1. **K-fold k=5** estratificado por `condition` y `category_id` durante
   el tuning de Optuna.
2. **Calibración Platt** sobre validación, para que las probabilidades
   tengan interpretación frecuentista.
3. **Análisis temporal**: entrenar con primer 50% del archivo ordenado
   por `last_updated` y evaluar con segundo 50%, prueba real de robustez
   bajo deriva.
4. **Embeddings multilingües** (e5 o LaBSE) sobre `title` y blending con
   LightGBM.
5. **Modelo de visión** sobre la primera picture y blending con el GBDT.
6. **API de inferencia** con FastAPI + Pydantic + SHAP en la respuesta
   para auditoría de decisiones individuales.
7. **Detección de fraud signals adicionales**: precio vs precio mediano
   de la categoría (z-score), sellers con tasa anormal de listings nuevos.

---

## Estructura del repo

```
.
├── src/
│   ├── config.py                 # SEED=42, paths, ratios de costo
│   ├── data.py                   # wrapper de build_dataset()
│   ├── features.py               # ColumnTransformer + flatten + derivaciones
│   ├── eda/
│   │   └── runner.py             # P1.1
│   ├── models/
│   │   ├── lr_text_only.py       # P1.3
│   │   ├── lr_baseline.py        # P1.3
│   │   ├── lightgbm_model.py     # P1.4 (modelo principal)
│   │   └── xgboost_model.py      # P1.5 (descartado, decisión documentada)
│   ├── eval/
│   │   ├── metrics.py            # F_0.5, expected_cost, helpers
│   │   ├── threshold.py          # P1.7
│   │   ├── error_analysis.py     # P1.8
│   │   └── run_test.py           # P1.9 — eval final sobre held-out
│   └── experiments/
│       └── ablation_listing_type.py  # P1.6
├── notebooks/                    # 01_eda.ipynb (preview rápido)
├── reports/                      # outputs de todas las fases (md + json + png)
├── models/                       # artifacts joblib (gitignored)
├── tests/                        # tests de regresión sobre features y métricas
├── new_or_used.py                # PROVISTO por el assessment, NO MODIFICAR
├── MLA_100k.jsonlines            # dataset (gitignored — 332 MB)
├── pyproject.toml
├── Makefile                      # pipeline reproducible
├── PLAN_PARTE1.md                # plan de ejecución original
└── README.md                     # este archivo
```

---

## Notas de defensa para la entrevista

- El test split del loader (`build_dataset()` últimos 10k) se evalúa
  **una sola vez** al final (`src/eval/run_test.py`). Toda la iteración
  de modelado se hace sobre splits internos del 90k de train.
- El experimento de ablación con/sin `listing_type_id` es la pieza
  central del reporte: responde directamente a *"¿el modelo aprendió
  'new vs used' o solo 'pricing tier'?"*.
- El threshold operativo (0.560) se elige por barrido sobre validación,
  no por default. Es un hiperparámetro documentado.
- Los pesos asimétricos (`sample_weight` 3:1) son señal de
  entrenamiento, no solo de evaluación. El modelo internaliza la
  asimetría desde el fit.

---

## Autor

Sergio Iván Villamizar Delgado — AI Manager @ Dichter & Neira. PhD en Ingeniería Eléctrica
(análisis de datos), Universidad Nacional de Colombia.
