# Item Condition Classifier — MercadoLibre (MLA)

Clasificador binario `new` / `used` sobre listings de Mercado Libre Argentina, como entrega de la **Parte 1** del Technical Assessment para LATAM E-commerce Scraping Team — Fintech.

## Objetivo

Construir una solución de ML end-to-end que prediga si un listing es nuevo o usado, alcanzando **≥0.86 de accuracy** sobre el split held-out devuelto por `build_dataset()` de `new_or_used.py`, con el rigor analítico y de comunicación esperado de un Tech Lead.

## Status

En desarrollo. Plan completo en [`PLAN_PARTE1.md`](PLAN_PARTE1.md).

| Fase | Status |
|------|--------|
| P1.0 Bootstrap | en progreso |
| P1.1 EDA + auditoría | pendiente |
| P1.2 Catálogo de features | pendiente |
| P1.3 Baselines LR | pendiente |
| P1.4 LightGBM + Optuna | pendiente |
| P1.5 XGBoost comparativo | pendiente |
| P1.6 Ablación `listing_type_id` | pendiente |
| P1.7 Threshold tuning | pendiente |
| P1.8 Error analysis | pendiente |
| P1.9 Eval final + REPORT | pendiente |

## Reproducibilidad

```bash
# 1. Instalar dependencias (recomendado: uv)
uv sync

# 2. Verificar que el dataset esté presente
ls -lh MLA_100k.jsonlines    # ~332 MB

# 3. Correr la pipeline completa
make all

# 4. Ver el reporte final
open reports/REPORT.md
```

Targets individuales: ver `make help`.

## Decisiones clave (cerradas)

| Decisión | Valor | Razón resumida |
|----------|-------|----------------|
| Modelo principal | LightGBM | rápido, eficiente en memoria, manejo nativo de missing |
| Modelo de comparación | XGBoost | bonus "compare model families" |
| Baselines | LR full + LR text-only | aísla la contribución de tabular vs texto |
| Tuning | Optuna TPE 50 trials, 30 min cap | más eficiente que GridSearchCV |
| CV | StratifiedKFold k=3 sobre X_train | el test del loader se toca **una sola vez** al final |
| Métrica primaria | accuracy ≥ 0.86 | requisito del enunciado |
| Métrica secundaria | **F_0.5 sobre clase `new`** | costo asimétrico: FP daña confianza del comprador > FN daña visibilidad del seller |
| Ratio de costo | **3:1 (FP:FN)** | cota inferior conservadora; inyectado vía `sample_weight` y reportado como `expected_cost` |
| Random seed | 42 | reproducibilidad |

Ver argumento completo de la métrica secundaria en `PLAN_PARTE1.md` sección 6, replicado al cierre en `reports/REPORT.md`.

## Estructura del repo

```
.
├── src/
│   ├── config.py              # SEED, paths, ratios de costo
│   ├── data.py                # wrapper sobre build_dataset()
│   ├── features.py            # ColumnTransformer + transformaciones derivadas
│   ├── eda/                   # análisis exploratorio
│   ├── models/                # LR baselines, LightGBM, XGBoost
│   ├── eval/                  # métricas, threshold, error analysis, run final
│   └── experiments/           # ablación listing_type_id
├── notebooks/                 # 01_eda.ipynb, 02_error_analysis.ipynb
├── reports/                   # documentos en Markdown + figuras + CSV de errores
│   ├── eda_findings.md
│   ├── feature_catalog.md
│   ├── ablation_listing_type.md
│   ├── REPORT.md              # documento final del entregable
│   └── figures/
├── models/                    # artefactos serializados (gitignored)
├── new_or_used.py             # provisto por el assessment — NO MODIFICAR
├── MLA_100k.jsonlines         # dataset (gitignored — 332 MB)
├── pyproject.toml
├── Makefile
├── PLAN_PARTE1.md             # plan de ejecución detallado
└── README.md
```

## Notas de defensa para la entrevista

- El test split del loader (`build_dataset` últimos 10k) se evalúa **una sola vez** al final (`src/eval/run_test.py`). Toda la iteración de modelado se hace sobre splits internos del 90k de train.
- El experimento de ablación con/sin `listing_type_id` es la pieza central del reporte: responde directamente a la pregunta *"¿el modelo aprendió 'new vs used' o solo 'pricing tier de Meli'?"*.
- El threshold operativo se elige por barrido sobre validación, no por default 0.5. Es un hiperparámetro documentado.
- Los pesos asimétricos (`sample_weight` 3:1) son señal de entrenamiento, no solo de evaluación. El modelo internaliza la asimetría desde el fit.
