# Plan de Ejecución — Parte 1: Item Condition Classifier

**Autor**: Sergio Iván Villamizar Delgado
**Contexto**: Technical Assessment — LATAM E-commerce Scraping Team (Fintech)
**Versión**: v1.0 — plan previo a ejecución
**Última actualización**: 2026-05-03

---

## 1. Objetivo

Construir un clasificador binario (`new` / `used`) sobre el dataset MLA_100k de Mercado Libre, alcanzando **≥0.86 de accuracy** sobre el split de test devuelto por `build_dataset()`, con el rigor analítico, la calidad de comunicación y el criterio de trade-offs que se espera de un Tech Lead.

**Filosofía de la entrega** (cita literal del PDF): *"We would rather see a smaller, polished system than a large unfinished one."* Cada decisión queda documentada con su razón; el reporte final pesa tanto como el código.

---

## 2. Restricciones del enunciado (no negociables)

1. Cargar datos exclusivamente vía `build_dataset()` de `new_or_used.py`. Cualquier otro método invalida la entrega.
2. Accuracy ≥0.86 sobre el held-out test (últimos 10.000 registros del archivo).
3. Métrica secundaria distinta de accuracy, justificada por escrito.
4. EDA documentado con findings sobre balance de clases, distribuciones, nulos y leakage.
5. Reproducibilidad: random seed fijo, splits deterministas, instrucciones de ejecución one-command.
6. Entrega en repo Github con `pyproject.toml` y reporte (Markdown/notebook/PDF).

---

## 3. Decisiones tomadas (CONFIRMADAS)

| Decisión | Valor confirmado | Razón |
|----------|------------------|-------|
| Modelo principal | LightGBM | Más rápido que XGBoost, manejo nativo de missing y categóricas, fácil de defender en entrevista |
| Modelo de comparación | XGBoost (sin tuning extensivo) | Cumple bonus "compare model families" sin sobreingeniar |
| Baseline obligatorio | Logistic Regression sobre TF-IDF + tabular | Sanity check + número a superar |
| Comparativo extra | LR solo TF-IDF sobre `title` (sin features tabulares) | Aísla la contribución del texto vs tabular; refuerza el argumento de que la señal vive en ambas |
| Hyperparameter tuning | Optuna TPE 30–50 trials con early stopping | Más eficiente que GridSearchCV |
| Estrategia de CV | StratifiedKFold k=3 sobre X_train; eval final único sobre X_test del loader | Conserva el test del loader intacto y previene overfit a un dev split |
| Clase positiva | `new` | Convención sklearn estándar para binario |
| **Métrica secundaria** | **F_0.5 sobre clase `new`** | Pondera precision al doble que recall — alineado con asimetría de costo |
| **Ratio de costo asimétrico** | **3:1 (FP:FN)** | Defendible y conservador — devolución/churn de comprador estimado en una orden de magnitud sobre visibilidad perdida del seller |
| Métrica de cross-check | Expected cost con ratio 3:1 | Sanity check del trade-off de costo |
| Señal de entrenamiento | `sample_weight` con peso 3.0 para "used", 1.0 para "new" | Inyecta el ratio 3:1 durante el fit, no solo en evaluación |
| Random seed | 42 (global, en `src/config.py`) | Reproducibilidad |
| **Versionado** | **Commit etapa por etapa** (un commit por fase Pn.x al cerrarla) | Trazabilidad del proceso para el reviewer; permite git log narrativo |

Estado: **decisiones cerradas — listo para ejecutar P1.0**.

---

## 4. Plan por fases — vista resumen

| # | Fase | Tiempo | Entregable principal |
|---|------|--------|----------------------|
| P1.0 | Bootstrap del repo | 0.5h | Estructura, `pyproject.toml`, `Makefile`, seeds globales |
| P1.1 | EDA + auditoría de label + distribution shift | 1.5h | `notebooks/01_eda.ipynb` + `reports/eda_findings.md` |
| P1.2 | Catálogo de features (campo-por-campo) | 0.5h | `reports/feature_catalog.md` |
| P1.3 | Pipeline + baseline LR (full) + LR text-only | 1.5h | `src/features.py`, `src/models/lr_baseline.py`, `src/models/lr_text_only.py` + tabla comparativa de baselines |
| P1.4 | LightGBM con Optuna | 1.5h | `src/models/lightgbm_model.py` + `models/lgbm_best.pkl` |
| P1.5 | Comparativo XGBoost | 0.5h | `src/models/xgboost_model.py` + tabla comparativa |
| P1.6 | Ablación con/sin `listing_type_id` | 0.5h | `reports/ablation_listing_type.md` (pieza central del reporte) |
| P1.7 | Threshold tuning + métrica secundaria | 0.5h | `reports/figures/` (PR curve, ROC, conf matrix) |
| P1.8 | Análisis de errores top-100 | 0.5h | `reports/errors_top100.csv` + `notebooks/02_error_analysis.ipynb` |
| P1.9 | Eval final sobre X_test + reporte | 1h | `reports/REPORT.md` + `README.md` |
|  | **Total** | **8h** |  |

---

## 5. Detalle por fase

### P1.0 — Bootstrap (0.5h)

Estructura de carpetas:

```
.
├── src/
│   ├── __init__.py
│   ├── config.py                 # SEED=42, paths, constantes globales
│   ├── data.py                   # wrapper sobre build_dataset()
│   ├── features.py               # ColumnTransformer + transformaciones derivadas
│   ├── models/
│   │   ├── lr_baseline.py
│   │   ├── lightgbm_model.py
│   │   └── xgboost_model.py
│   ├── eval/
│   │   ├── metrics.py            # F_0.5, expected_cost, custom feval
│   │   ├── threshold.py
│   │   ├── error_analysis.py
│   │   └── run_test.py           # eval final UNA sola vez sobre X_test
│   └── experiments/
│       └── ablation_listing_type.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_error_analysis.ipynb
├── reports/
│   ├── eda_findings.md
│   ├── feature_catalog.md
│   ├── ablation_listing_type.md
│   ├── REPORT.md                 # documento final
│   ├── errors_top100.csv
│   ├── optuna_study.json
│   └── figures/
├── models/                       # artefactos serializados (gitignore)
├── new_or_used.py                # NO MODIFICAR
├── MLA_100k.jsonlines            # NO MODIFICAR (gitignore)
├── pyproject.toml
├── Makefile
├── README.md
└── .gitignore
```

**`pyproject.toml` — dependencias pineadas**: python ≥3.11, lightgbm, xgboost, scikit-learn, pandas, numpy, optuna, shap, matplotlib, seaborn, jupyter, pytest.

**Makefile targets**:
- `make eda` — ejecuta el notebook EDA y exporta findings
- `make train` — entrena LightGBM y guarda artefacto
- `make eval` — corre eval final sobre X_test (una sola vez)
- `make report` — regenera `REPORT.md` con números actualizados
- `make all` — encadena todo end-to-end

**`.gitignore`**: `MLA_100k.jsonlines` (332 MB), `models/*.pkl`, `__pycache__/`, `.ipynb_checkpoints/`, `.venv/`.

---

### P1.1 — EDA + auditoría de label + distribution shift (1.5h)

Notebook `notebooks/01_eda.ipynb` con cuatro bloques.

**Bloque A — Estado del dataset**:
- Tamaños X_train / X_test (90k / 10k confirmados).
- Balance de clases en train y test (esperado ~54/46 según verificación previa).
- Tasa de nulos por campo (los 48). Reportar top-10 con más nulos.
- Distribución de `listing_type_id`, `category_id` top-20, `currency_id`, `site_id`.

**Bloque B — Distribution shift train vs test (crítico)**:
- KS test sobre `price` (continuas).
- Chi-cuadrado sobre `condition`, `listing_type_id`, `currency_id`, `category_id` top-20.
- Si hay shift significativo (p<0.05 con efecto material), documentarlo como riesgo. La validación interna debería entonces usar los últimos 10–20% del train (split temporal-mimic) en lugar de KFold puro.

**Bloque C — Auditoría de calidad de label**:
- Conteo de `condition=None` (no se pueden usar para entrenar).
- Filas con `available_quantity=0 ∧ sold_quantity>0` (inconsistencia lógica).
- Precios ≤0 con `condition!=None`.
- Verificación final: ¿`attributes` contiene id "ITEM_CONDITION" en alguna de las 100k filas? (En la muestra de 5k era 0; verificar global.)

**Bloque D — Leakage candidates (cuantificados)**:
- `permalink` con substring "usado"/"nuevo" (cobertura ya medida: 575 / 3.558).
- `title` y `subtitle` con palabras "usado", "permuto", "como nuevo", "sellado", "en caja".
- `tags` × `condition`: tabla de contingencia.
- `warranty` no-null × `condition` (esperado: 45.1% / 32.1% según verificación previa).
- `sold_quantity` × `condition` (esperado: new mean=4.38, used mean=0.09).
- `listing_type_id` × `condition` (esperado: free→92.6% used, gold_special→1.85% used).

**Output**: `reports/eda_findings.md` con bullets de hallazgos clave (sin código), referenciado desde el reporte final.

---

### P1.2 — Catálogo de features (0.5h)

Tabla en `reports/feature_catalog.md` con decisión campo-por-campo. Plantilla resumida:

| Campo | Tipo raw | Decisión | Justificación |
|-------|----------|----------|---------------|
| `id`, `parent_item_id` | string | drop | identificadores |
| `permalink`, `thumbnail`, `secure_thumbnail` | string | drop | URLs; `permalink` contiene "usado"/"nuevo" en 4.1k filas (leakage parcial) |
| `pictures` | list | derive `n_pictures` | densidad informativa del listing |
| `seller_id` | int | drop | leakage de identidad del vendedor (memoriza) |
| `seller_address` | dict | derive `country_id`, `state_id` | geografía útil sin memorizar seller |
| `title` | string | TF-IDF char_wb (3,5) max_features=20k | tolera mayúsculas/acentos/typos |
| `subtitle` | string | derive `has_subtitle`, drop contenido | sparse + leakage potencial |
| `descriptions` | list | derive `has_description` | low coverage para usar contenido |
| `category_id` | string | OneHot top-N + "other" | granularidad sin explotar dim |
| `currency_id` | string | OneHot | bajo cardinal |
| `price`, `base_price` | int | log1p + flag was_zero | distribución sesgada |
| `original_price` | int | derive `has_discount`, `discount_pct` | mayoría null |
| `available_quantity`, `initial_quantity`, `sold_quantity` | int | keep + derive `sold_ratio = sold/initial` | señal de listing premium |
| `listing_type_id` | string | OneHot — feature dominante (sujeto a ablación) | señal real, validar con P1.6 |
| `buying_mode` | string | OneHot | bajo cardinal |
| `condition` | string | TARGET — no feature | objetivo |
| `accepts_mercadopago` | bool | keep | señal de seller serio |
| `non_mercado_pago_payment_methods` | list | derive `n_payment_methods`, multi-hot tipos top | listing rico |
| `tags` | list | multi-hot top-10 (`dragged_bids_and_visits`, `good_quality_thumbnail`, etc.) | señal de listing |
| `attributes` | list | derive `n_attributes` + multi-hot keys top-20 | señal de detalle |
| `warranty` | string | derive `has_warranty` | 45% new vs 32% used |
| `start_time`, `stop_time` | iso datetime | derive `listing_duration_days` | duración intencional |
| `date_created`, `last_updated` | iso datetime | derive `listing_age_days`, `time_since_update_days` | features temporales relativas |
| `status`, `sub_status` | string/list | OneHot / multi-hot | filtros de estado |
| `automatic_relist` | bool | keep | señal de listing repetitivo |
| `shipping` | dict | derive `free_shipping`, `mode`, `local_pickup` | logística |
| `coverage_areas` | list | derive `has_coverage` | poco usado |
| `differential_pricing` | dict | derive `has_diff_pricing` | mayoría null |
| `deal_ids` | list | derive `n_deals` | participación en ofertas |
| `geolocation`, `location` | dict | drop | redundante con seller_address |
| `international_delivery_mode` | string | OneHot | bajo cardinal |
| `seller_contact`, `video_id`, `catalog_product_id`, `official_store_id` | string/null | derive `is_*_present` flags | mayoría null |
| `variations` | list | derive `n_variations` | tallas/colores |
| `listing_source` | string | drop | constante o low-cardinality opaco |

Total estimado: **~80 features finales** después de OneHot (sin contar TF-IDF char_wb).

---

### P1.3 — Pipeline + baseline LR (full) + LR text-only (1.5h)

`src/features.py`: `ColumnTransformer` con tres ramas.

**Numérica**: `SimpleImputer(strategy="median")` + flag binario `was_missing` + `StandardScaler` (desactivado vía `with_mean=False` cuando se enchufa al pipeline GBDT).

**Categórica**: `SimpleImputer(strategy="constant", fill_value="__missing__")` + `OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist")`.

**Texto** (`title`): `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=20000, lowercase=True, strip_accents="unicode")`.

**Dos baselines de LR para aislar la contribución de cada bloque**:

1. `src/models/lr_baseline.py` — **LR full**: tabular + texto. `LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)`. Es el número-piso que el GBDT debe superar.
2. `src/models/lr_text_only.py` — **LR text-only**: solo TF-IDF char_wb sobre `title`. Aísla la contribución del texto. Si este modelo solo ya supera 0.80, demuestra que el título es muy informativo (los usados se delatan léxicamente: "usado", "permuto", "negociable", "como nuevo").

Entrenar ambos sobre 80% de X_train (split estratificado, seed=42), validar sobre 20%.

Reportar tabla comparativa con: accuracy, precision(new), recall(new), F_0.5(new), F1_macro.

| Modelo | Accuracy | F_0.5(new) | Recall(used) | Comentario |
|--------|----------|------------|--------------|------------|
| LR text-only | … | … | … | aislamos contribución del título |
| LR full | … | … | … | piso a superar por GBDT |

**Expectativas**: LR text-only ~0.78–0.83; LR full ~0.83–0.86. Si LR full ya pasa 0.86, eso es información: el problema es lineal y el GBDT solo aporta marginalmente. En cualquier caso, esta tabla anclará el reporte y demuestra que la señal vive tanto en texto como en features tabulares — argumento defendible cuando el reviewer pregunte *"¿por qué no usaste solo TF-IDF si era suficiente?"*.

---

### P1.4 — LightGBM con Optuna + early stopping (2h)

`src/models/lightgbm_model.py`: pipeline = `features.py` → `LGBMClassifier`.

**Estrategia**:
1. **K-fold estratificado k=3** sobre X_train, estratificando por `condition`.
2. **Optuna TPE, 30–50 trials**. Espacio de búsqueda:
   - `num_leaves`: 31–255 (log)
   - `max_depth`: 5–12
   - `learning_rate`: 0.01–0.2 (log)
   - `min_child_samples`: 5–100
   - `feature_fraction`: 0.6–1.0
   - `bagging_fraction`: 0.6–1.0
   - `lambda_l1`, `lambda_l2`: 0–10 (log)
   - `n_estimators`: 2000 con `early_stopping_rounds=50` sobre fold validation
3. **Pesos asimétricos**: `sample_weight` con peso 3.0 para "used", 1.0 para "new". Implementa el ratio 3:1 (FP:FN) durante el fit.
4. **Métrica de optimización Optuna**: F_0.5 sobre clase "new" (custom feval).
5. Hard cap: 50 trials o 30 minutos (lo que llegue primero).
6. Persistir mejor configuración + estudio en `reports/optuna_study.json` y modelo en `models/lgbm_best.pkl`.

**Expectativa**: 0.89–0.92 accuracy.

---

### P1.5 — Comparativo XGBoost (0.5h)

`src/models/xgboost_model.py`: misma pipeline de features. Hiperparámetros razonables sin tuning extensivo:
- `max_depth=8`, `learning_rate=0.05`, `n_estimators=1000` con `early_stopping_rounds=50`
- `scale_pos_weight=3.0` (ratio asimétrico)
- `tree_method="hist"` (más rápido)

Reportar tabla comparativa con: accuracy, F_0.5(new), tiempo de entrenamiento, tamaño del modelo serializado. Trade-offs explicados en una línea cada uno (LightGBM más rápido y eficiente en memoria; XGBoost más maduro en producción y más robusto en categóricas con missing nativo).

---

### P1.6 — Ablación con/sin `listing_type_id` (0.5h)

Esta es la **pieza central del reporte**. Va en `src/experiments/ablation_listing_type.py`: re-entrenar el LightGBM con los hiperparámetros del best, dos veces:
- (A) features completas
- (B) features completas SIN `listing_type_id`

Tabla en `reports/ablation_listing_type.md`:

| Setup | Accuracy | F_0.5(new) | Recall(used) | Expected cost (3:1) |
|-------|----------|------------|--------------|---------------------|
| Full features | … | … | … | … |
| Sin listing_type_id | … | … | … | … |
| Δ (drop) | … | … | … | … |

**Interpretación esperada**:
- Δ accuracy ≤ 1.5pp → el modelo aprendió señal robusta más allá del pricing tier de Meli; defensible en producción.
- Δ accuracy ≥ 5pp → el modelo es esencialmente un proxy de listing tier; reportarlo honestamente y proponer feature engineering adicional como next step.

Este experimento responde directamente a la crítica que un reviewer va a hacer en la entrevista live: *"¿tu modelo aprendió a distinguir nuevo de usado, o solo a distinguir sellers premium de sellers casuales?"*. Llegar con la tabla ya hecha es la diferencia.

---

### P1.7 — Threshold tuning + métrica secundaria (0.5h)

`src/eval/threshold.py`: sobre el set de validación del mejor LightGBM, barrer thresholds en `[0.3, 0.7]` step 0.01.

Tabla por threshold con: precision(new), recall(new), F_0.5(new), recall(used), expected_cost.

Elegir threshold óptimo por máximo F_0.5(new). Documentar como **hiperparámetro operacional** en el reporte (es una decisión consciente, no el default 0.5).

Generar en `reports/figures/`:
- `pr_curve_used.png` — PR curve sobre clase used
- `roc_curve_new.png` — ROC sobre clase new
- `confusion_matrix_optimal_threshold.png`
- `f05_vs_threshold.png` — curva del barrido

---

### P1.8 — Análisis de errores top-100 (0.5h)

`src/eval/error_analysis.py`: sobre X_test al threshold óptimo:
- **Top-50 FP**: predijo new con `proba ≥ threshold` pero era used, ordenados por `proba` descendente.
- **Top-50 FN**: predijo used con `proba < threshold` pero era new, ordenados por `proba` ascendente.

CSV en `reports/errors_top100.csv` con columnas: `title`, `category_id`, `listing_type_id`, `price`, `condition_real`, `proba`, `len_title`, `n_pictures`, `has_warranty`.

Notebook `notebooks/02_error_analysis.ipynb`: SHAP values sobre los top-100 errores. Identificar patrones (¿categorías particulares? ¿rangos de precio? ¿títulos cortos?).

**Output**: 1 párrafo en el reporte explicando los patrones de fallo dominantes y qué se haría con más tiempo.

---

### P1.9 — Eval final + reporte (1h)

`src/eval/run_test.py`: cargar `models/lgbm_best.pkl`, aplicar threshold óptimo, evaluar sobre `X_test` del `build_dataset()` **una sola vez**. Reportar:
- Accuracy (verificar ≥0.86)
- F_0.5(new)
- Recall(used)
- Confusion matrix
- Expected cost (ratio 3:1)

`reports/REPORT.md` (estructura del documento final):
1. Dataset overview (1 párrafo)
2. EDA findings clave (3–5 bullets)
3. Decisiones de features (link a `feature_catalog.md`)
4. Modelo elegido y por qué (LightGBM vs XGBoost vs LR baseline)
5. Hiperparámetros finales
6. **Ablación `listing_type_id`** (tabla + interpretación)
7. Métrica secundaria + argumento de negocio
8. Threshold operacional
9. Métricas finales sobre held-out test (tabla)
10. Análisis de errores (1 párrafo + link al notebook)
11. Limitaciones
12. What I would do with more time

`README.md` con instrucciones de reproducción:

```bash
# 1. instalar
uv sync   # o: pip install -e .

# 2. correr todo
make all

# 3. inspeccionar resultados
open reports/REPORT.md
```

---

## 6. Argumento completo de la métrica secundaria

(Texto preparado tal cual va al reporte final.)

> **Métrica secundaria elegida**: F_0.5 sobre la clase "new".
>
> **Convención**: clase positiva = "new", clase negativa = "used".
>
> **Argumento de negocio**: en el ecosistema de Mercado Libre, los dos tipos de error tienen impacto asimétrico sobre la confianza del marketplace.
>
> Un Falso Positivo (predicción "new" siendo realmente "used") significa que un comprador adquiere un producto creyendo que es nuevo y recibe uno usado. Las consecuencias son: pérdida de confianza del comprador, reclamo formal, devolución logística costosa, posible fraude del seller, y daño reputacional de la plataforma. Es un evento *no recuperable* en términos de confianza.
>
> Un Falso Negativo (predicción "used" siendo realmente "new") significa que un seller publicó un producto nuevo legítimo y la plataforma lo clasifica como usado. Las consecuencias son: ranking de búsqueda subóptimo y menor visibilidad. Pero si el comprador finalmente compra, recibe algo mejor de lo esperado. Es un evento *recuperable*.
>
> Por esta asimetría, queremos que cuando el modelo afirma "new", esté altamente seguro. Esto se traduce en maximizar Precision sobre la clase "new", aceptando sacrificar algo de Recall.
>
> El F_β score con β=0.5 pondera la Precision al doble que el Recall, encodando matemáticamente esta preferencia. Es equivalente a maximizar Recall sobre la clase "used", lo cual es la formulación natural en lenguaje de Trust & Safety.
>
> **Métrica de cross-check**: Expected Cost con ratio 3:1 (FP cuesta 3× FN). El ratio se justifica como cota inferior conservadora del costo real (devoluciones + soporte + churn de comprador en una orden de magnitud por encima del costo de visibilidad perdida del seller). Si F_0.5 y Expected Cost coinciden en preferir el mismo modelo y threshold, la decisión es robusta.
>
> **Señal de entrenamiento**: el ratio se inyecta vía `sample_weight` durante el fit (peso 3.0 para "used", peso 1.0 para "new"), no solo se reporta en evaluación. Esto guía al modelo a internalizar la asimetría desde el aprendizaje.

---

## 7. Riesgos y mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| Distribution shift train vs test | Media | Detectado en P1.1; si existe, usar últimos 10–20% del train como validación interna en lugar de KFold puro |
| Optuna se pasa del tiempo | Media | Hard cap de 50 trials o 30 min por estudio (lo que llegue primero) |
| El modelo solo aprende `listing_type_id` | Media-alta (sospechoso por los números del EDA) | Ablación P1.6 lo expone; si es el caso, lo reportamos honestamente y el reporte gana en lugar de perder |
| Memoria insuficiente al cargar 100k JSON | Baja | El JSONL pesa 332 MB y queda manejable en RAM; fallback a `ijson` si hace falta |
| TF-IDF char_wb explota dimensionalidad | Baja | `max_features=20000` cap |
| El reporte queda incompleto al final | Baja | El `REPORT.md` se versiona desde el día 1 y se llena sección por sección al cerrar cada fase |

---

## 8. Lo que queda explícitamente fuera (cuts documentados)

- **Sin embeddings ni transformers** sobre title (BERT, e5-multilingual, etc.). TF-IDF char_wb es el sweet spot costo/beneficio para 8h.
- **Sin procesamiento de imágenes** (`pictures`). Requiere modelo de visión y rompe el presupuesto. Se menciona como next step.
- **Sin tuning extensivo de XGBoost**. Solo hiperparámetros razonables para comparativa entre familias.
- **Sin calibración de probabilidades** (Platt/isotonic). El threshold tuning sustituye en parte y es más interpretable para producción.
- **Sin stacking ni blending**. Un solo modelo final.
- **Sin tests unitarios extensivos**. Solo smoke tests del pipeline (que el ColumnTransformer no se rompa al ver categorías nuevas).

Estas decisiones se listan explícitamente en la sección "Limitations" del REPORT.md, con una línea de justificación cada una.

---

## 9. What I would do with more time

(Sección preparada para el reporte final, copiable verbatim.)

1. **Embeddings multilingües sobre title** (e5-multilingual o LaBSE) y blending con LightGBM.
2. **Calibración de probabilidades** (Platt scaling) sobre validación, para que el threshold tenga interpretación probabilística directa.
3. **CV estratificado por `category_id`** además de `condition`, para asegurar generalización inter-categoría.
4. **Análisis temporal**: entrenar con primer 50% del archivo y evaluar con segundo 50%. Es la prueba real de robustez en producción.
5. **Detección de fraud signals adicionales**: precio vs precio mediano de la categoría (z-score), seller con tasa anormal de listings nuevos, etc.
6. **Comparativa con un modelo de visión** sobre la primera picture: ¿una CNN puede inferir "new vs used" mejor que el texto? Bonus académico.
7. **Servicio de inferencia con FastAPI** + Pydantic para serializar la decisión, con SHAP values en la respuesta para auditoría.

---

## 10. Próximos pasos inmediatos

1. ~~Confirmar los defaults~~ — **CERRADO** (sección 3 actualizada).
2. Ejecutar **P1.0 (Bootstrap)** + **P1.1 (EDA)** en una primera sesión consecutiva.
3. Revisar findings del EDA antes de continuar a P1.2 — algunas decisiones de features dependen de lo que el EDA confirme (especialmente la presencia/ausencia de leakage en campos no auditados todavía).

---

## Anexo A — Hallazgos previos al EDA formal (verificación rápida ya hecha)

| Hallazgo | Valor | Implicancia |
|----------|-------|-------------|
| Balance de clases | 53.8% new / 46.2% used | Casi balanceado; no justifica métrica enfocada en imbalance, sí justifica métrica enfocada en costo asimétrico |
| `attributes` con id="ITEM_CONDITION" en muestra de 5k | 0 hits | El leakage típicamente "más obvio" no aplica a este snapshot; verificar en las 100k igual |
| Archivo ordenado por `date_created` | NO (verificado) | El split del loader es estable pero NO temporal; el argumento de "no shuffle por preservar orden temporal" es falso |
| `permalink` con "usado" / "nuevo" | 575 / 3.558 | Leakage parcial; drop obligatorio |
| `sold_quantity` por clase | new mean=4.38 / used mean=0.09 | Dirección INVERSA a la intuición común; new tiene más por relisteo |
| `warranty` no-null por clase | new=45.1% / used=32.1% | Señal real, defendible |
| `listing_type_id` × condition | free→92.6% used; gold_special→1.85% used | Feature dominante; ablación obligatoria |
| `tags` top | `dragged_bids_and_visits` (73.9k de 100k) | Señal de listing, no del producto — útil |

---

*Este plan es un documento vivo. Cualquier desviación durante la ejecución queda registrada en el reporte final con su razón.*
