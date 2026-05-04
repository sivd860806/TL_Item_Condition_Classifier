# Documentación técnica del código — Parte 1

**Propósito**: explicar **archivo por archivo y función por función** todo el código del proyecto, con los argumentos de diseño detrás de cada decisión. Sirve como documento de referencia para la entrevista live: cualquier pregunta del tipo *"explicame esta línea / función / decisión"* tiene su respuesta acá.

**Generado**: 2026-05-03

---

## 1. Estructura del repo

```
.
├── src/
│   ├── __init__.py
│   ├── config.py              # constantes globales, seed, ratios de costo
│   ├── data.py                # ÚNICO wrapper alrededor de build_dataset()
│   ├── features.py            # flatten + ColumnTransformer
│   ├── eda/
│   │   ├── __init__.py
│   │   └── runner.py          # P1.1 — EDA en 4 bloques
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py         # F_0.5, expected_cost, reporte unificado
│   │   ├── threshold.py       # P1.7 — barrido + curvas PR/ROC
│   │   ├── error_analysis.py  # P1.8 — top-50 FP + top-50 FN
│   │   └── run_test.py        # P1.9 — eval final + REPORT.md
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lr_baseline.py     # P1.3 — LR full (tabular + texto)
│   │   ├── lr_text_only.py    # P1.3 — LR solo TF-IDF de title
│   │   ├── lightgbm_model.py  # P1.4 — modelo principal
│   │   └── xgboost_model.py   # P1.5 — comparativo (descartado en run final)
│   └── experiments/
│       ├── __init__.py
│       └── ablation_listing_type.py   # P1.6 — ablación estrella
├── notebooks/
│   └── 01_eda.ipynb           # wrapper sobre src/eda/runner.py
├── reports/                   # outputs (markdown + json + figuras)
├── new_or_used.py             # PROVISTO POR EL ASSESSMENT — NO MODIFICAR
├── MLA_100k.jsonlines         # dataset (gitignored, 332 MB)
├── pyproject.toml             # deps pineadas + metadata del proyecto
├── Makefile                   # targets atómicos para cada fase
└── README.md
```

---

## 2. `src/config.py` — constantes globales

**Por qué existe**: cualquier valor reproducible (seed, paths, ratios) vive en un solo lugar. Si cambia algo (e.g. el ratio de costo asimétrico), cambia en una sola línea y se propaga.

### Constantes definidas

| Constante | Valor | Uso |
|-----------|-------|-----|
| `SEED` | `42` | Fija aleatoriedad en numpy, random builtin, sklearn `random_state`, Optuna `seed`. |
| `ROOT_DIR` | `Path(__file__).resolve().parent.parent` | Raíz del repo, calculada del path del archivo. |
| `DATA_FILE` | `ROOT_DIR / "MLA_100k.jsonlines"` | Path absoluto al JSONL. |
| `LOADER_FILE` | `ROOT_DIR / "new_or_used.py"` | Para validar presencia. |
| `REPORTS_DIR`, `FIGURES_DIR`, `MODELS_DIR`, `NOTEBOOKS_DIR` | Sub-paths de `ROOT_DIR` | Crean automáticamente con `mkdir(exist_ok=True)`. |
| `TRAIN_SIZE` | `90_000` | Invariante del loader (assert lo verifica). |
| `TEST_SIZE` | `10_000` | Idem. |
| `ACCURACY_THRESHOLD` | `0.86` | Mínimo del enunciado, usado en `run_test.py` para validar. |
| `POSITIVE_LABEL` | `"new"` | Convención de clase positiva (consistente en todo el proyecto). |
| `NEGATIVE_LABEL` | `"used"` | Convención. |
| `CLASSES` | `("used", "new")` | Tupla ordenada para confusion_matrix. |
| `COST_FP` | `3.0` | Costo de FP en el ratio asimétrico 3:1 (ver REPORT.md §7). |
| `COST_FN` | `1.0` | Costo de FN. |
| `COST_RATIO` | `3.0` | Derivado, para reportes. |
| `OPTUNA_N_TRIALS` | `50` | Cap de trials de Optuna. |
| `OPTUNA_TIMEOUT_SECONDS` | `1800` | 30 min hard cap (regla del enunciado). |
| `CV_FOLDS` | `3` | Ahora no se usa (usamos 80/20 stratified); reservado para extensión. |

### Función pública

```python
def set_global_seed(seed: int = SEED) -> None:
    """Fija seed en numpy y random builtin."""
```

**Decisión**: NO seteamos seed para LightGBM/XGBoost en esta función — los pasamos explícitamente en cada llamada a `fit`. Esto evita efectos sorpresa entre librerías y deja claro dónde se controla la aleatoriedad de cada modelo.

---

## 3. `src/data.py` — wrapper sobre `build_dataset`

**Por qué existe**: la regla del enunciado dice "use `build_dataset`, do not load the file in a different way". Centralizamos la única invocación válida en un wrapper para:
1. Garantizar que ningún otro módulo invente otra forma de cargar.
2. Hacer `chdir(ROOT_DIR)` antes de llamarlo, porque `build_dataset` usa la ruta relativa `"MLA_100k.jsonlines"` y rompe si se invoca desde otra carpeta.
3. Restaurar el `cwd` original al terminar (pattern `try/finally`).

### Función pública

```python
def load_data() -> tuple[
    list[dict], list[str | None],
    list[dict], list[str | None]
]:
    """
    Returns
    -------
    X_train : list[dict]   90k listings con campo `condition`
    y_train : list[str]    etiquetas correspondientes
    X_test  : list[dict]   10k listings SIN el campo `condition`
    y_test  : list[str]    etiquetas reales del held-out
    """
```

**Implementación clave**:

```python
cwd_original = Path.cwd()
os.chdir(ROOT_DIR)
try:
    X_train, y_train, X_test, y_test = build_dataset()
finally:
    os.chdir(cwd_original)
```

**Verificación de invariantes** (asserts replicados del `__main__` del loader):

```python
assert len(X_train) == len(y_train) == 90_000
assert len(X_test) == len(y_test) == 10_000
assert all("condition" not in x for x in X_test)
assert set(y_train) <= {"new", "used", None}
```

Si alguno falla, sabemos que algo cambió en el archivo provisto y el resto del pipeline puede fallar silenciosamente.

---

## 4. `src/features.py` — flatten + ColumnTransformer

Tres responsabilidades:
1. Convertir la lista de dicts crudos del loader a un `DataFrame` con features derivadas.
2. Calcular `ref_date` para deltas temporales sin leakage.
3. Construir un `ColumnTransformer` configurable con tres ramas (numérica, categórica, texto).

### 4.1 `_flatten_one(r, ref_date)` — convierte UN listing en dict plano

Recibe el listing crudo (un dict con 48 campos potencialmente anidados) y produce un dict plano con ~43 features derivadas. Decisiones documentadas en `feature_catalog.md`.

**Categorías de features generadas**:

| Categoría | Ejemplo de features | Lógica |
|-----------|---------------------|--------|
| Texto | `title`, `title_len`, `has_subtitle`, `has_description` | Conteo de chars + flags binarios |
| Categóricas | `listing_type_id`, `buying_mode`, `category_id`, `currency_id`, `status`, `country_id`, `state_id`, `shipping_mode` | Acceso directo o anidado vía `_safe_get` |
| Booleanas | `accepts_mercadopago`, `automatic_relist`, `has_warranty`, `has_video`, `has_catalog_product`, `has_official_store`, `has_diff_pricing` | Dado el alto % de nulos en muchos campos, los flags binarios son la única señal explotable |
| Numéricas | `price_log = log1p(price)`, `base_price_log`, `discount_pct`, `available_quantity`, `initial_quantity`, `sold_quantity`, `sold_ratio = sold/initial`, `n_pictures`, `n_attributes`, `n_variations`, `n_payment_methods` | `log1p` para distribuciones sesgadas; ratios para normalización |
| Tags multi-hot | `tag_dragged_bids_and_visits`, `tag_good_quality_thumbnail`, `tag_dragged_visits`, `tag_free_relist`, `tag_poor_quality_thumbnail` | Top-5 por frecuencia (del EDA); el resto se ignora |
| Shipping | `free_shipping`, `local_pickup` | Extracción del dict anidado |
| Temporales | `listing_duration_days`, `listing_age_days`, `time_since_update_days` | Deltas relativos a `ref_date` |

**Por qué `log1p(price)` y no `price` raw**: la distribución de precio en MLA es muy sesgada (median=ARS 250, max millones). `log1p` estabiliza el rango, mejora LR (que es sensible a escala) y no perjudica al GBDT (invariante a transformaciones monótonas).

### 4.2 `compute_ref_date(records) -> datetime`

Calcula la fecha de referencia para los deltas temporales. **Regla crítica**: se computa siempre sobre el set de entrenamiento, nunca sobre el test, porque eso introduciría leakage temporal.

```python
def compute_ref_date(records: list[dict]) -> datetime:
    candidates = []
    for r in records:
        d = _parse_iso_date(r.get("last_updated"))
        if d is not None:
            candidates.append(d)
    if not candidates:
        return datetime(2015, 12, 31)  # fallback safe
    return max(candidates)
```

El `ref_date` se persiste en el `joblib` del modelo (`lgbm_best.joblib["ref_date"]`) y se reutiliza al evaluar test. Implementado en `run_test.py` línea ~50.

### 4.3 `flatten_records(records, ref_date=None) -> pd.DataFrame`

Loop sobre `_flatten_one` y construcción del DataFrame. Si `ref_date` es `None`, lo computa sobre los `records` provistos (uso típico: train). Para test, se pasa el `ref_date` calculado en train.

### 4.4 `build_preprocessor(...)` — ColumnTransformer con 3 ramas

```python
def build_preprocessor(
    *,
    scale_numeric: bool = True,
    text_max_features: int = 20_000,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
```

**Tres ramas**:

| Rama | Pipeline | Decisiones de diseño |
|------|----------|---------------------|
| **Numérica** | `SimpleImputer(strategy="median")` → opcional `StandardScaler(with_mean=True)` | Mediana porque robusta a outliers (precios extremos). Scaler solo cuando se usa LR (`scale_numeric=True`); para GBDT `False` porque GBDT es invariante a escala. |
| **Categórica** | `SimpleImputer(strategy="constant", fill_value="__missing__")` → `OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist", sparse_output=True)` | `min_frequency=50` evita que `category_id` (10.491 valores únicos) explote a 10k columnas. `handle_unknown="infrequent_if_exist"` permite que categorías nuevas en test caigan al bucket "infrequent" sin romper. |
| **Texto** (`title`) | `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=20000, min_df=5, sublinear_tf=True, strip_accents="unicode")` | `char_wb` (character n-grams within word boundaries) tolera mayúsculas, acentos, typos del español argentino. `min_df=5` filtra n-grams muy raros (probablemente ruido). `sublinear_tf=True` aplica log a la frecuencia de términos (estándar en TF-IDF). |

**Argumentos opcionales `numeric_cols` y `categorical_cols`**: agregados específicamente para soportar ablaciones (P1.6). Cuando se entrena la variante "sin `listing_type_id`", se pasa la lista de categóricas sin esa columna.

`remainder="drop"` y `sparse_threshold=0.3` son configuración estándar para ColumnTransformer cuando hay TF-IDF sparse — las features densas se densifican solo si menos del 30% del output es sparse, lo cual no se cumple aquí (hay 20k columnas TF-IDF), así que el output queda sparse.

### 4.5 `build_text_only_pipeline()`

Vectorizer aislado para el baseline LR text-only (`src/models/lr_text_only.py`). Misma config de TF-IDF pero sin ColumnTransformer; el pipeline solo tiene 2 pasos (TF-IDF → LR).

---

## 5. `src/eda/runner.py` — EDA en 4 bloques

**Función `main()`**: carga datos, ejecuta los 4 bloques, escribe `eda_findings.md` y `eda_findings.json`, genera 3 figuras.

### 5.1 `block_a_dataset_state(X_train, y_train, X_test, y_test) -> dict`

Estado del dataset:
- `Counter(y_train)` y `Counter(y_test)` — balance de clases
- Tasa de nulos por campo: itera los 48 campos, cuenta `_is_null(x.get(f))` sobre las 90k filas
- Cardinalidad de categóricas clave: `len(set(vals))` + `Counter().most_common(3)`

**Output**: figura `eda_class_balance.png` con dos barras (train, test) y los conteos absolutos.

### 5.2 `block_b_distribution_shift(X_train, y_train, X_test, y_test) -> dict`

Tres tests estadísticos para detectar shift train→test:

| Test | Sobre | Implementación | Lectura |
|------|-------|----------------|---------|
| KS (Kolmogorov-Smirnov) | `log1p(price)` | `scipy.stats.ks_2samp` | D, p-value |
| Chi² | `listing_type_id`, `buying_mode`, `currency_id`, `status` | `scipy.stats.chi2_contingency` con tabla 2×n | χ², dof, p-value |
| Chi² | Balance de clases | Idem, tabla 2×2 | Si p<0.05 → shift en clase |

**Output**: figura `eda_price_distribution_shift.png` con los dos histogramas superpuestos + la D del test.

### 5.3 `block_c_label_quality(X_train, y_train, X_test, y_test) -> dict`

Auditoría de calidad de label:
- Conteo de `condition=None` en train y test
- Filas con `available_quantity=0 ∧ sold_quantity>0` (inconsistencia lógica)
- Precios `≤0` y `0<price<10`
- **Verificación crítica**: búsqueda exhaustiva de `attributes` con `id == "ITEM_CONDITION"` (match estricto). El primer intento usaba substring loose `"condici" in name` y producía 1.405 falsos positivos (capturaba "Aire **acondici**onado"). Corregido a:

```python
aid = (a.get("id") or "").strip().upper()
aname_norm = (a.get("name") or "").strip().lower().replace("ó", "o")
if aid == "ITEM_CONDITION" or aname_norm in {
    "condicion", "condicion del item", ...
}:
```

Resultado real: **0 hits** sobre las 90k filas. Documentado en `eda_findings.md`.

### 5.4 `block_d_leakage(X_train, y_train) -> dict`

Cuantificación de leakage candidato:
- `permalink` con substring "usado"/"nuevo" → 516/3.231 filas
- `title` con keywords used-leaning ("usado", "permuto", "como nuevo", "antiguo") y new-leaning ("sellado", "en caja", "0km")
- `listing_type_id × condition` cross-tabulation (la feature dominante)
- `warranty no-null × condition` (45.25% new vs 32.09% used)
- `sold_quantity × condition` (mean new=4.25 vs used=0.10 — dirección inversa a la intuición)
- `tags` top-10 con %used por tag (free_relist → 96.14% used)

**Output**: figura `eda_listing_type_pct_used.png` con barras por listing tier.

### 5.5 `render_findings_md(a, b, c, d) -> str`

Toma los dicts de los 4 bloques y arma `eda_findings.md` en Markdown. Usa f-strings con formato numérico explícito (`.2f`, `.4g`) para evitar dependencia de pandas styler.

---

## 6. `src/eval/metrics.py` — métricas custom

**Convención clavada**: clase positiva = `"new"`, clase negativa = `"used"`. Todas las funciones respetan esta convención usando `pos_label=POSITIVE_LABEL` que viene de `config.py`.

### 6.1 `f05_new(y_true, y_pred) -> float`

```python
return float(fbeta_score(y_true, y_pred, beta=0.5, pos_label=POSITIVE_LABEL))
```

**β=0.5** pondera Precision al doble que Recall:

`F_β = (1 + β²) · (P · R) / (β² · P + R)`

Para β=0.5: `F_0.5 = 1.25 · (P · R) / (0.25 · P + R)`. La derivada parcial respecto a P es 4× la respecto a R, encodando matemáticamente la preferencia del costo asimétrico.

### 6.2 `precision_new`, `recall_new`, `recall_used`

Wrappers sobre sklearn con `pos_label` fijo. `recall_used` es matemáticamente equivalente a especificidad de `new`; existe como alias para usar el lenguaje natural de Trust & Safety en el reporte ("qué fracción de items usados detectamos").

### 6.3 `expected_cost(y_true, y_pred, cost_fp=3.0, cost_fn=1.0) -> float`

```python
cm = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
tn, fp, fn, tp = cm.ravel()
return (cost_fp * fp + cost_fn * fn) / n
```

Métrica de cross-check que opera en términos de costo, no de F-score. Si F_0.5 y expected_cost prefieren el mismo modelo/threshold, la decisión es robusta. Si discrepan, hay que investigar.

### 6.4 `metrics_report(y_true, y_pred) -> dict`

Reporta TODAS las métricas en un solo dict, incluyendo confusion matrix con nombres explícitos:

```python
"confusion_matrix": {
    "TN_used_used": int(tn),
    "FP_used_pred_new": int(fp),
    "FN_new_pred_used": int(fn),
    "TP_new_new": int(tp),
}
```

Los nombres explícitos evitan ambigüedad cuando se lee el JSON crudo (algunas implementaciones usan orden distinto de las clases).

### 6.5 `print_metrics(name, metrics) -> None`

Helper de impresión consistente, usado por todos los scripts. Formato fijo de columnas.

---

## 7. `src/models/lr_text_only.py` — baseline mínimo

Pipeline:
1. `TfidfVectorizer(char_wb (3,5), max_features=20k)` sobre solo `title`
2. `LogisticRegression(C=1.0, solver=liblinear, max_iter=1000)`

Entrenado con `sample_weight = where(y=="used", 3.0, 1.0)` — ratio 3:1 inyectado al fit.

**Por qué `solver=liblinear`**: maneja bien matrices sparse, no necesita L2 fija (acepta C≠1), y converge rápido para este tamaño. `lbfgs` también funcionaría pero es más lento sobre sparse.

**Output**: top-15 ngrams positivos (hacia "used") y negativos (hacia "new") con sus coeficientes. Eso es interpretabilidad de bajo costo:

```python
top_used = np.argsort(used_signal)[-15:][::-1]
top_new = np.argsort(used_signal)[:15]
```

Hallazgo clave: ngrams hacia "used" son palabras subjetivas (`'usad'`, `'como '`, `'antig'`); hacia "new" son especificaciones técnicas (`'ml '`, `'cc '`, `'mm '`, `'0w '`). Patrón interpretable, defendible en entrevista.

---

## 8. `src/models/lr_baseline.py` — LR full

Pipeline igual al text-only pero con el `ColumnTransformer` completo del `features.py`:

```python
Pipeline([
    ("preprocessor", build_preprocessor(scale_numeric=True)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear", random_state=SEED)),
])
```

`scale_numeric=True` aquí porque LR sí es sensible a escala. `class_weight=None` porque ya manejamos asimetría con `sample_weight`.

**Resultado** (val interno): accuracy=0.8621, F_0.5(new)=0.9081. **Ya pasa el umbral de 0.86 del enunciado** — eso es información: el problema es razonablemente lineal en el espacio de features que diseñamos. LightGBM va a aportar +3-4pp de mejora, no +10.

---

## 9. `src/models/lightgbm_model.py` — modelo principal

### 9.1 Pre-fit del preprocessor (clave para velocidad)

```python
pre = build_preprocessor(scale_numeric=False)  # GBDT no necesita scaling
X_tr = pre.fit_transform(df_tr)   # se hace UNA vez
X_va = pre.transform(df_va)
```

**Decisión arquitectónica**: pre-fittear el preprocessor fuera del loop de Optuna. Si lo metiéramos dentro del loop, cada trial recomputaría TF-IDF + OneHot — eso sería 50× más lento. Trade-off: técnicamente introducimos un leakage minúsculo (el preprocessor "ve" la val durante fit, pero solo aprende vocab y categorías frecuentes, no la etiqueta). Documentado.

### 9.2 `make_objective(X_tr, y_tr, X_va, y_va, w_tr)` — closure

Crea la función objetivo de Optuna cerrada sobre los splits ya fitteados. Cada trial:

1. Sugiere hiperparámetros del espacio de búsqueda definido
2. Entrena `LGBMClassifier(**params)` con `sample_weight=w_tr` y `early_stopping(50)`
3. Calcula `f05_new` sobre validación al threshold default (0.5 — el threshold operativo se tunea aparte en P1.7)
4. Retorna F_0.5 (Optuna maximiza)

**Espacio de búsqueda** (justificación de cada rango):

| Hiperparámetro | Rango | Por qué |
|----------------|-------|---------|
| `learning_rate` | `[0.01, 0.2]` log | Estándar; lr más alto requiere menos iters pero overfitea |
| `num_leaves` | `[31, 255]` log | Trade-off complexity vs overfit; LightGBM recomienda <2^max_depth |
| `max_depth` | `[5, 12]` | Profundidad razonable; >12 raramente útil sobre 90k filas |
| `min_child_samples` | `[5, 100]` log | Regularización por min samples per leaf |
| `feature_fraction` | `[0.6, 1.0]` | Subsample de features por árbol (anti-overfit) |
| `bagging_fraction` | `[0.6, 1.0]` | Subsample de filas por árbol |
| `reg_alpha` | `[1e-8, 10]` log | L1 regularization |
| `reg_lambda` | `[1e-8, 10]` log | L2 regularization |
| `n_estimators` | `2000` (fijo) | Cap alto; early_stopping decide cuándo cortar |

### 9.3 Optuna TPE con cap de tiempo

```python
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
)
study.optimize(objective, n_trials=50, timeout=1800)
```

**TPE (Tree-structured Parzen Estimator)** es más sample-efficient que random search para espacios continuos con dependencias. `seed=SEED` hace la búsqueda determinista (los mismos 50 trials se exploran si se re-corre).

`timeout=1800` (30 min) es el cap. Si los 50 trials no caben, el study se detiene cuando llega el timeout y guarda los trials completados. En la práctica corrió 50/50 en ~25 min.

### 9.4 Refit final + persistencia

Después del tuning, re-entrena con los mejores hiperparámetros sobre el mismo train interno (no sobre todo el train original — eso sería data leakage para la métrica de val). Persiste:

```python
joblib.dump({
    "preprocessor": pre,
    "model": best_model,
    "ref_date": ref,
    "best_params": study.best_params,
    "metrics_val": metrics,
}, MODELS_DIR / "lgbm_best.joblib")
```

El `ref_date` se guarda explícitamente para que la evaluación final pueda usarlo y los deltas temporales sean reproducibles entre train y test.

---

## 10. `src/models/xgboost_model.py` — comparativo (descartado en run final)

Misma pipeline de features pero con `XGBClassifier`. Hiperparámetros razonables sin tuning extensivo.

**Decisión TL: descartado del run final** porque la combinación XGBoost 3.x + matriz sparse de 20k features + `sample_weight` asimétrico + `early_stopping_rounds` = >40 min para un solo fit. Documentado en `REPORT.md` §4 y §11. El código queda en el repo como evidencia del intento (señal de rigor: lo intentamos, medimos, decidimos descartar).

---

## 11. `src/experiments/ablation_listing_type.py` — ablación estrella

Re-entrena el LightGBM con los mejores hiperparámetros (recuperados del joblib de P1.4) sobre **dos variantes**:

- **A (FULL)**: features completas
- **B (SIN listing_type_id)**: drop de la columna del DataFrame

### 11.1 `_train_eval(df_tr, df_va, y_tr, y_va, w_tr, best_params, label)`

Una sola función que se llama dos veces (variante A y B). Internamente:

```python
cols_present = set(df_tr.columns)
cat_cols_used = [c for c in CATEGORICAL_COLS if c in cols_present]
num_cols_used = [c for c in NUMERIC_COLS if c in cols_present]
pre = build_preprocessor(
    scale_numeric=False,
    numeric_cols=num_cols_used,
    categorical_cols=cat_cols_used,
)
```

Esa lógica adaptativa es el fix del bug que tuvimos: el primer intento dropeaba la columna pero el `build_preprocessor` seguía esperándola en `CATEGORICAL_COLS`. Ahora el preprocessor se construye con las columnas presentes en el DataFrame.

### 11.2 Tabla comparativa + veredicto auto-generado

Después del entrenamiento, calcula deltas:

```python
delta = {
    "accuracy": metrics_A["accuracy"] - metrics_B["accuracy"],
    "f05_new": metrics_A["f05_new"] - metrics_B["f05_new"],
    ...
}
```

Y un veredicto basado en la magnitud del drop:

```python
if delta_acc_pp <= 1.5:
    verdict = "modelo aprendió señal robusta"
elif delta_acc_pp >= 5.0:
    verdict = "modelo es proxy de tier — reportar honestamente"
else:
    verdict = f"caída intermedia ({delta_acc_pp:.2f}pp), defendible"
```

**Resultado real**: caída +4.33 pp → veredicto intermedio, defendible. Modelo retiene 95.2% del accuracy sin la feature dominante.

---

## 12. `src/eval/threshold.py` — barrido + curvas

### 12.1 Barrido de thresholds

```python
thresholds = np.arange(0.30, 0.71, 0.01)
results = []
for t in thresholds:
    y_pred_int = (proba_va >= t).astype(np.int8)
    y_pred_str = ["new" if v == 1 else "used" for v in y_pred_int]
    m = metrics_report(y_va_str, y_pred_str)
    m["threshold"] = float(t)
    results.append(m)
```

41 thresholds evaluados. Mejor por F_0.5(new): **0.56**. Comparado con el default 0.5 — gana +0.4 pp en F_0.5 y -0.0009 en expected_cost a costa de -0.6 pp en accuracy. Trade-off documentado.

### 12.2 Cuatro figuras en `reports/figures/`

| Figura | Qué muestra | Cómo se usa en entrevista |
|--------|-------------|---------------------------|
| `f05_vs_threshold.png` | F_0.5 + Accuracy + Expected cost vs threshold | "Mostrame por qué elegiste 0.56" → curva muestra el peak |
| `pr_curve_used.png` | Precision-Recall sobre clase `used` con PR-AUC | Threshold-independent view del modelo |
| `roc_curve_new.png` | ROC sobre clase `new` con AUC | Estándar; útil para comparar con modelos futuros |
| `confusion_matrix_optimal.png` | Heatmap de la confusion matrix al threshold óptimo | Visual rápido del balance FP vs FN |

### 12.3 Persistencia en el artefacto

```python
artifact["best_threshold"] = float(best["threshold"])
artifact["metrics_at_best_threshold"] = best
artifact["pr_auc_used"] = float(pr_auc)
artifact["roc_auc_new"] = float(roc_auc)
joblib.dump(artifact, artifact_path)
```

El threshold se guarda en el mismo joblib del modelo, no en un archivo aparte. Esto evita que `run_test.py` tenga que coordinar dos archivos para evaluar. Una sola fuente de verdad.

---

## 13. `src/eval/error_analysis.py` — top errores

### 13.1 Identificación de top errores

```python
top_fp = errors[errors["error_type"] == "FP"] \
    .sort_values("proba_new", ascending=False).head(50)
top_fn = errors[errors["error_type"] == "FN"] \
    .sort_values("proba_new", ascending=True).head(50)
```

**FP** se ordenan por `proba_new` descendente — los más confiados de "new" siendo "used" son los errores más graves. **FN** por `proba_new` ascendente — los más confiados de "used" siendo "new" son los más graves del otro lado.

### 13.2 CSV con campos completos

`reports/errors_top100.csv` con 100 filas y columnas: `title`, `category_id`, `listing_type_id`, `price`, `n_pictures`, `has_warranty`, `condition_real`, `proba_new`, `y_pred`, `error_type`, `len_title`, `correct`, `rank_position`. Permite que el reviewer haga análisis ad-hoc abriendo el CSV en Excel.

### 13.3 Patrones agregados

```python
fp_by_lt = errors[errors["error_type"] == "FP"].groupby("listing_type_id").size()
fn_by_lt = errors[errors["error_type"] == "FN"].groupby("listing_type_id").size()
fp_by_cat = errors[errors["error_type"] == "FP"].groupby("category_id").size().head(10)
fn_by_cat = errors[errors["error_type"] == "FN"].groupby("category_id").size().head(10)
```

**Hallazgo del run real**: errores se concentran en `bronze` (84% del volumen), proporcionalmente al volumen de bronze en el dataset. No hay sesgo categórico inesperado.

---

## 14. `src/eval/run_test.py` — eval final + REPORT generator

Dos responsabilidades:

### 14.1 Evaluación final sobre held-out X_test (UNA SOLA VEZ)

Carga el artefacto del modelo, transforma `X_test` con el preprocessor fitteado en train (no re-fittea), aplica el threshold operativo, calcula métricas:

```python
metrics = metrics_report(y_test, y_pred)
metrics["threshold_used"] = float(thr)
metrics["passes_086_threshold"] = bool(metrics["accuracy"] >= ACCURACY_THRESHOLD)
```

Resultado: **0.8910** sobre 10k filas nunca antes vistas durante tuning.

### 14.2 Generador de REPORT.md

`_render_report(test_metrics, eda, lr_full, lr_text, lgbm, sweep, ablation, thr)` toma los JSON de las fases anteriores y compone el reporte final con tablas en Markdown. **Render automático** = el REPORT siempre está sincronizado con los números actuales. Si re-corremos `make eval`, el reporte se regenera con los nuevos valores.

Helper `_safe(d, *keys, default="—")` para acceso seguro a dicts anidados — si una métrica no está disponible muestra "—" en lugar de crashear.

---

## 15. Cómo se conectan los módulos

```
   ┌──────────────────────────────────────────────────────────────┐
   │ new_or_used.py (provisto, NO MODIFICAR)                      │
   │     build_dataset() → (X_train, y_train, X_test, y_test)     │
   └────────────────────────┬─────────────────────────────────────┘
                            ↓
   ┌──────────────────────────────────────────────────────────────┐
   │ src/data.py                                                  │
   │     load_data() = wrapper estricto + asserts                 │
   └────────────────────────┬─────────────────────────────────────┘
                            ↓
   ┌──────────────────────────────────────────────────────────────┐
   │ src/features.py                                              │
   │     compute_ref_date(X_train) → ref_date                     │
   │     flatten_records(X, ref_date) → DataFrame                 │
   │     build_preprocessor(scale, cols) → ColumnTransformer      │
   └────┬──────────┬─────────┬─────────┬─────────┬───────────────┘
        ↓          ↓         ↓         ↓         ↓
    ┌───────┐  ┌──────┐  ┌────────┐  ┌─────┐  ┌──────────┐
    │ EDA   │  │ LR   │  │ LightGBM│ │ XGB │  │ Ablation │
    │ P1.1  │  │ P1.3 │  │ P1.4    │ │ P1.5│  │ P1.6     │
    └───┬───┘  └──┬───┘  └────┬────┘ └──┬──┘  └────┬─────┘
        │         │           │         │           │
        ↓         ↓           ↓         ↓           ↓
   reports/eda  reports/lr  models/   models/   reports/
                            lgbm_best xgboost   ablation
                            .joblib   .joblib
                            (artefacto principal)
                                 │
                                 ↓
        ┌────────────────────────┴───────────────────────┐
        ↓                                                ↓
   ┌────────────┐  ┌───────────────┐  ┌──────────────┐
   │ threshold  │  │ error_analysis│  │ run_test     │
   │ P1.7       │  │ P1.8          │  │ P1.9         │
   │ (actualiza │  │               │  │              │
   │ joblib con │  │               │  │ → REPORT.md  │
   │ best_thr)  │  │               │  │              │
   └────────────┘  └───────────────┘  └──────────────┘
```

**Invariantes**:
1. `X_test` solo se toca en `run_test.py` (la última caja).
2. `ref_date` se calcula en train y se persiste en el joblib.
3. Todos los módulos usan `SEED=42` desde `config.py`.
4. El threshold operativo se persiste en el mismo joblib del modelo.
5. Cada fase escribe su JSON en `reports/`; `run_test.py` los recompone en el REPORT.md final.

---

## 16. Dependencias clave (`pyproject.toml`)

| Paquete | Versión mínima | Por qué |
|---------|----------------|---------|
| `lightgbm` | `>=4.5.0` | Modelo principal. v4.5+ corrige bugs de sparse handling. |
| `xgboost` | `>=2.1.0` | Comparativo (descartado en run final, código queda como evidencia). |
| `scikit-learn` | `>=1.5.0` | Pipeline, ColumnTransformer, métricas, train_test_split. v1.5 introduce `min_frequency` en OneHotEncoder. |
| `pandas` | `>=2.2.0` | DataFrame de features. |
| `numpy` | `>=1.26.0` | Arrays, máscaras de sample_weight. |
| `optuna` | `>=4.0.0` | Hyperparameter tuning. v4 estabiliza la API. |
| `shap` | `>=0.46.0` | Instalado pero no usado en run final por tiempo. Reservado para next steps. |
| `scipy` | `>=1.13.0` | KS test, chi² test en EDA. |
| `matplotlib`, `seaborn` | `>=3.9`, `>=0.13` | Figuras. |
| `joblib` | `>=1.4.0` | Serialización de modelos. |
| `jupyter`, `ipykernel` | `>=1.0`, `>=6.29` | Notebook EDA. |

---

## 17. Cómo defender en la entrevista (cheat sheet)

Si te preguntan sobre **cualquier línea de código**:

1. **Identificá el archivo** según la sección de este documento.
2. **Buscá la subsección** que documenta esa función.
3. La respuesta a "¿por qué?" está en la columna "decisión de diseño" o "trade-off" de cada subsección.

Ejemplos de preguntas y sección donde está la respuesta:
- *"¿Por qué `log1p(price)` y no `price`?"* → sección 4.1
- *"¿Por qué pre-fitear el preprocessor antes de Optuna?"* → sección 9.1
- *"¿Por qué F_0.5 con β=0.5 y no β=1?"* → sección 6.1 + REPORT.md §7
- *"¿Por qué `solver=liblinear` en LR?"* → sección 7
- *"¿Por qué descartaste XGBoost?"* → sección 10 + REPORT.md §4 §11
- *"¿Cómo funciona la ablación?"* → sección 11
- *"¿Por qué 0.56 y no 0.5?"* → sección 12.1 + REPORT.md §8
- *"¿Cómo evitaste leakage temporal?"* → sección 4.2 + COMPLIANCE_AUDIT.md §5.3
