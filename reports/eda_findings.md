# EDA — Hallazgos clave

Generado por `src/eda/runner.py`. Reproducible vía `make eda`.

Toda métrica computada sobre `X_train` (90k filas) salvo aclaración.

## A. Estado del dataset

- **Tamaños**: train = 90,000 filas, test = 10,000 filas.
- **Balance de clases (train)**: new = 48,352 (53.72%), used = 41,648 (46.28%).
- **Balance de clases (test)**: new = 5,406 (54.06%), used = 4,594 (45.94%).
- **Top-15 campos por tasa de nulos**:

| Campo | % nulos |
|-------|---------|
| `listing_source` | 100.00% |
| `coverage_areas` | 100.00% |
| `differential_pricing` | 100.00% |
| `subtitle` | 100.00% |
| `catalog_product_id` | 99.99% |
| `original_price` | 99.86% |
| `deal_ids` | 99.76% |
| `official_store_id` | 99.17% |
| `sub_status` | 99.01% |
| `seller_contact` | 97.80% |
| `location` | 97.80% |
| `video_id` | 97.03% |
| `variations` | 91.81% |
| `attributes` | 87.61% |
| `warranty` | 60.84% |

Figura: `reports/figures/eda_class_balance.png`.

## B. Distribution shift train vs test

- **KS test sobre log1p(price)**: D = 0.0122, p = 0.1374.
- **Chi² sobre balance de clases train vs test**: χ² = 0.3943, p = 0.5301 → sin shift relevante.
- **Chi² sobre categóricas**:

| Campo | χ² | dof | p |
|-------|-----|-----|---|
| `listing_type_id` | 14.64 | 6 | 0.02324 |
| `buying_mode` | 1.51 | 2 | 0.4701 |
| `currency_id` | 0.66 | 1 | 0.4155 |
| `status` | 2.63 | 3 | 0.4519 |

**Implicancia**: si algún p<0.05 con efecto material, documentar que la validación interna debe usar últimos 10–20% del train (mimic temporal) en lugar de KFold puro.

Figura: `reports/figures/eda_price_distribution_shift.png`.

## C. Auditoría de calidad de label

- `condition=None` en train: **0**; en test: **0**.
- `condition` fuera de {new, used, None}: **0**.
- Filas con `available_quantity=0 ∧ sold_quantity>0`: **0** (0.00% del train).
- `price ≤ 0`: **0**.  `0 < price < 10`: **406**.
- Verificación global de `attributes` con `id == "ITEM_CONDITION"` (match estricto, sin tildes y sin substring loose): **0** filas. El leakage típico de `attributes` codificando la etiqueta NO existe en este snapshot. (Una primera versión del filtro usaba `"condici" in name` y producía 1405 falsos positivos por capturar "Aire **acondici**onado".)

## D. Leakage candidates cuantificados

### D.1 `permalink`
- Contiene 'usado': **516** filas. Contiene 'nuevo': **3231** filas. **Drop obligatorio**.

### D.2 `title` — palabras predictoras

Keywords used-leaning:

| keyword | hits |
|---------|------|
| `uso` | 610 |
| `usado` | 516 |
| `como nuevo` | 194 |
| `permuto` | 171 |
| `detalle` | 101 |
| `permuta` | 30 |
| `negociable` | 1 |
| `estado:` | 1 |

Keywords new-leaning:

| keyword | hits |
|---------|------|
| `en caja` | 233 |
| `sellado` | 194 |
| `0km` | 110 |
| `nuevo sin uso` | 20 |

**Implicancia**: el título es señal léxica fuerte. TF-IDF char_wb va a capturar esto sin ingeniería manual.

### D.3 `listing_type_id` × condition  (FEATURE DOMINANTE)

| listing_type_id | total | new | used | %used |
|-----------------|-------|-----|------|-------|
| `bronze` | 56,904 | 35,410 | 21,494 | 37.77% |
| `free` | 19,260 | 1,428 | 17,832 | 92.59% |
| `silver` | 8,195 | 6,575 | 1,620 | 19.77% |
| `gold_special` | 2,693 | 2,641 | 52 | 1.93% |
| `gold` | 2,170 | 1,869 | 301 | 13.87% |
| `gold_premium` | 765 | 416 | 349 | 45.62% |
| `gold_pro` | 13 | 13 | 0 | 0.00% |

**Implicancia**: `listing_type_id` es prácticamente un clasificador por sí solo (spread de %used entre tiers). NO es leakage técnico (lo elige el seller al listar), pero hay que correr la **ablación P1.6 con/sin este campo** para responder *"¿el modelo aprendió new-vs-used o solo pricing tier?"*.

Figura: `reports/figures/eda_listing_type_pct_used.png`.

### D.4 `warranty`
- new con warranty: **45.25%** (21,877/48,352)
- used con warranty: **32.09%** (13,366/41,648)
- Señal real, defendible. Derivar `has_warranty: bool`.

### D.5 `sold_quantity`
- new : mean = **4.25**, median = 0
- used: mean = **0.10**, median = 0
- **Dirección inversa a la intuición común**: new tiene mucho más sold_quantity por re-listeo masivo. Feature útil pero hay que documentar que correlaciona con identidad de seller (no usar `seller_id` por leakage).

### D.6 `tags` top-10

| tag | total | %used |
|-----|-------|-------|
| `dragged_bids_and_visits` | 66,516 | 46.37% |
| `good_quality_thumbnail` | 1,537 | 11.32% |
| `dragged_visits` | 723 | 74.97% |
| `free_relist` | 259 | 96.14% |
| `poor_quality_thumbnail` | 13 | 0.00% |

**Implicancia**: tags son señal de *listing* (cómo se publica), no del producto. Multi-hot top-10 es low-cost y captura señal robusta.

## Resumen ejecutivo (para el REPORT.md)

- Dataset balanceado (~54/46); métrica enfocada en costo asimétrico (no en imbalance) está justificada.
- Sin distribution shift relevante en clase entre train y test (p = 0.53); KFold puro sobre train es defendible para la métrica primaria. Hay shift menor en `listing_type_id` (p = 0.023) que vale la pena monitorear pero no obliga a cambiar la estrategia de CV.
- Leakage real cuantificado: drop `permalink`, `id`, `seller_id`, `thumbnail`, `secure_thumbnail`, `pictures` (URLs).
- **Atributos con `ITEM_CONDITION` codificada: 0 hits** (verificación global con match estricto). El leakage típicamente "más obvio" no aplica a este snapshot — vale la pena escribirlo en el reporte para demostrar rigor verificado, no asumido.
- `listing_type_id` es la feature dominante (spread de %used desde 1.93% en `gold_special` hasta 92.59% en `free`); ablación P1.6 obligatoria para defender el modelo en entrevista.
- `sold_quantity` tiene dirección **inversa a la intuición común**: new mean = 4.25, used mean = 0.10 (re-listeo masivo de productos nuevos por sellers de escala).
- TF-IDF char_wb sobre `title` capturará palabras predictoras (1624 hits used + 557 hits new sobre 90k filas).
