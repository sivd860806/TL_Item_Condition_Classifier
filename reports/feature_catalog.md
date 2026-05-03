# Catálogo de features — decisiones campo por campo

Documento de referencia que justifica, para cada uno de los 48 campos del listing crudo, **si entra al pipeline de modelado, cómo, y por qué**. Cada decisión está alineada con los hallazgos del EDA (ver `eda_findings.md`) y con la regla de "no leakage" del enunciado.

**Convenciones**:
- **drop**: campo no entra al modelo
- **keep raw**: el valor original se usa tal cual como feature
- **derive**: se computa una o más features a partir del campo original
- **target**: variable a predecir (no es feature)

---

## Identificadores y URLs

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `id` | drop | Identificador único del listing; no generaliza. Dejarlo sería memorizar. |
| `parent_item_id` | drop | Identificador de padre en jerarquía de variantes; no generaliza. |
| `permalink` | drop | Contiene 'usado' en 516 filas y 'nuevo' en 3.231 filas (3.6% del train) → **leakage parcial confirmado**. Drop obligatorio. |
| `thumbnail`, `secure_thumbnail` | drop | URLs de imágenes. Sin contenido textual útil sin descargar la imagen. |
| `pictures` | derive `n_pictures` | El conteo es señal de seller serio; las URLs en sí no aportan sin CV. |
| `seller_id` | drop | Identidad del vendedor → riesgo alto de memorización. Si lo dejáramos, el modelo aprendería "este seller siempre vende usado", lo cual no generaliza a sellers nuevos. |
| `seller_address` | derive `country_id`, `state_id` | Geografía a nivel país/provincia es señal útil. El barrio (`city`) se descarta por alta cardinalidad y memorización. |
| `geolocation`, `location` | drop | Redundante con `seller_address`; `location` además tiene 97.8% de nulos. |

---

## Texto

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `title` | TF-IDF char_wb (3,5) max_features=20.000 | El EDA muestra 1.624 hits 'used-leaning' ('uso', 'usado', 'permuto', 'como nuevo') y 557 hits 'new-leaning' ('sellado', 'en caja', '0km'). Char-grams toleran mayúsculas, acentos y typos del español argentino. Cap a 20k features evita explotar dimensionalidad. |
| `subtitle` | derive `has_subtitle` | 100% de nulos en train; el flag binario es lo único informativo. |
| `descriptions` | derive `has_description` | El campo es una lista; el contenido en sí está vacío en la mayoría de casos. Solo el flag de presencia. |

---

## Categóricas (entran a OneHot)

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `listing_type_id` | OneHot — **feature dominante** | EDA: spread brutal de %used desde 1.93% (`gold_special`) hasta 92.59% (`free`). NO es leakage técnico (lo elige el seller al listar) pero es prácticamente un clasificador por sí solo → **ablación P1.6 obligatoria**. |
| `buying_mode` | OneHot | 3 categorías (`buy_it_now`, `classified`, `auction`); bajo cardinal, defendible sin transformación. |
| `currency_id` | OneHot | 2 categorías (`ARS` 99.4%, `USD` 0.6%); USD puede correlacionar con productos premium nuevos. |
| `category_id` | OneHot con `min_frequency=50` + "infrequent_if_exist" | 10.491 categorías únicas. Sin frequency cap explotaría a >10k columnas, agarrando ruido. Las raras se agrupan en bucket "infrequent". |
| `site_id` | drop | Constante: `MLA` en el 100% del train. No aporta señal. |
| `status` | OneHot | 4 categorías (`active`, `paused`, `closed`, otros); estados como `paused` pueden correlacionar con listings problemáticos. |
| `international_delivery_mode` | OneHot | Bajo cardinal; flag de comercio internacional. |
| `automatic_relist` | keep raw (bool) | Señal directa de listing repetitivo, típico de productos nuevos a escala. |

---

## Numéricas (entran al pipeline numérico)

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `price` | derive `price_log = log1p(price)` + flag `was_zero` | Distribución muy sesgada (median ARS 250, max millones). `log1p` estabiliza el rango. Para GBDT no cambia mucho, para LR sí; unificamos pipeline. |
| `base_price` | derive `base_price_log` | Casi idéntico a `price` en mayoría de casos; lo dejamos como feature secundaria. |
| `original_price` | derive `has_discount`, `discount_pct = 1 - price/original_price` | 99.86% de nulos; el flag de presencia y el % de descuento son lo único explotable. |
| `available_quantity` | keep raw | Inventario disponible. |
| `initial_quantity` | keep raw | Inventario al momento de publicar. |
| `sold_quantity` | keep raw | EDA: new mean=4.25 vs used mean=0.10 → señal fuerte (dirección **inversa** a la intuición común; los nuevos se re-listean a escala). |
| Derivada | `sold_ratio = sold_quantity / max(initial_quantity, 1)` | Tasa de venta, normalizada por inventario inicial. |

---

## Listas (multi-hot o conteos)

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `tags` | multi-hot top-5 (`dragged_bids_and_visits`, `good_quality_thumbnail`, `dragged_visits`, `free_relist`, `poor_quality_thumbnail`) | EDA muestra `free_relist` → 96.14% used; `dragged_visits` → 74.97% used. Señal robusta. |
| `attributes` | derive `n_attributes` | EDA verificó: 0 hits con `id == "ITEM_CONDITION"` (match estricto). Sin leakage por este lado. El conteo es señal de listing detallado. |
| `variations` | derive `n_variations` | Listings con variaciones (tallas/colores) tienden a ser productos nuevos en cadena. |
| `non_mercado_pago_payment_methods` | derive `n_payment_methods` | Conteo de medios de pago alternativos; señal de listing rico. |
| `coverage_areas` | drop | 100% nulos. |
| `deal_ids` | drop | 99.76% nulos. |
| `sub_status` | drop | 99.01% nulos. |

---

## Booleanas y flags

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `accepts_mercadopago` | keep raw | Señal de seller que pasa el filtro de Meli; correlaciona con tier de listing. |
| `automatic_relist` | keep raw (ya en categóricas arriba) | — |
| `warranty` | derive `has_warranty` | EDA: 45.25% new con warranty vs 32.09% used → señal real, defendible. |

---

## Fechas

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `start_time` | derive `listing_duration_days = stop_time - start_time` (junto con stop_time) | Duración intencional del listing. |
| `stop_time` | derive (combinada con `start_time`) | — |
| `date_created` | derive `listing_age_days = ref_date - date_created` | Antigüedad del listing al momento del snapshot. |
| `last_updated` | derive `time_since_update_days = ref_date - last_updated` | Frescura del último cambio. |

`ref_date` se computa una vez como el `max(last_updated)` del train; queda fijo para el test, evitando data leakage temporal.

---

## Anidadas (dict)

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `shipping` | derive `free_shipping` (bool), `shipping_mode` (cat), `local_pickup` (bool) | El logística es señal de seller serio; multi-feature simple. |
| `differential_pricing` | derive `has_diff_pricing` | 100% nulos; solo flag de presencia. |
| `seller_contact` | drop | 97.8% nulos; el contenido (números/emails) sería leakage de identidad. |
| `seller_address.country.id` | derive `country_id` | Categórica geográfica. |
| `seller_address.state.id` | derive `state_id` | Categórica geográfica. |

---

## Identificadores opacos / metadata

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `catalog_product_id` | derive `has_catalog_product` | 99.99% nulos; flag de presencia indica catálogo oficial. |
| `official_store_id` | derive `has_official_store` | 99.17% nulos; flag indica tienda oficial. |
| `video_id` | derive `has_video` | 97.03% nulos; flag indica listing rico en media. |
| `listing_source` | drop | 100% nulos (constante o vacío). |

---

## Target

| Campo | Decisión | Justificación |
|-------|----------|---------------|
| `condition` | **target** (no feature) | Variable a predecir: 'new' (positivo) vs 'used' (negativo). |

---

## Resumen del catálogo

| Tipo de transformación | # campos crudos | # features finales (aprox) |
|------------------------|-----------------|----------------------------|
| Drop                   | ~12             | 0                          |
| Keep raw (numéricas)   | ~6              | ~6                         |
| Keep raw (booleanas)   | ~3              | ~3                         |
| OneHot                 | ~7              | ~50                        |
| Multi-hot              | ~1 (tags)       | ~5                         |
| Derive (flags)         | ~10             | ~10                        |
| Derive (numéricas)     | ~6              | ~6                         |
| TF-IDF char_wb (title) | 1               | hasta 20.000 (sparse)      |

**Total estimado**: ~80 features densas + hasta 20k features sparse de TF-IDF. La sparse matrix resultante tiene una fracción muy baja de no-ceros, lo cual es manejable por LightGBM/XGBoost/LR sin problemas.

---

## Riesgos identificados (a defender en entrevista)

1. **`listing_type_id` puede dominar**: el modelo podría aprender "free → used" antes que la señal real del producto. Mitigación: ablación P1.6 (con/sin este campo) reportada explícitamente.
2. **`sold_quantity` es proxy de seller a escala**: correlaciona con identidad del seller (que dropeamos). Vale la pena monitorearla en SHAP del análisis de errores P1.8.
3. **TF-IDF char_wb puede sobreajustar a typos específicos**: el `min_df` implícito y el cap de 20k features ayudan; vale verificar feature importance del LR text-only para detectar n-grams ruidosos.
4. **`category_id` con 10.491 valores únicos**: el `min_frequency=50` los reduce a ~150 categorías reales + "infrequent". Defendible por el cap de dimensionalidad.
