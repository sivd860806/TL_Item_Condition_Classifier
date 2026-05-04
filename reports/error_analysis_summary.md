# Analisis de errores — top-50 FP + top-50 FN

Generado al threshold optimo **0.560** (calibrado en P1.7) sobre la validacion
interna del split del 80/20.

## Resumen

| Tipo | Conteo | % de validacion |
|------|-------:|----------------:|
| Correctas | 16,092 | 89.40% |
| FP (predijo new, era used) | 289 | 1.61% |
| FN (predijo used, era new) | 1,619 | 8.99% |

**Lectura**: con el sample_weight 3:1 y threshold optimizado, el modelo es
conservador al afirmar "new". Esperamos FP < FN en numero absoluto, lo cual
deberia confirmarse arriba.

## FP por `listing_type_id` (donde el modelo se equivoca diciendo "new" siendo "used")

```
listing_type_id
bronze          235
silver           38
free              6
gold_special      5
gold              4
gold_premium      1
```

## FN por `listing_type_id` (donde el modelo es muy conservador y dice "used" siendo "new")

```
listing_type_id
bronze          1249
free             257
silver            89
gold              21
gold_premium       2
gold_special       1
```

## Top-10 categorias mas problematicas

**FP**:
```
category_id
MLA1227     6
MLA2044     6
MLA3530     4
MLA1383     3
MLA1643     3
MLA1893     3
MLA34375    3
MLA8290     3
MLA37748    3
MLA1070     2
```

**FN**:
```
category_id
MLA1227      76
MLA2044      28
MLA3530      21
MLA41287     19
MLA15171     15
MLA370638    10
MLA41259     10
MLA1474      10
MLA15226      9
MLA1468       9
```

## Patrones que vale la pena explorar con mas tiempo

- **Si los FP se concentran en `bronze` o `silver`**: el modelo confunde listings
  comerciales serios pero de productos usados (e.g. refurbs vendidos por tiendas).
  Mejora posible: feature de "es seller con multiples listings de la misma categoria".
- **Si los FN se concentran en `free`**: el modelo aprendio "free=used" demasiado
  fuerte. Mejora: regularizacion mas agresiva o ablacion de listing_type_id (P1.6).
- **Errores por longitud de titulo**: titulos muy cortos (<20 chars) suelen ser
  ambigos. Mejora: bucket `title_too_short` como feature.

## Detalle

CSV completo en `reports/errors_top100.csv` con columnas: `title`, `category_id`,
`listing_type_id`, `price`, `n_pictures`, `has_warranty`, `condition_real`,
`proba_new`, `y_pred`, `error_type`, `rank_position`.
