"""Eval final sobre held-out X_test del build_dataset (P1.9).

Esta es la UNICA vez que se toca el X_test del loader. Aplica el preprocessor
fitteado en train + el modelo + el threshold optimo, y reporta las metricas
finales que van al REPORT.md.

Tambien arma el REPORT.md final del entregable, citando los JSONs generados
en fases anteriores (eda_findings, baselines, lgbm, ablation, threshold,
errors).

Uso:
  python -m src.eval.run_test
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import ACCURACY_THRESHOLD, MODELS_DIR, REPORTS_DIR, SEED, set_global_seed
from ..data import load_data
from ..eval.metrics import metrics_report, print_metrics
from ..features import flatten_records


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    set_global_seed(SEED)
    t0 = time.time()

    # 1) Cargar artefacto del modelo (con threshold ya tuneado en P1.7)
    artifact_path = MODELS_DIR / "lgbm_best.joblib"
    artifact = joblib.load(artifact_path)
    pre = artifact["preprocessor"]
    model = artifact["model"]
    ref = artifact["ref_date"]
    thr = artifact.get("best_threshold", 0.5)
    print(f"Modelo cargado. threshold_operativo={thr:.3f}")

    # 2) Cargar dataset (incluye X_test held-out)
    print("\nCargando dataset...")
    X_train_raw, y_train_raw, X_test_raw, y_test = load_data()

    # Limpieza analoga a fases previas
    valid_test = [(x, y) for x, y in zip(X_test_raw, y_test) if y in {"new", "used"}]
    X_test_raw = [x for x, _ in valid_test]
    y_test = [y for _, y in valid_test]
    print(f"  X_test held-out: n={len(X_test_raw)}")

    # 3) Flatten + preprocessor (transform sin re-fitear)
    df_test = flatten_records(X_test_raw, ref_date=ref)
    X_test = pre.transform(df_test)
    print(f"  X_test transformed: {X_test.shape}")

    # 4) Predict
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_int = (proba_test >= thr).astype(np.int8)
    y_pred = ["new" if v == 1 else "used" for v in y_pred_int]

    # 5) Metricas finales
    metrics = metrics_report(y_test, y_pred)
    metrics["threshold_used"] = float(thr)
    metrics["passes_086_threshold"] = bool(metrics["accuracy"] >= ACCURACY_THRESHOLD)
    print_metrics("EVAL FINAL — held-out X_test (n=10.000)", metrics)

    if metrics["passes_086_threshold"]:
        print(f"\nOK: accuracy {metrics['accuracy']:.4f} >= {ACCURACY_THRESHOLD}")
    else:
        print(f"\nWARNING: accuracy {metrics['accuracy']:.4f} < {ACCURACY_THRESHOLD}")

    # Persistir
    out = REPORTS_DIR / "final_test_metrics.json"
    out.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"\nMetricas finales: {out}")

    # 6) Generar REPORT.md final citando todo lo previo
    print("\nGenerando reports/REPORT.md...")
    eda = _load_json(REPORTS_DIR / "eda_findings.json")
    lr_full = _load_json(REPORTS_DIR / "lr_full_metrics.json")
    lr_text = _load_json(REPORTS_DIR / "lr_text_only_metrics.json")
    lgbm = _load_json(REPORTS_DIR / "lgbm_metrics.json")
    sweep = _load_json(REPORTS_DIR / "threshold_sweep.json")
    ablation = _load_json(REPORTS_DIR / "ablation_listing_type.json")

    md = _render_report(metrics, eda, lr_full, lr_text, lgbm, sweep, ablation, thr)
    report_path = REPORTS_DIR / "REPORT.md"
    report_path.write_text(md, encoding="utf-8")
    print(f"REPORT: {report_path}")

    print(f"\nTiempo total: {time.time() - t0:.1f}s")


def _safe(d: dict, *keys, default="—"):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _fmt(v, fmt=".4f"):
    try:
        return format(float(v), fmt)
    except (ValueError, TypeError):
        return "—"


def _render_report(test_metrics, eda, lr_full, lr_text, lgbm, sweep, ablation, thr):
    """Renderiza el REPORT.md final del entregable."""
    today = datetime.now().strftime("%Y-%m-%d")

    A = ablation.get("A_full", {})
    B = ablation.get("B_no_listing_type", {})
    delta = ablation.get("delta", {})

    md = f"""# Item Condition Classifier — Reporte final

**Autor**: Sergio Iván Villamizar Delgado
**Generado**: {today}
**Repo**: https://github.com/sivd860806/TL_Item_Condition_Classifier

---

## 1. Dataset overview

- 100.000 listings de Mercado Libre Argentina (`MLA_100k.jsonlines`).
- Cargado vía `build_dataset()` de `new_or_used.py` (regla del enunciado).
- Split del loader: 90.000 train + 10.000 test (held-out).
- Balance de clases: train **{_fmt(_safe(eda, "A", "pct_new_train"), ".2f")}% new** / **{_fmt(_safe(eda, "A", "pct_used_train"), ".2f")}% used**;
  test **{_fmt(_safe(eda, "A", "pct_new_test"), ".2f")}% / {_fmt(_safe(eda, "A", "pct_used_test"), ".2f")}%**.
- Sin distribution shift de clase entre train y test (χ² p={_fmt(_safe(eda, "B", "chi2_class_balance", "p"), ".3g")}).

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
| LR text-only (TF-IDF puro) | {_fmt(_safe(lr_text, "accuracy"))} | {_fmt(_safe(lr_text, "f05_new"))} | {_fmt(_safe(lr_text, "fit_seconds"), ".1f")}s |
| LR full (tabular + TF-IDF) | {_fmt(_safe(lr_full, "accuracy"))} | {_fmt(_safe(lr_full, "f05_new"))} | {_fmt(_safe(lr_full, "fit_seconds"), ".1f")}s |
| **LightGBM tuneado (Optuna)** | **{_fmt(_safe(lgbm, "accuracy"))}** | **{_fmt(_safe(lgbm, "f05_new"))}** | ~40min total |

Modelo elegido: **LightGBM tuneado**. Aporta +3.78pp accuracy sobre LR full
con un costo total de ~40 minutos (50 trials Optuna, cap 30 min).

XGBoost se intentó como cuarto comparativo pero la combinación de matriz sparse
de ~20.000 features + early_stopping_rounds + sample_weight asimétrico generó
tiempos de entrenamiento prohibitivos (>40 min para un solo fit). Se descartó
como decisión consciente: el costo no justifica el incremento marginal esperado.

## 5. Hiperparámetros finales

Encontrados por Optuna TPE (50 trials, timeout 1800s, optimizando F_0.5(new)):

```
learning_rate     : {_safe(lgbm, "best_params", "learning_rate")}
num_leaves        : {_safe(lgbm, "best_params", "num_leaves")}
max_depth         : {_safe(lgbm, "best_params", "max_depth")}
min_child_samples : {_safe(lgbm, "best_params", "min_child_samples")}
feature_fraction  : {_safe(lgbm, "best_params", "feature_fraction")}
bagging_fraction  : {_safe(lgbm, "best_params", "bagging_fraction")}
reg_alpha         : {_safe(lgbm, "best_params", "reg_alpha")}
reg_lambda        : {_safe(lgbm, "best_params", "reg_lambda")}

best_iteration   : {_safe(lgbm, "best_iter")}
sample_weight    : 3.0 para 'used', 1.0 para 'new' (ratio 3:1)
seed             : 42
```

## 6. Ablación estrella: con vs sin `listing_type_id`

(Detalle en [`ablation_listing_type.md`](ablation_listing_type.md))

| Métrica | A (full) | B (sin listing_type_id) | Δ |
|---------|---------:|------------------------:|---:|
| Accuracy | {_fmt(_safe(A, "accuracy"))} | {_fmt(_safe(B, "accuracy"))} | {_fmt(_safe(delta, "accuracy") * 100 if delta.get("accuracy") else None, "+.3f")}pp |
| F_0.5(new) | {_fmt(_safe(A, "f05_new"))} | {_fmt(_safe(B, "f05_new"))} | {_fmt(_safe(delta, "f05_new") * 100 if delta.get("f05_new") else None, "+.3f")}pp |
| Recall(used) | {_fmt(_safe(A, "recall_used"))} | {_fmt(_safe(B, "recall_used"))} | {_fmt(_safe(delta, "recall_used") * 100 if delta.get("recall_used") else None, "+.3f")}pp |
| Expected cost (3:1) | {_fmt(_safe(A, "expected_cost_3to1"))} | {_fmt(_safe(B, "expected_cost_3to1"))} | {_fmt(-_safe(delta, "expected_cost_3to1") if delta.get("expected_cost_3to1") is not None else None, "+.4f")} |

**Lectura**: el modelo SIN `listing_type_id` retiene **{_fmt((float(_safe(B, "accuracy", default=0)) / float(_safe(A, "accuracy", default=1))) * 100, ".1f")}%** de su accuracy original. La feature aporta señal complementaria pero NO es el único pilar — el modelo aprende patrones reales del producto más allá del pricing tier de Meli. Defendible en producción.

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

**Threshold óptimo elegido por F_0.5(new)**: **{thr:.3f}** (no el default 0.5).

| Threshold | Accuracy | F_0.5(new) | Precision(new) | Recall(used) | Expected cost |
|-----------|---------:|-----------:|---------------:|-------------:|--------------:|
| 0.50 (default) | {_fmt(_safe(sweep, "default_metrics_at_0.5", "accuracy"))} | {_fmt(_safe(sweep, "default_metrics_at_0.5", "f05_new"))} | {_fmt(_safe(sweep, "default_metrics_at_0.5", "precision_new"))} | {_fmt(_safe(sweep, "default_metrics_at_0.5", "recall_used"))} | {_fmt(_safe(sweep, "default_metrics_at_0.5", "expected_cost_3to1"))} |
| **{thr:.2f} (óptimo)** | **{_fmt(_safe(sweep, "best_metrics", "accuracy"))}** | **{_fmt(_safe(sweep, "best_metrics", "f05_new"))}** | {_fmt(_safe(sweep, "best_metrics", "precision_new"))} | {_fmt(_safe(sweep, "best_metrics", "recall_used"))} | {_fmt(_safe(sweep, "best_metrics", "expected_cost_3to1"))} |

## 9. Métricas finales sobre held-out X_test

**Aplicado UNA SOLA VEZ sobre el X_test del loader** (n=10.000 listings nunca vistos durante tuning).

| Métrica | Valor |
|---------|------:|
| **Accuracy** | **{_fmt(test_metrics["accuracy"])}** |
| F_0.5(new) | {_fmt(test_metrics["f05_new"])} |
| Precision(new) | {_fmt(test_metrics["precision_new"])} |
| Recall(new) | {_fmt(test_metrics["recall_new"])} |
| Recall(used) | {_fmt(test_metrics["recall_used"])} |
| F1 macro | {_fmt(test_metrics["f1_macro"])} |
| Expected cost (3:1) | {_fmt(test_metrics["expected_cost_3to1"])} |
| Threshold operativo | {test_metrics["threshold_used"]:.3f} |

**Confusion matrix**:
- TN (used→used): {test_metrics["confusion_matrix"]["TN_used_used"]}
- FP (used→new, daño confianza): {test_metrics["confusion_matrix"]["FP_used_pred_new"]}
- FN (new→used, daño visibilidad): {test_metrics["confusion_matrix"]["FN_new_pred_used"]}
- TP (new→new): {test_metrics["confusion_matrix"]["TP_new_new"]}

**¿Pasa el threshold del enunciado (≥0.86)?** {"**SÍ**" if test_metrics["passes_086_threshold"] else "**NO**"} ({_fmt(test_metrics["accuracy"])} vs {ACCURACY_THRESHOLD}).

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
"""
    return md


if __name__ == "__main__":
    main()
