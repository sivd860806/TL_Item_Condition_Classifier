"""Ablacion estrella — modelo CON vs SIN `listing_type_id`.

Pregunta de negocio que responde:
  "El modelo aprendio 'new vs used' o solo 'pricing tier de Meli'?"

Diseno:
  - Cargar los hiperparametros optimos del LightGBM tuneado (P1.4).
  - Re-entrenar dos veces sobre el mismo split, con misma semilla:
      A) Full features
      B) Full features menos `listing_type_id`
  - Reportar tabla con: accuracy, F_0.5(new), recall(used), expected_cost.

Lectura:
  - Delta accuracy <= 1.5pp -> el modelo aprendio senal robusta. Defendible.
  - Delta accuracy >= 5pp  -> es proxy de tier; documentar honestamente.

Uso:
  python -m src.experiments.ablation_listing_type
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split

from ..config import (
    COST_FN,
    COST_FP,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    set_global_seed,
)
from ..data import load_data
from ..eval.metrics import metrics_report, print_metrics
from ..features import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    build_preprocessor,
    compute_ref_date,
    flatten_records,
)


def _y_to_int(y: list[str]) -> np.ndarray:
    return np.array([1 if v == "new" else 0 for v in y], dtype=np.int8)


def _y_to_str(y_int: np.ndarray) -> list[str]:
    return ["new" if v == 1 else "used" for v in y_int]


def _train_eval(
    df_tr, df_va, y_tr, y_va, w_tr, best_params: dict, label: str,
):
    """Entrena LightGBM con los mejores params sobre (df_tr, df_va) y reporta.

    El preprocessor se construye usando solo las columnas que existen en df_tr,
    para soportar ablaciones que dropean columnas (e.g. listing_type_id).
    """
    cols_present = set(df_tr.columns)
    cat_cols_used = [c for c in CATEGORICAL_COLS if c in cols_present]
    num_cols_used = [c for c in NUMERIC_COLS if c in cols_present]
    pre = build_preprocessor(
        scale_numeric=False,
        numeric_cols=num_cols_used,
        categorical_cols=cat_cols_used,
    )
    X_tr = pre.fit_transform(df_tr)
    X_va = pre.transform(df_va)

    params = {
        "n_estimators": 2000,
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": SEED,
        "verbosity": -1,
        "n_jobs": -1,
        "bagging_freq": 5,
        **best_params,
    }
    t0 = time.time()
    clf = LGBMClassifier(**params)
    clf.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[early_stopping(50, verbose=False)],
    )
    fit_s = time.time() - t0

    proba = clf.predict_proba(X_va)[:, 1]
    y_pred_str = _y_to_str((proba >= 0.5).astype(np.int8))
    m = metrics_report(_y_to_str(y_va), y_pred_str)
    m["fit_seconds"] = fit_s
    m["best_iter"] = int(clf.best_iteration_) if clf.best_iteration_ else None
    m["n_features"] = X_tr.shape[1]
    print_metrics(label, m)
    return m


def main() -> None:
    set_global_seed(SEED)

    # 1) Recuperar mejores hiperparametros del LightGBM tuneado (P1.4)
    artifact_path = MODELS_DIR / "lgbm_best.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No existe {artifact_path}. Corre primero `make train` (P1.4)."
        )
    artifact = joblib.load(artifact_path)
    best_params = artifact["best_params"]
    print(f"Hiperparametros recuperados de P1.4:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 2) Cargar y splitear igual que en P1.4 (mismo seed -> mismo split)
    print("\nCargando dataset...")
    X_raw, y_str, _, _ = load_data()
    valid = [(x, y) for x, y in zip(X_raw, y_str) if y in {"new", "used"}]
    X_raw = [x for x, _ in valid]
    y_str = [y for _, y in valid]

    ref = compute_ref_date(X_raw)
    df_full = flatten_records(X_raw, ref_date=ref)

    df_tr, df_va, y_tr_str, y_va_str = train_test_split(
        df_full, y_str,
        test_size=0.20,
        stratify=y_str,
        random_state=SEED,
    )
    y_tr = _y_to_int(y_tr_str)
    y_va = _y_to_int(y_va_str)
    w_tr = np.where(y_tr == 0, COST_FP, COST_FN).astype(np.float32)

    # 3) Variante A: full features
    print("\n" + "=" * 70)
    print("Variante A: FULL features (incluyendo listing_type_id)")
    print("=" * 70)
    metrics_A = _train_eval(
        df_tr, df_va, y_tr, y_va, w_tr, best_params,
        "A) FULL features",
    )

    # 4) Variante B: drop listing_type_id
    print("\n" + "=" * 70)
    print("Variante B: SIN listing_type_id")
    print("=" * 70)
    df_tr_B = df_tr.drop(columns=["listing_type_id"])
    df_va_B = df_va.drop(columns=["listing_type_id"])
    metrics_B = _train_eval(
        df_tr_B, df_va_B, y_tr, y_va, w_tr, best_params,
        "B) SIN listing_type_id",
    )

    # 5) Tabla comparativa + escritura del reporte
    delta = {
        "accuracy": metrics_A["accuracy"] - metrics_B["accuracy"],
        "f05_new": metrics_A["f05_new"] - metrics_B["f05_new"],
        "recall_used": metrics_A["recall_used"] - metrics_B["recall_used"],
        "expected_cost_3to1": metrics_B["expected_cost_3to1"]
                              - metrics_A["expected_cost_3to1"],
    }
    print("\n" + "=" * 70)
    print("DELTA (A - B): cuanto cae el modelo al sacar listing_type_id")
    print("=" * 70)
    for k, v in delta.items():
        sign = "+" if v >= 0 else ""
        print(f"  {k:<24s} {sign}{v * 100:.3f}pp"
              if k != "expected_cost_3to1"
              else f"  {k:<24s} {sign}{v:.4f}")

    # 6) Render markdown del reporte
    delta_acc_pp = delta["accuracy"] * 100
    if delta_acc_pp <= 1.5:
        verdict = (
            "**Caida de accuracy <=1.5pp**: el modelo aprendio senal robusta mas "
            "alla del pricing tier de Meli. Defendible en produccion."
        )
    elif delta_acc_pp >= 5.0:
        verdict = (
            f"**Caida de accuracy >={delta_acc_pp:.1f}pp >= 5pp**: el modelo es "
            "esencialmente un proxy del listing tier. Hay que reportarlo "
            "honestamente y proponer feature engineering adicional."
        )
    else:
        verdict = (
            f"**Caida intermedia ({delta_acc_pp:.2f}pp)**: el modelo depende "
            "moderadamente de listing_type_id. La feature aporta senal pero el "
            "modelo no es solo un proxy del tier; el resto de features hace trabajo."
        )

    md = f"""# Ablacion: con vs sin `listing_type_id`

Re-entrenamiento del LightGBM tuneado (mismos hiperparametros, mismo split,
misma semilla) sobre dos variantes:
- **A**: full features (incluyendo listing_type_id)
- **B**: full features menos listing_type_id

## Resultados

| Metrica | A (full) | B (sin listing_type_id) | Delta (A - B) |
|---------|---------:|------------------------:|--------------:|
| Accuracy | {metrics_A['accuracy']:.4f} | {metrics_B['accuracy']:.4f} | {delta['accuracy'] * 100:+.3f} pp |
| F_0.5 (new) | {metrics_A['f05_new']:.4f} | {metrics_B['f05_new']:.4f} | {delta['f05_new'] * 100:+.3f} pp |
| Recall (used) | {metrics_A['recall_used']:.4f} | {metrics_B['recall_used']:.4f} | {delta['recall_used'] * 100:+.3f} pp |
| Expected cost (3:1) | {metrics_A['expected_cost_3to1']:.4f} | {metrics_B['expected_cost_3to1']:.4f} | {-delta['expected_cost_3to1']:+.4f} |
| n_features tras preprocessor | {metrics_A['n_features']} | {metrics_B['n_features']} | {metrics_A['n_features'] - metrics_B['n_features']} |
| best_iter LightGBM | {metrics_A['best_iter']} | {metrics_B['best_iter']} | — |

## Confusion matrices

**A (full)**:
- TN={metrics_A['confusion_matrix']['TN_used_used']}
- FP={metrics_A['confusion_matrix']['FP_used_pred_new']}
- FN={metrics_A['confusion_matrix']['FN_new_pred_used']}
- TP={metrics_A['confusion_matrix']['TP_new_new']}

**B (sin listing_type_id)**:
- TN={metrics_B['confusion_matrix']['TN_used_used']}
- FP={metrics_B['confusion_matrix']['FP_used_pred_new']}
- FN={metrics_B['confusion_matrix']['FN_new_pred_used']}
- TP={metrics_B['confusion_matrix']['TP_new_new']}

## Veredicto

{verdict}

## Como interpretarlo en la entrevista

`listing_type_id` es la feature dominante segun el EDA (spread de %used desde
1.93% en `gold_special` hasta 92.59% en `free`). NO es leakage tecnico (lo
elige el seller al listar, no depende de la etiqueta).

Pero el riesgo era que el modelo se apoyara casi exclusivamente en este tier
y aprendiera "vendedor pago listing premium -> producto nuevo" en vez de la
senal real del producto.

Esta tabla cuantifica esa dependencia con un experimento ablation y permite
afirmar con numeros si el modelo es defendible en produccion.
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "ablation_listing_type.md"
    out.write_text(md, encoding="utf-8")
    print(f"\nReporte: {out}")

    # JSON de respaldo
    payload = {
        "A_full": metrics_A,
        "B_no_listing_type": metrics_B,
        "delta": delta,
        "verdict_summary": verdict,
    }
    (REPORTS_DIR / "ablation_listing_type.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
