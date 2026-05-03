"""Modelo comparativo — XGBoost SIN tuning extensivo.

Razon de ser: cumplir el bonus del enunciado ("compare at least two model
families") sin gastar tiempo de tuning. Hiperparametros razonables fijos,
mismo pipeline de features, mismo sample_weight asimetrico.

Lo que se va a comparar en el reporte:
  - Accuracy y F_0.5(new) (esperamos ~igual a LightGBM tuneado o levemente abajo)
  - Tiempo de entrenamiento (XGBoost suele ser mas lento)
  - Tamano del modelo serializado

Uso:
  python -m src.models.xgboost_model
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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
from ..features import build_preprocessor, compute_ref_date, flatten_records


def _y_to_int(y: list[str]) -> np.ndarray:
    return np.array([1 if v == "new" else 0 for v in y], dtype=np.int8)


def _y_to_str(y_int: np.ndarray) -> list[str]:
    return ["new" if v == 1 else "used" for v in y_int]


def main() -> None:
    set_global_seed(SEED)
    t0 = time.time()

    print("Cargando dataset...")
    X_raw_train, y_train_raw, _, _ = load_data()
    valid = [(x, y) for x, y in zip(X_raw_train, y_train_raw) if y in {"new", "used"}]
    X_raw_train = [x for x, _ in valid]
    y_train_raw = [y for _, y in valid]

    ref = compute_ref_date(X_raw_train)
    df_full = flatten_records(X_raw_train, ref_date=ref)

    df_tr, df_va, y_tr_str, y_va_str = train_test_split(
        df_full, y_train_raw,
        test_size=0.20,
        stratify=y_train_raw,
        random_state=SEED,
    )
    y_tr = _y_to_int(y_tr_str)
    y_va = _y_to_int(y_va_str)

    print("Fitting preprocessor (sin scaling)...")
    pre = build_preprocessor(scale_numeric=False)
    X_tr = pre.fit_transform(df_tr)
    X_va = pre.transform(df_va)
    print(f"  X_tr={X_tr.shape}  X_va={X_va.shape}")

    # Hiperparametros razonables sin tuning (justificados en REPORT.md)
    params = dict(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=1000,
        scale_pos_weight=1.0 / COST_FP,  # XGBoost interpreta el ratio invertido a sample_weight
        # Nota: usamos sample_weight directo para que el calibrado coincida con LightGBM
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
    )

    # XGBoost interpreta clase positiva = 1 (new); FP = predijo 1 siendo 0 (used).
    # Para penalizar FP usamos sample_weight en used (clase 0) = COST_FP, new = COST_FN.
    # Por eso desactivamos scale_pos_weight (=1) y usamos sample_weight directo.
    params["scale_pos_weight"] = 1.0
    w_tr = np.where(y_tr == 0, COST_FP, COST_FN).astype(np.float32)

    print("\nFitting XGBoost (sin tuning, 1000 iters max, early stop 50)...")
    t_fit = time.time()
    clf = XGBClassifier(**params)
    clf.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    fit_seconds = time.time() - t_fit
    print(f"  fit en {fit_seconds:.1f}s  best_iter={clf.best_iteration}")

    proba_va = clf.predict_proba(X_va)[:, 1]
    y_pred = _y_to_str((proba_va >= 0.5).astype(np.int8))
    metrics = metrics_report(y_va_str, y_pred)
    metrics["fit_seconds"] = fit_seconds
    metrics["best_iter"] = int(clf.best_iteration)
    metrics["params"] = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    print_metrics("XGBoost (no tuning) — val interno (threshold=0.5)", metrics)

    # Persistir
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODELS_DIR / "xgboost_baseline.joblib"
    joblib.dump({
        "preprocessor": pre,
        "model": clf,
        "ref_date": ref,
        "metrics_val": metrics,
    }, artifact_path)
    size_mb = os.path.getsize(artifact_path) / (1024 * 1024)
    metrics["model_size_mb"] = round(size_mb, 2)
    print(f"\nArtefacto guardado: {artifact_path}  ({size_mb:.2f} MB)")

    metrics_path = REPORTS_DIR / "xgboost_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"Metricas guardadas: {metrics_path}")
    print(f"\nTiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
