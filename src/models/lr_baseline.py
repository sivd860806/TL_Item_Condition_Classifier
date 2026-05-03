"""Baseline 1 — Logistic Regression sobre features tabulares + TF-IDF de title.

Es el número-piso que LightGBM debe superar. Se entrena con sample_weight
asimétrico (ratio 3:1) para alinear con la métrica secundaria.

Uso:
  python -m src.models.lr_baseline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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


def main() -> None:
    set_global_seed(SEED)
    t0 = time.time()

    print("Cargando dataset...")
    X_raw_train, y_train, X_raw_test, _ = load_data()

    # Filtrar etiquetas None del train (si las hubiera; el EDA confirmó 0)
    valid_train = [(x, y) for x, y in zip(X_raw_train, y_train) if y in {"new", "used"}]
    X_raw_train = [x for x, _ in valid_train]
    y_train = [y for _, y in valid_train]

    print(f"  X_train={len(X_raw_train)}  X_test={len(X_raw_test)}")

    # ref_date computado sobre train (no leakar el del test)
    ref = compute_ref_date(X_raw_train)
    print(f"  ref_date = {ref}")

    print("\nFlatten + features derivadas...")
    df_train = flatten_records(X_raw_train, ref_date=ref)
    print(f"  df_train shape = {df_train.shape}  cols={df_train.columns.tolist()[:8]}...")

    # Split estratificado interno 80/20 — el test del loader NO se toca acá
    X_tr, X_va, y_tr, y_va = train_test_split(
        df_train, y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=SEED,
    )
    print(f"  train interno = {len(X_tr)}   val interno = {len(X_va)}")

    # sample_weight asimétrico (ratio 3:1 — used pesa 3x para penalizar FP)
    sample_w = np.where(np.array(y_tr) == "used", COST_FP, COST_FN)
    print(f"  sample_weight: used={COST_FP}  new={COST_FN}")

    print("\nConstruyendo pipeline LR full...")
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight=None,  # ya manejamos asimetría con sample_weight
            max_iter=1000,
            solver="liblinear",
            random_state=SEED,
        )),
    ])

    print("Fitting...")
    t_fit = time.time()
    pipeline.fit(X_tr, y_tr, clf__sample_weight=sample_w)
    fit_seconds = time.time() - t_fit
    print(f"  fit en {fit_seconds:.1f}s")

    # Métricas sobre validación interna
    y_pred = pipeline.predict(X_va)
    metrics = metrics_report(y_va, y_pred)
    metrics["fit_seconds"] = fit_seconds
    metrics["n_train"] = len(X_tr)
    metrics["n_val"] = len(X_va)
    print_metrics("LR FULL — val interno (20% del train)", metrics)

    # Persistir resultados para el REPORT.md
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "lr_full.joblib"
    joblib.dump({"pipeline": pipeline, "ref_date": ref}, model_path)
    print(f"\nModelo guardado: {model_path}")

    metrics_path = REPORTS_DIR / "lr_full_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"Métricas guardadas: {metrics_path}")

    print(f"\nTiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
