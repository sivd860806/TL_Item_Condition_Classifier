"""Baseline 2 — Logistic Regression solo sobre TF-IDF de title.

Aisla la contribucion de la senal textual. Si este modelo solo ya supera el
0.80 de accuracy, demuestra que el titulo es muy informativo (los usados se
delatan lexicamente: 'usado', 'permuto', 'como nuevo').

Sirve como argumento defendible cuando el reviewer pregunte:
  "por que no usaste solo TF-IDF si era suficiente?"
La respuesta es: lo medi, y aporta X pp menos que el modelo full.

Uso:
  python -m src.models.lr_text_only
"""
from __future__ import annotations

import json
import time

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
from ..features import build_text_only_pipeline


def main() -> None:
    set_global_seed(SEED)
    t0 = time.time()

    print("Cargando dataset...")
    X_raw_train, y_train, _, _ = load_data()

    valid = [(x, y) for x, y in zip(X_raw_train, y_train) if y in {"new", "used"}]
    X_raw_train = [x for x, _ in valid]
    y_train = [y for _, y in valid]

    # Solo necesitamos el title
    titles = [x.get("title") or "" for x in X_raw_train]
    print(f"  n = {len(titles)}  (titulo medio = {np.mean([len(t) for t in titles]):.1f} chars)")

    # Split estratificado interno
    t_tr, t_va, y_tr, y_va = train_test_split(
        titles, y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=SEED,
    )

    sample_w = np.where(np.array(y_tr) == "used", COST_FP, COST_FN)

    print("\nConstruyendo pipeline LR text-only (TF-IDF char_wb (3,5))...")
    pipeline = Pipeline([
        ("tfidf", build_text_only_pipeline(text_max_features=20_000)),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear",
            random_state=SEED,
        )),
    ])

    print("Fitting...")
    t_fit = time.time()
    pipeline.fit(t_tr, y_tr, clf__sample_weight=sample_w)
    fit_seconds = time.time() - t_fit
    print(f"  fit en {fit_seconds:.1f}s")

    y_pred = pipeline.predict(t_va)
    metrics = metrics_report(y_va, y_pred)
    metrics["fit_seconds"] = fit_seconds
    metrics["n_train"] = len(t_tr)
    metrics["n_val"] = len(t_va)
    print_metrics("LR TEXT-ONLY — val interno", metrics)

    # Top features informativas (interpretabilidad para la entrevista)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feat_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]  # binary classifier; coef hacia clase 'used'
    classes = list(clf.classes_)
    used_idx = classes.index("used")
    sign = 1 if used_idx == 1 else -1
    used_signal = sign * coefs

    top_used = np.argsort(used_signal)[-15:][::-1]
    top_new = np.argsort(used_signal)[:15]
    print("\nTop-15 ngrams hacia 'used':")
    for i in top_used:
        print(f"  {feat_names[i]!r:<30} coef={used_signal[i]:+.3f}")
    print("\nTop-15 ngrams hacia 'new':")
    for i in top_new:
        print(f"  {feat_names[i]!r:<30} coef={used_signal[i]:+.3f}")

    metrics["top_used_ngrams"] = [(str(feat_names[i]), float(used_signal[i]))
                                    for i in top_used]
    metrics["top_new_ngrams"] = [(str(feat_names[i]), float(used_signal[i]))
                                   for i in top_new]

    # Persist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "lr_text_only.joblib")
    (REPORTS_DIR / "lr_text_only_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nMetricas guardadas: {REPORTS_DIR / 'lr_text_only_metrics.json'}")
    print(f"Tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
