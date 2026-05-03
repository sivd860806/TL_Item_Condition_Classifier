"""Modelo principal — LightGBM con Optuna + early stopping.

Diseño:
  1. Pre-fittear el preprocessor (sin scaling — GBDT lo ignora) sobre el
     train interno (80% de X_train del loader).
  2. Optuna TPE con N trials sobre un single 80/20 split de validación.
     Optimiza F_0.5 sobre clase 'new' calculado al threshold 0.5.
     (El threshold operativo se tunea aparte en P1.7.)
  3. Early stopping=50 dentro de cada trial sobre la validación.
  4. Sample_weight asimétrico 3:1 (used:new) — ratio de costo inyectado al fit.
  5. Persistir mejor modelo + estudio Optuna + métricas.

Uso normal (8h budget, 50 trials, ~20–30 min):
  python -m src.models.lightgbm_model

Uso rápido (smoke test):
  python -m src.models.lightgbm_model --n-trials 3 --timeout 60
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from ..config import (
    COST_FN,
    COST_FP,
    MODELS_DIR,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SECONDS,
    REPORTS_DIR,
    SEED,
    set_global_seed,
)
from ..data import load_data
from ..eval.metrics import metrics_report, print_metrics
from ..features import build_preprocessor, compute_ref_date, flatten_records


# Silenciar logs verbose de LightGBM y Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _y_to_int(y: list[str]) -> np.ndarray:
    """'new' -> 1, 'used' -> 0 (clase positiva = new)."""
    return np.array([1 if v == "new" else 0 for v in y], dtype=np.int8)


def _y_to_str(y_int: np.ndarray) -> list[str]:
    return ["new" if v == 1 else "used" for v in y_int]


def _f05_at_default_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """F_0.5 sobre clase 'new' al threshold 0.5 (durante tuning).
    El threshold operativo se tunea en P1.7."""
    y_pred = (proba >= 0.5).astype(np.int8)
    return float(fbeta_score(y_true, y_pred, beta=0.5, pos_label=1))


def make_objective(X_tr, y_tr, X_va, y_va, w_tr):
    """Construye la función objetivo de Optuna cerrada sobre los splits."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "objective": "binary",
            "metric": "binary_logloss",
            "random_state": SEED,
            "verbosity": -1,
            "n_jobs": -1,
        }
        clf = LGBMClassifier(**params)
        clf.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(50, verbose=False)],
        )
        proba_va = clf.predict_proba(X_va)[:, 1]  # P(new)
        return _f05_at_default_threshold(y_va, proba_va)

    return objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=OPTUNA_N_TRIALS,
                        help=f"Número de trials Optuna (default {OPTUNA_N_TRIALS})")
    parser.add_argument("--timeout", type=int, default=OPTUNA_TIMEOUT_SECONDS,
                        help=f"Timeout en segundos (default {OPTUNA_TIMEOUT_SECONDS})")
    parser.add_argument("--no-final-refit", action="store_true",
                        help="Saltarse el refit final con todo el train (smoke test)")
    args = parser.parse_args()

    set_global_seed(SEED)
    t0 = time.time()

    print("Cargando dataset...")
    X_raw_train, y_train_raw, X_raw_test, _ = load_data()
    valid = [(x, y) for x, y in zip(X_raw_train, y_train_raw) if y in {"new", "used"}]
    X_raw_train = [x for x, _ in valid]
    y_train_raw = [y for _, y in valid]
    print(f"  X_train={len(X_raw_train)}  X_test={len(X_raw_test)}")

    # ref_date computado sobre el train completo
    ref = compute_ref_date(X_raw_train)
    print(f"  ref_date = {ref}")

    print("\nFlatten + features derivadas...")
    df_full = flatten_records(X_raw_train, ref_date=ref)
    print(f"  df shape = {df_full.shape}")

    # Split estratificado 80/20 — el test del loader NUNCA se toca
    df_tr, df_va, y_tr_str, y_va_str = train_test_split(
        df_full, y_train_raw,
        test_size=0.20,
        stratify=y_train_raw,
        random_state=SEED,
    )
    y_tr = _y_to_int(y_tr_str)
    y_va = _y_to_int(y_va_str)
    print(f"  train interno = {len(df_tr)}   val interno = {len(df_va)}")

    # Preprocessor — fitteado UNA vez sobre train interno (sin scaling para GBDT)
    print("\nFitting preprocessor (sin scaling)...")
    pre = build_preprocessor(scale_numeric=False)
    X_tr = pre.fit_transform(df_tr)
    X_va = pre.transform(df_va)
    print(f"  X_tr={X_tr.shape}  X_va={X_va.shape}  density={X_tr.nnz / (X_tr.shape[0] * X_tr.shape[1]) if hasattr(X_tr, 'nnz') else 'dense'}")

    # Sample weights asimétricos (used pesa 3.0, new pesa 1.0)
    w_tr = np.where(y_tr == 0, COST_FP, COST_FN).astype(np.float32)
    print(f"  sample_weight: used={COST_FP}  new={COST_FN}")

    # Optuna TPE
    print(f"\nOptuna TPE — n_trials={args.n_trials}  timeout={args.timeout}s")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="lightgbm_f05",
    )
    objective = make_objective(X_tr, y_tr, X_va, y_va, w_tr)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                    show_progress_bar=False)

    print(f"\nMejor F_0.5(new) val: {study.best_value:.4f}")
    print(f"Mejor params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Refit con los mejores hiperparámetros
    if not args.no_final_refit:
        print("\nRefit con los mejores hiperparámetros sobre train interno (80%)...")
    else:
        print("\n(saltando refit final por --no-final-refit)")

    best_params = {
        "n_estimators": 2000,
        "objective": "binary",
        "metric": "binary_logloss",
        "random_state": SEED,
        "verbosity": -1,
        "n_jobs": -1,
        "bagging_freq": 5,
        **study.best_params,
    }
    best_model = LGBMClassifier(**best_params)
    best_model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[early_stopping(50, verbose=False)],
    )

    # Métricas finales sobre validación interna
    proba_va = best_model.predict_proba(X_va)[:, 1]
    y_pred_str = _y_to_str((proba_va >= 0.5).astype(np.int8))
    metrics = metrics_report(y_va_str, y_pred_str)
    metrics["best_iter"] = int(best_model.best_iteration_) if best_model.best_iteration_ else None
    metrics["best_params"] = study.best_params
    metrics["n_trials_completed"] = len(study.trials)
    metrics["fit_seconds_total"] = time.time() - t0
    print_metrics("LightGBM TUNED — val interno (threshold=0.5)", metrics)

    # Persistir
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Modelo + preprocessor + ref_date — todo lo que se necesita para inferencia
    artifact_path = MODELS_DIR / "lgbm_best.joblib"
    joblib.dump({
        "preprocessor": pre,
        "model": best_model,
        "ref_date": ref,
        "best_params": study.best_params,
        "metrics_val": metrics,
    }, artifact_path)
    print(f"\nArtefacto guardado: {artifact_path}")

    metrics_path = REPORTS_DIR / "lgbm_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"Métricas guardadas: {metrics_path}")

    study_path = REPORTS_DIR / "optuna_study.json"
    study_data = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params,
              "duration_s": t.duration.total_seconds() if t.duration else None}
            for t in study.trials
        ],
    }
    study_path.write_text(json.dumps(study_data, indent=2, default=str), encoding="utf-8")
    print(f"Estudio Optuna: {study_path}")

    print(f"\nTiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
