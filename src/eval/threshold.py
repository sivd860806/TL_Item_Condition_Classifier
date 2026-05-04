"""Threshold tuning + metrica secundaria final (P1.7).

Por que esto importa:
- El modelo emite P(new) en [0,1]. El threshold de 0.5 rara vez es optimo bajo
  costo asimetrico (FP:FN = 3:1).
- Tuneamos el threshold sobre validacion para maximizar F_0.5(new), que es la
  metrica secundaria elegida.
- El threshold optimo es un hiperparametro operacional documentado.

Genera:
- reports/threshold_sweep.json    todas las metricas por threshold
- reports/figures/f05_vs_threshold.png
- reports/figures/pr_curve_used.png
- reports/figures/roc_curve_new.png
- reports/figures/confusion_matrix_optimal.png

Uso:
  python -m src.eval.threshold
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from ..config import (
    COST_FN,
    COST_FP,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    set_global_seed,
)
from ..data import load_data
from ..eval.metrics import metrics_report, print_metrics
from ..features import flatten_records


def _y_to_int(y: list[str]) -> np.ndarray:
    return np.array([1 if v == "new" else 0 for v in y], dtype=np.int8)


def _save_fig(fig, name: str):
    p = FIGURES_DIR / f"{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    set_global_seed(SEED)
    t0 = time.time()

    # 1) Cargar artefacto del LightGBM tuneado
    artifact_path = MODELS_DIR / "lgbm_best.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"No existe {artifact_path}. Corre primero `make train` (P1.4)."
        )
    artifact = joblib.load(artifact_path)
    pre = artifact["preprocessor"]
    model = artifact["model"]
    ref = artifact["ref_date"]
    print(f"Modelo cargado de {artifact_path}")

    # 2) Replicar split exacto de P1.4 (mismo seed)
    print("\nCargando dataset y replicando split de P1.4...")
    X_raw, y_str, _, _ = load_data()
    valid = [(x, y) for x, y in zip(X_raw, y_str) if y in {"new", "used"}]
    X_raw = [x for x, _ in valid]
    y_str = [y for _, y in valid]

    df_full = flatten_records(X_raw, ref_date=ref)
    _, df_va, _, y_va_str = train_test_split(
        df_full, y_str,
        test_size=0.20,
        stratify=y_str,
        random_state=SEED,
    )
    y_va = _y_to_int(y_va_str)
    print(f"  validacion interna: n={len(df_va)}")

    # 3) Predict probabilidades
    X_va = pre.transform(df_va)
    proba_va = model.predict_proba(X_va)[:, 1]  # P(new)
    print(f"  proba range: [{proba_va.min():.3f}, {proba_va.max():.3f}]  mean={proba_va.mean():.3f}")

    # 4) Barrer thresholds
    print("\nBarrido de thresholds [0.30, 0.70] step 0.01...")
    thresholds = np.arange(0.30, 0.71, 0.01)
    results = []
    for t in thresholds:
        y_pred_int = (proba_va >= t).astype(np.int8)
        y_pred_str = ["new" if v == 1 else "used" for v in y_pred_int]
        m = metrics_report(y_va_str, y_pred_str)
        m["threshold"] = float(t)
        results.append(m)

    # 5) Mejor threshold por F_0.5(new)
    best_idx = int(np.argmax([r["f05_new"] for r in results]))
    best = results[best_idx]
    print(f"\nMejor threshold por F_0.5(new): {best['threshold']:.2f}")
    print_metrics(f"AL THRESHOLD OPTIMO {best['threshold']:.2f}", best)

    # Comparacion con threshold default 0.5
    default_idx = int(np.argmin(np.abs(thresholds - 0.5)))
    default = results[default_idx]
    print("\nComparacion threshold default (0.5) vs optimo:")
    print(f"  threshold=0.50  acc={default['accuracy']:.4f}  F_0.5={default['f05_new']:.4f}  cost={default['expected_cost_3to1']:.4f}")
    print(f"  threshold={best['threshold']:.2f}  acc={best['accuracy']:.4f}  F_0.5={best['f05_new']:.4f}  cost={best['expected_cost_3to1']:.4f}")

    # 6) Persistir resultados del barrido
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sweep_path = REPORTS_DIR / "threshold_sweep.json"
    sweep_path.write_text(json.dumps({
        "best_threshold": best["threshold"],
        "best_metrics": best,
        "default_metrics_at_0.5": default,
        "sweep": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nSweep guardado: {sweep_path}")

    # 7) FIGURAS
    print("\nGenerando figuras...")

    # 7.1 F_0.5(new) vs threshold
    fig, ax = plt.subplots(figsize=(8, 4))
    f05s = [r["f05_new"] for r in results]
    accs = [r["accuracy"] for r in results]
    costs = [r["expected_cost_3to1"] for r in results]
    ax.plot(thresholds, f05s, marker="o", label="F_0.5 (new)", color="#4C72B0")
    ax.plot(thresholds, accs, marker="s", label="Accuracy", color="#55A868", alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(thresholds, costs, marker="^", linestyle="--",
             label="Expected cost (3:1)", color="#C44E52", alpha=0.6)
    ax2.set_ylabel("Expected cost (3:1)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax.axvline(best["threshold"], color="black", linestyle=":",
                label=f"Optimo ({best['threshold']:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F_0.5(new) / Accuracy")
    ax.set_title("Threshold sweep — F_0.5(new), Accuracy y Expected Cost")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower center")
    fig.tight_layout()
    p = _save_fig(fig, "f05_vs_threshold")
    print(f"  {p}")

    # 7.2 PR curve sobre clase 'used' (negativa)
    # Para PR sobre 'used' invertimos: y=1 si used, score = 1 - P(new)
    y_used = (y_va == 0).astype(np.int8)
    score_used = 1 - proba_va
    prec_u, rec_u, _ = precision_recall_curve(y_used, score_used)
    pr_auc = auc(rec_u, prec_u)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_u, prec_u, color="#4C72B0", linewidth=2)
    ax.fill_between(rec_u, prec_u, alpha=0.15, color="#4C72B0")
    ax.set_xlabel("Recall (used)")
    ax.set_ylabel("Precision (used)")
    ax.set_title(f"PR curve — clase 'used'  (PR-AUC = {pr_auc:.4f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = _save_fig(fig, "pr_curve_used")
    print(f"  {p}")

    # 7.3 ROC curve sobre clase 'new'
    fpr, tpr, _ = roc_curve(y_va, proba_va)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#55A868", linewidth=2,
             label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5,
             label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC curve — clase 'new'")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = _save_fig(fig, "roc_curve_new")
    print(f"  {p}")

    # 7.4 Confusion matrix al threshold optimo
    y_pred_opt = (proba_va >= best["threshold"]).astype(np.int8)
    cm = confusion_matrix(y_va, y_pred_opt, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["used", "new"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["used", "new"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ threshold={best['threshold']:.2f}\n"
                 f"acc={best['accuracy']:.4f}  F_0.5(new)={best['f05_new']:.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = _save_fig(fig, "confusion_matrix_optimal")
    print(f"  {p}")

    # 8) Persistir threshold optimo en el artefacto del modelo
    artifact["best_threshold"] = float(best["threshold"])
    artifact["metrics_at_best_threshold"] = best
    artifact["pr_auc_used"] = float(pr_auc)
    artifact["roc_auc_new"] = float(roc_auc)
    joblib.dump(artifact, artifact_path)
    print(f"\nArtefacto actualizado con best_threshold={best['threshold']:.2f}")

    print(f"\nTiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
