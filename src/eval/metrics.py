"""Métricas custom alineadas con el argumento de negocio del REPORT.

Convención (consistente en todo el proyecto):
- Clase positiva: 'new'
- Clase negativa: 'used'

Por la asimetría de costo (un comprador engañado pesa más que un seller
con menos visibilidad), F_0.5 sobre 'new' es la métrica secundaria. La
métrica de cross-check es expected_cost con ratio 3:1 (FP:FN).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

from ..config import COST_FN, COST_FP, NEGATIVE_LABEL, POSITIVE_LABEL


# -----------------------------------------------------------------------------
# Métrica secundaria principal
# -----------------------------------------------------------------------------
def f05_new(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """F_β=0.5 sobre clase 'new'. Pondera precision al doble que recall."""
    return float(fbeta_score(y_true, y_pred, beta=0.5, pos_label=POSITIVE_LABEL))


# -----------------------------------------------------------------------------
# Métricas auxiliares por clase
# -----------------------------------------------------------------------------
def precision_new(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))


def recall_new(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))


def recall_used(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Equivalente matemáticamente a especificidad sobre 'new'.
    Es la formulación natural en lenguaje de Trust & Safety:
    qué fracción de items realmente usados detectamos correctamente."""
    return float(recall_score(y_true, y_pred, pos_label=NEGATIVE_LABEL, zero_division=0))


# -----------------------------------------------------------------------------
# Cross-check económico
# -----------------------------------------------------------------------------
def expected_cost(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    cost_fp: float = COST_FP,
    cost_fn: float = COST_FN,
) -> float:
    """Costo esperado por predicción.

    FP = predijo 'new' siendo 'used' (comprador engañado)
    FN = predijo 'used' siendo 'new' (seller perjudicado en visibilidad)
    """
    cm = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    tn, fp, fn, tp = cm.ravel()
    n = len(y_true)
    if n == 0:
        return 0.0
    return float((cost_fp * fp + cost_fn * fn) / n)


# -----------------------------------------------------------------------------
# Reporte completo (lo que se imprime en cada fase)
# -----------------------------------------------------------------------------
def metrics_report(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    """Diccionario con todas las métricas de interés para el REPORT."""
    cm = confusion_matrix(y_true, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f05_new": f05_new(y_true, y_pred),
        "precision_new": precision_new(y_true, y_pred),
        "recall_new": recall_new(y_true, y_pred),
        "recall_used": recall_used(y_true, y_pred),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "expected_cost_3to1": expected_cost(y_true, y_pred, COST_FP, COST_FN),
        "confusion_matrix": {
            "TN_used_used": int(tn),
            "FP_used_pred_new": int(fp),
            "FN_new_pred_used": int(fn),
            "TP_new_new": int(tp),
        },
    }


def print_metrics(name: str, metrics: dict) -> None:
    """Imprime un reporte de métricas en formato consistente."""
    print(f"\n=== {name} ===")
    print(f"Accuracy            : {metrics['accuracy']:.4f}")
    print(f"F_0.5 (new)         : {metrics['f05_new']:.4f}")
    print(f"Precision (new)     : {metrics['precision_new']:.4f}")
    print(f"Recall    (new)     : {metrics['recall_new']:.4f}")
    print(f"Recall    (used)    : {metrics['recall_used']:.4f}")
    print(f"F1 macro            : {metrics['f1_macro']:.4f}")
    print(f"Expected cost (3:1) : {metrics['expected_cost_3to1']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"Confusion matrix    : TN={cm['TN_used_used']}  FP={cm['FP_used_pred_new']}  "
          f"FN={cm['FN_new_pred_used']}  TP={cm['TP_new_new']}")
