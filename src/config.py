"""Configuración global del proyecto.

Cualquier constante reproducible vive aquí. Si algo cambia (seed, ratio de
costo, ruta de datos), debe cambiar en un único lugar.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# ---- Reproducibilidad -------------------------------------------------------
SEED: int = 42


def set_global_seed(seed: int = SEED) -> None:
    """Fija seed en numpy y random builtin. LightGBM/XGBoost reciben seed
    explícito en sus llamadas a fit; lo manejamos local para evitar efectos
    sorpresa en otras libs."""
    random.seed(seed)
    np.random.seed(seed)


# ---- Paths -----------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_FILE: Path = ROOT_DIR / "MLA_100k.jsonlines"
LOADER_FILE: Path = ROOT_DIR / "new_or_used.py"

REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = ROOT_DIR / "models"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

# Crear directorios si no existen (idempotente)
for _d in (REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---- Reglas del assessment -------------------------------------------------
TRAIN_SIZE: int = 90_000
TEST_SIZE: int = 10_000
ACCURACY_THRESHOLD: float = 0.86  # mínimo requerido por el enunciado


# ---- Convenciones de la tarea binaria --------------------------------------
POSITIVE_LABEL: str = "new"   # clase positiva (sklearn convention)
NEGATIVE_LABEL: str = "used"  # clase negativa
CLASSES: tuple[str, str] = (NEGATIVE_LABEL, POSITIVE_LABEL)


# ---- Costos asimétricos ----------------------------------------------------
# Argumento: un FP (predijo 'new', era 'used') daña confianza del comprador
# más que un FN (predijo 'used', era 'new') daña visibilidad del seller.
# Ratio conservador: FP = 3 × FN.
# Ver REPORT.md sección "Métrica secundaria" para el argumento completo.
COST_FP: float = 3.0
COST_FN: float = 1.0
COST_RATIO: float = COST_FP / COST_FN  # 3.0


# ---- Hyperparam tuning -----------------------------------------------------
OPTUNA_N_TRIALS: int = 50
OPTUNA_TIMEOUT_SECONDS: int = 1800  # 30 min hard cap
CV_FOLDS: int = 3
