"""Wrapper sobre `build_dataset()` del archivo `new_or_used.py` provisto.

Esta es la **única** forma autorizada de cargar el dataset (regla del
assessment). Cualquier otra ruta de carga invalida la entrega.

El wrapper existe por dos razones:
1. `build_dataset()` usa una ruta relativa al cwd ("MLA_100k.jsonlines").
   Si el script se lanza desde otra carpeta, falla. Acá hacemos chdir
   explícito y lo restauramos.
2. Aislamos en un solo lugar la dependencia con el archivo provisto, por si
   en el futuro queremos agregar caching o sampling para iteración rápida
   (sin romper la regla — todo iría sobre el output de build_dataset).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .config import ROOT_DIR

# Hacemos importable el archivo provisto sin moverlo de la raíz
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def load_data() -> tuple[list[dict[str, Any]], list[str | None],
                          list[dict[str, Any]], list[str | None]]:
    """Carga el dataset usando `build_dataset` del archivo provisto.

    Returns
    -------
    X_train : list[dict]   90k listings con campo `condition`
    y_train : list[str]    etiquetas correspondientes
    X_test  : list[dict]   10k listings SIN el campo `condition` (lo borra el loader)
    y_test  : list[str]    etiquetas reales correspondientes (held-out)
    """
    # Import diferido para que el módulo sea importable aunque el archivo
    # del dataset no exista (e.g. en CI sin datos)
    from new_or_used import build_dataset  # type: ignore[import-not-found]

    cwd_original = Path.cwd()
    os.chdir(ROOT_DIR)
    try:
        X_train, y_train, X_test, y_test = build_dataset()
    finally:
        os.chdir(cwd_original)

    # Verificaciones de invariantes (mismas que el __main__ del loader)
    assert len(X_train) == len(y_train) == 90_000
    assert len(X_test) == len(y_test) == 10_000
    assert all("condition" not in x for x in X_test)
    assert set(y_train) <= {"new", "used", None}

    return X_train, y_train, X_test, y_test
