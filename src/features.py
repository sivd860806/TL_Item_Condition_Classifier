"""Pipeline de features: flatten de los listings + ColumnTransformer.

Dos pasos:
  1. `flatten_records()` toma la lista de dicts cruda (output de build_dataset)
     y devuelve un DataFrame con derivaciones aplicadas. Decisiones campo-por-campo
     en `reports/feature_catalog.md`.
  2. `build_preprocessor()` arma un ColumnTransformer con tres ramas
     (numérica + categórica + texto). Es reutilizado por LR y por GBDT.

Reglas para evitar leakage temporal:
  - `ref_date` (fecha de referencia para deltas) se computa con el train y se
     pasa al test. Nunca se usa el max del test.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Top tags multi-hot (justificadas en feature_catalog.md sección "Listas")
# -----------------------------------------------------------------------------
TOP_TAGS = (
    "dragged_bids_and_visits",
    "good_quality_thumbnail",
    "dragged_visits",
    "free_relist",
    "poor_quality_thumbnail",
)


# -----------------------------------------------------------------------------
# Helpers de extracción segura
# -----------------------------------------------------------------------------
def _safe_get(d: dict | None, *keys, default=None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _is_filled(v) -> bool:
    """True si v tiene contenido (no None, no '', no [], no {})."""
    return v is not None and v != "" and v != [] and v != {}


def _parse_iso_date(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        # ML usa ISO con sufijo 'Z' tipo "2015-09-15T10:00:00.000Z"
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _days_between(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    return (later - earlier).total_seconds() / 86400.0


# -----------------------------------------------------------------------------
# Flatten record-by-record
# -----------------------------------------------------------------------------
def _flatten_one(r: dict, ref_date: datetime | None = None) -> dict:
    """Convierte un listing en un dict plano con las features derivadas."""
    out: dict[str, Any] = {}

    # ---- Texto -----------------------------------------------------------
    out["title"] = (r.get("title") or "")
    out["title_len"] = len(out["title"])
    out["has_subtitle"] = _is_filled(r.get("subtitle"))
    out["has_description"] = _is_filled(r.get("descriptions"))

    # ---- Categóricas planas ---------------------------------------------
    out["listing_type_id"] = r.get("listing_type_id")
    out["buying_mode"] = r.get("buying_mode")
    out["currency_id"] = r.get("currency_id")
    out["category_id"] = r.get("category_id")
    out["status"] = r.get("status")
    out["international_delivery_mode"] = r.get("international_delivery_mode")

    # ---- Booleanas / flags simples --------------------------------------
    out["accepts_mercadopago"] = bool(r.get("accepts_mercadopago"))
    out["automatic_relist"] = bool(r.get("automatic_relist"))
    out["has_warranty"] = _is_filled(r.get("warranty"))
    out["has_video"] = _is_filled(r.get("video_id"))
    out["has_catalog_product"] = _is_filled(r.get("catalog_product_id"))
    out["has_official_store"] = _is_filled(r.get("official_store_id"))
    out["has_diff_pricing"] = _is_filled(r.get("differential_pricing"))

    # ---- Numéricas + transforms -----------------------------------------
    price = r.get("price")
    out["price_log"] = float(np.log1p(price)) if price and price > 0 else 0.0
    out["price_was_zero"] = bool(price is None or price <= 0)

    base_price = r.get("base_price")
    out["base_price_log"] = float(np.log1p(base_price)) if base_price and base_price > 0 else 0.0

    original_price = r.get("original_price")
    out["has_discount"] = original_price is not None and price is not None and original_price > price
    out["discount_pct"] = (
        1.0 - price / original_price
        if (original_price and price and original_price > 0)
        else 0.0
    )

    out["available_quantity"] = r.get("available_quantity") or 0
    out["initial_quantity"] = r.get("initial_quantity") or 0
    out["sold_quantity"] = r.get("sold_quantity") or 0
    out["sold_ratio"] = (
        out["sold_quantity"] / out["initial_quantity"]
        if out["initial_quantity"] > 0
        else 0.0
    )

    # ---- Listas: conteos -------------------------------------------------
    out["n_pictures"] = len(r.get("pictures") or [])
    out["n_attributes"] = len(r.get("attributes") or [])
    out["n_variations"] = len(r.get("variations") or [])
    out["n_payment_methods"] = len(r.get("non_mercado_pago_payment_methods") or [])

    # ---- Tags multi-hot --------------------------------------------------
    tags_set = set(r.get("tags") or [])
    for t in TOP_TAGS:
        out[f"tag_{t}"] = t in tags_set

    # ---- Shipping --------------------------------------------------------
    out["free_shipping"] = bool(_safe_get(r, "shipping", "free_shipping", default=False))
    out["local_pickup"] = bool(_safe_get(r, "shipping", "local_pickup", default=False))
    out["shipping_mode"] = _safe_get(r, "shipping", "mode")

    # ---- Geografía -------------------------------------------------------
    out["country_id"] = _safe_get(r, "seller_address", "country", "id")
    out["state_id"] = _safe_get(r, "seller_address", "state", "id")

    # ---- Fechas ----------------------------------------------------------
    start = _parse_iso_date(r.get("start_time"))
    stop = _parse_iso_date(r.get("stop_time"))
    created = _parse_iso_date(r.get("date_created"))
    updated = _parse_iso_date(r.get("last_updated"))

    out["listing_duration_days"] = _days_between(stop, start) or 0.0
    if ref_date is not None:
        out["listing_age_days"] = _days_between(ref_date, created) or 0.0
        out["time_since_update_days"] = _days_between(ref_date, updated) or 0.0
    else:
        out["listing_age_days"] = 0.0
        out["time_since_update_days"] = 0.0

    return out


def compute_ref_date(records: list[dict]) -> datetime:
    """Fecha de referencia para deltas: max(last_updated) del set provisto.

    Se calcula UNA vez sobre el train y se reutiliza en test, para no leakear
    el max del test al modelo.
    """
    candidates = []
    for r in records:
        d = _parse_iso_date(r.get("last_updated"))
        if d is not None:
            candidates.append(d)
    if not candidates:
        return datetime(2015, 12, 31)
    return max(candidates)


def flatten_records(records: list[dict], ref_date: datetime | None = None) -> pd.DataFrame:
    """Convierte una lista de listings en un DataFrame de features derivadas.

    Parameters
    ----------
    records : list[dict]
        Output de build_dataset (X_train o X_test).
    ref_date : datetime, opcional
        Fecha de referencia para los deltas temporales. Si None, se computa
        sobre los propios records (uso típico para el train; para el test
        debe pasarse el ref_date computado sobre el train).

    Returns
    -------
    pd.DataFrame
        Una fila por listing; columnas según `_flatten_one`.
    """
    if ref_date is None:
        ref_date = compute_ref_date(records)
    rows = [_flatten_one(r, ref_date) for r in records]
    df = pd.DataFrame(rows)
    return df


# -----------------------------------------------------------------------------
# ColumnTransformer
# -----------------------------------------------------------------------------
NUMERIC_COLS = (
    "title_len",
    "price_log", "price_was_zero",
    "base_price_log",
    "has_discount", "discount_pct",
    "available_quantity", "initial_quantity", "sold_quantity", "sold_ratio",
    "n_pictures", "n_attributes", "n_variations", "n_payment_methods",
    "listing_duration_days", "listing_age_days", "time_since_update_days",
    "accepts_mercadopago", "automatic_relist",
    "has_warranty", "has_video", "has_catalog_product",
    "has_official_store", "has_diff_pricing",
    "has_subtitle", "has_description",
    "free_shipping", "local_pickup",
    *(f"tag_{t}" for t in TOP_TAGS),
)

CATEGORICAL_COLS = (
    "listing_type_id",
    "buying_mode",
    "currency_id",
    "category_id",
    "status",
    "international_delivery_mode",
    "shipping_mode",
    "country_id",
    "state_id",
)

TEXT_COL = "title"


def build_preprocessor(
    *,
    scale_numeric: bool = True,
    text_max_features: int = 20_000,
) -> ColumnTransformer:
    """Construye el ColumnTransformer con las tres ramas estándar.

    Parameters
    ----------
    scale_numeric : bool
        True para LR (necesita features escaladas), False para GBDT (invariante).
    text_max_features : int
        Cap del vocabulario de TF-IDF char_wb. 20k es el sweet spot identificado.
    """
    # Rama numérica
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=True)))
    numeric_pipe = Pipeline(numeric_steps)

    # Rama categórica
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(
            min_frequency=50,
            handle_unknown="infrequent_if_exist",
            sparse_output=True,
        )),
    ])

    # Rama de texto (TF-IDF char-grams sobre title)
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=text_max_features,
            lowercase=True,
            strip_accents="unicode",
            min_df=5,
            sublinear_tf=True,
        )),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(NUMERIC_COLS)),
            ("cat", categorical_pipe, list(CATEGORICAL_COLS)),
            ("txt", text_pipe, TEXT_COL),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
    return pre


def build_text_only_pipeline(text_max_features: int = 20_000) -> TfidfVectorizer:
    """Vectorizer aislado para el baseline LR text-only (sin features tabulares)."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=text_max_features,
        lowercase=True,
        strip_accents="unicode",
        min_df=5,
        sublinear_tf=True,
    )
