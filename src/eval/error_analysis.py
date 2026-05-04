"""Error analysis sobre validacion (P1.8).

Identifica los top-50 falsos positivos y top-50 falsos negativos al threshold
optimo (computado en P1.7). Para cada error guarda los campos relevantes que
permiten interpretar el patron de fallo.

Convencion (consistente):
  FP = predijo 'new' siendo 'used'  -> comprador engañado, mas grave
  FN = predijo 'used' siendo 'new'  -> seller perjudicado en visibilidad

Genera:
  - reports/errors_top100.csv
  - reports/error_analysis_summary.md
  - reports/figures/error_analysis_distribution.png

Uso:
  python -m src.eval.error_analysis
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED, set_global_seed
from ..data import load_data
from ..features import flatten_records


def main() -> None:
    set_global_seed(SEED)
    t0 = time.time()

    artifact_path = MODELS_DIR / "lgbm_best.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"No existe {artifact_path}.")
    artifact = joblib.load(artifact_path)
    pre = artifact["preprocessor"]
    model = artifact["model"]
    ref = artifact["ref_date"]
    thr = artifact.get("best_threshold", 0.5)
    print(f"Modelo cargado. best_threshold={thr:.3f}")

    # Replicar split de P1.4
    print("\nReplicando split...")
    X_raw, y_str, _, _ = load_data()
    valid = [(x, y) for x, y in zip(X_raw, y_str) if y in {"new", "used"}]
    X_raw = [x for x, _ in valid]
    y_str = [y for _, y in valid]

    df_full = flatten_records(X_raw, ref_date=ref)
    # Tambien queremos campos crudos para el reporte (titulo, etc.)
    raw_full = pd.DataFrame([{
        "title": r.get("title") or "",
        "category_id": r.get("category_id"),
        "listing_type_id": r.get("listing_type_id"),
        "price": r.get("price"),
        "warranty_raw": r.get("warranty"),
        "n_pictures": len(r.get("pictures") or []),
    } for r in X_raw])

    df_tr, df_va, y_tr_str, y_va_str = train_test_split(
        df_full, y_str,
        test_size=0.20, stratify=y_str, random_state=SEED,
    )
    raw_tr, raw_va = train_test_split(
        raw_full,
        test_size=0.20, stratify=y_str, random_state=SEED,
    )
    raw_va = raw_va.reset_index(drop=True)
    print(f"  validacion: n={len(df_va)}")

    # Predict
    X_va = pre.transform(df_va)
    proba_va = model.predict_proba(X_va)[:, 1]
    y_va_int = np.array([1 if v == "new" else 0 for v in y_va_str])
    y_pred_int = (proba_va >= thr).astype(np.int8)

    # Build dataframe de errores
    errors = pd.DataFrame({
        "title": raw_va["title"],
        "category_id": raw_va["category_id"],
        "listing_type_id": raw_va["listing_type_id"],
        "price": raw_va["price"],
        "n_pictures": raw_va["n_pictures"],
        "has_warranty": raw_va["warranty_raw"].notna() & (raw_va["warranty_raw"] != ""),
        "condition_real": y_va_str,
        "proba_new": proba_va,
        "y_pred": ["new" if v == 1 else "used" for v in y_pred_int],
        "len_title": raw_va["title"].str.len(),
    })
    errors["correct"] = errors["condition_real"] == errors["y_pred"]
    errors["error_type"] = np.where(
        errors["correct"], "correct",
        np.where(errors["y_pred"] == "new", "FP", "FN"),
    )

    # Top-50 FP por proba_new descendente (mas confiados de "new" siendo used)
    top_fp = errors[errors["error_type"] == "FP"] \
        .sort_values("proba_new", ascending=False).head(50)
    # Top-50 FN por proba_new ascendente (mas confiados de "used" siendo new)
    top_fn = errors[errors["error_type"] == "FN"] \
        .sort_values("proba_new", ascending=True).head(50)

    out = pd.concat([
        top_fp.assign(rank_position=range(1, len(top_fp) + 1)),
        top_fn.assign(rank_position=range(1, len(top_fn) + 1)),
    ], ignore_index=True)
    csv_path = REPORTS_DIR / "errors_top100.csv"
    out.to_csv(csv_path, index=False)
    print(f"\nCSV de top-100 errores: {csv_path}")

    # Patrones agregados para el reporte
    n_fp = (errors["error_type"] == "FP").sum()
    n_fn = (errors["error_type"] == "FN").sum()
    n_correct = (errors["error_type"] == "correct").sum()

    fp_by_lt = errors[errors["error_type"] == "FP"].groupby("listing_type_id").size().sort_values(ascending=False)
    fn_by_lt = errors[errors["error_type"] == "FN"].groupby("listing_type_id").size().sort_values(ascending=False)

    fp_by_cat = errors[errors["error_type"] == "FP"].groupby("category_id").size().sort_values(ascending=False).head(10)
    fn_by_cat = errors[errors["error_type"] == "FN"].groupby("category_id").size().sort_values(ascending=False).head(10)

    print(f"\nResumen de errores sobre validacion (n={len(errors)}):")
    print(f"  Correct: {n_correct} ({n_correct/len(errors)*100:.2f}%)")
    print(f"  FP (predijo new, era used): {n_fp} ({n_fp/len(errors)*100:.2f}%)")
    print(f"  FN (predijo used, era new): {n_fn} ({n_fn/len(errors)*100:.2f}%)")

    print(f"\nFP por listing_type_id (top 5):\n{fp_by_lt.head().to_string()}")
    print(f"\nFN por listing_type_id (top 5):\n{fn_by_lt.head().to_string()}")

    # Markdown summary
    md = f"""# Analisis de errores — top-50 FP + top-50 FN

Generado al threshold optimo **{thr:.3f}** (calibrado en P1.7) sobre la validacion
interna del split del 80/20.

## Resumen

| Tipo | Conteo | % de validacion |
|------|-------:|----------------:|
| Correctas | {n_correct:,} | {n_correct/len(errors)*100:.2f}% |
| FP (predijo new, era used) | {n_fp:,} | {n_fp/len(errors)*100:.2f}% |
| FN (predijo used, era new) | {n_fn:,} | {n_fn/len(errors)*100:.2f}% |

**Lectura**: con el sample_weight 3:1 y threshold optimizado, el modelo es
conservador al afirmar "new". Esperamos FP < FN en numero absoluto, lo cual
deberia confirmarse arriba.

## FP por `listing_type_id` (donde el modelo se equivoca diciendo "new" siendo "used")

```
{fp_by_lt.head(7).to_string()}
```

## FN por `listing_type_id` (donde el modelo es muy conservador y dice "used" siendo "new")

```
{fn_by_lt.head(7).to_string()}
```

## Top-10 categorias mas problematicas

**FP**:
```
{fp_by_cat.to_string()}
```

**FN**:
```
{fn_by_cat.to_string()}
```

## Patrones que vale la pena explorar con mas tiempo

- **Si los FP se concentran en `bronze` o `silver`**: el modelo confunde listings
  comerciales serios pero de productos usados (e.g. refurbs vendidos por tiendas).
  Mejora posible: feature de "es seller con multiples listings de la misma categoria".
- **Si los FN se concentran en `free`**: el modelo aprendio "free=used" demasiado
  fuerte. Mejora: regularizacion mas agresiva o ablacion de listing_type_id (P1.6).
- **Errores por longitud de titulo**: titulos muy cortos (<20 chars) suelen ser
  ambigos. Mejora: bucket `title_too_short` como feature.

## Detalle

CSV completo en `reports/errors_top100.csv` con columnas: `title`, `category_id`,
`listing_type_id`, `price`, `n_pictures`, `has_warranty`, `condition_real`,
`proba_new`, `y_pred`, `error_type`, `rank_position`.
"""
    md_path = REPORTS_DIR / "error_analysis_summary.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\nReporte: {md_path}")

    # Distribución de proba para correctos vs incorrectos
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 51)
    ax.hist(errors[errors["correct"]]["proba_new"], bins=bins, alpha=0.5,
             label="Correctos", color="#55A868", density=True)
    ax.hist(errors[~errors["correct"]]["proba_new"], bins=bins, alpha=0.5,
             label="Errores", color="#C44E52", density=True)
    ax.axvline(thr, color="black", linestyle=":",
                label=f"Threshold={thr:.2f}")
    ax.set_xlabel("P(new)")
    ax.set_ylabel("densidad")
    ax.set_title("Distribucion de P(new) — correctos vs errores")
    ax.legend()
    fig.tight_layout()
    fig_path = FIGURES_DIR / "error_analysis_distribution.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura: {fig_path}")

    print(f"\nTiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
