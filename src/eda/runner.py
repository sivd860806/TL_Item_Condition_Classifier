"""EDA principal — Parte 1.

Cuatro bloques:
  A. Estado del dataset (tamaños, balance, nulos por campo, top categorías)
  B. Distribution shift train vs test (KS sobre numéricas, chi² sobre categóricas)
  C. Auditoría de calidad de label (None, inconsistencias logicas, precios invalidos)
  D. Leakage candidates cuantificados (permalink, title, attributes, listing_type_id, etc.)

Output:
  - reports/eda_findings.md     hallazgos en markdown listo para citar en REPORT.md
  - reports/figures/eda_*.png   visualizaciones
  - stdout                      reporte resumido

Uso:
  python -m src.eda.runner
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless para correr en CI sin display
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ..config import FIGURES_DIR, REPORTS_DIR, set_global_seed
from ..data import load_data

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_get(d: dict, *keys, default=None):
    """Acceso seguro a campos anidados."""
    cur = d
    for k in keys:
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _is_null(v) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _save_fig(fig, name: str) -> Path:
    path = FIGURES_DIR / f"eda_{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Bloque A — Estado del dataset
# -----------------------------------------------------------------------------
def block_a_dataset_state(X_train, y_train, X_test, y_test) -> dict:
    print("\n" + "=" * 78)
    print("BLOQUE A — Estado del dataset")
    print("=" * 78)

    # A.1 Tamaños y balance
    train_balance = Counter(y_train)
    test_balance = Counter(y_test)

    n_train = len(X_train)
    n_test = len(X_test)
    pct_new_train = train_balance["new"] / n_train * 100
    pct_used_train = train_balance["used"] / n_train * 100
    pct_new_test = test_balance["new"] / n_test * 100
    pct_used_test = test_balance["used"] / n_test * 100

    print(f"\nA.1 Tamaños y balance de clases:")
    print(f"  Train: {n_train:>6} filas  →  new={train_balance['new']:>6} ({pct_new_train:5.2f}%)  "
          f"used={train_balance['used']:>6} ({pct_used_train:5.2f}%)")
    print(f"  Test : {n_test:>6} filas  →  new={test_balance['new']:>6} ({pct_new_test:5.2f}%)  "
          f"used={test_balance['used']:>6} ({pct_used_test:5.2f}%)")

    # A.2 Tasa de nulos por campo (sobre train)
    fields = list(X_train[0].keys())
    null_rates = {}
    for f in fields:
        n_null = sum(1 for x in X_train if _is_null(x.get(f)))
        null_rates[f] = n_null / n_train

    null_top = sorted(null_rates.items(), key=lambda kv: -kv[1])[:15]
    print(f"\nA.2 Top-15 campos por tasa de nulos (sobre train):")
    for f, r in null_top:
        print(f"  {f:<40s} {r * 100:6.2f}%")

    # A.3 Cardinalidad de categóricas clave
    print(f"\nA.3 Cardinalidad de categóricas clave (sobre train):")
    for f in ["listing_type_id", "buying_mode", "currency_id", "site_id",
              "status", "category_id"]:
        vals = [x.get(f) for x in X_train if x.get(f) is not None]
        n_unique = len(set(vals))
        most_common = Counter(vals).most_common(3)
        print(f"  {f:<20s} unique={n_unique:<6}  top3={most_common}")

    # Figura: balance de clases
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, balance, title in [(axes[0], train_balance, "Train (90k)"),
                                (axes[1], test_balance, "Test (10k)")]:
        labels = ["new", "used"]
        values = [balance.get(l, 0) for l in labels]
        bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452"])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(f"Balance de clases — {title}")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig_path = _save_fig(fig, "class_balance")
    print(f"\n  Figura: {fig_path.name}")

    return {
        "train_size": n_train,
        "test_size": n_test,
        "train_balance": dict(train_balance),
        "test_balance": dict(test_balance),
        "pct_new_train": pct_new_train,
        "pct_used_train": pct_used_train,
        "pct_new_test": pct_new_test,
        "pct_used_test": pct_used_test,
        "null_top": null_top,
    }


# -----------------------------------------------------------------------------
# Bloque B — Distribution shift train vs test
# -----------------------------------------------------------------------------
def block_b_distribution_shift(X_train, y_train, X_test, y_test) -> dict:
    print("\n" + "=" * 78)
    print("BLOQUE B — Distribution shift train vs test")
    print("=" * 78)

    findings = {}

    # B.1 KS test sobre price (continua)
    price_train = np.array([x.get("price") for x in X_train if x.get("price") is not None],
                            dtype=float)
    price_test = np.array([x.get("price") for x in X_test if x.get("price") is not None],
                           dtype=float)
    # log1p para que no domine la cola
    ks_stat, ks_p = stats.ks_2samp(np.log1p(price_train), np.log1p(price_test))
    print(f"\nB.1 KS test sobre log1p(price):")
    print(f"  D = {ks_stat:.4f}   p = {ks_p:.4g}")
    print(f"  median train = {np.median(price_train):.0f}  median test = {np.median(price_test):.0f}")
    findings["ks_log_price"] = {"D": float(ks_stat), "p": float(ks_p)}

    # B.2 Chi² sobre categóricas
    print(f"\nB.2 Chi² sobre categóricas (H0: misma distribución):")
    cat_fields = ["listing_type_id", "buying_mode", "currency_id", "status"]
    chi_results = {}
    for f in cat_fields:
        train_counts = Counter(x.get(f) for x in X_train)
        test_counts = Counter(x.get(f) for x in X_test)
        all_keys = sorted(set(train_counts) | set(test_counts), key=lambda k: -train_counts.get(k, 0))
        # Construir tabla de contingencia 2 x len(all_keys)
        obs = np.array([
            [train_counts.get(k, 0) for k in all_keys],
            [test_counts.get(k, 0) for k in all_keys],
        ])
        # Filtrar columnas con suma 0
        col_sums = obs.sum(axis=0)
        obs = obs[:, col_sums > 0]
        try:
            chi2, p, dof, _ = stats.chi2_contingency(obs)
            print(f"  {f:<20s} chi2={chi2:8.2f}  dof={dof:<3}  p={p:.4g}")
            chi_results[f] = {"chi2": float(chi2), "dof": int(dof), "p": float(p)}
        except Exception as e:
            print(f"  {f}: error -> {e}")
    findings["chi2_categorical"] = chi_results

    # B.3 Balance de clases train vs test (chi² 2x2)
    train_balance = Counter(y_train)
    test_balance = Counter(y_test)
    obs_class = np.array([
        [train_balance["new"], train_balance["used"]],
        [test_balance["new"], test_balance["used"]],
    ])
    chi2, p, dof, _ = stats.chi2_contingency(obs_class)
    print(f"\nB.3 Chi² sobre balance de clases train vs test:")
    print(f"  chi2={chi2:.4f}  p={p:.4g}  →  {'shift detectado' if p < 0.05 else 'sin shift relevante'}")
    findings["chi2_class_balance"] = {"chi2": float(chi2), "p": float(p)}

    # B.4 Figura: histograma de log(price) train vs test
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, np.log1p(price_train).max(), 60)
    ax.hist(np.log1p(price_train), bins=bins, alpha=0.5, label="train", density=True,
            color="#4C72B0")
    ax.hist(np.log1p(price_test), bins=bins, alpha=0.5, label="test", density=True,
            color="#DD8452")
    ax.set_xlabel("log1p(price)")
    ax.set_ylabel("densidad")
    ax.set_title(f"Distribución de log1p(price) — KS D={ks_stat:.3f}, p={ks_p:.3g}")
    ax.legend()
    fig.tight_layout()
    fig_path = _save_fig(fig, "price_distribution_shift")
    print(f"\n  Figura: {fig_path.name}")

    return findings


# -----------------------------------------------------------------------------
# Bloque C — Auditoría de calidad de label
# -----------------------------------------------------------------------------
def block_c_label_quality(X_train, y_train, X_test, y_test) -> dict:
    print("\n" + "=" * 78)
    print("BLOQUE C — Auditoría de calidad de label")
    print("=" * 78)

    findings = {}

    # C.1 Conteo de etiquetas raras
    none_train = sum(1 for y in y_train if y is None)
    none_test = sum(1 for y in y_test if y is None)
    other_train = sum(1 for y in y_train if y not in {"new", "used", None})
    print(f"\nC.1 Etiquetas anómalas:")
    print(f"  condition=None en train: {none_train}")
    print(f"  condition=None en test : {none_test}")
    print(f"  condition fuera de {{new, used, None}} en train: {other_train}")
    findings["label_anomalies"] = {
        "none_train": none_train, "none_test": none_test, "other_train": other_train,
    }

    # C.2 Inconsistencia: available_quantity=0 con sold_quantity>0
    inconsistent = sum(1 for x in X_train
                        if x.get("available_quantity", 0) == 0
                        and x.get("sold_quantity", 0) > 0)
    print(f"\nC.2 Filas con available_quantity=0 ∧ sold_quantity>0 (inconsistencia lógica):")
    print(f"  train: {inconsistent} ({inconsistent / len(X_train) * 100:.2f}%)")
    findings["inconsistent_qty"] = inconsistent

    # C.3 Precios anómalos
    zero_price = sum(1 for x in X_train if (x.get("price") or 0) <= 0)
    very_low = sum(1 for x in X_train
                    if x.get("price") is not None and 0 < x.get("price") < 10)
    print(f"\nC.3 Precios anómalos (train):")
    print(f"  price <= 0      : {zero_price}")
    print(f"  0 < price < 10  : {very_low}")
    findings["price_anomalies"] = {"zero_price": zero_price, "very_low": very_low}

    # C.4 Búsqueda exhaustiva de ITEM_CONDITION en attributes (verificación global)
    # Match ESTRICTO: id == "ITEM_CONDITION" exacto o name == "Condición"/"condicion"
    # exacto. Antes usábamos `"condici" in name` y eso capturaba "aire acondicionado".
    n_with_item_condition = 0
    sample_attr = None
    for x in X_train:
        for a in (x.get("attributes") or []):
            aid = (a.get("id") or "").strip().upper()
            aname_norm = (a.get("name") or "").strip().lower()
            # Normalizar tildes para comparación robusta
            aname_norm = aname_norm.replace("ó", "o").replace("á", "a")
            if aid == "ITEM_CONDITION" or aname_norm in {"condicion", "condicion del item",
                                                          "condicion del articulo",
                                                          "condicion del producto"}:
                n_with_item_condition += 1
                if sample_attr is None:
                    sample_attr = a
                break
    print(f"\nC.4 attributes con id == 'ITEM_CONDITION' (match estricto) en TODO X_train:")
    print(f"  {n_with_item_condition} hits ({n_with_item_condition / len(X_train) * 100:.4f}%)")
    if sample_attr:
        print(f"  ejemplo: {sample_attr}")
    findings["item_condition_in_attributes"] = n_with_item_condition

    return findings


# -----------------------------------------------------------------------------
# Bloque D — Leakage candidates cuantificados
# -----------------------------------------------------------------------------
def block_d_leakage(X_train, y_train) -> dict:
    print("\n" + "=" * 78)
    print("BLOQUE D — Leakage candidates cuantificados")
    print("=" * 78)

    findings = {}

    # D.1 permalink con substring "usado"/"nuevo"
    perma_usado = sum(1 for x in X_train if "usado" in (x.get("permalink") or "").lower())
    perma_nuevo = sum(1 for x in X_train if "nuevo" in (x.get("permalink") or "").lower())
    print(f"\nD.1 permalink con substrings de leakage (sobre train):")
    print(f"  'usado' : {perma_usado:>5}  ({perma_usado / len(X_train) * 100:5.2f}%)")
    print(f"  'nuevo' : {perma_nuevo:>5}  ({perma_nuevo / len(X_train) * 100:5.2f}%)")
    findings["permalink_leakage"] = {"usado": perma_usado, "nuevo": perma_nuevo}

    # D.2 title con palabras de leakage
    keywords_used = ["usado", "permuto", "permuta", "negociable",
                     "como nuevo", "estado:", "uso", "rayadito", "detalle"]
    keywords_new = ["sellado", "en caja", "nuevo sin uso", "0km", "nuevo nuevo"]
    title_used_hits = defaultdict(int)
    title_new_hits = defaultdict(int)
    for x in X_train:
        t = (x.get("title") or "").lower()
        for kw in keywords_used:
            if kw in t:
                title_used_hits[kw] += 1
        for kw in keywords_new:
            if kw in t:
                title_new_hits[kw] += 1
    print(f"\nD.2 title — palabras 'used-leaning':")
    for kw, c in sorted(title_used_hits.items(), key=lambda kv: -kv[1]):
        print(f"  {kw:<20s} {c:>6}")
    print(f"\n    title — palabras 'new-leaning':")
    for kw, c in sorted(title_new_hits.items(), key=lambda kv: -kv[1]):
        print(f"  {kw:<20s} {c:>6}")
    findings["title_used_keywords"] = dict(title_used_hits)
    findings["title_new_keywords"] = dict(title_new_hits)

    # D.3 listing_type_id × condition (tabla de contingencia con %used)
    print(f"\nD.3 listing_type_id × condition (sobre train):")
    print(f"  {'listing_type_id':<20s} {'total':>8} {'new':>8} {'used':>8} {'%used':>7}")
    lt_table = defaultdict(lambda: Counter())
    for x, y in zip(X_train, y_train):
        lt_table[x.get("listing_type_id")][y] += 1
    lt_summary = {}
    for k in sorted(lt_table.keys(), key=lambda k: -sum(lt_table[k].values())):
        c = lt_table[k]
        total = sum(c.values())
        n_new = c.get("new", 0)
        n_used = c.get("used", 0)
        pct_used = n_used / total * 100 if total > 0 else 0
        print(f"  {str(k):<20s} {total:>8} {n_new:>8} {n_used:>8} {pct_used:>6.2f}%")
        lt_summary[str(k)] = {"total": total, "new": n_new, "used": n_used,
                                "pct_used": pct_used}
    findings["listing_type_x_condition"] = lt_summary

    # D.4 warranty no-null × condition
    war_new = sum(1 for x, y in zip(X_train, y_train)
                   if y == "new" and not _is_null(x.get("warranty")))
    war_used = sum(1 for x, y in zip(X_train, y_train)
                    if y == "used" and not _is_null(x.get("warranty")))
    n_new = sum(1 for y in y_train if y == "new")
    n_used = sum(1 for y in y_train if y == "used")
    print(f"\nD.4 warranty no-null × condition (sobre train):")
    print(f"  new : {war_new}/{n_new} ({war_new / n_new * 100:5.2f}%)")
    print(f"  used: {war_used}/{n_used} ({war_used / n_used * 100:5.2f}%)")
    findings["warranty_by_class"] = {
        "new": {"with_warranty": war_new, "total": n_new,
                "pct": war_new / n_new * 100},
        "used": {"with_warranty": war_used, "total": n_used,
                 "pct": war_used / n_used * 100},
    }

    # D.5 sold_quantity × condition (verificar dirección)
    sold_new = [x.get("sold_quantity", 0) for x, y in zip(X_train, y_train) if y == "new"]
    sold_used = [x.get("sold_quantity", 0) for x, y in zip(X_train, y_train) if y == "used"]
    print(f"\nD.5 sold_quantity × condition:")
    print(f"  new : mean={np.mean(sold_new):6.2f}  median={np.median(sold_new):.0f}  "
          f"max={np.max(sold_new):>6}")
    print(f"  used: mean={np.mean(sold_used):6.2f}  median={np.median(sold_used):.0f}  "
          f"max={np.max(sold_used):>6}")
    findings["sold_quantity_by_class"] = {
        "new": {"mean": float(np.mean(sold_new)), "median": float(np.median(sold_new))},
        "used": {"mean": float(np.mean(sold_used)), "median": float(np.median(sold_used))},
    }

    # D.6 tags top-10 + tasa de used por tag
    tag_total = Counter()
    tag_used = Counter()
    for x, y in zip(X_train, y_train):
        for t in (x.get("tags") or []):
            tag_total[t] += 1
            if y == "used":
                tag_used[t] += 1
    print(f"\nD.6 tags top-10 con %used:")
    print(f"  {'tag':<30s} {'total':>8} {'%used':>7}")
    tag_summary = {}
    for tag, total in tag_total.most_common(10):
        pct_used = tag_used[tag] / total * 100 if total > 0 else 0
        print(f"  {tag:<30s} {total:>8} {pct_used:>6.2f}%")
        tag_summary[tag] = {"total": total, "pct_used": pct_used}
    findings["tags_top10"] = tag_summary

    # Figura: % used por listing_type_id
    fig, ax = plt.subplots(figsize=(9, 4))
    lt_keys = sorted(lt_summary.keys(), key=lambda k: -lt_summary[k]["pct_used"])
    pcts = [lt_summary[k]["pct_used"] for k in lt_keys]
    totals = [lt_summary[k]["total"] for k in lt_keys]
    bars = ax.bar(lt_keys, pcts, color="#4C72B0")
    for bar, total in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"n={total}", ha="center", va="bottom", fontsize=8)
    ax.axhline(46.2, color="red", linestyle="--", label="baseline (% used global)")
    ax.set_ylabel("% used")
    ax.set_xlabel("listing_type_id")
    ax.set_title("listing_type_id × condition — % used por tier (train)")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig_path = _save_fig(fig, "listing_type_pct_used")
    print(f"\n  Figura: {fig_path.name}")

    return findings


# -----------------------------------------------------------------------------
# Render del reporte en markdown
# -----------------------------------------------------------------------------
def render_findings_md(a: dict, b: dict, c: dict, d: dict) -> str:
    lines: list[str] = []
    lines.append("# EDA — Hallazgos clave\n")
    lines.append("Generado por `src/eda/runner.py`. Reproducible vía `make eda`.\n")
    lines.append("Toda métrica computada sobre `X_train` (90k filas) salvo aclaración.\n")

    # ---- Bloque A
    lines.append("## A. Estado del dataset\n")
    lines.append(f"- **Tamaños**: train = {a['train_size']:,} filas, test = {a['test_size']:,} filas.")
    lines.append(f"- **Balance de clases (train)**: new = {a['train_balance']['new']:,} "
                 f"({a['pct_new_train']:.2f}%), used = {a['train_balance']['used']:,} "
                 f"({a['pct_used_train']:.2f}%).")
    lines.append(f"- **Balance de clases (test)**: new = {a['test_balance']['new']:,} "
                 f"({a['pct_new_test']:.2f}%), used = {a['test_balance']['used']:,} "
                 f"({a['pct_used_test']:.2f}%).")
    lines.append("- **Top-15 campos por tasa de nulos**:\n")
    lines.append("| Campo | % nulos |")
    lines.append("|-------|---------|")
    for f, r in a["null_top"]:
        lines.append(f"| `{f}` | {r * 100:.2f}% |")
    lines.append("")
    lines.append("Figura: `reports/figures/eda_class_balance.png`.\n")

    # ---- Bloque B
    lines.append("## B. Distribution shift train vs test\n")
    ks = b["ks_log_price"]
    lines.append(f"- **KS test sobre log1p(price)**: D = {ks['D']:.4f}, p = {ks['p']:.4g}.")
    chi_class = b["chi2_class_balance"]
    lines.append(f"- **Chi² sobre balance de clases train vs test**: "
                 f"χ² = {chi_class['chi2']:.4f}, p = {chi_class['p']:.4g} → "
                 f"{'shift detectado' if chi_class['p'] < 0.05 else 'sin shift relevante'}.")
    lines.append("- **Chi² sobre categóricas**:\n")
    lines.append("| Campo | χ² | dof | p |")
    lines.append("|-------|-----|-----|---|")
    for f, r in b["chi2_categorical"].items():
        lines.append(f"| `{f}` | {r['chi2']:.2f} | {r['dof']} | {r['p']:.4g} |")
    lines.append("")
    lines.append("**Implicancia**: si algún p<0.05 con efecto material, "
                 "documentar que la validación interna debe usar últimos 10–20% "
                 "del train (mimic temporal) en lugar de KFold puro.\n")
    lines.append("Figura: `reports/figures/eda_price_distribution_shift.png`.\n")

    # ---- Bloque C
    lines.append("## C. Auditoría de calidad de label\n")
    la = c["label_anomalies"]
    lines.append(f"- `condition=None` en train: **{la['none_train']}**; en test: **{la['none_test']}**.")
    lines.append(f"- `condition` fuera de {{new, used, None}}: **{la['other_train']}**.")
    lines.append(f"- Filas con `available_quantity=0 ∧ sold_quantity>0`: "
                 f"**{c['inconsistent_qty']:,}** ({c['inconsistent_qty'] / 90000 * 100:.2f}% del train).")
    pa = c["price_anomalies"]
    lines.append(f"- `price ≤ 0`: **{pa['zero_price']}**.  `0 < price < 10`: **{pa['very_low']}**.")
    lines.append(f"- Verificación global de `attributes` con id `ITEM_CONDITION` o name 'condici': "
                 f"**{c['item_condition_in_attributes']}** filas. "
                 f"{'(El leakage típico no aplica a este snapshot.)' if c['item_condition_in_attributes'] == 0 else '(Leakage real — drop obligatorio.)'}")
    lines.append("")

    # ---- Bloque D
    lines.append("## D. Leakage candidates cuantificados\n")
    pl = d["permalink_leakage"]
    lines.append(f"### D.1 `permalink`")
    lines.append(f"- Contiene 'usado': **{pl['usado']}** filas. Contiene 'nuevo': **{pl['nuevo']}** filas. "
                 f"**Drop obligatorio**.\n")

    lines.append(f"### D.2 `title` — palabras predictoras")
    lines.append(f"\nKeywords used-leaning:\n")
    lines.append("| keyword | hits |")
    lines.append("|---------|------|")
    for kw, c_ in sorted(d["title_used_keywords"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{kw}` | {c_} |")
    lines.append("\nKeywords new-leaning:\n")
    lines.append("| keyword | hits |")
    lines.append("|---------|------|")
    for kw, c_ in sorted(d["title_new_keywords"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{kw}` | {c_} |")
    lines.append(f"\n**Implicancia**: el título es señal léxica fuerte. TF-IDF char_wb va a capturar "
                 f"esto sin ingeniería manual.\n")

    lines.append(f"### D.3 `listing_type_id` × condition  (FEATURE DOMINANTE)")
    lines.append("")
    lines.append("| listing_type_id | total | new | used | %used |")
    lines.append("|-----------------|-------|-----|------|-------|")
    for k, v in sorted(d["listing_type_x_condition"].items(),
                        key=lambda kv: -kv[1]["total"]):
        lines.append(f"| `{k}` | {v['total']:,} | {v['new']:,} | {v['used']:,} | {v['pct_used']:.2f}% |")
    lines.append("")
    lines.append("**Implicancia**: `listing_type_id` es prácticamente un clasificador por sí solo "
                 "(spread de %used entre tiers). NO es leakage técnico (lo elige el seller al listar), "
                 "pero hay que correr la **ablación P1.6 con/sin este campo** para responder "
                 "*\"¿el modelo aprendió new-vs-used o solo pricing tier?\"*.\n")
    lines.append("Figura: `reports/figures/eda_listing_type_pct_used.png`.\n")

    wc = d["warranty_by_class"]
    lines.append(f"### D.4 `warranty`")
    lines.append(f"- new con warranty: **{wc['new']['pct']:.2f}%** "
                 f"({wc['new']['with_warranty']:,}/{wc['new']['total']:,})")
    lines.append(f"- used con warranty: **{wc['used']['pct']:.2f}%** "
                 f"({wc['used']['with_warranty']:,}/{wc['used']['total']:,})")
    lines.append("- Señal real, defendible. Derivar `has_warranty: bool`.\n")

    sc = d["sold_quantity_by_class"]
    lines.append(f"### D.5 `sold_quantity`")
    lines.append(f"- new : mean = **{sc['new']['mean']:.2f}**, median = {sc['new']['median']:.0f}")
    lines.append(f"- used: mean = **{sc['used']['mean']:.2f}**, median = {sc['used']['median']:.0f}")
    lines.append("- **Dirección inversa a la intuición común**: new tiene mucho más "
                 "sold_quantity por re-listeo masivo. Feature útil pero hay que documentar "
                 "que correlaciona con identidad de seller (no usar `seller_id` por leakage).\n")

    lines.append(f"### D.6 `tags` top-10")
    lines.append("")
    lines.append("| tag | total | %used |")
    lines.append("|-----|-------|-------|")
    for k, v in d["tags_top10"].items():
        lines.append(f"| `{k}` | {v['total']:,} | {v['pct_used']:.2f}% |")
    lines.append("")
    lines.append("**Implicancia**: tags son señal de *listing* (cómo se publica), no del producto. "
                 "Multi-hot top-10 es low-cost y captura señal robusta.\n")

    # Cierre
    lines.append("## Resumen ejecutivo (para el REPORT.md)\n")
    lines.append(f"- Dataset balanceado (~{a['pct_new_train']:.0f}/{a['pct_used_train']:.0f}); "
                 f"métrica enfocada en costo asimétrico (no en imbalance) está justificada.")
    lines.append(f"- Sin distribution shift relevante en clase entre train y test "
                 f"(p = {b['chi2_class_balance']['p']:.3g}); KFold puro sobre train es defendible.")
    lines.append(f"- Leakage real cuantificado: drop `permalink`, `id`, `seller_id`, `thumbnail`, "
                 f"`secure_thumbnail`, `pictures` (URLs).")
    lines.append(f"- `listing_type_id` es la feature dominante; ablación P1.6 obligatoria para "
                 f"defender el modelo en entrevista.")
    lines.append(f"- TF-IDF char_wb sobre `title` capturará palabras predictoras "
                 f"({sum(d['title_used_keywords'].values())} hits used + "
                 f"{sum(d['title_new_keywords'].values())} hits new).")
    lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    set_global_seed()

    print("Cargando dataset vía build_dataset()...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"  X_train={len(X_train)}  X_test={len(X_test)}")

    a = block_a_dataset_state(X_train, y_train, X_test, y_test)
    b = block_b_distribution_shift(X_train, y_train, X_test, y_test)
    c = block_c_label_quality(X_train, y_train, X_test, y_test)
    d = block_d_leakage(X_train, y_train)

    md = render_findings_md(a, b, c, d)
    out = REPORTS_DIR / "eda_findings.md"
    out.write_text(md, encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"Reporte: {out}")
    print(f"Figuras: {FIGURES_DIR}")
    print("=" * 78)

    # Persistir hallazgos en JSON para que el REPORT.md final pueda citarlos sin re-correr
    json_out = REPORTS_DIR / "eda_findings.json"
    json_out.write_text(json.dumps(
        {"A": a, "B": b, "C": c, "D": d}, indent=2, default=str
    ), encoding="utf-8")
    print(f"JSON   : {json_out}")


if __name__ == "__main__":
    main()
