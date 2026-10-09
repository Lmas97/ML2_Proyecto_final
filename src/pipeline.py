"""
Funciones de alto nivel que encadenan clasificador -> grafo -> BCubed.
Las usan los notebooks 02 y 03 para no repetir código.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import clustering as CL
from . import config as C
from . import metrics as ME


def evaluate_model(model, df: pd.DataFrame, test_pairs: pd.DataFrame, features: list[str],
                   threshold: float = C.EDGE_THRESHOLD, seed: int = C.SEED) -> dict:
    """Predice probabilidades, construye perfiles y calcula BCubed en los 3 subconjuntos."""
    proba = model.predict_proba(test_pairs[features].to_numpy(np.float32))[:, 1]
    pred = CL.cluster_all_blocks(df, test_pairs, proba, threshold, seed)
    ev, blocks = ME.evaluate_subsets(df, pred)
    return {"proba": proba, "pred": pred, "subsets": ev, "blocks": blocks}


def evaluate_rule(df: pd.DataFrame) -> dict:
    test = df[df.subset != "train"]
    pred = CL.milojevic_hybrid(test)
    ev, blocks = ME.evaluate_subsets(df, pred)
    return {"pred": pred, "subsets": ev, "blocks": blocks}


def long_table(results: list[dict]) -> pd.DataFrame:
    """Convierte la lista de modelos evaluados en una tabla larga (una fila por subconjunto)."""
    rows = []
    for r in results:
        for subset, m in r["subsets"].items():
            rows.append({"algorithm": r["algorithm"], "ratio": r["ratio"], "scoring": r["scoring"],
                         "subset": subset, **m})
    return pd.DataFrame(rows)


def best_per_algorithm(table: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """
    Mejor configuración de cada algoritmo por BCubed F1 'macro' en cada
    subconjunto (criterio de las Tablas 4, 6 y 8 del artículo) + la regla.
    """
    rows = []
    for subset in ["todos", "grandes", "aleatorios"]:
        rows.append({"subset": subset, "modelo": "Reglas (Milojević)", "config_sim": "-",
                     **rule["subsets"][subset]})
        sub = table[table.subset == subset]
        for algo in ["Decision Tree", "Random Forest", "Hist-GBDT"]:
            r = sub[sub.algorithm == algo].sort_values("macro_b3f", ascending=False).iloc[0]
            rows.append({"subset": subset, "modelo": algo, "config_sim": f"{r.scoring}, {r.ratio}",
                         **{k: r[k] for k in rule["subsets"][subset]}})
    return pd.DataFrame(rows)


def compare_with_article(best: pd.DataFrame) -> pd.DataFrame:
    art = pd.DataFrame(C.ARTICLE_BCUBED, columns=[
        "subset", "modelo", "config_articulo", "micro_b3p", "micro_b3r", "micro_b3f",
        "macro_b3p", "macro_b3r", "macro_b3f", "true_clusters", "pred_clusters"])
    m = art.merge(best, on=["subset", "modelo"], suffixes=("_art", "_sim"))
    m["dif_macro_b3f"] = m.macro_b3f_sim - m.macro_b3f_art
    cols = ["subset", "modelo", "config_articulo", "config_sim"]
    for k in ["micro_b3f", "macro_b3p", "macro_b3r", "macro_b3f", "true_clusters", "pred_clusters"]:
        cols += [f"{k}_art", f"{k}_sim"]
    return m[cols + ["dif_macro_b3f"]]
