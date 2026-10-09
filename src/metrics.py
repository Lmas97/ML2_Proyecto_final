"""
Métricas BCubed (Amigó et al., 2009) y su agregación como en el artículo.

Para una autoría a:
  precisión(a) = |cluster(a) ∩ autor(a)| / |cluster(a)|
  recall(a)    = |cluster(a) ∩ autor(a)| / |autor(a)|
Se promedian sobre las autorías y F1 = media armónica de P y R.

Agregación (nomenclatura del ARTÍCULO, invertida respecto al uso habitual):
  * "micro": BCubed por bloque y luego promedio simple entre bloques
             (b3f = promedio de los F1 por bloque);
  * "macro": todas las autorías del conjunto como un solo bloque.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def bcubed(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    t = pd.factorize(np.asarray(true))[0]
    p = pd.factorize(np.asarray(pred))[0]
    cont = pd.crosstab(t, p).to_numpy().astype(float)        # n_tc
    size_c = cont.sum(0)                                      # |cluster|
    size_t = cont.sum(1)                                      # |autor|
    n = cont.sum()
    prec = (cont ** 2 / size_c[None, :]).sum() / n
    rec = (cont ** 2 / size_t[:, None]).sum() / n
    f = 2 * prec * rec / (prec + rec)
    return prec, rec, f


def per_block(df: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    rows = []
    for block, sub in df.loc[pred.index].groupby("block", sort=False):
        p, r, f = bcubed(sub.author_id.values, pred.loc[sub.index].values)
        rows.append({"block": block, "subset": sub.subset.iloc[0], "b3p": p, "b3r": r, "b3f": f,
                     "authorships": len(sub), "true_clusters": sub.author_id.nunique(),
                     "pred_clusters": pred.loc[sub.index].nunique()})
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, pred: pd.Series, blocks_table: pd.DataFrame | None = None) -> dict:
    bt = per_block(df, pred) if blocks_table is None else blocks_table
    sub = df.loc[pred.index]
    P, R, F = bcubed(sub.author_id.values, pred.values)
    return {"micro_b3p": bt.b3p.mean(), "micro_b3r": bt.b3r.mean(), "micro_b3f": bt.b3f.mean(),
            "macro_b3p": P, "macro_b3r": R, "macro_b3f": F,
            "true_clusters": bt.true_clusters.mean(), "pred_clusters": bt.pred_clusters.mean()}


SUBSETS = {"todos": ["test_large", "test_random"], "grandes": ["test_large"], "aleatorios": ["test_random"]}


def evaluate_subsets(df: pd.DataFrame, pred: pd.Series) -> dict:
    """Métricas para todos los bloques de prueba, los grandes y los aleatorios."""
    bt = per_block(df, pred)
    out = {}
    for name, subs in SUBSETS.items():
        mask = df.loc[pred.index].subset.isin(subs)
        p = pred[mask.values]
        out[name] = aggregate(df, p, bt[bt.subset.isin(subs)])
    return out, bt


def pairwise_report(y: np.ndarray, yhat: np.ndarray) -> pd.DataFrame:
    rep = classification_report(y, yhat, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T
