"""
Explicaciones locales al estilo LIME (Ribeiro et al., 2016).

Implementación propia y compacta del principio de LIME tabular (el paquete
`lime` no compila en todas las versiones recientes de Python):
  1. se discretiza cada variable en cuartiles (o categorías si es discreta);
  2. se generan perturbaciones: cada variable conserva el valor de la instancia
     con probabilidad 1/2, o toma el valor de una fila de entrenamiento al azar;
  3. representación interpretable z_j = 1 si la perturbación cae en el mismo
     intervalo que la instancia;
  4. se ajusta una regresión Ridge ponderada por un kernel exponencial sobre la
     probabilidad de clase 1 del modelo. Los coeficientes son la contribución
     de cada condición ("coauthors_match = 1", "year_diff > 9", ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def _bins(col: np.ndarray):
    vals = np.unique(col)
    if len(vals) <= 6:
        return None
    return np.unique(np.quantile(col, [0.25, 0.5, 0.75]))


def _describe(name, value, edges):
    if edges is None:
        return f"{name} = {value:g}"
    k = np.searchsorted(edges, value, side="right")
    if k == 0:
        return f"{name} <= {edges[0]:.2f}"
    if k == len(edges):
        return f"{name} > {edges[-1]:.2f}"
    return f"{edges[k - 1]:.2f} < {name} <= {edges[k]:.2f}"


def explain_instance(model, x: np.ndarray, X_train: np.ndarray, features: list[str], n_samples: int = 4000,
                     n_top: int = 10, seed: int = 0) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    p = len(features)
    edges = [_bins(X_train[:, j]) for j in range(p)]

    def bin_of(v, j):
        return v if edges[j] is None else np.searchsorted(edges[j], v, side="right")

    rows = X_train[rng.integers(0, len(X_train), n_samples)]
    keep = rng.random((n_samples, p)) < 0.5
    Xs = np.where(keep, x[None, :], rows).astype(np.float32)
    Xs[0] = x
    Z = np.column_stack([bin_of(Xs[:, j], j) == bin_of(x[j], j) for j in range(p)]).astype(float)
    d = np.sqrt(((1 - Z) ** 2).sum(1))
    width = 0.75 * np.sqrt(p)
    w = np.exp(-(d ** 2) / width ** 2)
    target = model.predict_proba(Xs)[:, 1]
    reg = Ridge(alpha=1.0).fit(Z, target, sample_weight=w)
    prob = float(model.predict_proba(x[None, :].astype(np.float32))[0, 1])
    tab = pd.DataFrame({"condicion": [_describe(f, x[j], edges[j]) for j, f in enumerate(features)],
                        "peso": reg.coef_})
    tab = tab.reindex(tab.peso.abs().sort_values(ascending=False).index).head(n_top)
    return tab.reset_index(drop=True), prob
