"""
Agrupamiento de autorías en perfiles de autor.

* `graph_clusters`: grafo de autorías (arista si el clasificador predice clase 1,
  peso = probabilidad) + comunidades por *label propagation* asíncrono
  (NetworkX), exactamente como en el artículo.
* `milojevic_hybrid`: línea base basada solo en iniciales (Milojević, 2013).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from . import config as C


def graph_clusters(n_nodes: int, i: np.ndarray, j: np.ndarray, proba: np.ndarray,
                   threshold: float = C.EDGE_THRESHOLD, seed: int = C.SEED) -> np.ndarray:
    """Etiqueta de comunidad (0..k-1) de cada nodo local 0..n_nodes-1."""
    keep = proba >= threshold
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_weighted_edges_from(zip(i[keep].tolist(), j[keep].tolist(), proba[keep].tolist()))
    labels = np.empty(n_nodes, dtype=int)
    for c, comm in enumerate(nx.community.asyn_lpa_communities(G, weight="weight", seed=seed)):
        labels[list(comm)] = c
    return labels


def cluster_all_blocks(df: pd.DataFrame, pairs: pd.DataFrame, proba: np.ndarray,
                       threshold: float = C.EDGE_THRESHOLD, seed: int = C.SEED) -> pd.Series:
    """Asigna un perfil predicho ('bloque#k') a cada autoría de los bloques presentes en `pairs`."""
    pred = {}
    for block, idx in pairs.groupby("block", sort=False).indices.items():
        gi, gj = pairs.i.to_numpy()[idx], pairs.j.to_numpy()[idx]
        nodes = np.unique(np.concatenate([gi, gj]))
        local = {g: k for k, g in enumerate(nodes)}
        li = np.fromiter((local[x] for x in gi), int, len(gi))
        lj = np.fromiter((local[x] for x in gj), int, len(gj))
        lab = graph_clusters(len(nodes), li, lj, proba[idx], threshold, seed)
        for g, lb in zip(nodes, lab):
            pred[g] = f"{block}#{lb}"
    return pd.Series(pred).reindex(df.index[df.block.isin(pairs.block.unique())])


def milojevic_hybrid(df: pd.DataFrame) -> pd.Series:
    """
    Regla híbrida de Milojević (2013): dentro de cada apellido + inicial,
    si aparece MÁS DE UNA inicial de segundo nombre se separa por todas las
    iniciales ('all initials'); en otro caso todo el bloque es un único autor
    ('first initial').
    """
    mid = df.middle_part.fillna("").str[:1]
    out = pd.Series(index=df.index, dtype=object)
    for block, idx in df.groupby("block").groups.items():
        m = mid.loc[idx]
        if m[m != ""].nunique() > 1:
            out.loc[idx] = block + "#" + m
        else:
            out.loc[idx] = block + "#0"
    return out
