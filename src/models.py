"""
Clasificadores de pares: Decision Tree, Random Forest y Hist-GBDT.

Procedimiento del artículo (Fig. 2 y sección "Results"):
  * cinco proporciones de clases en el entrenamiento (submuestreo);
  * búsqueda aleatoria de hiperparámetros con validación cruzada de 5 particiones;
  * dos funciones objetivo: accuracy y average precision  -> 3 x 5 x 2 = 30 modelos.

[SUPUESTO] las particiones de la validación cruzada se hacen por BLOQUE
(GroupKFold), para que pares del mismo bloque no queden a la vez en
entrenamiento y validación (el artículo divide "a nivel de bloques").
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

from . import config as C

ALGORITHMS = {
    "Decision Tree": (DecisionTreeClassifier, {
        "criterion": ["gini", "entropy"], "max_depth": [5, 10, 15, 25, None],
        "min_samples_leaf": [1, 5, 10, 50, 100], "max_features": [None, "sqrt", 0.5]}),
    "Random Forest": (RandomForestClassifier, {
        "n_estimators": [10, 30, 50, 100], "criterion": ["gini", "entropy"],
        "max_depth": [5, 10, 15, 25, None], "min_samples_leaf": [1, 5, 10, 50],
        "max_features": ["sqrt", 0.3]}),
    "Hist-GBDT": (HistGradientBoostingClassifier, {
        "learning_rate": loguniform(0.02, 0.3), "max_depth": [3, 5, 10, None],
        "max_iter": randint(50, 300), "min_samples_leaf": [20, 50, 100]}),
}


def make_model(algo: str, params: dict | None = None, seed: int = C.SEED):
    cls, _ = ALGORITHMS[algo]
    kw = {"random_state": seed}
    if algo == "Random Forest":
        kw["n_jobs"] = -1
    return cls(**kw, **(params or {}))


def downsample(y: np.ndarray, ratio: tuple[int, int] | None, seed: int = C.SEED) -> np.ndarray:
    """
    Índices de un submuestreo con proporción {0:a, 1:b}.
    Se reduce la clase sobrante; la otra se conserva completa.
    """
    idx = np.arange(len(y))
    if ratio is None:
        return idx
    a, b = ratio
    rng = np.random.default_rng(seed)
    i0, i1 = idx[y == 0], idx[y == 1]
    if len(i0) * b > len(i1) * a:          # sobra clase 0
        i0 = rng.choice(i0, size=int(len(i1) * a / b), replace=False)
    else:                                  # sobra clase 1
        i1 = rng.choice(i1, size=int(len(i0) * b / a), replace=False)
    return np.sort(np.concatenate([i0, i1]))


def tune(algo: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_iter: int = C.N_ITER_SEARCH,
         seed: int = C.SEED) -> dict:
    """
    Búsqueda aleatoria con dos métricas a la vez. Devuelve los mejores
    hiperparámetros para cada métrica y la tabla completa de resultados.
    """
    _, space = ALGORITHMS[algo]
    search = RandomizedSearchCV(
        make_model(algo, seed=seed), space, n_iter=n_iter, cv=GroupKFold(n_splits=C.CV_FOLDS),
        scoring=C.SCORINGS, refit=False, random_state=seed, n_jobs=1 if algo == "Random Forest" else -1)
    search.fit(X, y, groups=groups)
    res = pd.DataFrame(search.cv_results_)
    best = {}
    for name in C.SCORINGS:
        r = res.loc[res[f"mean_test_{name}"].idxmax()]
        best[name] = {"params": r["params"], "cv_score": float(r[f"mean_test_{name}"])}
    return {"best": best, "cv_results": res}


def train_all(train_pairs: pd.DataFrame, features: list[str], algos=None, ratios=None,
              n_iter: int = C.N_ITER_SEARCH, seed: int = C.SEED, verbose: bool = True) -> list[dict]:
    """
    Entrena la grilla completa del artículo:
    algoritmo x proporción de clases x función objetivo.
    """
    algos = algos or list(ALGORITHMS)
    ratios = ratios or C.RATIOS
    y_all = train_pairs.y.to_numpy()
    X_all = train_pairs[features].to_numpy(np.float32)
    g_all = train_pairs.block.to_numpy()
    out = []
    for algo in algos:
        for rname, ratio in ratios.items():
            t0 = time.time()
            sel = downsample(y_all, ratio, seed)
            X, y, g = X_all[sel], y_all[sel], g_all[sel]
            # [ESCALADO] la búsqueda usa a lo sumo C.MAX_SEARCH_ROWS pares (el artículo
            # también limita a 100.000 registros en su análisis de tiempos); el modelo
            # final se reentrena con todos los pares de la proporción elegida.
            s_idx = np.arange(len(y))
            if len(y) > C.MAX_SEARCH_ROWS:
                s_idx = np.sort(np.random.default_rng(seed).choice(len(y), C.MAX_SEARCH_ROWS, replace=False))
            tuned = tune(algo, X[s_idx], y[s_idx], g[s_idx], n_iter=n_iter, seed=seed)
            fitted = {}
            for sname, info in tuned["best"].items():
                key = tuple(sorted(info["params"].items(), key=lambda kv: kv[0]))
                if key not in fitted:
                    fitted[key] = make_model(algo, info["params"], seed).fit(X, y)
                out.append({"algorithm": algo, "ratio": rname, "scoring": sname, "params": info["params"],
                            "cv_score": info["cv_score"], "n_train": len(y), "pct_class1_train": y.mean(),
                            "model": fitted[key]})
            if verbose:
                msg = f"{algo:14s} {rname:10s} n={len(y):7d}  {time.time() - t0:5.1f}s"
                print(msg)
                with open(C.PROCESSED / "train_log.txt", "a") as fh:
                    fh.write(msg + "\n")
    return out
