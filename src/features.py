"""
Variables de comparación por pares (Tabla 2 del artículo).

Todas las comparaciones se hacen SOLO dentro de cada bloque. Para que el cálculo
sea rápido incluso en bloques de ~800 autorías (~320 mil pares) se usan matrices
dispersas: p. ej. la intersección de coautores entre todas las autorías del
bloque es simplemente  M · Mᵀ  con M la matriz binaria autoría × coautor.

Códigos para variables categóricas de igualdad:
    1 = iguales, 0 = distintos, -1 = falta en una autoría, -2 = falta en ambas
Para similitudes numéricas, -1 indica que el campo falta en alguna autoría.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist
from scipy import sparse, stats
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config as C
from . import names as N
from .simulation import TOP_JOURNALS

FEATURES = [
    "first_name_equal", "first_name_jw", "first_name_phonetic", "middle_name_equal",
    "middle_initial_equal", "signature_compatible", "matching_signatures",
    "affil_jw", "country_match", "city_match",
    "journal_top", "journal_astro", "journal_astron", "journal_astrop", "journal_geo", "journal_earth",
    "year_diff", "age_max",
    "top_words_match", "top_bigrams_match", "title_sim", "abstract_sim", "acronyms_match",
    "keywords_match", "coauthors_match", "block_size_pct",
]
NAME_FEATURES = FEATURES[:7]
FEATURE_GROUPS = {
    "Nombre": NAME_FEATURES, "Afiliación": ["affil_jw", "country_match", "city_match"],
    "Revista": [f for f in FEATURES if f.startswith("journal")], "Año": ["year_diff", "age_max"],
    "Título/resumen": ["top_words_match", "top_bigrams_match", "title_sim", "abstract_sim", "acronyms_match"],
    "Palabras clave": ["keywords_match"], "Coautores": ["coauthors_match"], "Bloque": ["block_size_pct"],
}

# [SUPUESTO] distribución de referencia del tamaño de bloque en ADS: lognormal con
# media 21 y desviación ~50 (valores reportados en el artículo, p. 3899).
_SIG = np.sqrt(np.log(1 + (50 / 21) ** 2))
_REF_BLOCK = stats.lognorm(s=_SIG, scale=21 / np.exp(_SIG ** 2 / 2))


def _split(col: pd.Series, sep: str) -> list[list[str]]:
    return [[t.strip() for t in s.split(sep) if t.strip()] if isinstance(s, str) and s else [] for s in col]


def _binary(lists, vocab=None):
    cv = CountVectorizer(analyzer=lambda x: x, binary=True, vocabulary=vocab, lowercase=False)
    return cv.fit_transform(lists).astype(np.float32).tocsr()


def _top_k(tfidf: sparse.csr_matrix, k=5) -> sparse.csr_matrix:
    """Matriz binaria con los k términos de mayor TF-IDF de cada documento."""
    rows, cols = [], []
    for r in range(tfidf.shape[0]):
        s, e = tfidf.indptr[r], tfidf.indptr[r + 1]
        if e > s:
            top = tfidf.indices[s:e][np.argsort(-tfidf.data[s:e])[:k]]
            rows += [r] * len(top)
            cols += list(top)
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=tfidf.shape)


def corpus_representations(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    """
    Representaciones a nivel de autoría que no dependen del par:
    top-5 palabras y bigramas, embeddings de título y resumen (25 dim),
    matrices binarias de coautores / países / ciudades, palabras clave y acrónimos
    con sus pesos IDF.

    Sustituto del Doc2Vec del artículo: TF-IDF + SVD truncada (LSA) de 25
    dimensiones, ajustada solo con los bloques de entrenamiento.
    """
    text = (df.title.fillna("") + " " + df.abstract.fillna("")).str.strip()
    uni = TfidfVectorizer(token_pattern=r"[a-z]+", min_df=2).fit_transform(text)
    # min_df=3: solo bigramas que se repiten en el corpus (colocaciones), no combinaciones al azar
    bi = TfidfVectorizer(token_pattern=r"[a-z]+", ngram_range=(2, 2), min_df=3).fit_transform(text)
    train = (df.subset == "train").values
    emb = {}
    for field in ("title", "abstract"):
        vec = TfidfVectorizer(token_pattern=r"[a-z]+", sublinear_tf=True)
        vec.fit(df.loc[train, field].fillna(""))
        X = vec.transform(df[field].fillna(""))
        svd = TruncatedSVD(n_components=C.SIM["embedding_dim"], random_state=seed).fit(X[train])
        E = normalize(svd.transform(X)).astype(np.float32)
        emb[field] = (E, (df[field].fillna("") != "").values)
    kw = _binary(_split(df.keywords, ";"))
    acr = _binary(_split(df.acronyms, " "))
    n = len(df)
    idf = lambda M: np.log((1 + n) / (1 + np.asarray(M.sum(0)).ravel())) + 1  # noqa: E731
    return {
        "top_words": _top_k(uni), "top_bigrams": _top_k(bi), "emb": emb,
        "keywords": kw, "keywords_idf": idf(kw).astype(np.float32),
        "acronyms": acr, "acronyms_idf": idf(acr).astype(np.float32),
        "coauthors": _binary(_split(df.coauthors, ";")),
        "countries": _binary(_split(df.countries, "|")), "cities": _binary(_split(df.cities, "|")),
    }


def _eq_code(a: np.ndarray, b: np.ndarray, present_a: np.ndarray, present_b: np.ndarray) -> np.ndarray:
    out = np.where(a == b, 1, 0).astype(np.int8)
    out[present_a ^ present_b] = -1
    out[~present_a & ~present_b] = -2
    return out


def _gram(M: sparse.csr_matrix, idx: np.ndarray, weights=None) -> np.ndarray:
    sub = M[idx]
    left = sub.multiply(weights).tocsr() if weights is not None else sub
    return (left @ sub.T).toarray()


def name_features(blk: pd.DataFrame, iu: np.ndarray, ju: np.ndarray) -> dict:
    """Variables de nombre (dependen solo de first_part / middle_part / signature)."""
    first = blk.first_part.fillna("").to_numpy()
    middle = blk.middle_part.fillna("").to_numpy()
    f_full = np.array([len(x) > 1 for x in first])
    m_full = np.array([len(x) > 1 for x in middle])
    m_any = np.array([len(x) > 0 for x in middle])
    uf, inv_f = np.unique(first, return_inverse=True)
    jw = cdist(list(uf), list(uf), scorer=JaroWinkler.normalized_similarity, dtype=np.float32)
    ph = np.array([N.phonetic(x) for x in uf])
    out = {}
    out["first_name_equal"] = _eq_code(first[iu], first[ju], f_full[iu], f_full[ju])
    both_f = f_full[iu] & f_full[ju]
    out["first_name_jw"] = np.where(both_f, jw[inv_f[iu], inv_f[ju]], -1).astype(np.float32)
    out["first_name_phonetic"] = np.where(both_f, (ph[inv_f[iu]] == ph[inv_f[ju]]).astype(np.int8), -1)
    out["middle_name_equal"] = _eq_code(middle[iu], middle[ju], m_full[iu], m_full[ju])
    mi = np.array([x[:1] for x in middle])
    out["middle_initial_equal"] = _eq_code(mi[iu], mi[ju], m_any[iu], m_any[ju])
    sigs = blk.signature.to_numpy()
    us, inv_s = np.unique(sigs, return_inverse=True)
    comp = N.compatibility_matrix(list(us))
    out["signature_compatible"] = comp[inv_s[iu], inv_s[ju]].astype(np.int8)
    # ¿las firmas compatibles con s1 son también compatibles con s2 y viceversa?
    same_set = (comp[:, None, :] == comp[None, :, :]).all(-1)
    out["matching_signatures"] = same_set[inv_s[iu], inv_s[ju]].astype(np.int8)
    return out


def block_pairs(df: pd.DataFrame, reps: dict, block: str, names_only: bool = False) -> pd.DataFrame:
    """Todas las parejas (i<j) de un bloque con sus variables y la etiqueta."""
    idx = np.flatnonzero(df.block.values == block)
    blk = df.iloc[idx]
    n = len(idx)
    iu, ju = np.triu_indices(n, k=1)
    out = {"block": np.repeat(block, len(iu)), "i": idx[iu], "j": idx[ju]}
    out.update(name_features(blk, iu, ju))
    if not names_only:
        aff = blk.affiliation.fillna("").to_numpy()
        ua, inv_a = np.unique(aff, return_inverse=True)
        jw_a = cdist(list(ua), list(ua), scorer=JaroWinkler.normalized_similarity, dtype=np.float32)
        has_aff = aff != ""
        out["affil_jw"] = np.where(has_aff[iu] & has_aff[ju], jw_a[inv_a[iu], inv_a[ju]], -1).astype(np.float32)
        for key, name in (("countries", "country_match"), ("cities", "city_match")):
            G = _gram(reps[key], idx)
            has = np.diag(G) > 0
            out[name] = np.where(has[iu] & has[ju], (G[iu, ju] > 0).astype(np.int8), -1).astype(np.int8)
        jour = blk.journal.to_numpy()
        words = [j.lower().split() for j in jour]
        flags = {"journal_top": np.array([j in TOP_JOURNALS for j in jour])}
        for pre in ("astro", "astron", "astrop", "geo", "earth"):
            flags[f"journal_{pre}"] = np.array([any(w.startswith(pre) for w in ws) for ws in words])
        for k, v in flags.items():
            out[k] = (v[iu] & v[ju]).astype(np.int8)
        yr = blk.year.to_numpy()
        out["year_diff"] = np.abs(yr[iu] - yr[ju]).astype(np.int16)
        out["age_max"] = (2018 - np.minimum(yr[iu], yr[ju])).astype(np.int16)
        for key, name in (("top_words", "top_words_match"), ("top_bigrams", "top_bigrams_match")):
            G = _gram(reps[key], idx)
            has = np.diag(G) > 0
            out[name] = np.where(has[iu] & has[ju], (G[iu, ju] > 0).astype(np.int8), -1).astype(np.int8)
        for field, name in (("title", "title_sim"), ("abstract", "abstract_sim")):
            E, has = reps["emb"][field]
            S = E[idx] @ E[idx].T
            h = has[idx]
            out[name] = np.where(h[iu] & h[ju], S[iu, ju], -1).astype(np.float32)
        for key, name in (("acronyms", "acronyms_match"), ("keywords", "keywords_match")):
            G = _gram(reps[key], idx, reps[f"{key}_idf"])
            has = np.asarray(reps[key][idx].sum(1)).ravel() > 0
            out[name] = np.where(has[iu] & has[ju], G[iu, ju], -1).astype(np.float32)
        G = _gram(reps["coauthors"], idx)
        out["coauthors_match"] = (G[iu, ju] > 0).astype(np.int8)
        out["block_size_pct"] = np.full(len(iu), _REF_BLOCK.cdf(n), dtype=np.float32)
    a = blk.author_id.to_numpy()
    out["y"] = (a[iu] == a[ju]).astype(np.int8)
    return pd.DataFrame(out)


def all_pairs(df: pd.DataFrame, reps: dict | None = None, names_only: bool = False) -> pd.DataFrame:
    if reps is None and not names_only:
        reps = corpus_representations(df)
    parts = [block_pairs(df, reps, b, names_only) for b in df.block.unique()]
    pairs = pd.concat(parts, ignore_index=True)
    sub = df.subset.to_numpy()
    pairs.insert(1, "subset", sub[pairs.i.to_numpy()])
    return pairs


def load_or_build_pairs(df: pd.DataFrame, tag: str = "base") -> pd.DataFrame:
    """Calcula los pares una sola vez y los guarda en data/processed (no versionado)."""
    import hashlib
    key = pd.util.hash_pandas_object(df[["authorship_id", "signature", "affiliation", "author_id", "title"]],
                                     index=False).values
    path = C.PROCESSED / f"pairs_{tag}_{hashlib.md5(key.tobytes()).hexdigest()[:8]}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    pairs = all_pairs(df)
    pairs.to_pickle(path)
    return pairs
