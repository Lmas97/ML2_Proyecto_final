"""
Simulación de autorías al estilo ADS.

Idea general (modelo generativo):
  1. Se definen los bloques (apellido + inicial) con los tamaños y el número de
     perfiles reportados por el artículo (Tablas 3, 7 y 9).
  2. Cada perfil es un autor real con atributos latentes: nombre canónico,
     instituciones a lo largo de su carrera, temas de investigación, frases y
     acrónimos propios, grupo de coautores, revistas preferidas y años activos.
  3. Cada autoría hereda esos atributos con ruido (nombre abreviado, cambio de
     institución, coautores nuevos, campos faltantes según la Tabla 1).

El clasificador NUNCA ve los atributos latentes: solo los campos "observados"
(nombre tal como aparece, afiliación, título, resumen, etc.), a partir de los
cuales se calculan las variables de la Tabla 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import names as N

# --------------------------------------------------------------------------
# Catálogos del "mundo" simulado
# --------------------------------------------------------------------------
CITIES = [
    ("Beijing", "China"), ("Nanjing", "China"), ("Shanghai", "China"), ("Hefei", "China"),
    ("Kunming", "China"), ("Wuhan", "China"), ("Guangzhou", "China"),
    ("Seoul", "South Korea"), ("Daejeon", "South Korea"), ("Pohang", "South Korea"), ("Busan", "South Korea"),
    ("Pasadena", "USA"), ("Baltimore", "USA"), ("Tucson", "USA"), ("Berkeley", "USA"), ("Boulder", "USA"),
    ("Chicago", "USA"), ("Austin", "USA"), ("Princeton", "USA"), ("Ann Arbor", "USA"), ("Amherst", "USA"),
    ("Oxford", "UK"), ("London", "UK"), ("Edinburgh", "UK"), ("Manchester", "UK"),
    ("Berlin", "Germany"), ("Munich", "Germany"), ("Heidelberg", "Germany"), ("Bonn", "Germany"),
    ("Potsdam", "Germany"), ("Paris", "France"), ("Marseille", "France"), ("Toulouse", "France"),
    ("Rome", "Italy"), ("Padova", "Italy"), ("Bologna", "Italy"), ("Naples", "Italy"),
    ("Madrid", "Spain"), ("Granada", "Spain"), ("Tokyo", "Japan"), ("Kyoto", "Japan"),
    ("Moscow", "Russia"), ("Sao Paulo", "Brazil"), ("Puebla", "Mexico"), ("Santiago", "Chile"),
    ("Sydney", "Australia"), ("Leiden", "Netherlands"), ("Pune", "India"), ("Bogota", "Colombia"),
]
INST_PATTERNS = ["University of {c}", "{c} Observatory", "Institute of Astronomy, {c}",
                 "Center for Astrophysics {c}", "National Laboratory for Space Science {c}"]
DEPARTMENTS = ["Department of Physics", "Department of Astronomy", "School of Physics",
               "Key Laboratory of Dark Matter", "Space Research Division", ""]

TOP_JOURNALS = ["The Astrophysical Journal", "Astronomy and Astrophysics",
                "Monthly Notices of the Royal Astronomical Society", "The Astronomical Journal",
                "Nature", "Science"]
OTHER_JOURNALS = ["Astronomische Nachrichten", "Publications of the Astronomical Society of the Pacific",
                  "Icarus", "Journal of Geophysical Research", "Earth and Planetary Science Letters",
                  "Geophysical Research Letters", "Astrophysics and Space Science", "Physical Review D",
                  "Solar Physics", "Astroparticle Physics", "Acta Astronomica Sinica",
                  "Research in Astronomy and Astrophysics", "Astronomy Letters", "New Astronomy",
                  "Planetary and Space Science", "Journal of Cosmology and Astroparticle Physics",
                  "Classical and Quantum Gravity", "Physical Review Letters", "Space Science Reviews",
                  "Earth Moon and Planets", "Proceedings of the SPIE", "Advances in Space Research"]
JOURNALS = TOP_JOURNALS + OTHER_JOURNALS
KW_FACETS = ["galaxies", "stars", "cosmology", "Sun", "planets and satellites", "ISM",
             "instrumentation", "methods: numerical", "techniques: spectroscopic", "X-rays",
             "gravitational waves", "astroparticle physics"]

WESTERN_SURNAMES = [
    "smith", "johnson", "brown", "taylor", "anderson", "thomas", "jackson", "white", "harris", "martin",
    "thompson", "garcia", "martinez", "robinson", "clark", "rodriguez", "lewis", "walker", "hall", "allen",
    "young", "king", "wright", "lopez", "hill", "scott", "green", "adams", "baker", "gonzalez", "nelson",
    "mitchell", "perez", "roberts", "turner", "phillips", "campbell", "parker", "evans", "edwards",
    "collins", "stewart", "sanchez", "morris", "rogers", "reed", "cook", "morgan", "bell", "murphy",
    "bailey", "rivera", "cooper", "richardson", "cox", "howard", "ward", "torres", "peterson", "gray",
    "ramirez", "james", "watson", "brooks", "kelly", "sanders", "price", "bennett", "wood", "barnes",
    "ross", "henderson", "coleman", "jenkins", "perry", "powell", "long", "patterson", "hughes", "flores",
    "washington", "butler", "simmons", "foster", "gonzales", "bryant", "alexander", "russo", "griffin",
    "diaz", "hayes", "muller", "schmidt", "schneider", "fischer", "weber", "meyer", "wagner", "becker",
    "rossi", "ferrari", "esposito", "bianchi", "romano", "colombo", "ricci", "marino", "greco", "bruno",
    "dubois", "moreau", "laurent", "simon", "michel", "lefebvre", "leroy", "ivanov", "smirnov", "kuznetsov",
    "popov", "sokolov", "tanaka", "suzuki", "takahashi", "watanabe", "silva", "santos", "oliveira", "souza",
]
ASIAN_SURNAMES = {"chinese": ["li", "liu", "zhou", "huang", "zhao", "wu", "xu", "sun", "ma", "zhu"],
                  "korean": ["kim", "park", "choi", "jung", "kang", "cho", "yoon", "jang"]}
_CONS, _VOW = "bcdfghklmnprstvz", "aeiou"


def _make_words(rng, n, used):
    """Pseudopalabras pronunciables y únicas (vocabulario sintético)."""
    out = []
    while len(out) < n:
        k = rng.integers(2, 4)
        w = "".join(_CONS[rng.integers(0, len(_CONS))] + _VOW[rng.integers(0, 5)] for _ in range(k))
        w += _CONS[rng.integers(0, len(_CONS))] if rng.random() < 0.5 else ""
        if w not in used:
            used.add(w)
            out.append(w)
    return out


def build_world(rng: np.random.Generator) -> dict:
    """Instituciones, vocabulario por tema, frases, acrónimos, palabras clave y revistas."""
    S = C.SIM
    insts = []
    for city, country in CITIES:
        for pat in rng.choice(INST_PATTERNS, size=2, replace=False):
            insts.append({"name": pat.format(c=city), "city": city, "country": country})
    used = set()
    general = _make_words(rng, S["general_words"], used)
    # [SUPUESTO] los temas comparten vocabulario (todo es astronomía): las palabras
    # de cada tema se toman de un banco común, así que los temas se solapan.
    bank = _make_words(rng, S["topic_word_bank"], used)
    topics = []
    for t in range(S["n_topics"]):
        words = list(rng.choice(bank, size=S["words_per_topic"], replace=False))
        phrases = [f"{words[a]} {words[b]}" for a, b in rng.choice(len(words), size=(12, 2), replace=True) if a != b]
        acr = ["".join(rng.choice(list("ABCDEFGHIJKLMNOPRSTUVWXZ"), size=rng.integers(3, 6))) for _ in range(8)]
        kws = [f"{rng.choice(KW_FACETS)}: {w}" for w in rng.choice(words, size=15, replace=False)]
        geo = t >= S["n_topics"] - 4  # últimos temas: ciencias planetarias / de la Tierra
        jw = np.array([3.0 if (j in TOP_JOURNALS and not geo) else
                       (4.0 if geo and any(k in j for k in ("Geo", "Earth", "Planet", "Icarus")) else 0.4)
                       for j in JOURNALS])
        topics.append({"words": words, "phrases": phrases, "acronyms": acr, "keywords": kws,
                       "journal_w": jw / jw.sum()})
    co_pool = [f"{s}.{l}" for s in WESTERN_SURNAMES for l in "abcdefghjklmnprst"]
    co_pool += [f"{s}.{l}" for o in ASIAN_SURNAMES.values() for s in o for l in "hjlswxyz"]
    big_collab = list(rng.choice(co_pool, size=600, replace=False))
    return {"institutions": insts, "general": general, "topics": topics,
            "coauthor_pool": co_pool, "big_collab": big_collab}


# --------------------------------------------------------------------------
# Especificación de bloques
# --------------------------------------------------------------------------
def _train_block_sizes(n_blocks, total, median, cap=350, nmin=3):
    """Tamaños lognormales con mediana fija; sigma se ajusta para sumar `total`."""
    q = (np.arange(n_blocks) + 0.5) / n_blocks

    def sizes(sig):
        return np.clip(np.round(stats.lognorm.ppf(q, s=sig, scale=median)), nmin, cap).astype(int)

    lo, hi = 0.1, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sizes(mid).sum() < total else (lo, mid)
    s = sizes(hi)
    s[-1] += total - s.sum()  # ajuste fino en el bloque más grande
    return s


def _train_profiles(rng, sizes, mean_target):
    """[SUPUESTO] 55 % de bloques con un único autor; el resto crece con el tamaño."""
    u_one = rng.random(len(sizes))
    u_k = rng.random(len(sizes))

    def ks(lam):
        k = 1 + stats.poisson.ppf(u_k, lam * np.sqrt(sizes / 5.0)).astype(int)
        k[u_one < 0.55] = 1
        return np.minimum(k, sizes)

    lo, hi = 0.01, 20
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ks(mid).mean() < mean_target else (lo, mid)
    return ks(hi)


def build_block_specs(rng: np.random.Generator) -> pd.DataFrame:
    S = C.SIM
    rows = []
    sc = S["large_block_scale"]
    for name, n, k, origin in C.LARGE_BLOCKS:
        rows.append({"block": name, "subset": "test_large", "n": int(round(n * sc)),
                     "k": max(min(k, 2), int(round(k * sc))), "origin": origin,
                     "n_article": n, "k_article": k})
    for name, n, k in C.RANDOM_BLOCKS:
        rows.append({"block": name, "subset": "test_random", "n": n, "k": k, "origin": "western",
                     "n_article": n, "k_article": k})
    sizes = _train_block_sizes(S["train_blocks"], S["train_authorships"], S["train_median_block"])
    ks = _train_profiles(rng, sizes, S["train_profiles_mean"])
    used = {r["block"] for r in rows}
    order = rng.permutation(len(sizes))
    for i in order:
        while True:
            if rng.random() < S["train_share_asian"]:
                origin = str(rng.choice(["chinese", "korean"]))
                surname = str(rng.choice(ASIAN_SURNAMES[origin]))
                initial = str(rng.choice(list(N.ASIAN_FIRST_SYLLABLE[origin].keys())))
            else:
                origin, surname = "western", str(rng.choice(WESTERN_SURNAMES))
                initial = "abcdefghijklmnoprstvw"[rng.integers(0, 21)]
            key = f"{surname}.{initial}"
            if key not in used:
                used.add(key)
                break
        rows.append({"block": key, "subset": "train", "n": int(sizes[i]), "k": int(ks[i]),
                     "origin": origin, "n_article": np.nan, "k_article": np.nan})
    return pd.DataFrame(rows)


def _allocate(n, k, s):
    """Reparte n autorías entre k autores con pesos Zipf (cada autor >= 1)."""
    if k == 1:
        return np.array([n])
    w = np.arange(1, k + 1, dtype=float) ** (-s)
    rest = n - k
    raw = rest * w / w.sum()
    c = np.floor(raw).astype(int)
    c[np.argsort(-(raw - c))[: rest - c.sum()]] += 1
    return c + 1


def _bisect_s(ns, ks, target, fixed_num=0.0, fixed_den=0.0):
    """Exponente s tal que (pares clase 1) / (pares totales) = target."""
    def frac(s):
        num = sum((c * (c - 1)).sum() for c in (_allocate(n, k, s) for n, k in zip(ns, ks)))
        return (num + fixed_num) / ((np.asarray(ns) * (np.asarray(ns) - 1)).sum() + fixed_den)

    lo, hi = 0.0, 12.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if frac(mid) < target else (lo, mid)
    return round(hi, 4)


def calibrate_zipf(specs: pd.DataFrame) -> pd.Series:
    """
    Exponente Zipf de productividad para cada bloque, calibrado para reproducir
    el % de pares de clase 1 de la Tabla 3 (y los supuestos por bloque).
    """
    s_block = pd.Series(np.nan, index=specs.index)
    fixed = C.SIM["block_class1"]
    for b, target in fixed.items():
        row = specs[specs.block == b].iloc[0]
        s_block[row.name] = _bisect_s([row.n], [row.k], target)
    for subset, target in C.SIM["target_class1"].items():
        sub = specs[(specs.subset == subset) & ~specs.block.isin(fixed)]
        fx = specs[(specs.subset == subset) & specs.block.isin(fixed)]
        fnum = sum((c * (c - 1)).sum() for c in (_allocate(n, k, s) for n, k, s in
                                                  zip(fx.n, fx.k, s_block[fx.index])))
        fden = float((fx.n * (fx.n - 1)).sum())
        s_block[sub.index] = _bisect_s(list(sub.n), list(sub.k), target, fnum, fden)
    return s_block


# --------------------------------------------------------------------------
# Autores y autorías
# --------------------------------------------------------------------------
def _make_author(rng, world, block_row, initial, prev_authors):
    S = C.SIM
    origin = block_row["origin"]
    first, middle = N.sample_canonical_name(rng, initial, origin, S["p_middle_western"],
                                            S["p_two_syllables_asian"])
    insts = world["institutions"]
    home_country = {"chinese": "China", "korean": "South Korea"}.get(origin)
    if home_country and rng.random() < S["p_home_country"]:
        cands = [i for i, x in enumerate(insts) if x["country"] == home_country]
    else:
        cands = list(range(len(insts)))
    inst1 = int(rng.choice(cands))
    inst2 = int(rng.integers(0, len(insts))) if rng.random() < S["p_move"] else inst1
    start = int(rng.integers(1975, 2015))
    end = int(min(2018, start + 3 + rng.exponential(12)))
    move_year = int(rng.integers(start, end + 1))
    T = S["n_topics"]
    if prev_authors and rng.random() < S["p_shared_topic"]:
        main = prev_authors[int(rng.integers(0, len(prev_authors)))]["topics"][0]   # misma subárea
    else:
        main = int(rng.integers(0, T))
    secondary = [int(t) for t in rng.choice(T, size=int(rng.integers(1, 3)), replace=False) if t != main]
    tw = np.array([0.55] + [0.45 / len(secondary)] * len(secondary)) if secondary else np.array([1.0])
    topics = [main] + secondary
    words = world["topics"][main]["words"]
    phrases = [f"{words[a]} {words[b]}" for a, b in rng.choice(len(words), size=(3, 2))]
    acr = ["".join(rng.choice(list("ABCDEFGHIJKLMNOPRSTUVWXZ"), size=rng.integers(3, 6))) for _ in range(3)]
    pool = list(rng.choice(world["coauthor_pool"], size=int(rng.integers(5, 26)), replace=False))
    # colegas de la misma institución comparten parte del grupo de coautores
    same_inst = [a for a in prev_authors if a["inst1"] == inst1]
    if same_inst:
        other = same_inst[0]["coauthors"]
        pool += list(rng.choice(other, size=max(1, int(S["p_shared_coauthors"] * len(other))), replace=False))
    jw = sum(w * world["topics"][t]["journal_w"] for t, w in zip(topics, tw))
    jpref = rng.dirichlet(jw * 20 + 0.01)
    return {"first": first, "middle": middle, "inst1": inst1, "inst2": inst2, "move_year": move_year,
            "start": start, "end": end, "topics": topics, "topic_w": tw, "phrases": phrases,
            "acronyms": acr, "coauthors": pool, "journal_p": jpref}


def _affiliation_string(rng, inst):
    name = inst["name"]
    if rng.random() < 0.3:
        name = name.replace("University", "Univ.").replace("Observatory", "Obs.").replace("Institute", "Inst.")
    dept = DEPARTMENTS[rng.integers(0, len(DEPARTMENTS))]
    parts = [p for p in (dept, name, inst["city"], inst["country"]) if p]
    return ", ".join(parts)


def _text(rng, world, author, length, n_phrases, focus):
    """Texto de una autoría: palabras del tema central del artículo (`focus`),
    de los demás temas del autor y vocabulario general."""
    toks = []
    topics, tw = author["topics"], author["topic_w"]
    for _ in range(length):
        if rng.random() < C.SIM["p_topic_word"]:
            t = focus if rng.random() < 0.75 else topics[rng.choice(len(topics), p=tw)]
            ws = world["topics"][t]["words"]
            toks.append(ws[min(int(rng.exponential(20)), len(ws) - 1)])   # palabras frecuentes del tema
        else:
            toks.append(world["general"][int(rng.integers(0, len(world["general"])))])
    for _ in range(n_phrases):
        src = author["phrases"] if rng.random() < 0.5 else world["topics"][focus]["phrases"]
        pos = int(rng.integers(0, len(toks) + 1))
        toks[pos:pos] = src[int(rng.integers(0, len(src)))].split()
    return " ".join(toks)


def _authorship(rng, world, author, block_row, surname):
    S, cov = C.SIM, C.SIM["coverage"]
    year = int(rng.integers(author["start"], author["end"] + 1))
    insts = world["institutions"]
    inst = author["inst1"] if year < author["move_year"] else author["inst2"]
    affs, cities, countries = [], set(), set()
    if rng.random() < cov["affiliation"]:
        chosen = [int(rng.integers(0, len(insts)))] if rng.random() < S["p_visiting_affil"] else [inst]
        if rng.random() < S["p_double_affil"]:
            chosen.append(int(rng.integers(0, len(insts))))
        for c in chosen:
            affs.append(_affiliation_string(rng, insts[c]))
            if rng.random() > S["p_ner_fail"]:
                cities.add(insts[c]["city"])
            if rng.random() > S["p_ner_fail"]:
                countries.add(insts[c]["country"])
    # cada artículo trata sobre uno de los temas del autor (no siempre el principal)
    focus = author["topics"][rng.choice(len(author["topics"]), p=author["topic_w"])]
    title = _text(rng, world, author, int(rng.integers(*S["title_len"])), int(rng.random() < 0.6), focus) \
        if rng.random() < cov["title"] else ""
    abstract = _text(rng, world, author, int(rng.integers(*S["abstract_len"])), int(rng.integers(1, 4)), focus) \
        if rng.random() < cov["abstract"] else ""
    kws = []
    if rng.random() < cov["keywords"]:
        for _ in range(int(rng.integers(3, 8))):
            t = author["topics"][rng.choice(len(author["topics"]), p=author["topic_w"])]
            kws.append(world["topics"][t]["keywords"][int(rng.integers(0, 15))])
    n_acr = rng.choice(4, p=[0.35, 0.35, 0.2, 0.1])
    acr = [author["acronyms"][int(rng.integers(0, 3))] if rng.random() < 0.6 else
           world["topics"][author["topics"][0]]["acronyms"][int(rng.integers(0, 8))] for _ in range(n_acr)]
    n_co = 1 + rng.poisson(3)
    co = [author["coauthors"][int(rng.integers(0, len(author["coauthors"])))] if rng.random() < S["p_usual_coauthor"]
          else world["coauthor_pool"][int(rng.integers(0, len(world["coauthor_pool"])))] for _ in range(n_co)]
    if rng.random() < S["p_big_collab"]:
        co += list(rng.choice(world["big_collab"], size=int(rng.integers(100, 300)), replace=False))
    journal = JOURNALS[rng.choice(len(JOURNALS), p=author["journal_p"])]
    return {"year": year, "affiliation": "; ".join(affs), "cities": "|".join(sorted(cities)),
            "countries": "|".join(sorted(countries)), "title": title, "abstract": abstract,
            "keywords": "; ".join(dict.fromkeys(kws)), "acronyms": " ".join(dict.fromkeys(acr)),
            "coauthors": "; ".join(dict.fromkeys(co)), "journal": journal}


def render_names(df: pd.DataFrame, p_abbrev: float) -> pd.DataFrame:
    """(Re)calcula el nombre observado de cada autoría. Usado también en la extensión."""
    S = C.SIM
    out = df.copy()
    parts = [N.render_name(f, m, o, ua, uf, un, p_abbrev,
                           S["p_keep_middle_initial"] if o == "western" else S["p_keep_middle_initial_asian"],
                           S["p_name_typo"], S["p_nickname"])
             for f, m, o, ua, uf, un in zip(out.latent_first, out.latent_middle.fillna(""), out.origin,
                                            out.u_abbr, out.u_form, out.u_noise)]
    out["first_part"] = [p[0] for p in parts]
    out["middle_part"] = [p[1] for p in parts]
    out["signature"] = [N.signature(f, m) for f, m in parts]
    out["name"] = [N.display_name(s, f, m) for s, (f, m) in zip(out.surname, parts)]
    return out


def simulate(seed: int = C.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera todo el conjunto simulado.

    Devuelve (autorías, especificación de bloques con su exponente Zipf).
    """
    rng = np.random.default_rng(seed)
    world = build_world(rng)
    specs = build_block_specs(rng)
    specs["zipf_s"] = calibrate_zipf(specs)
    rows = []
    for _, b in specs.iterrows():
        surname, initial = b["block"].rsplit(".", 1)
        counts = _allocate(b["n"], b["k"], b["zipf_s"])
        authors = []
        for a_idx, cnt in enumerate(counts):
            a = _make_author(rng, world, b, initial, authors)
            authors.append(a)
            for _ in range(cnt):
                rec = _authorship(rng, world, a, b, surname)
                rec.update({"block": b["block"], "subset": b["subset"], "origin": b["origin"],
                            "surname": surname, "author_id": f"{b['block']}#{a_idx:03d}",
                            "latent_first": a["first"], "latent_middle": a["middle"],
                            "u_abbr": rng.random(), "u_form": rng.random(), "u_noise": rng.random()})
                rows.append(rec)
    df = pd.DataFrame(rows)
    # orden aleatorio dentro de cada bloque (el orden no debe delatar al autor)
    df = df.sample(frac=1.0, random_state=seed).sort_values("block", kind="stable").reset_index(drop=True)
    df.insert(0, "authorship_id", [f"A{i:05d}" for i in range(len(df))])
    df = render_names(df, C.SIM["p_abbreviated"])
    cols = ["authorship_id", "block", "subset", "name", "first_part", "middle_part", "signature",
            "affiliation", "cities", "countries", "year", "journal", "title", "abstract", "keywords",
            "acronyms", "coauthors", "author_id", "surname", "origin", "latent_first", "latent_middle",
            "u_abbr", "u_form", "u_noise"]
    return df[cols], specs
