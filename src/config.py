"""
Parámetros globales del proyecto.

Aquí se concentran:
  * las cifras REPORTADAS por Mihaljević & Santamaría (2021) que la simulación
    intenta reproducir (Tablas 1, 3, 4, 6, 7, 8 y 9 del artículo), y
  * los SUPUESTOS que fue necesario tomar porque el artículo no reporta el dato.

Cada parámetro de simulación lleva la etiqueta [ARTÍCULO], [SUPUESTO] o
[ESCALADO] para que sea fácil rastrear de dónde sale cada número.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"          # caché de pares (no se versiona)
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _p in (DATA, PROCESSED, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 2021

# --------------------------------------------------------------------------
# Cifras del artículo
# --------------------------------------------------------------------------
# Tabla 1: cobertura de cada campo en ADS
COVERAGE = {"title": 0.999, "abstract": 0.787, "affiliation": 0.78, "keywords": 0.653}

# Tabla 3: estadísticas del conjunto de datos etiquetado
ARTICLE_TABLE3 = {
    "train":      {"blocks": 98, "authorships": 2487, "labeled": 2318, "mean_block": 23.7, "median_block": 5,
                   "profiles_mean": 2.6, "profiles_median": 1, "pairs": 204_299, "pct_class1": 64.9},
    "test_all":   {"blocks": 39, "authorships": 11_567, "labeled": 9545, "mean_block": 244.7, "median_block": 35,
                   "profiles_mean": 14.4, "profiles_median": 2, "pairs": 5_503_931, "pct_class1": 47.0},
    "test_large": {"blocks": 6, "authorships": 10_006, "labeled": 7996, "mean_block": 1332.7, "median_block": 1304,
                   "profiles_mean": 79.2, "profiles_median": 80.5, "pairs": 5_423_970, "pct_class1": 46.5},
    "test_random": {"blocks": 33, "authorships": 1561, "labeled": 1549, "mean_block": 46.9, "median_block": 24,
                    "profiles_mean": 2.6, "profiles_median": 1, "pairs": 79_961, "pct_class1": 77.4},
}

# Tabla 7: los seis bloques grandes del conjunto de prueba
# (nombre, autorías desambiguadas, perfiles reales, origen del apellido)
LARGE_BLOCKS = [
    ("binzel.r", 1599, 2, "western"),
    ("lee.j", 1295, 181, "korean"),
    ("zhang.y", 1211, 122, "chinese"),
    ("wang.j", 1313, 107, "chinese"),
    ("russell.c", 1522, 9, "western"),
    ("chen.y", 1056, 54, "chinese"),
]

# Tabla 9: los 33 bloques aleatorios del conjunto de prueba
# (nombre, autorías desambiguadas, perfiles reales)
RANDOM_BLOCKS = [
    ("ables.s", 11, 1), ("adam.m", 16, 6), ("annibali.f", 39, 1), ("aramo.c", 100, 1),
    ("botvina.a", 12, 1), ("campbell-brown.m", 51, 1), ("carpenter.j", 190, 7), ("cotzomi.j", 30, 1),
    ("desiante.r", 54, 1), ("ermakov.s", 35, 5), ("gretskov.p", 23, 1), ("katgert.p", 63, 1),
    ("klein.u", 228, 5), ("kuga.k", 10, 2), ("luthcke.s", 23, 1), ("melioli.c", 15, 1),
    ("mendez.r", 162, 4), ("mercuri.s", 12, 1), ("miller.v", 22, 6), ("moreno.c", 21, 3),
    ("morlok.a", 24, 1), ("naef.d", 70, 1), ("nestorov.g", 34, 1), ("phan.n", 21, 1),
    ("ranjan.s", 10, 3), ("ravelo.a", 10, 1), ("ribeiro.a", 75, 10), ("rovira.m", 65, 2),
    ("roy.b", 12, 4), ("salama.f", 45, 1), ("sarasso.m", 16, 1), ("vargas.c", 13, 5),
    ("vincent.m", 36, 6),
]

# Tablas 4, 6 y 8: métricas BCubed reportadas (mejor modelo de cada algoritmo)
# Nota: el artículo llama "micro" al promedio de los BCubed por bloque y
# "macro" al cálculo con todas las autorías juntas. Respetamos su nomenclatura.
ARTICLE_BCUBED = [
    # subset, model, config, micro p, r, f, macro p, r, f, true clusters, predicted
    ("todos", "Reglas (Milojević)", "-", .973, .855, .890, .963, .638, .768, 14.41, 3.615),
    ("todos", "Decision Tree", "accuracy, {0:5,1:1}", .936, .911, .909, .749, .924, .827, 14.41, 10.821),
    ("todos", "Random Forest", "accuracy, {0:5,1:1}", .947, .956, .947, .811, .946, .874, 14.41, 10.974),
    ("todos", "Hist-GBDT", "average precision, {0:5,1:1}", .944, .960, .945, .787, .960, .865, 14.41, 9.333),
    ("grandes", "Reglas (Milojević)", "-", .967, .545, .646, .968, .583, .727, 79.167, 13.5),
    ("grandes", "Decision Tree", "accuracy, {0:5,1:1}", .682, .916, .766, .705, .922, .799, 79.167, 52.0),
    ("grandes", "Random Forest", "average precision, {0:5,1:1}", .761, .939, .832, .778, .944, .853, 79.167, 54.5),
    ("grandes", "Hist-GBDT", "average precision, {0:5,1:1}", .732, .957, .817, .751, .960, .843, 79.167, 44.5),
    ("aleatorios", "Reglas (Milojević)", "-", .974, .911, .934, .939, .927, .933, 2.636, 1.818),
    ("aleatorios", "Decision Tree", "accuracy, {0:1,1:1}", .961, .983, .971, .958, .986, .972, 2.636, 2.364),
    ("aleatorios", "Random Forest", "average precision, {0:2,1:1}", .974, .978, .975, .980, .978, .979, 2.636, 2.727),
    ("aleatorios", "Hist-GBDT", "accuracy, {0:1,1:1}", .957, .976, .965, .955, .984, .969, 2.636, 2.455),
]

# Tabla 5: reporte de clasificación de pares (RF, accuracy, {0:5,1:1})
ARTICLE_TABLE5 = {"precision_0": .83, "recall_0": .98, "f1_0": .90,
                  "precision_1": .97, "recall_1": .77, "f1_1": .86, "accuracy": .88}

# Tabla 7: BCubed por bloque grande (RF)
ARTICLE_TABLE7 = {
    "binzel.r": (.999, 1.0, .999, 2, 1), "lee.j": (.569, .918, .702, 181, 110),
    "zhang.y": (.557, .881, .682, 122, 72), "wang.j": (.641, .934, .761, 107, 76),
    "russell.c": (1.0, .984, .992, 9, 12), "chen.y": (.801, .920, .856, 54, 21),
}

# --------------------------------------------------------------------------
# Parámetros de la simulación
# --------------------------------------------------------------------------
SIM = {
    # [ESCALADO] los bloques grandes se reducen a la mitad (autorías y perfiles)
    # para que el experimento corra en un portátil: 5,4 M de pares -> ~1,3 M.
    "large_block_scale": 0.5,
    # [ARTÍCULO] tamaños del conjunto de entrenamiento (Tabla 3)
    "train_blocks": 98, "train_authorships": 2487, "train_median_block": 5,
    "train_profiles_mean": 2.6,
    # [SUPUESTO] el artículo no reporta cuántos bloques de entrenamiento tienen
    # apellidos de origen asiático; se supone 15 %.
    "train_share_asian": 0.15,
    # [ARTÍCULO] % de pares de clase 1 a reproducir (Tabla 3); se usan para
    # calibrar el exponente Zipf de productividad dentro de cada subconjunto.
    "target_class1": {"train": 0.649, "test_random": 0.774, "test_large": 0.465},
    # [SUPUESTO derivado de la Tabla 7] en binzel.r y russell.c el RF del artículo
    # obtiene precisión ~1 con 1 y 12 clusters: casi todas las autorías son de un
    # autor dominante. Se fija su % de pares de clase 1; el exponente Zipf de los
    # otros cuatro bloques grandes se calibra para que el total dé 46,5 %.
    "block_class1": {"binzel.r": 0.99, "russell.c": 0.90},
    # [SUPUESTO] "la mayoría de nombres de pila están abreviados" (artículo, p. 3898)
    # -> 60 % de las autorías muestran solo iniciales.
    "p_abbreviated": 0.60,
    # [SUPUESTO] probabilidad de que un autor occidental tenga segundo nombre
    "p_middle_western": 0.45,
    # [SUPUESTO] probabilidad de nombre de pila de dos sílabas (p. ej. Jian-Cheng)
    "p_two_syllables_asian": 0.75,
    # [SUPUESTO] probabilidad de conservar la inicial del segundo nombre al abreviar
    "p_keep_middle_initial": 0.85,
    # [SUPUESTO] en nombres asiáticos de dos sílabas la 2.ª inicial casi siempre se
    # conserva al abreviar (Wang, J.-C.), como es habitual en ADS
    "p_keep_middle_initial_asian": 0.95,
    # [SUPUESTO] errores de transcripción/transliteración del nombre completo
    "p_name_typo": 0.03,
    # [SUPUESTO] uso de apodo (Bob por Robert) en nombres occidentales completos
    "p_nickname": 0.05,
    # [ARTÍCULO] coberturas de la Tabla 1
    "coverage": COVERAGE,
    # [SUPUESTO] el autor cambia de institución una vez en su carrera
    "p_move": 0.35,
    # [SUPUESTO] autoría firmada desde una institución visitante
    "p_visiting_affil": 0.10,
    # [SUPUESTO] autoría con dos afiliaciones
    "p_double_affil": 0.15,
    # [SUPUESTO] el NER no logra extraer ciudad / país del texto de afiliación
    "p_ner_fail": 0.10,
    # [SUPUESTO] autorías en grandes colaboraciones (100-300 coautores);
    # el artículo menciona >1.600 documentos con más de 100 autores.
    "p_big_collab": 0.03,
    # [SUPUESTO] fracción de coautores tomados del grupo habitual del autor
    "p_usual_coauthor": 0.85,
    # [SUPUESTO] dentro de un bloque homónimo, probabilidad de que un autor
    # comparta subárea temática con otro autor del mismo bloque
    "p_shared_topic": 0.35,
    # [SUPUESTO] fracción del grupo de coautores que comparten dos homónimos de la misma institución
    "p_shared_coauthors": 0.10,
    # [SUPUESTO] fracción de autores asiáticos con institución en su país de origen
    "p_home_country": 0.65,
    # Corpus sintético
    "n_topics": 30, "words_per_topic": 60, "general_words": 400, "topic_word_bank": 700,
    # [SUPUESTO] fracción de palabras de un texto que vienen de los temas del autor
    "p_topic_word": 0.40,
    "title_len": (7, 14), "abstract_len": (60, 130),
    "embedding_dim": 25,   # [ARTÍCULO] Doc2Vec de 25 dimensiones
}

# --------------------------------------------------------------------------
# Modelado
# --------------------------------------------------------------------------
# [ARTÍCULO] cinco proporciones de clases {clase 0 : clase 1}
RATIOS = {"original": None, "{0:1,1:1}": (1, 1), "{0:2,1:1}": (2, 1),
          "{0:1,1:2}": (1, 2), "{0:5,1:1}": (5, 1)}
# [ARTÍCULO] dos funciones objetivo en la búsqueda de hiperparámetros
SCORINGS = {"accuracy": "accuracy", "average precision": "average_precision"}
# [ARTÍCULO] 200 iteraciones de búsqueda aleatoria con 5 particiones.
# [SUPUESTO/ESCALADO] usamos menos iteraciones por costo de cómputo; se puede
# subir a 200 cambiando este valor (el código no cambia).
N_ITER_SEARCH = 10
CV_FOLDS = 5
# [ESCALADO] tamaño máximo de la muestra usada en la búsqueda de hiperparámetros
MAX_SEARCH_ROWS = 40_000
EDGE_THRESHOLD = 0.5    # [ARTÍCULO] arista si el clasificador predice clase 1
