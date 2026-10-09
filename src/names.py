"""
Nombres de autor: generación, representación abreviada, firmas y compatibilidad.

Reproduce la lógica descrita en la sección "Pairwise comparisons" del artículo:
  * la *firma* de una autoría es el par (nombre, segundo nombre), p. ej. 'jane-mary';
  * dos firmas son *compatibles* si cada parte es prefijo de la otra, admitiendo
    apodos ('tony' ~ 'anthony') y concatenaciones ('pei-yu' ~ 'peiyu').
"""
from __future__ import annotations

import unicodedata

import jellyfish
import numpy as np

# Nombres occidentales por inicial (pool pequeño a propósito: genera homónimos)
WESTERN_FIRST = {
    "a": ["alexander", "andrew", "anna", "alice", "antonio", "albert", "alan", "amanda", "anthony"],
    "b": ["benjamin", "barbara", "brian", "bruno", "beatrice", "boris", "bradley"],
    "c": ["charles", "christopher", "carlos", "catherine", "claire", "carmen", "christian", "craig"],
    "d": ["david", "daniel", "dominique", "diana", "dmitry", "douglas", "denis"],
    "e": ["edward", "elena", "eric", "emily", "eduardo", "elizabeth", "erik"],
    "f": ["francesco", "frank", "felix", "fiona", "fernando", "frederic", "fabio"],
    "g": ["george", "gregory", "giovanni", "gabriela", "gerhard", "gloria", "georgi"],
    "h": ["henry", "helen", "hans", "hector", "hannah", "harold"],
    "i": ["ivan", "isabel", "ian", "irina", "igor", "ines"],
    "j": ["john", "james", "jane", "joseph", "julia", "jorge", "jennifer", "jacob", "jean", "joan"],
    "k": ["kevin", "katherine", "karl", "kenji", "klaus", "karen", "kazuo"],
    "l": ["laura", "louis", "lucia", "lars", "leonard", "linda", "luca"],
    "m": ["michael", "maria", "martin", "marco", "margaret", "matthew", "miguel", "monica", "mark"],
    "n": ["nicholas", "natalia", "nathan", "norbert", "nicole", "nikolai"],
    "o": ["oliver", "olga", "oscar", "otto", "olivia"],
    "p": ["peter", "paul", "patricia", "pablo", "pierre", "petra", "philip"],
    "q": ["quentin", "quinn"],
    "r": ["robert", "richard", "rachel", "rafael", "roberto", "rosa", "ralf", "ronald"],
    "s": ["stephen", "sarah", "samuel", "sergei", "susan", "simone", "stefan", "sandra"],
    "t": ["thomas", "timothy", "teresa", "tobias", "takeshi", "tatiana"],
    "u": ["ulrich", "ursula", "uwe", "umberto", "ulf"],
    "v": ["victor", "valentina", "vladimir", "vera", "vincent", "vanessa"],
    "w": ["william", "walter", "wolfgang", "wendy", "werner"],
    "x": ["xavier", "ximena"],
    "y": ["yuri", "yvonne", "yosef"],
    "z": ["zoltan", "zoe", "zachary"],
}

NICKNAMES = {
    "robert": "bob", "william": "bill", "anthony": "tony", "james": "jim", "michael": "mike",
    "elizabeth": "liz", "katherine": "kate", "david": "dave", "thomas": "tom",
    "alexander": "alex", "christopher": "chris", "daniel": "dan", "stephen": "steve",
    "samuel": "sam", "joseph": "joe", "benjamin": "ben", "nicholas": "nick", "edward": "ed",
    "richard": "rick", "matthew": "matt", "timothy": "tim", "peter": "pete", "andrew": "andy",
}
_NICK_TO_FULL = {v: k for k, v in NICKNAMES.items()}

# Sílabas de nombres de pila de origen chino / coreano por inicial
ASIAN_FIRST_SYLLABLE = {
    "chinese": {
        "j": ["jian", "jun", "jing", "jie", "jin", "jia", "jiang", "ju"],
        "y": ["yan", "yi", "yu", "ying", "yong", "yang", "yue", "yun"],
        "l": ["li", "lin", "liang", "lei", "lu"], "x": ["xiao", "xin", "xue", "xiang"],
        "h": ["hong", "hui", "hai", "hua"], "w": ["wei", "wen", "wu"],
        "z": ["zhi", "zhong", "zhen"], "s": ["shu", "sheng", "song"],
    },
    "korean": {
        "j": ["jae", "jong", "jin", "ji", "jun", "joon", "jung", "jeong"],
        "s": ["sung", "seung", "soo", "sang"], "h": ["hyun", "hee", "ho", "hye"],
        "m": ["min", "myung"], "y": ["young", "yong", "yeon"],
    },
}
ASIAN_SECOND_SYLLABLE = {
    "chinese": ["cheng", "fei", "hua", "ming", "ping", "qiang", "rong", "tao", "wei", "xin",
                "yu", "zhong", "dong", "feng", "gang", "hong", "jie", "lin", "long", "bo"],
    "korean": ["ho", "hoon", "soo", "won", "min", "woo", "hyun", "sik", "young", "su",
               "chul", "hwan", "kyu", "seok", "wook", "yeol", "ung", "ik", "il", "gyu"],
}


def strip_accents(text: str) -> str:
    """Elimina diacríticos y pasa a minúsculas ('Sørensen' -> 'sorensen')."""
    text = text.replace("ø", "o").replace("Ø", "O")
    norm = unicodedata.normalize("NFKD", text)
    return "".join(c for c in norm if not unicodedata.combining(c)).lower()


def block_key(surname: str, first: str) -> str:
    """Clave de bloque del artículo: apellido + primera inicial, sin diacríticos."""
    clean = "".join(c for c in strip_accents(surname) if c.isalpha() or c == "-")
    return f"{clean}.{strip_accents(first)[0]}"


# --------------------------------------------------------------------------
# Generación de nombres canónicos de autores
# --------------------------------------------------------------------------
def sample_canonical_name(rng: np.random.Generator, initial: str, origin: str, p_middle: float,
                          p_two_syll: float) -> tuple[str, str]:
    """Devuelve (nombre, segundo_nombre) canónicos de un autor real."""
    if origin in ASIAN_FIRST_SYLLABLE:
        pool = ASIAN_FIRST_SYLLABLE[origin].get(initial)
        if pool is None:  # inicial poco frecuente: se usa la sílaba con esa inicial
            pool = [initial + "an", initial + "in"]
        first = str(rng.choice(pool))
        middle = str(rng.choice(ASIAN_SECOND_SYLLABLE[origin])) if rng.random() < p_two_syll else ""
        return first, middle
    pool = WESTERN_FIRST.get(initial, [initial + "ane"])
    first = str(rng.choice(pool))
    middle = ""
    if rng.random() < p_middle:
        letter = "abcdefghijklmnoprstvw"[rng.integers(0, 21)]
        middle = str(rng.choice(WESTERN_FIRST[letter]))
    return first, middle


def _typo(rng: np.random.Generator, name: str) -> str:
    """Error de transcripción: duplica, omite o cambia una letra interna."""
    if len(name) < 4:
        return name
    i = int(rng.integers(1, len(name) - 1))
    kind = rng.integers(0, 3)
    if kind == 0:
        return name[:i] + name[i] + name[i:]
    if kind == 1:
        return name[:i] + name[i + 1:]
    return name[:i] + "aeiou"[rng.integers(0, 5)] + name[i + 1:]


def render_name(first: str, middle: str, origin: str, u_abbr: float, u_form: float, u_noise: float,
                p_abbrev: float, p_keep_mid: float, p_typo: float, p_nick: float) -> tuple[str, str]:
    """
    Convierte el nombre canónico en la forma que aparece en una autoría concreta.

    Los números aleatorios (u_*) se pasan desde fuera para que el experimento de
    extensión pueda cambiar SOLO `p_abbrev` manteniendo todo lo demás idéntico
    (números aleatorios comunes).

    Devuelve (parte_nombre, parte_segundo_nombre) tal como se leería en ADS,
    p. ej. ('j', 'c') para 'Wang, J.-C.' o ('jiancheng', '') para 'Wang, Jiancheng'.
    """
    if u_abbr < p_abbrev:                       # forma abreviada: solo iniciales
        mid = middle[0] if (middle and u_form < p_keep_mid) else ""
        return first[0], mid
    # forma completa
    f = first
    if origin == "western" and u_noise < p_nick and first in NICKNAMES:
        f = NICKNAMES[first]
    elif u_noise > 1 - p_typo:
        f = _typo(np.random.default_rng(int(u_noise * 1e9)), first)
    if not middle:
        return f, ""
    if origin in ASIAN_FIRST_SYLLABLE:
        if u_form < 0.70:
            return f, middle                    # Jian-Cheng
        if u_form < 0.85:
            return f + middle, ""               # Jiancheng (concatenado)
        return f, middle[0]                     # Jian C.
    if u_form < 0.40:
        return f, middle                        # Jane Mary
    if u_form < 0.80:
        return f, middle[0]                     # Jane M.
    return f, ""                                # Jane (se omite el segundo nombre)


def display_name(surname: str, first: str, middle: str) -> str:
    """Forma legible 'Apellido, Nombre M.'"""
    def part(p):
        return (p[0].upper() + ".") if len(p) == 1 else p.capitalize()
    s = surname.capitalize() + ", " + part(first)
    if middle:
        s += ("-" if len(middle) > 1 and len(first) > 1 else " ") + part(middle)
    return s


def signature(first: str, middle: str) -> str:
    """Firma del artículo: 'nombre-segundo' (cadena vacía si falta una parte)."""
    return f"{first}-{middle}"


# --------------------------------------------------------------------------
# Compatibilidad de firmas
# --------------------------------------------------------------------------
def _canon(part: str) -> str:
    return _NICK_TO_FULL.get(part, part)


def _part_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return True
    a, b = _canon(a), _canon(b)
    return a.startswith(b) or b.startswith(a)


def signatures_compatible(s1: str, s2: str) -> bool:
    """
    'abbigail-joan' ~ 'abbigail-', 'anthony-' ~ 'tony-m', 'pei-yu' ~ 'peiyu-',
    'j-' ~ 'jane-mary'; pero 'j-r' !~ 'jane-mary' y 'maria-f' !~ 'maria-t'.
    """
    f1, m1 = s1.split("-", 1)
    f2, m2 = s2.split("-", 1)
    # concatenaciones: 'peiyu-' frente a 'pei-yu'
    if not m1 and m2 and len(f2) > 1 and len(m2) > 1 and f1 == f2 + m2:
        return True
    if not m2 and m1 and len(f1) > 1 and len(m1) > 1 and f2 == f1 + m1:
        return True
    return _part_compatible(f1, f2) and _part_compatible(m1, m2)


def compatibility_matrix(sigs: list[str]) -> np.ndarray:
    n = len(sigs)
    m = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = signatures_compatible(sigs[i], sigs[j])
    return m


def phonetic(name: str) -> str:
    """Código fonético (Metaphone; el artículo usa Double Metaphone)."""
    return jellyfish.metaphone(name) if len(name) > 1 else ""
