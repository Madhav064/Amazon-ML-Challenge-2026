"""Deterministic, symmetric text normalization for business names and addresses.

All rules are hand-written linguistic conventions (abbreviations, legal forms, script transliteration);
no external data is used. The same functions are applied to every source, so normalization never
leaks source identity.
"""
import re
import unicodedata

# ----------------------------------------------------------------------------- Indic transliteration
# The nine major Indic Unicode blocks share the ISCII-derived layout, so one offset table serves all.
_INDIC_BASES = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00)
_V = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e",
      0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li", 0x72: "", 0x73: ""}
_C = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
      0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
      0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l",
      0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h", 0x58: "q", 0x59: "kh", 0x5A: "g",
      0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y"}
_M = {0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e",
      0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x62: "li", 0x63: "li", 0x55: "", 0x56: "i", 0x57: ""}
_VIRAMA, _NUKTA = 0x4D, 0x3C
_NUKTA_SHIFT = {"j": "z", "ph": "f", "k": "q", "d": "r", "dh": "rh"}
_NASAL = {0x01: "n", 0x02: "n", 0x03: "h", 0x70: "n"}
_MALAYALAM_CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k", 0x09CE: "t"}


def _indic_off(cp):
    if 0x0900 <= cp <= 0x0D7F:
        return cp - _INDIC_BASES[(cp - 0x0900) >> 7]
    return None


def transliterate(s: str) -> str:
    out = []
    pending = False  # consonant waiting for its inherent vowel
    for ch in s:
        cp = ord(ch)
        if cp in _MALAYALAM_CHILLU:
            if pending:
                out.append("a")
            out.append(_MALAYALAM_CHILLU[cp]); pending = False
            continue
        off = _indic_off(cp)
        if off is None:
            if cp in (0x200C, 0x200D):
                continue
            pending = False  # word-final schwa deletion
            out.append(ch)
            continue
        if off in _C:
            if pending:
                out.append("a")
            out.append(_C[off]); pending = True
        elif off in _M:
            out.append(_M[off]); pending = False
        elif off == _VIRAMA:
            pending = False
        elif off == _NUKTA:
            if out and out[-1] in _NUKTA_SHIFT:
                out[-1] = _NUKTA_SHIFT[out[-1]]
        elif off in (0x3D, 0x71):
            pass
        elif off in _NASAL:
            if pending:
                out.append("a")
            out.append(_NASAL[off]); pending = False
        elif off in _V:
            if pending:
                out.append("a")
            out.append(_V[off]); pending = False
        elif 0x66 <= off <= 0x6F:
            if pending:
                out.append("a")
            out.append(chr(ord("0") + off - 0x66)); pending = False
        else:
            if pending:
                out.append("a")
            out.append(" "); pending = False
    return "".join(out)


_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def fold(s: str) -> str:
    """Unicode -> lowercase ASCII: strip accents, transliterate Indic scripts."""
    if _NON_ASCII.search(s):
        s = transliterate(s.replace("റ്റ", "ട്ട"))
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.encode("ascii", "ignore").decode()
    return s.lower()


# ----------------------------------------------------------------------------- names
LEGAL_CANON = {
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "cie": "co", "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc",
    "pc": "pc", "pllc": "pllc", "pa": "pa", "public": "public",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci", "snc": "snc", "ei": "ei",
}
LEGAL = set(LEGAL_CANON.values())
STOP = {"and", "the", "of", "de", "du", "des", "la", "le", "les", "l", "d", "et", "a", "an", "en"}
HONORIFIC = {"mr", "mrs", "ms", "miss", "smt", "shri", "shree", "sri", "sree", "messrs", "the"}

_RE_ID = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)")
_RE_DBA = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|doing business as|trading as|t\s*/\s*a|aka|a\.k\.a\.?)\b\s*:?")
_RE_URL = re.compile(r"https?://|www\.")
_RE_DOMAIN = re.compile(r"\.(?:co\.in|com|net|org|in|co|biz|info|fr|us)\b")
_RE_MS = re.compile(r"^\s*m\s*/\s*s\b\.?")
_RE_UPPER_L = re.compile(r"\b[A-Z]+l[A-Z]+\b")
_RE_LEET = re.compile(r"(?<=[a-z])[0135](?=[a-z])|(?<=[a-z]{3})[0135]\b")
_LEET = str.maketrans("0135", "oies")
_RE_NONALNUM = re.compile(r"[^a-z0-9]+")


_RE_INDIC = re.compile(r"[ऀ-ൿ]")
# legal forms as they come out of transliterated native-script names ("limittad", "praivet", "kampani")
_LEGAL_SKELETON = {"lmt": "ltd", "lmtt": "ltd", "prvt": "pvt", "prpt": "pvt", "kmpn": "co", "kmpni": "co",
                   "krprsn": "corp", "alalp": "llp", "alp": "llp"}


def _tokens(s: str):
    return [t for t in _RE_NONALNUM.split(s) if t]


def normalize_name(raw: str):
    """Returns (full normalized name, core name tokens joined, compact core)."""
    s = _RE_UPPER_L.sub(lambda m: m.group(0).replace("l", "I"), raw)
    s = fold(s)
    s = _RE_ID.sub(" ", s)
    parts = _RE_DBA.split(s)
    if len(parts) > 1 and parts[-1].strip():
        s = parts[-1]
    s = _RE_URL.sub(" ", s)
    s = _RE_DOMAIN.sub(" ", s)
    s = _RE_MS.sub(" ", s)
    s = s.replace("&", " and ").replace("'", "").replace("`", "").replace(".", "")
    s = _RE_LEET.sub(lambda m: m.group(0).translate(_LEET), s)
    toks = [LEGAL_CANON.get(t, t) for t in _tokens(s)]
    if _RE_INDIC.search(raw):
        toks = [_LEGAL_SKELETON.get(skeleton(t), t) for t in toks]
    while len(toks) > 1 and toks[0] in HONORIFIC:
        toks = toks[1:]
    core = [t for t in toks if t not in LEGAL and t not in STOP] or toks
    compact = "".join(core)
    if len(core) == 1 and len(compact) >= 8 and compact.endswith("com"):
        compact = compact[:-3]
        core = [compact]
    return " ".join(toks), " ".join(core), compact


# ----------------------------------------------------------------------------- addresses
ADDR_CANON = {
    "street": "st", "st": "st", "str": "st", "saint": "st", "ste": "ste", "suite": "ste",
    "road": "rd", "rd": "rd", "avenue": "ave", "ave": "ave", "av": "ave", "drive": "dr", "dr": "dr",
    "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd",
    "highway": "hwy", "hwy": "hwy", "place": "pl", "pl": "pl", "parkway": "pkwy", "pkwy": "pkwy",
    "circle": "cir", "cir": "cir", "terrace": "ter", "ter": "ter", "square": "sq", "sq": "sq", "trail": "trl", "trl": "trl",
    "north": "n", "south": "s", "east": "e", "west": "w", "apartment": "apt", "apt": "apt",
    "building": "bldg", "bldg": "bldg", "floor": "fl", "fl": "fl", "flr": "fl",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "nagar": "nagar", "ngr": "nagar", "sector": "sec", "sec": "sec", "phase": "ph", "ph": "ph",
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp", "railway": "rly", "rly": "rly",
    "crossing": "crsg", "crsg": "crsg", "cross": "crs", "colony": "colony", "society": "soc", "soc": "soc",
    "rue": "rue", "r": "rue", "chemin": "ch", "ch": "ch", "impasse": "imp", "imp": "imp", "route": "rte", "rte": "rte",
    "allee": "all", "general": "gen", "gen": "gen",
}
ADDR_DROP = {"no", "number", "door", "hno", "h", "null", "na", "a", "nan", "none", "unit", "pmb"}
_RE_NUM = re.compile(r"\d+")
_RE_ADDR_SPLIT = re.compile(r"[^a-z0-9]+")
_RE_ORD = re.compile(r"^(\d+)(st|nd|rd|th)$")


def normalize_address(raw: str):
    """Returns (normalized token string, space-joined numbers without leading zeros)."""
    if not raw:
        return "", ""
    s = fold(raw)
    s = s.replace("n/a", " ").replace("c/o", " ")
    toks = []
    for t in _RE_ADDR_SPLIT.split(s):
        if not t:
            continue
        m = _RE_ORD.match(t)
        if m:
            t = str(int(m.group(1))) + m.group(2)
        elif t.isdigit():
            t = str(int(t))
        t = ADDR_CANON.get(t, t)
        if t in ADDR_DROP:
            continue
        toks.append(t)
    nums = []
    for n in _RE_NUM.findall(s):
        n = str(int(n))
        if n not in nums:
            nums.append(n)
    return " ".join(toks), " ".join(nums)


# ----------------------------------------------------------------------------- phonetic skeleton
_SK_MULTI = [("chh", "c"), ("ch", "c"), ("sh", "s"), ("ph", "f"), ("kh", "k"), ("gh", "k"), ("th", "t"), ("dh", "t"),
             ("bh", "p"), ("jh", "j"), ("ck", "k"), ("x", "ks"), ("q", "k"), ("w", "v"), ("z", "s")]
_SK_SINGLE = str.maketrans({"g": "k", "d": "t", "b": "p", "c": "k", "y": "i"})
_RE_VOWEL = re.compile(r"[aeiouh]")
_RE_DUP = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    """Consonant skeleton that is robust to vowel spelling, transliteration and voicing differences."""
    if not tok or tok.isdigit():
        return tok
    t = tok
    for a, b in _SK_MULTI:
        if a in t:
            t = t.replace(a, b)
    t = t.translate(_SK_SINGLE)
    head = "a" if t[0] in "aeiou" else t[0]
    return _RE_DUP.sub(r"\1", head + _RE_VOWEL.sub("", t[1:]))
