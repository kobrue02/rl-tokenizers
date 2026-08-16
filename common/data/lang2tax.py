"""Maps this project's lang_Script codes to Joshi et al.'s 6-level linguistic
resource taxonomy (0 = "The Left-Behinds", ..., 5 = "The Winners" -- Joshi
et al. 2020, "The State and Fate of Linguistic Diversity and Inclusion in
the NLP World"), via Microsoft's published lang2tax.txt (cached here
verbatim, 2485 lines):
https://microsoft.github.io/linguisticdiversity/assets/lang2tax.txt

lang2tax.txt maps plain lowercase English language NAMES to a resource
level 0-5 -- NOT ISO codes -- so bridging it to our lang_Script codes needs
a name lookup (via langcodes, CLDR-backed) plus alias handling, since ISO
639-3's canonical name for a code often differs from Joshi's glottonym
(e.g. ISO "Kalaallisut" vs. Joshi's "greenlandic").

Match strategy, in order (resolves 217/259 BOUQuET codes this project
uses automatically):
  1. Exact match on the langcodes display name (lowercased, "(individual
     language)" clarifier stripped).
  2. The same name with hyphens/apostrophes/diacritics normalized away.
  3. A qualifier word (standard/western/eastern/.../upper/lower/greater)
     stripped from the front, OR just the first or last word alone (e.g.
     "Western Frisian" -> "frisian", "Mandarin Chinese" -> "mandarin").
_MANUAL_ALIASES covers the remainder: codes where the ISO name and Joshi's
glottonym share no derivable substring at all (e.g. "Ika" for Arhuaco,
"Jingpho" for Kachin), each verified against lang2tax.txt before being
added, plus two upgrades from an accurate-but-generic automatic match to a
more specific entry Joshi's file also has at a DIFFERENT level (kmr/ckb
resolve to bare "kurdish"=0 automatically, but "kurdish (kurmanji)"=1 and
"kurdish (sorani)"=1 are more accurate for those codes specifically).

The remaining ~42/259 codes (mostly very low-resource languages) are NOT in
Joshi's taxonomy at all -- a real coverage gap in that external resource.
load_resource_levels() reports these explicitly rather than silently
dropping them.
"""

import os
import re
import unicodedata

import langcodes

_TAX_FILE = os.path.join(os.path.dirname(__file__), "lang2tax.txt")

_QUALIFIERS = {
    "standard", "western", "eastern", "northern", "southern", "central",
    "coastal", "north", "south", "east", "west", "upper", "lower", "greater",
}

# Verified individually against lang2tax.txt -- see module docstring for
# which have no derivable automatic match vs. which override a generic
# automatic match with a more specific one.
_MANUAL_ALIASES = {
    "ben_Beng": "bengali", "ben_Latn": "bengali",
    "pan_Guru": "eastern punjabi",
    "kac_Latn": "jingpho",
    "kir_Cyrl": "kirghiz",
    "tgl_Latn": "tagalog",
    "gil_Latn": "kiribati",
    "kal_Latn": "greenlandic",
    "ilo_Latn": "ilocano",
    "lug_Latn": "luganda",
    "nya_Latn": "chichewa",
    "bba_Latn": "bariba",
    "mas_Latn": "maasai",
    "taq_Latn": "tuareg", "taq_Tfng": "tuareg",
    "ijc_Latn": "ijo",
    "tzm_Tfng": "berber",
    "arh_Latn": "ika",
    "fuc_Latn": "fulfulde",
    "kmr_Latn": "kurdish (kurmanji)",
    "ckb_Arab": "kurdish (sorani)",
}


def _load_tax(_cache={}):
    if _cache:
        return _cache
    with open(_TAX_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "," not in line:
                continue
            name, level = line.rsplit(",", 1)
            name = name.strip().lower()
            level = int(level)
            _cache[name] = level
            # A few entries are themselves parenthesized ("norwegian
            # (nynorsk)") -- register the paren-stripped form too, but never
            # let it override an already-distinct entry (bare "kurdish" is
            # its own separately-leveled entry).
            if "(" in name:
                stripped = re.sub(r"[()]", "", name)
                stripped = re.sub(r"\s+", " ", stripped).strip()
                _cache.setdefault(stripped, level)
    return _cache


def _normalize(s):
    s = s.replace("ʼ", "'").replace("’", "'").replace("'", "").replace("-", " ")
    nfkd = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def _candidates(name):
    """Returns candidate lookup keys in explicit, deterministic priority
    order (most specific first) -- returning a set here instead is a real
    bug: set iteration order depends on PYTHONHASHSEED, so when two
    candidates were both valid but differently-leveled tax-file entries,
    which one won was non-deterministic across process runs (reproduced
    directly: repeated runs over the same 259 codes gave different
    resource-level distributions)."""
    name = re.sub(r"\(.*?\)", "", name.lower()).strip()
    normalized = _normalize(name)
    ordered = [name, normalized]
    words = normalized.split()
    if len(words) > 1:
        if words[0] in _QUALIFIERS:
            ordered.append(" ".join(words[1:]))
        ordered.append(words[-1])
        ordered.append(words[0])
    seen = set()
    result = []
    for c in ordered:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def resolve(code):
    """Returns the resource level (0-5) for one lang_Script code, or None if
    it can't be resolved against lang2tax.txt (see module docstring)."""
    tax = _load_tax()
    if code in _MANUAL_ALIASES:
        return tax.get(_MANUAL_ALIASES[code])
    if "_" not in code:
        return None
    lang, script = code.split("_", 1)
    try:
        name = langcodes.Language.get(f"{lang}-{script}").language_name()
    except Exception:
        return None
    for cand in _candidates(name):
        if cand in tax:
            return tax[cand]
    return None


def load_resource_levels(codes):
    """Returns (levels: {code: int}, unresolved: [code, ...]) for an
    iterable of lang_Script codes."""
    levels = {}
    unresolved = []
    for code in codes:
        level = resolve(code)
        if level is None:
            unresolved.append(code)
        else:
            levels[code] = level
    return levels, unresolved
