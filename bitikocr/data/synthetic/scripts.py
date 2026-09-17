"""The two alphabets Uzbek documents are written in.

Uzbek has been written in both Latin and Cyrillic within living memory, and
an archive holds both — often the same office name spelled either way. A
record therefore carries the script it is written in, and the corpus is
stored once, in Latin, then transliterated on demand.

The transliteration is the standard Uzbek correspondence. It is a spelling
convention, not a dictionary, so a handful of irregular words are given
explicitly instead; see :func:`in_script`.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["SCRIPTS", "Script", "in_script", "to_cyrillic"]

#: Which alphabet a document is written in.
Script = Literal["latin", "cyrillic"]

#: Every script a document may use.
SCRIPTS: tuple[Script, ...] = ("latin", "cyrillic")

#: A word given in both scripts, because transliterating it would be wrong.
Bilingual = tuple[str, str]

# Longest first: "yo'" is y + o' rather than yo + tutuq, "o'" must be tried
# before "o", and "sh" before "s".
_DIGRAPHS: tuple[tuple[str, str], ...] = (
    ("yo‘", "йў"),
    ("yo'", "йў"),
    ("o‘", "ў"),
    ("o'", "ў"),
    ("g‘", "ғ"),
    ("g'", "ғ"),
    ("sh", "ш"),
    ("ch", "ч"),
    ("ya", "я"),
    ("yo", "ё"),
    ("yu", "ю"),
    ("ye", "е"),
    ("ts", "ц"),
)

_LETTERS: dict[str, str] = {
    "a": "а",
    "b": "б",
    "d": "д",
    "e": "е",
    "f": "ф",
    "g": "г",
    "h": "ҳ",
    "i": "и",
    "j": "ж",
    "k": "к",
    "l": "л",
    "m": "м",
    "n": "н",
    "o": "о",
    "p": "п",
    "q": "қ",
    "r": "р",
    "s": "с",
    "t": "т",
    "u": "у",
    "v": "в",
    "x": "х",
    "y": "й",
    "z": "з",
}

# The tutuq belgisi marks a glottal stop in Latin and a hard sign in Cyrillic.
_APOSTROPHES = "'‘’ʻʼ"


def _match_case(source: str, converted: str) -> str:
    """Give the converted text the capitalisation of the text it came from."""
    if source.isupper() and len(source) > 1:
        return converted.upper()
    if source[:1].isupper():
        return converted[:1].upper() + converted[1:]
    return converted


def _transliterate_word(word: str) -> str:
    """Transliterate one whitespace-free run of Latin Uzbek."""
    lowered = word.lower()
    out: list[str] = []
    index = 0
    at_start = True

    while index < len(lowered):
        for latin, cyrillic in _DIGRAPHS:
            if lowered.startswith(latin, index):
                out.append(cyrillic)
                index += len(latin)
                at_start = False
                break
        else:
            char = lowered[index]
            if char == "e" and at_start:
                # A word-initial "e" is the open э, not the iotated е.
                out.append("э")
            elif char in _APOSTROPHES:
                out.append("ъ")
            else:
                out.append(_LETTERS.get(char, char))
            index += 1
            at_start = False

    return _match_case(word, "".join(out))


def to_cyrillic(text: str) -> str:
    """Transliterate Uzbek Latin text into Uzbek Cyrillic.

    Args:
        text: Text written in the Uzbek Latin alphabet.

    Returns:
        The same text in Cyrillic, preserving capitalisation, digits,
        punctuation and spacing.
    """
    pieces: list[str] = []
    word: list[str] = []

    for char in text:
        if char.isalpha() or char in _APOSTROPHES:
            word.append(char)
            continue
        if word:
            pieces.append(_transliterate_word("".join(word)))
            word = []
        pieces.append(char)

    if word:
        pieces.append(_transliterate_word("".join(word)))
    return "".join(pieces)


def in_script(entry: str | Bilingual, script: Script) -> str:
    """Render a corpus entry in the requested script.

    Args:
        entry: Either Latin text to transliterate, or an explicit
            ``(latin, cyrillic)`` pair for a word the rules get wrong.
        script: The alphabet to render in.

    Returns:
        The entry written in that script.
    """
    if isinstance(entry, tuple):
        return entry[0] if script == "latin" else entry[1]
    return entry if script == "latin" else to_cyrillic(entry)
