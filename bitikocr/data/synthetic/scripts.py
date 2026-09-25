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

import random
import re
from typing import Literal

__all__ = [
    "LATIN_APOSTROPHES",
    "SCRIPTS",
    "Script",
    "in_script",
    "russian_spelling",
    "spell_as_writer",
    "to_cyrillic",
]

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


# -- how a writer actually spells ----------------------------------------------

# A word, for the writer's spelling: a run of anything but whitespace.
_WORD = re.compile(r"\S+")

# How many letters of a word a writer's habit is decided on: enough to tell
# names apart, few enough that Каримов and Каримова count as one.
_STEM_LENGTH = 6

#: The Uzbek Cyrillic letters Russian has no equivalent for, and the Russian
#: letter a writer schooled in Russian puts in their place.
_RUSSIAN_LETTERS: dict[str, str] = {
    "ҳ": "х",
    "Ҳ": "Х",
    "қ": "к",
    "Қ": "К",
    "ғ": "г",
    "Ғ": "Г",
    "ў": "у",
    "Ў": "У",
}

#: Name stems whose Russian form differs by more than those letters, as the
#: civil register wrote them in Russian: Иброҳимов became Ибрагимов.
#: Matched at the start of a word, so a surname or patronymic built on the
#: stem changes with it.
_RUSSIAN_FORMS: tuple[tuple[str, str], ...] = (
    ("Иброҳим", "Ибрагим"),
    ("Аҳмад", "Ахмед"),
    ("Абдураҳмон", "Абдурахман"),
    ("Абдураҳим", "Абдурахим"),
    ("Раҳмон", "Рахман"),
    ("Юсуф", "Юсуп"),
    ("Исмоил", "Исмаил"),
    ("Сафар", "Сапар"),
    ("Шариф", "Шарип"),
    ("Зариф", "Зарип"),
    ("Латиф", "Латип"),
    ("Ражаб", "Ражап"),
    ("Ёқуб", "Якуб"),
    ("Муҳаммад", "Мухаммед"),
    ("Маҳмуд", "Махмуд"),
    ("Солай", "Салай"),
    ("Бозор", "Базар"),
    ("Давлат", "Давлет"),
    ("Мамат", "Мамед"),
)

#: How a writer spells the tutuq and the o' and g' of Latin Uzbek: the
#: straight apostrophe of a keyboard, or the turned comma the standard asks
#: for, or a backtick.
LATIN_APOSTROPHES: tuple[str, ...] = ("'", "‘", "`")


def russian_spelling(word: str) -> str:
    """Spell a Cyrillic Uzbek word as a writer schooled in Russian would.

    Args:
        word: One word, in Uzbek Cyrillic.

    Returns:
        The word with its stem in the Russian form, when it has one, and
        the Uzbek-only letters replaced with the nearest Russian ones.
    """
    for uzbek, russian in _RUSSIAN_FORMS:
        if word.startswith(uzbek):
            word = russian + word[len(uzbek) :]
            break
    return "".join(_RUSSIAN_LETTERS.get(char, char) for char in word)


def spell_as_writer(
    text: str,
    script: Script,
    share: float,
    apostrophe: str,
    rng: random.Random,
    habits: dict[str, bool] | None = None,
) -> str:
    """Spell a text the way one writer does rather than the way the standard does.

    Many of the archive's writers learnt to write in Russian, and write
    Uzbek Cyrillic with Russian letters and Russian forms of names, some on
    every word and some now and then. In Latin the tutuq is typed however
    the writer is used to.

    Args:
        text: The text in its standard spelling.
        script: The alphabet it is written in.
        share: The share of Cyrillic words the writer spells the Russian
            way; 0 keeps the standard spelling.
        apostrophe: The mark the writer uses for the tutuq, in Latin.
        rng: Random source deciding which words the writer changes.
        habits: What the writer already decided for each word stem, shared
            across every text on one page. A writer spells a name the same
            way each time it comes up, and a family's surnames alike, so
            words that begin alike are decided once.

    Returns:
        The text as this writer would write it.
    """
    if script == "latin":
        return "".join(
            apostrophe if char in _APOSTROPHES else char for char in text
        )
    if share <= 0:
        return text
    decided = {} if habits is None else habits

    def respell(match: re.Match[str]) -> str:
        word = match.group()
        stem = word[:_STEM_LENGTH].casefold()
        if stem not in decided:
            decided[stem] = rng.random() < share
        return russian_spelling(word) if decided[stem] else word

    return _WORD.sub(respell, text)
