"""Uzbek vocabulary the record samplers draw on.

Everything is stored once in the Latin alphabet and transliterated on demand,
except the handful of words the transliteration rules get wrong, which are
given as explicit ``(latin, cyrillic)`` pairs.

This is a working vocabulary, not a gazetteer: it is large enough that thirty
generated records rarely repeat, and small enough to read and correct by
hand. Widen the lists to widen the data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from bitikocr.data.synthetic.scripts import Bilingual, Script, in_script

__all__ = [
    "DISTRICTS",
    "MONTHS",
    "ORGANISATION_ROLES",
    "STREETS",
    "Address",
    "Person",
    "sample_address",
    "sample_mahalla",
    "sample_office",
    "sample_organisation",
    "sample_person",
    "sample_place",
]

#: Month names. Cyrillic is spelled out because the Latin forms iotate
#: irregularly: "sentabr" is "сентябр", not "сентабр".
MONTHS: tuple[Bilingual, ...] = (
    ("yanvar", "январ"),
    ("fevral", "феврал"),
    ("mart", "март"),
    ("aprel", "апрел"),
    ("may", "май"),
    ("iyun", "июн"),
    ("iyul", "июл"),
    ("avgust", "август"),
    ("sentabr", "сентябр"),
    ("oktabr", "октябр"),
    ("noyabr", "ноябр"),
    ("dekabr", "декабр"),
)

_UNITS: tuple[str, ...] = (
    "bir",
    "ikki",
    "uch",
    "to'rt",
    "besh",
    "olti",
    "yetti",
    "sakkiz",
    "to'qqiz",
)

_TEENS_TENS: dict[int, str] = {
    10: "o'n",
    20: "yigirma",
    30: "o'ttiz",
    40: "qirq",
    50: "ellik",
    60: "oltmish",
    70: "yetmish",
    80: "sakson",
    90: "to'qson",
}

_MALE_NAMES: tuple[str, ...] = (
    "Otabek",
    "Shavkat",
    "Bahodir",
    "Javohir",
    "Ulug'bek",
    "Sardor",
    "Jasur",
    "Aziz",
    "Rustam",
    "Bekzod",
    "Doniyor",
    "Farrux",
    "Ibrohim",
    "Kamol",
    "Mirzo",
    "Nodir",
    "Olim",
    "Qodir",
    "Sanjar",
    "Temur",
    "Umid",
    "Xurshid",
    "Zafar",
    "Alisher",
    "Botir",
    "Dilshod",
    "Elyor",
    "Fazliddin",
    "G'ayrat",
    "Husan",
)

_FEMALE_NAMES: tuple[str, ...] = (
    "Mohira",
    "Nodira",
    "Gulnora",
    "Dilorom",
    "Zulfiya",
    "Malika",
    "Sevara",
    "Shahzoda",
    "Tamara",
    "Umida",
    "Feruza",
    "Kamola",
    "Lola",
    "Marjona",
    "Nafisa",
    "Ozoda",
    "Rayhona",
    "Sabina",
    "Shohista",
    "Yulduz",
    "Zebo",
    "Aziza",
    "Barno",
    "Charos",
    "Dildora",
    "Gulchehra",
    "Hulkar",
    "Iroda",
    "Munisa",
    "Nigora",
)

_SURNAME_STEMS: tuple[str, ...] = (
    "Karim",
    "Rahim",
    "Sobir",
    "Ergash",
    "Yo'ldosh",
    "Bekchan",
    "Matyoqub",
    "Qurbonboy",
    "Usmon",
    "Solay",
    "Abduraxmon",
    "Habib",
    "Nazar",
    "Ochil",
    "Po'lat",
    "Qodir",
    "Sattor",
    "Toshpo'lat",
    "Xudoyber",
    "Yusuf",
    "Zokir",
    "Alim",
    "Bozor",
    "Davron",
    "Egam",
    "Fayzi",
    "G'ulom",
    "Hakim",
    "Islom",
    "Jo'ra",
)

_NATIONALITIES: tuple[str, ...] = (
    "o'zbek",
    "qoraqalpoq",
    "qozoq",
    "tojik",
    "rus",
    "turkman",
    "qirg'iz",
)

#: Region mapped to the districts and cities inside it.
DISTRICTS: dict[str, tuple[str, ...]] = {
    "Xorazm": ("Urganch", "Yangiariq", "Xiva", "Bog'ot", "Gurlan", "Shovot"),
    "Sirdaryo": ("Guliston", "Yangiyer", "Sirdaryo", "Boyovut", "Sardoba"),
    "Toshkent": ("Chirchiq", "Angren", "Bekobod", "Yangiyo'l", "Parkent"),
    "Samarqand": ("Urgut", "Kattaqo'rg'on", "Bulung'ur", "Jomboy", "Payariq"),
    "Buxoro": ("G'ijduvon", "Kogon", "Vobkent", "Romitan", "Shofirkon"),
    "Farg'ona": ("Qo'qon", "Marg'ilon", "Quvasoy", "Rishton", "Beshariq"),
    "Andijon": ("Asaka", "Xonobod", "Shahrixon", "Baliqchi", "Izboskan"),
    "Namangan": ("Chust", "Pop", "To'raqo'rg'on", "Uychi", "Kosonsoy"),
    "Qashqadaryo": ("Qarshi", "Shahrisabz", "Kitob", "G'uzor", "Koson"),
    "Surxondaryo": ("Termiz", "Denov", "Sherobod", "Boysun", "Jarqo'rg'on"),
    "Navoiy": ("Zarafshon", "Nurota", "Karmana", "Xatirchi", "Konimex"),
    "Jizzax": ("Jizzax", "Gagarin", "Zomin", "G'allaorol", "Paxtakor"),
}

#: Street names, as an Uzbek address writes them. Most streets are named
#: after a writer, a poet or an idea rather than numbered, so the list is
#: names rather than a pattern.
STREETS: tuple[str, ...] = (
    "A. Qodiriy",
    "Alisher Navoiy",
    "Mustaqillik",
    "Amir Temur",
    "Bobur",
    "Ibn Sino",
    "Al-Xorazmiy",
    "Zulfiya",
    "Cho'lpon",
    "Fitrat",
    "Sh. Rashidov",
    "J. Ataniyazov",
    "Ulug'bek",
    "O'zbekiston ovozi",
    "Do'stlik",
    "Bog'bon",
    "Tinchlik",
    "Gulzor",
)

#: Organisations that write letters of their own. A legal entity writes on
#: its own letterhead and seals what it signs, unlike a private citizen.
_ORGANISATION_NAMES: tuple[str, ...] = (
    "Gulzor",
    "Navro'z",
    "Sharq",
    "Zamin",
    "Oltin vodiy",
    "Yangi asr",
    "Baraka",
    "Nurafshon",
)

#: Legal forms an Uzbek company takes, and the institutions that are not
#: companies at all but still write and seal letters.
_ORGANISATION_FORMS: tuple[str, ...] = (
    "MChJ",
    "OAJ",
    "XK",
    "QK",
)

_INSTITUTIONS: tuple[str, ...] = (
    "umumta'lim maktabi",
    "bolalar bog'chasi",
    "tibbiyot birlashmasi",
    "kasb-hunar kolleji",
    "madaniyat markazi",
)

#: What the person who signs for an organisation is called.
ORGANISATION_ROLES: tuple[str, ...] = (
    "direktori",
    "raisi",
    "boshlig'i",
    "mudiri",
)

#: Names a neighbourhood committee — the mahalla — goes by. Its seal is what
#: certifies a resident's signature.
_MAHALLAS: tuple[str, ...] = (
    "Gulzor",
    "Do'stlik",
    "Bog'bon",
    "Navro'z",
    "Chorbog'",
    "Yangiobod",
    "Bahor",
    "Mustaqillik",
    "Guliston",
    "Obod",
)

_CAUSES_OF_DEATH: tuple[str, ...] = (
    "yurak ishemik kasalligi",
    "o'pka arteriyasi tromboemboliyasi",
    "miya qon aylanishining o'tkir buzilishi",
    "o'tkir yurak yetishmovchiligi",
    "surunkali buyrak yetishmovchiligi",
    "o'pka shishi",
    "qon bosimining keskin ko'tarilishi",
    "onkologik kasallik asoratlari",
    "qandli diabet asoratlari",
    "keksalik",
)


@dataclass(frozen=True)
class Person:
    """One person's name, as a civil register writes it.

    Args:
        surname: Family name.
        given_name: First name.
        patronymic: Father's name in its Slavic-style possessive form.
        is_female: Whether the patronymic and any "child of" suffix are
            feminine.
    """

    surname: str
    given_name: str
    patronymic: str
    is_female: bool

    def full_given_name(self) -> str:
        """Return the given name followed by the patronymic."""
        return f"{self.given_name} {self.patronymic}"

    def initials(self) -> str:
        """Return the name as a clerk signs it: initials then surname."""
        return f"{self.given_name[0]}. {self.patronymic[0]}. {self.surname}"


def _feminise(stem: str) -> str:
    """Return the feminine form of a surname stem."""
    return f"{stem}ova" if not stem.endswith("o") else f"{stem}yeva"


def sample_person(
    rng: random.Random,
    script: Script,
    is_female: bool | None = None,
    surname_stem: str | None = None,
    father_name: str | None = None,
) -> Person:
    """Sample one person's name.

    Args:
        rng: Random source.
        script: Alphabet to write the name in.
        is_female: Force a gender instead of sampling one.
        surname_stem: Share a family name with an already-sampled relative.
        father_name: Build the patronymic from a known father's given name.

    Returns:
        The sampled name, already written in ``script``.
    """
    female = rng.random() < 0.5 if is_female is None else is_female
    stem = surname_stem or rng.choice(_SURNAME_STEMS)
    surname = _feminise(stem) if female else f"{stem}ov"
    given = rng.choice(_FEMALE_NAMES if female else _MALE_NAMES)

    father = father_name or rng.choice(_MALE_NAMES)
    patronymic = f"{father}ovna" if female else f"{father}ovich"

    return Person(
        surname=in_script(surname, script),
        given_name=in_script(given, script),
        patronymic=in_script(patronymic, script),
        is_female=female,
    )


def surname_stem(rng: random.Random) -> str:
    """Sample a family name stem, so relatives can share one."""
    return rng.choice(_SURNAME_STEMS)


def sample_place(rng: random.Random, script: Script) -> dict[str, str]:
    """Sample a place of birth or death.

    Args:
        rng: Random source.
        script: Alphabet to write the place names in.

    Returns:
        ``country``, ``region``, ``district`` and ``settlement``, each
        already written in ``script`` with its Uzbek suffix.
    """
    region = rng.choice(list(DISTRICTS))
    district = rng.choice(DISTRICTS[region])
    is_city = rng.random() < 0.5

    return {
        "country": in_script("O'zbekiston", script),
        "region": in_script(f"{region} viloyati", script),
        "district": in_script(
            f"{district} {'shahri' if is_city else 'tumani'}", script
        ),
        "settlement": in_script(
            district if is_city else f"{district} shaharchasi", script
        ),
    }


@dataclass(frozen=True)
class Address:
    """A home address, as a letter's header block writes it.

    Args:
        city: The city or district the street is in.
        street: Street name, without the word "street".
        house: House number, which may carry a letter such as "7a".
        flat: Flat or entrance number, empty when the home is a house.
    """

    city: str
    street: str
    house: str
    flat: str = ""

    def short(self, script: Script) -> str:
        """Return the address up to the house number, naming no dwelling.

        A sentence usually has to inflect the word for the dwelling —
        "uyning chegarasi", "uyim chegarasidan" — so it takes the address
        this far and supplies that word itself.

        Args:
            script: Alphabet to write the fixed words in.

        Returns:
            The address, e.g. "Urganch shahar A. Qodiriy ko'chasi 11".
        """
        street_word = in_script("ko'chasi", script)
        return f"{self.city} {self.street} {street_word} {self.house}"

    def line(self, script: Script) -> str:
        """Return the address as a header block writes it.

        Args:
            script: Alphabet to write the fixed words in.

        Returns:
            The address, e.g. "Urganch shahar A. Qodiriy ko'chasi 11 uy".
        """
        parts = [self.short(script), in_script("uy", script)]
        if self.flat:
            parts.append(f"{self.flat} {in_script('xonadon', script)}")
        return " ".join(parts)


def sample_address(
    rng: random.Random, script: Script, city: str | None = None
) -> Address:
    """Sample a home address.

    Args:
        rng: Random source.
        script: Alphabet to write the names in.
        city: Keep an already-sampled city instead of drawing one, so that
            neighbours in the same letter share it.

    Returns:
        The address, already written in ``script``.
    """
    if city is None:
        region = rng.choice(list(DISTRICTS))
        district = rng.choice(DISTRICTS[region])
        city = in_script(f"{district} shahar", script)

    house = str(rng.randint(1, 120))
    # A few houses on a street carry a letter rather than a plain number.
    if rng.random() < 0.15:
        house += in_script(rng.choice("abv"), script)

    return Address(
        city=city,
        street=in_script(rng.choice(STREETS), script),
        house=house,
        flat=str(rng.randint(1, 60)) if rng.random() < 0.35 else "",
    )


def sample_mahalla(rng: random.Random, script: Script) -> str:
    """Sample the name of a neighbourhood committee."""
    return in_script(rng.choice(_MAHALLAS), script)


def sample_organisation(rng: random.Random, script: Script) -> str:
    """Sample the name of a legal entity, as its letterhead writes it.

    Args:
        rng: Random source.
        script: Alphabet to write the name in.

    Returns:
        A company such as ``"Gulzor" MChJ``, or an institution such as
        ``Navro'z umumta'lim maktabi``.
    """
    name = rng.choice(_ORGANISATION_NAMES)
    if rng.random() < 0.5:
        return in_script(f'"{name}" {rng.choice(_ORGANISATION_FORMS)}', script)
    return in_script(f"{name} {rng.choice(_INSTITUTIONS)}", script)


def sample_office(rng: random.Random, script: Script) -> str:
    """Sample the name of a civil registry office."""
    region = rng.choice(list(DISTRICTS))
    district = rng.choice(DISTRICTS[region])
    kind = rng.choice(("shahri", "tumani"))
    return in_script(f"{district} {kind} FHDYO bo'limi", script)


def sample_nationality(rng: random.Random, script: Script) -> str:
    """Sample a nationality as written on a certificate."""
    return in_script(rng.choice(_NATIONALITIES), script)


def sample_cause_of_death(rng: random.Random, script: Script) -> str:
    """Sample a cause of death."""
    return in_script(rng.choice(_CAUSES_OF_DEATH), script)


def month_name(rng: random.Random, script: Script) -> tuple[int, str]:
    """Sample a month, returning its number and its name in ``script``."""
    index = rng.randrange(len(MONTHS))
    return index + 1, in_script(MONTHS[index], script)


def year_in_words(year: int, script: Script) -> str:
    """Spell a four-digit year out in Uzbek, as the forms require.

    Args:
        year: The year to spell, from 1900 to 2099.
        script: Alphabet to write it in.

    Returns:
        The year in words, e.g. "ikki ming o'n beshinchi".

    Raises:
        ValueError: If the year is outside the supported range.
    """
    if not 1900 <= year <= 2099:
        raise ValueError(f"Cannot spell the year {year}")

    thousands = "ming" if year < 2000 else "ikki ming"
    remainder = year % 1000 if year < 2000 else year % 2000
    words = [thousands] if year >= 2000 else ["bir ming"]

    if year < 2000:
        remainder = year - 1000

    hundreds, remainder = divmod(remainder, 100)
    if hundreds:
        words.append(f"{_UNITS[hundreds - 1]} yuz")

    tens, units = divmod(remainder, 10)
    if tens:
        words.append(_TEENS_TENS[tens * 10])
    if units:
        words.append(_UNITS[units - 1])

    ordinal = _ordinal(words.pop())
    words.append(ordinal)
    return in_script(" ".join(words), script)


def _ordinal(word: str) -> str:
    """Turn the last word of a spelled-out number into an ordinal."""
    if word.endswith(("a", "i", "o", "u", "e")):
        return f"{word}nchi"
    return f"{word}inchi"
