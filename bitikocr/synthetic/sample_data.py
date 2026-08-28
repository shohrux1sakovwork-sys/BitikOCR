"""Hand-written field values used to smoke-test and demo the generators.

These are transcriptions of real archive pages, kept here so that generating
a preview batch never depends on a text generator being wired up. They are
deliberately small: a production run should feed the generators from a real
field-value sampler instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["SAMPLE_FIELDS", "sample_fields_for"]

_ARIZA_SAMPLES: tuple[dict[str, Any], ...] = (
    {
        "recipient": "Гулистон шаҳар ҳокими И. Й. Эрбековга",
        "applicant": (
            "Сирдарё вилояти Янгиер ш. Бахт кўчаси 4 уйда яшовчи фуқаро "
            "Усмонов С-дан"
        ),
        "body": (
            "Аризам мазмуни шундан иборатки менга Н. Зокиров кўчаси "
            '"Тинчлик" майдони ҳудудидан маиший хизмат кўрсатиш шахобчаси '
            "қуриш учун ер майдони ажратиб беришингизни сўрайман."
        ),
        "signature_name": "Усмонов С.",
        "date": "06.09.06 й",
        "reg_number": "У-1120/А",
        "reg_date": "06.09.06",
        "page_number": "4",
    },
    {
        "recipient": "Урганч шаҳар ҳокими О. Сапаевга",
        "applicant": (
            "Урганч шаҳар Мустақиллик кўчаси 46-уйда яшовчи фуқаро Солаева "
            "Дилором томонидан 17-маҳалла"
        ),
        "body": (
            "Бераман ушбу аризани шу ҳақда ким, менга юқорида кўрсатилган "
            "турар жойимга марҳума онам Солаева Панишажон номига эгалик "
            "ҳуқуқи белгилаб қарор чиқариб беришингизни сўрайман."
        ),
        "signature_name": "Солаева Дилором",
        "reg_number": "С-1025",
        "reg_date": "16.03.2016",
        "page_number": "3",
    },
    {
        "recipient": "Янгиариқ туман ҳокими К. Б. Сабировга",
        "applicant": (
            "Янгиариқ туман Оҳлабарг кўча 3 уйда яшовчи Раҳимова Тамара "
            "томонидан"
        ),
        "body": (
            "Бераман ушбу аризани шу ҳақда ким мен яшаб турган турар жой "
            "биноға эгалик ҳуқуқини белгилаб беришингизни сўрайман"
        ),
        "phone": "998907254253",
        "reg_number": "Р-605",
        "reg_date": "11 март",
        "page_number": "131",
    },
    {
        "recipient": "Гулистон шаҳар ҳокими И. Эрбековга",
        "applicant": "Абдураҳмонова кўчаси 9/2 уйда яшовчи Ҳабибов Ҳ дан",
        "body": (
            "менинг маҳалла ҳудудидаги ким ошди савдоси орқали сотиб олган "
            "ер майдонимга уй жой қуришга рухсат беришингизни сўрайман."
        ),
        "date": "14.09.06 й",
        "reg_number": "Ҳ-1133/А",
        "reg_date": "18.09.06",
        "page_number": "46",
    },
)

_DEATH_CERTIFICATE_SAMPLES: tuple[dict[str, Any], ...] = (
    {
        "surname": "Bekchanov",
        "given_name_patronymic": "Shavkat Ergashovich",
        "citizenship": "O'zbekiston",
        "death_year": "2014",
        "death_month": "sentabr",
        "death_day": "24",
        "death_year_in_words": "ikki ming o'n to'rtinchi yil",
        "age_at_death": "53",
        "record_year": "2014",
        "record_month": "sentabr",
        "record_day": "30",
        "record_number": "820",
        "cause_of_death": "O'pka arteriyasi tromboemboliyasi",
        "death_place_country": "O'zbekiston",
        "death_place_region": "Xorazm",
        "death_place_district": "Urganch",
        "death_place_settlement": "Urganch",
        "registration_office": "Urganch shahar FHDYO bo'limi",
        "issue_year": "2014",
        "issue_month": "sentabr",
        "issue_day": "30",
        "registrar_name": "N. Qurbonboyeva",
        "serial_number": "0072725",
    },
    {
        "surname": "Раҳимова",
        "given_name_patronymic": "Гулнора Тўраевна",
        "citizenship": "Ўзбекистон",
        "death_year": "2009",
        "death_month": "март",
        "death_day": "7",
        "death_year_in_words": "икки минг тўққизинчи йил",
        "age_at_death": "71",
        "record_year": "2009",
        "record_month": "март",
        "record_day": "9",
        "record_number": "114",
        "cause_of_death": "юрак ишемик касаллиги",
        "death_place_country": "Ўзбекистон",
        "death_place_region": "Сирдарё",
        "death_place_district": "Гулистон",
        "death_place_settlement": "Гулистон шаҳар",
        "registration_office": "Гулистон шаҳар ФҲДЁ бўлими",
        "issue_year": "2009",
        "issue_month": "март",
        "issue_day": "9",
        "registrar_name": "Ҳ. Эргашев",
        "serial_number": "0311482",
        "stamp_ring": "ЎЗБЕКИСТОН РЕСПУБЛИКАСИ * СИРДАРЁ ВИЛОЯТИ ФҲДЁ БЎЛИМИ *",
        "stamp_center": ["ФҲДЁ"],
    },
    {
        "surname": "Yo'ldoshev",
        "given_name_patronymic": "Bahodir Karimovich",
        "citizenship": "O'zbekiston",
        "death_year": "2019",
        "death_month": "dekabr",
        "death_day": "2",
        "death_year_in_words": "ikki ming o'n to'qqizinchi yil",
        "age_at_death": "66",
        "record_year": "2019",
        "record_month": "dekabr",
        "record_day": "4",
        "record_number": "1207",
        "cause_of_death": "Miya qon aylanishining o'tkir buzilishi",
        "death_place_country": "O'zbekiston",
        "death_place_region": "Xorazm",
        "death_place_district": "Yangiariq tumani",
        "death_place_settlement": "Yangiariq shaharchasi",
        "registration_office": "Yangiariq tumani FHDYO bo'limi",
        "issue_year": "2019",
        "issue_month": "dekabr",
        "issue_day": "4",
        "registrar_name": "S. Matyoqubova",
        "serial_number": "0459013",
    },
)

#: Document type mapped to the field sets available for it.
SAMPLE_FIELDS: Mapping[str, tuple[dict[str, Any], ...]] = {
    "ariza": _ARIZA_SAMPLES,
    "death_certificate": _DEATH_CERTIFICATE_SAMPLES,
}


def sample_fields_for(document_type: str) -> tuple[dict[str, Any], ...]:
    """Return the built-in field sets for a document type.

    Args:
        document_type: A registered document type name.

    Returns:
        The available field sets, in a stable order.

    Raises:
        KeyError: If no samples exist for that document type.
    """
    try:
        return SAMPLE_FIELDS[document_type]
    except KeyError as error:
        known = ", ".join(sorted(SAMPLE_FIELDS))
        raise KeyError(
            f"No sample fields for {document_type!r}. Known types: {known}"
        ) from error
