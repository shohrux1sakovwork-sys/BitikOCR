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

_BIRTH_CERTIFICATE_SAMPLES: tuple[dict[str, Any], ...] = (
    {
        "child_surname": "KARIMOV",
        "child_given_name": "OTABEK SHAVKAT O'G'LI",
        "child_birth_date": "2015 YIL YANVAR 12",
        "child_birth_year_words": "IKKI MING O'N BESHINCHI",
        "birth_country": "O'ZBEKISTON",
        "birth_region": "XORAZM VILOYATI",
        "birth_district": "URGANCH SHAHRI",
        "birth_settlement": "URGANCH",
        "record_year": "2015",
        "record_month": "YANVAR",
        "record_day": "19",
        "record_number": "1-2108-20-T-003",
        "father_surname": "KARIMOV",
        "father_given_name": "SHAVKAT ERGASHOVICH",
        "father_nationality": "O'ZBEK",
        "father_citizenship": "O'ZBEKISTON",
        "mother_surname": "KARIMOVA",
        "mother_given_name": "NODIRA BAXTIYOROVNA",
        "mother_nationality": "O'ZBEK",
        "mother_citizenship": "O'ZBEKISTON",
        "registry_office": "URGANCH SHAHRI FHDYO BO'LIMI",
        "issue_year": "2015",
        "issue_month": "YANVAR",
        "issue_day": "19",
        "registry_head_name": "M. X. QURBONBOYEVA",
        "form_series": "III-XX",
        "form_number": "0072725",
    },
    {
        "child_surname": "YO'LDOSHEVA",
        "child_given_name": "MOHIRA BAHODIR QIZI",
        "child_birth_date": "2019 YIL DEKABR 2",
        "child_birth_year_words": "IKKI MING O'N TO'QQIZINCHI",
        "birth_country": "O'ZBEKISTON",
        "birth_region": "SIRDARYO VILOYATI",
        "birth_district": "GULISTON SHAHRI",
        "birth_settlement": "GULISTON",
        "record_year": "2019",
        "record_month": "DEKABR",
        "record_day": "6",
        "record_number": "1-3104-11-T-118",
        "father_surname": "YO'LDOSHEV",
        "father_given_name": "BAHODIR KARIMOVICH",
        "father_nationality": "O'ZBEK",
        "father_citizenship": "O'ZBEKISTON",
        "mother_surname": "YO'LDOSHEVA",
        "mother_given_name": "GULNORA TO'RAYEVNA",
        "mother_nationality": "O'ZBEK",
        "mother_citizenship": "O'ZBEKISTON",
        "registry_office": "GULISTON SHAHRI FHDYO BO'LIMI",
        "issue_year": "2019",
        "issue_month": "DEKABR",
        "issue_day": "6",
        "registry_head_name": "H. ERGASHEV",
        "form_series": "II-SR",
        "form_number": "0459013",
        "stamp_ring": (
            "O'ZBEKISTON RESPUBLIKASI * SIRDARYO VILOYATI FHDYO BO'LIMI *"
        ),
        "stamp_center": ["FHDYO"],
    },
    {
        "child_surname": "RAHIMOV",
        "child_given_name": "JAVOHIR ULUG'BEK O'G'LI",
        "child_birth_date": "2022 YIL SENTABR 24",
        "child_birth_year_words": "IKKI MING YIGIRMA IKKINCHI",
        "birth_country": "O'ZBEKISTON",
        "birth_region": "XORAZM VILOYATI",
        "birth_district": "YANGIARIQ TUMANI",
        "birth_settlement": "YANGIARIQ SHAHARCHASI",
        "record_year": "2022",
        "record_month": "SENTABR",
        "record_day": "30",
        "record_number": "1-2108-04-T-820",
        "father_surname": "RAHIMOV",
        "father_given_name": "ULUG'BEK SOBIROVICH",
        "father_nationality": "O'ZBEK",
        "father_citizenship": "O'ZBEKISTON",
        "mother_surname": "RAHIMOVA",
        "mother_given_name": "TAMARA IBRAGIMOVNA",
        "mother_nationality": "O'ZBEK",
        "mother_citizenship": "O'ZBEKISTON",
        "registry_office": "YANGIARIQ TUMANI FHDYO BO'LIMI",
        "issue_year": "2022",
        "issue_month": "SENTABR",
        "issue_day": "30",
        "registry_head_name": "S. MATYOQUBOVA",
        "form_series": "I-XR",
        "form_number": "0311482",
    },
)


#: Document type mapped to the field sets available for it.
SAMPLE_FIELDS: Mapping[str, tuple[dict[str, Any], ...]] = {
    "ariza": _ARIZA_SAMPLES,
    "birth_certificate": _BIRTH_CERTIFICATE_SAMPLES,
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
