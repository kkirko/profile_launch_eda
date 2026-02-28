from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from latex_parser import clean_text, is_tool_token, parse_period


COUNTRY_SYNONYMS: Dict[str, set[str]] = {
    "Russia": {"russia", "россия", "рф", "russian federation", "российская федерация", "ru"},
    "Belarus": {"belarus", "беларусь", "белоруссия", "republic of belarus", "республика беларусь"},
    "Kazakhstan": {"kazakhstan", "казахстан", "republic of kazakhstan", "республика казахстан"},
    "UAE": {"uae", "u.a.e", "united arab emirates", "оаэ", "объединенные арабские эмираты"},
    "Ukraine": {"ukraine", "украина"},
    "Poland": {"poland", "польша"},
    "Qatar": {"qatar", "катар"},
    "Georgia": {"georgia", "грузия", "sakartvelo"},
    "Turkey": {"turkey", "турция"},
    "Uzbekistan": {"uzbekistan", "узбекистан"},
    "Armenia": {"armenia", "армения"},
    "Lithuania": {"lithuania", "литва"},
    "Serbia": {"serbia", "сербия"},
    "Germany": {"germany", "германия", "deutschland"},
    "United Kingdom": {"united kingdom", "uk", "great britain", "england", "великобритания"},
    "USA": {"usa", "us", "united states", "соединенные штаты", "сша"},
    "Cyprus": {"cyprus", "кипр"},
    "Latvia": {"latvia", "латвия"},
    "Bulgaria": {"bulgaria", "болгария"},
    "Montenegro": {"montenegro", "черногория"},
    "Netherlands": {"netherlands", "нидерланды", "holland"},
    "Spain": {"spain", "испания"},
    "Finland": {"finland", "финляндия"},
    "Austria": {"austria", "австрия"},
    "Canada": {"canada", "канада"},
    "Thailand": {"thailand", "тайланд", "таиланд"},
    "Israel": {"israel", "израиль"},
}

CITY_SYNONYMS: Dict[str, set[str]] = {
    "Moscow": {"moscow", "москва", "moskva"},
    "Saint Petersburg": {
        "saint petersburg",
        "st petersburg",
        "st. petersburg",
        "st-petersburg",
        "saint-petersburg",
        "санкт петербург",
        "санкт-петербург",
        "спб",
    },
    "Minsk": {"minsk", "минск"},
    "Almaty": {"almaty", "алматы", "alma-ata", "алма-ата", "alma ata"},
    "Kyiv": {"kyiv", "kiev", "киев"},
    "Tashkent": {"tashkent", "ташкент"},
    "Yerevan": {"yerevan", "ереван"},
    "Warsaw": {"warsaw", "варшава"},
    "Vilnius": {"vilnius", "вильнюс"},
    "Tbilisi": {"tbilisi", "тбилиси"},
    "Doha": {"doha", "доха"},
    "Dubai": {"dubai", "дубай"},
    "Abu Dhabi": {"abu dhabi", "абу даби", "abu-dhabi"},
    "Samara": {"samara", "самара"},
    "Novosibirsk": {"novosibirsk", "новосибирск"},
}

CITY_TO_COUNTRY: Dict[str, str] = {
    "Moscow": "Russia",
    "Saint Petersburg": "Russia",
    "Minsk": "Belarus",
    "Almaty": "Kazakhstan",
    "Kyiv": "Ukraine",
    "Tashkent": "Uzbekistan",
    "Yerevan": "Armenia",
    "Warsaw": "Poland",
    "Vilnius": "Lithuania",
    "Tbilisi": "Georgia",
    "Doha": "Qatar",
    "Dubai": "UAE",
    "Abu Dhabi": "UAE",
    "Samara": "Russia",
    "Novosibirsk": "Russia",
}

DOMAIN_SYNONYMS: Dict[str, set[str]] = {
    "Product/Project Management": {
        "product project",
        "product/project",
        "product management",
        "project management",
        "program management",
        "product/project management",
        "product project management",
    },
    "Engineering/IT": {
        "engineering",
        "engineering software",
        "engineering/it",
        "engineering/software",
        "software",
        "software engineering",
        "it",
    },
    "Design/Creative": {
        "design",
        "design creative",
        "design/creative",
        "ux ui",
        "ux/ui",
    },
    "Operations/Administration": {
        "operations",
        "operations admin",
        "operations/administration",
        "operations/admin",
        "administration",
        "ops",
    },
    "Finance/Legal/HR": {
        "finance legal hr",
        "finance/legal/hr",
        "finance",
        "legal",
        "hr",
        "human resources",
    },
    "Marketing/Sales": {
        "marketing sales",
        "marketing/sales",
        "marketing",
        "sales",
    },
    "Data/ML/Analytics": {
        "data",
        "data analytics",
        "data/ml/analytics",
        "analytics",
        "bi",
    },
    "Healthcare/Education": {
        "healthcare education",
        "healthcare/education",
        "healthcare",
        "education",
    },
}

ROLE_FAMILY_RULES = [
    (
        "Finance/Legal/HR",
        [
            r"finance",
            r"account",
            r"audit",
            r"legal",
            r"\bhr\b",
            r"human resources",
            r"recruit",
            r"talent acquisition",
            r"sourcer",
            r"talent partner",
            r"hrbp",
            r"рекрутер",
            r"подбор",
            r"сорсинг",
            r"кадров",
            r"финанс",
            r"юрист",
        ],
    ),
    (
        "Design/Creative",
        [
            r"\bux\b",
            r"\bui\b",
            r"ux researcher",
            r"ux research",
            r"ux designer",
            r"product designer",
            r"interaction designer",
            r"дизайнер интерфейс",
            r"ux[- ]?исследов",
            r"design",
            r"designer",
            r"дизайн",
            r"creative",
        ],
    ),
    (
        "Product/Project Management",
        [
            r"product",
            r"project",
            r"program",
            r"delivery",
            r"scrum",
            r"agile",
            r"business analyst",
            r"бизнес[- ]?аналит",
            r"system analyst",
            r"системн\w* аналит",
            r"requirements?",
            r"требован",
            r"\bbpmn\b",
            r"\buml\b",
            r"use case",
            r"продукт",
            r"проект",
        ],
    ),
    (
        "Data/ML/Analytics",
        [
            r"\bdata\b",
            r"analytics",
            r"data analyst",
            r"bi analyst",
            r"\bbi\b",
            r"machine learning",
            r"\bml\b",
            r"\bai\b",
            r"аналитик данных",
            r"аналитика данных",
            r"data science",
            r"дата[- ]?сайнс",
        ],
    ),
    (
        "Engineering/IT",
        [
            r"engineer",
            r"developer",
            r"backend",
            r"frontend",
            r"fullstack",
            r"devops",
            r"sre",
            r"software",
            r"qa",
            r"test",
            r"разработ",
            r"инженер",
            r"тестиров",
            r"программист",
            r"архитектор",
        ],
    ),
    (
        "Marketing/Sales",
        [
            r"marketing",
            r"sales",
            r"growth",
            r"crm",
            r"brand",
            r"маркет",
            r"продаж",
        ],
    ),
    (
        "Operations/Administration",
        [
            r"operations?",
            r"administration",
            r"admin",
            r"supply",
            r"logistic",
            r"operational",
            r"операцион",
            r"админ",
            r"логист",
        ],
    ),
    (
        "Healthcare/Education",
        [
            r"health",
            r"medical",
            r"clinic",
            r"мед",
            r"education",
            r"teacher",
            r"tutor",
            r"образован",
            r"research",
            r"науч",
        ],
    ),
]

SENIORITY_MAP = [
    ("C-level", [r"\bc-level\b", r"\bcxo\b", r"\bceo\b", r"\bcto\b", r"\bcfo\b", r"\bvp\b", r"chief", r"director", r"head", r"директор", r"руководит", r"начальник"]),
    ("Lead", [r"\blead\b", r"team lead", r"tech lead", r"лид", r"ведущ"]),
    ("Senior", [r"\bsenior\b", r"старш"]),
    ("Middle", [r"\bmiddle\b", r"mid", r"мидл", r"специалист"]),
    ("Junior", [r"\bjunior\b", r"jun", r"intern", r"младш", r"стажер"]),
]

INDUSTRY_MAP = {
    "it": "IT/Software",
    "software": "IT/Software",
    "saas": "IT/Software",
    "финтех": "Banking/FinTech",
    "fintech": "Banking/FinTech",
    "bank": "Banking/FinTech",
    "банк": "Banking/FinTech",
    "retail": "E-commerce/Retail",
    "e-commerce": "E-commerce/Retail",
    "маркетплейс": "E-commerce/Retail",
    "telecom": "Telecom",
    "телеком": "Telecom",
    "logistics": "Logistics/Transport",
    "transport": "Logistics/Transport",
    "consult": "Consulting",
    "консалт": "Consulting",
    "energy": "Energy",
    "oil": "Energy",
    "нефт": "Energy",
    "edu": "Education",
    "образован": "Education",
    "health": "Healthcare",
    "мед": "Healthcare",
    "gov": "Public Sector",
    "гос": "Public Sector",
    "министер": "Public Sector",
}

COMPANY_SYNONYMS: Dict[str, set[str]] = {
    "Сбер": {
        "сбер",
        "сбербанк",
        "sber",
        "sberbank",
        "sberbank of russia",
        "russian sberbank",
        "пао сбербанк",
        "pao sberbank",
        "pjsc sberbank",
        "sberbank corporate banking corporate solutions center",
    },
    "ВТБ": {
        "втб",
        "банк втб",
        "vtb",
        "vtb bank",
        "pao vtb",
        "pjsc vtb",
        "пао втб",
    },
    "Газпромбанк": {
        "газпромбанк",
        "gazprombank",
        "оао газпромбанк",
        "gazprombank oao",
    },
    "Яндекс": {
        "яндекс",
        "yandex",
        "ооо яндекс",
        "yandex llc",
    },
    "EPAM": {
        "epam",
        "epam systems",
        "epam system",
        "эпам",
    },
    "Газпром нефть": {
        "gazprom neft",
        "газпром нефть",
        "пao газпром нефть",
    },
    "ДОМ.РФ": {
        "дом рф",
        "dom rf",
        "domrf",
        "домрф",
    },
    "ИК СИБИНТЕК": {
        "ик сибинтек",
        "сибинтек",
        "ik sibintek",
    },
    "Лаборатория Касперского": {
        "лаборатория касперского",
        "kaspersky",
        "kaspersky lab",
    },
    "Avito": {
        "avito",
        "авито",
    },
    "Komus": {
        "komus",
        "комус",
    },
    "Газпром": {
        "gazprom",
        "газпром",
    },
    "NLMK": {
        "nlmk",
        "нлмк",
    },
    "МТС Банк": {
        "mts bank",
        "мтс банк",
    },
    "Ozon Bank": {
        "ozon bank",
        "озон банк",
    },
    "Промсвязьбанк": {
        "promsvyazbank",
        "промсвязьбанк",
    },
    "SberKorus": {
        "sberkorus",
        "сберкорус",
    },
    "Альфа-Банк": {
        "alfa bank",
        "альфа банк",
    },
    "Мираторг": {
        "miratorg",
        "мираторг",
    },
    "ADANI": {
        "adani",
        "адани",
    },
    "Lenta": {
        "lenta",
        "лента",
    },
    "Severstal": {
        "severstal",
        "северсталь",
    },
    "Technoserv": {
        "technoserv",
        "техносерв",
    },
    "IT_One": {
        "it one",
        "it_one",
    },
    "Metro Cash&Carry": {
        "metro cash and carry",
        "метро cash and carry",
        "metro cash carry",
        "metro cash&carry",
    },
}

COMPANY_NOISE_TOKENS = {
    "ooo",
    "ооо",
    "ao",
    "ао",
    "oao",
    "оао",
    "pao",
    "пао",
    "zao",
    "зао",
    "ip",
    "ип",
    "llc",
    "inc",
    "ltd",
    "corp",
    "jsc",
    "pjsc",
    "gmbh",
    "co",
    "group",
    "holding",
    "company",
}

COMPANY_NOT_SPECIFIED = {"not specified", "unknown", "n/a", "na", "none"}

CYR_TO_LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

_COMPANY_SYNONYM_LOOKUP: Dict[str, str] = {}


def canonical_text(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = text.replace("\\&", "&")
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("\\", " ")
    text = text.replace("ё", "е").lower().strip()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text


def _geo_token(token: object) -> str:
    text = canonical_text(token)
    if not text:
        return ""
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_geo_separators(text: str) -> str:
    out = text
    out = re.sub(r"[|•;]+", ",", out)
    out = re.sub(r"\s*/\s*", ",", out)
    out = re.sub(r"\s+(?:-|—|–)\s+", ",", out)
    out = re.sub(r"\s*,\s*", ",", out)
    out = re.sub(r",{2,}", ",", out)
    return out.strip(" ,")


def _match_synonym(token: str, mapping: Dict[str, set[str]]) -> str:
    t = _geo_token(token)
    if not t:
        return ""
    padded = f" {t} "
    for canonical, variants in mapping.items():
        for variant in variants:
            v = _geo_token(variant)
            if not v:
                continue
            if t == v:
                return canonical
            if f" {v} " in padded:
                return canonical
    return ""


def _clean_company_text(value: object) -> str:
    text = canonical_text(value)
    if not text:
        return ""
    text = re.sub(r"&+", " and ", text)
    text = re.sub(r"[|/•]+", " ", text)
    text = re.sub(r"[,\.;:!\?\"'`«»“”„_]+", " ", text)
    text = re.sub(r"[\(\)\[\]\{\}]", " ", text)
    text = re.sub(r"[—–-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def company_clean_key(value: object) -> str:
    text = _clean_company_text(value)
    if not text:
        return ""

    tokens = []
    for token in text.split():
        t = token.strip()
        if not t:
            continue
        if t in COMPANY_NOISE_TOKENS:
            continue
        if len(t) == 1 and not t.isdigit():
            continue
        tokens.append(t)

    key = " ".join(tokens).strip()
    return key


def transliterate_to_latin(value: object) -> str:
    text = canonical_text(value)
    if not text:
        return ""
    out_chars = [CYR_TO_LAT.get(ch, ch) for ch in text]
    out = "".join(out_chars)
    out = re.sub(r"[^a-z0-9\s]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def company_key_translit(value: object) -> str:
    key = company_clean_key(value)
    if not key:
        return ""
    return transliterate_to_latin(key)


def _company_synonym_lookup() -> Dict[str, str]:
    global _COMPANY_SYNONYM_LOOKUP
    if _COMPANY_SYNONYM_LOOKUP:
        return _COMPANY_SYNONYM_LOOKUP

    lookup: Dict[str, str] = {}
    for canonical, variants in COMPANY_SYNONYMS.items():
        for variant in set(variants) | {canonical}:
            key = company_clean_key(variant)
            if key:
                lookup[key] = canonical
                lookup[transliterate_to_latin(key)] = canonical
    _COMPANY_SYNONYM_LOOKUP = lookup
    return _COMPANY_SYNONYM_LOOKUP


def hash_user_id(value: object) -> str:
    raw = clean_text(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def normalize_empty_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    object_cols = out.select_dtypes(include=["object"]).columns
    for col in object_cols:
        out[col] = out[col].astype(str).str.strip()
        out.loc[out[col].str.lower().isin({"", "none", "null", "nan", "n/a", "na", "-"}), col] = np.nan
    return out


def parse_bool_series(series: pd.Series, default_false: bool = True) -> pd.Series:
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False})
    )
    if default_false:
        return mapped.astype("boolean").fillna(False).astype(bool)
    return mapped


def find_job_indices(columns: Iterable[str]) -> List[int]:
    idx = set()
    pattern = re.compile(r"^talentCard\.jobs\[(\d+)\]\.")
    for col in columns:
        m = pattern.match(col)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def build_jobs_long_talent(df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    indices = find_job_indices(df.columns)
    fields = [
        "company",
        "region",
        "industry",
        "job_title",
        "seniority",
        "employment_period",
        "chronology_order",
        "responsibilities",
        "achievements",
    ]

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        user_hash = row["user_hash"]
        for i in indices:
            payload: Dict[str, object] = {"user_hash": user_hash, "job_index": i, "source": "talentCard"}
            for field in fields:
                col = f"talentCard.jobs[{i}].{field}"
                payload[field] = row[col] if col in df.columns else np.nan

            has_data = any(clean_text(payload[x]) for x in ["company", "region", "industry", "job_title", "employment_period"])
            if not has_data:
                continue

            payload["responsibilities_len"] = len(clean_text(payload["responsibilities"]))
            payload["achievements_len"] = len(clean_text(payload["achievements"]))
            payload["has_numbers_in_achievements"] = bool(re.search(r"\d|%|\$|€|₽", clean_text(payload["achievements"])))

            period = parse_period(payload.get("employment_period", ""), as_of_date)
            payload["start_date"] = period.start_date
            payload["end_date"] = period.end_date
            payload["is_present"] = period.is_present
            payload["period_parse_ok"] = period.parse_ok
            payload["period_parse_quality"] = period.parse_quality

            if pd.notna(period.start_date) and pd.notna(period.end_date):
                months = (
                    (period.end_date.year - period.start_date.year) * 12
                    + (period.end_date.month - period.start_date.month)
                    + 1
                )
                payload["duration_months"] = float(months) if 1 <= months <= 720 else np.nan
            else:
                payload["duration_months"] = np.nan

            payload["chronology_order"] = pd.to_numeric(payload.get("chronology_order"), errors="coerce")
            rows.append(payload)

    jobs = pd.DataFrame(rows)
    if jobs.empty:
        cols = [
            "user_hash",
            "job_index",
            "source",
            "company",
            "region",
            "industry",
            "job_title",
            "seniority",
            "employment_period",
            "chronology_order",
            "responsibilities_len",
            "achievements_len",
            "has_numbers_in_achievements",
            "start_date",
            "end_date",
            "is_present",
            "period_parse_ok",
            "period_parse_quality",
            "duration_months",
        ]
        return pd.DataFrame(columns=cols)

    return jobs


def normalize_region(region: object) -> str:
    raw = clean_text(region)
    if not raw:
        return "Not specified"
    c = _normalize_geo_separators(canonical_text(raw))
    if not c:
        return "Not specified"

    if c in {"not specified", "unknown", "n/a"}:
        return "Not specified"
    if c in {"other"}:
        return "Other"
    if c in {"remote", "remotely", "удаленно", "удаленно,россия", "удаленно,remote", "онлайн"}:
        return "Remote"

    if "," not in c:
        city_full = _match_synonym(c, CITY_SYNONYMS)
        if city_full:
            return city_full
        country_full = _match_synonym(c, COUNTRY_SYNONYMS)
        if country_full:
            return country_full

    tokens = [t.strip() for t in c.split(",") if t.strip()]
    if not tokens:
        tokens = [c]

    country = ""
    city = ""

    # Prefer token-level mapping first, then whole-string fallback.
    for token in tokens:
        if not country:
            country = _match_synonym(token, COUNTRY_SYNONYMS)
        if not city:
            city = _match_synonym(token, CITY_SYNONYMS)

    if not country:
        country = _match_synonym(c, COUNTRY_SYNONYMS)
    if not city:
        city = _match_synonym(c, CITY_SYNONYMS)

    # Country-only values should collapse into one canonical country label.
    if len(tokens) == 1 and country and not city:
        return country

    # For city+country combinations, keep a single city-level canonical region.
    if city:
        return city
    if country:
        return country

    cleaned = ", ".join([token.strip().title() for token in tokens if token.strip()])
    return cleaned if cleaned else "Not specified"


def normalize_company(company: object) -> str:
    raw = clean_text(company)
    if not raw:
        return "Not specified"

    key = company_clean_key(raw)
    if not key or key in COMPANY_NOT_SPECIFIED:
        return "Not specified"
    if key == "other":
        return "Other"

    lookup = _company_synonym_lookup()
    key_lat = transliterate_to_latin(key)
    if key in lookup:
        return lookup[key]
    if key_lat in lookup:
        return lookup[key_lat]

    # Handles patterns like "sberbank --- ...", "пао сбербанк ..."
    if key.startswith("sberbank") or key.startswith("сбербанк") or key == "sber":
        return "Сбер"

    tokens = key.split()
    if len(tokens) == 1 and len(tokens[0]) <= 5:
        return tokens[0].upper()

    display = _clean_company_text(raw)
    display_tokens = [tok for tok in display.split() if tok not in COMPANY_NOISE_TOKENS]
    display = " ".join(display_tokens).strip()
    if not display:
        display = key
    display_tokens = display.split()
    if len(display_tokens) == 1 and len(display_tokens[0]) <= 5:
        return display_tokens[0].upper()
    return display


def normalize_industry(industry: object) -> str:
    raw = clean_text(industry)
    if not raw:
        return "Not specified"
    c = canonical_text(raw)
    for key, value in INDUSTRY_MAP.items():
        if key in c:
            return value
    return raw


def normalize_domain(raw: object) -> str:
    text = canonical_text(raw)
    if not text:
        return "Not specified"

    text = re.sub(r"[/|•]+", " ", text)
    text = re.sub(r"[\-_,;:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in {"not specified", "unknown", "n a", "na", "none"}:
        return "Not specified"
    if text in {"other", "другое"}:
        return "Other"

    for canonical, variants in DOMAIN_SYNONYMS.items():
        for variant in variants:
            v = canonical_text(variant)
            if text == v:
                return canonical

    # Fallback by semantic rules.
    for canonical, patterns in ROLE_FAMILY_RULES:
        if any(re.search(p, text) for p in patterns):
            return canonical

    return "Other"


def guess_country(region_norm: object) -> str:
    text = canonical_text(region_norm)
    if not text:
        return "Unknown"

    for country in COUNTRY_SYNONYMS:
        if canonical_text(country) == text:
            return country

    for city, country in CITY_TO_COUNTRY.items():
        if canonical_text(city) == text:
            return country

    m_country = _match_synonym(text, COUNTRY_SYNONYMS)
    if m_country:
        return m_country

    m_city = _match_synonym(text, CITY_SYNONYMS)
    if m_city:
        return CITY_TO_COUNTRY.get(m_city, "Unknown")
    return "Unknown"


def normalize_seniority(value: object) -> str:
    text = canonical_text(value)
    if not text:
        return "Unknown"
    for label, patterns in SENIORITY_MAP:
        if any(re.search(p, text) for p in patterns):
            return label
    return "Unknown"


def infer_role_family(selected_position: object, job_title: object, specialist_category: object) -> str:
    text = " ".join([clean_text(selected_position), clean_text(job_title), clean_text(specialist_category)])
    text = canonical_text(text)
    if not text:
        return "Other"
    for label, patterns in ROLE_FAMILY_RULES:
        if any(re.search(p, text) for p in patterns):
            return label
    specialist_domain = normalize_domain(specialist_category)
    if specialist_domain not in {"Other", "Not specified"}:
        return specialist_domain
    return "Other"


def split_skill_tokens(raw: object) -> List[str]:
    text = clean_text(raw)
    if not text:
        return []
    chunks = re.split(r"[;,\n\|]+", text)
    tokens: List[str] = []
    for ch in chunks:
        token = re.sub(r"\s+", " ", ch).strip(" .:-_()[]{}")
        if len(token) < 2:
            continue
        tokens.append(token)
    return tokens


def merge_skills_tools(
    overall_skills: object,
    overall_tools: object,
    latex_skills: List[str] | None,
    latex_tools: List[str] | None,
) -> Tuple[List[str], List[str]]:
    # Priority:
    # 1) LaTeX skills/tools sections
    # 2) talentCard.overall_skills
    # 3) talentCard.overall_tools
    skills: List[str] = []
    tools: List[str] = []

    if latex_skills:
        skills.extend(latex_skills)
    if latex_tools:
        tools.extend(latex_tools)

    skills.extend(split_skill_tokens(overall_skills))
    tools.extend(split_skill_tokens(overall_tools))

    final_skills: List[str] = []
    final_tools: List[str] = []

    for token in skills:
        if is_tool_token(token):
            final_tools.append(token)
        else:
            final_skills.append(token)
    final_tools.extend([tok for tok in tools if tok])

    # Deduplicate by canonical representation.
    seen = set()
    skills_out: List[str] = []
    for token in final_skills:
        canon = canonical_text(token)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        skills_out.append(token.strip())

    seen = set()
    tools_out: List[str] = []
    for token in final_tools:
        canon = canonical_text(token)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        tools_out.append(token.strip())

    return skills_out, tools_out


def build_mapping_table(raw_series: pd.Series, norm_series: pd.Series, value_name: str) -> pd.DataFrame:
    frame = pd.DataFrame({"raw": raw_series.fillna("Not specified").astype(str), "norm": norm_series.fillna("Not specified").astype(str)})
    out = frame.value_counts().reset_index(name="count")
    out = out.sort_values(["count", "raw"], ascending=[False, True])
    out = out.rename(columns={"raw": f"{value_name}_raw", "norm": f"{value_name}_norm"})
    return out


def summarize_user_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for user_hash, grp in jobs.groupby("user_hash"):
        g = grp.copy()
        g["sort_date"] = g["start_date"].fillna(pd.Timestamp("1900-01-01"))
        g = g.sort_values(["sort_date", "chronology_order", "job_index"], ascending=[True, True, True])

        durations = g["duration_months"].dropna()
        valid = g.dropna(subset=["start_date", "end_date"]).sort_values("start_date")

        total_exp_years = np.nan
        if not valid.empty:
            total_exp_years = round((valid["end_date"].max() - valid["start_date"].min()).days / 365.25, 2)

        current = None
        present = g[g["is_present"] == True]  # noqa: E712
        if not present.empty:
            current = present.sort_values(["start_date", "job_index"], ascending=[False, True]).iloc[0]
        elif not valid.empty:
            current = valid.sort_values(["end_date", "start_date"], ascending=[False, False]).iloc[0]
        else:
            current = g.iloc[-1]

        gaps = 0
        max_gap = 0.0
        overlaps = 0
        if len(valid) >= 2:
            for i in range(1, len(valid)):
                prev_end = valid.iloc[i - 1]["end_date"]
                cur_start = valid.iloc[i]["start_date"]
                delta = (cur_start - prev_end).days
                if delta > 31:
                    gaps += 1
                    max_gap = max(max_gap, delta / 30.44)
                elif delta < -31:
                    overlaps += 1

        companies_series = g["company"].fillna("").astype(str).str.strip()
        regions_series = g["region"].fillna("").astype(str).str.strip()
        industries_series = g["industry"].fillna("").astype(str).str.strip()

        rows.append(
            {
                "user_hash": user_hash,
                "jobs_count_talentCard": int(len(g)),
                "companies_count": int(companies_series[companies_series.ne("")].nunique()),
                "regions_count": int(regions_series[regions_series.ne("")].nunique()),
                "industries_count": int(industries_series[industries_series.ne("")].nunique()),
                "total_experience_years": total_exp_years,
                "sum_tenure_years": round(durations.sum() / 12, 2) if len(durations) else np.nan,
                "avg_tenure_months": round(durations.mean(), 1) if len(durations) else np.nan,
                "median_tenure_months": round(durations.median(), 1) if len(durations) else np.nan,
                "career_gaps_count": int(gaps),
                "max_gap_months": round(max_gap, 1),
                "overlaps_count": int(overlaps),
                "current_company_talentCard": clean_text(current.get("company", "")),
                "current_region_talentCard": clean_text(current.get("region", "")),
                "current_job_title_talentCard": clean_text(current.get("job_title", "")),
                "current_industry_talentCard": clean_text(current.get("industry", "")),
                "current_seniority_talentCard": clean_text(current.get("seniority", "")),
                "resp_coverage": round((g["responsibilities_len"] > 0).mean(), 3),
                "ach_coverage": round((g["achievements_len"] > 0).mean(), 3),
                "impact_coverage": round(((g["achievements_len"] > 0) & g["has_numbers_in_achievements"]).mean(), 3),
                "period_parse_success": round(g["period_parse_ok"].fillna(False).mean(), 3),
            }
        )

    return pd.DataFrame(rows)


def aggregate_tokens(user_df: pd.DataFrame, token_col: str) -> pd.DataFrame:
    counter = Counter()
    for tokens in user_df[token_col]:
        for tok in (tokens or []):
            key = canonical_text(tok)
            if key:
                counter[key] += 1
    out = pd.DataFrame(counter.items(), columns=["token_norm", "count"]).sort_values("count", ascending=False)
    out["token_display"] = out["token_norm"].map(lambda x: x.title() if re.search(r"[a-z]", x) else x)
    return out.reset_index(drop=True)


def cooccurrence_pairs(user_df: pd.DataFrame, token_col: str, min_count: int = 2) -> pd.DataFrame:
    counter = Counter()
    for tokens in user_df[token_col]:
        uniq = sorted({canonical_text(t) for t in (tokens or []) if canonical_text(t)})
        for a, b in combinations(uniq, 2):
            counter[(a, b)] += 1

    rows = [(a, b, c) for (a, b), c in counter.items() if c >= min_count]
    out = pd.DataFrame(rows, columns=["token_a", "token_b", "count"]).sort_values("count", ascending=False)
    return out.reset_index(drop=True)


def leadership_level(title: object, seniority_norm: object, exp_years: float) -> str:
    text = canonical_text(title) + " " + canonical_text(seniority_norm)
    score = 0
    if any(x in text for x in ["head", "director", "chief", "vp", "руковод", "директор", "начальник", "c-level"]):
        score += 2
    if pd.notna(exp_years):
        if float(exp_years) >= 10:
            score += 1
        if float(exp_years) >= 15:
            score += 1
    if score >= 3:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def experience_bin(exp_years: float) -> str:
    if pd.isna(exp_years):
        return "Unknown"
    years = float(exp_years)
    if years <= 2:
        return "0-2"
    if years <= 5:
        return "3-5"
    if years <= 9:
        return "6-9"
    if years <= 14:
        return "10-14"
    return "15+"


def profile_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["summary_present"] = out["talentCard.overall_summary"].fillna("").astype(str).str.strip().ne("").astype(int)
    out["skills_present"] = out["skills_count"].fillna(0).astype(float).gt(0).astype(int)
    out["jobs_present"] = out["jobs_count_talentCard"].fillna(0).astype(float).gt(0).astype(int)

    out["profile_quality_score"] = (
        out["summary_present"] * 20
        + out["skills_present"] * 20
        + out["jobs_present"] * 20
        + out["resp_coverage"].fillna(0).clip(0, 1) * 10
        + out["ach_coverage"].fillna(0).clip(0, 1) * 10
        + out["impact_coverage"].fillna(0).clip(0, 1) * 10
        + out["period_parse_success"].fillna(0).clip(0, 1) * 10
    ).round(1)

    out["profile_quality_bucket"] = pd.cut(
        out["profile_quality_score"],
        bins=[-1, 40, 60, 75, 90, 101],
        labels=["0-40", "41-60", "61-75", "76-90", "91-100"],
    ).astype(str)
    return out
