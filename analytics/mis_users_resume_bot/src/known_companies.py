from __future__ import annotations

from typing import Callable, Dict, List, Set


KNOWN_COMPANIES_CANONICAL: List[str] = [
    # Big4
    "PwC",
    "Deloitte",
    "EY",
    "KPMG",
    # Strategy
    "McKinsey & Company",
    "Boston Consulting Group (BCG)",
    "Bain & Company",
    # Global IT / Consulting / Outsourcing
    "Accenture",
    "IBM",
    "Capgemini",
    "Cognizant",
    "Tata Consultancy Services (TCS)",
    "Infosys",
    "Wipro",
    "EPAM",
    "Luxoft",
    "GlobalLogic",
    # Big Tech / SaaS
    "Google",
    "Amazon",
    "Microsoft",
    "Apple",
    "Meta",
    "Netflix",
    "Adobe",
    "SAP",
    "Oracle",
    "Salesforce",
    # RU/CIS banks
    "Сбер",
    "ВТБ",
    "Альфа-Банк",
    "Т-Банк",
    "Газпромбанк",
    "Райффайзенбанк",
    # RU/CIS tech/ecom
    "Яндекс",
    "VK",
    "Ozon",
    "Wildberries",
    "Avito",
    # RU/CIS retail/telecom
    "X5 Group",
    "Магнит",
    "МТС",
    "Beeline",
    "МегаФон",
    "Ростелеком",
    "Tele2",
    # RU/CIS energy/industry
    "Газпром",
    "Роснефть",
    "Лукойл",
    "Норникель",
    "Северсталь",
    "Русал",
    "Росатом",
    "РЖД",
    "Аэрофлот",
    # Security / ERP / IT
    "Kaspersky",
    "1C",
    "Т1",
    "Сбертех",
    # Previously used
    "AVO",
]


KNOWN_COMPANY_ALIASES: Dict[str, List[str]] = {
    "PwC": [
        "PWC",
        "PwC",
        "PricewaterhouseCoopers",
        "Price Waterhouse Coopers",
        "Прайсвотерхаускуперс",
        "Прайс Уотерхаус Куперс",
    ],
    "EY": [
        "Ernst & Young",
        "Ernst and Young",
        "E&Y",
        "Эрнст энд Янг",
    ],
    "Deloitte": [
        "Deloitte Consulting",
        "Deloitte Touche Tohmatsu",
        "Делойт",
    ],
    "KPMG": [
        "КПМГ",
    ],
    "McKinsey & Company": [
        "McKinsey",
        "McKinsey & Company",
    ],
    "Boston Consulting Group (BCG)": [
        "BCG",
        "Boston Consulting Group",
        "БКГ",
    ],
    "Bain & Company": [
        "Bain",
        "Bain & Company",
    ],
    "Сбер": [
        "Сбербанк",
        "Sber",
        "Sberbank",
        "Sberbank of Russia",
        "PAO Sberbank",
        "PJSC Sberbank",
        "ПАО Сбербанк",
    ],
    "EPAM": [
        "EPAM Systems",
        "EPAM Systems, Inc.",
        "ООО ЭПАМ",
        "Эпам",
    ],
    "Альфа-Банк": [
        "Alfa Bank",
        "Alfa-Bank",
        "Альфа банк",
    ],
    "Т-Банк": [
        "Тинькофф",
        "Tinkoff",
        "Tinkoff Bank",
        "T-Bank",
        "Т Банк",
    ],
    "Beeline": [
        "ВымпелКом",
        "VimpelCom",
        "Bee Line",
    ],
    "AVO": [
        "avo в банке",
        "avo интегратор",
        "AVO (в банке)",
        "AVO (интегратор)",
    ],
    "Т1": [
        "T1",
        "Т1 Консалтинг",
        "T1 Consulting",
    ],
    "VK": [
        "ВК",
        "Mail.ru Group",
        "VK Company",
    ],
    "Ozon": [
        "Ozon Tech",
        "Озон",
    ],
    "МТС": [
        "MTS",
        "МТС Банк",
    ],
    "Ростелеком": [
        "Rostelecom",
    ],
    "РЖД": [
        "RZD",
        "Российские железные дороги",
    ],
    "Сбертех": [
        "SberTech",
        "Sber Technology",
    ],
}


def build_known_companies_norm(normalize_company_fn: Callable[[object], str]) -> Set[str]:
    out: Set[str] = set()
    for canonical in KNOWN_COMPANIES_CANONICAL:
        norm = normalize_company_fn(canonical)
        if norm and norm not in {"Not specified", "Other"}:
            out.add(norm)
        for variant in KNOWN_COMPANY_ALIASES.get(canonical, []):
            n = normalize_company_fn(variant)
            if n and n not in {"Not specified", "Other"}:
                out.add(n)
    return out
