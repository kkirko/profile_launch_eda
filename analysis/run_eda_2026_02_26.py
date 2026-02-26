from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "snapshot_tue26.csv"
OUT_DIR = BASE_DIR / "analysis" / "output_2026_02_26"
FIG_DIR = OUT_DIR / "figures"

TARGET_DATE = pd.Timestamp("2026-02-26", tz="UTC").date()

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["font.family"] = "DejaVu Sans"

CATEGORY_FALLBACK = {
    "Product/Project Management": "Project / Program Management",
    "Engineering/IT": "Software Engineering",
    "Operations/Administration": "Operations / Administration / Supply Chain",
    "Finance/Legal/HR": "Finance / Accounting / Audit",
    "Marketing/Sales": "Marketing / Sales / Growth",
    "Design/Creative": "Design / Creative",
    "Healthcare/Education": "Healthcare / Education",
    "Other": "Other",
}

ROLE_PATTERNS: Dict[str, List[str]] = {
    "Executive / General Management": [
        r"\bceo\b",
        r"\bcfo\b",
        r"\bcto\b",
        r"\bcoo\b",
        r"\bchief\b",
        r"\bexecutive\b",
        r"\bvice president\b",
        r"\bvp\b",
        r"генеральн",
        r"топ-менедж",
        r"вице-президент",
    ],
    "Product Management": [
        r"\bproduct manager\b",
        r"\bproduct owner\b",
        r"\bhead of product\b",
        r"\bcpo\b",
        r"продукт",
        r"monetization",
    ],
    "Project / Program Management": [
        r"\bproject manager\b",
        r"\bprogram manager\b",
        r"\bdelivery manager\b",
        r"\bpm\b",
        r"\bpmo\b",
        r"руководител[ья]\s+проект",
        r"менеджер проектов",
        r"управлени[ея]\s+проект",
        r"scrum master",
    ],
    "Business / System Analysis": [
        r"\bbusiness analyst\b",
        r"\bsystems analyst\b",
        r"бизнес-аналит",
        r"системн[а-я]*\s+аналит",
        r"requirements",
        r"functional design",
    ],
    "Data / BI Analytics": [
        r"\bdata analyst\b",
        r"\bdata science\b",
        r"\bbi analyst\b",
        r"\bbi-?аналит",
        r"аналитик данных",
        r"power bi",
        r"tableau",
        r"qlik",
    ],
    "Software Engineering": [
        r"\bsoftware engineer\b",
        r"\bdeveloper\b",
        r"\bbackend\b",
        r"\bfrontend\b",
        r"\bfull[- ]?stack\b",
        r"разработчик",
        r"\.net",
        r"\bjava\b",
        r"\bpython\b",
        r"\breact\b",
        r"\bangular\b",
    ],
    "QA / Testing": [
        r"\bqa\b",
        r"\btesting\b",
        r"\btest engineer\b",
        r"\bquality assurance\b",
        r"тестиров",
        r"обеспечени[ея]\s+качества",
    ],
    "DevOps / SRE / Infrastructure": [
        r"\bdevops\b",
        r"\bsre\b",
        r"\bcloud\b",
        r"kubernetes",
        r"infrastructure",
        r"инфраструктур",
        r"ci\s*/\s*cd",
        r"ci-cd",
    ],
    "Design / Creative": [
        r"\bux\b",
        r"\bui\b",
        r"\bdesigner\b",
        r"дизайнер",
        r"\b3d\b",
        r"visual",
        r"creative",
    ],
    "Marketing / Sales / Growth": [
        r"\bmarketing\b",
        r"\bsales\b",
        r"\bpr\b",
        r"\bcommunications\b",
        r"\bgrowth\b",
        r"go-to-market",
        r"маркет",
        r"продаж",
        r"коммуникац",
    ],
    "Finance / Accounting / Audit": [
        r"\bfinance\b",
        r"\bfinancial\b",
        r"\baccount",
        r"\baudit\b",
        r"\bifrs\b",
        r"финансов",
        r"мсфо",
        r"бюджет",
    ],
    "HR / People": [
        r"\bhr\b",
        r"\bhrbp\b",
        r"human resources",
        r"people partner",
        r"talent",
        r"рекрут",
        r"персонал",
        r"кадров",
    ],
    "Legal / Compliance": [
        r"\blegal\b",
        r"\bcompliance\b",
        r"юрист",
        r"регулятор",
        r"санкц",
        r"\bkyc\b",
    ],
    "Operations / Administration / Supply Chain": [
        r"\boperations\b",
        r"\boperational\b",
        r"операцион",
        r"логист",
        r"supply chain",
        r"административ",
        r"\bвэд\b",
    ],
    "Healthcare / Education": [
        r"\bhealthcare\b",
        r"\bpharma\b",
        r"\beducation\b",
        r"\bedtech\b",
        r"медицин",
        r"образован",
    ],
}

ROLE_PRIORITY = [
    "Executive / General Management",
    "Product Management",
    "Project / Program Management",
    "Business / System Analysis",
    "Data / BI Analytics",
    "Software Engineering",
    "DevOps / SRE / Infrastructure",
    "QA / Testing",
    "Design / Creative",
    "Marketing / Sales / Growth",
    "Finance / Accounting / Audit",
    "HR / People",
    "Legal / Compliance",
    "Operations / Administration / Supply Chain",
    "Healthcare / Education",
    "Other",
]

REGION_RULES: Dict[str, List[str]] = {
    "Russia & CIS": [
        r"росси",
        r"\bрф\b",
        r"москв",
        r"санкт",
        r"\bснг\b",
        r"казахстан",
        r"беларус",
        r"узбекистан",
        r"армени",
        r"кыргыз",
        r"\bcis\b",
    ],
    "Europe": [
        r"\beurope\b",
        r"\beu\b",
        r"европ",
        r"\buk\b",
        r"germany",
        r"italy",
        r"france",
        r"spain",
        r"poland",
    ],
    "Middle East": [
        r"\buae\b",
        r"dubai",
        r"\bmena\b",
        r"\bgcc\b",
        r"middle east",
        r"saudi",
        r"qatar",
    ],
    "North America": [
        r"\busa\b",
        r"\bсша\b",
        r"north america",
        r"canada",
    ],
    "APAC": [
        r"\bapac\b",
        r"\basia\b",
        r"singapore",
        r"china",
        r"india",
        r"japan",
        r"korea",
    ],
}

EMPLOYER_TYPE_RULES: Dict[str, List[str]] = {
    "Banking / Fintech": [
        r"\bbank\b",
        r"бан[кк]",
        r"финтех",
        r"fintech",
        r"payment",
        r"кредит",
        r"treasury",
    ],
    "Telecom": [
        r"telecom",
        r"телеком",
        r"связ",
        r"operator",
        r"мобильн",
    ],
    "IT Product / SaaS": [
        r"\bsaas\b",
        r"product",
        r"digital platform",
        r"software",
        r"it services",
        r"продуктов",
        r"цифров",
    ],
    "System Integration / Outsourcing": [
        r"system integration",
        r"системн[а-я]*\s+интеграц",
        r"outsourcing",
        r"аутсорс",
        r"consulting",
        r"консалт",
    ],
    "Government / Public Sector": [
        r"government",
        r"гос",
        r"state",
        r"министер",
        r"public sector",
    ],
    "Industrial / Manufacturing": [
        r"industrial",
        r"промышлен",
        r"manufactur",
        r"factory",
        r"production",
    ],
    "Healthcare / Pharma": [
        r"healthcare",
        r"pharma",
        r"медицин",
        r"клиник",
        r"hospital",
    ],
    "Retail / E-commerce / FMCG": [
        r"retail",
        r"e-?commerce",
        r"fmcg",
        r"beauty",
        r"subscription",
    ],
    "Construction / Real Estate": [
        r"construction",
        r"строител",
        r"real estate",
        r"proptech",
        r"bim",
    ],
    "Logistics / Transport": [
        r"logistics",
        r"transport",
        r"supply chain",
        r"логист",
    ],
    "Education / EdTech": [
        r"education",
        r"edtech",
        r"образован",
    ],
    "Media / Gaming": [
        r"media",
        r"gaming",
        r"gameplay",
        r"producer",
    ],
}

EMPLOYER_FIELD_ALIASES = {
    "employer",
    "employer_name",
    "company",
    "company_name",
    "current_company",
    "current_employer",
    "last_company",
    "last_employer",
    "latest_company",
    "latest_employer",
    "organization",
    "organisation",
    "org_name",
    "workplace",
    "current_workplace",
}

YEARS_RE = re.compile(r"(?<!\d)(\d{1,2})\s*\+?\s*(?:years?|year|лет|года|год)", re.IGNORECASE)


def first_sentence(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    parts = re.split(r"[.!?\n]", text, maxsplit=1)
    return parts[0].strip()


def extract_years_experience(text: str) -> float:
    matches = YEARS_RE.findall(str(text or ""))
    if not matches:
        return math.nan
    values = [int(v) for v in matches]
    return float(max(values))


def score_from_patterns(text_lc: str, first_lc: str, patterns: Iterable[str]) -> int:
    score = 0
    for pattern in patterns:
        if re.search(pattern, text_lc):
            score += 1
        if re.search(pattern, first_lc):
            score += 2
    return score


def infer_role_family(summary: str, specialist_category: str) -> str:
    text = str(summary or "")
    text_lc = text.lower()
    first_lc = first_sentence(text).lower()
    scores: Dict[str, int] = {}

    for family, patterns in ROLE_PATTERNS.items():
        scores[family] = score_from_patterns(text_lc, first_lc, patterns)

    best_family, best_score = "Other", 0
    for family in ROLE_PRIORITY:
        score = scores.get(family, 0)
        if score > best_score:
            best_family, best_score = family, score

    if best_score == 0:
        return CATEGORY_FALLBACK.get(str(specialist_category), "Other")

    return best_family


def infer_seniority(summary: str, years_experience: float) -> str:
    text = str(summary or "").lower()

    if re.search(r"\bjunior\b|стаж[её]р|начинающ", text):
        return "Junior"

    if re.search(
        r"\b(ceo|cfo|cto|coo|chief|executive|vice president|vp|founder|co-founder)\b|"
        r"генеральн|топ-менедж|вице-президент|директор",
        text,
    ):
        return "Executive"

    if re.search(r"\b(lead|principal|architect|head)\b|руководител|тимлид|team lead", text):
        return "Lead"

    if re.search(r"\bsenior\b|старш", text):
        return "Senior"

    if re.search(r"\bmiddle\b|\bmid[- ]level\b", text):
        return "Middle"

    if pd.notna(years_experience):
        years = float(years_experience)
        if years >= 15:
            return "Lead"
        if years >= 8:
            return "Senior"
        if years >= 3:
            return "Middle"
        return "Junior"

    return "Unknown"


def infer_region(summary: str) -> str:
    text = str(summary or "")
    text_lc = text.lower()
    found_regions = []

    for region, patterns in REGION_RULES.items():
        if any(re.search(p, text_lc) for p in patterns):
            found_regions.append(region)

    if len(found_regions) > 1:
        return "Global / Multi-region"
    if len(found_regions) == 1:
        return found_regions[0]

    if re.search(r"[а-яё]", text_lc):
        return "Russia & CIS (inferred)"

    return "Not specified"


def infer_employer_type(summary: str) -> str:
    text_lc = str(summary or "").lower()

    scores: Dict[str, int] = {}
    for employer_type, patterns in EMPLOYER_TYPE_RULES.items():
        scores[employer_type] = sum(1 for p in patterns if re.search(p, text_lc))

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "Not specified"
    return best


def normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def clean_text_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "n/a", "na", "-"}:
        return ""
    return text


def find_employer_fields(columns: Iterable[str]) -> List[str]:
    fields = []
    for col in columns:
        if normalize_col_name(col) in EMPLOYER_FIELD_ALIASES:
            fields.append(col)
    return fields


def extract_employer_from_row(row: pd.Series, employer_fields: List[str]) -> str:
    for field in employer_fields:
        value = clean_text_value(row.get(field))
        if value:
            return value
    return ""


def detect_employer_source_field(row: pd.Series, employer_fields: List[str]) -> str:
    for field in employer_fields:
        value = clean_text_value(row.get(field))
        if value:
            return field
    return ""


def parse_skills(raw: str) -> List[str]:
    if pd.isna(raw):
        return []
    items = re.findall(r'"([^\"]+)"', str(raw))
    cleaned = []
    for item in items:
        value = item.strip().strip("()[]{} ")
        value = re.sub(r"\s+", " ", value)
        if len(value) >= 2:
            cleaned.append(value)
    return cleaned


def save_countplot(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    top_n: int | None = None,
    horizontal: bool = True,
    pre_counted: bool = False,
) -> None:
    data = series.copy() if pre_counted else series.value_counts(dropna=False)
    if top_n is not None:
        data = data.head(top_n)

    total = data.sum()

    fig, ax = plt.subplots(figsize=(13, max(5, 0.45 * len(data) + 2)))
    colors = sns.color_palette("viridis", n_colors=len(data))

    if horizontal:
        ax.barh(data.index.astype(str), data.values, color=colors)
        ax.invert_yaxis()
        for y_idx, val in enumerate(data.values):
            pct = val / total * 100 if total else 0
            ax.text(val + 0.2, y_idx, f"{val} ({pct:.1f}%)", va="center", fontsize=10)
    else:
        ax.bar(data.index.astype(str), data.values, color=colors)
        ax.tick_params(axis="x", rotation=30)
        for x_idx, val in enumerate(data.values):
            pct = val / total * 100 if total else 0
            ax.text(x_idx, val + 0.2, f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    sns.despine(ax=ax, left=False, bottom=False)
    if horizontal:
        fig.tight_layout(rect=(0.25, 0, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_no_data(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_created_date_context(df: pd.DataFrame, target_date: pd.Timestamp, path: Path) -> None:
    created_dates = df["created_at_dt"].dt.date.value_counts().sort_index()
    x_labels = [str(x) for x in created_dates.index]
    y_values = created_dates.values

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4c78a8" if str(d) != str(target_date) else "#f58518" for d in created_dates.index]
    bars = ax.bar(x_labels, y_values, color=colors)

    for bar, val in zip(bars, y_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(val), ha="center", fontsize=10)

    ax.set_title("Profile Creation Dates (UTC)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Profiles")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_experience_distribution(series: pd.Series, path: Path) -> None:
    values = series.dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(0, max(26, int(values.max()) + 2), 2)
    sns.histplot(values, bins=bins, kde=True, color="#4c78a8", ax=ax)

    if not values.empty:
        mean_v = values.mean()
        med_v = values.median()
        ax.axvline(mean_v, color="#e45756", linestyle="--", linewidth=2, label=f"Mean: {mean_v:.1f}")
        ax.axvline(med_v, color="#72b7b2", linestyle="--", linewidth=2, label=f"Median: {med_v:.1f}")
        ax.legend()

    ax.set_title("Years of Experience Distribution")
    ax.set_xlabel("Years")
    ax.set_ylabel("Profiles")
    ax.grid(alpha=0.25)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_role_seniority_heatmap(df: pd.DataFrame, path: Path) -> None:
    table = pd.crosstab(df["role_family"], df["seniority"]) \
        .reindex(columns=["Junior", "Middle", "Senior", "Lead", "Executive", "Unknown"], fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(14, max(6, 0.5 * len(table) + 2)))
    sns.heatmap(table, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Profiles"}, ax=ax)
    ax.set_title("Role Family x Seniority")
    ax.set_xlabel("Seniority")
    ax.set_ylabel("Role Family")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, rows: int = 10) -> str:
    if df.empty:
        return "(no data)"
    return df.head(rows).to_string(index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df["created_at_dt"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["updated_at_dt"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
    employer_fields = find_employer_fields(df.columns)

    cohort = df[df["created_at_dt"].dt.date == TARGET_DATE].copy()

    cohort["full_name"] = cohort["first_name"].fillna("").str.strip() + " " + cohort["last_name"].fillna("").str.strip()
    cohort["years_experience"] = cohort["summary"].map(extract_years_experience)
    cohort["role_family"] = cohort.apply(
        lambda row: infer_role_family(row.get("summary", ""), row.get("specialist_category", "")),
        axis=1,
    )
    cohort["seniority"] = cohort.apply(
        lambda row: infer_seniority(row.get("summary", ""), row.get("years_experience")),
        axis=1,
    )
    cohort["region"] = cohort["summary"].map(infer_region)
    cohort["employer_type"] = cohort["summary"].map(infer_employer_type)
    if employer_fields:
        cohort["employer_name"] = cohort.apply(lambda row: extract_employer_from_row(row, employer_fields), axis=1)
        cohort["employer_source_field"] = cohort.apply(
            lambda row: detect_employer_source_field(row, employer_fields),
            axis=1,
        )
    else:
        cohort["employer_name"] = ""
        cohort["employer_source_field"] = ""
    cohort["has_explicit_employer"] = cohort["employer_name"].str.len().gt(0)

    all_skills = []
    for raw in cohort["skills"]:
        all_skills.extend(parse_skills(raw))

    skills_series = pd.Series(all_skills)
    skill_counts = skills_series.value_counts()

    plot_created_date_context(df, TARGET_DATE, FIG_DIR / "01_created_date_context.png")
    save_countplot(
        cohort["specialist_category"].fillna("Not specified"),
        "Specialist Category Distribution",
        "Profiles",
        "Category",
        FIG_DIR / "02_specialist_category.png",
    )
    save_countplot(
        cohort["role_family"].fillna("Other"),
        "Inferred Role Family Distribution",
        "Profiles",
        "Role Family",
        FIG_DIR / "03_role_family.png",
    )
    save_countplot(
        cohort["seniority"].fillna("Unknown"),
        "Inferred Seniority Distribution",
        "Seniority",
        "Profiles",
        FIG_DIR / "04_seniority.png",
        horizontal=False,
    )
    save_countplot(
        cohort["region"].fillna("Not specified"),
        "Inferred Region Distribution",
        "Profiles",
        "Region",
        FIG_DIR / "05_region.png",
    )
    save_countplot(
        cohort["employer_type"].fillna("Not specified"),
        "Inferred Employer Type Distribution",
        "Profiles",
        "Employer Type",
        FIG_DIR / "06_employer_type.png",
    )

    top_employers = cohort.loc[cohort["has_explicit_employer"], "employer_name"].value_counts()
    if not top_employers.empty:
        save_countplot(
            top_employers.rename_axis("employer").reset_index(name="count").set_index("employer")["count"],
            "Top Employers (Dedicated Employer Field)",
            "Mentions",
            "Employer",
            FIG_DIR / "07_employer_mentions.png",
            top_n=15,
            pre_counted=True,
        )
    else:
        if employer_fields:
            plot_no_data(
                FIG_DIR / "07_employer_mentions.png",
                "Top Employers (Dedicated Employer Field)",
                "Employer fields exist, but no non-empty values in this cohort.",
            )
        else:
            plot_no_data(
                FIG_DIR / "07_employer_mentions.png",
                "Top Employers (Dedicated Employer Field)",
                "No dedicated employer column found in source CSV.",
            )

    plot_experience_distribution(cohort["years_experience"], FIG_DIR / "08_experience_distribution.png")
    plot_role_seniority_heatmap(cohort, FIG_DIR / "09_role_seniority_heatmap.png")

    if not skill_counts.empty:
        save_countplot(
            skill_counts,
            "Top Skill Keywords",
            "Mentions",
            "Skill",
            FIG_DIR / "10_top_skills.png",
            top_n=25,
            pre_counted=True,
        )

    enriched_path = OUT_DIR / "profiles_2026_02_26_enriched.csv"
    cohort.to_csv(enriched_path, index=False)

    quality = cohort.isna().mean().mul(100).round(1).sort_values(ascending=False)

    role_table = cohort["role_family"].value_counts().rename("count").to_frame()
    role_table["share_%"] = (role_table["count"] / len(cohort) * 100).round(1)

    seniority_table = cohort["seniority"].value_counts().rename("count").to_frame()
    seniority_table["share_%"] = (seniority_table["count"] / len(cohort) * 100).round(1)

    region_table = cohort["region"].value_counts().rename("count").to_frame()
    region_table["share_%"] = (region_table["count"] / len(cohort) * 100).round(1)

    employer_type_table = cohort["employer_type"].value_counts().rename("count").to_frame()
    employer_type_table["share_%"] = (employer_type_table["count"] / len(cohort) * 100).round(1)

    experience_non_null = cohort["years_experience"].dropna()
    experience_coverage = len(experience_non_null) / len(cohort) * 100 if len(cohort) else 0.0
    employer_field_info = ", ".join(employer_fields) if employer_fields else "(none)"

    report_path = OUT_DIR / "EDA_report_2026_02_26.md"
    report = f"""# EDA: Profiles Created on 2026-02-26 (UTC)

## Scope
- Source file: `{DATA_PATH.name}`
- Total profiles in source: **{len(df)}**
- Profiles on target date `2026-02-26`: **{len(cohort)}** ({len(cohort)/len(df)*100:.1f}% of source)

## Data Quality (Filtered Cohort)
```
{quality.to_string()}
```

## Methodology (Inferred Fields)
- `role_family`: keyword-based mapping from `summary` first sentence + full text; fallback to `specialist_category`.
- `seniority`: title keyword rules (Executive/Lead/Senior/Middle/Junior) + years-of-experience fallback.
- `region`: geo keyword matching; if no explicit region and summary is Cyrillic, marked as `Russia & CIS (inferred)`.
- `employer_name`: extracted strictly from dedicated employer columns only.
- Dedicated employer column(s) found in source: **{employer_field_info}**
- `employer_type`: keyword mapping by industry context in summary (separate from employer_name).

## Key Metrics
- Profiles with parsed years-of-experience: **{experience_coverage:.1f}%**
- Mean years-of-experience: **{experience_non_null.mean():.1f}**
- Median years-of-experience: **{experience_non_null.median():.1f}**
- Profiles with non-empty dedicated employer field: **{cohort['has_explicit_employer'].mean()*100:.1f}%**

### Role Family
```
{role_table.to_string()}
```

### Seniority
```
{seniority_table.to_string()}
```

### Region
```
{region_table.to_string()}
```

### Employer Type
```
{employer_type_table.to_string()}
```

### Top Employers (Dedicated Employer Field)
```
{top_employers.to_string() if not top_employers.empty else '(none)'}
```

### Top Skills (Keyword-level)
```
{skill_counts.head(30).to_string() if not skill_counts.empty else '(none)'}
```

## Visualizations
1. `figures/01_created_date_context.png`
2. `figures/02_specialist_category.png`
3. `figures/03_role_family.png`
4. `figures/04_seniority.png`
5. `figures/05_region.png`
6. `figures/06_employer_type.png`
7. `figures/07_employer_mentions.png`
8. `figures/08_experience_distribution.png`
9. `figures/09_role_seniority_heatmap.png`
10. `figures/10_top_skills.png`
"""
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved enriched dataset: {enriched_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
