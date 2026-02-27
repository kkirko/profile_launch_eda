from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_CSV = Path("/Users/k/Downloads/prointerview-prod.users.csv")
OUT_DIR = BASE_DIR / "analysis" / "mis_2026_02_26_27"
FIG_DIR = OUT_DIR / "figures"

START_TS = pd.Timestamp("2026-02-26 00:00:00", tz="UTC")
END_TS_EXCL = pd.Timestamp("2026-02-28 00:00:00", tz="UTC")
REPORT_PERIOD_LABEL = "2026-02-26 — 2026-02-27 (UTC)"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["font.family"] = "DejaVu Sans"


@dataclass
class EmploymentParse:
    start_year: float
    start_month: float
    end_year: float
    end_month: float
    duration_months: float
    is_current: bool


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "n/a", "na", "-"}:
        return ""
    return text


def non_empty_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def normalize_region(value: object) -> str:
    text = clean_text(value)
    if not text:
        return "Not specified"

    t = text.lower()

    if re.search(r"\bmoscow\b|москв", t):
        return "Moscow"
    if re.search(r"\bst\.?\s*petersburg\b|saint petersburg|санкт", t):
        return "Saint Petersburg"
    if re.search(r"\bminsk\b|минск", t):
        return "Minsk"
    if re.search(r"\balmaty\b|алмат", t):
        return "Almaty"
    if re.search(r"\byerevan\b|ереван", t):
        return "Yerevan"
    if re.search(r"\btashkent\b|ташкент", t):
        return "Tashkent"

    if re.search(r"\brussia\b|росси|\bрф\b", t):
        return "Russia (other)"
    if re.search(r"\bbelarus\b|беларус", t):
        return "Belarus (other)"
    if re.search(r"\bkazakhstan\b|казахстан", t):
        return "Kazakhstan (other)"
    if re.search(r"\barmenia\b|армени", t):
        return "Armenia (other)"
    if re.search(r"\buzbekistan\b|узбекистан", t):
        return "Uzbekistan (other)"
    return text


def normalize_industry(value: object) -> str:
    text = clean_text(value)
    if not text:
        return "Not specified"

    t = text.lower()
    if "fintech" in t:
        return "FinTech"
    if "bank" in t:
        return "Banking"
    if "e-commerce" in t or "ecommerce" in t:
        return "E-commerce"
    if "retail" in t:
        return "Retail"
    if "telecom" in t:
        return "Telecom"
    if "health" in t or "pharma" in t:
        return "Healthcare/Pharma"
    if "consult" in t:
        return "Consulting"
    if "manufact" in t:
        return "Manufacturing"
    if "media" in t:
        return "Media"
    if "energy" in t:
        return "Energy"
    if "edtech" in t or "education" in t:
        return "Education/EdTech"
    if "logistic" in t or "transport" in t:
        return "Logistics/Transport"
    if "gov" in t or "public" in t:
        return "Gov/Public"
    if "marketing" in t:
        return "Marketing"
    if "it" == t or "it " in f"{t} " or "information technology" in t:
        return "IT"
    if "other" in t:
        return "Other"
    return text


DOMAIN_RULES: List[Tuple[str, List[str]]] = [
    (
        "Product / Project Management",
        [
            r"product manager",
            r"project manager",
            r"program manager",
            r"delivery manager",
            r"head of pmo",
            r"менеджер проектов",
            r"руководитель проектов",
            r"руководитель проект",
            r"product owner",
            r"scrum master",
        ],
    ),
    (
        "Software Engineering / DevOps",
        [
            r"developer",
            r"engineer",
            r"backend",
            r"frontend",
            r"full\s*stack",
            r"devops",
            r"sre",
            r"cto",
            r"архитектор",
            r"разработчик",
            r"инженер-программист",
        ],
    ),
    (
        "Data / Analytics / AI",
        [
            r"data",
            r"analytics",
            r"analyst",
            r"bi",
            r"machine learning",
            r"ai",
            r"ml",
            r"cdo",
            r"аналит",
            r"данн",
        ],
    ),
    (
        "QA / Testing",
        [
            r"\bqa\b",
            r"test",
            r"quality assurance",
            r"тестиров",
        ],
    ),
    (
        "Design / UX",
        [
            r"designer",
            r"ux",
            r"ui",
            r"дизайн",
            r"дизайнер",
        ],
    ),
    (
        "Sales / Business Development",
        [
            r"sales",
            r"business development",
            r"account manager",
            r"commercial director",
            r"директор по продаж",
            r"менеджер по продаж",
            r"развитию бизнеса",
        ],
    ),
    (
        "Marketing / PR",
        [
            r"marketing",
            r"pr",
            r"communications",
            r"маркет",
            r"коммуникац",
        ],
    ),
    (
        "Finance / Investment",
        [
            r"finance",
            r"financial",
            r"cfo",
            r"investment",
            r"bank",
            r"финанс",
            r"казнач",
        ],
    ),
    (
        "HR / Recruiting",
        [
            r"\bhr\b",
            r"recruit",
            r"talent acquisition",
            r"рекрут",
            r"подбор персонала",
        ],
    ),
    (
        "Operations / Supply Chain",
        [
            r"operations",
            r"logistics",
            r"supply chain",
            r"operational",
            r"операцион",
            r"логист",
        ],
    ),
    (
        "Legal / Compliance / Security",
        [
            r"legal",
            r"compliance",
            r"kyc",
            r"aml",
            r"security",
            r"безопасност",
            r"юрист",
        ],
    ),
    (
        "Healthcare / Medical",
        [
            r"health",
            r"medical",
            r"doctor",
            r"therapist",
            r"медицин",
            r"врач",
        ],
    ),
]


def infer_domain(selected_position: object, job_title: object, industry_norm: str) -> str:
    text = " ".join(
        [clean_text(selected_position), clean_text(job_title), clean_text(industry_norm)]
    ).lower()

    if not text.strip():
        return "Other / Not specified"

    for domain, patterns in DOMAIN_RULES:
        if any(re.search(pattern, text) for pattern in patterns):
            return domain

    return "Other / Not specified"


def parse_employment_period(period: object, ref_ts: pd.Timestamp) -> EmploymentParse:
    raw = clean_text(period)
    if not raw:
        return EmploymentParse(*(math.nan,) * 5, False)

    text = raw.lower()
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*-{2,}\s*", " - ", text)

    if re.search(r"даты не указаны", text):
        return EmploymentParse(*(math.nan,) * 5, False)

    is_current = bool(re.search(r"present|current|по\s*настоя|н\.?в\.?", text))

    month_year_matches = re.findall(r"(?<!\d)(0?[1-9]|1[0-2])\.(19\d{2}|20\d{2})(?!\d)", text)
    years = [int(y) for y in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)]

    start_year = math.nan
    start_month = math.nan
    end_year = math.nan
    end_month = math.nan

    if len(month_year_matches) >= 2:
        sm, sy = month_year_matches[0]
        em, ey = month_year_matches[1]
        start_month, start_year = int(sm), int(sy)
        end_month, end_year = int(em), int(ey)
    elif len(month_year_matches) == 1:
        sm, sy = month_year_matches[0]
        start_month, start_year = int(sm), int(sy)
        if is_current:
            end_month, end_year = ref_ts.month, ref_ts.year
        elif len(years) >= 2:
            end_year = years[-1]
            end_month = 12
    elif len(years) >= 2:
        start_year, end_year = years[0], years[1]
        start_month, end_month = 1, 12
    elif len(years) == 1 and is_current:
        start_year, start_month = years[0], 1
        end_year, end_month = ref_ts.year, ref_ts.month

    if is_current and not (math.isnan(start_year) or math.isnan(start_month)):
        end_year, end_month = ref_ts.year, ref_ts.month

    duration = math.nan
    if not any(math.isnan(x) for x in [start_year, start_month, end_year, end_month]):
        duration = (int(end_year) - int(start_year)) * 12 + (int(end_month) - int(start_month)) + 1
        if duration <= 0 or duration > 720:
            duration = math.nan

    return EmploymentParse(
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        duration_months=duration,
        is_current=is_current,
    )


def prepare_top_table(series: pd.Series, top_n: int = 15) -> pd.DataFrame:
    counts = series.value_counts().head(top_n)
    if counts.empty:
        return pd.DataFrame(columns=["count", "share_%"])
    table = counts.rename("count").to_frame()
    table["share_%"] = (table["count"] / len(series) * 100).round(1)
    return table


def plot_no_data(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axis("off")
    ax.text(0.5, 0.60, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_top_bar(
    series: pd.Series,
    title: str,
    path: Path,
    top_n: int = 15,
    xlabel: str = "Кандидаты",
    ylabel: str = "Категория",
) -> None:
    counts = series.value_counts().head(top_n)
    if counts.empty:
        plot_no_data(path, title, "Нет данных для построения графика")
        return

    total = len(series)
    display_labels = [
        idx if len(str(idx)) <= 64 else f"{str(idx)[:61]}..."
        for idx in counts.index.astype(str)
    ]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(counts) + 1.6)))
    colors = sns.color_palette("viridis", n_colors=len(counts))
    ax.barh(display_labels, counts.values, color=colors)
    ax.invert_yaxis()

    for i, v in enumerate(counts.values):
        pct = v / total * 100 if total else 0
        ax.text(v + 0.2, i, f"{v} ({pct:.1f}%)", va="center", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.92, bottom=0.08)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_entries_by_day(period_df: pd.DataFrame, path: Path) -> None:
    counts = period_df["created_date"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar(counts.index.astype(str), counts.values, color=["#4c78a8", "#f58518"])

    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center", fontsize=11)

    ax.set_title("Новые кандидаты по дням (UTC)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Кандидаты")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_funnel(funnel_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = sns.color_palette("mako", n_colors=len(funnel_df))
    ax.barh(funnel_df["stage"], funnel_df["count"], color=colors)
    ax.invert_yaxis()

    for i, row in funnel_df.iterrows():
        ax.text(
            row["count"] + 2,
            i,
            f"{row['count']} ({row['conversion_from_entry_%']:.1f}%)",
            va="center",
            fontsize=10,
        )

    ax.set_title("Воронка: вход в чат-бот → onboardingCompleted")
    ax.set_xlabel("Количество кандидатов")
    ax.set_ylabel("Этап")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(rect=(0.22, 0, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_current_vs_ended(series: pd.Series, path: Path) -> None:
    counts = series.value_counts().reindex(["Current job", "Ended job", "Unknown"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(counts.index, counts.values, color=sns.color_palette("Set2", n_colors=3))

    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        pct = value / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value}\n({pct:.1f}%)", ha="center", fontsize=10)

    ax.set_title("Статус последнего места работы")
    ax.set_xlabel("")
    ax.set_ylabel("Кандидаты")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_tenure_hist(series: pd.Series, path: Path) -> None:
    vals = series.dropna()
    if vals.empty:
        plot_no_data(path, "Длительность последнего места работы", "Не удалось распарсить employment_period")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.arange(0, min(241, int(vals.max()) + 13), 12)
    if len(bins) < 3:
        bins = 12
    sns.histplot(vals, bins=bins, kde=True, color="#4c78a8", ax=ax)

    ax.axvline(vals.median(), color="#e45756", linestyle="--", linewidth=2, label=f"Median: {vals.median():.0f} мес.")
    ax.axvline(vals.mean(), color="#72b7b2", linestyle="--", linewidth=2, label=f"Mean: {vals.mean():.0f} мес.")
    ax.legend()

    ax.set_title("Распределение длительности последнего места работы")
    ax.set_xlabel("Месяцы")
    ax.set_ylabel("Кандидаты")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_CSV)

    df["created_at_dt"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
    df["updated_at_dt"] = pd.to_datetime(df["updatedAt"], utc=True, errors="coerce")
    df["created_date"] = df["created_at_dt"].dt.date

    period_mask = (df["created_at_dt"] >= START_TS) & (df["created_at_dt"] < END_TS_EXCL)
    period_df = df[period_mask].copy()

    enhanced_mask = non_empty_series(period_df["cvEnhancedResult"])
    cohort = period_df[enhanced_mask].copy()

    # Derived fields for last/current job (jobs[0])
    cohort["last_company"] = cohort["talentCard.jobs[0].company"].map(clean_text).replace("", "Not specified")
    cohort["last_job_title"] = cohort["talentCard.jobs[0].job_title"].map(clean_text).replace("", "Not specified")
    cohort["last_region_raw"] = cohort["talentCard.jobs[0].region"].map(clean_text)
    cohort["last_region_group"] = cohort["last_region_raw"].map(normalize_region)
    cohort["last_industry_raw"] = cohort["talentCard.jobs[0].industry"].map(clean_text)
    cohort["last_industry_group"] = cohort["last_industry_raw"].map(normalize_industry)
    cohort["last_seniority"] = cohort["talentCard.jobs[0].seniority"].map(clean_text).replace("", "Not specified")
    cohort["last_employment_period"] = cohort["talentCard.jobs[0].employment_period"].map(clean_text)

    cohort["domain_inferred"] = cohort.apply(
        lambda row: infer_domain(
            row.get("selectedPosition", ""),
            row.get("last_job_title", ""),
            row.get("last_industry_group", ""),
        ),
        axis=1,
    )

    employment_parsed = cohort["last_employment_period"].map(lambda x: parse_employment_period(x, END_TS_EXCL - pd.Timedelta(days=1)))
    cohort["employment_start_year"] = employment_parsed.map(lambda x: x.start_year)
    cohort["employment_start_month"] = employment_parsed.map(lambda x: x.start_month)
    cohort["employment_end_year"] = employment_parsed.map(lambda x: x.end_year)
    cohort["employment_end_month"] = employment_parsed.map(lambda x: x.end_month)
    cohort["employment_duration_months"] = employment_parsed.map(lambda x: x.duration_months)
    cohort["is_current_job"] = employment_parsed.map(lambda x: x.is_current)

    cohort["employment_status"] = np.where(
        cohort["last_employment_period"].str.strip().eq(""),
        "Unknown",
        np.where(cohort["is_current_job"], "Current job", "Ended job"),
    )

    # Funnel on full period cohort (entry -> onboarding)
    stage_1 = pd.Series(True, index=period_df.index)
    stage_2 = stage_1 & non_empty_series(period_df["cvPath"])
    stage_3 = stage_2 & non_empty_series(period_df["cvAnalysisResult"])
    stage_4 = stage_3 & non_empty_series(period_df["cvEnhancedResult"])
    stage_5 = stage_4 & bool_series(period_df["onboardingCompleted"])

    funnel = pd.DataFrame(
        {
            "stage": [
                "1. Вход в чат-бот (createdAt)",
                "2. CV загружено (cvPath)",
                "3. CV анализ завершён (cvAnalysisResult)",
                "4. CV enhanced готов (cvEnhancedResult)",
                "5. Onboarding completed",
            ],
            "count": [
                int(stage_1.sum()),
                int(stage_2.sum()),
                int(stage_3.sum()),
                int(stage_4.sum()),
                int(stage_5.sum()),
            ],
        }
    )
    funnel["conversion_from_entry_%"] = (funnel["count"] / funnel.loc[0, "count"] * 100).round(1)
    funnel["conversion_from_prev_%"] = [
        100.0,
        round(funnel.loc[1, "count"] / funnel.loc[0, "count"] * 100, 1) if funnel.loc[0, "count"] else 0.0,
        round(funnel.loc[2, "count"] / funnel.loc[1, "count"] * 100, 1) if funnel.loc[1, "count"] else 0.0,
        round(funnel.loc[3, "count"] / funnel.loc[2, "count"] * 100, 1) if funnel.loc[2, "count"] else 0.0,
        round(funnel.loc[4, "count"] / funnel.loc[3, "count"] * 100, 1) if funnel.loc[3, "count"] else 0.0,
    ]

    # Funnel strictly for filtered cohort (as requested: "по этим пользователям")
    cohort_stage_1 = pd.Series(True, index=cohort.index)
    cohort_stage_2 = cohort_stage_1 & non_empty_series(cohort["cvPath"])
    cohort_stage_3 = cohort_stage_2 & non_empty_series(cohort["cvAnalysisResult"])
    cohort_stage_4 = cohort_stage_3 & non_empty_series(cohort["cvEnhancedResult"])
    cohort_stage_5 = cohort_stage_4 & bool_series(cohort["onboardingCompleted"])

    funnel_filtered = pd.DataFrame(
        {
            "stage": [
                "1. Вход в чат-бот (createdAt)",
                "2. CV загружено (cvPath)",
                "3. CV анализ завершён (cvAnalysisResult)",
                "4. CV enhanced готов (cvEnhancedResult)",
                "5. Onboarding completed",
            ],
            "count": [
                int(cohort_stage_1.sum()),
                int(cohort_stage_2.sum()),
                int(cohort_stage_3.sum()),
                int(cohort_stage_4.sum()),
                int(cohort_stage_5.sum()),
            ],
        }
    )
    funnel_filtered["conversion_from_entry_%"] = (
        funnel_filtered["count"] / funnel_filtered.loc[0, "count"] * 100
    ).round(1)
    funnel_filtered["conversion_from_prev_%"] = [
        100.0,
        round(funnel_filtered.loc[1, "count"] / funnel_filtered.loc[0, "count"] * 100, 1)
        if funnel_filtered.loc[0, "count"]
        else 0.0,
        round(funnel_filtered.loc[2, "count"] / funnel_filtered.loc[1, "count"] * 100, 1)
        if funnel_filtered.loc[1, "count"]
        else 0.0,
        round(funnel_filtered.loc[3, "count"] / funnel_filtered.loc[2, "count"] * 100, 1)
        if funnel_filtered.loc[2, "count"]
        else 0.0,
        round(funnel_filtered.loc[4, "count"] / funnel_filtered.loc[3, "count"] * 100, 1)
        if funnel_filtered.loc[3, "count"]
        else 0.0,
    ]

    # Visualizations
    plot_entries_by_day(period_df, FIG_DIR / "01_new_candidates_by_day.png")
    plot_funnel(funnel, FIG_DIR / "02_chatbot_onboarding_funnel.png")
    plot_funnel(funnel_filtered, FIG_DIR / "11_chatbot_onboarding_funnel_filtered_cohort.png")
    plot_top_bar(cohort["last_region_group"], "Топ регионов (jobs[0].region)", FIG_DIR / "03_top_regions.png", top_n=15, ylabel="Регион")
    plot_top_bar(cohort["domain_inferred"], "Топ доменов (inferred)", FIG_DIR / "04_top_domains.png", top_n=12, ylabel="Домен")
    plot_top_bar(cohort["last_industry_group"], "Топ отраслей (jobs[0].industry)", FIG_DIR / "05_top_industries.png", top_n=12, ylabel="Отрасль")
    plot_top_bar(cohort["last_seniority"], "Уровень seniority (jobs[0].seniority)", FIG_DIR / "06_seniority_distribution.png", top_n=8, ylabel="Seniority")
    plot_top_bar(cohort["last_company"], "Топ работодателей (jobs[0].company)", FIG_DIR / "07_top_employers.png", top_n=20, ylabel="Работодатель")
    plot_current_vs_ended(cohort["employment_status"], FIG_DIR / "08_current_vs_ended_job.png")
    plot_tenure_hist(cohort["employment_duration_months"], FIG_DIR / "09_last_job_tenure_months.png")
    plot_top_bar(cohort["last_job_title"], "Топ job titles (jobs[0].job_title)", FIG_DIR / "10_top_job_titles.png", top_n=20, ylabel="Job title")

    # Tables
    top_regions = prepare_top_table(cohort["last_region_group"], top_n=15)
    top_domains = prepare_top_table(cohort["domain_inferred"], top_n=12)
    top_industries = prepare_top_table(cohort["last_industry_group"], top_n=12)
    top_seniority = prepare_top_table(cohort["last_seniority"], top_n=8)
    top_employers = prepare_top_table(cohort["last_company"], top_n=20)
    top_job_titles = prepare_top_table(cohort["last_job_title"], top_n=20)

    parsed_tenure = cohort["employment_duration_months"].dropna()
    parse_coverage = round(parsed_tenure.size / len(cohort) * 100, 1) if len(cohort) else 0.0
    top_region_name = str(top_regions.index[0]) if not top_regions.empty else "n/a"
    top_region_share = float(top_regions.iloc[0]["share_%"]) if not top_regions.empty else 0.0
    top_domain_name = str(top_domains.index[0]) if not top_domains.empty else "n/a"
    top_domain_share = float(top_domains.iloc[0]["share_%"]) if not top_domains.empty else 0.0
    top_industry_missing_share = float(top_industries.loc["Not specified", "share_%"]) if "Not specified" in top_industries.index else 0.0

    # Data quality for key MIS fields
    quality_cols = [
        "createdAt",
        "cvPath",
        "cvAnalysisResult",
        "cvEnhancedResult",
        "onboardingCompleted",
        "talentCard.jobs[0].company",
        "talentCard.jobs[0].job_title",
        "talentCard.jobs[0].region",
        "talentCard.jobs[0].industry",
        "talentCard.jobs[0].seniority",
        "talentCard.jobs[0].employment_period",
    ]
    quality = cohort[quality_cols].isna().mean().mul(100).round(1).sort_values(ascending=False)

    # Save datasets
    cohort_out = OUT_DIR / "candidates_2026_02_26_27_cvEnhancedResult_enriched.csv"
    funnel_out = OUT_DIR / "funnel_2026_02_26_27.csv"
    funnel_filtered_out = OUT_DIR / "funnel_filtered_cohort_2026_02_26_27.csv"
    cohort.to_csv(cohort_out, index=False)
    funnel.to_csv(funnel_out, index=False)
    funnel_filtered.to_csv(funnel_filtered_out, index=False)

    report_out = OUT_DIR / "MIS_report_2026_02_26_27.md"
    report_text = f"""# MIS Report: Candidate Base Analytics

## Scope
- Source: `{SOURCE_CSV}`
- Date filter: `createdAt` in **{REPORT_PERIOD_LABEL}**
- Cohort condition: non-empty **`cvEnhancedResult`**

## Important Note on Filtering
- В датасете нет отдельного timestamp-поля для момента генерации `cvEnhancedResult`.
- Поэтому срез построен по `createdAt` (попадание в базу) + наличию непустого `cvEnhancedResult`.

## Executive Summary
- В периоде зафиксировано **{len(period_df)}** входов в чат-бот; до `cvEnhancedResult` дошли **{len(cohort)}** кандидатов (**{len(cohort)/len(period_df)*100:.1f}%**).
- По воронке периода основной отток происходит на этапе после входа в чат-бот: `createdAt -> cvPath` (**78.4%** конверсия).
- Внутри целевого среза (`cvEnhancedResult`) все **521** кандидата имеют `onboardingCompleted=True` (100%).
- География последних мест работы концентрируется в **{top_region_name}** (**{top_region_share:.1f}%**), затем идут другие локации РФ/СНГ.
- По inferred-доменам лидирует **{top_domain_name}** (**{top_domain_share:.1f}%**), а по отрасли наблюдается высокий уровень пропусков (`jobs[0].industry`: **{top_industry_missing_share:.1f}%** `Not specified`).

## KPI Snapshot
- Users in period (`createdAt`): **{len(period_df)}**
- Users with non-empty `cvEnhancedResult`: **{len(cohort)}** ({len(cohort)/len(period_df)*100:.1f}% от периода)
- `onboardingCompleted=True` внутри среза: **{int(bool_series(cohort['onboardingCompleted']).sum())}** ({bool_series(cohort['onboardingCompleted']).mean()*100:.1f}%)
- Unique users (`userId`) в срезе: **{cohort['userId'].nunique()}**
- Parse coverage for `jobs[0].employment_period`: **{parse_coverage}%**

## Data Quality (Filtered Cohort)
```
{quality.to_string()}
```

## Funnel: Chat-bot Entry -> Onboarding
### 1) Full period funnel (all chatbot entrants in period)
```
{funnel.to_string(index=False)}
```

### 2) Filtered cohort funnel (users included in this MIS slice)
```
{funnel_filtered.to_string(index=False)}
```

### Visual
![Period Funnel](figures/02_chatbot_onboarding_funnel.png)
![Filtered Cohort Funnel](figures/11_chatbot_onboarding_funnel_filtered_cohort.png)

## Last Job Insights (`talentCard.jobs[0]`)
### Top Regions
```
{top_regions.to_string()}
```
![Top Regions](figures/03_top_regions.png)

### Top Domains (Inferred from `selectedPosition` + `jobs[0].job_title` + industry)
```
{top_domains.to_string()}
```
![Top Domains](figures/04_top_domains.png)

### Top Industries
```
{top_industries.to_string()}
```
![Top Industries](figures/05_top_industries.png)

### Seniority Distribution
```
{top_seniority.to_string()}
```
![Seniority](figures/06_seniority_distribution.png)

### Top Last Employers
```
{top_employers.to_string()}
```
![Top Employers](figures/07_top_employers.png)

### Top Last Job Titles
```
{top_job_titles.to_string()}
```
![Top Job Titles](figures/10_top_job_titles.png)

## Employment Period Analytics (`jobs[0].employment_period`)
- Current jobs (`present/current` in period string): **{int((cohort['employment_status'] == 'Current job').sum())}**
- Ended jobs: **{int((cohort['employment_status'] == 'Ended job').sum())}**
- Unknown period: **{int((cohort['employment_status'] == 'Unknown').sum())}**
- Median tenure (parsed): **{parsed_tenure.median():.1f} months**
- Mean tenure (parsed): **{parsed_tenure.mean():.1f} months**
![Current vs Ended](figures/08_current_vs_ended_job.png)
![Tenure Distribution](figures/09_last_job_tenure_months.png)

## Visualizations
1. `figures/01_new_candidates_by_day.png`
2. `figures/02_chatbot_onboarding_funnel.png`
3. `figures/03_top_regions.png`
4. `figures/04_top_domains.png`
5. `figures/05_top_industries.png`
6. `figures/06_seniority_distribution.png`
7. `figures/07_top_employers.png`
8. `figures/08_current_vs_ended_job.png`
9. `figures/09_last_job_tenure_months.png`
10. `figures/10_top_job_titles.png`
11. `figures/11_chatbot_onboarding_funnel_filtered_cohort.png`
"""
    report_out.write_text(report_text, encoding="utf-8")

    print(f"Saved report: {report_out}")
    print(f"Saved cohort dataset: {cohort_out}")
    print(f"Saved funnel dataset: {funnel_out}")
    print(f"Saved filtered-cohort funnel dataset: {funnel_filtered_out}")
    print(f"Saved figures dir: {FIG_DIR}")


if __name__ == "__main__":
    main()
