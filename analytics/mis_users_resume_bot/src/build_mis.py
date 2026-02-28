from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feature_builders import (
    aggregate_tokens,
    build_jobs_long_talent,
    build_mapping_table,
    canonical_text,
    experience_bin,
    guess_country,
    hash_user_id,
    infer_role_family,
    leadership_level,
    merge_skills_tools,
    normalize_company,
    normalize_empty_strings,
    normalize_industry,
    normalize_region,
    normalize_seniority,
    parse_bool_series,
    summarize_user_jobs,
)
from latex_parser import clean_text, parse_cv_latex


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 220

DATE_COLUMNS = [
    "createdAt",
    "updatedAt",
    "bannedAt",
    "cvGenerationStartedAt",
    "cvAnalysisStartedAt",
]


def pct(n: int, d: int) -> float:
    return round((n / d * 100), 1) if d else 0.0


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def shorten(value: object, max_len: int = 44) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def non_empty(value: object) -> bool:
    return clean_text(value) != ""


def limit_categories(series: pd.Series, max_n: int, other_label: str = "Other") -> pd.Series:
    values = series.fillna("Not specified").astype(str)
    top = values.value_counts().head(max_n - 1 if values.nunique() > max_n else max_n).index
    if values.nunique() <= max_n:
        return values
    return values.map(lambda x: x if x in top else other_label)


def row_normalize_percent(df: pd.DataFrame) -> pd.DataFrame:
    share = df.div(df.sum(axis=1).replace(0, np.nan), axis=0) * 100
    return share.fillna(0)


def build_columns_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for col in df.columns:
        series = df[col]
        non_null = series.notna()
        rate = round(float(non_null.mean()), 4)

        if non_null.any():
            lengths = series[non_null].astype(str).str.len()
            stats = f"p50={int(lengths.quantile(0.5))};p95={int(lengths.quantile(0.95))};max={int(lengths.max())}"
        else:
            stats = ""

        rows.append(
            {
                "col": col,
                "dtype": str(series.dtype),
                "non_null_rate": rate,
                "example_len_stats": stats,
            }
        )
    return pd.DataFrame(rows)


def distribution_with_display(df: pd.DataFrame, raw_col: str, norm_col: str, field_name: str) -> pd.DataFrame:
    work = df[[raw_col, norm_col]].copy()
    work[raw_col] = work[raw_col].fillna("Not specified").astype(str)
    work[norm_col] = work[norm_col].fillna("Not specified").astype(str)

    counts = work.groupby(norm_col).size().rename("count").reset_index()
    counts = counts.sort_values("count", ascending=False)

    display_map = (
        work.groupby([norm_col, raw_col]).size().rename("cnt").reset_index().sort_values([norm_col, "cnt"], ascending=[True, False])
    )
    display_top = display_map.drop_duplicates(subset=[norm_col])[[norm_col, raw_col]]

    out = counts.merge(display_top, on=norm_col, how="left")
    out = out.rename(columns={raw_col: f"{field_name}_display", norm_col: f"{field_name}_norm"})
    return out


def plot_line(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.plot(df[x_col], df[y_col], marker="o", linewidth=2.1, color="#355C7D")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_barh(df: pd.DataFrame, label_col: str, value_col: str, out_path: Path, title: str, top_n: int) -> None:
    if df.empty:
        return
    top = df.head(top_n).copy()
    labels = [shorten(x, 55) for x in top[label_col].astype(str)]
    values = top[value_col].astype(float).values

    fig, ax = plt.subplots(figsize=(12.5, max(5.5, 0.5 * len(top) + 2)))
    colors = sns.color_palette("viridis", n_colors=len(top))
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()

    total = float(top[value_col].sum())
    for i, value in enumerate(values):
        ax.text(value + max(values) * 0.01, i, f"{int(value)} ({pct(int(value), int(total)):.1f}%)", va="center", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel("Candidates")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_share(matrix: pd.DataFrame, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    if matrix.empty:
        return
    fig_h = max(5.8, 0.42 * matrix.shape[0] + 2)
    fig_w = max(10.5, 0.5 * matrix.shape[1] + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    display = matrix.copy()
    display.columns = [shorten(c, 24) for c in display.columns]
    display.index = [shorten(i, 36) for i in display.index]

    sns.heatmap(
        display,
        cmap="YlGnBu",
        annot=False,
        linewidths=0.3,
        cbar_kws={"label": "Share, %"},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_stacked_100(matrix_counts: pd.DataFrame, out_path: Path, title: str, x_label: str) -> pd.DataFrame:
    if matrix_counts.empty:
        return pd.DataFrame()

    share = row_normalize_percent(matrix_counts)
    fig, ax = plt.subplots(figsize=(13.5, 6.8))

    x = np.arange(len(share.index))
    bottom = np.zeros(len(share.index))
    colors = sns.color_palette("tab20", n_colors=share.shape[1])

    for idx, col in enumerate(share.columns):
        vals = share[col].values
        ax.bar(x, vals, bottom=bottom, color=colors[idx], label=shorten(col, 28), width=0.82)
        bottom += vals

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([shorten(i, 28) for i in share.index], rotation=45, ha="right")
    ax.set_ylabel("Share, %")
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return share


def plot_donut(series: pd.Series, out_path: Path, title: str) -> pd.DataFrame:
    data = series.fillna("Not specified").astype(str).value_counts()
    if data.empty:
        return pd.DataFrame(columns=["category", "count", "share_%"])

    if len(data) > 6:
        top = data.head(5)
        other = data.iloc[5:].sum()
        data = pd.concat([top, pd.Series({"Other": other})])

    if len(data) < 3:
        return pd.DataFrame(columns=["category", "count", "share_%"])

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    colors = sns.color_palette("Set2", n_colors=len(data))
    wedges, texts, autotexts = ax.pie(
        data.values,
        labels=[shorten(i, 18) for i in data.index],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        colors=colors,
        pctdistance=0.74,
        labeldistance=1.02,
    )
    for t in texts:
        t.set_fontsize(8)
    for t in autotexts:
        t.set_fontsize(7)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    out = data.rename_axis("category").reset_index(name="count")
    out["share_%"] = (out["count"] / out["count"].sum() * 100).round(1)
    return out


def parse_latex_dataset(users: pd.DataFrame, as_of_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, object]] = []
    jobs_rows: List[Dict[str, object]] = []

    for _, row in users.iterrows():
        parsed = parse_cv_latex(row.get("cvEnhancedResult", np.nan), as_of_date=as_of_date)

        summary_rows.append(
            {
                "user_hash": row["user_hash"],
                "latex_found": parsed.get("latex_found", False),
                "expheader_count": parsed.get("expheader_count", 0),
                "jobs_count_latex": parsed.get("jobs_count_latex", 0),
                "skills_section_found": parsed.get("skills_section_found", False),
                "skills_count_latex": parsed.get("skills_count", 0),
                "tools_count_latex": parsed.get("tools_count", 0),
                "skills_list_latex": parsed.get("skills_list", []),
                "tools_list_latex": parsed.get("tools_list", []),
                "current_company_latex": parsed.get("current_company_latex", ""),
                "current_region_latex": parsed.get("current_region_latex", ""),
                "current_job_title_latex": parsed.get("current_job_title_latex", ""),
                "current_source_latex": parsed.get("current_source_latex", ""),
                "current_company_expheader": parsed.get("current_company_expheader", ""),
                "current_region_expheader": parsed.get("current_region_expheader", ""),
                "current_job_title_expheader": parsed.get("current_job_title_expheader", ""),
                "header_role_latex": parsed.get("header_role_latex", ""),
                "header_location_latex": parsed.get("header_location_latex", ""),
                "english_level": parsed.get("english_level", "Unknown"),
                "languages_count": parsed.get("languages_count", 0),
                "languages_section_found": parsed.get("languages_section_found", False),
                "degree_level": parsed.get("degree_level", "Unknown"),
                "education_text_len": parsed.get("education_text_len", 0),
                "education_section_found": parsed.get("education_section_found", False),
            }
        )

        for job in parsed.get("jobs_latex", []):
            jobs_rows.append(
                {
                    "user_hash": row["user_hash"],
                    "source": job.get("source", "latex_expheader"),
                    "job_index": job.get("job_index", 0),
                    "company": job.get("company", ""),
                    "region": job.get("region", ""),
                    "job_title": job.get("job_title", ""),
                    "employment_period": job.get("employment_period", ""),
                    "start_date": job.get("start_date", pd.NaT),
                    "end_date": job.get("end_date", pd.NaT),
                    "is_present": job.get("is_present", False),
                    "period_parse_ok": job.get("period_parse_ok", False),
                    "period_parse_quality": job.get("period_parse_quality", ""),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(jobs_rows)


def choose_with_source(candidates: List[Tuple[object, str]], default: str = "Not specified") -> Tuple[str, str]:
    for value, source in candidates:
        text = clean_text(value)
        if text:
            return text, source
    return default, "not_specified"


def build_not_specified_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("domain", "domain_filled", "domain_source", {"inferred"}),
        ("region", "region_filled", "region_source", {"talentCard", "latex_header"}),
        ("company", "company_filled", "company_source", {"talentCard"}),
        ("seniority", "seniority_filled", "seniority_source", {"inferred_job_title", "inferred_header_role"}),
        ("industry", "industry_filled", "industry_source", set()),
    ]

    rows: List[Dict[str, object]] = []
    for field, value_col, source_col, fallback_sources in specs:
        source = df[source_col].fillna("not_specified").astype(str)
        value = df[value_col].fillna("Not specified").astype(str)

        missing_count = int(value.eq("Not specified").sum())
        missing_share = round((value.eq("Not specified").mean()) * 100, 1)
        fallback_mask = source.isin(fallback_sources)
        if field == "region":
            fallback_mask = fallback_mask | source.str.startswith("alt_col:")
        fallback_share = round((fallback_mask.mean()) * 100, 1)

        breakdown = (source.value_counts(normalize=True) * 100).round(1)
        breakdown_text = "; ".join([f"{k}:{v:.1f}%" for k, v in breakdown.items()])

        rows.append(
            {
                "field": field,
                "total_missing_count": missing_count,
                "share_missing_%": missing_share,
                "share_filled_by_fallback": fallback_share,
                "source_breakdown": breakdown_text,
            }
        )

    return pd.DataFrame(rows)


def top_counts(series: pd.Series, n: int, col_name: str) -> pd.DataFrame:
    s = series.fillna("").astype(str).str.strip()
    s = s[s.ne("")]
    if s.empty:
        return pd.DataFrame(columns=[col_name, "count", "share_%"])
    out = s.value_counts().head(n).rename_axis(col_name).reset_index(name="count")
    out["share_%"] = (out["count"] / out["count"].sum() * 100).round(1)
    return out


def top_tokens_from_list(df: pd.DataFrame, list_col: str, n: int, token_name: str) -> pd.DataFrame:
    rows: List[str] = []
    for tokens in df[list_col]:
        for token in (tokens or []):
            t = clean_text(token)
            if t:
                rows.append(t)
    if not rows:
        return pd.DataFrame(columns=[token_name, "count", "share_%"])
    series = pd.Series(rows)
    out = series.value_counts().head(n).rename_axis(token_name).reset_index(name="count")
    out["share_%"] = (out["count"] / out["count"].sum() * 100).round(1)
    return out


def discover_alt_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r"(location|city|country|region|geo)", flags=re.IGNORECASE)
    rows: List[Dict[str, object]] = []
    for col in df.columns:
        if not pattern.search(col):
            continue
        if col.startswith("talentCard.jobs["):
            continue
        if col in {"region_filled", "region_norm", "current_region_talentCard", "current_region_latex", "header_location_latex"}:
            continue
        if col == "talentCard":
            continue
        if str(df[col].dtype) not in {"object", "string"}:
            continue
        series = df[col].fillna("").astype(str).str.strip()
        non_empty = int(series.ne("").sum())
        if non_empty == 0:
            continue
        rows.append(
            {
                "column": col,
                "non_empty_count": non_empty,
                "non_empty_rate_%": round(non_empty / len(df) * 100, 1),
                "median_len": int(series[series.ne("")].str.len().median()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["column", "non_empty_count", "non_empty_rate_%", "median_len"])
    return pd.DataFrame(rows).sort_values("non_empty_count", ascending=False)


def build_notebook(path: Path) -> None:
    notebook_json = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MIS users resume bot\n",
                    "Запуск воспроизводимой сборки MIS без вывода PII и без публикации сырого LaTeX.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import sys\n",
                    "sys.path.append(str(Path('analytics/mis_users_resume_bot/src').resolve()))\n",
                    "from build_mis import run\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "run(input_path='/mnt/data/prointerview-prod.users.csv', base_dir='analytics/mis_users_resume_bot')",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook_json, ensure_ascii=False, indent=2), encoding="utf-8")


def build_readme(base_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    profile = tables["dataset_profile"]
    validation = tables["validation_summary"]
    diagnostics = tables["not_specified_diagnostics"]
    geo_audit = tables["geo_mapping_audit"]
    region_ns_breakdown = tables["not_specified_deep_dive_region_not_specified_breakdown"]
    company_ns_breakdown = tables["not_specified_deep_dive_company_not_specified_breakdown"]
    region_alt_cols = tables["not_specified_deep_dive_region_alt_columns"]

    p = dict(zip(profile["metric"], profile["value"]))
    v = dict(zip(validation["metric"], validation["value"]))

    top_domains = tables["domain_distribution_plot"].head(3)
    top_regions = tables["region_distribution_plot"].head(3)
    top_companies = tables["company_distribution_plot"].head(3)
    top_tools = tables["tools_top"].head(5)

    industry_cov = float(v.get("coverage_industry_talentCard_%", 0))
    russia_collapsed = geo_audit[
        geo_audit["raw_region"].astype(str).str.contains(r"russia|россия|рф|russian federation", case=False, regex=True)
        & geo_audit["region_norm"].astype(str).eq("Russia")
    ]["count"].sum()
    region_ns_share = float(
        region_ns_breakdown.loc[region_ns_breakdown["metric"] == "share_of_dataset_%", "value"].iloc[0]
    ) if not region_ns_breakdown.empty else 0.0
    company_ns_share = float(
        company_ns_breakdown.loc[company_ns_breakdown["metric"] == "share_of_dataset_%", "value"].iloc[0]
    ) if not company_ns_breakdown.empty else 0.0
    alt_geo_used = (
        int(region_alt_cols["selected_in_fallback"].fillna(False).sum())
        if "selected_in_fallback" in region_alt_cols.columns
        else 0
    )

    observations = [
        f"База содержит {int(float(p['users_total']))} профилей.",
        f"Покрытие `cvEnhancedResult`: {float(v['coverage_cvEnhancedResult_%']):.1f}%.",
        f"Покрытие `talentCard.jobs`: {float(v['coverage_talentCard_jobs_%']):.1f}%.",
        f"Покрытие `talentCard.overall_skills`: {float(v['coverage_overall_skills_%']):.1f}%.",
        f"Покрытие `talentCard.specialist_category`: {float(v['coverage_specialist_category_%']):.1f}%.",
        f"LaTeX-парсинг нашел `ExpHeader` у {float(v['share_users_with_expheader_%']):.1f}% пользователей.",
        f"LaTeX skills section найдена у {float(v['share_users_with_skills_section_%']):.1f}% пользователей.",
        f"Гео-мэппинг схлопнул варианты Russia/Россия/РФ: {int(russia_collapsed)} записей перешли в канон `Russia`.",
        f"Industry анализируется только на subset с заполненным industry: {industry_cov:.1f}% пользователей.",
        f"`region=Not specified` остается у {region_ns_share:.1f}% базы; `company=Not specified` — у {company_ns_share:.1f}%.",
        f"В fallback цепочку региона добавлено альтернативных geo-колонок: {alt_geo_used}.",
        f"Топ tools: {', '.join(top_tools['token_display'].tolist())}.",
    ]

    diagnostics_md = diagnostics.to_markdown(index=False)
    validation_md = validation.to_markdown(index=False)
    top_domains_md = top_domains[["domain_filled", "count"]].to_markdown(index=False) if not top_domains.empty else "(empty)"
    top_regions_md = top_regions[["region_display", "count"]].to_markdown(index=False) if not top_regions.empty else "(empty)"
    top_companies_md = top_companies[["company_display", "count"]].to_markdown(index=False) if not top_companies.empty else "(empty)"
    deep_dive_summary_md = tables["not_specified_deep_dive_summary"].to_markdown(index=False)

    readme = f"""# MIS: Users Resume Bot (Candidate Analytics)

## 1) Summary
- Users total: **{int(float(p['users_total']))}**
- Coverage `cvEnhancedResult`: **{float(v['coverage_cvEnhancedResult_%']):.1f}%**
- Coverage `talentCard.jobs`: **{float(v['coverage_talentCard_jobs_%']):.1f}%**
- Coverage `talentCard.overall_skills`: **{float(v['coverage_overall_skills_%']):.1f}%**
- Coverage `talentCard.specialist_category`: **{float(v['coverage_specialist_category_%']):.1f}%**

Top-3 domains:
{top_domains_md}

Top-3 regions:
{top_regions_md}

Top-3 companies:
{top_companies_md}

### Key observations
{"\n".join([f"- {x}" for x in observations])}

## 2) Coverage / Parsing validation
{validation_md}

Columns inventory (real CSV structure + non-null profile + length stats):
- `outputs/tables/columns_inventory.csv`

## 3) Domains & Geography
![Weekly signups](outputs/figures/01_registrations_weekly.png)
![Top domains](outputs/figures/02_top_domains.png)
![Top regions](outputs/figures/03_top_regions.png)
![Domain x Region heatmap (row share)](outputs/figures/06_heatmap_domain_region_share.png)
![Region composition by domains (100% stacked)](outputs/figures/07_stacked_region_domain_top10.png)

## 4) Companies & Seniority
![Top companies](outputs/figures/04_top_companies.png)
![Top industries (subset)](outputs/figures/05_top_industries_subset.png)
![Seniority x Company (100% stacked)](outputs/figures/08_stacked_seniority_company_top15.png)
![Domain x Company (100% stacked)](outputs/figures/09_stacked_domain_company_top15.png)

## 5) Skills & Stack
![Top tools](outputs/figures/10_top_tools.png)
![Top skills](outputs/figures/11_top_skills.png)
![Domain x Tools heatmap (row share)](outputs/figures/12_heatmap_domain_tools_share.png)

## 6) Стратификация выборки
<p>
  <img src="outputs/figures/13_donut_seniority_filled.png" width="48%" />
  <img src="outputs/figures/14_donut_cv_generation_language.png" width="48%" />
</p>

![Top-20 strata](outputs/figures/15_strata_top20.png)

Ключевые таблицы стратификации:
- `outputs/tables/strata_top20.csv`
- `outputs/tables/domain_distribution.csv`
- `outputs/tables/role_family_distribution.csv`
- `outputs/tables/seniority_distribution.csv`
- `outputs/tables/experience_bin_distribution.csv`
- `outputs/tables/leadership_distribution.csv`
- `outputs/tables/cv_generation_language_distribution.csv`

## 7) Not specified diagnostics
{diagnostics_md}

Пустоты уменьшались по fallback-цепочкам:
- `region_filled`: `latex_expheader -> talentCard -> latex_header -> alt_geo_columns`
- `seniority_filled`: `talentCard -> inferred_job_title -> inferred_header_role`
- `domain_filled`: `talentCard.specialist_category -> inferred role family`

## 8) Other/Not specified deep dive
{deep_dive_summary_md}

Таблицы deep dive:
- `outputs/tables/not_specified_deep_dive_domain_titles.csv`
- `outputs/tables/not_specified_deep_dive_domain_selected_position.csv`
- `outputs/tables/not_specified_deep_dive_domain_tools.csv`
- `outputs/tables/not_specified_deep_dive_domain_skills.csv`
- `outputs/tables/not_specified_deep_dive_region_not_specified_breakdown.csv`
- `outputs/tables/not_specified_deep_dive_region_not_specified_domain.csv`
- `outputs/tables/not_specified_deep_dive_region_not_specified_seniority.csv`
- `outputs/tables/not_specified_deep_dive_region_alt_columns.csv`
- `outputs/tables/not_specified_deep_dive_company_not_specified_breakdown.csv`
- `outputs/tables/not_specified_deep_dive_company_not_specified_job_titles.csv`
- `outputs/tables/not_specified_deep_dive_company_not_specified_region.csv`

Графики deep dive:
![Domain Other/Not specified titles](outputs/figures/16_deep_dive_domain_titles.png)
![Region Not specified by domain](outputs/figures/17_deep_dive_region_not_specified_domain.png)
![Company Not specified top titles](outputs/figures/18_deep_dive_company_not_specified_titles.png)

## 9) Appendix
Артефакты:
- Figures: `outputs/figures/*.png`
- Tables: `outputs/tables/*.csv`
- Notebook: `notebooks/mis_users_resume_bot.ipynb`
- Geo mapping audit: `outputs/tables/geo_mapping_audit.csv`

How to reproduce:
```bash
python analytics/mis_users_resume_bot/src/build_mis.py \
  --input /mnt/data/prointerview-prod.users.csv \
  --base-dir analytics/mis_users_resume_bot
```
"""

    (base_dir / "README.md").write_text(readme, encoding="utf-8")


def run(input_path: str, base_dir: str) -> Dict[str, pd.DataFrame]:
    base = Path(base_dir)
    tables_dir = base / "outputs" / "tables"
    figures_dir = base / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Prevent stale artifacts from previous report versions.
    for old_csv in tables_dir.glob("*.csv"):
        old_csv.unlink()
    for old_png in figures_dir.glob("*.png"):
        old_png.unlink()

    users_raw = pd.read_csv(input_path, low_memory=False)
    columns_inventory = build_columns_inventory(users_raw)

    users = normalize_empty_strings(users_raw)

    for col in DATE_COLUMNS:
        if col in users.columns:
            users[col] = pd.to_datetime(users[col], errors="coerce", utc=True)

    users["onboardingCompleted"] = parse_bool_series(users.get("onboardingCompleted", pd.Series(index=users.index, dtype=object)), default_false=True)
    users["isBanned"] = parse_bool_series(users.get("isBanned", pd.Series(index=users.index, dtype=object)), default_false=True)

    users["user_hash"] = users["userId"].map(hash_user_id)
    users["cvEnhancedResult_present"] = users.get("cvEnhancedResult", "").fillna("").astype(str).str.strip().ne("")

    as_of = users["updatedAt"].dropna().max() if "updatedAt" in users.columns else pd.NaT
    if pd.isna(as_of):
        as_of = pd.Timestamp.utcnow()

    jobs_talent = build_jobs_long_talent(users, as_of)
    jobs_talent_summary = summarize_user_jobs(jobs_talent) if not jobs_talent.empty else pd.DataFrame(columns=["user_hash"])

    latex_summary, jobs_latex = parse_latex_dataset(users, as_of)

    users_enriched = users.merge(jobs_talent_summary, on="user_hash", how="left")
    users_enriched = users_enriched.merge(latex_summary, on="user_hash", how="left")

    users_enriched["jobs_count_talentCard"] = users_enriched["jobs_count_talentCard"].fillna(0).astype(int)
    users_enriched["jobs_count_latex"] = users_enriched["jobs_count_latex"].fillna(0).astype(int)

    alt_geo_inventory = discover_alt_geo_columns(users)
    selected_alt_geo_cols: List[str] = []
    if not alt_geo_inventory.empty:
        selected_alt_geo_cols = (
            alt_geo_inventory.loc[alt_geo_inventory["non_empty_rate_%"] >= 1.0, "column"].head(2).astype(str).tolist()
        )

    # Filled fields with explicit source fallback chains.
    company_rows = users_enriched.apply(
        lambda r: choose_with_source(
            [
                (r.get("current_company_expheader", ""), "latex_expheader"),
                (r.get("current_company_talentCard", ""), "talentCard"),
            ]
        ),
        axis=1,
    )
    users_enriched["company_filled"] = company_rows.map(lambda x: x[0])
    users_enriched["company_source"] = company_rows.map(lambda x: x[1])

    region_rows = users_enriched.apply(
        lambda r: choose_with_source(
            [
                (r.get("current_region_expheader", ""), "latex_expheader"),
                (r.get("current_region_talentCard", ""), "talentCard"),
                (r.get("header_location_latex", ""), "latex_header"),
                *[(r.get(col, ""), f"alt_col:{col}") for col in selected_alt_geo_cols],
            ]
        ),
        axis=1,
    )
    users_enriched["region_filled"] = region_rows.map(lambda x: x[0])
    users_enriched["region_source"] = region_rows.map(lambda x: x[1])

    users_enriched["current_job_title_filled"] = users_enriched.apply(
        lambda r: choose_with_source(
            [
                (r.get("current_job_title_expheader", ""), "latex_expheader"),
                (r.get("current_job_title_talentCard", ""), "talentCard"),
                (r.get("header_role_latex", ""), "latex_header"),
            ]
        )[0],
        axis=1,
    )

    def infer_seniority(row: pd.Series) -> Tuple[str, str]:
        s_talent = normalize_seniority(row.get("current_seniority_talentCard", ""))
        if s_talent != "Unknown":
            return s_talent, "talentCard"

        s_title = normalize_seniority(row.get("current_job_title_filled", ""))
        if s_title != "Unknown":
            return s_title, "inferred_job_title"

        s_header = normalize_seniority(row.get("header_role_latex", ""))
        if s_header != "Unknown":
            return s_header, "inferred_header_role"

        return "Not specified", "not_specified"

    seniority_rows = users_enriched.apply(infer_seniority, axis=1)
    users_enriched["seniority_filled"] = seniority_rows.map(lambda x: x[0])
    users_enriched["seniority_source"] = seniority_rows.map(lambda x: x[1])

    def infer_domain(row: pd.Series) -> Tuple[str, str]:
        domain = clean_text(row.get("talentCard.specialist_category", ""))
        if domain:
            return domain, "talentCard"

        infer_text = " ".join(
            [
                clean_text(row.get("selectedPosition", "")),
                clean_text(row.get("current_job_title_filled", "")),
                clean_text(row.get("header_role_latex", "")),
            ]
        )
        if clean_text(infer_text):
            inferred = infer_role_family(row.get("selectedPosition", ""), infer_text, "")
            return ("Other" if inferred == "Other" else inferred), "inferred"

        return "Not specified", "not_specified"

    domain_rows = users_enriched.apply(infer_domain, axis=1)
    users_enriched["domain_filled"] = domain_rows.map(lambda x: x[0])
    users_enriched["domain_source"] = domain_rows.map(lambda x: x[1])

    users_enriched["industry_filled"] = users_enriched["current_industry_talentCard"].fillna("").astype(str).str.strip()
    users_enriched["industry_source"] = np.where(users_enriched["industry_filled"].ne(""), "talentCard", "not_specified")
    users_enriched.loc[users_enriched["industry_filled"].eq(""), "industry_filled"] = "Not specified"

    users_enriched["region_norm"] = users_enriched["region_filled"].map(normalize_region)
    users_enriched["company_norm"] = users_enriched["company_filled"].map(normalize_company)
    users_enriched["industry_norm"] = users_enriched["industry_filled"].map(normalize_industry)
    users_enriched["country_guess"] = users_enriched["region_norm"].map(guess_country)

    users_enriched["role_family"] = users_enriched.apply(
        lambda r: infer_role_family(r.get("selectedPosition", ""), r.get("current_job_title_filled", ""), r.get("header_role_latex", "")),
        axis=1,
    )
    users_enriched["experience_bin"] = users_enriched["total_experience_years"].map(experience_bin)
    users_enriched["leadership_level"] = users_enriched.apply(
        lambda r: leadership_level(r.get("current_job_title_filled", ""), r.get("seniority_filled", ""), r.get("total_experience_years", np.nan)),
        axis=1,
    )

    merged_skills = users_enriched.apply(
        lambda r: merge_skills_tools(
            r.get("talentCard.overall_skills", ""),
            r.get("talentCard.overall_tools", ""),
            r.get("skills_list_latex", []) if isinstance(r.get("skills_list_latex", []), list) else [],
            r.get("tools_list_latex", []) if isinstance(r.get("tools_list_latex", []), list) else [],
        ),
        axis=1,
    )
    users_enriched["skills_list"] = merged_skills.map(lambda x: x[0])
    users_enriched["tools_list"] = merged_skills.map(lambda x: x[1])
    users_enriched["skills_count"] = users_enriched["skills_list"].map(len)
    users_enriched["tools_count"] = users_enriched["tools_list"].map(len)

    # Core coverage and validation.
    total_users = len(users_enriched)
    cover_cv_enhanced = pct(int(users_enriched["cvEnhancedResult_present"].sum()), total_users)
    cover_jobs = pct(int((users_enriched["jobs_count_talentCard"] > 0).sum()), total_users)
    cover_overall_skills = pct(int(users.get("talentCard.overall_skills", "").fillna("").astype(str).str.strip().ne("").sum()), total_users)
    cover_spec_cat = pct(int(users.get("talentCard.specialist_category", "").fillna("").astype(str).str.strip().ne("").sum()), total_users)
    cover_industry = pct(int(users_enriched["industry_filled"].ne("Not specified").sum()), total_users)

    compare_company = users_enriched[["user_hash", "current_company_talentCard", "current_company_expheader"]].copy()
    compare_company["company_talent_norm"] = compare_company["current_company_talentCard"].map(normalize_company)
    compare_company["company_latex_norm"] = compare_company["current_company_expheader"].map(normalize_company)
    comparable_mask = compare_company["company_talent_norm"].ne("Not specified") & compare_company["company_latex_norm"].ne("Not specified")
    comparable_n = int(comparable_mask.sum())
    matches_n = int((compare_company.loc[comparable_mask, "company_talent_norm"] == compare_company.loc[comparable_mask, "company_latex_norm"]).sum())

    jobs_diff = users_enriched[["jobs_count_talentCard", "jobs_count_latex"]].copy()
    jobs_diff["diff_latex_minus_talent"] = jobs_diff["jobs_count_latex"] - jobs_diff["jobs_count_talentCard"]
    jobs_diff_distribution = jobs_diff["diff_latex_minus_talent"].value_counts().rename_axis("diff").reset_index(name="users").sort_values("diff")

    mismatch_samples = compare_company.loc[
        comparable_mask & (compare_company["company_talent_norm"] != compare_company["company_latex_norm"]),
        ["user_hash", "current_company_talentCard", "current_company_expheader", "company_talent_norm", "company_latex_norm"],
    ].head(200)

    validation_summary = pd.DataFrame(
        {
            "metric": [
                "users_total",
                "coverage_cvEnhancedResult_%",
                "coverage_talentCard_jobs_%",
                "coverage_overall_skills_%",
                "coverage_specialist_category_%",
                "coverage_industry_talentCard_%",
                "users_with_latex_block",
                "share_users_with_latex_block_%",
                "users_with_expheader",
                "share_users_with_expheader_%",
                "users_with_skills_section",
                "share_users_with_skills_section_%",
                "users_with_languages_section",
                "share_users_with_languages_section_%",
                "users_with_education_section",
                "share_users_with_education_section_%",
                "company_comparable_users",
                "current_company_matches",
                "current_company_match_rate_%",
            ],
            "value": [
                total_users,
                cover_cv_enhanced,
                cover_jobs,
                cover_overall_skills,
                cover_spec_cat,
                cover_industry,
                int(users_enriched["latex_found"].fillna(False).sum()),
                pct(int(users_enriched["latex_found"].fillna(False).sum()), total_users),
                int((users_enriched["expheader_count"].fillna(0) > 0).sum()),
                pct(int((users_enriched["expheader_count"].fillna(0) > 0).sum()), total_users),
                int(users_enriched["skills_section_found"].fillna(False).sum()),
                pct(int(users_enriched["skills_section_found"].fillna(False).sum()), total_users),
                int(users_enriched["languages_section_found"].fillna(False).sum()),
                pct(int(users_enriched["languages_section_found"].fillna(False).sum()), total_users),
                int(users_enriched["education_section_found"].fillna(False).sum()),
                pct(int(users_enriched["education_section_found"].fillna(False).sum()), total_users),
                comparable_n,
                matches_n,
                pct(matches_n, comparable_n),
            ],
        }
    )

    # Distributions.
    users_enriched["created_week"] = (
        users_enriched["createdAt"].dt.tz_convert(None).dt.to_period("W").astype(str)
        if getattr(users_enriched["createdAt"].dt, "tz", None) is not None
        else users_enriched["createdAt"].dt.to_period("W").astype(str)
    )
    weekly_signups = users_enriched.groupby("created_week").size().reset_index(name="new_users").sort_values("created_week")

    domain_distribution = users_enriched["domain_filled"].fillna("Not specified").astype(str).value_counts().rename_axis("domain_filled").reset_index(name="count")
    role_family_distribution = users_enriched["role_family"].fillna("Other").astype(str).value_counts().rename_axis("role_family").reset_index(name="count")

    seniority_order = ["Junior", "Middle", "Senior", "Lead", "C-level", "Not specified"]
    seniority_distribution = users_enriched["seniority_filled"].fillna("Not specified").astype(str).value_counts().rename_axis("seniority_filled").reset_index(name="count")
    seniority_distribution["_ord"] = seniority_distribution["seniority_filled"].map({k: i for i, k in enumerate(seniority_order)}).fillna(999)
    seniority_distribution = seniority_distribution.sort_values(["_ord", "count"], ascending=[True, False]).drop(columns=["_ord"])

    experience_distribution = users_enriched["experience_bin"].fillna("Unknown").astype(str).value_counts().rename_axis("experience_bin").reset_index(name="count")
    leadership_distribution = users_enriched["leadership_level"].fillna("Low").astype(str).value_counts().rename_axis("leadership_level").reset_index(name="count")

    cv_generation_language_distribution = (
        users_enriched.get("cvGenerationLanguage", pd.Series(index=users_enriched.index, dtype=object))
        .fillna("Not specified")
        .astype(str)
        .value_counts()
        .rename_axis("cvGenerationLanguage")
        .reset_index(name="count")
    )

    region_distribution = distribution_with_display(users_enriched, "region_filled", "region_norm", "region")
    company_distribution = distribution_with_display(users_enriched, "company_filled", "company_norm", "company")
    industry_distribution = distribution_with_display(users_enriched, "industry_filled", "industry_norm", "industry")

    domain_distribution_plot = domain_distribution[
        ~domain_distribution["domain_filled"].astype(str).str.lower().isin({"other", "not specified"})
    ].copy()
    region_distribution_plot = region_distribution[
        ~region_distribution["region_norm"].astype(str).str.lower().isin({"other", "not specified"})
    ].copy()
    company_distribution_plot = company_distribution[
        ~company_distribution["company_norm"].astype(str).str.lower().isin({"other", "not specified"})
    ].copy()

    if domain_distribution_plot.empty:
        domain_distribution_plot = domain_distribution.copy()
    if region_distribution_plot.empty:
        region_distribution_plot = region_distribution.copy()
    if company_distribution_plot.empty:
        company_distribution_plot = company_distribution.copy()

    industry_subset = industry_distribution[industry_distribution["industry_norm"] != "Not specified"].copy()

    country_distribution = users_enriched["country_guess"].fillna("Unknown").astype(str).value_counts().rename_axis("country").reset_index(name="count")
    english_level_distribution = users_enriched["english_level"].fillna("Unknown").astype(str).value_counts().rename_axis("english_level").reset_index(name="count")
    degree_level_distribution = users_enriched["degree_level"].fillna("Unknown").astype(str).value_counts().rename_axis("degree_level").reset_index(name="count")

    # Skills/stack outputs.
    skills_top = aggregate_tokens(users_enriched, "skills_list")
    tools_top = aggregate_tokens(users_enriched, "tools_list")

    # Domain x Region (heatmap row-share) and stacked by regions.
    domain_analysis = users_enriched[
        ~users_enriched["domain_filled"].fillna("Not specified").astype(str).str.lower().isin({"other", "not specified"})
    ].copy()
    if domain_analysis.empty:
        domain_analysis = users_enriched.copy()

    domain_small = limit_categories(domain_analysis["domain_filled"], max_n=10, other_label="Other")
    region_small = limit_categories(domain_analysis["region_norm"], max_n=15, other_label="Other")

    domain_region_counts = pd.crosstab(domain_small, region_small)
    domain_region_share = row_normalize_percent(domain_region_counts)

    region_domain_base = users_enriched[
        ~users_enriched["region_norm"].fillna("Not specified").astype(str).str.lower().isin({"not specified", "other"})
    ].copy()
    if region_domain_base.empty:
        region_domain_base = users_enriched.copy()

    region_top10 = limit_categories(region_domain_base["region_norm"], max_n=10, other_label="Other")
    domain_for_regions = limit_categories(region_domain_base["domain_filled"], max_n=10, other_label="Other")
    region_domain_counts = pd.crosstab(region_top10, domain_for_regions)

    # Seniority x Company and Domain x Company as 100% stacked bars.
    company_analysis = users_enriched[
        ~users_enriched["company_norm"].fillna("Not specified").astype(str).str.lower().isin({"not specified", "other"})
    ].copy()
    if company_analysis.empty:
        company_analysis = users_enriched.copy()

    company_small = limit_categories(company_analysis["company_norm"], max_n=15, other_label="Other")
    seniority_small = company_analysis["seniority_filled"].fillna("Not specified")
    seniority_company_counts = pd.crosstab(company_small, seniority_small)
    for cat in seniority_order:
        if cat not in seniority_company_counts.columns:
            seniority_company_counts[cat] = 0
    seniority_company_counts = seniority_company_counts[seniority_order]
    seniority_company_counts = seniority_company_counts.sort_values(by=seniority_order, ascending=False)

    domain_company_counts = pd.crosstab(company_small, company_analysis["domain_filled"])

    # Domain x Tools heatmap (top-8 domains x top-20 tools, row share).
    domain_top8 = domain_distribution_plot.head(8)["domain_filled"].astype(str).tolist()
    tools_top20 = tools_top.head(20)["token_norm"].tolist()
    token_display_map = dict(zip(tools_top["token_norm"], tools_top["token_display"]))

    user_tool_sets = users_enriched["tools_list"].map(lambda xs: {canonical_text(x) for x in (xs or []) if canonical_text(x)})
    domain_tools_matrix = pd.DataFrame(0.0, index=domain_top8, columns=[token_display_map.get(t, t) for t in tools_top20])

    for domain in domain_top8:
        mask = users_enriched["domain_filled"].fillna("Not specified").astype(str).eq(domain)
        denom = int(mask.sum())
        if denom == 0:
            continue
        sets_subset = user_tool_sets[mask]
        for tool_norm in tools_top20:
            hits = int(sets_subset.map(lambda s: tool_norm in s).sum())
            domain_tools_matrix.loc[domain, token_display_map.get(tool_norm, tool_norm)] = round(hits / denom * 100, 1)

    # Extra skills slices by domain/region.
    tools_by_domain_rows: List[Dict[str, object]] = []
    for domain in domain_top8[:6]:
        mask = users_enriched["domain_filled"].astype(str).eq(domain)
        local = aggregate_tokens(users_enriched.loc[mask], "tools_list").head(5)
        for _, r in local.iterrows():
            tools_by_domain_rows.append({"domain": domain, "tool": r["token_display"], "count": int(r["count"])})
    tools_by_domain_top = pd.DataFrame(tools_by_domain_rows)

    region_top10_values = region_distribution.head(10)["region_norm"].tolist()
    tools_by_region_rows: List[Dict[str, object]] = []
    for region in region_top10_values:
        mask = users_enriched["region_norm"].astype(str).eq(region)
        local = aggregate_tokens(users_enriched.loc[mask], "tools_list").head(5)
        for _, r in local.iterrows():
            tools_by_region_rows.append({"region_norm": region, "tool": r["token_display"], "count": int(r["count"])})
    tools_by_region_top = pd.DataFrame(tools_by_region_rows)

    # Stratification.
    strata_top20 = (
        users_enriched.groupby(["domain_filled", "seniority_filled", "region_norm"]).size().reset_index(name="count").sort_values("count", ascending=False).head(20)
    )
    strata_top20["strata"] = strata_top20.apply(
        lambda r: f"{r['domain_filled']} | {r['seniority_filled']} | {r['region_norm']}", axis=1
    )

    # Missingness and diagnostics.
    missing_fields = [
        "createdAt",
        "updatedAt",
        "cvEnhancedResult",
        "talentCard.specialist_category",
        "talentCard.overall_skills",
        "talentCard.overall_tools",
        "cvGenerationLanguage",
        "isBanned",
    ]
    missingness = pd.DataFrame(
        {
            "field": missing_fields,
            "missing_%": [round(users[c].isna().mean() * 100, 1) if c in users.columns else 100.0 for c in missing_fields],
            "filled_%": [round((1 - users[c].isna().mean()) * 100, 1) if c in users.columns else 0.0 for c in missing_fields],
        }
    ).sort_values("missing_%", ascending=False)

    not_specified_diagnostics = build_not_specified_diagnostics(users_enriched)

    deep_fields = ["domain", "region", "company", "seniority"]
    not_specified_deep_dive_summary = not_specified_diagnostics[
        not_specified_diagnostics["field"].isin(deep_fields)
    ].copy()

    # Domain = Other/Not specified deep dive.
    domain_deep = users_enriched[
        users_enriched["domain_filled"].fillna("Not specified").astype(str).isin(["Other", "Not specified"])
    ].copy()
    domain_titles = (
        domain_deep.groupby(["domain_filled", "current_job_title_filled"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )
    domain_selected_position = (
        domain_deep.groupby(["domain_filled", "selectedPosition"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )

    domain_tools_rows: List[Dict[str, object]] = []
    domain_skills_rows: List[Dict[str, object]] = []
    for dom in ["Other", "Not specified"]:
        subset = domain_deep[domain_deep["domain_filled"] == dom]
        local_tools = top_tokens_from_list(subset, "tools_list", 20, "tool")
        local_skills = top_tokens_from_list(subset, "skills_list", 20, "skill")
        if not local_tools.empty:
            local_tools["domain_filled"] = dom
            domain_tools_rows.append(local_tools)
        if not local_skills.empty:
            local_skills["domain_filled"] = dom
            domain_skills_rows.append(local_skills)
    domain_tools_deep = pd.concat(domain_tools_rows, ignore_index=True) if domain_tools_rows else pd.DataFrame(columns=["tool", "count", "share_%", "domain_filled"])
    domain_skills_deep = pd.concat(domain_skills_rows, ignore_index=True) if domain_skills_rows else pd.DataFrame(columns=["skill", "count", "share_%", "domain_filled"])

    # Region = Not specified deep dive.
    region_ns = users_enriched[users_enriched["region_norm"].fillna("Not specified").astype(str).str.lower().eq("not specified")].copy()
    region_ns_total = len(region_ns)
    region_ns_breakdown = pd.DataFrame(
        {
            "metric": [
                "users_region_not_specified",
                "share_of_dataset_%",
                "cvEnhancedResult_present_%",
                "latex_found_%",
                "expheader_gt0_%",
                "jobs_count_talentCard_gt0_%",
            ],
            "value": [
                region_ns_total,
                round(region_ns_total / len(users_enriched) * 100, 1) if len(users_enriched) else 0.0,
                round(region_ns["cvEnhancedResult_present"].mean() * 100, 1) if region_ns_total else 0.0,
                round(region_ns["latex_found"].fillna(False).mean() * 100, 1) if region_ns_total else 0.0,
                round((region_ns["expheader_count"].fillna(0) > 0).mean() * 100, 1) if region_ns_total else 0.0,
                round((region_ns["jobs_count_talentCard"] > 0).mean() * 100, 1) if region_ns_total else 0.0,
            ],
        }
    )
    region_ns_domain = top_counts(region_ns["domain_filled"], 20, "domain_filled")
    region_ns_seniority = top_counts(region_ns["seniority_filled"], 20, "seniority_filled")

    if not alt_geo_inventory.empty:
        region_alt_columns = alt_geo_inventory.copy()
        region_alt_columns["selected_in_fallback"] = region_alt_columns["column"].isin(selected_alt_geo_cols)
    else:
        region_alt_columns = pd.DataFrame(
            [{"column": "none", "non_empty_count": 0, "non_empty_rate_%": 0.0, "median_len": 0, "selected_in_fallback": False}]
        )

    # Company = Not specified deep dive.
    company_ns = users_enriched[users_enriched["company_norm"].fillna("Not specified").astype(str).str.lower().eq("not specified")].copy()
    company_ns_total = len(company_ns)
    company_ns_breakdown = pd.DataFrame(
        {
            "metric": [
                "users_company_not_specified",
                "share_of_dataset_%",
                "expheader_gt0_%",
                "jobs_count_talentCard_gt0_%",
            ],
            "value": [
                company_ns_total,
                round(company_ns_total / len(users_enriched) * 100, 1) if len(users_enriched) else 0.0,
                round((company_ns["expheader_count"].fillna(0) > 0).mean() * 100, 1) if company_ns_total else 0.0,
                round((company_ns["jobs_count_talentCard"] > 0).mean() * 100, 1) if company_ns_total else 0.0,
            ],
        }
    )
    company_ns_job_titles = top_counts(company_ns["current_job_title_filled"], 20, "current_job_title_filled")
    company_ns_regions = top_counts(company_ns["region_norm"], 20, "region_norm")

    geo_mapping_audit = (
        users_enriched.groupby(["region_filled", "region_norm"]).size().reset_index(name="count").rename(columns={"region_filled": "raw_region"}).sort_values("count", ascending=False)
    )

    region_mappings = build_mapping_table(users_enriched["region_filled"], users_enriched["region_norm"], "region")
    company_mappings = build_mapping_table(users_enriched["company_filled"], users_enriched["company_norm"], "company")
    industry_mappings = build_mapping_table(users_enriched["industry_filled"], users_enriched["industry_norm"], "industry")

    dataset_profile = pd.DataFrame(
        {
            "metric": [
                "users_total",
                "columns_total",
                "createdAt_min",
                "createdAt_max",
                "coverage_cvEnhancedResult_%",
                "coverage_talentCard_jobs_%",
                "coverage_overall_skills_%",
                "coverage_specialist_category_%",
            ],
            "value": [
                total_users,
                users.shape[1],
                str(users["createdAt"].min()) if "createdAt" in users.columns else "",
                str(users["createdAt"].max()) if "createdAt" in users.columns else "",
                cover_cv_enhanced,
                cover_jobs,
                cover_overall_skills,
                cover_spec_cat,
            ],
        }
    )

    tables: Dict[str, pd.DataFrame] = {
        "columns_inventory": columns_inventory,
        "dataset_profile": dataset_profile,
        "validation_summary": validation_summary,
        "jobs_count_diff_distribution": jobs_diff_distribution,
        "mismatch_samples": mismatch_samples,
        "missingness": missingness,
        "not_specified_diagnostics": not_specified_diagnostics,
        "weekly_signups": weekly_signups,
        "domain_distribution": domain_distribution,
        "domain_distribution_plot": domain_distribution_plot,
        "role_family_distribution": role_family_distribution,
        "seniority_distribution": seniority_distribution,
        "experience_bin_distribution": experience_distribution,
        "leadership_distribution": leadership_distribution,
        "cv_generation_language_distribution": cv_generation_language_distribution,
        "region_distribution": region_distribution,
        "region_distribution_plot": region_distribution_plot,
        "country_distribution": country_distribution,
        "company_distribution": company_distribution,
        "company_distribution_plot": company_distribution_plot,
        "industry_distribution": industry_distribution,
        "industry_distribution_subset": industry_subset,
        "english_level_distribution": english_level_distribution,
        "degree_level_distribution": degree_level_distribution,
        "skills_top": skills_top,
        "tools_top": tools_top,
        "tools_by_domain_top": tools_by_domain_top,
        "tools_by_region_top": tools_by_region_top,
        "domain_region_heatmap_share": domain_region_share.reset_index(),
        "region_domain_stacked_top10": row_normalize_percent(region_domain_counts).reset_index(),
        "seniority_company_stacked100": row_normalize_percent(seniority_company_counts).reset_index(),
        "domain_company_stacked100": row_normalize_percent(domain_company_counts).reset_index(),
        "domain_tools_heatmap": domain_tools_matrix.reset_index().rename(columns={"index": "domain_filled"}),
        "region_mappings": region_mappings,
        "company_mappings": company_mappings,
        "industry_mappings": industry_mappings,
        "geo_mapping_audit": geo_mapping_audit,
        "strata_top20": strata_top20,
        "not_specified_deep_dive_summary": not_specified_deep_dive_summary,
        "not_specified_deep_dive_domain_titles": domain_titles,
        "not_specified_deep_dive_domain_selected_position": domain_selected_position,
        "not_specified_deep_dive_domain_tools": domain_tools_deep,
        "not_specified_deep_dive_domain_skills": domain_skills_deep,
        "not_specified_deep_dive_region_not_specified_breakdown": region_ns_breakdown,
        "not_specified_deep_dive_region_not_specified_domain": region_ns_domain,
        "not_specified_deep_dive_region_not_specified_seniority": region_ns_seniority,
        "not_specified_deep_dive_region_alt_columns": region_alt_columns,
        "not_specified_deep_dive_company_not_specified_breakdown": company_ns_breakdown,
        "not_specified_deep_dive_company_not_specified_job_titles": company_ns_job_titles,
        "not_specified_deep_dive_company_not_specified_region": company_ns_regions,
    }

    for name, table in tables.items():
        save_table(table, tables_dir / f"{name}.csv")

    # Figures.
    plot_line(
        weekly_signups,
        x_col="created_week",
        y_col="new_users",
        out_path=figures_dir / "01_registrations_weekly.png",
        title="Registrations by week",
        xlabel="Week",
        ylabel="Users",
    )
    plot_barh(
        domain_distribution_plot,
        "domain_filled",
        "count",
        figures_dir / "02_top_domains.png",
        "Top domains (excluding Other/Not specified)",
        top_n=10,
    )
    plot_barh(
        region_distribution_plot.rename(columns={"region_display": "region"}),
        "region",
        "count",
        figures_dir / "03_top_regions.png",
        "Top regions (excluding Other/Not specified)",
        top_n=15,
    )
    plot_barh(
        company_distribution_plot.rename(columns={"company_display": "company"}),
        "company",
        "count",
        figures_dir / "04_top_companies.png",
        "Top companies (excluding Other/Not specified)",
        top_n=15,
    )
    if not industry_subset.empty:
        plot_barh(industry_subset.rename(columns={"industry_display": "industry"}), "industry", "count", figures_dir / "05_top_industries_subset.png", "Top industries (subset with non-empty industry)", top_n=15)

    plot_heatmap_share(
        domain_region_share,
        figures_dir / "06_heatmap_domain_region_share.png",
        "Domain x Region (row-normalized share)",
        "Region",
        "Domain",
    )

    plot_stacked_100(
        region_domain_counts,
        figures_dir / "07_stacked_region_domain_top10.png",
        "Region composition by domains (100% stacked)",
        "Region",
    )

    plot_stacked_100(
        seniority_company_counts,
        figures_dir / "08_stacked_seniority_company_top15.png",
        "Seniority x Company (100% stacked)",
        "Company",
    )

    plot_stacked_100(
        domain_company_counts,
        figures_dir / "09_stacked_domain_company_top15.png",
        "Domain x Company (100% stacked)",
        "Company",
    )

    plot_barh(tools_top.rename(columns={"token_display": "tool"}), "tool", "count", figures_dir / "10_top_tools.png", "Top tools / stack", top_n=20)
    plot_barh(skills_top.rename(columns={"token_display": "skill"}), "skill", "count", figures_dir / "11_top_skills.png", "Top skills", top_n=20)

    if not domain_tools_matrix.empty:
        plot_heatmap_share(
            domain_tools_matrix,
            figures_dir / "12_heatmap_domain_tools_share.png",
            "Domain x Tools (row-normalized share)",
            "Tool",
            "Domain",
        )

    seniority_donut = plot_donut(users_enriched["seniority_filled"], figures_dir / "13_donut_seniority_filled.png", "Seniority mix")
    cv_lang_donut = plot_donut(users_enriched.get("cvGenerationLanguage", pd.Series(index=users_enriched.index, dtype=object)), figures_dir / "14_donut_cv_generation_language.png", "CV generation language mix")
    if not seniority_donut.empty:
        save_table(seniority_donut, tables_dir / "seniority_donut_distribution.csv")
    if not cv_lang_donut.empty:
        save_table(cv_lang_donut, tables_dir / "cv_generation_language_donut_distribution.csv")

    if not strata_top20.empty:
        plot_barh(
            strata_top20.rename(columns={"strata": "strata_label"}),
            "strata_label",
            "count",
            figures_dir / "15_strata_top20.png",
            "Top-20 strata: domain x seniority x region",
            top_n=20,
        )

    if not domain_titles.empty:
        deep_titles_plot = domain_titles.copy()
        deep_titles_plot["domain_title"] = deep_titles_plot.apply(
            lambda r: f"{r['domain_filled']} | {r['current_job_title_filled']}", axis=1
        )
        plot_barh(
            deep_titles_plot.rename(columns={"domain_title": "label"}),
            "label",
            "count",
            figures_dir / "16_deep_dive_domain_titles.png",
            "Domain Other/Not specified: top current job titles",
            top_n=20,
        )

    if not region_ns_domain.empty:
        plot_barh(
            region_ns_domain,
            "domain_filled",
            "count",
            figures_dir / "17_deep_dive_region_not_specified_domain.png",
            "Region Not specified: top domains",
            top_n=20,
        )

    if not company_ns_job_titles.empty:
        plot_barh(
            company_ns_job_titles,
            "current_job_title_filled",
            "count",
            figures_dir / "18_deep_dive_company_not_specified_titles.png",
            "Company Not specified: top current job titles",
            top_n=20,
        )

    build_notebook(base / "notebooks" / "mis_users_resume_bot.ipynb")
    build_readme(base, tables)

    readme_text = (base / "README.md").read_text(encoding="utf-8")
    md_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme_text)
    html_links = re.findall(r'<img[^>]*src="([^"]+)"', readme_text)
    links = md_links + html_links
    readme_links_check = pd.DataFrame(
        {
            "link": links,
            "exists": [((base / link).exists()) for link in links],
        }
    )
    save_table(readme_links_check, tables_dir / "readme_links_check.csv")
    tables["readme_links_check"] = readme_links_check

    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate MIS report for users dataset")
    parser.add_argument("--input", default="/mnt/data/prointerview-prod.users.csv", help="Path to source users CSV")
    parser.add_argument("--base-dir", default="analytics/mis_users_resume_bot", help="Base MIS directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input

    if not Path(input_path).exists() and input_path == "/mnt/data/prointerview-prod.users.csv":
        alt = "/Users/k/Downloads/prointerview-prod.users.csv"
        if Path(alt).exists():
            input_path = alt

    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    run(input_path=input_path, base_dir=args.base_dir)
    print(f"MIS generated in {args.base_dir}")


if __name__ == "__main__":
    main()
