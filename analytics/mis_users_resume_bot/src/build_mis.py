from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
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
    company_clean_key,
    company_key_translit,
    experience_bin,
    guess_country,
    hash_user_id,
    infer_role_family,
    leadership_level,
    merge_skills_tools,
    normalize_company,
    normalize_domain,
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

GENERIC_EXCLUDE = {"other", "not specified", "unknown"}


def pct(n: int, d: int) -> float:
    return round((n / d * 100), 1) if d else 0.0


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def shorten(value: object, max_len: int = 44) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def display_short(value: object, max_len: int = 60) -> str:
    text = str(value) if value is not None else ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def wrap_label(value: object, max_len: int = 44, wrap_width: int = 28) -> str:
    text = shorten(value, max_len=max_len)
    return textwrap.fill(text, width=wrap_width)


def filter_generic(df: pd.DataFrame, col: str) -> pd.DataFrame:
    mask = ~df[col].fillna("").astype(str).str.strip().str.lower().isin(GENERIC_EXCLUDE)
    return df[mask].copy()


def is_meaningful_barplot(df: pd.DataFrame, label_col: str, value_col: str) -> bool:
    if df.empty:
        return False
    uniq = df[label_col].fillna("").astype(str).str.strip().replace("", np.nan).dropna().nunique()
    total = float(df[value_col].fillna(0).sum())
    return bool(uniq >= 2 and total >= 2)


def non_empty(value: object) -> bool:
    return clean_text(value) != ""


def infer_lang_from_text(value: object) -> str:
    text = clean_text(value)
    if not text:
        return "en"
    low = text.lower()
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"
    if any(token in low for token in ["навыки", "опыт", "образование", "языки"]):
        return "ru"
    return "en"


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
    out = out.rename(columns={raw_col: f"{field_name}_display_full", norm_col: f"{field_name}_norm"})
    out[f"{field_name}_display_short"] = out[f"{field_name}_display_full"].map(lambda x: display_short(x, max_len=60))
    out[f"{field_name}_display"] = out[f"{field_name}_display_short"]
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


def plot_barh(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    top_n: int,
    max_label_len: int = 36,
    wrap_width: int = 22,
) -> bool:
    if not is_meaningful_barplot(df, label_col, value_col):
        return False
    top = df.head(top_n).copy()
    if not is_meaningful_barplot(top, label_col, value_col):
        return False
    labels = [wrap_label(x, max_len=max_label_len, wrap_width=wrap_width) for x in top[label_col].astype(str)]
    values = top[value_col].astype(float).values

    fig, ax = plt.subplots(figsize=(14.0, max(6.0, 0.55 * len(top) + 2.2)))
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
    fig.subplots_adjust(left=0.41, right=0.97, top=0.92, bottom=0.08)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


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
    if matrix_counts.shape[0] < 2 or matrix_counts.to_numpy().sum() < 2:
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

    if len(data) < 2:
        return pd.DataFrame(columns=["category", "count", "share_%"])

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    colors = sns.color_palette("Set2", n_colors=len(data))
    wedges, _, autotexts = ax.pie(
        data.values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        colors=colors,
        pctdistance=0.76,
    )
    for t in autotexts:
        t.set_fontsize(7)
    ax.legend(
        wedges,
        [shorten(i, 16) for i in data.index],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=7,
    )
    ax.set_title(title, fontsize=9)
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


def build_company_mapping_collisions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["cluster_key", "total_count", "company_norm_list", "top_raw_examples"])

    work = df.copy()
    work["company_raw"] = work["company_raw"].fillna("").astype(str).str.strip()
    work["company_norm"] = work["company_norm"].fillna("Not specified").astype(str)
    work = work[work["company_raw"].ne("")]
    work = work[~work["company_norm"].str.lower().isin({"not specified", "other", "unknown"})]
    work["cluster_key"] = work["company_raw"].map(company_key_translit)
    work = work[work["cluster_key"].ne("")]

    rows: List[Dict[str, object]] = []
    for cluster_key, grp in work.groupby("cluster_key"):
        norm_counts = grp["company_norm"].value_counts()
        if len(norm_counts) <= 1:
            continue

        raw_counts = grp["company_raw"].value_counts().head(6)
        rows.append(
            {
                "cluster_key": cluster_key,
                "total_count": int(len(grp)),
                "company_norm_list": " | ".join([f"{idx} ({int(cnt)})" for idx, cnt in norm_counts.items()]),
                "top_raw_examples": " | ".join([f"{idx} ({int(cnt)})" for idx, cnt in raw_counts.items()]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["cluster_key", "total_count", "company_norm_list", "top_raw_examples"])
    return pd.DataFrame(rows).sort_values(["total_count", "cluster_key"], ascending=[False, True]).reset_index(drop=True)


def months_between(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    if pd.isna(later) or pd.isna(earlier):
        return np.nan
    months = (later.year - earlier.year) * 12 + (later.month - earlier.month)
    if later.day < earlier.day:
        months -= 1
    return float(max(months, 0))


def plot_bar(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    max_label_len: int = 28,
) -> bool:
    if not is_meaningful_barplot(df, label_col, value_col):
        return False
    data = df.copy()
    labels = [display_short(v, max_len=max_label_len) for v in data[label_col].astype(str)]
    values = data[value_col].astype(float).values

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = sns.color_palette("viridis", n_colors=len(data))
    ax.bar(labels, values, color=colors)
    total = float(values.sum())
    for i, val in enumerate(values):
        ax.text(i, val + max(values) * 0.02, f"{int(val)} ({pct(int(val), int(total)):.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Candidates")
    ax.tick_params(axis="x", rotation=20)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_hist(
    series: pd.Series,
    out_path: Path,
    title: str,
    xlabel: str,
    bins: int = 15,
) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.hist(s.values, bins=bins, color="#355C7D", alpha=0.85, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Candidates")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


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


def _img_md(base_dir: Path, rel_path: str, title: str) -> str:
    return f"![{title}]({rel_path})" if (base_dir / rel_path).exists() else ""


def _to_markdown_top(df: pd.DataFrame, cols: List[str], n: int = 3) -> str:
    if df.empty:
        return "(empty)"
    return df[cols].head(n).to_markdown(index=False)


def build_readme(base_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    profile = tables["dataset_profile"]
    validation = tables["validation_summary"]
    diagnostics = tables["not_specified_diagnostics"]
    geo_audit = tables["geo_mapping_audit"]
    region_ns_breakdown = tables["not_specified_deep_dive_region_not_specified_breakdown"]
    company_ns_breakdown = tables["not_specified_deep_dive_company_not_specified_breakdown"]
    region_alt_cols = tables["not_specified_deep_dive_region_alt_columns"]
    employment_summary = tables.get("employment_status_summary", pd.DataFrame())
    employment_domain = tables.get("employment_status_by_domain", pd.DataFrame())
    employment_region = tables.get("employment_status_by_region", pd.DataFrame())
    employment_seniority = tables.get("employment_status_by_seniority", pd.DataFrame())
    cv_language_coverage = tables.get("cv_language_coverage", pd.DataFrame())
    cv_language_dist = tables.get("cv_generation_language_distribution", pd.DataFrame())

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
    region_ns_share = (
        float(region_ns_breakdown.loc[region_ns_breakdown["metric"] == "share_of_dataset_%", "value"].iloc[0])
        if not region_ns_breakdown.empty
        else 0.0
    )
    company_ns_share = (
        float(company_ns_breakdown.loc[company_ns_breakdown["metric"] == "share_of_dataset_%", "value"].iloc[0])
        if not company_ns_breakdown.empty
        else 0.0
    )
    alt_geo_used = int(region_alt_cols["selected_in_fallback"].fillna(False).sum()) if "selected_in_fallback" in region_alt_cols.columns else 0
    no_latex_count = 0
    no_latex_share = 0.0
    cv_lang_mix = ""
    if not cv_language_coverage.empty:
        cov = dict(zip(cv_language_coverage["metric"], cv_language_coverage["value"]))
        no_latex_count = int(float(cov.get("no_latex_count", 0)))
        no_latex_share = float(cov.get("no_latex_share_%", 0.0))
    if not cv_language_dist.empty:
        parts = []
        total_lang = cv_language_dist["count"].sum()
        for _, r in cv_language_dist.iterrows():
            share = pct(int(r["count"]), int(total_lang))
            parts.append(f"{r['cvGenerationLanguage']}: {share:.1f}%")
        cv_lang_mix = ", ".join(parts)
    emp_share = {}
    if not employment_summary.empty:
        for _, row in employment_summary.iterrows():
            emp_share[str(row["employment_status"])] = float(row.get("share_%", 0.0))

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
        f"CV language среди пользователей с cvEnhancedResult: {cv_lang_mix if cv_lang_mix else 'n/a'}; no_latex={no_latex_count} ({no_latex_share:.1f}%).",
        f"Топ tools: {', '.join(top_tools['token_display'].tolist())}.",
        f"Статус занятости: employed {emp_share.get('employed', 0.0):.1f}%, not_employed {emp_share.get('not_employed', 0.0):.1f}%, unknown {emp_share.get('unknown', 0.0):.1f}%.",
    ]

    lines: List[str] = []
    lines.append("# MIS: Users Resume Bot (Candidate Analytics)")
    lines.append("")
    lines.append("## 1) Summary")
    lines.append(f"- Users total: **{int(float(p['users_total']))}**")
    lines.append(f"- Coverage `cvEnhancedResult`: **{float(v['coverage_cvEnhancedResult_%']):.1f}%**")
    lines.append(f"- Coverage `talentCard.jobs`: **{float(v['coverage_talentCard_jobs_%']):.1f}%**")
    lines.append(f"- Coverage `talentCard.overall_skills`: **{float(v['coverage_overall_skills_%']):.1f}%**")
    lines.append(f"- Coverage `talentCard.specialist_category`: **{float(v['coverage_specialist_category_%']):.1f}%**")
    lines.append("")
    lines.append("Top-3 domains (excluding Other/Not specified):")
    lines.append(_to_markdown_top(top_domains.rename(columns={"domain_display_short": "domain"}), ["domain", "count"]))
    lines.append("")
    lines.append("Top-3 regions (excluding Other/Not specified):")
    lines.append(_to_markdown_top(top_regions.rename(columns={"region_display_short": "region"}), ["region", "count"]))
    lines.append("")
    lines.append("Top-3 companies (excluding Other/Not specified):")
    lines.append(_to_markdown_top(top_companies.rename(columns={"company_display_short": "company"}), ["company", "count"]))
    lines.append("")
    lines.append("### Key observations")
    lines.extend([f"- {x}" for x in observations])
    lines.append("")
    lines.append("## 2) Coverage / Parsing validation")
    lines.append(validation.to_markdown(index=False))
    lines.append("")
    lines.append("Columns inventory (real CSV structure + non-null profile + length stats):")
    lines.append("- `outputs/tables/columns_inventory.csv`")
    lines.append("")
    lines.append("## 3) Domains & Geography")
    for rel, title in [
        ("outputs/figures/02_top_domains.png", "Top domains"),
        ("outputs/figures/03_top_regions.png", "Top regions"),
        ("outputs/figures/06_heatmap_domain_region_share.png", "Domain x Region heatmap (row share)"),
        ("outputs/figures/07_stacked_region_domain_top10.png", "Region composition by domains (100% stacked)"),
    ]:
        img = _img_md(base_dir, rel, title)
        if img:
            lines.append(img)
    lines.append("")
    lines.append("## 4) Companies & Seniority")
    for rel, title in [
        ("outputs/figures/04_top_companies.png", "Top companies"),
        ("outputs/figures/05_top_industries_subset.png", "Top industries (subset)"),
        ("outputs/figures/08_stacked_seniority_company_top15.png", "Seniority x Company (100% stacked)"),
        ("outputs/figures/09_stacked_domain_company_top15.png", "Domain x Company (100% stacked)"),
    ]:
        img = _img_md(base_dir, rel, title)
        if img:
            lines.append(img)
    lines.append("")
    lines.append("## 5) Skills & Stack")
    for rel, title in [
        ("outputs/figures/10_top_tools.png", "Top tools"),
        ("outputs/figures/11_top_skills.png", "Top skills"),
        ("outputs/figures/12_heatmap_domain_tools_share.png", "Domain x Tools heatmap (row share)"),
    ]:
        img = _img_md(base_dir, rel, title)
        if img:
            lines.append(img)
    lines.append("")
    lines.append("## 6) Стратификация выборки")
    lines.append("CV language (among users with cvEnhancedResult): `ru/en`; `no_latex` учитывается только как покрытие.")
    if no_latex_count:
        lines.append(f"- no_latex_count: **{no_latex_count}** ({no_latex_share:.1f}% базы)")
    donut_a = (base_dir / "outputs/figures/13_donut_seniority_filled.png").exists()
    donut_b = (base_dir / "outputs/figures/14_donut_cv_generation_language.png").exists()
    if donut_a and donut_b:
        lines.append('<p><img src="outputs/figures/13_donut_seniority_filled.png" width="49%"><img src="outputs/figures/14_donut_cv_generation_language.png" width="49%"></p>')
    else:
        if donut_a:
            lines.append("![Seniority mix](outputs/figures/13_donut_seniority_filled.png)")
        if donut_b:
            lines.append("![CV generation language mix](outputs/figures/14_donut_cv_generation_language.png)")
    img_strata = _img_md(base_dir, "outputs/figures/15_strata_top20.png", "Top-20 strata")
    if img_strata:
        lines.append("")
        lines.append(img_strata)
    lines.append("")
    lines.append("Ключевые таблицы стратификации:")
    lines.extend(
        [
            "- `outputs/tables/strata_top20.csv`",
            "- `outputs/tables/domain_distribution.csv`",
            "- `outputs/tables/role_family_distribution.csv`",
            "- `outputs/tables/seniority_distribution.csv`",
            "- `outputs/tables/experience_bin_distribution.csv`",
            "- `outputs/tables/leadership_distribution.csv`",
            "- `outputs/tables/cv_generation_language_distribution.csv`",
            "- `outputs/tables/cv_language_coverage.csv`",
            "- `outputs/tables/language_audit.csv`",
        ]
    )
    lines.append("")
    lines.append("## 7) Employment status (working vs not working)")
    if not employment_summary.empty:
        lines.append(employment_summary.to_markdown(index=False))
    emp_notes: List[str] = []
    if not employment_domain.empty:
        d_emp = employment_domain[(employment_domain["employment_status"] == "employed") & (employment_domain["row_total"] >= 10)]
        d_not = employment_domain[(employment_domain["employment_status"] == "not_employed") & (employment_domain["row_total"] >= 10)]
        if not d_emp.empty:
            top = d_emp.sort_values("row_share_%", ascending=False).iloc[0]
            emp_notes.append(
                f"Домен с максимальной долей employed: `{top['domain_filled_short']}` ({float(top['row_share_%']):.1f}%)."
            )
        if not d_not.empty:
            top = d_not.sort_values("row_share_%", ascending=False).iloc[0]
            emp_notes.append(
                f"Домен с максимальной долей not_employed: `{top['domain_filled_short']}` ({float(top['row_share_%']):.1f}%)."
            )
    if not employment_region.empty:
        r_not = employment_region[(employment_region["employment_status"] == "not_employed") & (employment_region["row_total"] >= 8)]
        if not r_not.empty:
            top = r_not.sort_values("row_share_%", ascending=False).iloc[0]
            emp_notes.append(
                f"Регион с максимальной долей not_employed: `{top['region_norm_short']}` ({float(top['row_share_%']):.1f}%)."
            )
    if not employment_seniority.empty:
        s_not = employment_seniority[
            (employment_seniority["employment_status"] == "not_employed")
            & (employment_seniority["row_total"] >= 8)
            & (~employment_seniority["seniority_filled"].astype(str).str.lower().isin({"not specified", "unknown"}))
        ]
        if not s_not.empty:
            top = s_not.sort_values("row_share_%", ascending=False).iloc[0]
            emp_notes.append(
                f"Сеньорность с максимальной долей not_employed: `{top['seniority_filled_short']}` ({float(top['row_share_%']):.1f}%)."
            )
    if emp_notes:
        lines.append("")
        lines.append("Ключевые наблюдения:")
        lines.extend([f"- {x}" for x in emp_notes])
    lines.append("")
    for rel, title in [
        ("outputs/figures/16_employment_status_overall.png", "Employment status overall"),
        ("outputs/figures/17_employment_status_by_domain.png", "Employment status by domain (100% stacked)"),
        ("outputs/figures/18_months_since_last_end_hist.png", "Months since last end-date (not employed)"),
        ("outputs/figures/19_not_employed_top_last_companies.png", "Not employed: top last companies"),
        ("outputs/figures/20_not_employed_top_last_titles.png", "Not employed: top last titles"),
        ("outputs/figures/21_not_employed_history_top_companies.png", "Not employed: historical top companies"),
    ]:
        img = _img_md(base_dir, rel, title)
        if img:
            lines.append(img)
    lines.append("")
    lines.append("Таблицы employment status:")
    lines.extend(
        [
            "- `outputs/tables/employment_status_summary.csv`",
            "- `outputs/tables/employment_status_by_domain.csv`",
            "- `outputs/tables/employment_status_by_region.csv`",
            "- `outputs/tables/employment_status_by_seniority.csv`",
            "- `outputs/tables/not_employed_top_last_companies.csv`",
            "- `outputs/tables/not_employed_top_last_titles.csv`",
            "- `outputs/tables/not_employed_months_since_last_end.csv`",
            "- `outputs/tables/not_employed_history_top_companies.csv`",
            "- `outputs/tables/not_employed_history_top_titles.csv`",
        ]
    )
    lines.append("")
    lines.append("## 8) Not specified research")
    lines.append(diagnostics.to_markdown(index=False))
    lines.append("")
    lines.append("Пустоты уменьшались по fallback-цепочкам:")
    lines.append("- `region_filled`: `latex_expheader -> talentCard -> latex_header -> alt_geo_columns`")
    lines.append("- `seniority_filled`: `talentCard -> inferred_job_title -> inferred_header_role`")
    lines.append("- `domain_filled`: `talentCard.specialist_category -> inferred role family`")
    lines.append("")
    lines.append("Таблицы исследования Not specified:")
    lines.extend(
        [
            "- `outputs/tables/not_specified_deep_dive_summary.csv`",
            "- `outputs/tables/not_specified_deep_dive_region_not_specified_breakdown.csv`",
            "- `outputs/tables/not_specified_deep_dive_region_not_specified_domain.csv`",
            "- `outputs/tables/not_specified_deep_dive_region_not_specified_seniority.csv`",
            "- `outputs/tables/not_specified_deep_dive_region_alt_columns.csv`",
            "- `outputs/tables/not_specified_deep_dive_company_not_specified_breakdown.csv`",
            "- `outputs/tables/not_specified_deep_dive_company_not_specified_job_titles.csv`",
            "- `outputs/tables/not_specified_deep_dive_company_not_specified_region.csv`",
        ]
    )
    lines.append("")
    lines.append("## 9) Domain Other")
    lines.append("Исследование домена `Other` вынесено в отдельный отчёт: `REPORT_OTHERS.md`.")
    lines.append("")
    lines.append("## 10) Appendix")
    lines.append("Артефакты:")
    lines.extend(
        [
            "- Figures: `outputs/figures/*.png`",
            "- Tables: `outputs/tables/*.csv`",
            "- Other report: `REPORT_OTHERS.md` + `outputs/others/*`",
            "- Notebook: `notebooks/mis_users_resume_bot.ipynb`",
            "- Geo mapping audit: `outputs/tables/geo_mapping_audit.csv`",
            "- Geo mapping top-50: `outputs/tables/geo_mapping_top50.csv`",
            "- Company mapping collisions: `outputs/tables/company_mapping_collisions.csv`",
        ]
    )
    lines.append("")
    lines.append("How to reproduce:")
    lines.append("```bash")
    lines.append("python analytics/mis_users_resume_bot/src/build_mis.py \\")
    lines.append("  --input /mnt/data/prointerview-prod.users.csv \\")
    lines.append("  --base-dir analytics/mis_users_resume_bot")
    lines.append("```")
    lines.append("")

    (base_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _other_bucket(text: str) -> str:
    t = canonical_text(text)
    rules = [
        ("Data/Analytics", [r"\bdata\b", r"analyt", r"\bbi\b", r"ml", r"ai", r"sql", r"python"]),
        ("SWE/IT", [r"engineer", r"developer", r"devops", r"sre", r"backend", r"frontend", r"fullstack", r"it"]),
        ("Product/Project", [r"product", r"project", r"program", r"scrum", r"pm\b", r"delivery"]),
        ("Design", [r"design", r"ux", r"ui", r"figma"]),
        ("Sales/Marketing", [r"sales", r"marketing", r"growth", r"crm"]),
        ("HR", [r"hr", r"recruit", r"talent", r"кадр"]),
        ("Finance/Legal", [r"finance", r"audit", r"account", r"legal", r"юрист", r"финанс"]),
        ("Ops", [r"operations", r"logistic", r"supply", r"admin", r"операц", r"админ"]),
        ("Education/Research", [r"research", r"education", r"teacher", r"lect", r"науч", r"образован"]),
        ("Healthcare", [r"health", r"medical", r"clinic", r"мед"]),
    ]
    for label, patterns in rules:
        if any(re.search(p, t) for p in patterns):
            return label
    return "Other-unknown"


def build_others_report(base_dir: Path, others_tables: Dict[str, pd.DataFrame], others_stats: Dict[str, float]) -> None:
    def md(df_name: str, cols: List[str], n: int) -> str:
        frame = others_tables.get(df_name, pd.DataFrame())
        if frame.empty:
            return "(empty)"
        return frame[cols].head(n).to_markdown(index=False)

    lines: List[str] = []
    lines.append("# REPORT_OTHERS: Domain Other Deep Dive")
    lines.append("")
    lines.append("## 1) Size")
    lines.append(f"- Other users: **{int(others_stats.get('other_users', 0))}**")
    lines.append(f"- Share of all users: **{others_stats.get('share_all_pct', 0.0):.1f}%**")
    lines.append(f"- Other users with `cvEnhancedResult`: **{int(others_stats.get('other_with_cv_count', 0))}**")
    lines.append(f"- Share among users with `cvEnhancedResult`: **{others_stats.get('share_among_cv_pct', 0.0):.1f}%**")
    lines.append("")
    lines.append("## 2) Coverage")
    lines.append(md("coverage", ["metric", "value"], 20))
    lines.append("")
    lines.append("## 3) Moved out of Other by remapping")
    lines.append(md("moved_out_of_other_by_remapping", ["old_label_keywords", "new_domain", "count"], 50))
    lines.append("")
    lines.append("## 4) Who Are These Candidates")
    lines.append("Top-30 titles across all jobs history:")
    lines.append(md("titles_all_history", ["job_title", "count", "share_%"], 30))
    lines.append("")
    lines.append("Top-20 current titles:")
    lines.append(md("current_titles", ["current_job_title_filled", "count", "share_%"], 20))
    lines.append("")
    lines.append("Top-30 companies across all jobs history:")
    lines.append(md("companies_all_history", ["company", "count", "share_%"], 30))
    lines.append("")
    lines.append("Most frequent title keywords:")
    lines.append(md("title_keywords", ["title_keyword", "count", "share_%"], 30))
    lines.append("")
    lines.append("## 5) Skills & Stack")
    lines.append("Top tools:")
    lines.append(md("tools_top", ["tool", "count", "share_%"], 30))
    lines.append("")
    lines.append("Top skills:")
    lines.append(md("skills_top", ["skill", "count", "share_%"], 30))
    lines.append("")
    lines.append("## 6) Geography & Seniority")
    lines.append("Top regions:")
    lines.append(md("regions_top", ["region_norm", "count", "share_%"], 20))
    lines.append("")
    lines.append("Seniority:")
    lines.append(md("seniority_top", ["seniority_filled", "count", "share_%"], 20))
    lines.append("")
    lines.append("## 7) Rule-Based Buckets")
    lines.append(md("bucket_distribution", ["bucket", "count", "share_%"], 20))
    lines.append("")
    lines.append("## 8) Figures")
    for rel, title in [
        ("outputs/others/figures/01_other_titles_history_top30.png", "Top titles across all jobs"),
        ("outputs/others/figures/02_other_companies_history_top30.png", "Top companies across all jobs"),
        ("outputs/others/figures/03_other_tools_top30.png", "Top tools in Other"),
        ("outputs/others/figures/04_other_bucket_distribution.png", "Rule-based buckets"),
    ]:
        img = _img_md(base_dir, rel, title)
        if img:
            lines.append(img)
    lines.append("")
    lines.append("## 9) Appendix")
    lines.append("- Tables: `outputs/others/tables/*.csv`")
    lines.append("- Figures: `outputs/others/figures/*.png`")
    lines.append("")

    (base_dir / "REPORT_OTHERS.md").write_text("\n".join(lines), encoding="utf-8")


def run(input_path: str, base_dir: str) -> Dict[str, pd.DataFrame]:
    base = Path(base_dir)
    tables_dir = base / "outputs" / "tables"
    figures_dir = base / "outputs" / "figures"
    others_tables_dir = base / "outputs" / "others" / "tables"
    others_figures_dir = base / "outputs" / "others" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    others_tables_dir.mkdir(parents=True, exist_ok=True)
    others_figures_dir.mkdir(parents=True, exist_ok=True)

    # Prevent stale artifacts from previous report versions.
    for old_csv in tables_dir.glob("*.csv"):
        old_csv.unlink()
    for old_png in figures_dir.glob("*.png"):
        old_png.unlink()
    for old_csv in others_tables_dir.glob("*.csv"):
        old_csv.unlink()
    for old_png in others_figures_dir.glob("*.png"):
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
    users_enriched["has_latex"] = users_enriched["cvEnhancedResult_present"].fillna(False).astype(bool)

    # User-level title corpus from all jobs (talentCard + LaTeX).
    title_parts: List[pd.DataFrame] = []
    if not jobs_talent.empty:
        title_parts.append(jobs_talent[["user_hash", "job_title"]].copy())
    if not jobs_latex.empty:
        title_parts.append(jobs_latex[["user_hash", "job_title"]].copy())
    if title_parts:
        titles_all = pd.concat(title_parts, ignore_index=True)
        titles_all["job_title"] = titles_all["job_title"].fillna("").astype(str).str.strip()
        titles_all = titles_all[titles_all["job_title"].ne("")]
        all_titles_by_user = (
            titles_all.groupby("user_hash")["job_title"]
            .apply(lambda x: " | ".join(pd.Series(x).drop_duplicates().astype(str).tolist()[:25]))
            .reset_index(name="all_titles_text")
        )
        users_enriched = users_enriched.merge(all_titles_by_user, on="user_hash", how="left")
    else:
        users_enriched["all_titles_text"] = ""
    users_enriched["all_titles_text"] = users_enriched["all_titles_text"].fillna("")

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
        domain_talent = normalize_domain(row.get("talentCard.specialist_category", ""))

        infer_text = " ".join(
            [
                clean_text(row.get("selectedPosition", "")),
                clean_text(row.get("current_job_title_filled", "")),
                clean_text(row.get("all_titles_text", "")),
                clean_text(row.get("header_role_latex", "")),
            ]
        )
        if clean_text(infer_text):
            inferred = normalize_domain(
                infer_role_family(row.get("selectedPosition", ""), infer_text, row.get("header_role_latex", ""))
            )
            if inferred not in {"Other", "Not specified"}:
                return inferred, "inferred"
            if domain_talent not in {"Other", "Not specified"}:
                return domain_talent, "talentCard"
            if inferred == "Other" or domain_talent == "Other":
                return "Other", "inferred"
            return "Not specified", "not_specified"

        if domain_talent not in {"Other", "Not specified"}:
            return domain_talent, "talentCard"
        if domain_talent == "Other":
            return "Other", "talentCard"
        return "Not specified", "not_specified"

    domain_rows = users_enriched.apply(infer_domain, axis=1)
    users_enriched["domain_filled"] = domain_rows.map(lambda x: normalize_domain(x[0]))
    users_enriched["domain_source"] = domain_rows.map(lambda x: x[1])

    users_enriched["industry_filled"] = users_enriched["current_industry_talentCard"].fillna("").astype(str).str.strip()
    users_enriched["industry_source"] = np.where(users_enriched["industry_filled"].ne(""), "talentCard", "not_specified")
    users_enriched.loc[users_enriched["industry_filled"].eq(""), "industry_filled"] = "Not specified"

    users_enriched["region_norm"] = users_enriched["region_filled"].map(normalize_region)
    users_enriched["company_norm"] = users_enriched["company_filled"].map(normalize_company)
    users_enriched["industry_norm"] = users_enriched["industry_filled"].map(normalize_industry)
    users_enriched["country_guess"] = users_enriched["region_norm"].map(guess_country)

    users_enriched["role_family"] = users_enriched.apply(
        lambda r: normalize_domain(
            infer_role_family(
                r.get("selectedPosition", ""),
                " ".join([clean_text(r.get("current_job_title_filled", "")), clean_text(r.get("all_titles_text", ""))]),
                r.get("header_role_latex", ""),
            )
        ),
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

    domain_distribution = users_enriched["domain_filled"].fillna("Not specified").astype(str).value_counts().rename_axis("domain_filled").reset_index(name="count")
    domain_distribution["domain_display_full"] = domain_distribution["domain_filled"].astype(str)
    domain_distribution["domain_display_short"] = domain_distribution["domain_display_full"].map(lambda x: display_short(x, max_len=60))
    domain_distribution["domain_display"] = domain_distribution["domain_display_short"]
    role_family_distribution = users_enriched["role_family"].fillna("Other").astype(str).value_counts().rename_axis("role_family").reset_index(name="count")

    seniority_order = ["Junior", "Middle", "Senior", "Lead", "C-level", "Not specified"]
    seniority_distribution = users_enriched["seniority_filled"].fillna("Not specified").astype(str).value_counts().rename_axis("seniority_filled").reset_index(name="count")
    seniority_distribution["_ord"] = seniority_distribution["seniority_filled"].map({k: i for i, k in enumerate(seniority_order)}).fillna(999)
    seniority_distribution = seniority_distribution.sort_values(["_ord", "count"], ascending=[True, False]).drop(columns=["_ord"])

    experience_distribution = users_enriched["experience_bin"].fillna("Unknown").astype(str).value_counts().rename_axis("experience_bin").reset_index(name="count")
    leadership_distribution = users_enriched["leadership_level"].fillna("Low").astype(str).value_counts().rename_axis("leadership_level").reset_index(name="count")

    users_enriched["cv_language_filled"] = np.where(
        users_enriched["has_latex"],
        users_enriched.get("cvEnhancedResult", "").map(infer_lang_from_text),
        "no_latex",
    )
    users_enriched["language_source"] = np.where(users_enriched["has_latex"], "inferred_from_latex", "no_latex")
    users_enriched["raw_cvGenerationLanguage"] = (
        users_enriched.get("cvGenerationLanguage", pd.Series(index=users_enriched.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cv_generation_language_distribution = (
        users_enriched[users_enriched["has_latex"]]
        .groupby("cv_language_filled")
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"cv_language_filled": "cvGenerationLanguage"})
        .sort_values("count", ascending=False)
    )
    cv_language_coverage = pd.DataFrame(
        {
            "metric": ["has_latex_count", "no_latex_count", "has_latex_share_%", "no_latex_share_%"],
            "value": [
                int(users_enriched["has_latex"].sum()),
                int((~users_enriched["has_latex"]).sum()),
                round(users_enriched["has_latex"].mean() * 100, 1),
                round((~users_enriched["has_latex"]).mean() * 100, 1),
            ],
        }
    )
    language_audit = (
        users_enriched.groupby(["has_latex", "raw_cvGenerationLanguage", "cv_language_filled", "language_source"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )

    region_distribution = distribution_with_display(users_enriched, "region_filled", "region_norm", "region")
    company_distribution = distribution_with_display(users_enriched, "company_filled", "company_norm", "company")
    industry_distribution = distribution_with_display(users_enriched, "industry_filled", "industry_norm", "industry")

    domain_distribution_plot = filter_generic(domain_distribution, "domain_filled")
    domain_distribution_plot = domain_distribution_plot.sort_values("count", ascending=False).reset_index(drop=True)
    region_distribution_plot = filter_generic(region_distribution, "region_norm")
    region_distribution_plot = region_distribution_plot.sort_values("count", ascending=False).reset_index(drop=True)
    company_distribution_plot = filter_generic(company_distribution, "company_norm")
    company_distribution_plot = company_distribution_plot.sort_values("count", ascending=False).reset_index(drop=True)
    seniority_distribution_plot = filter_generic(seniority_distribution, "seniority_filled")
    seniority_distribution_plot = seniority_distribution_plot.sort_values("count", ascending=False).reset_index(drop=True)

    industry_subset = industry_distribution[industry_distribution["industry_norm"] != "Not specified"].copy()

    country_distribution = users_enriched["country_guess"].fillna("Unknown").astype(str).value_counts().rename_axis("country").reset_index(name="count")
    english_level_distribution = users_enriched["english_level"].fillna("Unknown").astype(str).value_counts().rename_axis("english_level").reset_index(name="count")
    degree_level_distribution = users_enriched["degree_level"].fillna("Unknown").astype(str).value_counts().rename_axis("degree_level").reset_index(name="count")

    # Skills/stack outputs.
    skills_top = aggregate_tokens(users_enriched, "skills_list")
    tools_top = aggregate_tokens(users_enriched, "tools_list")

    # Domain x Region (heatmap row-share) and stacked by regions.
    domain_analysis = users_enriched[
        ~users_enriched["domain_filled"].fillna("Not specified").astype(str).str.lower().isin(GENERIC_EXCLUDE)
    ].copy()
    domain_top_values = domain_distribution_plot.head(10)["domain_filled"].astype(str).tolist()
    region_top_values = region_distribution_plot.head(15)["region_norm"].astype(str).tolist()
    domain_analysis = domain_analysis[
        domain_analysis["domain_filled"].astype(str).isin(domain_top_values)
        & domain_analysis["region_norm"].astype(str).isin(region_top_values)
    ].copy()

    domain_region_counts = pd.crosstab(domain_analysis["domain_filled"], domain_analysis["region_norm"])
    domain_region_share = row_normalize_percent(domain_region_counts)

    region_domain_base = users_enriched[
        ~users_enriched["region_norm"].fillna("Not specified").astype(str).str.lower().isin(GENERIC_EXCLUDE)
        & users_enriched["domain_filled"].astype(str).isin(domain_top_values)
    ].copy()
    region_top10_values = region_distribution_plot.head(10)["region_norm"].astype(str).tolist()
    region_domain_base = region_domain_base[region_domain_base["region_norm"].astype(str).isin(region_top10_values)].copy()
    region_domain_counts = pd.crosstab(region_domain_base["region_norm"], region_domain_base["domain_filled"])

    # Seniority x Company and Domain x Company as 100% stacked bars.
    company_analysis = users_enriched[
        ~users_enriched["company_norm"].fillna("Not specified").astype(str).str.lower().isin(GENERIC_EXCLUDE)
    ].copy()
    top_company_values = company_distribution_plot.head(15)["company_norm"].astype(str).tolist()
    company_analysis = company_analysis[company_analysis["company_norm"].astype(str).isin(top_company_values)].copy()

    seniority_small = company_analysis["seniority_filled"].fillna("Not specified")
    seniority_small = seniority_small.where(~seniority_small.astype(str).str.lower().isin(GENERIC_EXCLUDE), np.nan)
    seniority_company_counts = pd.crosstab(company_analysis["company_norm"], seniority_small).dropna(axis=1, how="all")

    domain_company_base = company_analysis[
        company_analysis["domain_filled"].astype(str).isin(domain_top_values)
    ].copy()
    domain_company_counts = pd.crosstab(domain_company_base["company_norm"], domain_company_base["domain_filled"])

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

    region_top10_values = region_distribution_plot.head(10)["region_norm"].tolist()
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

    # Company mapping collision audit.
    company_audit_parts: List[pd.DataFrame] = []
    company_audit_parts.append(
        users_enriched[["company_filled", "company_norm"]]
        .rename(columns={"company_filled": "company_raw"})
        .assign(source="current_filled")
    )
    if not jobs_talent.empty:
        company_audit_parts.append(
            jobs_talent[["company"]]
            .rename(columns={"company": "company_raw"})
            .assign(company_norm=lambda d: d["company_raw"].map(normalize_company), source="jobs_talentCard")
        )
    if not jobs_latex.empty:
        company_audit_parts.append(
            jobs_latex[["company"]]
            .rename(columns={"company": "company_raw"})
            .assign(company_norm=lambda d: d["company_raw"].map(normalize_company), source="jobs_latex")
        )
    company_audit = pd.concat(company_audit_parts, ignore_index=True)
    company_mapping_collisions = build_company_mapping_collisions(company_audit)

    # Employment status (working vs not working) from combined jobs history.
    analysis_date_candidates: List[pd.Timestamp] = []
    if "updatedAt" in users.columns and users["updatedAt"].notna().any():
        analysis_date_candidates.append(users["updatedAt"].dropna().max())
    if "createdAt" in users.columns and users["createdAt"].notna().any():
        analysis_date_candidates.append(users["createdAt"].dropna().max())
    analysis_date = max(analysis_date_candidates) if analysis_date_candidates else pd.Timestamp.utcnow()
    if isinstance(analysis_date, pd.Timestamp) and analysis_date.tzinfo is not None:
        analysis_date = analysis_date.tz_convert(None)

    jobs_all_parts: List[pd.DataFrame] = []
    if not jobs_talent.empty:
        jt = jobs_talent.copy()
        jt["source"] = "talentCard"
        jt = jt.rename(columns={"employment_period": "period_raw", "period_parse_ok": "parse_ok"})
        jobs_all_parts.append(
            jt[
                [
                    "user_hash",
                    "source",
                    "company",
                    "region",
                    "job_title",
                    "period_raw",
                    "start_date",
                    "end_date",
                    "is_present",
                    "parse_ok",
                ]
            ]
        )
    if not jobs_latex.empty:
        jl = jobs_latex.copy()
        jl["source"] = jl["source"].fillna("latex")
        jl = jl.rename(columns={"employment_period": "period_raw", "period_parse_ok": "parse_ok"})
        jobs_all_parts.append(
            jl[
                [
                    "user_hash",
                    "source",
                    "company",
                    "region",
                    "job_title",
                    "period_raw",
                    "start_date",
                    "end_date",
                    "is_present",
                    "parse_ok",
                ]
            ]
        )

    if jobs_all_parts:
        jobs_long_all = pd.concat(jobs_all_parts, ignore_index=True)
        jobs_long_all["company_display_full"] = jobs_long_all["company"].fillna("").astype(str).str.strip()
        jobs_long_all["company_display_short"] = jobs_long_all["company_display_full"].map(lambda x: display_short(x, max_len=60))
        jobs_long_all["company_norm"] = jobs_long_all["company_display_full"].map(normalize_company)
        jobs_long_all["region_display_full"] = jobs_long_all["region"].fillna("").astype(str).str.strip()
        jobs_long_all["region_display_short"] = jobs_long_all["region_display_full"].map(lambda x: display_short(x, max_len=60))
        jobs_long_all["region_norm"] = jobs_long_all["region_display_full"].map(normalize_region)
        jobs_long_all["job_title"] = jobs_long_all["job_title"].fillna("").astype(str).str.strip()
        jobs_long_all["job_title_short"] = jobs_long_all["job_title"].map(lambda x: display_short(x, max_len=60))
        jobs_long_all["period_raw"] = jobs_long_all["period_raw"].fillna("").astype(str).str.strip()
        jobs_long_all["parse_ok"] = jobs_long_all["parse_ok"].fillna(False).astype(bool)
        jobs_long_all["is_present"] = jobs_long_all["is_present"].fillna(False).astype(bool)
        jobs_long_all = jobs_long_all[
            jobs_long_all[["company_display_full", "job_title", "period_raw"]].astype(str).apply(lambda r: any(v.strip() for v in r), axis=1)
        ].copy()
    else:
        jobs_long_all = pd.DataFrame(
            columns=[
                "user_hash",
                "source",
                "company_display_full",
                "company_display_short",
                "company_norm",
                "region_display_full",
                "region_display_short",
                "region_norm",
                "job_title",
                "job_title_short",
                "period_raw",
                "start_date",
                "end_date",
                "is_present",
                "parse_ok",
            ]
        )

    status_rows: List[Dict[str, object]] = []
    if not jobs_long_all.empty:
        for user_hash, grp in jobs_long_all.groupby("user_hash"):
            g = grp.copy()
            employed = bool(g["is_present"].fillna(False).any())
            valid_closed = g[g["parse_ok"] & g["end_date"].notna()].copy()

            status = "unknown"
            last_end = pd.NaT
            months_since_last_end = np.nan
            last_company_norm = "Not specified"
            last_company_display_full = "Not specified"
            last_job_title = "Not specified"

            if employed:
                status = "employed"
            elif not valid_closed.empty:
                last_end = valid_closed["end_date"].max()
                if pd.notna(last_end) and last_end < analysis_date:
                    status = "not_employed"
                    months_since_last_end = months_between(analysis_date, last_end)
                else:
                    status = "unknown"

                last_row = valid_closed.sort_values(["end_date", "start_date"], ascending=[False, False]).iloc[0]
                last_company_norm = clean_text(last_row.get("company_norm", "")) or "Not specified"
                last_company_display_full = clean_text(last_row.get("company_display_full", "")) or "Not specified"
                last_job_title = clean_text(last_row.get("job_title", "")) or "Not specified"

            status_rows.append(
                {
                    "user_hash": user_hash,
                    "employment_status": status,
                    "last_end_date": last_end,
                    "months_since_last_end": months_since_last_end,
                    "last_company_norm": last_company_norm,
                    "last_company_display_full": last_company_display_full,
                    "last_company_display_short": display_short(last_company_display_full, max_len=60),
                    "last_job_title": last_job_title,
                    "last_job_title_short": display_short(last_job_title, max_len=60),
                }
            )

    employment_user = users_enriched[["user_hash", "domain_filled", "region_norm", "seniority_filled"]].merge(
        pd.DataFrame(status_rows), on="user_hash", how="left"
    )
    employment_user["employment_status"] = employment_user["employment_status"].fillna("unknown")
    employment_user["months_since_last_end"] = pd.to_numeric(employment_user["months_since_last_end"], errors="coerce")
    employment_user["last_company_norm"] = employment_user["last_company_norm"].fillna("Not specified")
    employment_user["last_company_display_full"] = employment_user["last_company_display_full"].fillna("Not specified")
    employment_user["last_company_display_short"] = employment_user["last_company_display_short"].fillna("Not specified")
    employment_user["last_job_title"] = employment_user["last_job_title"].fillna("Not specified")
    employment_user["last_job_title_short"] = employment_user["last_job_title_short"].fillna("Not specified")

    status_order = ["employed", "not_employed", "unknown"]
    employment_status_summary = (
        employment_user["employment_status"]
        .value_counts()
        .reindex(status_order, fill_value=0)
        .rename_axis("employment_status")
        .reset_index(name="count")
    )
    employment_status_summary["share_%"] = (employment_status_summary["count"] / max(len(employment_user), 1) * 100).round(1)

    def cross_status(segment: pd.Series, segment_name: str, top_n: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        seg = segment.fillna("Not specified").astype(str)
        if top_n is not None:
            seg_top = seg[~seg.str.lower().isin(GENERIC_EXCLUDE)].value_counts().head(top_n).index.tolist()
            seg = seg.where(seg.isin(seg_top), np.nan)
        base = pd.DataFrame({"segment": seg, "employment_status": employment_user["employment_status"]}).dropna(subset=["segment"]).copy()
        counts = pd.crosstab(base["segment"], base["employment_status"])
        for st in status_order:
            if st not in counts.columns:
                counts[st] = 0
        counts = counts[status_order]
        shares = row_normalize_percent(counts)
        rows: List[Dict[str, object]] = []
        for idx in counts.index:
            row_total = int(counts.loc[idx].sum())
            for st in status_order:
                rows.append(
                    {
                        segment_name: idx,
                        f"{segment_name}_short": display_short(idx, max_len=60),
                        "employment_status": st,
                        "count": int(counts.loc[idx, st]),
                        "row_share_%": round(float(shares.loc[idx, st]), 1),
                        "row_total": row_total,
                    }
                )
        return pd.DataFrame(rows), counts

    employment_status_by_domain, domain_status_counts = cross_status(employment_user["domain_filled"], "domain_filled", top_n=10)
    employment_status_by_region, region_status_counts = cross_status(employment_user["region_norm"], "region_norm", top_n=15)
    employment_status_by_seniority, seniority_status_counts = cross_status(employment_user["seniority_filled"], "seniority_filled", top_n=None)

    not_employed_users = employment_user[employment_user["employment_status"] == "not_employed"].copy()
    not_employed_months_since_last_end = not_employed_users[
        ["user_hash", "last_end_date", "months_since_last_end", "last_company_norm", "last_job_title"]
    ].copy()

    not_employed_last_companies = (
        not_employed_users[~not_employed_users["last_company_norm"].astype(str).str.lower().isin(GENERIC_EXCLUDE)]
        .groupby("last_company_norm")
        .agg(
            count=("user_hash", "count"),
            company_display_full=("last_company_display_full", lambda x: x.value_counts().index[0] if len(x) else ""),
        )
        .reset_index()
        .rename(columns={"last_company_norm": "company_norm"})
        .sort_values("count", ascending=False)
        .head(30)
    )
    if not not_employed_last_companies.empty:
        not_employed_last_companies["share_%"] = (not_employed_last_companies["count"] / not_employed_last_companies["count"].sum() * 100).round(1)
        not_employed_last_companies["company_display_short"] = not_employed_last_companies["company_display_full"].map(lambda x: display_short(x, max_len=60))
    else:
        not_employed_last_companies = pd.DataFrame(columns=["company_norm", "count", "company_display_full", "company_display_short", "share_%"])

    not_employed_last_titles = (
        not_employed_users[not_employed_users["last_job_title"].astype(str).str.strip().ne("")]
        .groupby("last_job_title")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
        .head(30)
    )
    if not not_employed_last_titles.empty:
        not_employed_last_titles["share_%"] = (not_employed_last_titles["count"] / not_employed_last_titles["count"].sum() * 100).round(1)
        not_employed_last_titles["job_title_short"] = not_employed_last_titles["last_job_title"].map(lambda x: display_short(x, max_len=60))
    else:
        not_employed_last_titles = pd.DataFrame(columns=["last_job_title", "count", "share_%", "job_title_short"])

    not_employed_history_jobs = jobs_long_all[jobs_long_all["user_hash"].isin(set(not_employed_users["user_hash"]))].copy()
    not_employed_history_companies = (
        not_employed_history_jobs[~not_employed_history_jobs["company_norm"].astype(str).str.lower().isin(GENERIC_EXCLUDE)]
        .groupby("company_norm")
        .agg(
            count=("user_hash", "count"),
            company_display_full=("company_display_full", lambda x: x.value_counts().index[0] if len(x) else ""),
        )
        .reset_index()
        .sort_values("count", ascending=False)
        .head(30)
    )
    if not not_employed_history_companies.empty:
        not_employed_history_companies["share_%"] = (
            not_employed_history_companies["count"] / not_employed_history_companies["count"].sum() * 100
        ).round(1)
        not_employed_history_companies["company_display_short"] = not_employed_history_companies["company_display_full"].map(
            lambda x: display_short(x, max_len=60)
        )
    else:
        not_employed_history_companies = pd.DataFrame(columns=["company_norm", "count", "company_display_full", "company_display_short", "share_%"])

    not_employed_history_titles = top_counts(not_employed_history_jobs["job_title"], 30, "job_title")
    if not not_employed_history_titles.empty:
        not_employed_history_titles["job_title_short"] = not_employed_history_titles["job_title"].map(lambda x: display_short(x, max_len=60))

    geo_mapping_audit = (
        users_enriched.groupby(["region_filled", "region_norm"]).size().reset_index(name="count").rename(columns={"region_filled": "raw_region"}).sort_values("count", ascending=False)
    )
    geo_mapping_top50 = geo_mapping_audit.head(50).copy()

    region_mappings = build_mapping_table(users_enriched["region_filled"], users_enriched["region_norm"], "region")
    company_mappings = build_mapping_table(users_enriched["company_filled"], users_enriched["company_norm"], "company")
    industry_mappings = build_mapping_table(users_enriched["industry_filled"], users_enriched["industry_norm"], "industry")

    top_domains_full = domain_distribution_plot[["domain_display_full", "count"]].head(10).copy()
    top_regions_full = region_distribution_plot[["region_display_full", "count"]].head(15).copy()
    top_companies_full = company_distribution_plot[["company_display_full", "count"]].head(15).copy()
    top_seniority_full = seniority_distribution_plot[["seniority_filled", "count"]].head(10).copy()

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
        "domain_distribution": domain_distribution,
        "domain_distribution_plot": domain_distribution_plot,
        "top_domains_full": top_domains_full,
        "role_family_distribution": role_family_distribution,
        "seniority_distribution": seniority_distribution,
        "seniority_distribution_plot": seniority_distribution_plot,
        "top_seniority_full": top_seniority_full,
        "experience_bin_distribution": experience_distribution,
        "leadership_distribution": leadership_distribution,
        "cv_generation_language_distribution": cv_generation_language_distribution,
        "cv_language_coverage": cv_language_coverage,
        "language_audit": language_audit,
        "region_distribution": region_distribution,
        "region_distribution_plot": region_distribution_plot,
        "top_regions_full": top_regions_full,
        "country_distribution": country_distribution,
        "company_distribution": company_distribution,
        "company_distribution_plot": company_distribution_plot,
        "top_companies_full": top_companies_full,
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
        "company_mapping_collisions": company_mapping_collisions,
        "industry_mappings": industry_mappings,
        "geo_mapping_audit": geo_mapping_audit,
        "geo_mapping_top50": geo_mapping_top50,
        "jobs_long_all": jobs_long_all,
        "employment_status_summary": employment_status_summary,
        "employment_status_by_domain": employment_status_by_domain,
        "employment_status_by_region": employment_status_by_region,
        "employment_status_by_seniority": employment_status_by_seniority,
        "not_employed_months_since_last_end": not_employed_months_since_last_end,
        "not_employed_top_last_companies": not_employed_last_companies,
        "not_employed_top_last_titles": not_employed_last_titles,
        "not_employed_history_top_companies": not_employed_history_companies,
        "not_employed_history_top_titles": not_employed_history_titles,
        "strata_top20": strata_top20,
        "not_specified_deep_dive_summary": not_specified_deep_dive_summary,
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
    plot_barh(
        domain_distribution_plot,
        "domain_display_short",
        "count",
        figures_dir / "02_top_domains.png",
        "Top domains (excluding Other/Not specified)",
        top_n=10,
        max_label_len=34,
        wrap_width=22,
    )
    plot_barh(
        region_distribution_plot.rename(columns={"region_display_short": "region"}),
        "region",
        "count",
        figures_dir / "03_top_regions.png",
        "Top regions (excluding Other/Not specified)",
        top_n=15,
        max_label_len=34,
        wrap_width=22,
    )
    plot_barh(
        company_distribution_plot.rename(columns={"company_display_short": "company"}),
        "company",
        "count",
        figures_dir / "04_top_companies.png",
        "Top companies (excluding Other/Not specified)",
        top_n=15,
        max_label_len=34,
        wrap_width=22,
    )
    if not industry_subset.empty:
        plot_barh(
            industry_subset.rename(columns={"industry_display": "industry"}),
            "industry",
            "count",
            figures_dir / "05_top_industries_subset.png",
            "Top industries (subset with non-empty industry)",
            top_n=15,
            max_label_len=34,
            wrap_width=22,
        )

    if not domain_region_share.empty:
        plot_heatmap_share(
            domain_region_share,
            figures_dir / "06_heatmap_domain_region_share.png",
            "Domain x Region (row-normalized share, excluding Other/Not specified)",
            "Region",
            "Domain",
        )

    if not region_domain_counts.empty:
        plot_stacked_100(
            region_domain_counts,
            figures_dir / "07_stacked_region_domain_top10.png",
            "Region composition by domains (100% stacked, excluding Other/Not specified)",
            "Region",
        )

    if not seniority_company_counts.empty:
        plot_stacked_100(
            seniority_company_counts,
            figures_dir / "08_stacked_seniority_company_top15.png",
            "Seniority x Company (100% stacked, excluding Other/Not specified)",
            "Company",
        )

    if not domain_company_counts.empty:
        plot_stacked_100(
            domain_company_counts,
            figures_dir / "09_stacked_domain_company_top15.png",
            "Domain x Company (100% stacked, excluding Other/Not specified)",
            "Company",
        )

    plot_barh(
        tools_top.rename(columns={"token_display": "tool"}),
        "tool",
        "count",
        figures_dir / "10_top_tools.png",
        "Top tools / stack",
        top_n=20,
        max_label_len=34,
        wrap_width=22,
    )
    plot_barh(
        skills_top.rename(columns={"token_display": "skill"}),
        "skill",
        "count",
        figures_dir / "11_top_skills.png",
        "Top skills",
        top_n=20,
        max_label_len=34,
        wrap_width=22,
    )

    if not domain_tools_matrix.empty:
        plot_heatmap_share(
            domain_tools_matrix,
            figures_dir / "12_heatmap_domain_tools_share.png",
            "Domain x Tools (row-normalized share, excluding Other/Not specified)",
            "Tool",
            "Domain",
        )

    seniority_donut = plot_donut(users_enriched["seniority_filled"], figures_dir / "13_donut_seniority_filled.png", "Seniority mix")
    cv_lang_series = users_enriched.loc[users_enriched["has_latex"], "cv_language_filled"].astype(str)
    cv_lang_series = cv_lang_series[cv_lang_series.isin(["ru", "en"])]
    cv_lang_donut = plot_donut(cv_lang_series, figures_dir / "14_donut_cv_generation_language.png", "CV language (among users with cvEnhancedResult)")
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
            max_label_len=36,
            wrap_width=24,
        )

    plot_bar(
        employment_status_summary,
        "employment_status",
        "count",
        figures_dir / "16_employment_status_overall.png",
        "Employment status overall",
        max_label_len=20,
    )

    if not domain_status_counts.empty:
        plot_stacked_100(
            domain_status_counts,
            figures_dir / "17_employment_status_by_domain.png",
            "Employment status by domain (100% stacked)",
            "Domain",
        )

    plot_hist(
        not_employed_months_since_last_end["months_since_last_end"]
        if "months_since_last_end" in not_employed_months_since_last_end.columns
        else pd.Series(dtype=float),
        figures_dir / "18_months_since_last_end_hist.png",
        "Months since last closed job (not employed)",
        "Months",
        bins=16,
    )

    plot_barh(
        not_employed_last_companies.rename(columns={"company_display_short": "company"}),
        "company",
        "count",
        figures_dir / "19_not_employed_top_last_companies.png",
        "Not employed: top last companies",
        top_n=20,
        max_label_len=34,
        wrap_width=22,
    )

    plot_barh(
        not_employed_last_titles.rename(columns={"job_title_short": "title"}),
        "title",
        "count",
        figures_dir / "20_not_employed_top_last_titles.png",
        "Not employed: top last titles",
        top_n=20,
        max_label_len=34,
        wrap_width=22,
    )

    plot_barh(
        not_employed_history_companies.rename(columns={"company_display_short": "company"}),
        "company",
        "count",
        figures_dir / "21_not_employed_history_top_companies.png",
        "Not employed history: top companies",
        top_n=20,
        max_label_len=34,
        wrap_width=22,
    )

    # Remapping effect: moved out of Other/Not specified.
    users_enriched["domain_raw_norm"] = users_enriched.get("talentCard.specialist_category", "").map(normalize_domain)

    def moved_out_reason(row: pd.Series) -> str:
        text = canonical_text(
            " ".join(
                [
                    clean_text(row.get("selectedPosition", "")),
                    clean_text(row.get("all_titles_text", "")),
                    clean_text(row.get("header_role_latex", "")),
                    clean_text(row.get("current_job_title_filled", "")),
                ]
            )
        )
        if re.search(r"recruit|talent acquisition|sourcer|talent partner|hrbp|рекрутер|подбор|сорсинг|кадр", text):
            return "Other/Not specified -> Recruiter/TA/HR keywords"
        if re.search(r"business analyst|бизнес[- ]?аналит|system analyst|системн\\w* аналит|requirements?|требован|bpmn|uml|use case", text):
            return "Other/Not specified -> Business/System Analyst keywords"
        if re.search(r"\\bux\\b|\\bui\\b|product designer|interaction designer|ux researcher|ux research|дизайнер интерфейс|ux[- ]?исследов", text):
            return "Other/Not specified -> UX/UI/Design keywords"
        if re.search(r"engineer|developer|devops|qa|test|разработ|инженер|тестиров", text):
            return "Other/Not specified -> Engineering keywords"
        if re.search(r"data|analytics|analyst|bi|machine learning|\\bml\\b|\\bai\\b|аналитик данных", text):
            return "Other/Not specified -> Data/Analytics keywords"
        return "Other/Not specified -> generic remapping"

    moved_mask = users_enriched["domain_raw_norm"].isin(["Other", "Not specified"]) & ~users_enriched["domain_filled"].isin(["Other", "Not specified"])
    moved_out_of_other = users_enriched.loc[moved_mask, ["domain_filled"]].copy()
    if not moved_out_of_other.empty:
        moved_out_of_other["old_label_keywords"] = users_enriched.loc[moved_mask].apply(moved_out_reason, axis=1)
        moved_out_of_other = (
            moved_out_of_other.groupby(["old_label_keywords", "domain_filled"])
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={"domain_filled": "new_domain"})
            .sort_values("count", ascending=False)
        )
    else:
        moved_out_of_other = pd.DataFrame(columns=["old_label_keywords", "new_domain", "count"])

    # Separate report for domain=Other.
    other_subset = users_enriched[users_enriched["domain_filled"].astype(str).eq("Other")].copy()
    all_jobs = pd.concat(
        [
            jobs_talent[["user_hash", "job_title", "company", "region"]].copy() if not jobs_talent.empty else pd.DataFrame(columns=["user_hash", "job_title", "company", "region"]),
            jobs_latex[["user_hash", "job_title", "company", "region"]].copy() if not jobs_latex.empty else pd.DataFrame(columns=["user_hash", "job_title", "company", "region"]),
        ],
        ignore_index=True,
    )
    other_jobs = all_jobs[all_jobs["user_hash"].isin(set(other_subset["user_hash"]))].copy()
    other_jobs["job_title"] = other_jobs["job_title"].fillna("").astype(str).str.strip()
    other_jobs["company"] = other_jobs["company"].fillna("").astype(str).str.strip()

    other_titles_all = top_counts(other_jobs["job_title"], 30, "job_title")
    other_current_titles = top_counts(other_subset["current_job_title_filled"], 20, "current_job_title_filled")
    other_companies_all = top_counts(other_jobs["company"], 30, "company")
    other_tools_top = top_tokens_from_list(other_subset, "tools_list", 30, "tool")
    other_skills_top = top_tokens_from_list(other_subset, "skills_list", 30, "skill")
    other_regions_top = top_counts(other_subset["region_norm"], 20, "region_norm")
    other_seniority_top = top_counts(other_subset["seniority_filled"], 20, "seniority_filled")

    all_title_tokens: List[str] = []
    for val in other_jobs["job_title"].fillna("").astype(str):
        chunks = re.split(r"[/,;|()\\-]+|\\s+", canonical_text(val))
        for token in chunks:
            tok = token.strip()
            if len(tok) >= 3 and tok not in {"and", "for", "the", "with"}:
                all_title_tokens.append(tok)
    title_keywords = top_counts(pd.Series(all_title_tokens, dtype=object), 30, "title_keyword")

    bucket_rows: List[str] = []
    all_titles_by_user = other_jobs.groupby("user_hash")["job_title"].apply(lambda x: " ".join([clean_text(v) for v in x if clean_text(v)])).to_dict()
    for _, row in other_subset.iterrows():
        profile_text = " ".join(
            [
                clean_text(row.get("selectedPosition", "")),
                clean_text(row.get("current_job_title_filled", "")),
                clean_text(all_titles_by_user.get(row["user_hash"], "")),
                " ".join(row.get("skills_list", [])[:25]) if isinstance(row.get("skills_list", []), list) else "",
                " ".join(row.get("tools_list", [])[:25]) if isinstance(row.get("tools_list", []), list) else "",
            ]
        )
        bucket_rows.append(_other_bucket(profile_text))
    other_bucket_dist = top_counts(pd.Series(bucket_rows, dtype=object), 20, "bucket")

    other_total = len(other_subset)
    cv_total = int(users_enriched["cvEnhancedResult_present"].sum())
    other_cv_count = int(other_subset["cvEnhancedResult_present"].sum())
    other_coverage = pd.DataFrame(
        {
            "metric": [
                "other_users_total",
                "share_all_users_%",
                "other_users_with_cvEnhancedResult",
                "share_among_cvEnhancedResult_users_%",
                "jobs_latex_present_%",
                "jobs_talent_present_%",
                "skills_or_tools_present_%",
                "region_specified_%",
                "seniority_specified_%",
            ],
            "value": [
                other_total,
                round(other_total / len(users_enriched) * 100, 1) if len(users_enriched) else 0.0,
                other_cv_count,
                round(other_cv_count / cv_total * 100, 1) if cv_total else 0.0,
                round((other_subset["expheader_count"].fillna(0) > 0).mean() * 100, 1) if other_total else 0.0,
                round((other_subset["jobs_count_talentCard"] > 0).mean() * 100, 1) if other_total else 0.0,
                round(((other_subset["skills_count"] > 0) | (other_subset["tools_count"] > 0)).mean() * 100, 1) if other_total else 0.0,
                round((~other_subset["region_norm"].astype(str).str.lower().isin(GENERIC_EXCLUDE)).mean() * 100, 1) if other_total else 0.0,
                round((~other_subset["seniority_filled"].astype(str).str.lower().isin(GENERIC_EXCLUDE)).mean() * 100, 1) if other_total else 0.0,
            ],
        }
    )

    others_tables = {
        "coverage": other_coverage,
        "moved_out_of_other_by_remapping": moved_out_of_other,
        "titles_all_history": other_titles_all,
        "current_titles": other_current_titles,
        "companies_all_history": other_companies_all,
        "tools_top": other_tools_top,
        "skills_top": other_skills_top,
        "regions_top": other_regions_top,
        "seniority_top": other_seniority_top,
        "title_keywords": title_keywords,
        "bucket_distribution": other_bucket_dist,
    }
    for name, table in others_tables.items():
        save_table(table, others_tables_dir / f"{name}.csv")

    plot_barh(
        other_titles_all,
        "job_title",
        "count",
        others_figures_dir / "01_other_titles_history_top30.png",
        "Other domain: top titles across all jobs history",
        top_n=30,
        max_label_len=34,
        wrap_width=22,
    )
    plot_barh(
        other_companies_all,
        "company",
        "count",
        others_figures_dir / "02_other_companies_history_top30.png",
        "Other domain: top companies across all jobs history",
        top_n=30,
        max_label_len=34,
        wrap_width=22,
    )
    plot_barh(
        other_tools_top,
        "tool",
        "count",
        others_figures_dir / "03_other_tools_top30.png",
        "Other domain: top tools",
        top_n=30,
        max_label_len=34,
        wrap_width=22,
    )
    plot_barh(
        other_bucket_dist,
        "bucket",
        "count",
        others_figures_dir / "04_other_bucket_distribution.png",
        "Other domain: rule-based bucket distribution",
        top_n=20,
        max_label_len=34,
        wrap_width=22,
    )

    build_others_report(
        base,
        others_tables=others_tables,
        others_stats={
            "other_users": other_total,
            "share_all_pct": round(other_total / len(users_enriched) * 100, 1) if len(users_enriched) else 0.0,
            "other_with_cv_count": other_cv_count,
            "share_among_cv_pct": round(other_cv_count / cv_total * 100, 1) if cv_total else 0.0,
        },
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
