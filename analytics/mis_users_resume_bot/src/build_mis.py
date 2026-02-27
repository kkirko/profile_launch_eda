from __future__ import annotations

import argparse
import json
import os
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
    cooccurrence_pairs,
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
    profile_quality_score,
    summarize_user_jobs,
)
from latex_parser import parse_cv_latex


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


def shorten(text: str, max_len: int = 56) -> str:
    t = str(text)
    return t if len(t) <= max_len else t[: max_len - 3] + "..."


def barh_counts(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    top_n: int = 20,
) -> None:
    if df.empty:
        return
    top = df.head(top_n).copy()
    labels = [shorten(v, 62) for v in top[label_col].astype(str)]
    values = top[value_col].astype(float).values

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(top) + 1.5)))
    colors = sns.color_palette("viridis", n_colors=len(top))
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()

    total = float(df[value_col].sum()) if df[value_col].sum() else 0.0
    for i, v in enumerate(values):
        share = pct(int(v), int(total)) if total else 0.0
        ax.text(v + max(values) * 0.01, i, f"{int(v)} ({share:.1f}%)", va="center", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(label_col.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.90, bottom=0.10)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def line_series(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.plot(df[x_col], df[y_col], marker="o", linewidth=2.2, color="#355C7D")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_funnel(funnel: pd.DataFrame, out_path: Path) -> None:
    if funnel.empty:
        return
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    colors = sns.color_palette("mako", n_colors=len(funnel))
    ax.barh(funnel["stage"], funnel["count"], color=colors)
    ax.invert_yaxis()

    base = int(funnel.iloc[0]["count"]) if len(funnel) else 0
    for i, row in funnel.iterrows():
        ax.text(row["count"] + max(funnel["count"]) * 0.01, i, f"{int(row['count'])} ({pct(int(row['count']), base):.1f}%)", va="center", fontsize=10)

    ax.set_title("Funnel: onboardingCompleted -> cvPath -> cvAnalysis completed -> cvEnhancedResult")
    ax.set_xlabel("Users")
    ax.set_ylabel("Stage")
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.40, right=0.98, top=0.90, bottom=0.12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def heatmap(
    matrix: pd.DataFrame,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    annotate: bool = True,
) -> None:
    if matrix.empty:
        return
    fig_h = max(6, 0.42 * matrix.shape[0] + 2)
    fig_w = max(10, 0.42 * matrix.shape[1] + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(matrix, cmap="YlGnBu", annot=annotate, fmt=".0f", linewidths=0.35, cbar_kws={"label": "Users"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def histogram(series: pd.Series, out_path: Path, title: str, xlabel: str) -> None:
    values = series.dropna().astype(float)
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    sns.histplot(values, bins=24, kde=True, color="#2E86AB", ax=ax)
    ax.axvline(values.median(), color="#F18F01", linestyle="--", label=f"median: {values.median():.1f}")
    ax.axvline(values.mean(), color="#C73E1D", linestyle="--", label=f"mean: {values.mean():.1f}")
    ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Users")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def top_n_with_other(series: pd.Series, top_n: int) -> pd.Series:
    top_values = set(series.value_counts().head(top_n).index)
    return series.map(lambda x: x if x in top_values else "Other")


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


def build_notebook(path: Path) -> None:
    notebook_json = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MIS users resume bot\n",
                    "Ноутбук для воспроизводимого запуска MIS-отчета (без вывода PII и сырых LaTeX).",
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
                    "run(input_path='/Users/k/Downloads/prointerview-prod.users.csv', base_dir='analytics/mis_users_resume_bot')",
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
    users_summary = tables["dataset_profile"]
    funnel = tables["funnel"]
    domains = tables["domain_distribution"]
    regions = tables["region_distribution"]
    companies = tables["company_distribution"]
    industries = tables["industry_distribution"]
    skills_top = tables["skills_top"]
    tools_top = tables["tools_top"]
    validation = tables["validation_summary"]

    total_users = int(users_summary.loc[users_summary["metric"] == "users_total", "value"].iloc[0])
    onboarding = int(funnel.loc[funnel["stage"] == "1) onboardingCompleted", "count"].iloc[0])
    cv_path = int(funnel.loc[funnel["stage"] == "2) cvPath exists", "count"].iloc[0])
    analysis_completed = int(funnel.loc[funnel["stage"] == "3) cvAnalysisStatus=completed", "count"].iloc[0])
    enhanced = int(funnel.loc[funnel["stage"] == "4) cvEnhancedResult exists", "count"].iloc[0])

    top_domain = domains.head(3)
    top_region = regions.head(3)
    top_company = companies.head(3)
    top_industry = industries.head(3)

    val_map = dict(zip(validation["metric"], validation["value"]))
    expheader_share = float(val_map.get("share_users_with_expheader_%", 0))
    skills_share = float(val_map.get("share_users_with_skills_section_%", 0))
    company_match = float(val_map.get("current_company_match_rate_%", 0))

    observations = [
        f"`onboardingCompleted` у {onboarding} из {total_users} пользователей ({pct(onboarding, total_users):.1f}%).",
        f"До этапа `cvAnalysisStatus=completed` доходят {analysis_completed} пользователей ({pct(analysis_completed, total_users):.1f}% от базы).",
        f"`cvEnhancedResult` присутствует у {enhanced} пользователей ({pct(enhanced, total_users):.1f}%).",
        f"LaTeX с `\\ExpHeader` распознан у {expheader_share:.1f}% пользователей, skills-секция найдена у {skills_share:.1f}%.",
        f"Согласованность current company между talentCard и LaTeX (нормализовано) составляет {company_match:.1f}% среди сравнимых профилей.",
        f"Топ-домен: {top_domain.iloc[0]['domain']} ({int(top_domain.iloc[0]['count'])} пользователей).",
        f"Топ-регион (current): {top_region.iloc[0]['region_display']} ({int(top_region.iloc[0]['count'])}).",
        f"Топ-компания (current): {top_company.iloc[0]['company_display']} ({int(top_company.iloc[0]['count'])}).",
        f"Топ-индустрия current job: {top_industry.iloc[0]['industry_display']} ({int(top_industry.iloc[0]['count'])}).",
        f"Самые частые инструменты: {', '.join(tools_top.head(5)['token_display'].tolist())}.",
    ]

    observations_md = "\n".join([f"- {line}" for line in observations])
    missingness_md = tables["missingness"].to_markdown(index=False)
    validation_md = validation.to_markdown(index=False)
    jobs_diff_md = tables["jobs_count_diff_distribution"].to_markdown(index=False)

    readme = f"""# MIS: Users Resume Bot

## 1) MIS Summary
- Users total: **{total_users}**
- onboardingCompleted: **{onboarding}** ({pct(onboarding, total_users):.1f}%)
- cvPath exists: **{cv_path}** ({pct(cv_path, total_users):.1f}%)
- cvAnalysisStatus=completed: **{analysis_completed}** ({pct(analysis_completed, total_users):.1f}%)
- cvEnhancedResult exists: **{enhanced}** ({pct(enhanced, total_users):.1f}%)

Top-3 domains:
{top_domain[['domain','count']].to_markdown(index=False)}

Top-3 regions:
{top_region[['region_display','count']].to_markdown(index=False)}

Top-3 companies:
{top_company[['company_display','count']].to_markdown(index=False)}

### Key observations
{observations_md}

## 2) Data Coverage & Quality
### Missingness (key fields)
{missingness_md}

### LaTeX parsing validation
{validation_md}

### Jobs count comparison (talentCard vs LaTeX)
{jobs_diff_md}

## 3) Domains & Geography
![Weekly signups](outputs/figures/01_registrations_weekly.png)
![Top domains](outputs/figures/04_top_domains.png)
![Top regions](outputs/figures/05_top_regions.png)
![Domain x Region heatmap](outputs/figures/09_heatmap_domain_region.png)

## 4) Companies & Seniority
![Top companies](outputs/figures/06_top_companies.png)
![Top industries](outputs/figures/07_top_industries.png)
![Heatmap seniority x company](outputs/figures/10_heatmap_seniority_company.png)
![Heatmap domain x company](outputs/figures/14_heatmap_domain_company.png)

## 5) Skills & Stack
![Top tools](outputs/figures/11_top_tools.png)
![Top skills](outputs/figures/12_top_skills.png)
![Heatmap domain x tools](outputs/figures/13_heatmap_domain_tools.png)

## 6) Product funnel and status
![Funnel](outputs/figures/02_funnel.png)
![CV analysis status](outputs/figures/03_cv_analysis_status.png)
![Experience distribution](outputs/figures/08_experience_hist.png)
![Profile quality score](outputs/figures/15_profile_quality_score.png)

## 7) Appendix
Generated tables:
- `outputs/tables/validation_summary.csv`
- `outputs/tables/mismatch_samples.csv`
- `outputs/tables/region_mappings.csv`
- `outputs/tables/company_mappings.csv`
- `outputs/tables/industry_mappings.csv`
- `outputs/tables/domain_region_heatmap.csv`
- `outputs/tables/seniority_company_heatmap.csv`
- `outputs/tables/domain_tools_heatmap.csv`

How to reproduce:
```bash
/opt/anaconda3/bin/python analytics/mis_users_resume_bot/src/build_mis.py \
  --input /Users/k/Downloads/prointerview-prod.users.csv \
  --base-dir analytics/mis_users_resume_bot
```
"""

    (base_dir / "README.md").write_text(readme, encoding="utf-8")


def run(input_path: str, base_dir: str) -> Dict[str, pd.DataFrame]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    base = Path(base_dir)
    outputs = base / "outputs"
    figures_dir = outputs / "figures"
    tables_dir = outputs / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    users = pd.read_csv(input_path, low_memory=False)
    users = normalize_empty_strings(users)

    for col in DATE_COLUMNS:
        if col in users.columns:
            users[col] = pd.to_datetime(users[col], errors="coerce", utc=True)

    users["onboardingCompleted"] = parse_bool_series(users.get("onboardingCompleted", pd.Series(index=users.index, dtype=object)), default_false=True)
    users["isBanned"] = parse_bool_series(users.get("isBanned", pd.Series(index=users.index, dtype=object)), default_false=True)

    users["user_hash"] = users["userId"].map(hash_user_id)
    users["cvPath_present"] = users.get("cvPath", "").fillna("").astype(str).str.strip().ne("")
    users["cvEnhancedResult_present"] = users.get("cvEnhancedResult", "").fillna("").astype(str).str.strip().ne("")
    users["cvAnalysis_completed"] = users.get("cvAnalysisStatus", "").fillna("").astype(str).str.strip().str.lower().eq("completed")

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

    users_enriched["current_company_raw"] = users_enriched["current_company_latex"].fillna("")
    users_enriched.loc[users_enriched["current_company_raw"].astype(str).str.strip().eq(""), "current_company_raw"] = (
        users_enriched["current_company_talentCard"].fillna("")
    )

    users_enriched["current_region_raw"] = users_enriched["current_region_latex"].fillna("")
    users_enriched.loc[users_enriched["current_region_raw"].astype(str).str.strip().eq(""), "current_region_raw"] = (
        users_enriched["current_region_talentCard"].fillna("")
    )

    users_enriched["current_job_title_raw"] = users_enriched["current_job_title_latex"].fillna("")
    users_enriched.loc[users_enriched["current_job_title_raw"].astype(str).str.strip().eq(""), "current_job_title_raw"] = (
        users_enriched["current_job_title_talentCard"].fillna("")
    )

    users_enriched["current_industry_raw"] = users_enriched["current_industry_talentCard"].fillna("")
    users_enriched["current_seniority_raw"] = users_enriched["current_seniority_talentCard"].fillna("")

    users_enriched["company_norm"] = users_enriched["current_company_raw"].map(normalize_company)
    users_enriched["region_norm"] = users_enriched["current_region_raw"].map(normalize_region)
    users_enriched["industry_norm"] = users_enriched["current_industry_raw"].map(normalize_industry)
    users_enriched["country_guess"] = users_enriched["region_norm"].map(guess_country)

    users_enriched["domain"] = users_enriched.get("talentCard.specialist_category", pd.Series(index=users_enriched.index, dtype=object)).fillna("Not specified").astype(str)
    users_enriched["role_family"] = users_enriched.apply(
        lambda r: infer_role_family(r.get("selectedPosition", ""), r.get("current_job_title_raw", ""), r.get("domain", "")),
        axis=1,
    )

    users_enriched["seniority_norm"] = users_enriched["current_seniority_raw"].map(normalize_seniority)
    mask_unknown = users_enriched["seniority_norm"].eq("Unknown")
    users_enriched.loc[mask_unknown, "seniority_norm"] = users_enriched.loc[mask_unknown, "current_job_title_raw"].map(normalize_seniority)

    users_enriched["experience_bin"] = users_enriched["total_experience_years"].map(experience_bin)
    users_enriched["leadership_level"] = users_enriched.apply(
        lambda r: leadership_level(r.get("current_job_title_raw", ""), r.get("seniority_norm", ""), r.get("total_experience_years", np.nan)),
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

    users_enriched = profile_quality_score(users_enriched)

    # Coverage and validation.
    total_users = len(users_enriched)
    cv_enhanced_users = int(users_enriched["cvEnhancedResult_present"].sum())
    expheader_users = int((users_enriched["expheader_count"].fillna(0) > 0).sum())
    skills_section_users = int(users_enriched["skills_section_found"].fillna(False).sum())

    compare_company = users_enriched[["user_hash", "current_company_talentCard", "current_company_latex"]].copy()
    compare_company["company_talent_norm"] = compare_company["current_company_talentCard"].map(normalize_company)
    compare_company["company_latex_norm"] = compare_company["current_company_latex"].map(normalize_company)
    comparable_mask = compare_company["company_talent_norm"].ne("Not specified") & compare_company["company_latex_norm"].ne("Not specified")
    comparable_n = int(comparable_mask.sum())
    matches_n = int((compare_company.loc[comparable_mask, "company_talent_norm"] == compare_company.loc[comparable_mask, "company_latex_norm"]).sum())

    validation_summary = pd.DataFrame(
        {
            "metric": [
                "users_total",
                "users_with_cvEnhancedResult",
                "share_users_with_cvEnhancedResult_%",
                "users_with_expheader",
                "share_users_with_expheader_%",
                "users_with_skills_section",
                "share_users_with_skills_section_%",
                "company_comparable_users",
                "current_company_matches",
                "current_company_match_rate_%",
            ],
            "value": [
                total_users,
                cv_enhanced_users,
                pct(cv_enhanced_users, total_users),
                expheader_users,
                pct(expheader_users, total_users),
                skills_section_users,
                pct(skills_section_users, total_users),
                comparable_n,
                matches_n,
                pct(matches_n, comparable_n),
            ],
        }
    )

    jobs_count_diff = users_enriched[["jobs_count_talentCard", "jobs_count_latex"]].copy()
    jobs_count_diff["diff_latex_minus_talent"] = jobs_count_diff["jobs_count_latex"] - jobs_count_diff["jobs_count_talentCard"]
    jobs_count_diff_distribution = jobs_count_diff["diff_latex_minus_talent"].value_counts().rename_axis("diff").reset_index(name="users")
    jobs_count_diff_distribution = jobs_count_diff_distribution.sort_values("diff")

    mismatch_samples = compare_company.loc[
        comparable_mask & (compare_company["company_talent_norm"] != compare_company["company_latex_norm"]),
        ["user_hash", "current_company_talentCard", "current_company_latex", "company_talent_norm", "company_latex_norm"],
    ].copy()
    mismatch_samples = mismatch_samples.head(200)

    # Product layer.
    users_enriched["created_week"] = users_enriched["createdAt"].dt.to_period("W").astype(str)
    weekly_signups = users_enriched.groupby("created_week").size().reset_index(name="new_users").sort_values("created_week")

    funnel = pd.DataFrame(
        {
            "stage": [
                "1) onboardingCompleted",
                "2) cvPath exists",
                "3) cvAnalysisStatus=completed",
                "4) cvEnhancedResult exists",
            ],
            "count": [
                int(users_enriched["onboardingCompleted"].sum()),
                int((users_enriched["onboardingCompleted"] & users_enriched["cvPath_present"]).sum()),
                int((users_enriched["onboardingCompleted"] & users_enriched["cvPath_present"] & users_enriched["cvAnalysis_completed"]).sum()),
                int((users_enriched["onboardingCompleted"] & users_enriched["cvPath_present"] & users_enriched["cvAnalysis_completed"] & users_enriched["cvEnhancedResult_present"]).sum()),
            ],
        }
    )

    cv_analysis_status = users_enriched.get("cvAnalysisStatus", pd.Series(index=users_enriched.index, dtype=object)).fillna("not_started").astype(str)
    cv_analysis_status = cv_analysis_status.value_counts().rename_axis("status").reset_index(name="count")

    # Distributions.
    domain_distribution = users_enriched["domain"].fillna("Not specified").astype(str).value_counts().rename_axis("domain").reset_index(name="count")

    region_distribution = (
        users_enriched.assign(region_display=users_enriched["current_region_raw"].fillna("").replace("", "Not specified"))
        .groupby(["region_display", "region_norm"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    company_distribution = (
        users_enriched.assign(company_display=users_enriched["current_company_raw"].fillna("").replace("", "Not specified"))
        .groupby(["company_display", "company_norm"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    industry_distribution = (
        users_enriched.assign(industry_display=users_enriched["current_industry_raw"].fillna("").replace("", "Not specified"))
        .groupby(["industry_display", "industry_norm"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    seniority_distribution = users_enriched["seniority_norm"].value_counts().rename_axis("seniority").reset_index(name="count")

    # Skills / tools.
    skills_top = aggregate_tokens(users_enriched, "skills_list")
    tools_top = aggregate_tokens(users_enriched, "tools_list")
    tools_pairs = cooccurrence_pairs(users_enriched, "tools_list", min_count=3)

    # Heatmaps.
    domain_small = top_n_with_other(users_enriched["domain"].fillna("Not specified"), 12)
    region_small = top_n_with_other(users_enriched["region_norm"].fillna("Not specified"), 25)
    domain_region_heatmap = pd.crosstab(domain_small, region_small)

    company_small = top_n_with_other(users_enriched["company_norm"].fillna("Not specified"), 30)
    seniority_order = ["Junior", "Middle", "Senior", "Lead", "C-level", "Unknown"]
    seniority_small = users_enriched["seniority_norm"].fillna("Unknown")
    seniority_company_heatmap = pd.crosstab(seniority_small, company_small)
    seniority_company_heatmap = seniority_company_heatmap.reindex(seniority_order, fill_value=0)

    domain_company_heatmap = pd.crosstab(domain_small, company_small)

    top_tool_norms = tools_top.head(35)["token_norm"].tolist()
    domain_tools_rows: List[Dict[str, object]] = []
    for _, row in users_enriched[["domain", "tools_list"]].iterrows():
        domain_raw = str(row["domain"]).strip()
        domain = domain_raw if domain_raw else "Not specified"
        tokens = {canonical_text(t) for t in (row["tools_list"] or []) if canonical_text(t) in top_tool_norms}
        for token in tokens:
            domain_tools_rows.append({"domain": domain, "tool_norm": token})

    if domain_tools_rows:
        domain_tools_df = pd.DataFrame(domain_tools_rows)
        domain_tools_heatmap = pd.crosstab(
            top_n_with_other(domain_tools_df["domain"], 12),
            top_n_with_other(domain_tools_df["tool_norm"], 35),
        )
    else:
        domain_tools_heatmap = pd.DataFrame()

    # Missingness.
    key_fields = [
        "createdAt",
        "updatedAt",
        "onboardingCompleted",
        "cvPath",
        "cvAnalysisStatus",
        "cvEnhancedResult",
        "talentCard.specialist_category",
        "talentCard.overall_summary",
        "talentCard.overall_skills",
        "talentCard.overall_tools",
        "isBanned",
        "banReason",
    ]
    missingness = pd.DataFrame(
        {
            "field": key_fields,
            "missing_%": [round(users[c].isna().mean() * 100, 1) if c in users.columns else 100.0 for c in key_fields],
            "filled_%": [round((1 - users[c].isna().mean()) * 100, 1) if c in users.columns else 0.0 for c in key_fields],
        }
    ).sort_values("missing_%", ascending=False)

    # Mapping tables raw->norm.
    region_mappings = build_mapping_table(users_enriched["current_region_raw"], users_enriched["region_norm"], "region")
    company_mappings = build_mapping_table(users_enriched["current_company_raw"], users_enriched["company_norm"], "company")
    industry_mappings = build_mapping_table(users_enriched["current_industry_raw"], users_enriched["industry_norm"], "industry")

    # Summary tables.
    dataset_profile = pd.DataFrame(
        {
            "metric": [
                "users_total",
                "columns_total",
                "createdAt_min",
                "createdAt_max",
                "onboardingCompleted_true",
                "banned_true",
            ],
            "value": [
                len(users),
                users.shape[1],
                str(users["createdAt"].min()) if "createdAt" in users.columns else "",
                str(users["createdAt"].max()) if "createdAt" in users.columns else "",
                int(users_enriched["onboardingCompleted"].sum()),
                int(users_enriched["isBanned"].sum()),
            ],
        }
    )

    profile_quality_distribution = users_enriched["profile_quality_bucket"].value_counts().rename_axis("bucket").reset_index(name="count")

    tables: Dict[str, pd.DataFrame] = {
        "dataset_profile": dataset_profile,
        "missingness": missingness,
        "weekly_signups": weekly_signups,
        "funnel": funnel,
        "cv_analysis_status": cv_analysis_status,
        "domain_distribution": domain_distribution,
        "region_distribution": region_distribution,
        "company_distribution": company_distribution,
        "industry_distribution": industry_distribution,
        "seniority_distribution": seniority_distribution,
        "skills_top": skills_top,
        "tools_top": tools_top,
        "tools_pairs": tools_pairs,
        "validation_summary": validation_summary,
        "jobs_count_diff_distribution": jobs_count_diff_distribution,
        "mismatch_samples": mismatch_samples,
        "region_mappings": region_mappings,
        "company_mappings": company_mappings,
        "industry_mappings": industry_mappings,
        "domain_region_heatmap": domain_region_heatmap.reset_index(),
        "seniority_company_heatmap": seniority_company_heatmap.reset_index(),
        "domain_company_heatmap": domain_company_heatmap.reset_index(),
        "domain_tools_heatmap": domain_tools_heatmap.reset_index() if not domain_tools_heatmap.empty else pd.DataFrame(),
        "profile_quality_distribution": profile_quality_distribution,
    }

    for name, frame in tables.items():
        save_table(frame, tables_dir / f"{name}.csv")

    # Figures.
    line_series(
        weekly_signups,
        x_col="created_week",
        y_col="new_users",
        out_path=figures_dir / "01_registrations_weekly.png",
        title="User registrations by week",
        xlabel="Week",
        ylabel="Users",
    )
    plot_funnel(funnel, figures_dir / "02_funnel.png")
    barh_counts(cv_analysis_status, "status", "count", figures_dir / "03_cv_analysis_status.png", "CV analysis status", "Users", top_n=12)
    barh_counts(domain_distribution, "domain", "count", figures_dir / "04_top_domains.png", "Top professional domains (specialist_category)", "Users", top_n=20)
    barh_counts(region_distribution.rename(columns={"region_display": "region"}), "region", "count", figures_dir / "05_top_regions.png", "Top current regions", "Users", top_n=20)
    barh_counts(company_distribution.rename(columns={"company_display": "company"}), "company", "count", figures_dir / "06_top_companies.png", "Top current companies", "Users", top_n=20)
    barh_counts(industry_distribution.rename(columns={"industry_display": "industry"}), "industry", "count", figures_dir / "07_top_industries.png", "Top current industries", "Users", top_n=20)
    histogram(users_enriched["total_experience_years"], figures_dir / "08_experience_hist.png", "Total experience (years)", "Years")
    heatmap(domain_region_heatmap, figures_dir / "09_heatmap_domain_region.png", "Heatmap: Domain x Region", "Region", "Domain")
    heatmap(seniority_company_heatmap, figures_dir / "10_heatmap_seniority_company.png", "Heatmap: Seniority x Company", "Company", "Seniority")
    barh_counts(tools_top.rename(columns={"token_display": "tool"}), "tool", "count", figures_dir / "11_top_tools.png", "Top tools/stack", "Users", top_n=25)
    barh_counts(skills_top.rename(columns={"token_display": "skill"}), "skill", "count", figures_dir / "12_top_skills.png", "Top skills", "Users", top_n=25)
    if not domain_tools_heatmap.empty:
        heatmap(domain_tools_heatmap, figures_dir / "13_heatmap_domain_tools.png", "Heatmap: Domain x Tools", "Tool", "Domain")
    heatmap(domain_company_heatmap, figures_dir / "14_heatmap_domain_company.png", "Heatmap: Domain x Company", "Company", "Domain")
    histogram(users_enriched["profile_quality_score"], figures_dir / "15_profile_quality_score.png", "Profile quality score", "Score")

    build_notebook(base / "notebooks" / "mis_users_resume_bot.ipynb")
    build_readme(base, tables)

    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIS report for users dataset")
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
