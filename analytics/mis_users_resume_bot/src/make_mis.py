from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from parsing import (
    add_employment_parsed_fields,
    clean_text,
    hash_user_id,
    melt_jobs_long,
    normalize_empty_strings,
    parse_bool_series,
    sort_user_jobs,
)
from features import (
    build_profile_score,
    build_skill_aggregates,
    cluster_profiles,
    crosstab_share,
    experience_bin,
    extract_user_skills_tools,
    guess_country,
    infer_role_family,
    leadership_score,
    normalize_industry,
    normalize_region,
)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["savefig.dpi"] = 220
plt.rcParams["font.family"] = "DejaVu Sans"

PII_COLUMNS = {
    "firstName",
    "lastName",
    "userName",
    "talentCard.first_name",
    "talentCard.last_name",
    "userId",
    "_id",
    "supabaseTalentId",
}

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


def truncate_label(text: str, max_len: int = 48) -> str:
    t = str(text)
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def plot_bar_top(
    series: pd.Series,
    out: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    top_n: int = 12,
) -> None:
    data = series.value_counts().head(top_n)
    if data.empty:
        return

    labels = [truncate_label(x, 52) for x in data.index.astype(str)]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(data) + 1.5)))
    colors = sns.color_palette("viridis", n_colors=len(data))
    ax.barh(labels, data.values, color=colors)
    ax.invert_yaxis()

    total = series.shape[0]
    for i, v in enumerate(data.values):
        ax.text(v + 0.2, i, f"{v} ({pct(v, total):.1f}%)", va="center", fontsize=10)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.25)
    fig.subplots_adjust(left=0.30, right=0.98, top=0.90, bottom=0.10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_line(daily: pd.DataFrame, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily["date"], daily["new_users"], marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Users")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_funnel(funnel_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.barh(funnel_df["stage"], funnel_df["count"], color=sns.color_palette("mako", n_colors=len(funnel_df)))
    ax.invert_yaxis()
    base = int(funnel_df.iloc[0]["count"]) if len(funnel_df) else 0
    for i, row in funnel_df.iterrows():
        ax.text(row["count"] + 2, i, f"{int(row['count'])} ({pct(int(row['count']), base):.1f}%)", va="center", fontsize=10)
    ax.set_title("Product Funnel: onboarding -> cvPath -> analysis -> enhanced")
    ax.set_xlabel("Users")
    ax.set_ylabel("Stage")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(rect=(0.24, 0, 1, 1))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_hist(series: pd.Series, out: Path, title: str, xlabel: str) -> None:
    vals = series.dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(vals, bins=25, kde=True, color="#4c78a8", ax=ax)
    ax.axvline(vals.median(), color="#e45756", linestyle="--", label=f"Median: {vals.median():.1f}")
    ax.axvline(vals.mean(), color="#72b7b2", linestyle="--", label=f"Mean: {vals.mean():.1f}")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Users")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out: Path, title: str, xlabel: str, ylabel: str) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(5, 0.4 * df.shape[0] + 2)))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.4, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_skill_network(pairs_df: pd.DataFrame, out: Path, top_edges: int = 40) -> None:
    pairs = pairs_df[pairs_df["count"] >= 3].head(top_edges)
    if pairs.empty:
        return

    g = nx.Graph()
    for _, row in pairs.iterrows():
        g.add_edge(row["token_a"], row["token_b"], weight=float(row["count"]))

    if g.number_of_nodes() == 0:
        return

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(g, seed=42, k=0.6)
    weights = np.array([g[u][v]["weight"] for u, v in g.edges()])
    widths = 0.5 + (weights - weights.min()) / (weights.max() - weights.min() + 1e-6) * 2.5

    nx.draw_networkx_nodes(g, pos, node_size=450, node_color="#4c78a8", alpha=0.9)
    nx.draw_networkx_edges(g, pos, width=widths, alpha=0.4, edge_color="#888888")
    nx.draw_networkx_labels(g, pos, font_size=8)
    plt.title("Skill/Tool Co-occurrence Network (top edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def aggregate_user_job_metrics(jobs_long: pd.DataFrame, user_hashes: pd.Series) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grouped = jobs_long.groupby("user_hash") if not jobs_long.empty else []

    for user_hash, grp in grouped:
        g = sort_user_jobs(grp)
        companies = [clean_text(v) for v in g["company"] if clean_text(v)]
        industries = [clean_text(v) for v in g["industry"] if clean_text(v)]
        regions = [clean_text(v) for v in g["region"] if clean_text(v)]

        valid_dates = g.dropna(subset=["start_date", "end_date"]).sort_values("start_date")

        total_experience_years = np.nan
        if not valid_dates.empty:
            first_start = valid_dates["start_date"].min()
            last_end = valid_dates["end_date"].max()
            total_experience_years = round((last_end - first_start).days / 365.25, 2)

        durations = g["duration_months"].dropna()

        # current job: present -> latest start_date, else latest end_date
        current_row = None
        present_rows = g[g["is_present"] == True]  # noqa: E712
        if not present_rows.empty:
            current_row = present_rows.sort_values(["start_date", "job_index"], ascending=[False, True]).iloc[0]
        elif not valid_dates.empty:
            current_row = valid_dates.sort_values(["end_date", "start_date"], ascending=[False, False]).iloc[0]
        elif not g.empty:
            current_row = g.sort_values(["chronology_order", "job_index"], ascending=[True, True]).iloc[0]

        gaps_count = 0
        max_gap_months = 0.0
        overlaps_count = 0
        if len(valid_dates) >= 2:
            ordered = valid_dates.sort_values("start_date")
            for i in range(1, len(ordered)):
                prev_end = ordered.iloc[i - 1]["end_date"]
                cur_start = ordered.iloc[i]["start_date"]
                delta_days = (cur_start - prev_end).days
                if delta_days > 31:
                    gaps_count += 1
                    max_gap_months = max(max_gap_months, delta_days / 30.44)
                if delta_days < -31:
                    overlaps_count += 1

        resp_non_empty = (g["responsibilities_raw"].fillna("").astype(str).str.strip().ne(""))
        ach_non_empty = (g["achievements_raw"].fillna("").astype(str).str.strip().ne(""))
        ach_num = g["has_numbers_in_achievements"].fillna(False).astype(bool)

        rows.append(
            {
                "user_hash": user_hash,
                "jobs_count": int(len(g)),
                "companies_count": int(len(set(companies))),
                "industries_count": int(len(set(industries))),
                "regions_count": int(len(set(regions))),
                "total_experience_years": total_experience_years,
                "sum_tenure_years": round(durations.sum() / 12, 2) if len(durations) else np.nan,
                "avg_tenure_months": round(durations.mean(), 1) if len(durations) else np.nan,
                "median_tenure_months": round(durations.median(), 1) if len(durations) else np.nan,
                "current_job_title": clean_text(current_row["job_title"]) if current_row is not None else "",
                "current_company": clean_text(current_row["company"]) if current_row is not None else "",
                "current_industry": clean_text(current_row["industry"]) if current_row is not None else "",
                "current_region": clean_text(current_row["region"]) if current_row is not None else "",
                "career_gaps_count": int(gaps_count),
                "max_gap_months": round(max_gap_months, 1),
                "overlaps_count": int(overlaps_count),
                "resp_coverage": round(resp_non_empty.mean(), 3) if len(g) else 0,
                "ach_coverage": round(ach_non_empty.mean(), 3) if len(g) else 0,
                "ach_numbers_coverage": round((ach_num & ach_non_empty).mean(), 3) if len(g) else 0,
                "period_parse_failed_share": round((g["parse_quality_flag"] == "failed").mean(), 3) if len(g) else 1.0,
            }
        )

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        metrics = pd.DataFrame({"user_hash": user_hashes.unique()})

    metrics = pd.DataFrame({"user_hash": user_hashes.unique()}).merge(metrics, on="user_hash", how="left")

    fill_zero_cols = [
        "jobs_count",
        "companies_count",
        "industries_count",
        "regions_count",
        "career_gaps_count",
        "overlaps_count",
    ]
    for c in fill_zero_cols:
        metrics[c] = metrics[c].fillna(0).astype(int)

    fill_float_cols = [
        "max_gap_months",
        "resp_coverage",
        "ach_coverage",
        "ach_numbers_coverage",
        "period_parse_failed_share",
    ]
    for c in fill_float_cols:
        metrics[c] = metrics[c].fillna(0.0)

    for c in ["current_job_title", "current_company", "current_industry", "current_region"]:
        metrics[c] = metrics[c].fillna("")

    return metrics


def build_tables_and_figures(df: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    tables_dir = output_dir / "tables"
    figs_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    work = normalize_empty_strings(df)

    # Dates and bools
    for col in DATE_COLUMNS:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce", utc=True)

    work["onboardingCompleted"] = parse_bool_series(work["onboardingCompleted"], default_false=True)
    # Explicit assumption required by task
    work["isBanned"] = parse_bool_series(work.get("isBanned", pd.Series(index=work.index, dtype=object)), default_false=True)

    work["user_hash"] = work["userId"].map(hash_user_id)

    # Presence flags
    work["cvPath_present"] = work["cvPath"].fillna("").astype(str).str.strip().ne("")
    work["cvAnalysisResult_present"] = work["cvAnalysisResult"].fillna("").astype(str).str.strip().ne("")
    work["cvEnhancedResult_present"] = work["cvEnhancedResult"].fillna("").astype(str).str.strip().ne("")

    # Jobs long + parse
    jobs_long = melt_jobs_long(work, include_text_fields=True)
    analysis_date = work["updatedAt"].dropna().max()
    if pd.isna(analysis_date):
        analysis_date = pd.Timestamp.utcnow().tz_localize("UTC")

    jobs_long = add_employment_parsed_fields(jobs_long, analysis_date)

    jobs_long["responsibilities_raw"] = jobs_long.get("responsibilities_raw", pd.Series(dtype=object)).fillna("")
    jobs_long["achievements_raw"] = jobs_long.get("achievements_raw", pd.Series(dtype=object)).fillna("")
    jobs_long["resp_len"] = jobs_long["responsibilities_raw"].astype(str).str.len()
    jobs_long["ach_len"] = jobs_long["achievements_raw"].astype(str).str.len()
    jobs_long["has_numbers_in_achievements"] = jobs_long["achievements_raw"].astype(str).str.contains(
        r"\d|%|\$|€|₽", regex=True
    )

    # User metrics from jobs
    user_job_metrics = aggregate_user_job_metrics(jobs_long, work["user_hash"])

    # User profile frame without direct PII
    cols_keep = [
        "user_hash",
        "createdAt",
        "updatedAt",
        "onboardingCompleted",
        "isBanned",
        "banReason",
        "cvGenerationStatus",
        "cvAnalysisStatus",
        "cvPath_present",
        "cvAnalysisResult_present",
        "cvEnhancedResult_present",
        "selectedPosition",
        "talentCard.specialist_category",
        "talentCard.overall_summary",
        "talentCard.overall_skills",
        "talentCard.overall_tools",
        "cvAnalysisResult",
    ]
    user_df = work[cols_keep].copy()
    user_df = user_df.merge(user_job_metrics, on="user_hash", how="left")

    user_df["current_region_norm"] = user_df["current_region"].map(normalize_region)
    user_df["country_guess"] = user_df["current_region_norm"].map(guess_country)
    user_df["current_industry_norm"] = user_df["current_industry"].map(normalize_industry)

    user_df["role_family"] = user_df.apply(
        lambda r: infer_role_family(
            r.get("selectedPosition", ""),
            r.get("current_job_title", ""),
            r.get("talentCard.specialist_category", ""),
        ),
        axis=1,
    )

    leadership = user_df.apply(lambda r: leadership_score(r.get("current_job_title", ""), r.get("total_experience_years", np.nan)), axis=1)
    user_df["leadership_score"] = leadership.map(lambda x: x[0])
    user_df["leadership_level"] = leadership.map(lambda x: x[1])
    user_df["experience_bin"] = user_df["total_experience_years"].map(experience_bin)

    # Skills + tools
    extracted = user_df.apply(
        lambda r: extract_user_skills_tools(
            r.get("talentCard.overall_skills", ""),
            r.get("talentCard.overall_tools", ""),
            r.get("talentCard.overall_summary", ""),
        ),
        axis=1,
    )
    user_df["skills_list"] = extracted.map(lambda x: x["skills_list"])
    user_df["tools_list"] = extracted.map(lambda x: x["tools_list"])
    user_df["skills_count"] = extracted.map(lambda x: x["skills_count"])
    user_df["tools_count"] = extracted.map(lambda x: x["tools_count"])
    user_df["top_skill_family"] = extracted.map(lambda x: x["top_skill_family"])

    # Profile quality score
    user_df = build_profile_score(user_df)

    # Clustering
    cluster_labels, cluster_summary, cluster_terms = cluster_profiles(user_df)
    user_df["cluster_id"] = cluster_labels

    # Product MIS tables
    daily = (
        user_df.assign(date=user_df["createdAt"].dt.date)
        .groupby("date")
        .size()
        .reset_index(name="new_users")
        .sort_values("date")
    )
    weekly = (
        user_df.assign(week=user_df["createdAt"].dt.to_period("W").astype(str))
        .groupby("week")
        .size()
        .reset_index(name="new_users")
    )

    s1 = pd.Series(True, index=user_df.index)
    s2 = s1 & user_df["onboardingCompleted"]
    s3 = s2 & user_df["cvPath_present"]
    s4 = s3 & user_df["cvAnalysisResult_present"]
    s5 = s4 & user_df["cvEnhancedResult_present"]

    funnel = pd.DataFrame(
        {
            "stage": [
                "1) Signup",
                "2) Onboarding completed",
                "3) CV uploaded (cvPath)",
                "4) CV analyzed (cvAnalysisResult)",
                "5) CV enhanced (cvEnhancedResult)",
            ],
            "count": [int(s1.sum()), int(s2.sum()), int(s3.sum()), int(s4.sum()), int(s5.sum())],
        }
    )
    funnel["conversion_from_signup_%"] = (funnel["count"] / funnel.iloc[0]["count"] * 100).round(1)

    cv_analysis_status = user_df["cvAnalysisStatus"].fillna("not_started").astype(str).value_counts().rename_axis("status").reset_index(name="count")
    cv_generation_status = user_df["cvGenerationStatus"].fillna("not_started").astype(str).value_counts().rename_axis("status").reset_index(name="count")

    banned = pd.DataFrame(
        {
            "metric": ["users_total", "banned_users", "banned_share_%"],
            "value": [len(user_df), int(user_df["isBanned"].sum()), pct(int(user_df["isBanned"].sum()), len(user_df))],
        }
    )
    ban_reasons = (
        user_df.loc[user_df["isBanned"], "banReason"]
        .fillna("not_specified")
        .astype(str)
        .value_counts()
        .rename_axis("ban_reason")
        .reset_index(name="count")
    )

    # Candidate landscape
    role_family = user_df["role_family"].value_counts().rename_axis("role_family").reset_index(name="count")
    role_family["share_%"] = (role_family["count"] / len(user_df) * 100).round(1)

    specialist_cat = (
        user_df["talentCard.specialist_category"]
        .fillna("Not specified")
        .astype(str)
        .value_counts()
        .rename_axis("specialist_category")
        .reset_index(name="count")
    )
    specialist_cat["share_%"] = (specialist_cat["count"] / len(user_df) * 100).round(1)

    current_industry = user_df["current_industry_norm"].fillna("Not specified").value_counts().rename_axis("industry").reset_index(name="count")
    current_industry["share_%"] = (current_industry["count"] / len(user_df) * 100).round(1)

    current_company = (
        user_df["current_company"].fillna("").astype(str).str.strip().replace({"": "Not specified"}).value_counts().rename_axis("company").reset_index(name="count")
    )
    current_company["share_%"] = (current_company["count"] / len(user_df) * 100).round(1)

    current_region = user_df["current_region_norm"].fillna("Not specified").value_counts().rename_axis("region").reset_index(name="count")
    current_region["share_%"] = (current_region["count"] / len(user_df) * 100).round(1)

    country = user_df["country_guess"].fillna("Unknown").value_counts().rename_axis("country").reset_index(name="count")
    country["share_%"] = (country["count"] / len(user_df) * 100).round(1)

    # Experience dynamics
    exp_bins = user_df["experience_bin"].value_counts().rename_axis("experience_bin").reset_index(name="count")
    exp_bins["share_%"] = (exp_bins["count"] / len(user_df) * 100).round(1)

    tenure_stats = pd.DataFrame(
        {
            "metric": [
                "users_with_jobs",
                "users_with_valid_experience",
                "median_total_experience_years",
                "mean_total_experience_years",
                "median_tenure_months",
                "mean_tenure_months",
                "users_with_career_gaps",
                "users_with_overlaps",
            ],
            "value": [
                int((user_df["jobs_count"] > 0).sum()),
                int(user_df["total_experience_years"].notna().sum()),
                round(user_df["total_experience_years"].median(), 2),
                round(user_df["total_experience_years"].mean(), 2),
                round(user_df["median_tenure_months"].median(), 2),
                round(user_df["avg_tenure_months"].mean(), 2),
                int((user_df["career_gaps_count"] > 0).sum()),
                int((user_df["overlaps_count"] > 0).sum()),
            ],
        }
    )

    exp_x_role = crosstab_share(user_df, "experience_bin", "role_family")
    exp_x_industry = crosstab_share(user_df, "experience_bin", "current_industry_norm")
    leader_x_role = crosstab_share(user_df, "leadership_level", "role_family")

    # Skills
    skills_agg = build_skill_aggregates(user_df)

    # Profile scorecard
    score_dist = user_df["profile_quality_bucket"].value_counts().rename_axis("bucket").reset_index(name="count")
    score_dist["share_%"] = (score_dist["count"] / len(user_df) * 100).round(1)

    score_components = pd.DataFrame(
        {
            "component": [
                "summary_present",
                "skills_present",
                "jobs_present",
                "resp_cov",
                "ach_cov",
                "ach_num_cov",
                "period_parse_success",
                "analysis_present",
            ],
            "avg_value": [
                user_df["summary_present"].mean(),
                user_df["skills_present"].mean(),
                user_df["jobs_present"].mean(),
                user_df["resp_cov"].mean(),
                user_df["ach_cov"].mean(),
                user_df["ach_num_cov"].mean(),
                user_df["period_parse_success"].mean(),
                user_df["analysis_present"].mean(),
            ],
        }
    )
    score_components["avg_value"] = score_components["avg_value"].round(3)

    # Data quality + anomalies
    quality_cols = [
        "createdAt",
        "updatedAt",
        "onboardingCompleted",
        "cvPath",
        "cvAnalysisResult",
        "cvEnhancedResult",
        "selectedPosition",
        "talentCard.overall_summary",
        "talentCard.overall_skills",
        "talentCard.overall_tools",
        "talentCard.specialist_category",
        "isBanned",
        "banReason",
    ]
    quality = pd.DataFrame(
        {
            "field": quality_cols,
            "null_%": [round(work[c].isna().mean() * 100, 1) for c in quality_cols],
            "non_null_%": [round((1 - work[c].isna().mean()) * 100, 1) for c in quality_cols],
        }
    ).sort_values("null_%", ascending=False)

    parse_flags = (
        jobs_long["parse_quality_flag"].fillna("failed").astype(str).value_counts().rename_axis("parse_quality_flag").reset_index(name="count")
    )

    anomalies = pd.DataFrame(
        {
            "risk": [
                "employment_period_parse_failed",
                "industry_not_specified",
                "region_not_specified",
                "users_with_zero_jobs",
                "profiles_score_below_60",
                "isBanned_missing_interpreted_false",
            ],
            "count": [
                int((jobs_long["parse_quality_flag"] == "failed").sum()),
                int((user_df["current_industry_norm"] == "Not specified").sum()),
                int((user_df["current_region_norm"] == "Not specified").sum()),
                int((user_df["jobs_count"] == 0).sum()),
                int((user_df["profile_quality_score"] < 60).sum()),
                int(work["isBanned"].isna().sum()) if "isBanned" in work.columns else 0,
            ],
            "share_%": [
                pct(int((jobs_long["parse_quality_flag"] == "failed").sum()), max(len(jobs_long), 1)),
                pct(int((user_df["current_industry_norm"] == "Not specified").sum()), len(user_df)),
                pct(int((user_df["current_region_norm"] == "Not specified").sum()), len(user_df)),
                pct(int((user_df["jobs_count"] == 0).sum()), len(user_df)),
                pct(int((user_df["profile_quality_score"] < 60).sum()), len(user_df)),
                pct(int(work["isBanned"].isna().sum()) if "isBanned" in work.columns else 0, len(work)),
            ],
        }
    )

    # Save tables (aggregated only)
    tables = {
        "dataset_profile": pd.DataFrame(
            {
                "metric": [
                    "n_rows",
                    "n_cols",
                    "createdAt_min",
                    "createdAt_max",
                    "users_unique_hashed",
                ],
                "value": [
                    len(work),
                    work.shape[1],
                    str(work["createdAt"].min()),
                    str(work["createdAt"].max()),
                    user_df["user_hash"].nunique(),
                ],
            }
        ),
        "data_quality": quality,
        "daily_signups": daily,
        "weekly_signups": weekly,
        "funnel": funnel,
        "cv_analysis_status": cv_analysis_status,
        "cv_generation_status": cv_generation_status,
        "banned_summary": banned,
        "ban_reasons": ban_reasons,
        "role_family": role_family,
        "specialist_category": specialist_cat,
        "country_distribution": country,
        "region_distribution": current_region,
        "industry_distribution": current_industry,
        "current_company_distribution": current_company,
        "experience_bins": exp_bins,
        "tenure_stats": tenure_stats,
        "experience_x_role_share": exp_x_role.reset_index(),
        "experience_x_industry_share": exp_x_industry.reset_index(),
        "leadership_x_role_share": leader_x_role.reset_index(),
        "skill_top": skills_agg["skills_top"].head(50),
        "tool_top": skills_agg["tools_top"].head(50),
        "skill_families": skills_agg["skill_families"],
        "skill_cooccurrence_top_pairs": skills_agg["cooccurrence_pairs"].head(100),
        "cluster_summary": cluster_summary,
        "cluster_terms": cluster_terms,
        "profile_score_distribution": score_dist,
        "profile_score_components": score_components,
        "employment_parse_flags": parse_flags,
        "anomalies_risks": anomalies,
    }

    for name, tdf in tables.items():
        save_table(tdf, tables_dir / f"{name}.csv")

    # Figures
    plot_line(daily, figs_dir / "01_signups_trend.png", "New users by day")
    plot_funnel(funnel, figs_dir / "02_funnel_onboarding_to_enhanced.png")
    plot_bar_top(cv_analysis_status.set_index("status")["count"], figs_dir / "03_cv_analysis_status.png", "CV analysis status", "Users", "Status", top_n=8)
    plot_bar_top(role_family.set_index("role_family")["count"], figs_dir / "04_role_family.png", "Role family", "Users", "Role", top_n=10)
    plot_bar_top(specialist_cat.set_index("specialist_category")["count"], figs_dir / "05_specialist_category.png", "Specialist category", "Users", "Category", top_n=10)
    plot_bar_top(country.set_index("country")["count"], figs_dir / "06_country_distribution.png", "Top countries (guess)", "Users", "Country", top_n=10)
    plot_bar_top(current_industry.set_index("industry")["count"], figs_dir / "07_industry_distribution.png", "Top industries", "Users", "Industry", top_n=12)
    plot_bar_top(current_company.set_index("company")["count"], figs_dir / "15_current_company_top.png", "Top current employers", "Users", "Company", top_n=20)
    plot_hist(user_df["total_experience_years"], figs_dir / "08_total_experience_years_hist.png", "Total experience years", "Years")
    plot_bar_top(cluster_summary.set_index("cluster_id")["users"], figs_dir / "09_cluster_sizes.png", "Cluster sizes", "Users", "Cluster", top_n=12)
    plot_hist(user_df["profile_quality_score"], figs_dir / "10_profile_quality_score_hist.png", "Profile quality score", "Score")
    plot_bar_top(skills_agg["skill_families"].set_index("skill_family")["count"], figs_dir / "11_skill_families.png", "Skill families", "Users", "Skill family", top_n=10)
    plot_skill_network(skills_agg["cooccurrence_pairs"], figs_dir / "12_skill_cooccurrence_network.png")
    plot_heatmap(exp_x_role, figs_dir / "13_experience_x_role_heatmap.png", "Experience x Role family (%)", "Role family", "Experience bin")
    plot_heatmap(leader_x_role, figs_dir / "14_leadership_x_role_heatmap.png", "Leadership x Role family (%)", "Role family", "Leadership")

    return tables


def build_report(tables: Dict[str, pd.DataFrame], output_dir: Path, report_path: Path) -> None:
    def top_rows(df: pd.DataFrame, n: int = 10) -> str:
        if df.empty:
            return "(no data)"
        return df.head(n).to_string(index=False)

    dataset_profile = tables["dataset_profile"]
    funnel = tables["funnel"]
    role_family = tables["role_family"]
    specialist = tables["specialist_category"]
    industry = tables["industry_distribution"]
    company = tables["current_company_distribution"]
    country = tables["country_distribution"]
    exp_bins = tables["experience_bins"]
    cluster_summary = tables["cluster_summary"]
    score_dist = tables["profile_score_distribution"]
    anomalies = tables["anomalies_risks"]
    quality = tables["data_quality"]
    tenure = tables["tenure_stats"]
    skill_families = tables["skill_families"]

    n_users = int(dataset_profile.loc[dataset_profile["metric"] == "n_rows", "value"].iloc[0])
    onboarding = int(funnel.loc[funnel["stage"] == "2) Onboarding completed", "count"].iloc[0])
    enhanced = int(funnel.loc[funnel["stage"] == "5) CV enhanced (cvEnhancedResult)", "count"].iloc[0])
    banned = int(tables["banned_summary"].loc[tables["banned_summary"]["metric"] == "banned_users", "value"].iloc[0])

    top_role = role_family.iloc[0] if not role_family.empty else None
    top_ind = industry.iloc[0] if not industry.empty else None
    top_country = country.iloc[0] if not country.empty else None

    report_text = f"""# MIS Report: Users EDA (Resume Bot)

## 1) Executive MIS Summary
- Users in dataset: **{n_users}**
- Onboarding completed: **{onboarding}** ({pct(onboarding, n_users):.1f}%)
- CV enhanced available: **{enhanced}** ({pct(enhanced, n_users):.1f}%)
- Banned users: **{banned}** ({pct(banned, n_users):.1f}%)
- Top role family: **{top_role['role_family']}** ({top_role['share_%']:.1f}%)
- Top industry: **{top_ind['industry']}** ({top_ind['share_%']:.1f}%)
- Top country (guess): **{top_country['country']}** ({top_country['share_%']:.1f}%)

### Key Findings
- Основной отток в продуктовой воронке происходит до этапа `onboarding completed`.
- Профили сильно различаются по полноте: большой хвост профилей с низким score из-за отсутствия структурированных job-данных и индустрий.
- В текущем срезе доминируют роли Product/Project и Engineering/Software, с заметным кластером Data/Analytics.
- Доля незаполненных `industry` и частично `region` ограничивает точность отраслевой и гео-аналитики.
- По карьерной динамике встречаются разрывы и пересечения периодов, что важно учитывать в matching/scoring.

## 2) Data Quality & Coverage
Допущения:
- `isBanned`: пустые значения интерпретированы как `False`.
- Периоды вида `present/по н.в.` завершаются датой анализа.
- Для `YYYY-YYYY` парсинг с флагом `year_only`.

### Null Coverage (key fields)
```
{top_rows(quality, 20)}
```

### Employment Period Parse Quality
```
{top_rows(tables['employment_parse_flags'], 10)}
```

## 3) Candidate Landscape
### Role Family
```
{top_rows(role_family, 12)}
```
### Specialist Category
```
{top_rows(specialist, 12)}
```
### Geography
```
{top_rows(country, 12)}
```
### Industry
```
{top_rows(industry, 12)}
```
### Current Employers (aggregated)
```
{top_rows(company, 20)}
```

## 4) Experience & Career Dynamics
### Tenure / Experience Summary
```
{top_rows(tenure, 12)}
```
### Experience Bins
```
{top_rows(exp_bins, 10)}
```

## 5) Skills & Tools
### Skill Families
```
{top_rows(skill_families, 12)}
```
### Top Skills / Tools
```
{top_rows(tables['skill_top'], 20)}
```

## 6) Segmentation
### Text Clusters (TF-IDF + KMeans)
```
{top_rows(cluster_summary, 12)}
```
### Cluster Top Terms
```
{top_rows(tables['cluster_terms'], 12)}
```
### Experience x Role (share %)
```
{top_rows(tables['experience_x_role_share'], 12)}
```
### Leadership x Role (share %)
```
{top_rows(tables['leadership_x_role_share'], 12)}
```

## 7) Profile Quality Scorecard
### Score Distribution
```
{top_rows(score_dist, 12)}
```
### Score Components
```
{top_rows(tables['profile_score_components'], 12)}
```

## 8) Anomalies & Risks
```
{top_rows(anomalies, 20)}
```

## 9) Visuals
![Signups](outputs/figures/01_signups_trend.png)
![Funnel](outputs/figures/02_funnel_onboarding_to_enhanced.png)
![CV Analysis Status](outputs/figures/03_cv_analysis_status.png)
![Role Family](outputs/figures/04_role_family.png)
![Specialist Category](outputs/figures/05_specialist_category.png)
![Country Distribution](outputs/figures/06_country_distribution.png)
![Industry Distribution](outputs/figures/07_industry_distribution.png)
![Top Employers](outputs/figures/15_current_company_top.png)
![Experience Hist](outputs/figures/08_total_experience_years_hist.png)
![Cluster Sizes](outputs/figures/09_cluster_sizes.png)
![Profile Score](outputs/figures/10_profile_quality_score_hist.png)
![Skill Families](outputs/figures/11_skill_families.png)
![Skill Network](outputs/figures/12_skill_cooccurrence_network.png)
![Experience x Role](outputs/figures/13_experience_x_role_heatmap.png)
![Leadership x Role](outputs/figures/14_leadership_x_role_heatmap.png)

## 10) Appendix
### Data Dictionary (key fields)
- `createdAt`, `updatedAt`: жизненный цикл пользователя
- `onboardingCompleted`: завершение онбординга
- `cvPath`, `cvAnalysisResult`, `cvEnhancedResult`: этапы обработки CV
- `talentCard.jobs[i].*`: история работ
- `talentCard.overall_summary/skills/tools`: агрегированный профиль навыков
- `talentCard.specialist_category`: категория специалиста

### Generated Artifacts
- Aggregated tables: `outputs/tables/*.csv`
- Figures: `outputs/figures/*.png`

## How to reproduce
```bash
python analytics/mis_users_resume_bot/src/make_mis.py --input /Users/k/Downloads/prointerview-prod.users.csv --output analytics/mis_users_resume_bot/outputs
```
"""
    report_path.write_text(report_text, encoding="utf-8")


def build_notebook(notebook_path: Path) -> None:
    # Minimal reproducible notebook without raw PII displays.
    notebook_json = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# EDA Users Resume Bot (MIS)\\n",
                    "Этот ноутбук запускает генерацию MIS-артефактов и показывает агрегаты без PII."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\\n",
                    "import pandas as pd\\n",
                    "import sys\\n",
                    "sys.path.append(str(Path('analytics/mis_users_resume_bot/src').resolve()))\\n",
                    "from make_mis import build_tables_and_figures, build_report\\n",
                    "from parsing import normalize_empty_strings\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "input_csv = Path('/Users/k/Downloads/prointerview-prod.users.csv')\\n",
                    "output_dir = Path('analytics/mis_users_resume_bot/outputs')\\n",
                    "report = Path('analytics/mis_users_resume_bot/README.md')\\n",
                    "df = pd.read_csv(input_csv, low_memory=False)\\n",
                    "tables = build_tables_and_figures(df, output_dir)\\n",
                    "build_report(tables, output_dir, report)\\n",
                    "print('MIS generated')"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pd.read_csv('analytics/mis_users_resume_bot/outputs/tables/dataset_profile.csv')"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    import json

    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook_json, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MIS analytics for resume-bot users")
    parser.add_argument("--input", required=True, help="Path to users CSV")
    parser.add_argument(
        "--output",
        default="analytics/mis_users_resume_bot/outputs",
        help="Output directory for aggregated tables and figures",
    )
    parser.add_argument(
        "--report",
        default="analytics/mis_users_resume_bot/README.md",
        help="Path to markdown MIS report",
    )
    parser.add_argument(
        "--notebook",
        default="analytics/mis_users_resume_bot/notebooks/eda_users_resume_bot.ipynb",
        help="Path to generated notebook",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    report_path = Path(args.report)
    notebook_path = Path(args.notebook)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # keep matplotlib cache writable in sandboxed env
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    df = pd.read_csv(input_path, low_memory=False)
    tables = build_tables_and_figures(df, output_dir)
    build_report(tables, output_dir, report_path)
    build_notebook(notebook_path)

    print(f"Report: {report_path}")
    print(f"Figures: {output_dir / 'figures'}")
    print(f"Tables: {output_dir / 'tables'}")
    print(f"Notebook: {notebook_path}")


if __name__ == "__main__":
    main()
