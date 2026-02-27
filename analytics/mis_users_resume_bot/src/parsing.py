from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class PeriodParse:
    start_date: pd.Timestamp | pd.NaT
    end_date: pd.Timestamp | pd.NaT
    is_present: bool
    parse_quality_flag: str


def hash_user_id(user_id: object) -> str:
    value = "" if pd.isna(user_id) else str(user_id)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "n/a", "na", "-"}:
        return ""
    return text


def normalize_empty_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    obj_cols = out.select_dtypes(include=["object"]).columns
    for col in obj_cols:
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
        return mapped.fillna(False)
    return mapped


def find_job_indices(columns: Iterable[str]) -> List[int]:
    idx = set()
    pattern = re.compile(r"^talentCard\.jobs\[(\d+)\]\.")
    for col in columns:
        m = pattern.match(col)
        if m:
            idx.add(int(m.group(1)))
    return sorted(idx)


def melt_jobs_long(df: pd.DataFrame, include_text_fields: bool = True) -> pd.DataFrame:
    job_indices = find_job_indices(df.columns)
    base_fields = [
        "region",
        "company",
        "job_title",
        "seniority",
        "industry",
        "employment_period",
        "chronology_order",
    ]
    extra_fields = ["responsibilities", "achievements"] if include_text_fields else []
    fields = base_fields + extra_fields

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        user_hash = row["user_hash"]
        for j in job_indices:
            item: Dict[str, object] = {"user_hash": user_hash, "job_index": j}
            for f in fields:
                col = f"talentCard.jobs[{j}].{f}"
                item[f] = row[col] if col in df.columns else np.nan

            has_title = clean_text(item.get("job_title", "")) != ""
            has_company = clean_text(item.get("company", "")) != ""
            has_period = clean_text(item.get("employment_period", "")) != ""
            if not (has_title or has_company or has_period):
                continue

            rows.append(item)

    jobs_long = pd.DataFrame(rows)
    if jobs_long.empty:
        return pd.DataFrame(
            columns=[
                "user_hash",
                "job_index",
                "chronology_order",
                "region",
                "company",
                "job_title",
                "seniority",
                "industry",
                "employment_period_raw",
                "responsibilities_raw",
                "achievements_raw",
            ]
        )

    jobs_long = jobs_long.rename(
        columns={
            "employment_period": "employment_period_raw",
            "responsibilities": "responsibilities_raw",
            "achievements": "achievements_raw",
        }
    )
    return jobs_long


def _parse_month_year_token(token: str) -> pd.Timestamp | pd.NaT:
    token = token.strip().lower()
    m = re.match(r"^(0?[1-9]|1[0-2])\.(19\d{2}|20\d{2})$", token)
    if not m:
        return pd.NaT
    month = int(m.group(1))
    year = int(m.group(2))
    return pd.Timestamp(year=year, month=month, day=1)


def _parse_year_token(token: str, end_of_year: bool = False) -> pd.Timestamp | pd.NaT:
    token = token.strip()
    if not re.match(r"^(19\d{2}|20\d{2})$", token):
        return pd.NaT
    year = int(token)
    if end_of_year:
        return pd.Timestamp(year=year, month=12, day=1)
    return pd.Timestamp(year=year, month=1, day=1)


def parse_employment_period(raw: object, analysis_date: pd.Timestamp) -> PeriodParse:
    text = clean_text(raw)
    if not text:
        return PeriodParse(pd.NaT, pd.NaT, False, "empty")

    normalized = text.lower().replace("—", "-").replace("–", "-")
    normalized = re.sub(r"\s*-{2,}\s*", " - ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    is_present = bool(re.search(r"\bpresent\b|\bcurrent\b|по\s*н\.?в\.?|по\s*настоя|н\.?в\.?", normalized))
    parts = [p.strip() for p in normalized.split("-") if p.strip()]

    if len(parts) < 2:
        years = re.findall(r"(19\d{2}|20\d{2})", normalized)
        if len(years) == 2:
            start = _parse_year_token(years[0], end_of_year=False)
            end = _parse_year_token(years[1], end_of_year=True)
            return PeriodParse(start, end, False, "year_only")
        if len(years) == 1 and is_present:
            start = _parse_year_token(years[0], end_of_year=False)
            end = pd.Timestamp(year=analysis_date.year, month=analysis_date.month, day=1)
            return PeriodParse(start, end, True, "year_to_present")
        return PeriodParse(pd.NaT, pd.NaT, is_present, "failed")

    left, right = parts[0], parts[1]

    start = _parse_month_year_token(left)
    end = _parse_month_year_token(right)

    if pd.isna(start):
        start = _parse_year_token(left, end_of_year=False)
    if pd.isna(end):
        end = _parse_year_token(right, end_of_year=True)

    if is_present:
        end = pd.Timestamp(year=analysis_date.year, month=analysis_date.month, day=1)

    if pd.isna(start) or pd.isna(end):
        years = re.findall(r"(19\d{2}|20\d{2})", normalized)
        if len(years) >= 2:
            start = _parse_year_token(years[0], end_of_year=False)
            end = _parse_year_token(years[1], end_of_year=True)
            return PeriodParse(start, end, is_present, "year_only")
        return PeriodParse(pd.NaT, pd.NaT, is_present, "failed")

    quality = "month_year"
    if re.match(r"^(19\d{2}|20\d{2})$", left) or re.match(r"^(19\d{2}|20\d{2})$", right):
        quality = "year_only"
    if is_present:
        quality = "present"

    return PeriodParse(start, end, is_present, quality)


def add_employment_parsed_fields(jobs_long: pd.DataFrame, analysis_date: pd.Timestamp) -> pd.DataFrame:
    if jobs_long.empty:
        return jobs_long

    parsed = jobs_long["employment_period_raw"].map(lambda x: parse_employment_period(x, analysis_date))
    out = jobs_long.copy()
    out["start_date"] = parsed.map(lambda x: x.start_date)
    out["end_date"] = parsed.map(lambda x: x.end_date)
    out["is_present"] = parsed.map(lambda x: x.is_present)
    out["parse_quality_flag"] = parsed.map(lambda x: x.parse_quality_flag)

    def duration_months(row: pd.Series) -> float:
        if pd.isna(row["start_date"]) or pd.isna(row["end_date"]):
            return math.nan
        months = (row["end_date"].year - row["start_date"].year) * 12 + (row["end_date"].month - row["start_date"].month) + 1
        if months <= 0 or months > 720:
            return math.nan
        return float(months)

    out["duration_months"] = out.apply(duration_months, axis=1)
    return out


def sort_user_jobs(jdf: pd.DataFrame) -> pd.DataFrame:
    # If dates exist, they are source of truth. chronology_order and job_index are fallbacks.
    tmp = jdf.copy()
    tmp["start_date_sort"] = tmp["start_date"].fillna(pd.Timestamp("1900-01-01"))
    tmp["chronology_order_sort"] = pd.to_numeric(tmp.get("chronology_order"), errors="coerce").fillna(9999)
    tmp = tmp.sort_values(["start_date_sort", "chronology_order_sort", "job_index"], ascending=[True, True, True])
    return tmp.drop(columns=["start_date_sort", "chronology_order_sort"])
