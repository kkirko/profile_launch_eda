from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class PeriodParseResult:
    start_date: pd.Timestamp | pd.NaT
    end_date: pd.Timestamp | pd.NaT
    is_present: bool
    parse_ok: bool
    parse_quality: str


TOOL_KEYWORDS = {
    "python",
    "sql",
    "postgresql",
    "postgres",
    "mysql",
    "oracle",
    "clickhouse",
    "bigquery",
    "snowflake",
    "tableau",
    "power bi",
    "excel",
    "vba",
    "airflow",
    "dbt",
    "spark",
    "hadoop",
    "kafka",
    "docker",
    "kubernetes",
    "linux",
    "git",
    "jira",
    "confluence",
    "notion",
    "miro",
    "figma",
    "sap",
    "1c",
    "bitrix",
    "google analytics",
    "yandex metrica",
    "amplitude",
    "mixpanel",
    "react",
    "angular",
    "node.js",
    "node",
    "typescript",
    "javascript",
    "java",
    "c++",
    "c#",
    "go",
    "aws",
    "azure",
    "gcp",
    "grafana",
    "prometheus",
    "terraform",
    "ansible",
    "ci/cd",
    "jenkins",
    "gitlab",
}

SKILLS_SECTION_TITLES = {
    "навыки",
    "skills",
    "core skills",
}

PREV_EXPERIENCE_TITLES = {
    "предыдущий опыт работы",
    "previous experience",
}

PRESENT_PATTERNS = [
    r"\bpresent\b",
    r"\bcurrent\b",
    r"по\s*н\.?в\.?",
    r"по\s*настоя",
    r"н\.?в\.?",
]


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "n/a", "na", "-"}:
        return ""
    return text


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonical_token(token: str) -> str:
    t = token.lower().strip()
    t = t.replace("ё", "е")
    t = normalize_spaces(t)
    return t


def is_tool_token(token: str) -> bool:
    t = canonical_token(token)
    if t in TOOL_KEYWORDS:
        return True
    return any(k in t for k in TOOL_KEYWORDS)


def unique_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        token = canonical_token(item)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(item.strip())
    return out


def extract_latex_block(cv_enhanced_result: object) -> str:
    raw = clean_text(cv_enhanced_result)
    if not raw:
        return ""

    # Typical payload keeps escaped newlines and fenced latex block.
    raw = raw.replace("\\r\\n", "\n").replace("\\n", "\n")
    fence = re.search(r"```latex\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return raw


def latex_to_plain(text: str) -> str:
    if not text:
        return ""
    t = text
    t = t.replace("\\&", "&")
    t = t.replace("\\%", "%")
    t = t.replace("\\_", "_")
    t = t.replace("\\#", "#")
    t = t.replace("\\textbf", "")
    t = t.replace("\\textit", "")
    t = t.replace("\\emph", "")
    t = re.sub(r"\\begin\{[^}]+\}", "\n", t)
    t = re.sub(r"\\end\{[^}]+\}", "\n", t)
    t = re.sub(r"\\item\b", "\n", t)
    t = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", t)
    t = t.replace("{", " ").replace("}", " ")
    t = t.replace("\\\\", "\n")
    t = re.sub(r"\n{2,}", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def extract_section_content(latex_text: str, section_titles: set[str]) -> str:
    if not latex_text:
        return ""

    matches = list(re.finditer(r"\\section\*\{([^}]*)\}", latex_text, flags=re.IGNORECASE))
    if not matches:
        return ""

    for idx, m in enumerate(matches):
        title = canonical_token(m.group(1))
        if title not in section_titles:
            continue
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(latex_text)
        return latex_text[start:end].strip()
    return ""


def split_tokens(text: str) -> List[str]:
    if not text:
        return []
    chunks = re.split(r"[\n;,\|•·]+", text)
    out = []
    for ch in chunks:
        token = re.sub(r"\s+", " ", ch).strip(" .:-_()[]{}")
        if len(token) < 2:
            continue
        out.append(token)
    return out


def extract_skills_tools(latex_text: str) -> Dict[str, object]:
    section = extract_section_content(latex_text, SKILLS_SECTION_TITLES)
    if not section:
        return {
            "skills_section_found": False,
            "skills_block_raw": "",
            "skills_list": [],
            "tools_list": [],
            "skills_count": 0,
            "tools_count": 0,
            "skills_text_len": 0,
        }

    plain = latex_to_plain(section)
    lines = [normalize_spaces(x) for x in plain.split("\n") if normalize_spaces(x)]

    skills: List[str] = []
    tools: List[str] = []

    if len(lines) >= 2:
        skills.extend(split_tokens(lines[0]))
        tools.extend(split_tokens(lines[1]))
        for extra in lines[2:]:
            for tok in split_tokens(extra):
                if is_tool_token(tok):
                    tools.append(tok)
                else:
                    skills.append(tok)
    else:
        for tok in split_tokens(plain):
            if is_tool_token(tok):
                tools.append(tok)
            else:
                skills.append(tok)

    skills = unique_keep_order(skills)
    tools = unique_keep_order(tools)

    return {
        "skills_section_found": True,
        "skills_block_raw": section,
        "skills_list": skills,
        "tools_list": tools,
        "skills_count": len(skills),
        "tools_count": len(tools),
        "skills_text_len": len(plain),
    }


def parse_expheaders(latex_text: str) -> List[Dict[str, str]]:
    if not latex_text:
        return []

    pattern = re.compile(
        r"\\ExpHeader\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}",
        flags=re.DOTALL,
    )
    jobs: List[Dict[str, str]] = []
    for m in pattern.finditer(latex_text):
        company = normalize_spaces(m.group(1))
        location = normalize_spaces(m.group(2))
        period_raw = normalize_spaces(m.group(3))
        role = normalize_spaces(m.group(4))
        jobs.append(
            {
                "source": "latex_expheader",
                "company": company,
                "region": location,
                "employment_period": period_raw,
                "job_title": role,
            }
        )
    return jobs


def parse_previous_experience_items(latex_text: str) -> List[Dict[str, str]]:
    block = extract_section_content(latex_text, PREV_EXPERIENCE_TITLES)
    if not block:
        return []

    normalized = block.replace("—", "--").replace("–", "--")
    chunks = re.split(r"\\item", normalized)

    jobs: List[Dict[str, str]] = []
    for chunk in chunks:
        line = normalize_spaces(latex_to_plain(chunk))
        if not line:
            continue

        m = re.match(
            r"^(?P<period>(?:\d{1,2}\.\d{4}|\d{4})\s*(?:-|--|to)\s*(?:\d{1,2}\.\d{4}|\d{4}|present|current|по\s*н\.?в\.?|по\s*настоя[^\s,;]*))\s*-{2,3}\s*(?P<company>[^-]+?)\s*-{2,3}\s*(?P<role>.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if not m:
            continue

        jobs.append(
            {
                "source": "latex_previous",
                "company": normalize_spaces(m.group("company")),
                "region": "",
                "employment_period": normalize_spaces(m.group("period")),
                "job_title": normalize_spaces(m.group("role")),
            }
        )

    return jobs


def _parse_month_year(token: str) -> pd.Timestamp | pd.NaT:
    t = normalize_spaces(token)
    m = re.match(r"^(0?[1-9]|1[0-2])[./](19\d{2}|20\d{2})$", t)
    if not m:
        return pd.NaT
    month = int(m.group(1))
    year = int(m.group(2))
    return pd.Timestamp(year=year, month=month, day=1)


def _parse_year(token: str, end_of_year: bool) -> pd.Timestamp | pd.NaT:
    t = normalize_spaces(token)
    if not re.match(r"^(19\d{2}|20\d{2})$", t):
        return pd.NaT
    year = int(t)
    month = 12 if end_of_year else 1
    day = 1
    return pd.Timestamp(year=year, month=month, day=day)


def parse_period(period_raw: object, as_of_date: pd.Timestamp) -> PeriodParseResult:
    raw = clean_text(period_raw)
    if not raw:
        return PeriodParseResult(pd.NaT, pd.NaT, False, False, "empty")

    text = raw.lower().replace("—", "-").replace("–", "-")
    text = re.sub(r"\s*-{2,}\s*", " - ", text)
    text = normalize_spaces(text)

    is_present = any(re.search(p, text) for p in PRESENT_PATTERNS)
    parts = [p.strip() for p in re.split(r"\s+-\s+", text) if p.strip()]

    if len(parts) < 2:
        years = re.findall(r"(19\d{2}|20\d{2})", text)
        if len(years) == 2:
            start = _parse_year(years[0], end_of_year=False)
            end = _parse_year(years[1], end_of_year=True)
            return PeriodParseResult(start, end, False, True, "year_only")
        if len(years) == 1 and is_present:
            start = _parse_year(years[0], end_of_year=False)
            end = pd.Timestamp(year=as_of_date.year, month=as_of_date.month, day=1)
            return PeriodParseResult(start, end, True, True, "year_to_present")
        return PeriodParseResult(pd.NaT, pd.NaT, is_present, False, "failed")

    left, right = parts[0], parts[1]

    start = _parse_month_year(left)
    end = _parse_month_year(right)

    if pd.isna(start):
        start = _parse_year(left, end_of_year=False)
    if pd.isna(end):
        end = _parse_year(right, end_of_year=True)

    if is_present:
        end = pd.Timestamp(year=as_of_date.year, month=as_of_date.month, day=1)

    if pd.isna(start) or pd.isna(end):
        years = re.findall(r"(19\d{2}|20\d{2})", text)
        if len(years) >= 2:
            start = _parse_year(years[0], end_of_year=False)
            end = _parse_year(years[1], end_of_year=True)
            return PeriodParseResult(start, end, is_present, True, "year_only")
        return PeriodParseResult(pd.NaT, pd.NaT, is_present, False, "failed")

    quality = "present" if is_present else "month_year"
    if re.fullmatch(r"(19\d{2}|20\d{2})", left) or re.fullmatch(r"(19\d{2}|20\d{2})", right):
        quality = "year_only"

    return PeriodParseResult(start, end, is_present, True, quality)


def pick_current_job(parsed_jobs: List[Dict[str, object]]) -> Dict[str, object]:
    if not parsed_jobs:
        return {}

    df = pd.DataFrame(parsed_jobs)
    if df.empty:
        return {}

    present = df[df["is_present"] == True]  # noqa: E712
    if not present.empty:
        row = present.sort_values(["start_date", "job_index"], ascending=[False, True]).iloc[0]
        return row.to_dict()

    valid = df.dropna(subset=["start_date", "end_date"])
    if not valid.empty:
        row = valid.sort_values(["end_date", "start_date"], ascending=[False, False]).iloc[0]
        return row.to_dict()

    row = df.sort_values(["job_index"], ascending=[True]).iloc[0]
    return row.to_dict()


def parse_cv_latex(cv_enhanced_result: object, as_of_date: Optional[pd.Timestamp] = None) -> Dict[str, object]:
    as_of = as_of_date or pd.Timestamp.utcnow()

    latex_text = extract_latex_block(cv_enhanced_result)
    if not latex_text:
        return {
            "latex_found": False,
            "expheader_count": 0,
            "skills_section_found": False,
            "skills_list": [],
            "tools_list": [],
            "skills_count": 0,
            "tools_count": 0,
            "jobs_latex": [],
            "current_company_latex": "",
            "current_region_latex": "",
            "current_job_title_latex": "",
        }

    exp_jobs = parse_expheaders(latex_text)
    prev_jobs = parse_previous_experience_items(latex_text)
    jobs = exp_jobs + prev_jobs

    parsed_jobs: List[Dict[str, object]] = []
    for idx, job in enumerate(jobs):
        period = parse_period(job.get("employment_period", ""), as_of)
        parsed_jobs.append(
            {
                "job_index": idx,
                "source": job.get("source", "latex_expheader"),
                "company": clean_text(job.get("company", "")),
                "region": clean_text(job.get("region", "")),
                "job_title": clean_text(job.get("job_title", "")),
                "employment_period": clean_text(job.get("employment_period", "")),
                "start_date": period.start_date,
                "end_date": period.end_date,
                "is_present": period.is_present,
                "period_parse_ok": period.parse_ok,
                "period_parse_quality": period.parse_quality,
            }
        )

    current = pick_current_job(parsed_jobs)
    skills_payload = extract_skills_tools(latex_text)

    return {
        "latex_found": True,
        "expheader_count": len(exp_jobs),
        "skills_section_found": skills_payload["skills_section_found"],
        "skills_list": skills_payload["skills_list"],
        "tools_list": skills_payload["tools_list"],
        "skills_count": skills_payload["skills_count"],
        "tools_count": skills_payload["tools_count"],
        "skills_text_len": skills_payload["skills_text_len"],
        "jobs_latex": parsed_jobs,
        "jobs_count_latex": len(parsed_jobs),
        "current_company_latex": clean_text(current.get("company", "")),
        "current_region_latex": clean_text(current.get("region", "")),
        "current_job_title_latex": clean_text(current.get("job_title", "")),
    }
