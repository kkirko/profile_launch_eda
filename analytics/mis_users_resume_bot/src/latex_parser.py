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

LANGUAGES_SECTION_TITLES = {
    "languages",
    "языки",
}

EDUCATION_SECTION_TITLES = {
    "education",
    "образование",
}

PRESENT_PATTERNS = [
    r"\bpresent\b",
    r"\bcurrent\b",
    r"по\s*н\.?в\.?",
    r"по\s*настоя",
    r"н\.?в\.?",
]

CONTACT_TOKENS = {
    "linkedin",
    "telegram",
    "github",
    "email",
    "mail",
    "phone",
    "tel",
    "github.com",
    "linkedin.com",
    "t.me",
    "@",
}

ENGLISH_HINTS = ["english", "англий", "англ"]


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
    return normalize_spaces(t)


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

    raw = raw.replace("\\r\\n", "\n").replace("\\n", "\n")
    fenced = re.search(r"```latex\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
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

    sections = list(re.finditer(r"\\section\*\{([^}]*)\}", latex_text, flags=re.IGNORECASE))
    if not sections:
        return ""

    for idx, match in enumerate(sections):
        title = canonical_token(match.group(1))
        if title not in section_titles:
            continue
        start = match.end()
        end = sections[idx + 1].start() if idx + 1 < len(sections) else len(latex_text)
        return latex_text[start:end].strip()
    return ""


def split_tokens(text: str) -> List[str]:
    if not text:
        return []
    chunks = re.split(r"[\n;,\|•·]+", text)
    out: List[str] = []
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
            for token in split_tokens(extra):
                if is_tool_token(token):
                    tools.append(token)
                else:
                    skills.append(token)
    else:
        for token in split_tokens(plain):
            if is_tool_token(token):
                tools.append(token)
            else:
                skills.append(token)

    skills = unique_keep_order(skills)
    tools = unique_keep_order(tools)

    return {
        "skills_section_found": True,
        "skills_list": skills,
        "tools_list": tools,
        "skills_count": len(skills),
        "tools_count": len(tools),
        "skills_text_len": len(plain),
    }


def parse_expheaders(latex_text: str) -> List[Dict[str, str]]:
    if not latex_text:
        return []

    pattern = re.compile(r"\\ExpHeader\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}", flags=re.DOTALL)
    jobs: List[Dict[str, str]] = []
    for match in pattern.finditer(latex_text):
        jobs.append(
            {
                "source": "latex_expheader",
                "company": normalize_spaces(match.group(1)),
                "region": normalize_spaces(match.group(2)),
                "employment_period": normalize_spaces(match.group(3)),
                "job_title": normalize_spaces(match.group(4)),
            }
        )
    return jobs


def parse_previous_experience_items(latex_text: str) -> List[Dict[str, str]]:
    block = extract_section_content(latex_text, PREV_EXPERIENCE_TITLES)
    if not block:
        return []

    chunks = re.split(r"\\item", block.replace("—", "--").replace("–", "--"))
    jobs: List[Dict[str, str]] = []

    for chunk in chunks:
        line = normalize_spaces(latex_to_plain(chunk))
        if not line:
            continue

        match = re.match(
            r"^(?P<period>(?:\d{1,2}\.\d{4}|\d{4})\s*(?:-|--|to)\s*(?:\d{1,2}\.\d{4}|\d{4}|present|current|по\s*н\.?в\.?|по\s*настоя[^\s,;]*))\s*-{2,3}\s*(?P<company>[^-]+?)\s*-{2,3}\s*(?P<role>.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        jobs.append(
            {
                "source": "latex_previous",
                "company": normalize_spaces(match.group("company")),
                "region": "",
                "employment_period": normalize_spaces(match.group("period")),
                "job_title": normalize_spaces(match.group("role")),
            }
        )

    return jobs


def _parse_month_year(token: str) -> pd.Timestamp | pd.NaT:
    t = normalize_spaces(token)
    match = re.match(r"^(0?[1-9]|1[0-2])[./](19\d{2}|20\d{2})$", t)
    if not match:
        return pd.NaT
    return pd.Timestamp(year=int(match.group(2)), month=int(match.group(1)), day=1)


def _parse_year(token: str, end_of_year: bool) -> pd.Timestamp | pd.NaT:
    t = normalize_spaces(token)
    if not re.match(r"^(19\d{2}|20\d{2})$", t):
        return pd.NaT
    return pd.Timestamp(year=int(t), month=12 if end_of_year else 1, day=1)


def parse_period(period_raw: object, as_of_date: pd.Timestamp) -> PeriodParseResult:
    raw = clean_text(period_raw)
    if not raw:
        return PeriodParseResult(pd.NaT, pd.NaT, False, False, "empty")

    text = raw.lower().replace("—", "-").replace("–", "-")
    text = re.sub(r"\s*-{2,}\s*", " - ", text)
    text = normalize_spaces(text)

    is_present = any(re.search(pattern, text) for pattern in PRESENT_PATTERNS)
    parts = [part.strip() for part in re.split(r"\s+-\s+", text) if part.strip()]

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

    return df.sort_values(["job_index"], ascending=True).iloc[0].to_dict()


def _contains_contact(text: str) -> bool:
    t = canonical_token(text)
    if not t:
        return False
    if any(token in t for token in CONTACT_TOKENS):
        return True
    if re.search(r"\+?\d[\d\s\-()]{6,}", t):
        return True
    if re.search(r"https?://|www\.", t):
        return True
    return False


def extract_role_line(latex_text: str) -> str:
    match = re.search(r"\\RoleLine\{([^{}]*)\}", latex_text, flags=re.DOTALL)
    if not match:
        return ""
    return normalize_spaces(latex_to_plain(match.group(1)))


def extract_header_location(latex_text: str) -> str:
    if not latex_text:
        return ""

    raw_lines = re.findall(r"\\SmallLine\{([^{}]*)\}", latex_text, flags=re.DOTALL)
    candidates: List[str] = []

    for raw in raw_lines:
        plain = normalize_spaces(latex_to_plain(raw))
        if not plain:
            continue

        parts = re.split(r"[•\|;]+", plain)
        for part in parts:
            piece = normalize_spaces(part)
            if not piece:
                continue
            if _contains_contact(piece):
                continue
            if re.search(r"\d{4}", piece):
                continue
            if len(piece) < 2 or len(piece) > 80:
                continue
            if not re.search(r"[A-Za-zА-Яа-я]", piece):
                continue
            candidates.append(piece)

    if not candidates:
        return ""

    def score(value: str) -> int:
        s = 0
        low = canonical_token(value)
        if "," in value:
            s += 2
        if any(token in low for token in ["moscow", "москва", "saint", "санкт", "минск", "dubai", "warsaw", "almaty", "tashkent", "yerevan", "russia", "belarus", "kazakhstan"]):
            s += 2
        if len(value) <= 35:
            s += 1
        if re.search(r"\d", value):
            s -= 2
        return s

    best = sorted(candidates, key=lambda x: (score(x), -len(x)), reverse=True)[0]
    return best


def _extract_english_level(text: str) -> str:
    low = canonical_token(text)
    if not low:
        return "Unknown"

    level_match = re.search(r"\b([ABC][12])\b", low, flags=re.IGNORECASE)
    if level_match and any(h in low for h in ENGLISH_HINTS):
        return level_match.group(1).upper()

    pattern = re.search(r"(english|англий\w*)[^\n,;:]{0,30}([ABC][12])", low, flags=re.IGNORECASE)
    if pattern:
        return pattern.group(2).upper()

    if any(h in low for h in ENGLISH_HINTS):
        if re.search(r"native|proficient|fluent|свобод", low):
            return "C2"
        if re.search(r"advanced|продвин", low):
            return "C1"
        if re.search(r"upper\s*intermediate", low):
            return "B2"
        if re.search(r"intermediate|средн", low):
            return "B1"
        if re.search(r"elementary|базов", low):
            return "A2"
        if re.search(r"beginner|началь", low):
            return "A1"

    return "Unknown"


def parse_languages(latex_text: str) -> Dict[str, object]:
    section = extract_section_content(latex_text, LANGUAGES_SECTION_TITLES)
    if not section:
        return {
            "languages_section_found": False,
            "english_level": "Unknown",
            "languages_count": 0,
        }

    plain = latex_to_plain(section)
    tokens = split_tokens(plain)
    language_candidates: List[str] = []

    for token in tokens:
        clean = re.sub(r"\b([ABC][12]|upper\s*intermediate|intermediate|advanced|native|fluent|beginner|elementary)\b", " ", token, flags=re.IGNORECASE)
        clean = normalize_spaces(clean)
        if len(clean) < 2:
            continue
        if not re.search(r"[A-Za-zА-Яа-я]", clean):
            continue
        language_candidates.append(clean)

    language_candidates = unique_keep_order(language_candidates)

    return {
        "languages_section_found": True,
        "english_level": _extract_english_level(plain),
        "languages_count": len(language_candidates),
    }


def parse_education(latex_text: str) -> Dict[str, object]:
    section = extract_section_content(latex_text, EDUCATION_SECTION_TITLES)
    if not section:
        return {
            "education_section_found": False,
            "degree_level": "Unknown",
            "education_text_len": 0,
        }

    plain = latex_to_plain(section)
    low = canonical_token(plain)

    degree_level = "Unknown"
    if re.search(r"phd|doctor|доктор|кандидат\s+наук", low):
        degree_level = "PhD"
    elif re.search(r"\bmba\b", low):
        degree_level = "MBA"
    elif re.search(r"msc|m\.sc|master|магистр", low):
        degree_level = "MSc"
    elif re.search(r"ba|b\.a|bachelor of arts", low):
        degree_level = "BA"
    elif re.search(r"bsc|b\.sc|bachelor|бакалавр", low):
        degree_level = "BSc"

    return {
        "education_section_found": True,
        "degree_level": degree_level,
        "education_text_len": len(plain),
    }


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
            "skills_text_len": 0,
            "jobs_latex": [],
            "jobs_count_latex": 0,
            "current_company_latex": "",
            "current_region_latex": "",
            "current_job_title_latex": "",
            "current_source_latex": "",
            "current_company_expheader": "",
            "current_region_expheader": "",
            "current_job_title_expheader": "",
            "header_role_latex": "",
            "header_location_latex": "",
            "english_level": "Unknown",
            "languages_count": 0,
            "degree_level": "Unknown",
            "education_text_len": 0,
            "languages_section_found": False,
            "education_section_found": False,
        }

    exp_jobs = parse_expheaders(latex_text)
    prev_jobs = parse_previous_experience_items(latex_text)
    jobs = exp_jobs + prev_jobs

    parsed_jobs: List[Dict[str, object]] = []
    parsed_exp_jobs: List[Dict[str, object]] = []
    for idx, job in enumerate(jobs):
        period = parse_period(job.get("employment_period", ""), as_of)
        parsed_item = {
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
        parsed_jobs.append(parsed_item)
        if parsed_item["source"] == "latex_expheader":
            parsed_exp_jobs.append(parsed_item)

    current = pick_current_job(parsed_jobs)
    current_exp = pick_current_job(parsed_exp_jobs)
    skills_payload = extract_skills_tools(latex_text)
    languages_payload = parse_languages(latex_text)
    education_payload = parse_education(latex_text)

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
        "current_source_latex": clean_text(current.get("source", "")),
        "current_company_expheader": clean_text(current_exp.get("company", "")),
        "current_region_expheader": clean_text(current_exp.get("region", "")),
        "current_job_title_expheader": clean_text(current_exp.get("job_title", "")),
        "header_role_latex": extract_role_line(latex_text),
        "header_location_latex": extract_header_location(latex_text),
        "english_level": languages_payload["english_level"],
        "languages_count": languages_payload["languages_count"],
        "languages_section_found": languages_payload["languages_section_found"],
        "degree_level": education_payload["degree_level"],
        "education_text_len": education_payload["education_text_len"],
        "education_section_found": education_payload["education_section_found"],
    }
