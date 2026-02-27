from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


REGION_REPLACEMENTS = {
    "москва": "Moscow",
    "moscow": "Moscow",
    "санкт-петербург": "Saint Petersburg",
    "санкт петербург": "Saint Petersburg",
    "st. petersburg": "Saint Petersburg",
    "saint petersburg": "Saint Petersburg",
    "минск": "Minsk",
    "minsk": "Minsk",
    "алматы": "Almaty",
    "almaty": "Almaty",
    "ереван": "Yerevan",
    "yerevan": "Yerevan",
    "ташкент": "Tashkent",
    "tashkent": "Tashkent",
}

COUNTRY_PATTERNS = [
    ("Russia", [r"росси", r"\brussia\b", r"\bрф\b"]),
    ("Kazakhstan", [r"казахстан", r"\bkazakhstan\b"]),
    ("Belarus", [r"беларус", r"\bbelarus\b"]),
    ("Armenia", [r"армени", r"\barmenia\b"]),
    ("Uzbekistan", [r"узбекистан", r"\buzbekistan\b"]),
    ("UAE", [r"\buae\b", r"dubai", r"abu dhabi"]),
    ("Poland", [r"\bpoland\b", r"польша", r"warsaw"]),
    ("Lithuania", [r"\blithuania\b", r"vilnius", r"литва"]),
    ("Ukraine", [r"\bukraine\b", r"киев", r"kyiv", r"украин"]),
]

INDUSTRY_RULES = [
    ("IT/Software", [r"\bit\b", r"software", r"saas", r"dev", r"product"]),
    ("Banking/FinTech", [r"fintech", r"bank", r"бан", r"payment"]),
    ("E-commerce/Retail", [r"e-?commerce", r"retail", r"маркетплейс"]),
    ("Telecom", [r"telecom", r"связ"]),
    ("Healthcare/Pharma", [r"health", r"pharma", r"мед", r"клиник"]),
    ("Consulting", [r"consult", r"консалт"]),
    ("Manufacturing/Industrial", [r"manufact", r"industrial", r"завод", r"производ"]),
    ("Media/Marketing", [r"media", r"marketing", r"реклама", r"pr"]),
    ("Energy/Oil&Gas", [r"energy", r"oil", r"gas", r"нефт", r"энерг"]),
    ("Education/EdTech", [r"edtech", r"education", r"образован"]),
    ("Logistics/Transport", [r"logistic", r"transport", r"авиа", r"доставк"]),
    ("Gov/Public", [r"gov", r"public", r"министер", r"гос"]),
]

ROLE_RULES = [
    ("Data/ML/Analytics", [r"data", r"analyst", r"analytics", r"bi", r"ml", r"ai", r"аналит", r"данн"]),
    (
        "Product/Project",
        [r"product", r"project", r"program", r"pmo", r"delivery", r"продукт", r"проект", r"scrum"],
    ),
    (
        "Engineering/Software",
        [r"developer", r"engineer", r"backend", r"frontend", r"full stack", r"devops", r"sre", r"разработчик", r"инженер"],
    ),
    ("Design", [r"design", r"designer", r"ux", r"ui", r"дизайн", r"дизайнер"]),
    (
        "Marketing/Sales",
        [r"marketing", r"sales", r"business development", r"account manager", r"маркет", r"продаж"],
    ),
    ("Finance/Legal/HR", [r"finance", r"cfo", r"legal", r"hr", r"recruit", r"финанс", r"юрист", r"кадр"]),
    (
        "Operations/Admin",
        [r"operations", r"operation", r"admin", r"supply chain", r"логист", r"операцион", r"администр"],
    ),
]

LEADERSHIP_PATTERNS = [
    r"\bhead\b",
    r"\blead\b",
    r"\bdirector\b",
    r"\bchief\b",
    r"\bvp\b",
    r"\bceo\b",
    r"\bcto\b",
    r"\bcfo\b",
    r"руководител",
    r"директор",
    r"начальник",
]

TECH_TOOL_PATTERNS = [
    r"python",
    r"java",
    r"javascript",
    r"typescript",
    r"sql",
    r"postgres",
    r"mysql",
    r"oracle",
    r"spark",
    r"hadoop",
    r"tableau",
    r"power\s*bi",
    r"excel",
    r"docker",
    r"kubernetes",
    r"aws",
    r"gcp",
    r"azure",
    r"git",
    r"jira",
    r"confluence",
    r"figma",
    r"react",
    r"angular",
    r"node",
    r"linux",
    r"sap",
    r"1c",
    r"bitrix",
    r"etl",
    r"airflow",
]

SKILL_FAMILY_RULES = [
    ("Data Engineering", [r"etl", r"spark", r"airflow", r"dwh", r"dbt", r"kafka"]),
    ("BI/Analytics", [r"tableau", r"power\s*bi", r"bi", r"dashboard", r"аналит"]),
    ("ML/AI", [r"machine learning", r"ml", r"ai", r"nlp", r"llm"]),
    ("Project/Product", [r"project", r"product", r"scrum", r"kanban", r"agile", r"pmo"]),
    ("Engineering", [r"python", r"java", r"react", r"frontend", r"backend", r"api"]),
    ("DevOps", [r"docker", r"kubernetes", r"devops", r"ci/cd", r"aws", r"azure"]),
    ("Design", [r"ux", r"ui", r"figma", r"adobe", r"design"]),
    ("Marketing/Sales", [r"marketing", r"sales", r"crm", r"seo", r"smm"]),
    ("Finance", [r"ifrs", r"финанс", r"budget", r"audit", r"tax"]),
]

STOPWORDS = {
    "and",
    "the",
    "for",
    "with",
    "from",
    "using",
    "work",
    "years",
    "year",
    "опыт",
    "лет",
    "год",
    "года",
    "работа",
    "работы",
    "компания",
    "company",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "nan", "n/a", "na", "-"}:
        return ""
    return text


def normalize_region(region: object) -> str:
    text = clean_text(region)
    if not text:
        return "Not specified"
    t = text.lower().replace("ё", "е")

    for key, val in REGION_REPLACEMENTS.items():
        if key in t:
            return val
    return text


def guess_country(region_norm: object) -> str:
    text = clean_text(region_norm)
    if not text:
        return "Unknown"
    t = text.lower().replace("ё", "е")

    for country, patterns in COUNTRY_PATTERNS:
        if any(re.search(p, t) for p in patterns):
            return country
    return "Unknown"


def normalize_industry(industry: object) -> str:
    text = clean_text(industry)
    if not text:
        return "Not specified"

    t = text.lower()
    if t == "other":
        return "Other"

    for label, patterns in INDUSTRY_RULES:
        if any(re.search(p, t) for p in patterns):
            return label
    return text


def infer_role_family(selected_position: object, current_title: object, specialist_category: object) -> str:
    text = " ".join([clean_text(selected_position), clean_text(current_title), clean_text(specialist_category)]).lower()
    if not text:
        return "Other"

    for label, patterns in ROLE_RULES:
        if any(re.search(p, text) for p in patterns):
            return label
    return "Other"


def leadership_score(title: object, exp_years: float) -> Tuple[int, str]:
    text = clean_text(title).lower()
    score = 0
    if any(re.search(p, text) for p in LEADERSHIP_PATTERNS):
        score += 2

    if pd.notna(exp_years):
        if exp_years >= 10:
            score += 1
        if exp_years >= 15:
            score += 1

    if score >= 3:
        level = "High"
    elif score >= 2:
        level = "Medium"
    else:
        level = "Low"
    return score, level


def experience_bin(exp_years: float) -> str:
    if pd.isna(exp_years):
        return "Unknown"
    v = float(exp_years)
    if v <= 2:
        return "0-2"
    if v <= 5:
        return "3-5"
    if v <= 9:
        return "6-9"
    if v <= 14:
        return "10-14"
    return "15+"


def _split_tokens(raw: object) -> List[str]:
    text = clean_text(raw)
    if not text:
        return []

    text = text.replace("[", " ").replace("]", " ").replace('"', " ").replace("'", " ")
    chunks = re.split(r"[;,\n\|/]+", text)
    tokens = []
    for ch in chunks:
        token = re.sub(r"\s+", " ", ch).strip(" .:-_()")
        if len(token) < 2:
            continue
        tokens.append(token)
    return tokens


def normalize_token(token: str) -> str:
    t = token.lower().strip()
    t = t.replace("ё", "е")
    t = re.sub(r"\s+", " ", t)
    return t


def is_tool_token(token: str) -> bool:
    t = normalize_token(token)
    return any(re.search(p, t) for p in TECH_TOOL_PATTERNS)


def skill_family(token: str) -> str:
    t = normalize_token(token)
    for family, patterns in SKILL_FAMILY_RULES:
        if any(re.search(p, t) for p in patterns):
            return family
    return "General"


def extract_user_skills_tools(overall_skills: object, overall_tools: object, overall_summary: object = None) -> Dict[str, object]:
    skill_tokens = _split_tokens(overall_skills)
    tool_tokens = _split_tokens(overall_tools)

    # fallback: if tools empty but some technical tokens are in skills
    normalized_skills = [normalize_token(t) for t in skill_tokens]
    normalized_tools = [normalize_token(t) for t in tool_tokens]

    for token in normalized_skills:
        if is_tool_token(token):
            normalized_tools.append(token)

    # keep tokens deduplicated but stable
    seen = set()
    skills_final = []
    for token in normalized_skills:
        if token in seen:
            continue
        seen.add(token)
        skills_final.append(token)

    seen_tools = set()
    tools_final = []
    for token in normalized_tools:
        if token in seen_tools:
            continue
        seen_tools.add(token)
        tools_final.append(token)

    families = [skill_family(tok) for tok in skills_final + tools_final]
    family_counter = Counter(families)
    top_family = family_counter.most_common(1)[0][0] if family_counter else "General"

    return {
        "skills_list": skills_final,
        "tools_list": tools_final,
        "skills_count": len(skills_final),
        "tools_count": len(tools_final),
        "top_skill_family": top_family,
    }


def build_skill_aggregates(user_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    skill_counter = Counter()
    tool_counter = Counter()
    family_counter = Counter()
    pair_counter = Counter()

    for _, row in user_df.iterrows():
        skills = row.get("skills_list", []) or []
        tools = row.get("tools_list", []) or []
        fam = row.get("top_skill_family", "General")

        for s in skills:
            if s and s not in STOPWORDS:
                skill_counter[s] += 1
        for t in tools:
            if t and t not in STOPWORDS:
                tool_counter[t] += 1
        family_counter[fam] += 1

        top_tokens = sorted(set([x for x in skills + tools if x and x not in STOPWORDS]))[:20]
        for a, b in combinations(top_tokens, 2):
            pair_counter[(a, b)] += 1

    skills_df = (
        pd.DataFrame(skill_counter.items(), columns=["token", "count"]).sort_values("count", ascending=False).reset_index(drop=True)
    )
    tools_df = (
        pd.DataFrame(tool_counter.items(), columns=["token", "count"]).sort_values("count", ascending=False).reset_index(drop=True)
    )
    families_df = (
        pd.DataFrame(family_counter.items(), columns=["skill_family", "count"]).sort_values("count", ascending=False).reset_index(drop=True)
    )
    pairs_df = (
        pd.DataFrame([(a, b, c) for (a, b), c in pair_counter.items()], columns=["token_a", "token_b", "count"])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "skills_top": skills_df,
        "tools_top": tools_df,
        "skill_families": families_df,
        "cooccurrence_pairs": pairs_df,
    }


def make_text_corpus(df: pd.DataFrame) -> pd.Series:
    cols = [
        "selectedPosition",
        "talentCard.overall_summary",
        "talentCard.overall_skills",
        "current_job_title",
        "current_industry",
    ]
    text = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    return text


def _row_l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms


def _spherical_kmeans(X: np.ndarray, k: int, n_iter: int = 30, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    centers = X[idx].copy()
    centers = _row_l2_normalize(centers)

    labels = np.zeros(n, dtype=int)
    prev_labels = None

    for _ in range(n_iter):
        sims = X @ centers.T
        labels = np.argmax(sims, axis=1)
        if prev_labels is not None and np.array_equal(labels, prev_labels):
            break
        prev_labels = labels.copy()

        for c in range(k):
            members = X[labels == c]
            if len(members) == 0:
                centers[c] = X[rng.integers(0, n)]
            else:
                centers[c] = members.mean(axis=0)
        centers = _row_l2_normalize(centers)

    sims = X @ centers.T
    assigned = sims[np.arange(n), labels]
    inertia = float(np.sum(1 - assigned))
    return labels, centers, inertia


def choose_k_for_kmeans(X_dense: np.ndarray, k_values: Sequence[int], random_state: int = 42) -> int:
    best_k = k_values[0]
    best_inertia = float("inf")
    for k in k_values:
        if k >= X_dense.shape[0] or k < 2:
            continue
        _, _, inertia = _spherical_kmeans(X_dense, k=k, random_state=random_state, n_iter=25)
        if inertia < best_inertia:
            best_inertia = inertia
            best_k = k
    return best_k


def cluster_profiles(
    df: pd.DataFrame,
    min_k: int = 6,
    max_k: int = 10,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    corpus = make_text_corpus(df)

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        token_pattern=r"(?u)\b[\w\-]{3,}\b",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.9,
    )

    X_word = word_vec.fit_transform(corpus)
    X_char = char_vec.fit_transform(corpus)
    X = hstack([X_word, X_char]).tocsr()
    X_dense = X.astype(np.float32).toarray()
    X_dense = _row_l2_normalize(X_dense)

    max_k = max(min_k, min(max_k, max(2, min(12, X.shape[0] - 1))))
    k_values = list(range(min_k, max_k + 1)) if max_k >= min_k else [min_k]
    k_values = [k for k in k_values if k < X.shape[0] and k >= 2]
    if not k_values:
        k_values = [2]

    k_opt = choose_k_for_kmeans(X_dense, k_values, random_state=random_state)
    labels, centers, _ = _spherical_kmeans(X_dense, k=k_opt, random_state=random_state, n_iter=35)

    # cluster top terms from word-level centroid part
    terms = np.array(word_vec.get_feature_names_out())
    centroids_word = centers[:, : len(terms)]

    rows = []
    for cl in range(k_opt):
        top_idx = np.argsort(centroids_word[cl])[-12:][::-1]
        top_terms = [t for t in terms[top_idx] if t not in STOPWORDS][:8]
        rows.append({"cluster_id": cl, "top_terms": ", ".join(top_terms)})

    terms_df = pd.DataFrame(rows)

    labels_s = pd.Series(labels, index=df.index, name="cluster_id")
    tmp = df.copy()
    tmp["cluster_id"] = labels_s

    summary = (
        tmp.groupby("cluster_id")
        .agg(
            users=("user_hash", "count"),
            median_experience_years=("total_experience_years", "median"),
            top_role_family=("role_family", lambda s: s.value_counts().index[0] if len(s.value_counts()) else "n/a"),
            top_industry=("current_industry", lambda s: s.value_counts().index[0] if len(s.value_counts()) else "n/a"),
            top_country=("country_guess", lambda s: s.value_counts().index[0] if len(s.value_counts()) else "n/a"),
        )
        .reset_index()
    )

    summary["share_%"] = (summary["users"] / summary["users"].sum() * 100).round(1)
    summary["median_experience_years"] = summary["median_experience_years"].round(1)
    summary = summary.sort_values("users", ascending=False)

    summary = summary.merge(terms_df, on="cluster_id", how="left")
    return labels_s, summary, terms_df


def build_profile_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["summary_present"] = out["talentCard.overall_summary"].fillna("").str.strip().ne("").astype(int)
    out["skills_present"] = out["talentCard.overall_skills"].fillna("").str.strip().ne("").astype(int)
    out["jobs_present"] = (out["jobs_count"] > 0).astype(int)

    out["resp_cov"] = out["resp_coverage"].clip(0, 1)
    out["ach_cov"] = out["ach_coverage"].clip(0, 1)
    out["ach_num_cov"] = out["ach_numbers_coverage"].clip(0, 1)
    out["period_parse_success"] = (1 - out["period_parse_failed_share"].fillna(1)).clip(0, 1)
    out["analysis_present"] = out["cvAnalysisResult"].fillna("").str.strip().ne("").astype(int)

    out["profile_quality_score"] = (
        out["summary_present"] * 20
        + out["skills_present"] * 15
        + out["jobs_present"] * 15
        + out["resp_cov"] * 15
        + out["ach_cov"] * 10
        + out["ach_num_cov"] * 10
        + out["period_parse_success"] * 10
        + out["analysis_present"] * 5
    ).round(1)

    out["profile_quality_bucket"] = pd.cut(
        out["profile_quality_score"],
        bins=[-1, 40, 60, 75, 90, 101],
        labels=["0-40", "41-60", "61-75", "76-90", "91-100"],
    ).astype(str)

    return out


def crosstab_share(df: pd.DataFrame, idx: str, cols: str) -> pd.DataFrame:
    ct = pd.crosstab(df[idx], df[cols])
    share = (ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(1)
    return share.fillna(0)
