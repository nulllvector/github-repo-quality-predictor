import sys
from typing import Dict, Any, Set, List
import numpy as np
import pandas as pd

# Set UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Recognized OSI-Approved Open Source Licenses
OSI_LICENSES: Set[str] = {
    "mit", "apache-2.0", "gpl-3.0", "bsd-3-clause", "agpl-3.0",
    "bsd-2-clause", "isc", "gpl-2.0", "mpl-2.0", "unlicense", "cc0-1.0", "lgpl-3.0"
}

# 2. Canonical Feature Order Expected by Trained Models
FEATURE_ORDER: List[str] = [
    'stargazers_count', 'forks_count', 'open_issues_count', 'size_kb',
    'has_wiki', 'has_pages', 'has_projects', 'has_discussions',
    'has_license', 'topics_count', 'description_length', 'repo_age_days',
    'days_since_last_push', 'log_star_velocity', 'log_fork_velocity',
    'fork_to_star_ratio', 'issue_to_fork_ratio', 'maintenance_ratio',
    'has_description', 'has_topics', 'license_type_osi',
    'license_type_custom', 'lang_JavaScript', 'lang_Python',
    'lang_TypeScript', 'owner_type_org'
]

# 3. Standard Quality Tier Mappings
TIER_MAP: Dict[int, str] = {
    0: "Tier 0 (Low / Dormant)",
    1: "Tier 1 (Emerging / Moderate)",
    2: "Tier 2 (Production-Grade)"
}


def compute_velocity(count: int, age_days: int, prior_days: int = 30) -> float:
    """
    Computes Bayesian-smoothed logarithmic velocity.
    Formula: log10(1 + count) / log10(1 + age_days + prior_days)
    The +30d prior dampens noise from 1-day old repositories with few stars.
    """
    effective_age = max(age_days, 1)
    return float(np.log10(1 + count) / np.log10(1 + effective_age + prior_days))


def extract_features_from_dict(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Unified Single Source of Truth for Feature Extraction.
    Accepts either:
      1. A raw GitHub REST API dictionary (with nested 'owner', 'license', ISO timestamps).
      2. A flat snapshot dictionary (used in lifecycle and adversarial stress testing).
    Returns a standardized 1-row DataFrame indexed strictly by FEATURE_ORDER.
    """
    now = pd.Timestamp.now(tz="UTC")

    # --- 1. Temporal Dynamics ---
    if "created_at" in data:
        created_at = pd.to_datetime(data.get("created_at"), utc=True)
        repo_age_days = max(int((now - created_at).total_seconds() // 86400), 1)
    else:
        repo_age_days = max(int(data.get("age_days", 1)), 1)

    if "pushed_at" in data:
        pushed_at = pd.to_datetime(data.get("pushed_at"), utc=True)
        days_since_push = max(int((now - pushed_at).total_seconds() // 86400), 0)
    else:
        days_since_push = max(int(data.get("push_days", 0)), 0)

    # --- 2. Raw Metrics ---
    stargazers = int(data.get("stargazers_count", data.get("stars", 0)))
    forks = int(data.get("forks_count", data.get("forks", 0)))
    open_issues = int(data.get("open_issues_count", data.get("open_issues", 0)))
    size_kb = int(data.get("size", data.get("size_kb", 2000)))

    # --- 3. License Identification ---
    if "license" in data and isinstance(data["license"], dict):
        lic_key = (data["license"].get("key") or "none").lower()
        has_license = int(lic_key != "none")
        license_osi = int(lic_key in OSI_LICENSES)
        license_custom = int(has_license == 1 and license_osi == 0)
    else:
        has_license = int(data.get("has_license", 0))
        license_osi = int(data.get("is_osi", data.get("license_type_osi", 0)))
        license_custom = int(has_license == 1 and license_osi == 0)

    # --- 4. Content & Packaging ---
    desc = data.get("description", "")
    desc_len = len(desc) if isinstance(desc, str) else int(data.get("desc_len", data.get("description_length", 0)))

    topics = data.get("topics", [])
    topics_cnt = len(topics) if isinstance(topics, list) else int(data.get("topics_cnt", data.get("topics_count", 0)))

    # --- 5. Velocity & Interaction Ratios ---
    log_star_vel = compute_velocity(stargazers, repo_age_days, prior_days=30)
    log_fork_vel = compute_velocity(forks, repo_age_days, prior_days=30)
    fork_star_ratio = float((forks + 1) / (stargazers + 1))
    issue_fork_ratio = float((open_issues + 1) / (forks + 1))
    maintenance_ratio = float(np.clip(days_since_push / (repo_age_days + 1), 0.0, 1.0))

    # --- 6. Governance & Ecosystem ---
    owner = data.get("owner", {})
    if isinstance(owner, dict):
        owner_org = int(owner.get("type") == "Organization")
    else:
        owner_org = int(data.get("is_org", data.get("owner_type_org", 0)))

    lang = data.get("language", data.get("lang", ""))

    features = {
        'stargazers_count': stargazers,
        'forks_count': forks,
        'open_issues_count': open_issues,
        'size_kb': size_kb,
        'has_wiki': int(bool(data.get("has_wiki", 0))),
        'has_pages': int(bool(data.get("has_pages", 0))),
        'has_projects': int(bool(data.get("has_projects", 0))),
        'has_discussions': int(bool(data.get("has_discussions", 0))),
        'has_license': has_license,
        'topics_count': topics_cnt,
        'description_length': desc_len,
        'repo_age_days': repo_age_days,
        'days_since_last_push': days_since_push,
        'log_star_velocity': log_star_vel,
        'log_fork_velocity': log_fork_vel,
        'fork_to_star_ratio': fork_star_ratio,
        'issue_to_fork_ratio': issue_fork_ratio,
        'maintenance_ratio': maintenance_ratio,
        'has_description': int(desc_len > 0),
        'has_topics': int(topics_cnt > 0),
        'license_type_osi': license_osi,
        'license_type_custom': license_custom,
        'lang_JavaScript': int(lang == "JavaScript"),
        'lang_Python': int(lang == "Python"),
        'lang_TypeScript': int(lang == "TypeScript"),
        'owner_type_org': owner_org
    }

    return pd.DataFrame([features])[FEATURE_ORDER]
