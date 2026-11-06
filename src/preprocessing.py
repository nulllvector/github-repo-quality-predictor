import os
import sys
import numpy as np
import pandas as pd

# Fix Windows console encoding for terminal outputs
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

RAW_DATA_PATH = "data/raw/github_repos.csv"
V1_PROCESSED_DATA_PATH = "data/processed/github_repos_clean.csv"
V2_PROCESSED_DATA_PATH = "data/processed/github_repos_v2.csv"

# Recognized OSI-approved open source licenses
OSI_LICENSES = {
    "mit", "apache-2.0", "gpl-3.0", "bsd-3-clause", "agpl-3.0",
    "bsd-2-clause", "isc", "gpl-2.0", "mpl-2.0", "unlicense", "cc0-1.0", "lgpl-3.0"
}


# =====================================================================
# Version 1 Legacy Preprocessing (Preserved for backwards compatibility)
# =====================================================================

def calculate_quality_score(df: pd.DataFrame, threshold: float = 42.5) -> pd.DataFrame:
    """Version 1: Calculates 100-pt baseline rubric and assigns binary is_high_quality label."""
    df = df.copy()
    now = pd.Timestamp.now(tz="UTC")

    df["repo_age_days"] = (now - pd.to_datetime(df["created_at"], utc=True)).dt.days
    df["days_since_last_push"] = (now - pd.to_datetime(df["pushed_at"], utc=True)).dt.days
    days_since_update = (now - pd.to_datetime(df["updated_at"], utc=True)).dt.days

    star_score = np.clip(20.0 * (np.log10(1 + df["stargazers_count"]) / np.log10(1 + 10000)), 0, 20)
    fork_score = np.clip(25.0 * (np.log10(1 + df["forks_count"]) / np.log10(1 + 2000)), 0, 25)

    pushed_score = np.select(
        [df["days_since_last_push"] <= 30, df["days_since_last_push"] <= 90, 
         df["days_since_last_push"] <= 180, df["days_since_last_push"] <= 365],
        [25.0, 18.0, 10.0, 4.0],
        default=0.0
    )

    updated_score = np.select(
        [days_since_update <= 7, days_since_update <= 30, days_since_update <= 90],
        [5.0, 3.0, 1.0],
        default=0.0
    )

    license_score = np.where(df["has_license"] == True, 15.0, 0.0)
    owner_score = np.where(df["owner_type"] == "Organization", 10.0, 0.0)

    quality_score = star_score + fork_score + pushed_score + updated_score + license_score + owner_score
    df["is_high_quality"] = (quality_score >= threshold).astype(int)
    return df


def preprocess_data(df: pd.DataFrame, threshold: float = 42.5) -> pd.DataFrame:
    """Version 1: Preprocesses dataset into binary classification format."""
    df = df[df["language"].isin(["Python", "JavaScript", "TypeScript"])].copy()
    df = calculate_quality_score(df, threshold=threshold)

    lang_dummies = pd.get_dummies(df["language"], prefix="lang", dtype=int)
    df = pd.concat([df, lang_dummies], axis=1)

    df["owner_type_org"] = (df["owner_type"] == "Organization").astype(int)
    for col in ["has_license", "has_wiki", "has_pages", "has_projects", "has_discussions"]:
        df[col] = df[col].astype(int)

    cols_to_drop = [
        "id", "name", "full_name", "created_at", "pushed_at", 
        "updated_at", "default_branch", "license_key", "owner_type", "language"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df


# =====================================================================
# Version 2 Upgraded Preprocessing & Rubric
# =====================================================================

def calculate_quality_score_v2(df: pd.DataFrame) -> pd.Series:
    """
    Version 2: Upgraded 100-Point Quality Rubric.
    Weights:
      1. Traction & Velocity (35 pts): Dual-path max(Cumulative, Velocity).
         - Cumulative: Forks (22 pts log) + Stars (13 pts log) -> heavily emphasizes forks.
         - Velocity: log_star_velocity * 25 pts.
      2. Maintenance & Liveness (25 pts): Push recency (20 pts) + Update recency (5 pts).
      3. Open Source Hygiene (15 pts): OSI standard license (15 pts) vs Custom (8 pts) vs None (0 pts).
      4. Documentation & Packaging (15 pts): Description (10 pts) + Curated topics (5 pts).
      5. Governance & Community (10 pts): Org-backed (6 pts) + Discussions/Wiki/Pages (4 pts).
    """
    now = pd.Timestamp.now(tz="UTC")

    # Reuse durations if already computed, else compute once
    repo_age_days = df["repo_age_days"] if "repo_age_days" in df.columns else (now - pd.to_datetime(df["created_at"], utc=True)).dt.days
    days_since_last_push = df["days_since_last_push"] if "days_since_last_push" in df.columns else (now - pd.to_datetime(df["pushed_at"], utc=True)).dt.days
    days_since_update = (now - pd.to_datetime(df["updated_at"], utc=True)).dt.days
    effective_age = np.maximum(repo_age_days, 1)

    # 1. Traction (35 pts) - Heavy on forks, dual-path with Bayesian smoothing (+30d prior)
    s_cum_fork = 22.0 * (np.log10(1 + df["forks_count"]) / np.log10(1 + 2000))
    s_cum_star = 13.0 * (np.log10(1 + df["stargazers_count"]) / np.log10(1 + 10000))
    s_cum = np.clip(s_cum_fork + s_cum_star, 0, 35)

    # Bayesian-smoothed velocity (+30 days prior prevents 1-day 1-star noise explosion)
    log_vel = np.log10(1 + df["stargazers_count"]) / np.log10(1 + effective_age + 30)
    s_vel = np.clip(log_vel * 25.0, 0, 35)
    traction_score = np.maximum(s_cum, s_vel)

    # 2. Maintenance & Liveness (25 pts)
    pushed_score = np.select(
        [days_since_last_push <= 30, days_since_last_push <= 90, 
         days_since_last_push <= 180, days_since_last_push <= 365],
        [20.0, 14.0, 8.0, 3.0], default=0.0
    )
    updated_score = np.select([days_since_update <= 14, days_since_update <= 60], [5.0, 2.0], default=0.0)
    maint_score = pushed_score + updated_score

    # 3. License Type (15 pts)
    lic_series = df["license_key"].fillna("none").str.lower()
    license_score = np.where(lic_series.isin(OSI_LICENSES), 15.0,
                    np.where(lic_series != "none", 8.0, 0.0))

    # 4. Documentation & Packaging (15 pts)
    desc_len = df["description_length"].fillna(0)
    topics_cnt = df["topics_count"].fillna(0)
    desc_score = np.where(desc_len >= 30, 10.0, np.where(desc_len > 0, 5.0, 0.0))
    topics_score = np.where(topics_cnt >= 3, 5.0, np.where(topics_cnt > 0, 2.0, 0.0))
    doc_score = desc_score + topics_score

    # 5. Governance & Community (10 pts)
    org_score = np.where(df["owner_type"] == "Organization", 6.0, 0.0)
    comm_tools = (df["has_discussions"].astype(int) + df["has_wiki"].astype(int) + df["has_pages"].astype(int)) > 0
    comm_score = np.where(comm_tools, 4.0, 0.0)
    gov_score = org_score + comm_score

    return traction_score + maint_score + license_score + doc_score + gov_score


def preprocess_data_v2(df: pd.DataFrame, low_cutoff: float = 40.0, high_cutoff: float = 65.0) -> pd.DataFrame:
    """
    Version 2 Preprocessing Pipeline:
    - Filters target languages
    - Computes 3-tier target via upgraded V2 rubric
    - Engineers velocity, ratio, and license features
    - Drops multicollinear watchers_count and raw IDs
    """
    df = df[df["language"].isin(["Python", "JavaScript", "TypeScript"])].copy()
    now = pd.Timestamp.now(tz="UTC")

    # Time durations
    df["repo_age_days"] = (now - pd.to_datetime(df["created_at"], utc=True)).dt.days
    df["days_since_last_push"] = (now - pd.to_datetime(df["pushed_at"], utc=True)).dt.days
    effective_age = np.maximum(df["repo_age_days"], 1)

    # 1. Target Label Assignment via Upgraded V2 Rubric
    quality_score = calculate_quality_score_v2(df)
    conditions = [
        quality_score < low_cutoff,
        (quality_score >= low_cutoff) & (quality_score < high_cutoff),
        quality_score >= high_cutoff
    ]
    # 0: Low / Dormant, 1: Emerging / Moderate, 2: Production-Grade
    df["quality_tier"] = np.select(conditions, [0, 1, 2], default=0).astype(int)

    # 2. Velocity & Ratio Feature Engineering (with +30d Bayesian smoothing)
    df["log_star_velocity"] = np.log10(1 + df["stargazers_count"]) / np.log10(1 + effective_age + 30)
    df["log_fork_velocity"] = np.log10(1 + df["forks_count"]) / np.log10(1 + effective_age + 30)
    df["fork_to_star_ratio"] = (df["forks_count"] + 1) / (df["stargazers_count"] + 1)
    df["issue_to_fork_ratio"] = (df["open_issues_count"] + 1) / (df["forks_count"] + 1)
    df["maintenance_ratio"] = np.clip(df["days_since_last_push"] / (effective_age + 1), 0.0, 1.0)
    df["has_description"] = (df["description_length"].fillna(0) > 0).astype(int)
    df["has_topics"] = (df["topics_count"].fillna(0) > 0).astype(int)

    # 3. Categorical & License Features
    lic_series = df["license_key"].fillna("none").str.lower()
    df["license_type_osi"] = lic_series.isin(OSI_LICENSES).astype(int)
    df["license_type_custom"] = ((~lic_series.isin(OSI_LICENSES)) & (lic_series != "none")).astype(int)

    lang_dummies = pd.get_dummies(df["language"], prefix="lang", dtype=int)
    df = pd.concat([df, lang_dummies], axis=1)

    df["owner_type_org"] = (df["owner_type"] == "Organization").astype(int)
    for col in ["has_license", "has_wiki", "has_pages", "has_projects", "has_discussions"]:
        df[col] = df[col].astype(int)

    # 4. Clean Drops (prevent leakage & eliminate duplicate watchers_count alias)
    cols_to_drop = [
        "id", "name", "full_name", "created_at", "pushed_at", "updated_at",
        "default_branch", "license_key", "owner_type", "language", "watchers_count"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df


def main():
    print("============================================================")
    print("Running Version 2 Preprocessing Pipeline...")
    print("============================================================")

    print(f"\nLoading raw dataset from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   Raw shape: {df.shape}")

    print("\n[Step 1/4] Calculating Upgraded V2 Quality Rubric & 3-Tier Target...")
    print("[Step 2/4] Engineering Velocity, Ratio, and License Features...")
    print("[Step 3/4] Encoding Categoricals and Dropping Multicollinear Columns...")
    df_v2 = preprocess_data_v2(df, low_cutoff=40.0, high_cutoff=65.0)

    print(f"\n[Step 4/4] Saving V2 processed dataset to: {V2_PROCESSED_DATA_PATH}")
    os.makedirs(os.path.dirname(V2_PROCESSED_DATA_PATH), exist_ok=True)
    df_v2.to_csv(V2_PROCESSED_DATA_PATH, index=False)

    print("\n" + "="*60)
    print("V2 Dataset Successfully Generated!")
    print("="*60)
    print(f"Shape: {df_v2.shape[0]} rows × {df_v2.shape[1]} columns")
    print(f"Null values: {df_v2.isnull().sum().sum()}")
    print("\nTarget Class Distribution:")
    counts = df_v2["quality_tier"].value_counts().sort_index()
    percentages = df_v2["quality_tier"].value_counts(normalize=True).sort_index() * 100
    tier_labels = {0: "Tier 0 (Low / Dormant)", 1: "Tier 1 (Emerging / Moderate)", 2: "Tier 2 (Production-Grade)"}
    for tier in [0, 1, 2]:
        print(f"   {tier_labels[tier]}: {counts[tier]} repos ({percentages[tier]:.2f}%)")

    print("\nEngineered Feature Set (26 features):")
    feature_cols = [c for c in df_v2.columns if c != "quality_tier"]
    print(feature_cols)


if __name__ == "__main__":
    main()