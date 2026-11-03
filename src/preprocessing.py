import numpy as np
import pandas as pd

RAW_DATA_PATH = "data/raw/github_repos.csv"
PROCESSED_DATA_PATH = "data/processed/github_repos_clean.csv"


def calculate_quality_score(df: pd.DataFrame, threshold: float = 42.5) -> pd.DataFrame:
    """Calculates 100-pt quality score components and assigns the binary is_high_quality label."""
    df = df.copy()
    now = pd.Timestamp.now(tz="UTC")

    # 1. Transform dates into clean day numbers FIRST (done once, used everywhere)
    df["repo_age_days"] = (now - pd.to_datetime(df["created_at"], utc=True)).dt.days
    df["days_since_last_push"] = (now - pd.to_datetime(df["pushed_at"], utc=True)).dt.days
    days_since_update = (now - pd.to_datetime(df["updated_at"], utc=True)).dt.days

    # 2. Quality Score Components
    # Stars (20 pts, log scale)
    star_score = np.clip(20.0 * (np.log10(1 + df["stargazers_count"]) / np.log10(1 + 10000)), 0, 20)

    # Forks (25 pts, log scale)
    fork_score = np.clip(25.0 * (np.log10(1 + df["forks_count"]) / np.log10(1 + 2000)), 0, 25)

    # Pushed At (25 pts, recency)
    pushed_score = np.select(
        [df["days_since_last_push"] <= 30, df["days_since_last_push"] <= 90, 
         df["days_since_last_push"] <= 180, df["days_since_last_push"] <= 365],
        [25.0, 18.0, 10.0, 4.0],
        default=0.0
    )

    # Updated At (5 pts, recency)
    updated_score = np.select(
        [days_since_update <= 7, days_since_update <= 30, days_since_update <= 90],
        [5.0, 3.0, 1.0],
        default=0.0
    )

    # License (15 pts) & Owner Type (10 pts)
    license_score = np.where(df["has_license"] == True, 15.0, 0.0)
    owner_score = np.where(df["owner_type"] == "Organization", 10.0, 0.0)

    # Total Score & Target Label
    quality_score = star_score + fork_score + pushed_score + updated_score + license_score + owner_score
    df["is_high_quality"] = (quality_score >= threshold).astype(int)

    return df


def preprocess_data(df: pd.DataFrame, threshold: float = 42.5) -> pd.DataFrame:
    # Filter strictly to the 3 target languages (drops 1 Java repo and 12 NaNs)
    df = df[df["language"].isin(["Python", "JavaScript", "TypeScript"])].copy()

    # Calculate quality rubric and target label
    df = calculate_quality_score(df, threshold=threshold)

    # 3. Encode Categoricals & Booleans
    lang_dummies = pd.get_dummies(df["language"], prefix="lang", dtype=int)
    df = pd.concat([df, lang_dummies], axis=1)

    df["owner_type_org"] = (df["owner_type"] == "Organization").astype(int)
    for col in ["has_license", "has_wiki", "has_pages", "has_projects", "has_discussions"]:
        df[col] = df[col].astype(int)

    # 4. Drop Raw String & ID Columns (prevent leakage)
    cols_to_drop = [
        "id", "name", "full_name", "created_at", "pushed_at", 
        "updated_at", "default_branch", "license_key", "owner_type", "language"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df


def main():
    # Dead simple: load -> clean -> save
    print("Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Preprocessing...")
    df_clean = preprocess_data(df, threshold=42.5)  
    # 42.5 is the median for the quality score, so it divides our dataset into 50/50 good and bad quality repos

    print("Saving processed data...")
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Done! Clean dataset saved with shape: {df_clean.shape}")


if __name__ == "__main__":
    main()
