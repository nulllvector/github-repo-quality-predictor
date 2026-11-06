import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import requests
import joblib
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import extract_features_from_dict, TIER_MAP

load_dotenv(ROOT_DIR / ".env")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

BENCHMARK_ROSTER: List[Dict[str, str]] = [
    {"repo": "facebook/react",          "category": "Production Titan"},
    {"repo": "torvalds/linux",          "category": "Production Titan"},
    {"repo": "psf/black",               "category": "Production Titan"},
    {"repo": "kubernetes/kubernetes",   "category": "Production Titan"},
    {"repo": "astral-sh/uv",            "category": "Viral Rocket (Young)"},
    {"repo": "ollama/ollama",           "category": "Viral Rocket (Young)"},
    {"repo": "vllm-project/vllm",       "category": "Viral Rocket (Young)"},
    {"repo": "angular/angular.js",      "category": "Dormant Legend"},
    {"repo": "request/request",         "category": "Dormant Legend"},
    {"repo": "moment/moment",           "category": "Dormant Legend"},
    {"repo": "fastapi/typer",           "category": "Established Utility"},
    {"repo": "samuelcolvin/watchfiles", "category": "Specialized Tool"},
    {"repo": "encode/databases",        "category": "Specialized Tool"},
    {"repo": "octocat/Hello-World",     "category": "Tutorial / Scrapbook"},
    {"repo": "github/gitignore",        "category": "Static Resource"}
]


def fetch_metadata(repo_name: str) -> Dict[str, Any]:
    """Fetches real-time repository metadata via GitHub REST API."""
    url = f"https://api.github.com/repos/{repo_name}"
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Quality-Predictor"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    response = requests.get(url, headers=headers, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(f"GitHub API Error {response.status_code}: {response.text}")
    return response.json()


def diagnose_prediction(pred_tier: int, X) -> str:
    """Generates an automated, human-interpretable diagnosis of primary feature drivers."""
    stars = X['stargazers_count'].iloc[0]
    forks = X['forks_count'].iloc[0]
    push_days = X['days_since_last_push'].iloc[0]
    age_days = X['repo_age_days'].iloc[0]

    if pred_tier == 2:
        if age_days < 1500 and stars > 50000:
            return "Extreme adoption velocity successfully decoupled young repo age"
        elif push_days > 365:
            return f"Massive historical scale ({stars:,}★, {forks:,}⑂) outscores {push_days}d inactivity penalty"
        else:
            return "High community scale + active push recency + verified license"
    elif pred_tier == 1:
        if push_days > 365 and stars > 1000:
            return f"Moderate historical stars ({stars:,}★) elevated above Tier 0 despite {push_days}d inactivity"
        else:
            return "Solid developer hygiene and maintenance with moderate adoption scale"
    else:
        return "Low social adoption combined with stagnant maintenance activity"


def main():
    print("=" * 115)
    print(" SUITE 1: LANDMARK ARCHETYPES BENCHMARK EVALUATION (Tuned LightGBM v2)")
    print("=" * 115)

    model_path = ROOT_DIR / "models" / "lightgbm_v2.pkl"
    if not model_path.exists():
        print(f"Error: Champion model artifact not found at {model_path}.")
        return

    model = joblib.load(model_path)

    header = f"{'Repository':<24} | {'Archetype':<22} | {'Stars':>7} | {'Push(d)':>7} | {'Predicted Tier':<19} | {'Conf':<6} | {'Diagnosis'}"
    print(header)
    print("-" * 115)

    for item in BENCHMARK_ROSTER:
        repo = item["repo"]
        try:
            raw = fetch_metadata(repo)
            time.sleep(0.2)
        except Exception as err:
            print(f"{repo:<24} | Fetch Error: {err}")
            continue

        X = extract_features_from_dict(raw)
        probabilities = model.predict_proba(X)[0]
        pred_tier = int(np.argmax(probabilities))
        confidence = probabilities[pred_tier]

        diagnosis = diagnose_prediction(pred_tier, X)
        stars_formatted = f"{raw.get('stargazers_count', 0):,}"

        row_str = (
            f"{repo:<24} | {item['category']:<22} | {stars_formatted:>7} | "
            f"{X['days_since_last_push'].iloc[0]:>7} | {TIER_MAP[pred_tier]:<19} | "
            f"{confidence*100:>5.1f}% | {diagnosis}"
        )
        print(row_str)

    print("\n" + "=" * 115)
    print(" 3-PHASE BENCHMARK SUITE STATUS")
    print("=" * 115)
    print(" [✓] Suite 1: Landmark Archetypes Benchmark  --> COMPLETE")
    print(" [ ] Suite 2: Temporal Lifecycle Simulation  --> PENDING (FastAPI from Day 1 to Year 6)")
    print(" [ ] Suite 3: Adversarial & Gaming Robustness --> PENDING (Fake stars & license deletion)")
    print("=" * 115)


if __name__ == "__main__":
    main()
