import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import joblib

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import extract_features_from_dict, TIER_MAP

ADVERSARIAL_TESTS = [
    {
        "id": "ATTACK-01",
        "name": "The $50 Fake Star-Bomb",
        "vector": "Buy 10,000 bot stars on an abandoned, unformatted junk repository.",
        "baseline": {
            "stars": 5, "forks": 0, "open_issues": 0, "age_days": 600, "push_days": 500,
            "is_org": 0, "has_license": 0, "is_osi": 0, "desc_len": 0, "topics_cnt": 0, "lang": "Python"
        },
        "manipulation": {"stars": 10000},
        "defense_criteria": "Must NOT reach Tier 2 (Lack of forks, dead push, and no license must prevent Tier 2)."
    },
    {
        "id": "ATTACK-02",
        "name": "The Zombie Commit Bot",
        "vector": "Set up a daily cron bot pushing empty commits to an abandoned 0-star personal repo.",
        "baseline": {
            "stars": 0, "forks": 0, "open_issues": 0, "age_days": 400, "push_days": 400,
            "is_org": 0, "has_license": 0, "is_osi": 0, "desc_len": 0, "topics_cnt": 0, "lang": "Python"
        },
        "manipulation": {"push_days": 0},
        "defense_criteria": "Must NOT reach Tier 2 (Green squares alone cannot replace traction or hygiene)."
    },
    {
        "id": "ATTACK-03",
        "name": "The SEO / Topic Stuffer",
        "vector": "Stuff 20 trending buzzword topics and 500-char marketing description on an empty repo.",
        "baseline": {
            "stars": 2, "forks": 0, "open_issues": 0, "age_days": 200, "push_days": 200,
            "is_org": 0, "has_license": 0, "is_osi": 0, "desc_len": 0, "topics_cnt": 0, "lang": "Python"
        },
        "manipulation": {"desc_len": 500, "topics_cnt": 20},
        "defense_criteria": "Must NOT reach Tier 2 (Keywords cannot substitute for real community scale)."
    },
    {
        "id": "ATTACK-04",
        "name": "The Synthetic Fork Farm",
        "vector": "A botnet generates 5,000 forks on a 2-star repo (abnormal fork_to_star_ratio = 1667.0).",
        "baseline": {
            "stars": 2, "forks": 0, "open_issues": 0, "age_days": 300, "push_days": 5,
            "is_org": 0, "has_license": 1, "is_osi": 1, "desc_len": 50, "topics_cnt": 2, "lang": "Python"
        },
        "manipulation": {"forks": 5000},
        "defense_criteria": "Evaluate whether abnormal fork inflation bypasses model checks."
    },
    {
        "id": "ATTACK-05",
        "name": "License Stripping Sabotage",
        "vector": "Maintainer deletes the open-source license from Facebook React (250k stars, active).",
        "baseline": {
            "stars": 252000, "forks": 46000, "open_issues": 1500, "age_days": 4900, "push_days": 0,
            "is_org": 1, "has_license": 1, "is_osi": 1, "desc_len": 150, "topics_cnt": 10, "lang": "JavaScript"
        },
        "manipulation": {"has_license": 0, "is_osi": 0},
        "defense_criteria": "Measure sensitivity: does removing OSI compliance impact confidence?"
    }
]


def evaluate_attack(test_case: Dict[str, Any], model) -> Dict[str, Any]:
    # 1. Evaluate clean baseline via centralized features module
    clean_dict = test_case["baseline"].copy()
    X_clean = extract_features_from_dict(clean_dict)
    proba_clean = model.predict_proba(X_clean)[0]
    pred_clean = int(np.argmax(proba_clean))

    # 2. Evaluate attacked state via centralized features module
    attacked_dict = clean_dict.copy()
    attacked_dict.update(test_case["manipulation"])
    X_attack = extract_features_from_dict(attacked_dict)
    proba_attack = model.predict_proba(X_attack)[0]
    pred_attack = int(np.argmax(proba_attack))

    # Assessment logic
    if test_case["id"] in ["ATTACK-01", "ATTACK-02", "ATTACK-03"]:
        defended = (pred_attack < 2)
        status = "DEFENDED (Resisted Tier 2)" if defended else "EXPLOITED (Fooled to Tier 2)"
    elif test_case["id"] == "ATTACK-04":
        defended = (pred_attack < 2)
        status = "DEFENDED (Resisted Tier 2)" if defended else "EXPLOITED (Forks elevated to Tier 2)"
    elif test_case["id"] == "ATTACK-05":
        p2_drop = proba_clean[2] - proba_attack[2]
        status = f"SENSITIVE (P2 dropped by {p2_drop*100:.2f}%)" if p2_drop > 0 else "INSENSITIVE"

    return {
        "id": test_case["id"],
        "name": test_case["name"],
        "vector": test_case["vector"],
        "clean_tier": pred_clean,
        "clean_conf": proba_clean[pred_clean],
        "clean_p": proba_clean,
        "attack_tier": pred_attack,
        "attack_conf": proba_attack[pred_attack],
        "attack_p": proba_attack,
        "status": status,
        "criteria": test_case["defense_criteria"]
    }


def main():
    print("=" * 115)
    print(" SUITE 3: ADVERSARIAL & GAMING ROBUSTNESS (BOUNDARY STRESS-TEST)")
    print("=" * 115)
    print("Testing Tuned LightGBM v2 against artificial metadata manipulation, star-bombing, and sabotage.")

    model_path = ROOT_DIR / "models" / "lightgbm_v2.pkl"
    if not model_path.exists():
        print(f"Error: Champion model artifact {model_path} not found.")
        return

    model = joblib.load(model_path)

    results = []
    print(f"{'Attack ID':<10} | {'Scenario Name':<28} | {'Clean State':<12} | {'Attacked State':<12} | {'Shift in P(Tier 2)':<19} | {'Defense Status'}")
    print("-" * 115)

    for tc in ADVERSARIAL_TESTS:
        res = evaluate_attack(tc, model)
        results.append(res)

        clean_str = f"T{res['clean_tier']} ({res['clean_conf']*100:.1f}%)"
        attack_str = f"T{res['attack_tier']} ({res['attack_conf']*100:.1f}%)"
        delta_p2 = (res['attack_p'][2] - res['clean_p'][2]) * 100
        delta_str = f"{delta_p2:+6.2f}%"

        print(f"{res['id']:<10} | {res['name']:<28} | {clean_str:<12} | {attack_str:<12} | {delta_str:<19} | {res['status']}")

    # Detailed Scenario Breakdowns
    print("\n" + "=" * 115)
    print(" DETAILED ATTACK ANALYSIS & FINDINGS")
    print("=" * 115)

    for r in results:
        print(f"\n[{r['id']}: {r['name'].upper()}]")
        print(f"  * Attack Vector:    {r['vector']}")
        print(f"  * Baseline Score:   Tier {r['clean_tier']} [P0={r['clean_p'][0]*100:.1f}%, P1={r['clean_p'][1]*100:.1f}%, P2={r['clean_p'][2]*100:.1f}%]")
        print(f"  * Attacked Score:   Tier {r['attack_tier']} [P0={r['attack_p'][0]*100:.1f}%, P1={r['attack_p'][1]*100:.1f}%, P2={r['attack_p'][2]*100:.1f}%]")
        print(f"  * Security Verdict: {r['status']}")
        print(f"  * Defense Criteria: {r['criteria']}")

    # Final Summary Banner
    print("\n" + "=" * 115)
    print(" 3-PHASE BENCHMARK SUITE STATUS")
    print("=" * 115)
    print(" [✓] Suite 1: Landmark Archetypes Benchmark  --> COMPLETE")
    print(" [✓] Suite 2: Temporal Lifecycle Simulation  --> COMPLETE")
    print(" [✓] Suite 3: Adversarial & Gaming Robustness --> COMPLETE")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()
