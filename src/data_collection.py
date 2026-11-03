import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("Please set your valid GITHUB_TOKEN in the .env file.")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

BASE_URL = "https://api.github.com/search/repositories"
OUTPUT_FILE = "data/raw/github_repos.csv"


def extract_features(repo: dict) -> dict:
    """Extracts only the fields we care about from a single repo JSON object."""
    return {
        "id": repo.get("id"),
        "name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "owner_type": repo.get("owner", {}).get("type"),
        "stargazers_count": repo.get("stargazers_count", 0),
        "forks_count": repo.get("forks_count", 0),
        "open_issues_count": repo.get("open_issues_count", 0),
        "watchers_count": repo.get("watchers_count", 0),
        "size_kb": repo.get("size", 0),
        "language": repo.get("language"),
        "has_wiki": repo.get("has_wiki", False),
        "has_pages": repo.get("has_pages", False),
        "has_projects": repo.get("has_projects", False),
        "has_discussions": repo.get("has_discussions", False),
        "has_license": bool(repo.get("license")),
        "license_key": repo.get("license", {}).get("key") if repo.get("license") else None,
        "topics_count": len(repo.get("topics", [])),
        "description_length": len(repo.get("description", "")) if repo.get("description") else 0,
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "default_branch": repo.get("default_branch")
    }


def fetch_page(query: str, page: int, per_page: int = 100) -> list:
    """Fetches one page of search results with rate limit handling."""
    params = {
        "q": query,
        "page": page,
        "per_page": per_page
    }

    while True:
        try:
            response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=10)
            
            # Status 200: Success
            if response.status_code == 200:
                data = response.json()
                return data.get("items", [])

            # Status 403: Rate limit reached
            elif response.status_code == 403:
                reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                sleep_duration = max(reset_time - int(time.time()), 2)
                print(f"⏳ Rate limit reached. Sleeping for {sleep_duration} seconds...")
                time.sleep(sleep_duration)
                continue

            # Status 5xx: Server glitch -> retry
            elif response.status_code >= 500:
                print(f"⚠️ Server error {response.status_code}. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            # Other client errors (e.g. 422)
            else:
                print(f"❌ Failed to fetch (Status {response.status_code}): {response.text}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error: {e}. Retrying in 5 seconds...")
            time.sleep(5)


def save_checkpoint(records: list, filepath: str):
    """Appends records to CSV and drops duplicates based on repository ID."""
    new_df = pd.DataFrame(records)
    
    if os.path.exists(filepath):
        existing_df = pd.read_csv(filepath)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["id"])
    else:
        combined_df = new_df.drop_duplicates(subset=["id"])
        
    combined_df.to_csv(filepath, index=False)
    return len(combined_df)


def collect_data():
    """Main collection pipeline iterating through languages, star ranges, and date windows."""
    languages = ["python", "javascript", "typescript"]
    
    # 8 granular star ranges
    star_ranges = [
        "0",
        "1..5",
        "6..20",
        "21..50",
        "51..150",
        "151..500",
        "501..2000",
        ">2000"
    ]
    
    # 4 pushed-date categories
    date_windows = [
        "pushed:>2026-08-01",              # Active / Fresh (< 90 days)
        "pushed:2025-11-01..2026-07-31",   # Stable / Moderate (3 months to 1 year)
        "pushed:2023-11-01..2025-10-31",   # Older / Dormant (1 to 3 years)
        "pushed:<2023-11-01"               # Abandoned (> 3 years)
    ]
    
    # 2 pages per combination = 200 repos per combo
    # 3 langs * 8 stars * 4 date windows * 200 = ~19,200 repos total
    pages_per_combo = 2
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print("=" * 60)
    print("🚀 Starting Data Collection...")
    print("=" * 60)

    combo_count = 0
    total_combos = len(languages) * len(star_ranges) * len(date_windows)

    for lang in languages:
        for stars in star_ranges:
            for date_window in date_windows:
                combo_count += 1
                query = f"language:{lang} stars:{stars} {date_window}"
                print(f"\n🔎 [{combo_count}/{total_combos}] Querying: '{query}'")
                
                combo_records = []
                for page in range(1, pages_per_combo + 1):
                    repos = fetch_page(query=query, page=page, per_page=100)
                    
                    if not repos:
                        print(f"   Page {page}: No more results.")
                        break
                    
                    parsed_repos = [extract_features(r) for r in repos]
                    combo_records.extend(parsed_repos)
                    print(f"   Page {page}: Fetched {len(parsed_repos)} repos.")
                    
                    time.sleep(2)

                # Checkpoint to CSV
                if combo_records:
                    total_in_file = save_checkpoint(combo_records, OUTPUT_FILE)
                    print(f"💾 Checkpoint saved! Total unique repos on disk: {total_in_file}")

    print(f"\n🎉 Data collection complete! Dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    collect_data()
