"""
job_search.py
Searches free job APIs for entry-level cloud / cybersecurity / AI roles
and appends new matches to a Google Sheet.

Env vars required (set as GitHub Actions secrets):
  ADZUNA_APP_ID, ADZUNA_APP_KEY
  
"""

import os
import json
import requests

DATA_FILE = "jobs_data.json"

KEYWORDS = [
    "Cloud Support Engineer", "SOC Analyst", "AI/ML Engineer", "Security Analyst", "Cloud Engineer I","Cybersecurity Analyst", "Cloud Security Engineer", "Cloud Security Analyst", "Cloud Security Specialist", "Cloud Security Consultant", "Cloud Security Architect", "Software Developer", "Software Engineer", "Cloud Architect", "Forward Deployed Engineer", "Solutions Engineer", "AI Engineer", "Machine Learning Engineer", "Data Scientist", "Data Engineer", "DevOps Engineer", "Site Reliability Engineer", "Cloud Operations Engineer", "Cloud Infrastructure Engineer", "Cloud Solutions Architect", "Cloud Security Consultant", "Cloud Security Specialist", "Cloud Security Architect", "Incident responder",
    "Jr. Cloud security analyst",
    "Jr. Cybersecurity specialist",
    "Information security analyst",
    "Cybersecurity analyst",
    "SOC security engineer",
    "Cloud security specialist",
    "DevSecOps engineer"
]

ENTRY_LEVEL_SIGNALS = ["entry level", "entry-level", "junior", "associate", "0-2 years", "new grad", "graduate"]
EXCLUDE_SIGNALS = ["senior", "sr.", "staff", "principal", "lead", "manager", "director", "5+ years", "7+ years",, "3+ years", "4+ years", "6+ years", "8+ years", "9+ years", "10+ years"]


def is_entry_level(title: str, description: str) -> bool:
    text = f"{title} {description}".lower()
    if any(bad in text for bad in EXCLUDE_SIGNALS):
        return False
    return any(good in text for good in ENTRY_LEVEL_SIGNALS) or "years" not in text


def search_adzuna():
    app_id = os.environ["ADZUNA_APP_ID"]
    app_key = os.environ["ADZUNA_APP_KEY"]
    results = []
    for kw in KEYWORDS:
        url = f"https://api.adzuna.com/v1/api/jobs/us/search/1"
        params = {
            "app_id": app_id, "app_key": app_key,
            "results_per_page": 20, "what": kw,
            "max_days_old": 2, "content-type": "application/json",
        }
        r = requests.get(url, params=params, timeout=20)
        if r.ok:
            for job in r.json().get("results", []):
                results.append({
                    "source": "Adzuna",
                    "title": job.get("title", ""),
                    "company": job.get("company", {}).get("display_name", ""),
                    "url": job.get("redirect_url", ""),
                    "description": job.get("description", ""),
                })
    return results


def search_arbeitnow():
    r = requests.get("https://www.arbeitnow.com/api/job-board-api", timeout=20)
    results = []
    if r.ok:
        for job in r.json().get("data", []):
            title = job.get("title", "")
            desc = job.get("description", "")
            if any(kw.lower() in (title + desc).lower() for kw in KEYWORDS):
                results.append({
                    "source": "Arbeitnow",
                    "title": title,
                    "company": job.get("company_name", ""),
                    "url": job.get("url", ""),
                    "description": desc,
                })
    return results


def search_remoteok():
    r = requests.get("https://remoteok.com/api", timeout=20, headers={"User-Agent": "job-agent"})
    results = []
    if r.ok:
        for job in r.json():
            if not isinstance(job, dict) or "position" not in job:
                continue
            title = job.get("position", "")
            desc = job.get("description", "") or ""
            if any(kw.lower() in (title + desc).lower() for kw in KEYWORDS):
                results.append({
                    "source": "RemoteOK",
                    "title": title,
                    "company": job.get("company", ""),
                    "url": job.get("url", ""),
                    "description": desc,
                })
    return results

def load_jobs():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def save_jobs(jobs):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)



def main():
    jobs = load_jobs()
    existing_urls = {j["url"] for j in jobs}

    all_jobs = []
    for fn in (search_adzuna, search_arbeitnow, search_remoteok):
        try:
            all_jobs.extend(fn())
        except Exception as e:
            print(f"{fn.__name__} failed: {e}")

    new_count = 0
    for job in all_jobs:
        if job["url"] in existing_urls:
            continue
        if not is_entry_level(job["title"], job["description"]):
            continue
        jobs.append({
            "source": job["source"],
            "title": job["title"],
            "company": job["company"],
            "url": job["url"],
            "description": job["description"][:2000],
            "status": "NEW",
            "tailored_output": "",
        })
        existing_urls.add(job["url"])
        new_count += 1
 
    save_jobs(jobs)
    print(f"Added {new_count} new job(s). Total tracked: {len(jobs)}.")


if __name__ == "__main__":
    main()
