"""
notion_sync.py
Syncs jobs_data.json into a Notion database so you get a live, cloud-based
kanban/table view of your job tracker.

Notion database must have these exact properties:
  Title            (Title)
  Company          (Text)
  Status           (Select: NEW, READY_TO_REVIEW, APPLIED)
  Source           (Select)
  URL              (URL)
  Description      (Text)
  Tailored Output  (Text)

Env vars required:
  NOTION_TOKEN
  NOTION_DATABASE_ID
"""

import os
import json
import requests

DATA_FILE = "jobs_data.json"
NOTION_VERSION = "2022-06-28"
BASE_URL = "https://api.notion.com/v1"


def headers():
    return {
        "Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def load_jobs():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jobs(jobs):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)


def chunk_text(text, size=1900):
    """Notion rich_text blocks cap at 2000 chars each; split long text into chunks."""
    text = text or ""
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def rich_text_property(text):
    return {"rich_text": [{"text": {"content": chunk}} for chunk in chunk_text(text)]}


def build_properties(job):
    return {
        "Title": {"title": [{"text": {"content": job.get("title", "")[:2000]}}]},
        "Company": rich_text_property(job.get("company", "")),
        "Status": {"select": {"name": job.get("status", "NEW")}},
        "Source": {"select": {"name": job.get("source", "Unknown")}},
        "URL": {"url": job.get("url") or None},
        "Description": rich_text_property(job.get("description", "")),
        "Tailored Output": rich_text_property(job.get("tailored_output", "")),
    }


def find_existing_page(url):
    if not url:
        return None
    payload = {"filter": {"property": "URL", "url": {"equals": url}}}
    r = requests.post(
        f"{BASE_URL}/databases/{os.environ['NOTION_DATABASE_ID']}/query",
        headers=headers(), json=payload, timeout=30,
    )
    r.raise_for_status()
    results = r.json().get("results", [])
    return results[0]["id"] if results else None


def create_page(job):
    payload = {
        "parent": {"database_id": os.environ["NOTION_DATABASE_ID"]},
        "properties": build_properties(job),
    }
    r = requests.post(f"{BASE_URL}/pages", headers=headers(), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["id"]


def update_page(page_id, job):
    payload = {"properties": build_properties(job)}
    r = requests.patch(f"{BASE_URL}/pages/{page_id}", headers=headers(), json=payload, timeout=30)
    r.raise_for_status()


def main():
    jobs = load_jobs()
    created, updated = 0, 0

    for job in jobs:
        existing_page_id = job.get("notion_page_id") or find_existing_page(job.get("url"))
        try:
            if existing_page_id:
                update_page(existing_page_id, job)
                job["notion_page_id"] = existing_page_id
                updated += 1
            else:
                page_id = create_page(job)
                job["notion_page_id"] = page_id
                created += 1
        except requests.HTTPError as e:
            print(f"Notion sync failed for {job.get('title')}: {e.response.text}")

    save_jobs(jobs)
    print(f"Notion sync: {created} created, {updated} updated.")


if __name__ == "__main__":
    main()
