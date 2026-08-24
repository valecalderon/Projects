"""
tailor_resume.py
For every sheet row marked NEW, generates tailored resume bullets + a cover
letter draft using the Gemini API (free), writes results back to the
sheet, and pings you on Telegram.

Env vars required:
  GEMINI_API_KEY
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  (optional but recommended)
"""

import os
import json
import requests

DATA_FILE = "jobs_data.json"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-flash-latest:generateContent"
)

def load_jobs():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def save_jobs(jobs):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)


def load_context():
    with open("resume_master.txt", "r", encoding="utf-8") as f:
        resume = f.read()
    with open("profile.json", "r", encoding="utf-8") as f:
        profile = json.load(f)
    return resume, profile


def build_prompt(resume, profile, title, company, description):
    return f"""You are helping tailor a job application. Use ONLY facts present
in the resume and profile below — never invent employers, dates, titles, or
certifications that aren't listed.

PROFILE (ground truth, do not contradict):
{json.dumps(profile, indent=2)}

FULL RESUME (raw material — pick the most relevant parts):
{resume}

JOB POSTING:
Title: {title}
Company: {company}
Description: {description}

Return exactly two sections:
1. "TAILORED BULLETS" — 4 to 6 existing resume bullets, lightly reworded to
   mirror this job's language and priorities. Do not add new claims.
2. "COVER LETTER" — a 180-220 word cover letter for this specific role,
   grounded only in the resume/profile facts above.
"""


def call_gemini(prompt):
    api_key = os.environ["GEMINI_API_KEY"]
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = requests.post(f"{GEMINI_URL}?key={api_key}", json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def notify_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text[:4000]}, timeout=20)


def main():
     jobs = load_jobs()
     resume, profile = load_context()
 
     tailored_count = 0
     for job in jobs:
        if job.get("status") != "NEW":
            continue
 
        title, company, url, description = (
            job["title"], job["company"], job["url"], job["description"],
        )
        prompt = build_prompt(resume, profile, title, company, description)
        try:
            result = call_gemini(prompt)
        except Exception as e:
            print(f"Gemini call failed for {title} @ {company}: {e}")
            continue
 
        job["tailored_output"] = result
        job["status"] = "READY_TO_REVIEW"
 
        notify_telegram(f"New tailored application ready:\n{title} @ {company}\n{url}")
        
        print(f"Tailored: {title} @ {company}")
        tailored_count += 1
 
     save_jobs(jobs)
     print(f"Tailored {tailored_count} job(s).")


if __name__ == "__main__":
    main()