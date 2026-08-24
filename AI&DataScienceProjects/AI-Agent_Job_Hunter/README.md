# Job hunting AI-agent
## What this agent actually does
By: Valeria Calderon
### Overview

The job hunting AI agent uses LArge ALnguage models, Gen AI, and Agentic Ai systems and addresses the time consuming task of searching for hours the internet for new job postings that match certain criteria.

This project is for saving time, and aiming for the right job openings for each person. The programs uses APIs to search for new job openings depending on certain keywords, and then saves it to a table in a notion databse in your private account. All steps on set up are below.

Tools & Frameworks: 
+ Python
+ Gemini API
+ Github Actions
+ Version Control
+ workflows
+ LLM

Every day , it will:

Search several job APIs/boards for postings matching keywords.
Filter out anything that isn't actually your keywords or isn't a real match.
Tailor a version of your resume bullets + a custom cover letter for each posting, using an LLM.
Save everything to a Notion Database (or local folder) so you can review, then apply yourself.


---
## Platforms & accounts you'll need (all free)

| Purpose | Tool | Why |
|---|---|---|
| Run the agent on a schedule | **GitHub Actions** (free, in any GitHub repo) | Free scheduled ("cron") runs, no server needed |
| Job listings | **Adzuna API** (free), **Arbeitnow API** (free, no key needed), **RemoteOK API** (free, no key), **USAJobs API** (free, great for entry-level gov cyber/IT roles) | Together these cover cloud/cyber/AI roles broadly without scraping (scraping LinkedIn/Indeed violates ToS) |
| AI tailoring | **Google Gemini API free tier** or **Groq API free tier** (both currently offer free API keys with generous limits) | Used to rewrite resume bullets + draft cover letters per job |
| Store results | **Google Sheets** (free, via a Google Cloud service account) | Simple dashboard you check each morning |
| Notifications | **Telegram Bot API** | Pings you when new tailored applications are ready |
| Resume parsing | **Python + pdfplumber/python-docx** | Extracts your resume text so the AI has it as context |

---

## Step 1 — Get your accounts and keys

1. **GitHub** account (you probably have one) → create a new repo, e.g. `job-agent`.
2. **Adzuna**: sign up at their developer portal → get `APP_ID` + `APP_KEY` (free).
3. **USAJobs**: register for an API key (free, instant).
4. **Gemini API key**: from Google AI Studio → free tier key.
5. **Telegram**: message @BotFather, `/newbot`, get a bot token + your chat ID (2 min, easiest notification option).

## Step 2 — Prepare your resume as structured input

Save two files in the repo:
- `resume_master.txt` — your full resume as plain text (every bullet you've ever written, even ones not currently on your resume — more raw material = better tailoring).
- `profile.json` — a few structured facts the AI should never invent, e.g.:



This file is your safety rail — it tells the AI exactly what's true so it can't accidentally fabricate experience on your cover letter.

---

## Step 3 — The job search script

See `job_search.py`. It:
- Queries Adzuna, Arbeitnow, RemoteOK, and USAJobs with keyword sets 
- Filters titles/descriptions for  signals (`"entry level"`, `"junior"`, `"0-2 years"`, `"associate"`)
- De-duplicates against jobs already logged in your sheet (so you don't get repeat tailoring).
- Writes new matches to the sheet with status `NEW`.

## Step 4 — The tailoring script

See `tailor_resume.py`. For every row marked `NEW`, it:
- Sends the job description + your `resume_master.txt` + `profile.json` to Gemini.
- Asks for: (a) which 4–6 of your existing bullets to lead with, rewritten to mirror the job's language, (b) a 200-word cover letter draft.
- Writes the output back into the sheet, marks status `READY_TO_REVIEW`.
- Sends you a Telegram message with the job title + link.

## Running it for free, on schedule

`schedule.yml` (GitHub Actions workflow) runs `job_search.py` every morning and `tailor_resume.py` right after, entirely on GitHub's free compute — nothing runs on your machine, no server costs.

<sub> References: claude AI, Github guides</sub>
