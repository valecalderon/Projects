# job hunting AI-agent with automatic notification to your phones
## What this agent actually does
Every day (on a free schedule), it will:

Search several job APIs/boards for new entry-level postings matching cloud, cybersecurity, and AI keywords.
Filter out anything that isn't actually entry-level or isn't a real match.
Tailor a version of your resume bullets + a custom cover letter for each posting, using an LLM.
Save everything to a Notion Database (or local folder) so you can review, then apply yourself.


---
## Platforms & accounts you'll need (all free tier)

| Purpose | Tool | Why |
|---|---|---|
| Run the agent on a schedule | **GitHub Actions** (free, in any GitHub repo) | Free scheduled ("cron") runs, no server needed |
| Job listings | **Adzuna API** (free), **Arbeitnow API** (free, no key needed), **RemoteOK API** (free, no key), **USAJobs API** (free, great for entry-level gov cyber/IT roles) | Together these cover cloud/cyber/AI roles broadly without scraping (scraping LinkedIn/Indeed violates ToS) |
| AI tailoring | **Google Gemini API free tier** or **Groq API free tier** (both currently offer free API keys with generous limits) | Used to rewrite resume bullets + draft cover letters per job |
| Store results | **Google Sheets** (free, via a Google Cloud service account) | Simple dashboard you check each morning |
| Notifications | **Telegram Bot API** (free) or plain email via Gmail SMTP | Pings you when new tailored applications are ready |
| Resume parsing | **Python + pdfplumber/python-docx** | Extracts your resume text so the AI has it as context |

---

## Step 1 — Get your accounts and keys (15–20 min)

1. **GitHub** account (you probably have one) → create a new repo, e.g. `job-agent`.
2. **Adzuna**: sign up at their developer portal → get `APP_ID` + `APP_KEY` (free).
3. **USAJobs**: register for an API key (free, instant).
4. **Gemini API key**: from Google AI Studio → free tier key.
5. **Telegram**: message @BotFather, `/newbot`, get a bot token + your chat ID (2 min, easiest notification option).

6. ## Step 2 — Prepare your resume as structured input

Save two files in the repo:
- `resume_master.txt` — your full resume as plain text (every bullet you've ever written, even ones not currently on your resume — more raw material = better tailoring).
- `profile.json` — a few structured facts the AI should never invent, e.g.:

```json
{
  "name": "Your Name",
  "target_roles": ["roles"],
  "location_pref": "Remote or XYC",
  "years_experience": 0,
  "certifications": ["Certs_name"],
  "must_not_claim": ["do not invent employers, dates, or certifications not listed here"]
}
```

This file is your safety rail — it tells the AI exactly what's true so it can't accidentally fabricate experience on your cover letter.

---

## Step 3 — The job search script

See `job_search.py`. It:
- Queries Adzuna, Arbeitnow, RemoteOK, and USAJobs with keyword sets 
- Filters titles/descriptions for entry-level signals (`"entry level"`, `"junior"`, `"0-2 years"`, `"associate"`) and excludes senior/lead/manager postings.
- De-duplicates against jobs already logged in your sheet (so you don't get repeat tailoring).
- Writes new matches to the sheet with status `NEW`.

## Step 4 — The tailoring script

See `tailor_resume.py`. For every row marked `NEW`, it:
- Sends the job description + your `resume_master.txt` + `profile.json` to Gemini.
- Asks for: (a) which 4–6 of your existing bullets to lead with, rewritten to mirror the job's language, (b) a 200-word cover letter draft.
- Writes the output back into the sheet, marks status `READY_TO_REVIEW`.
- Sends you a Telegram message with the job title + link.

## Step 5 — Your daily 10-minute routine

1. Open the Telegram notification / Notion.
2. Skim the tailored bullets and cover letter — fix anything off, this takes seconds since it's already 90% done.
3. Copy into your resume template, export as PDF (or use the `docx` skill in Claude if you want, separately).
4. Click "Apply" on the job page yourself.
5. Mark row as `APPLIED` in the sheet so it's tracked.

## Running it for free, on schedule

`schedule.yml` (GitHub Actions workflow) runs `job_search.py` every morning and `tailor_resume.py` right after, entirely on GitHub's free compute — nothing runs on your machine, no server costs.
