# Study Plan Optimization Engine

Heuristic study planner that converts course syllabi into optimized long‑term schedules with fatigue limits, dependency awareness, and urgency spacing. Supports CLI and Streamlit UI, plus an LLM pipeline to extract syllabus data directly from PDFs.

## Features
- Greedy scheduler with urgency curve and fatigue/hour caps; optional simulated annealing refinement to improve block placement.
- Global and per‑subject long‑term planning; short‑term planner available for nearest-exam focus.
- Metrics: expected score gain, exam coverage, fatigue compliance, deadline/dependency violations, hours scheduled.
- Streamlit dashboard to upload/inspect inputs, tweak constraints, run optimizer, and visualize calendars/charts.
- PDF ingestion via Gemini + LangChain → validated JSON subject files.

## Architecture (high level)
1) **Input loading** (`optimizer_loader.py`) — parse JSON into `Subject`, `Exam`, `Module`.  
2) **Preprocess** (`preprocess.py`) — map exams to dates, compute module weights/value-per-hour, expand to 1h `StudyBlock`s, derive dependency requirements.  
3) **Availability** (`availability.py`) — merge weekly template + ad-hoc overrides per day.  
4) **Scheduling** (`planner_core.py`, `long_term_planner.py`, `short_term_planner.py`) — greedy placement by priority (value * urgency), enforcing hours/fatigue caps, deadlines, dependencies; optional simulated annealing to refine.  
5) **Metrics + outputs** (`metrics.py`, `main.py`) — compute coverage/compliance/violations and write plans/summary JSON.  
6) **UI + LLM** (`app/streamlit_app.py`, `pdf_extractor.py`) — Streamlit front-end and Gemini-powered PDF → JSON pipeline validated with `models_pydantic.py`.

## Installation
Requires Python 3.10+. From repo root:
```bash
pip install -r requirement.txt
```
or with the project metadata:
```bash
pip install -e .
```

## Inputs
- Place subject JSON files in `Subjects_Input/` (see samples in that directory). Key fields: `subject_name`, `exams` (name/date/score_percentage), `modules` (id/name/exam_type/estimated_exam_percent/estimated_time_hrs/preparation_ease/fatigue_drain/dependency_modules).
- Start date, weekly hours, ad-hoc overrides, fatigue/hour caps, and SA settings are provided via CLI flags or config JSON.

## CLI usage
Generate plans from existing subject JSONs:
```bash
python main.py \
  --start-date 2025-03-01 \
  --weekly-hours '{"Mon":3,"Tue":3,"Wed":2,"Thu":3,"Fri":2,"Sat":5,"Sun":5}' \
  --adhoc-hours '{"2025-03-10":0}' \
  --daily-max-hours 8 \
  --daily-max-fatigue 45 \
  --no-sa-refine
```
Flags of interest:
- `--config <path>`: JSON containing the same keys as CLI flags.
- `--use-sa-refine/--no-sa-refine`, `--sa-iterations <int>`: toggle/tune simulated annealing.
- `--input-dir`, `--output-dir`: override default `Subjects_Input/` and `Plans_Output/`.

Outputs: per-subject `<NAME>_long_term_plan.json`, `global_long_term_plan.json`, and `summary_metrics.json` in `Plans_Output/`.

## Streamlit dashboard
Launch:
```bash
streamlit run app/streamlit_app.py
```
- Upload PDF or JSON syllabus files, edit exams, set availability/fatigue caps, and run the global planner.
- Visualize daily hours/fatigue, subject hour breakdowns, and a calendar view; download plan/summary JSONs.

## PDF → JSON (LLM pipeline)
- Requires Google Gemini API key (`GOOGLE_API_KEY` env var or `api_key` argument).
- CLI:
```bash
python pdf_extractor.py syllabus.pdf Subjects_Input
```
- Streamlit: upload PDFs in the “PDF Syllabus” tab and select a Gemini model (e.g., `gemini-2.5-flash-lite`). Extracted subjects are validated with `SubjectPydantic` and saved to `Subjects_Input/`.

## Configuration defaults (main)
- Weekly hours: `{"Mon":2,"Tue":3,"Wed":2,"Thu":3,"Fri":2,"Sat":6,"Sun":6}`
- Daily caps: `daily_max_hours=8.0`, `daily_max_fatigue=50`
- Simulated annealing: enabled, `sa_iterations=1000`
- Start date: today (override with `--start-date` or config)

## Authors & Course
- Kampanat Yingseree — 6633021421  
- Peerapat Patcharamontree — 6633176021  
Project for course **2110477 Artificial Intelligence II (1/2025), Chulalongkorn University**.
