from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import streamlit as st

# Ensure project root on path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from optimizer_loader import load_subjects_from_directory, load_subjects_from_json  # type: ignore  # noqa: E402
from main import plan_global  # type: ignore  # noqa: E402
from main import DEFAULT_WEEKLY_HOURS  # type: ignore  # noqa: E402

SUBJECTS_DIR = ROOT_DIR / "Subjects_Input"
OUTPUT_DIR = ROOT_DIR / "Plans_Output"

st.set_page_config(page_title="Study Planner (Global)", layout="wide")


# ----------------------- Session State ----------------------- #
def init_session_state() -> None:
    if "subjects" not in st.session_state:
        st.session_state["subjects"] = []
    if "weekly_hours_df" not in st.session_state:
        st.session_state["weekly_hours_df"] = pd.DataFrame(
            {"Day": list(DEFAULT_WEEKLY_HOURS.keys()), "Hours": list(DEFAULT_WEEKLY_HOURS.values())}
        )
    if "adhoc_df" not in st.session_state:
        st.session_state["adhoc_df"] = pd.DataFrame(columns=["Date", "Hours"])
    if "last_config" not in st.session_state:
        st.session_state["last_config"] = {}


init_session_state()


# ----------------------- Helpers ----------------------- #
def save_uploaded_files(uploaded_files: List[Any]) -> List[Path]:
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for uf in uploaded_files:
        path = SUBJECTS_DIR / uf.name
        with path.open("wb") as f:
            f.write(uf.getvalue())
        saved_paths.append(path)
    return saved_paths


def summarize_subjects(subjects) -> pd.DataFrame:
    rows = []
    for subj in subjects:
        next_exam = None
        if subj.exams:
            future_dates = []
            for exam in subj.exams:
                try:
                    future_dates.append(datetime.strptime(exam.exam_date, "%Y-%m-%d").date())
                except Exception:
                    continue
            next_exam = min(future_dates) if future_dates else None
        total_hours = sum(getattr(m, "estimated_time_hrs", 0.0) for m in subj.modules)
        rows.append(
            {
                "subject_name": subj.subject_name,
                "#exams": len(subj.exams),
                "#modules": len(subj.modules),
                "next_exam_date": next_exam.isoformat() if next_exam else "-",
                "total_estimated_hours": total_hours,
            }
        )
    return pd.DataFrame(rows)


def weekly_hours_from_df(df: pd.DataFrame) -> Dict[str, float]:
    hours = {}
    for _, row in df.iterrows():
        day = str(row.get("Day", "")).strip()
        try:
            hours_val = float(row.get("Hours", 0))
        except Exception:
            hours_val = 0.0
        hours[day] = max(hours_val, 0.0)
    return hours


def adhoc_hours_from_df(df: pd.DataFrame) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    for _, row in df.iterrows():
        date_str = str(row.get("Date", "")).strip()
        hours_val = row.get("Hours", None)
        if not date_str:
            continue
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            hours_val = float(hours_val) if hours_val is not None else 0.0
            overrides[date_str] = max(hours_val, 0.0)
        except Exception:
            continue
    return overrides


def validate_inputs(
    subjects,
    start_date_val: Optional[date],
    weekly_hours: Dict[str, float],
    adhoc_hours: Dict[str, float],
    daily_max_hours: float,
    daily_max_fatigue: int,
) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if not subjects:
        errors.append("No subjects loaded.")
    if not start_date_val:
        errors.append("Start date is required.")
    if daily_max_hours <= 0:
        errors.append("Daily max hours must be > 0.")
    if daily_max_fatigue <= 0:
        errors.append("Daily max fatigue must be > 0.")
    if any(h < 0 for h in weekly_hours.values()):
        errors.append("Weekly hours must be non-negative.")
    for date_str in adhoc_hours.keys():
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except Exception:
            errors.append(f"Invalid adhoc date: {date_str}")
    # Warning if start date after some exams
    if start_date_val:
        for subj in subjects:
            for exam in subj.exams:
                try:
                    exam_dt = datetime.strptime(exam.exam_date, "%Y-%m-%d").date()
                    if start_date_val > exam_dt:
                        warnings.append(
                            f"{subj.subject_name}: start_date after exam '{exam.exam_name}' ({exam.exam_date}); modules tied to it will be deprioritized."
                        )
                except Exception:
                    continue
    return {"errors": errors, "warnings": warnings}


def run_optimizer(
    subjects,
    start_date_val: date,
    weekly_hours: Dict[str, float],
    adhoc_hours: Dict[str, float],
    daily_max_hours: float,
    daily_max_fatigue: int,
    short_term_horizon_days: int,
    use_sa_refine: bool,
) -> Dict[str, Dict]:
    config = {
        "start_date": start_date_val.isoformat(),
        "weekly_hours": weekly_hours,
        "adhoc_hours": adhoc_hours,
        "daily_max_fatigue": daily_max_fatigue,
        "daily_max_hours": daily_max_hours,
        "short_term_horizon_days": short_term_horizon_days,
        "use_sa_refine": use_sa_refine,
        "sa_iterations": 1000,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = plan_global(subjects, config, start_date_val, OUTPUT_DIR)
    st.session_state["last_config"] = config
    return metrics


def load_json_file(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def download_button_for_file(label: str, path: Path, key: str) -> None:
    if not path.exists():
        st.warning(f"{path.name} not found.")
        return
    data = path.read_bytes()
    st.download_button(label=label, data=data, file_name=path.name, mime="application/json", key=key)


# ----------------------- Sidebar ----------------------- #
with st.sidebar:
    st.header("Configuration")
    start_date_val = st.date_input("Start date", value=date.today())
    generate_short_term = st.checkbox("Generate short-term plan", value=True)
    short_term_horizon_days = st.slider("Short-term horizon (days)", min_value=1, max_value=3, value=1)
    st.markdown("---")
    st.subheader("Constraints")
    daily_max_hours = st.number_input("Daily max hours", min_value=0.0, value=8.0, step=0.5)
    daily_max_fatigue = st.number_input("Daily max fatigue", min_value=1, value=7, step=1)
    st.markdown("---")
    st.subheader("Availability")
    st.caption("Edit weekly hours")
    st.session_state["weekly_hours_df"] = st.data_editor(
        st.session_state["weekly_hours_df"],
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        key="weekly_hours_editor",
    )
    st.caption("Adhoc overrides (date, hours)")
    st.session_state["adhoc_df"] = st.data_editor(
        st.session_state["adhoc_df"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="adhoc_editor",
    )
    st.markdown("---")
    st.subheader("Optimizer")
    use_sa_refine = st.checkbox("Use simulated annealing refine", value=True)
    st.caption("Urgency spacing is enabled by default in the optimizer (no toggle).")
    st.markdown("---")
    st.subheader("Diagnostics")
    show_diag = st.checkbox("Show per-subject diagnostics", value=False)
    if st.button("Reset session"):
        st.session_state.clear()
        st.experimental_rerun()


# ----------------------- Main Layout ----------------------- #
tabs = st.tabs(
    [
        "Inputs",
        "Validate & Generate",
        "Results — Global Long-Term",
        "Results — Global Short-Term",
        "Diagnostics",
    ]
)


# ----------------------- Tab 1: Inputs ----------------------- #
with tabs[0]:
    st.header("Subject Inputs")
    uploaded = st.file_uploader("Upload subject JSON files", type=["json"], accept_multiple_files=True)
    col_up1, col_up2 = st.columns([1, 1])
    if col_up1.button("Save uploads to Subjects_Input", use_container_width=True):
        if not uploaded:
            st.warning("No files uploaded.")
        else:
            saved = save_uploaded_files(uploaded)
            st.success(f"Saved {len(saved)} file(s) to Subjects_Input/.")
            st.session_state["subjects"] = load_subjects_from_directory(SUBJECTS_DIR)
    if col_up2.button("Load from Subjects_Input folder", use_container_width=True):
        st.session_state["subjects"] = load_subjects_from_directory(SUBJECTS_DIR)
        st.success(f"Loaded {len(st.session_state['subjects'])} subject(s) from folder.")

    if st.session_state["subjects"]:
        st.subheader("Parsed Subjects")
        st.dataframe(summarize_subjects(st.session_state["subjects"]), use_container_width=True)
    st.info("fatigue_drain is interpreted as PER-MODULE TOTAL (per-block fatigue = fatigue_drain / n_blocks).")


# ----------------------- Tab 2: Validate & Generate ----------------------- #
with tabs[1]:
    st.header("Validate & Generate")
    weekly_hours = weekly_hours_from_df(st.session_state["weekly_hours_df"])
    adhoc_hours = adhoc_hours_from_df(st.session_state["adhoc_df"])

    if st.button("Validate Inputs", key="validate_btn"):
        checks = validate_inputs(
            st.session_state["subjects"],
            start_date_val,
            weekly_hours,
            adhoc_hours,
            daily_max_hours,
            daily_max_fatigue,
        )
        if checks["errors"]:
            st.error("Validation errors:")
            for err in checks["errors"]:
                st.write(f"- {err}")
        else:
            st.success("Validation passed.")
        if checks["warnings"]:
            st.warning("Warnings:")
            for warn in checks["warnings"]:
                st.write(f"- {warn}")

    if st.button("Generate Global Plan", type="primary", key="generate_btn"):
        checks = validate_inputs(
            st.session_state["subjects"],
            start_date_val,
            weekly_hours,
            adhoc_hours,
            daily_max_hours,
            daily_max_fatigue,
        )
        if checks["errors"]:
            st.error("Fix validation errors first.")
        else:
            with st.spinner("Running optimizer..."):
                run_optimizer(
                    subjects=st.session_state["subjects"],
                    start_date_val=start_date_val,
                    weekly_hours=weekly_hours,
                    adhoc_hours=adhoc_hours,
                    daily_max_hours=daily_max_hours,
                    daily_max_fatigue=int(daily_max_fatigue),
                    short_term_horizon_days=short_term_horizon_days if generate_short_term else 1,
                    use_sa_refine=use_sa_refine,
                )
            st.success("Plans generated. See Results tabs.")


# ----------------------- Tab 3: Results — Global Long-Term ----------------------- #
with tabs[2]:
    st.header("Global Long-Term Plan")
    long_path = OUTPUT_DIR / "global_long_term_plan.json"
    summary_path = OUTPUT_DIR / "summary_metrics.json"
    if not long_path.exists():
        st.info("No global_long_term_plan.json found. Generate a plan first.")
    else:
        plan = load_json_file(long_path) or {}
        metrics = plan.get("metrics", {})
        st.subheader("Metrics")
        cols = st.columns(5)
        cols[0].metric("Expected score gain", f"{metrics.get('expected_score_gain', 0):.3f}")
        cols[1].metric("Exam coverage %", f"{metrics.get('exam_score_coverage_pct', 0)*100:.1f}%")
        cols[2].metric("Hours scheduled", f"{metrics.get('total_hours_scheduled', 0):.1f}")
        cols[3].metric("Fatigue compliance", f"{metrics.get('fatigue_compliance_rate', 0)*100:.1f}%")
        cols[4].metric("Dependency violations", metrics.get("dependency_violations", 0))

        schedule = plan.get("schedule", [])
        if schedule:
            sched_df = pd.DataFrame(schedule)
            sched_df["date"] = pd.to_datetime(sched_df["date"])
            st.subheader("Charts")
            chart_cols = st.columns(2)
            with chart_cols[0]:
                st.caption("Daily hours")
                st.bar_chart(sched_df.set_index("date")["hours_planned"])
            with chart_cols[1]:
                st.caption("Daily fatigue vs cap")
                fatigue_df = sched_df[["date", "fatigue_total"]].set_index("date")
                st.line_chart(fatigue_df)
                st.caption(f"Fatigue cap: {plan.get('daily_max_fatigue', '-')}")

            # Stacked bars by subject
            blocks_expanded = []
            for row in schedule:
                for blk in row.get("blocks", []):
                    blocks_expanded.append(
                        {
                            "date": row["date"],
                            "hours": blk.get("block_hours", 1.0),
                            "subject": blk.get("subject_name", "Unknown"),
                        }
                    )
            if blocks_expanded:
                st.caption("Hours by subject per day")
                bdf = pd.DataFrame(blocks_expanded)
                pivot = bdf.pivot_table(index="date", columns="subject", values="hours", aggfunc="sum").fillna(0)
                st.bar_chart(pivot)

            st.subheader("Schedule")
            subjects_available = sorted({blk.get("subject_name", "Unknown") for row in schedule for blk in row.get("blocks", [])})
            exams_available = sorted({blk.get("exam_type", "Unknown") for row in schedule for blk in row.get("blocks", [])})
            filt_col1, filt_col2 = st.columns(2)
            subject_filter = filt_col1.multiselect("Filter by subject", subjects_available, default=subjects_available)
            exam_filter = filt_col2.multiselect("Filter by exam type", exams_available, default=exams_available)

            for row in schedule:
                day_blocks = [
                    blk for blk in row.get("blocks", [])
                    if blk.get("subject_name", "Unknown") in subject_filter and blk.get("exam_type", "Unknown") in exam_filter
                ]
                if not day_blocks:
                    continue
                with st.expander(f"{row['date']} — hours {row.get('hours_planned',0)} / fatigue {row.get('fatigue_total',0)}"):
                    st.table(pd.DataFrame(day_blocks))

        st.subheader("Downloads")
        download_button_for_file("Download global_long_term_plan.json", long_path, key="dl_long")
        download_button_for_file("Download summary_metrics.json", summary_path, key="dl_summary")


# ----------------------- Tab 4: Results — Global Short-Term ----------------------- #
with tabs[3]:
    st.header("Global Short-Term Plan")
    short_path = OUTPUT_DIR / "global_short_term_plan.json"
    if not generate_short_term:
        st.info("Short-term generation disabled in sidebar. Enable and regenerate to view.")
    elif not short_path.exists():
        st.info("No global_short_term_plan.json found. Generate a plan first.")
    else:
        plan = load_json_file(short_path) or {}
        metrics = plan.get("metrics", {})
        st.subheader("Metrics")
        cols = st.columns(4)
        cols[0].metric("Expected score gain", f"{metrics.get('expected_score_gain', 0):.3f}")
        cols[1].metric("Exam coverage %", f"{metrics.get('exam_score_coverage_pct', 0)*100:.1f}%")
        cols[2].metric("Hours scheduled", f"{metrics.get('total_hours_scheduled', 0):.1f}")
        cols[3].metric("Fatigue compliance", f"{metrics.get('fatigue_compliance_rate', 0)*100:.1f}%")

        schedule = plan.get("schedule", [])
        if schedule:
            sched_df = pd.DataFrame(schedule)
            st.subheader("Schedule")
            st.table(sched_df[["date", "hours_planned", "fatigue_total"]])
        st.subheader("Downloads")
        download_button_for_file("Download global_short_term_plan.json", short_path, key="dl_short")


# ----------------------- Tab 5: Diagnostics ----------------------- #
with tabs[4]:
    st.header("Diagnostics")
    if not show_diag:
        st.info("Enable 'Show per-subject diagnostics' in the sidebar to view.")
    else:
        st.caption("Per-subject plans (diagnostic only):")
        for path in sorted(OUTPUT_DIR.glob("*_plan.json")):
            if "global_" in path.name:
                continue
            st.write(f"- {path.name}")
        st.caption("Raw subjects in Subjects_Input/:")
        for path in sorted(SUBJECTS_DIR.glob("*.json")):
            st.write(f"- {path.name}")
