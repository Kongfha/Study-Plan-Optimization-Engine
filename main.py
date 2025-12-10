from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

from availability import build_daily_availability
from long_term_planner import greedy_long_term, sa_refine_long_term
from metrics import compute_plan_metrics
from models import PlanMetrics, Subject
from optimizer_loader import load_subjects_from_directory
from planner_core import compute_urgency_for_day
from preprocess import build_exam_map, compute_final_weights, expand_to_blocks
from short_term_planner import greedy_short_term

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WEEKLY_HOURS = {"Mon": 2, "Tue": 3, "Wed": 2, "Thu": 3, "Fri": 2, "Sat": 6, "Sun": 6}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traditional AI study planner")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--start-date", dest="start_date", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--weekly-hours", dest="weekly_hours", type=str, help="JSON string for weekly hours")
    parser.add_argument("--adhoc-hours", dest="adhoc_hours", type=str, help="JSON string for adhoc hours")
    parser.add_argument("--daily-max-fatigue", dest="daily_max_fatigue", type=int, help="Daily fatigue cap")
    parser.add_argument("--daily-max-hours", dest="daily_max_hours", type=float, help="Daily hours cap")
    parser.add_argument("--short-term-horizon-days", dest="short_term_horizon_days", type=int, help="Horizon for short-term plan")
    parser.add_argument("--use-sa-refine", dest="use_sa_refine", action="store_true", help="Enable simulated annealing refinement")
    parser.add_argument("--no-sa-refine", dest="use_sa_refine", action="store_false", help="Disable simulated annealing refinement")
    parser.set_defaults(use_sa_refine=None)
    parser.add_argument("--sa-iterations", dest="sa_iterations", type=int, help="Simulated annealing iterations")
    parser.add_argument("--input-dir", dest="input_dir", type=str, default="Subjects_Input", help="Directory with subject JSON files")
    parser.add_argument("--output-dir", dest="output_dir", type=str, default="Plans_Output", help="Directory to write plans")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict:
    config = {
        "start_date": date.today().isoformat(),
        "weekly_hours": DEFAULT_WEEKLY_HOURS,
        "adhoc_hours": {},
        "daily_max_fatigue": 7,
        "daily_max_hours": 8.0,
        "short_term_horizon_days": 1,
        "use_sa_refine": True,
        "sa_iterations": 1000,
    }

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            file_config = json.load(f)
            config.update(file_config)

    if args.start_date:
        config["start_date"] = args.start_date
    if args.weekly_hours:
        config["weekly_hours"] = json.loads(args.weekly_hours)
    if args.adhoc_hours:
        config["adhoc_hours"] = json.loads(args.adhoc_hours)
    if args.daily_max_fatigue is not None:
        config["daily_max_fatigue"] = args.daily_max_fatigue
    if args.daily_max_hours is not None:
        config["daily_max_hours"] = args.daily_max_hours
    if args.short_term_horizon_days is not None:
        config["short_term_horizon_days"] = args.short_term_horizon_days
    if args.use_sa_refine is not None:
        config["use_sa_refine"] = args.use_sa_refine
    if args.sa_iterations is not None:
        config["sa_iterations"] = args.sa_iterations

    return config


def parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plan_subject(
    subject: Subject,
    config: Dict,
    start: date,
    output_dir: Path,
) -> Dict[str, PlanMetrics]:
    base_notes: List[str] = []
    exam_map = build_exam_map(subject)
    base_notes.extend(compute_final_weights(subject, start, exam_map))

    high_fatigue_modules = [m.module_id for m in subject.modules if m.fatigue_drain > config["daily_max_fatigue"]]
    if high_fatigue_modules:
        base_notes.append(
            f"Modules with fatigue drain above the daily cap ({config['daily_max_fatigue']}): {', '.join(high_fatigue_modules)}."
        )

    blocks, module_block_totals, dependency_requirements = expand_to_blocks(subject, start, exam_map)
    module_final_weights = {m.module_key or m.module_id: m.final_weight for m in subject.modules}
    module_exam_percents = {m.module_key or m.module_id: m.estimated_exam_percent for m in subject.modules}
    total_exam_percent = sum(module_exam_percents.values())

    future_exams = [d for d in exam_map.values() if d and d >= start]
    if future_exams:
        horizon_end = max(future_exams)
    else:
        horizon_end = start
        base_notes.append("No future exams on or after the start date; schedule left empty.")

    availability_long = build_daily_availability(
        start_date=start,
        end_date=horizon_end,
        weekly_hours=config["weekly_hours"],
        adhoc_hours=config.get("adhoc_hours", {}),
        daily_max_hours=config.get("daily_max_hours", 8.0),
    )

    # Long-term planning
    notes_long = list(base_notes)
    if future_exams:
        (
            schedule_map,
            day_hours,
            day_fatigue,
            remaining_blocks,
            deadline_violations,
            dependency_violations,
            total_days,
        ) = greedy_long_term(
            blocks=blocks,
            start_date=start,
            horizon_end=horizon_end,
            availability_map=availability_long,
            exam_map=exam_map,
            daily_max_fatigue=config["daily_max_fatigue"],
            dependency_requirements=dependency_requirements,
            module_block_totals=module_block_totals,
        )

        if config.get("use_sa_refine", True) and schedule_map:
            schedule_map, day_hours, day_fatigue = sa_refine_long_term(
                schedule_map=schedule_map,
                start_date=start,
                horizon_end=horizon_end,
                availability_map=availability_long,
                exam_map=exam_map,
                daily_max_fatigue=config["daily_max_fatigue"],
                dependency_requirements=dependency_requirements,
                module_block_totals=module_block_totals,
                iterations=config.get("sa_iterations", 1000),
            )
    else:
        schedule_map, day_hours, day_fatigue = {}, {}, {}
        remaining_blocks = blocks
        deadline_violations = 0
        dependency_violations = 0
        total_days = 0

    if remaining_blocks:
        notes_long.append(f"{len(remaining_blocks)} blocks remained unscheduled in the long-term plan.")

    metrics_long = compute_plan_metrics(
        schedule_map=schedule_map,
        day_fatigue=day_fatigue,
        module_block_totals=module_block_totals,
        module_final_weights=module_final_weights,
        module_exam_percents=module_exam_percents,
        total_exam_percent=total_exam_percent,
        exam_map=exam_map,
        dependency_requirements=dependency_requirements,
        daily_max_fatigue=config["daily_max_fatigue"],
        total_days=total_days,
        deadline_violations=deadline_violations,
        dependency_violations=dependency_violations,
    )

    long_term_output = build_plan_output(
        subject_name=subject.subject_name,
        mode="long_term",
        start_date=start,
        horizon_end=horizon_end,
        availability=config,
        schedule_map=schedule_map,
        day_hours=day_hours,
        day_fatigue=day_fatigue,
        exam_map=exam_map,
        metrics=metrics_long,
        notes=notes_long,
    )

    write_plan(output_dir / f"{subject.subject_name}_long_term_plan.json", long_term_output)

    # Short-term planning
    short_horizon_end = compute_short_horizon_end(start, config["short_term_horizon_days"], exam_map)
    availability_short = build_daily_availability(
        start_date=start,
        end_date=short_horizon_end,
        weekly_hours=config["weekly_hours"],
        adhoc_hours=config.get("adhoc_hours", {}),
        daily_max_hours=config.get("daily_max_hours", 8.0),
    )

    (
        schedule_short,
        day_hours_short,
        day_fatigue_short,
        remaining_short,
        deadline_short,
        dependency_short,
        total_days_short,
        short_horizon_end,
    ) = greedy_short_term(
        blocks=blocks,
        start_date=start,
        horizon_days=config["short_term_horizon_days"],
        availability_map=availability_short,
        exam_map=exam_map,
        daily_max_fatigue=config["daily_max_fatigue"],
        dependency_requirements=dependency_requirements,
        module_block_totals=module_block_totals,
    )

    notes_short = list(base_notes)
    if remaining_short:
        notes_short.append(f"{len(remaining_short)} blocks remained unscheduled in the short-term plan.")

    metrics_short = compute_plan_metrics(
        schedule_map=schedule_short,
        day_fatigue=day_fatigue_short,
        module_block_totals=module_block_totals,
        module_final_weights=module_final_weights,
        module_exam_percents=module_exam_percents,
        total_exam_percent=total_exam_percent,
        exam_map=exam_map,
        dependency_requirements=dependency_requirements,
        daily_max_fatigue=config["daily_max_fatigue"],
        total_days=total_days_short,
        deadline_violations=deadline_short,
        dependency_violations=dependency_short,
    )

    short_term_output = build_plan_output(
        subject_name=subject.subject_name,
        mode="short_term",
        start_date=start,
        horizon_end=short_horizon_end,
        availability=config,
        schedule_map=schedule_short,
        day_hours=day_hours_short,
        day_fatigue=day_fatigue_short,
        exam_map=exam_map,
        metrics=metrics_short,
        notes=notes_short,
    )

    write_plan(output_dir / f"{subject.subject_name}_short_term_plan.json", short_term_output)

    return {
        "long_term": metrics_long,
        "short_term": metrics_short,
    }


def plan_global(
    subjects: List[Subject],
    config: Dict,
    start: date,
    output_dir: Path,
) -> Dict[str, PlanMetrics]:
    notes: List[str] = []
    all_blocks = []
    module_block_totals: Dict[str, int] = {}
    dependency_requirements: Dict[str, Dict[str, int]] = {}
    module_final_weights: Dict[str, float] = {}
    module_exam_percents: Dict[str, float] = {}
    total_exam_percent = 0.0
    exam_map_global: Dict[str, date] = {}
    future_exams: List[date] = []

    for subject in subjects:
        exam_map = build_exam_map(subject)
        subj_notes = compute_final_weights(subject, start, exam_map)
        if subj_notes:
            notes.extend([f"{subject.subject_name}: {msg}" for msg in subj_notes])

        blocks, module_totals, deps = expand_to_blocks(subject, start, exam_map)
        all_blocks.extend(blocks)
        module_block_totals.update(module_totals)
        dependency_requirements.update(deps)
        module_final_weights.update({m.module_key or m.module_id: m.final_weight for m in subject.modules})
        module_exam_percents.update({m.module_key or m.module_id: m.estimated_exam_percent for m in subject.modules})
        total_exam_percent += sum(m.estimated_exam_percent for m in subject.modules)
        exam_map_global.update(exam_map)
        future_exams.extend([d for d in exam_map.values() if d and d >= start])

        high_fatigue_modules = [m.module_key or m.module_id for m in subject.modules if m.fatigue_drain > config["daily_max_fatigue"]]
        if high_fatigue_modules:
            notes.append(
                f"{subject.subject_name}: Modules above fatigue cap ({config['daily_max_fatigue']}): {', '.join(high_fatigue_modules)}."
            )

    if future_exams:
        horizon_end = max(future_exams)
    else:
        horizon_end = start
        notes.append("No future exams on or after the start date; global schedule left empty.")

    availability_long = build_daily_availability(
        start_date=start,
        end_date=horizon_end,
        weekly_hours=config["weekly_hours"],
        adhoc_hours=config.get("adhoc_hours", {}),
        daily_max_hours=config.get("daily_max_hours", 8.0),
    )

    if future_exams:
        (
            schedule_map,
            day_hours,
            day_fatigue,
            remaining_blocks,
            deadline_violations,
            dependency_violations,
            total_days,
        ) = greedy_long_term(
            blocks=all_blocks,
            start_date=start,
            horizon_end=horizon_end,
            availability_map=availability_long,
            exam_map=exam_map_global,
            daily_max_fatigue=config["daily_max_fatigue"],
            dependency_requirements=dependency_requirements,
            module_block_totals=module_block_totals,
        )

        if config.get("use_sa_refine", True) and schedule_map:
            schedule_map, day_hours, day_fatigue = sa_refine_long_term(
                schedule_map=schedule_map,
                start_date=start,
                horizon_end=horizon_end,
                availability_map=availability_long,
                exam_map=exam_map_global,
                daily_max_fatigue=config["daily_max_fatigue"],
                dependency_requirements=dependency_requirements,
                module_block_totals=module_block_totals,
                iterations=config.get("sa_iterations", 1000),
            )
    else:
        schedule_map, day_hours, day_fatigue = {}, {}, {}
        remaining_blocks = all_blocks
        deadline_violations = 0
        dependency_violations = 0
        total_days = 0

    if remaining_blocks:
        notes.append(f"{len(remaining_blocks)} blocks remained unscheduled in the global long-term plan.")

    metrics_long = compute_plan_metrics(
        schedule_map=schedule_map,
        day_fatigue=day_fatigue,
        module_block_totals=module_block_totals,
        module_final_weights=module_final_weights,
        module_exam_percents=module_exam_percents,
        total_exam_percent=total_exam_percent,
        exam_map=exam_map_global,
        dependency_requirements=dependency_requirements,
        daily_max_fatigue=config["daily_max_fatigue"],
        total_days=total_days,
        deadline_violations=deadline_violations,
        dependency_violations=dependency_violations,
    )

    long_term_output = build_plan_output(
        subject_name="All Subjects (Global)",
        mode="global_long_term",
        start_date=start,
        horizon_end=horizon_end,
        availability=config,
        schedule_map=schedule_map,
        day_hours=day_hours,
        day_fatigue=day_fatigue,
        exam_map=exam_map_global,
        metrics=metrics_long,
        notes=notes,
    )

    write_plan(output_dir / "global_long_term_plan.json", long_term_output)

    # Short-term global plan
    short_horizon_end = compute_short_horizon_end(start, config["short_term_horizon_days"], exam_map_global)
    availability_short = build_daily_availability(
        start_date=start,
        end_date=short_horizon_end,
        weekly_hours=config["weekly_hours"],
        adhoc_hours=config.get("adhoc_hours", {}),
        daily_max_hours=config.get("daily_max_hours", 8.0),
    )

    (
        schedule_short,
        day_hours_short,
        day_fatigue_short,
        remaining_short,
        deadline_short,
        dependency_short,
        total_days_short,
        short_horizon_end,
    ) = greedy_short_term(
        blocks=all_blocks,
        start_date=start,
        horizon_days=config["short_term_horizon_days"],
        availability_map=availability_short,
        exam_map=exam_map_global,
        daily_max_fatigue=config["daily_max_fatigue"],
        dependency_requirements=dependency_requirements,
        module_block_totals=module_block_totals,
    )

    notes_short = list(notes)
    if remaining_short:
        notes_short.append(f"{len(remaining_short)} blocks remained unscheduled in the global short-term plan.")

    metrics_short = compute_plan_metrics(
        schedule_map=schedule_short,
        day_fatigue=day_fatigue_short,
        module_block_totals=module_block_totals,
        module_final_weights=module_final_weights,
        module_exam_percents=module_exam_percents,
        total_exam_percent=total_exam_percent,
        exam_map=exam_map_global,
        dependency_requirements=dependency_requirements,
        daily_max_fatigue=config["daily_max_fatigue"],
        total_days=total_days_short,
        deadline_violations=deadline_short,
        dependency_violations=dependency_short,
    )

    short_term_output = build_plan_output(
        subject_name="All Subjects (Global)",
        mode="global_short_term",
        start_date=start,
        horizon_end=short_horizon_end,
        availability=config,
        schedule_map=schedule_short,
        day_hours=day_hours_short,
        day_fatigue=day_fatigue_short,
        exam_map=exam_map_global,
        metrics=metrics_short,
        notes=notes_short,
    )

    write_plan(output_dir / "global_short_term_plan.json", short_term_output)

    return {
        "long_term": metrics_long,
        "short_term": metrics_short,
    }


def compute_short_horizon_end(start: date, horizon_days: int, exam_map: Dict[str, date]) -> date:
    nearest_exam = None
    future_dates = [d for d in exam_map.values() if d and d >= start]
    if future_dates:
        nearest_exam = min(future_dates)
    end_date = start + timedelta(days=horizon_days - 1)
    if nearest_exam and nearest_exam <= end_date:
        end_date = nearest_exam - timedelta(days=1)
    return end_date


def build_plan_output(
    subject_name: str,
    mode: str,
    start_date: date,
    horizon_end: date,
    availability: Dict,
    schedule_map: Dict[date, List],
    day_hours: Dict[date, float],
    day_fatigue: Dict[date, int],
    exam_map: Dict[str, date],
    metrics: PlanMetrics,
    notes: List[str],
) -> Dict:
    schedule_entries = []
    for day in sorted(schedule_map.keys()):
        blocks_output = [
            {
                "module_id": block.module_id,
                "module_key": block.module_key,
                "module_name": block.module_name,
                "subject_name": block.subject_name,
                "exam_type": block.exam_type,
                "block_hours": block.block_hours,
                "final_weight": block.final_weight,
                "value_per_hour": block.value_per_hour,
                "urgency": compute_urgency_for_day(block, day, exam_map),
                "fatigue_drain": block.fatigue_drain,
            }
            for block in schedule_map.get(day, [])
        ]
        schedule_entries.append(
            {
                "date": day.isoformat(),
                "hours_planned": day_hours.get(day, 0.0),
                "fatigue_total": day_fatigue.get(day, 0),
                "blocks": blocks_output,
            }
        )

    return {
        "subject_name": subject_name,
        "mode": mode,
        "start_date": start_date.isoformat(),
        "horizon_end": horizon_end.isoformat(),
        "daily_max_fatigue": availability.get("daily_max_fatigue", 0),
        "availability_source": {
            "weekly_hours": availability.get("weekly_hours", {}),
            "adhoc_hours": availability.get("adhoc_hours", {}),
        },
        "schedule": schedule_entries,
        "metrics": {
            "expected_score_gain": metrics.expected_score_gain,
            "exam_score_coverage": metrics.exam_score_coverage,
            "exam_score_coverage_pct": metrics.exam_score_coverage_pct,
            "fatigue_compliance_rate": metrics.fatigue_compliance_rate,
            "deadline_violations": metrics.deadline_violations,
            "dependency_violations": metrics.dependency_violations,
            "total_hours_scheduled": metrics.total_hours_scheduled,
        },
        "notes": notes,
    }


def write_plan(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote plan to %s", path)


def write_summary(
    output_dir: Path,
    start_date: date,
    subject_metrics: Dict[str, Dict[str, PlanMetrics]],
    global_metrics: Dict[str, PlanMetrics] | None = None,
) -> None:
    summary_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "start_date": start_date.isoformat(),
        "subjects": [],
    }

    for subject_name, metrics in subject_metrics.items():
        summary_payload["subjects"].append(
            {
                "subject_name": subject_name,
                "long_term": metrics["long_term"].__dict__,
                "short_term": metrics["short_term"].__dict__,
            }
        )

    if global_metrics:
        summary_payload["global"] = {
            "long_term": global_metrics["long_term"].__dict__,
            "short_term": global_metrics["short_term"].__dict__,
        }

    write_plan(output_dir / "summary_metrics.json", summary_payload)


def main() -> None:
    args = parse_args()
    config = load_config(args)
    start = parse_date(config["start_date"])

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    subjects = load_subjects_from_directory(input_dir)
    if not subjects:
        logger.warning("No subject files found in %s", input_dir)
        return

    subject_metrics: Dict[str, Dict[str, PlanMetrics]] = {}
    for subject in subjects:
        metrics = plan_subject(subject, config, start, output_dir)
        subject_metrics[subject.subject_name] = metrics

    global_metrics = plan_global(subjects, config, start, output_dir)

    write_summary(output_dir, start, subject_metrics, global_metrics=global_metrics)


if __name__ == "__main__":
    main()
