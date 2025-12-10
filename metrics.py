from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Dict, List

from models import PlanMetrics, StudyBlock


def compute_plan_metrics(
    schedule_map: Dict[date, List[StudyBlock]],
    day_fatigue: Dict[date, float],
    module_block_totals: Dict[str, int],
    module_final_weights: Dict[str, float],
    module_exam_percents: Dict[str, float],
    total_exam_percent: float,
    exam_map: Dict[str, date],
    dependency_requirements: Dict[str, Dict[str, int]],
    daily_max_fatigue: int,
    total_days: int,
    deadline_violations: int,
    dependency_violations: int,
) -> PlanMetrics:
    scheduled_counts: Dict[str, int] = defaultdict(int)
    total_hours = 0.0

    for day, blocks in schedule_map.items():
        for block in blocks:
            scheduled_counts[block.module_key] += 1
            total_hours += block.block_hours

    expected_score_gain = 0.0
    exam_score_coverage = 0.0
    for module_key, count in scheduled_counts.items():
        n_blocks = module_block_totals.get(module_key, 0)
        final_weight = module_final_weights.get(module_key, 0.0)
        exam_percent = module_exam_percents.get(module_key, 0.0)
        if n_blocks > 0:
            expected_score_gain += final_weight * (count / n_blocks)
            exam_score_coverage += exam_percent * (count / n_blocks)
    exam_score_coverage_pct = 0.0
    if total_exam_percent > 1e-6:
        exam_score_coverage_pct = exam_score_coverage / total_exam_percent

    fatigue_compliance_rate = 1.0
    if total_days > 0:
        compliant_days = sum(1 for day in schedule_map if day_fatigue.get(day, 0) <= daily_max_fatigue)
        fatigue_compliance_rate = compliant_days / total_days

    deadline_violations += _count_deadline_violations(schedule_map, exam_map)
    dependency_violations += _count_dependency_violations(schedule_map, dependency_requirements)

    return PlanMetrics(
        expected_score_gain=expected_score_gain,
        exam_score_coverage=exam_score_coverage,
        exam_score_coverage_pct=exam_score_coverage_pct,
        total_hours_scheduled=total_hours,
        fatigue_compliance_rate=fatigue_compliance_rate,
        deadline_violations=deadline_violations,
        dependency_violations=dependency_violations,
    )


def _count_deadline_violations(schedule_map: Dict[date, List[StudyBlock]], exam_map: Dict[str, date]) -> int:
    violations = 0
    for day, blocks in schedule_map.items():
        for block in blocks:
            exam_day = block.exam_date or exam_map.get(block.exam_type) or exam_map.get(f"{block.subject_name}:{block.exam_type}")
            if exam_day and day >= exam_day:
                violations += 1
    return violations


def _count_dependency_violations(
    schedule_map: Dict[date, List[StudyBlock]],
    dependency_requirements: Dict[str, Dict[str, int]],
) -> int:
    violations = 0
    scheduled_counts: Dict[str, int] = defaultdict(int)

    for day in sorted(schedule_map.keys()):
        blocks = schedule_map[day]
        for block in blocks:
            if scheduled_counts[block.module_key] == 0:
                deps = dependency_requirements.get(block.module_key, {})
                for dep, required in deps.items():
                    if scheduled_counts.get(dep, 0) < required:
                        violations += 1
                        break
            scheduled_counts[block.module_key] += 1
    return violations
