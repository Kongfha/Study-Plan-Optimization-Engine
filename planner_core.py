from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Tuple

from models import StudyBlock

logger = logging.getLogger(__name__)


def compute_spacing_multiplier(days_to_exam: int, sweet_spot: int = 10, sigma: float = 15.0) -> float:
    if days_to_exam < 0:
        return 0.0
    # Gaussian-like bump centered at sweet_spot days before exam to avoid too-early or too-late cram
    return math.exp(-((days_to_exam - sweet_spot) ** 2) / (2 * sigma * sigma))


def compute_urgency_for_day(block: StudyBlock, day: date, exam_map: Dict[str, date]) -> float:
    exam_date = block.exam_date or exam_map.get(block.exam_type) or exam_map.get(f"{block.subject_name}:{block.exam_type}")
    if exam_date:
        days_to_exam = (exam_date - day).days
        if days_to_exam < 0:
            return 0.0
        base = 1.0 / (days_to_exam + 1)
        spacing = compute_spacing_multiplier(days_to_exam)
        # Keep a floor to still allow scheduling even if spacing is low
        return base * (0.5 + spacing)
    return 0.1


def compute_priority(block: StudyBlock, day: date, exam_map: Dict[str, date]) -> float:
    urgency = compute_urgency_for_day(block, day, exam_map)
    return block.value_per_hour * urgency


def greedy_schedule(
    blocks: List[StudyBlock],
    start_date: date,
    end_date: date,
    availability_map: Dict[date, float],
    exam_map: Dict[str, date],
    daily_max_fatigue: int,
    dependency_requirements: Dict[str, Dict[str, int]],
    module_block_totals: Dict[str, int],
) -> Tuple[Dict[date, List[StudyBlock]], Dict[date, float], Dict[date, float], List[StudyBlock], int, int, int]:
    if start_date > end_date:
        return {}, {}, {}, [], 0, 0, 0

    remaining = [b for b in blocks if b.final_weight > 0]
    schedule_map: Dict[date, List[StudyBlock]] = {d: [] for d in _iter_days(start_date, end_date, exam_map)}
    day_hours: Dict[date, float] = defaultdict(float)
    day_fatigue: Dict[date, float] = defaultdict(float)
    scheduled_counts: Dict[str, int] = defaultdict(int)
    exam_dates = {d for d in exam_map.values() if d}
    deadline_violations = 0
    dependency_violations = 0
    total_days = len(schedule_map)

    current = start_date
    while current <= end_date:
        if current in exam_dates:
            current += timedelta(days=1)
            continue

        available_hours = availability_map.get(current, 0.0)
        if available_hours <= 0:
            current += timedelta(days=1)
            continue

        expired = [b for b in remaining if b.exam_date and current >= b.exam_date]
        for b in expired:
            remaining.remove(b)
            deadline_violations += 1

        while True:
            if not remaining or day_hours[current] >= available_hours:
                break

            sorted_candidates = sorted(
                remaining,
                key=lambda b: (
                    -compute_priority(b, current, exam_map),
                    -b.preparation_ease,
                    b.fatigue_drain,
                ),
            )

            chosen = None
            for cand in sorted_candidates:
                if day_hours[current] + cand.block_hours > available_hours:
                    continue
                if day_fatigue[current] + cand.fatigue_drain > daily_max_fatigue:
                    continue
                exam_date = cand.exam_date or exam_map.get(cand.exam_type) or exam_map.get(f"{cand.subject_name}:{cand.exam_type}")
                if exam_date and current >= exam_date:
                    continue

                if scheduled_counts[cand.module_key] == 0:
                    deps = dependency_requirements.get(cand.module_key, {})
                    dep_ok = True
                    for dep, required in deps.items():
                        if scheduled_counts.get(dep, 0) < required:
                            dep_ok = False
                            break
                    if not dep_ok:
                        continue

                chosen = cand
                break

            if chosen is None:
                break

            schedule_map[current].append(chosen)
            remaining.remove(chosen)
            day_hours[current] += chosen.block_hours
            day_fatigue[current] += chosen.fatigue_drain
            scheduled_counts[chosen.module_key] += 1

        current += timedelta(days=1)

    return schedule_map, day_hours, day_fatigue, remaining, deadline_violations, dependency_violations, total_days


def _iter_days(start_date: date, end_date: date, exam_map: Dict[str, date]) -> List[date]:
    days: List[date] = []
    current = start_date
    exam_dates = {d for d in exam_map.values() if d}
    while current <= end_date:
        if current not in exam_dates:
            days.append(current)
        current += timedelta(days=1)
    return days
