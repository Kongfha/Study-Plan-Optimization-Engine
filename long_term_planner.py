from __future__ import annotations

import copy
import logging
import math
import random
from datetime import date
from typing import Dict, List, Tuple

from models import StudyBlock
from planner_core import compute_priority, greedy_schedule

logger = logging.getLogger(__name__)


def greedy_long_term(
    blocks: List[StudyBlock],
    start_date: date,
    horizon_end: date,
    availability_map: Dict[date, float],
    exam_map: Dict[str, date],
    daily_max_fatigue: int,
    dependency_requirements: Dict[str, Dict[str, int]],
    module_block_totals: Dict[str, int],
) -> Tuple[Dict[date, List[StudyBlock]], Dict[date, float], Dict[date, float], List[StudyBlock], int, int, int]:
    return greedy_schedule(
        blocks=blocks,
        start_date=start_date,
        end_date=horizon_end,
        availability_map=availability_map,
        exam_map=exam_map,
        daily_max_fatigue=daily_max_fatigue,
        dependency_requirements=dependency_requirements,
        module_block_totals=module_block_totals,
    )


def sa_refine_long_term(
    schedule_map: Dict[date, List[StudyBlock]],
    start_date: date,
    horizon_end: date,
    availability_map: Dict[date, float],
    exam_map: Dict[str, date],
    daily_max_fatigue: int,
    dependency_requirements: Dict[str, Dict[str, int]],
    module_block_totals: Dict[str, int],
    iterations: int = 1000,
    cooling: float = 0.995,
) -> Tuple[Dict[date, List[StudyBlock]], Dict[date, float], Dict[date, float]]:
    if not schedule_map:
        return schedule_map, {}, {}

    rng = random.Random(42)
    exam_dates = {d for d in exam_map.values() if d}
    all_days = [
        day
        for day in availability_map
        if start_date <= day <= horizon_end and day not in exam_dates
    ]

    current = copy.deepcopy(schedule_map)
    best = copy.deepcopy(schedule_map)
    current_obj = _objective(current, exam_map)
    best_obj = current_obj
    temperature = 1.0

    for _ in range(iterations):
        if temperature < 1e-4:
            break
        days_with_blocks = [d for d, blocks in current.items() if blocks]
        if not days_with_blocks:
            break

        day_a = rng.choice(days_with_blocks)
        block_idx = rng.randrange(len(current[day_a]))
        block = current[day_a][block_idx]
        day_b = rng.choice(all_days)
        if day_b == day_a:
            temperature *= cooling
            continue

        proposal = copy.deepcopy(current)
        proposal[day_a].pop(block_idx)
        proposal.setdefault(day_b, []).append(block)

        if not _is_feasible(
            proposal,
            availability_map,
            exam_map,
            daily_max_fatigue,
            dependency_requirements,
            module_block_totals,
        ):
            temperature *= cooling
            continue

        new_obj = _objective(proposal, exam_map)
        delta = new_obj - current_obj
        accept = delta >= 0 or math.exp(delta / max(temperature, 1e-6)) > rng.random()
        if accept:
            current = proposal
            current_obj = new_obj
            if new_obj > best_obj:
                best = proposal
                best_obj = new_obj

        temperature *= cooling

    day_hours, day_fatigue = _recompute_day_stats(best)
    return best, day_hours, day_fatigue


def _objective(schedule_map: Dict[date, List[StudyBlock]], exam_map: Dict[str, date]) -> float:
    total = 0.0
    for day, blocks in schedule_map.items():
        for block in blocks:
            total += compute_priority(block, day, exam_map)
    return total


def _recompute_day_stats(schedule_map: Dict[date, List[StudyBlock]]) -> Tuple[Dict[date, float], Dict[date, int]]:
    day_hours: Dict[date, float] = {}
    day_fatigue: Dict[date, float] = {}
    for day, blocks in schedule_map.items():
        day_hours[day] = sum(b.block_hours for b in blocks)
        day_fatigue[day] = sum(b.fatigue_drain for b in blocks)
    return day_hours, day_fatigue


def _is_feasible(
    schedule_map: Dict[date, List[StudyBlock]],
    availability_map: Dict[date, float],
    exam_map: Dict[str, date],
    daily_max_fatigue: int,
    dependency_requirements: Dict[str, Dict[str, int]],
    module_block_totals: Dict[str, int],
) -> bool:
    from collections import defaultdict
    scheduled_counts: Dict[str, int] = defaultdict(int)

    for day, blocks in schedule_map.items():
        available_hours = availability_map.get(day, 0.0)
        total_hours = sum(b.block_hours for b in blocks)
        total_fatigue = sum(b.fatigue_drain for b in blocks)
        if total_hours - 1e-6 > available_hours:
            return False
        if total_fatigue > daily_max_fatigue:
            return False

    for day in sorted(schedule_map.keys()):
        for block in schedule_map[day]:
            exam_date = block.exam_date or exam_map.get(block.exam_type) or exam_map.get(f"{block.subject_name}:{block.exam_type}")
            if exam_date and day >= exam_date:
                return False
            if scheduled_counts[block.module_key] == 0:
                deps = dependency_requirements.get(block.module_key, {})
                for dep, required in deps.items():
                    if scheduled_counts.get(dep, 0) < required:
                        return False
            scheduled_counts[block.module_key] += 1

    return True
