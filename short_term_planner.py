from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Tuple

from models import StudyBlock
from planner_core import greedy_schedule


def greedy_short_term(
    blocks: List[StudyBlock],
    start_date: date,
    horizon_days: int,
    availability_map: Dict[date, float],
    exam_map: Dict[str, date],
    daily_max_fatigue: int,
    dependency_requirements: Dict[str, Dict[str, int]],
    module_block_totals: Dict[str, int],
) -> Tuple[Dict[date, List[StudyBlock]], Dict[date, float], Dict[date, float], List[StudyBlock], int, int, int, date]:
    nearest_exam = _nearest_future_exam(exam_map, start_date)

    if nearest_exam:
        filtered_blocks = [b for b in blocks if b.exam_date == nearest_exam]
    else:
        filtered_blocks = list(blocks)

    horizon_end = start_date + timedelta(days=horizon_days - 1)
    if nearest_exam and nearest_exam <= horizon_end:
        horizon_end = nearest_exam - timedelta(days=1)

    schedule = greedy_schedule(
        blocks=filtered_blocks,
        start_date=start_date,
        end_date=horizon_end,
        availability_map=availability_map,
        exam_map=exam_map,
        daily_max_fatigue=daily_max_fatigue,
        dependency_requirements=dependency_requirements,
        module_block_totals=module_block_totals,
    )
    return (*schedule, horizon_end)


def _nearest_future_exam(exam_map: Dict[str, date], start: date) -> date | None:
    future_dates = [d for d in exam_map.values() if d and d >= start]
    if not future_dates:
        return None
    return min(future_dates)
