from __future__ import annotations

import logging
import math
from datetime import date, datetime
from typing import Dict, List, Tuple

from models import Module, StudyBlock, Subject

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.15
DEFAULT_BETA = 0.10


def build_exam_map(subject: Subject) -> Dict[str, date]:
    exam_map: Dict[str, date] = {}
    for exam in subject.exams:
        try:
            parsed = datetime.strptime(exam.exam_date, "%Y-%m-%d").date()
            exam.exam_day = parsed
            exam_map[exam.exam_name] = parsed
            exam_map[f"{subject.subject_name}:{exam.exam_name}"] = parsed
        except Exception as exc:
            logger.warning("Could not parse exam date %s (%s): %s", exam.exam_name, exam.exam_date, exc)
    return exam_map


def compute_final_weights(
    subject: Subject,
    start_date: date,
    exam_map: Dict[str, date],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> List[str]:
    notes: List[str] = []
    for module in subject.modules:
        exam_day = exam_map.get(module.exam_type)
        if exam_day and start_date > exam_day:
            module.final_weight = 0.0
            module.is_past_exam = True
            notes.append(
                f"Module {module.module_id} tied to past exam {module.exam_type} on {exam_day}; deprioritized."
            )
            continue

        ease_term = 1 + alpha * (module.preparation_ease - 3)
        fatigue_penalty = 1 - beta * max(module.fatigue_drain - 5, 0)
        module.final_weight = max(module.estimated_exam_percent * ease_term * fatigue_penalty, 0.0)

        if not exam_day:
            module.warning = f"Exam type {module.exam_type} not found; urgency lowered."
            notes.append(module.warning)
    return notes


def expand_to_blocks(
    subject: Subject,
    start_date: date,
    exam_map: Dict[str, date],
) -> Tuple[List[StudyBlock], Dict[str, int], Dict[str, Dict[str, int]]]:
    blocks: List[StudyBlock] = []
    module_block_totals: Dict[str, int] = {}
    dependency_requirements: Dict[str, Dict[str, int]] = {}
    block_counter = 0

    for module in subject.modules:
        module.module_key = f"{subject.subject_name}:{module.module_id}"
        n_blocks = max(1, math.ceil(module.estimated_time_hrs))
        module.n_blocks = n_blocks
        module.value_per_hour = module.final_weight / max(module.estimated_time_hrs, 1e-6)
        module_block_totals[module.module_key] = n_blocks

    for module in subject.modules:
        exam_day = exam_map.get(module.exam_type)
        module_key = module.module_key or module.module_id
        required_deps: Dict[str, int] = {}
        for dep in module.dependency_modules:
            dep_key = f"{subject.subject_name}:{dep}"
            dep_blocks = module_block_totals.get(dep_key, 0)
            dep_module = next((m for m in subject.modules if m.module_id == dep), None)
            dep_weight = dep_module.final_weight if dep_module else 0.0
            if dep_blocks <= 0 or dep_weight <= 0:
                required_deps[dep_key] = 0
            else:
                required_deps[dep_key] = max(1, math.ceil(0.3 * dep_blocks))
            if dep_key not in module_block_totals:
                logger.warning("Dependency %s referenced by %s not found in subject %s", dep, module.module_id, subject.subject_name)
        dependency_requirements[module_key] = required_deps

        for i in range(module.n_blocks):
            block_counter += 1
            per_block_fatigue = module.fatigue_drain / max(module.n_blocks, 1)
            blocks.append(
                StudyBlock(
                    block_uid=f"{module_key}_{i+1}",
                    module_id=module.module_id,
                    module_key=module_key,
                    subject_name=subject.subject_name,
                    module_name=module.module_name,
                    exam_type=module.exam_type,
                    exam_date=exam_day,
                    fatigue_drain=per_block_fatigue,
                    final_weight=module.final_weight,
                    value_per_hour=module.value_per_hour,
                    preparation_ease=module.preparation_ease,
                )
            )

    blocks.sort(key=lambda b: b.module_key)
    return blocks, module_block_totals, dependency_requirements
