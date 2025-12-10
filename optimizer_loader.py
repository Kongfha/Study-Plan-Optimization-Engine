from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from models import Exam, Module, Subject

logger = logging.getLogger(__name__)


def _parse_module(data: dict) -> Module:
    return Module(
        module_id=data["module_id"],
        module_name=data.get("module_name", ""),
        exam_type=data.get("exam_type", ""),
        estimated_exam_percent=float(data.get("estimated_exam_percent", 0.0)),
        estimated_time_hrs=float(data.get("estimated_time_hrs", 0.0)),
        preparation_ease=int(data.get("preparation_ease", 3)),
        fatigue_drain=int(data.get("fatigue_drain", 5)),
        dependency_modules=list(data.get("dependency_modules", [])),
    )


def _parse_exam(data: dict) -> Exam:
    return Exam(
        exam_name=data.get("exam_name", ""),
        exam_date=data.get("exam_date", ""),
        score_percentage=float(data.get("score_percentage", 0.0)),
    )


def _parse_subject(data: dict) -> Subject:
    exams = [_parse_exam(e) for e in data.get("exams", [])]
    modules = [_parse_module(m) for m in data.get("modules", [])]
    return Subject(
        subject_name=data.get("subject_name", ""),
        subject_credit=float(data.get("subject_credit", 0.0)),
        is_major=bool(data.get("is_major", False)),
        exams=exams,
        modules=modules,
        subject_id=str(data.get("subject_id", "")),
        semester=data.get("semester", ""),
        academic_year=str(data.get("academic_year", "")),
        instructors=list(data.get("instructors", [])),
    )


def load_subjects_from_json(path: Path) -> List[Subject]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "subjects" in payload and isinstance(payload["subjects"], list):
        subjects = [_parse_subject(item) for item in payload["subjects"]]
    else:
        subjects = [_parse_subject(payload)]

    logger.info("Loaded %d subjects from %s", len(subjects), path)
    return subjects


def load_subjects_from_directory(directory: Path) -> List[Subject]:
    subjects: List[Subject] = []
    for json_path in sorted(directory.glob("*.json")):
        try:
            subjects.extend(load_subjects_from_json(json_path))
        except Exception as exc:
            logger.exception("Failed to load %s: %s", json_path, exc)
    return subjects
