from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional


@dataclass
class Module:
    module_id: str
    module_name: str
    exam_type: str
    estimated_exam_percent: float
    estimated_time_hrs: float
    preparation_ease: int
    fatigue_drain: int  # per-hour fatigue intensity from input
    dependency_modules: List[str]
    final_weight: float = 0.0
    value_per_hour: float = 0.0
    n_blocks: int = 0
    is_past_exam: bool = False
    warning: Optional[str] = None
    module_key: Optional[str] = None


@dataclass
class Exam:
    exam_name: str
    exam_date: str
    score_percentage: float
    exam_day: Optional[date] = None


@dataclass
class Subject:
    subject_name: str
    subject_credit: float
    is_major: bool
    exams: List[Exam]
    modules: List[Module]
    subject_id: str = ""
    semester: str = ""
    academic_year: str = ""
    instructors: List[str] = field(default_factory=list)


@dataclass
class StudyPlanInput:
    subjects: List[Subject]
    daily_max_hours: float = 8.0
    daily_max_fatigue: int = 7


@dataclass
class StudyBlock:
    block_uid: str
    module_id: str
    module_name: str
    exam_type: str
    subject_name: str
    module_key: str
    exam_date: Optional[date]
    fatigue_drain: float  # total fatigue for this block (fatigue-per-hour * block_hours)
    final_weight: float
    value_per_hour: float
    preparation_ease: int
    block_hours: float = 1.0


@dataclass
class ScheduledBlock:
    day: date
    block: StudyBlock
    urgency: float


@dataclass
class PlanMetrics:
    expected_score_gain: float
    exam_score_coverage: float
    exam_score_coverage_pct: float
    total_hours_scheduled: float
    fatigue_compliance_rate: float
    deadline_violations: int
    dependency_violations: int
