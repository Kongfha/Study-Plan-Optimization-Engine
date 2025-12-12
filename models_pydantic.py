"""Pydantic models for LLM output validation and structured extraction."""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ModulePydantic(BaseModel):
    """Module within a subject - a specific topic or chapter to study."""

    module_id: str = Field(...,
                           description="Unique identifier for the module (e.g., 'M1', 'M2')")
    module_name: str = Field(...,
                             description="Name/title of the module or topic")
    exam_type: str = Field(
        ..., description="Which exam this module is for (e.g., 'Midterm', 'Final')")
    estimated_exam_percent: float = Field(
        ...,
        description="Estimated percentage this module contributes to the exam score (0-100)",
        ge=0,
        le=100
    )
    estimated_time_hrs: float = Field(
        ...,
        description="Estimated hours needed to study this module",
        gt=0
    )
    preparation_ease: int = Field(
        ...,
        description="How easy it is to prepare (1=very hard, 5=very easy)",
        ge=1,
        le=5
    )
    fatigue_drain: int = Field(
        ...,
        description="Total fatigue drain for the entire module (1-10 scale)",
        ge=1,
        le=10
    )
    dependency_modules: List[str] = Field(
        default_factory=list,
        description="List of module_ids that should be studied before this one"
    )
    is_past_exam: bool = Field(
        default=False,
        description="Whether this is a past exam paper module"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "module_id": "M1",
                "module_name": "Introduction to Linear Algebra",
                "exam_type": "Midterm",
                "estimated_exam_percent": 20.0,
                "estimated_time_hrs": 8.0,
                "preparation_ease": 3,
                "fatigue_drain": 5,
                "dependency_modules": [],
                "is_past_exam": False
            }
        }


class ExamPydantic(BaseModel):
    """Exam information for a subject."""

    exam_name: str = Field(...,
                           description="Name of the exam (e.g., 'Midterm', 'Final Exam')")
    exam_date: str = Field(...,
                           description="Date of the exam in YYYY-MM-DD format")
    score_percentage: float = Field(
        ...,
        description="Percentage this exam contributes to the final grade (0-100)",
        ge=0,
        le=100
    )

    @field_validator('exam_date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Ensure date is in correct format."""
        try:
            date.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError(
                f"exam_date must be in YYYY-MM-DD format, got: {v}")

    class Config:
        json_schema_extra = {
            "example": {
                "exam_name": "Midterm Exam",
                "exam_date": "2025-03-15",
                "score_percentage": 40.0
            }
        }


class SubjectPydantic(BaseModel):
    """Complete subject/course information including exams and study modules."""

    subject_name: str = Field(..., description="Name of the subject/course")
    subject_credit: float = Field(...,
                                  description="Credit hours for this subject", gt=0)
    is_major: bool = Field(...,
                           description="Whether this is a major/core subject")
    exams: List[ExamPydantic] = Field(
        default_factory=list,
        description="List of exams for this subject (at least 1 required if available)"
    )
    modules: List[ModulePydantic] = Field(
        ...,
        description="List of study modules/topics for this subject",
        min_length=1
    )
    subject_id: str = Field(default="", description="Optional subject ID/code")
    semester: str = Field(default="", description="Semester (e.g., '1/2024')")
    academic_year: str = Field(default="", description="Academic year")
    instructors: List[str] = Field(
        default_factory=list,
        description="List of instructor names"
    )

    @field_validator('modules')
    @classmethod
    def validate_exam_percentages(cls, modules: List[ModulePydantic]) -> List[ModulePydantic]:
        """Warn if exam percentages don't sum to ~100 per exam type."""
        exam_totals = {}
        for module in modules:
            exam_type = module.exam_type
            exam_totals[exam_type] = exam_totals.get(
                exam_type, 0) + module.estimated_exam_percent

        # Just validate, don't fail - sometimes estimates might not be exact
        for exam_type, total in exam_totals.items():
            if total > 110:  # Allow some margin
                import warnings
                warnings.warn(
                    f"Exam '{exam_type}' module percentages sum to {total}%, which exceeds 100%"
                )

        return modules

    class Config:
        json_schema_extra = {
            "example": {
                "subject_name": "Linear Algebra",
                "subject_credit": 3.0,
                "is_major": True,
                "subject_id": "MATH201",
                "semester": "1/2025",
                "academic_year": "2024-2025",
                "instructors": ["Dr. Smith"],
                "exams": [
                    {
                        "exam_name": "Midterm",
                        "exam_date": "2025-03-15",
                        "score_percentage": 40.0
                    },
                    {
                        "exam_name": "Final",
                        "exam_date": "2025-05-20",
                        "score_percentage": 60.0
                    }
                ],
                "modules": [
                    {
                        "module_id": "M1",
                        "module_name": "Vectors and Matrices",
                        "exam_type": "Midterm",
                        "estimated_exam_percent": 25.0,
                        "estimated_time_hrs": 8.0,
                        "preparation_ease": 3,
                        "fatigue_drain": 5,
                        "dependency_modules": [],
                        "is_past_exam": False
                    }
                ]
            }
        }
