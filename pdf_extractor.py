"""PDF syllabus extraction using LangChain and LLM with structured output."""

import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader

from models_pydantic import SubjectPydantic

# Load environment variables from .env file
load_dotenv()


EXTRACTION_PROMPT = """You are an expert educational data extractor. Your task is to extract structured information from a course syllabus PDF.

Extract the following information and format it according to the JSON schema provided:

## Course Syllabus Content:
{syllabus_text}

## Instructions:

1. **Subject Information**: Extract course name, credits, subject code, semester, academic year, and instructor names.

2. **Exams**: Identify all exams (midterm, final, quizzes, etc.) with:
   - Exact exam name
   - Exam date (in YYYY-MM-DD format - if year is not specified, infer from context or use 2025)
   - Percentage of final grade (ensure all exam percentages sum to 100%)
   - **IMPORTANT**: If no exams are explicitly mentioned in the syllabus, create at least one generic exam (e.g., "Final Exam" on 2025-05-15 with 100% weight) so the exams list is never empty.

3. **Study Modules**: Break down the course content into study modules/topics. For each module:
   - **module_id**: Use format "M1", "M2", "M3", etc.
   - **module_name**: The topic or chapter name
   - **exam_type**: One of the exam names from the exams list (e.g., if exams include "Midterm Exam" and "Final Exam", use those exact names)
   - **estimated_exam_percent**: Estimate what percentage of that exam this module contributes to (distribute the exam's total percentage across its modules)
   - **estimated_time_hrs**: Estimate study hours needed based on complexity (simple topics: 4-6 hrs, moderate: 8-12 hrs, complex: 15-20 hrs)
   - **preparation_ease**: Rate 1-5 (1=very hard, 5=very easy) based on complexity and prerequisites
   - **fatigue_drain**: Rate 1-10 per study hour (fatigue intensity) based on mental effort and topic difficulty
   - **dependency_modules**: List module_ids that should be studied before this one (e.g., ["M1", "M2"])
   - **is_past_exam**: Set to false for regular topics, true only if explicitly mentioned as past exam practice

4. **Estimation Guidelines**:
   - For modules covering fundamental concepts: preparation_ease=4-5, fatigue_drain=3-5
   - For modules with complex mathematics/theory: preparation_ease=2-3, fatigue_drain=7-9
   - For application/practice modules: preparation_ease=3-4, fatigue_drain=5-7
   - Ensure exam percentages across all modules for each exam type roughly sum to that exam's total percentage

5. **is_major**: Set to true if this appears to be a core/major subject for the program, false if it's an elective or general education course.

{format_instructions}

Important: Return ONLY the valid JSON object, no additional text or explanation.
"""


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text content from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text_parts = []

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{text}")

        full_text = "\n\n".join(text_parts)

        if not full_text.strip():
            raise ValueError("No text could be extracted from the PDF")

        return full_text

    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")


def extract_subject_from_pdf(
    pdf_path: Path,
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.1,
    api_key: Optional[str] = None
) -> SubjectPydantic:
    """
    Extract structured subject information from a PDF syllabus using LLM.

    Args:
        pdf_path: Path to the PDF syllabus file
        model_name: Google Gemini model to use (default: gemini-2.5-flash-lite)
        temperature: LLM temperature (lower = more consistent, default: 0.1)
        api_key: Google API key (if not provided, reads from GOOGLE_API_KEY env var)

    Returns:
        SubjectPydantic: Validated subject information

    Raises:
        ValueError: If extraction or validation fails
        RuntimeError: If PDF reading fails
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )

    # Extract text from PDF
    print(f"Extracting text from {pdf_path.name}...")
    syllabus_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(syllabus_text)} characters from PDF")

    # Set up LangChain components
    parser = JsonOutputParser(pydantic_object=SubjectPydantic)

    prompt = ChatPromptTemplate.from_template(
        template=EXTRACTION_PROMPT,
        partial_variables={
            "format_instructions": parser.get_format_instructions()}
    )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key
    )

    # Create the chain
    chain = prompt | llm | parser

    # Run extraction
    print(f"Running LLM extraction with {model_name}...")
    try:
        result = chain.invoke({"syllabus_text": syllabus_text})

        # Add default exam if none extracted
        if not result.get("exams"):
            print("⚠️  No exams found in extraction, adding default exam...")
            result["exams"] = [{
                "exam_name": "Final Exam",
                "exam_date": "2025-05-15",
                "score_percentage": 100.0
            }]

        # Validate with Pydantic
        subject = SubjectPydantic(**result)
        print(f"✓ Successfully extracted subject: {subject.subject_name}")
        print(f"  - {len(subject.exams)} exams")
        print(f"  - {len(subject.modules)} modules")

        return subject

    except Exception as e:
        raise ValueError(
            f"Failed to extract or validate subject data: {str(e)}")


def save_subject_to_json(subject: SubjectPydantic, output_path: Path) -> None:
    """Save a validated Subject to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(subject.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"✓ Saved to {output_path}")


def process_pdf_syllabus(
    pdf_path: Path,
    output_dir: Path,
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.1,
    api_key: Optional[str] = None
) -> Path:
    """
    Complete pipeline: Extract subject from PDF and save to JSON.

    Args:
        pdf_path: Path to PDF syllabus
        output_dir: Directory to save the output JSON
        model_name: Google Gemini model name
        temperature: LLM temperature
        api_key: Google API key

    Returns:
        Path to the saved JSON file
    """
    subject = extract_subject_from_pdf(
        pdf_path, model_name, temperature, api_key)

    # Create output filename based on subject name
    safe_name = "".join(c if c.isalnum() or c in (
        " ", "-", "_") else "_" for c in subject.subject_name)
    safe_name = safe_name.strip().replace(" ", "-").lower()
    output_path = output_dir / f"{safe_name}.json"

    save_subject_to_json(subject, output_path)

    return output_path


# CLI for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_path> [output_dir]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(
        sys.argv) > 2 else Path("Subjects_Input")

    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)

    try:
        output_path = process_pdf_syllabus(pdf_path, output_dir)
        print(f"\n✓ Success! Subject JSON saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
