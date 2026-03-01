"""
schema.py - JSON Schema Validation for Resume and Job Data

Ensures all input data files follow the expected structure before
they are ingested into Endee. Reports clear errors for malformed data.
"""

from typing import Dict, Any, List, Tuple

# ─── Schema definitions ────────────────────────────────────────────

RESUME_SCHEMA = {
    "required": ["id", "name", "title", "summary", "skills"],
    "optional": [
        "email", "location", "years_experience", "is_open_to_work",
        "experience", "education", "certifications", "languages",
    ],
    "types": {
        "id": str,
        "name": str,
        "email": str,
        "title": str,
        "location": str,
        "years_experience": (int, float),
        "is_open_to_work": bool,
        "summary": str,
        "skills": list,
        "experience": list,
        "education": list,
    },
}

JOB_SCHEMA = {
    "required": ["id", "title", "description", "required_skills"],
    "optional": [
        "company", "location", "min_experience", "remote_friendly",
        "preferred_skills", "responsibilities", "salary_range",
        "employment_type", "department",
    ],
    "types": {
        "id": str,
        "title": str,
        "company": str,
        "location": str,
        "min_experience": (int, float),
        "remote_friendly": bool,
        "description": str,
        "required_skills": list,
        "preferred_skills": list,
        "responsibilities": list,
        "salary_range": str,
    },
}


# ─── Validation functions ──────────────────────────────────────────

def validate_document(
    doc: Dict[str, Any],
    schema: Dict[str, Any],
    doc_label: str = "document",
) -> Tuple[bool, List[str]]:
    """
    Validate a document against a schema.

    Args:
        doc: The document dictionary to validate
        schema: The schema definition (RESUME_SCHEMA or JOB_SCHEMA)
        doc_label: Human-readable label for error messages

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Check required fields
    for field in schema["required"]:
        if field not in doc:
            errors.append(f"[{doc_label}] Missing required field: '{field}'")
        elif doc[field] is None or doc[field] == "":
            errors.append(f"[{doc_label}] Required field '{field}' is empty")

    # Check types
    for field, expected_type in schema["types"].items():
        if field in doc and doc[field] is not None:
            if not isinstance(doc[field], expected_type):
                actual = type(doc[field]).__name__
                expected = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
                errors.append(
                    f"[{doc_label}] Field '{field}' has wrong type: "
                    f"expected {expected}, got {actual}"
                )

    # Check skills are non-empty lists of strings
    for skills_field in ["skills", "required_skills", "preferred_skills"]:
        if skills_field in doc and isinstance(doc[skills_field], list):
            if len(doc[skills_field]) == 0:
                errors.append(f"[{doc_label}] '{skills_field}' list is empty")
            for i, skill in enumerate(doc[skills_field]):
                if not isinstance(skill, str):
                    errors.append(
                        f"[{doc_label}] '{skills_field}[{i}]' should be string, got {type(skill).__name__}"
                    )

    # Check experience entries
    if "experience" in doc and isinstance(doc.get("experience"), list):
        for i, exp in enumerate(doc["experience"]):
            if not isinstance(exp, dict):
                errors.append(f"[{doc_label}] experience[{i}] should be a dict")
            elif "title" not in exp:
                errors.append(f"[{doc_label}] experience[{i}] missing 'title'")

    # Check education entries
    if "education" in doc and isinstance(doc.get("education"), list):
        for i, edu in enumerate(doc["education"]):
            if not isinstance(edu, dict):
                errors.append(f"[{doc_label}] education[{i}] should be a dict")

    # Check ID format (should be safe for Endee)
    if "id" in doc and isinstance(doc["id"], str):
        if len(doc["id"]) > 128:
            errors.append(f"[{doc_label}] 'id' is too long (max 128 chars)")
        if " " in doc["id"]:
            errors.append(f"[{doc_label}] 'id' should not contain spaces")

    return len(errors) == 0, errors


def validate_resume(doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a resume document."""
    label = f"Resume:{doc.get('id', doc.get('name', '?'))}"
    return validate_document(doc, RESUME_SCHEMA, doc_label=label)


def validate_job(doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a job document."""
    label = f"Job:{doc.get('id', doc.get('title', '?'))}"
    return validate_document(doc, JOB_SCHEMA, doc_label=label)


# ─── Batch validation ──────────────────────────────────────────────

def validate_batch(
    docs: List[Dict[str, Any]],
    schema_type: str = "resume",
) -> Tuple[int, int, List[str]]:
    """
    Validate a batch of documents.

    Args:
        docs: List of document dicts
        schema_type: "resume" or "job"

    Returns:
        Tuple of (valid_count, invalid_count, all_errors)
    """
    validate_fn = validate_resume if schema_type == "resume" else validate_job
    all_errors: List[str] = []
    valid = 0
    invalid = 0

    # Check for duplicate IDs
    ids_seen = set()
    for doc in docs:
        doc_id = doc.get("id", "")
        if doc_id in ids_seen:
            all_errors.append(f"Duplicate ID found: '{doc_id}'")
            invalid += 1
            continue
        ids_seen.add(doc_id)

        is_valid, errors = validate_fn(doc)
        if is_valid:
            valid += 1
        else:
            invalid += 1
            all_errors.extend(errors)

    return valid, invalid, all_errors


if __name__ == "__main__":
    # Quick test
    test_resume = {
        "id": "test_001",
        "name": "Test User",
        "title": "Developer",
        "summary": "A test resume.",
        "skills": ["Python", "Java"],
    }
    ok, errs = validate_resume(test_resume)
    print(f"Valid: {ok}, Errors: {errs}")

    test_bad = {"id": 123, "title": "Bad"}
    ok2, errs2 = validate_resume(test_bad)
    print(f"Valid: {ok2}, Errors: {errs2}")
