from typing import Any

from sqlalchemy.orm import Session

from app.models.job import Job
from app.models.student_saved_analysis import StudentSavedAnalysis


def _extract_snapshot(result: dict[str, Any] | None) -> tuple[str | None, str | None, str | None]:
    if not isinstance(result, dict):
        return None, None, None

    emotion = result.get("emotion") if isinstance(result.get("emotion"), dict) else {}
    recommendation = result.get("recommendation") if isinstance(result.get("recommendation"), dict) else {}

    mood = emotion.get("predicted_mood") or emotion.get("emotion")

    confidence_value = None
    probabilities = emotion.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        try:
            max_prob = max(float(value) for value in probabilities.values())
            confidence_value = f"{max_prob * 100:.0f}%"
        except (TypeError, ValueError):
            confidence_value = None

    summary = recommendation.get("RecommendationText")

    return (
        str(mood) if mood is not None else None,
        str(confidence_value) if confidence_value is not None else None,
        str(summary) if summary is not None else None,
    )


def create_saved_analysis(db: Session, student_id: int, job: Job) -> StudentSavedAnalysis:
    existing = (
        db.query(StudentSavedAnalysis)
        .filter(StudentSavedAnalysis.student_id == student_id, StudentSavedAnalysis.job_id == job.job_id)
        .first()
    )
    if existing:
        return existing

    mood, confidence, summary = _extract_snapshot(job.result)
    item = StudentSavedAnalysis(
        student_id=student_id,
        job_id=job.job_id,
        mood=mood,
        confidence=confidence,
        summary=summary,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def list_saved_analyses(db: Session, student_id: int) -> list[StudentSavedAnalysis]:
    return (
        db.query(StudentSavedAnalysis)
        .filter(StudentSavedAnalysis.student_id == student_id)
        .order_by(StudentSavedAnalysis.saved_at.desc())
        .all()
    )
