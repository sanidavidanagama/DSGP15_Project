import os
from datetime import datetime
from typing import Any

import requests


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_TEACHER_ID = os.getenv("TEACHER_ID", "dev-teacher")


class ClassApiError(Exception):
    pass


def _parse_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            if "detail" in payload:
                return str(payload["detail"])
            if "message" in payload:
                return str(payload["message"])
        return str(payload)
    except Exception:
        return response.text or "Unexpected backend error"


def _request(method: str, path: str, teacher_id: str, **kwargs: Any) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers = {**headers, "X-Teacher-Id": teacher_id}

    try:
        response = requests.request(
            method=method,
            url=f"{BACKEND_BASE_URL}{path}",
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise ClassApiError(f"Failed to connect to backend: {exc}") from exc

    if response.status_code >= 400:
        raise ClassApiError(_parse_error_message(response))

    return response


def get_classes(teacher_id: str = DEFAULT_TEACHER_ID) -> list[dict[str, Any]]:
    response = _request("GET", "/classes", teacher_id=teacher_id)
    payload = response.json()
    if not isinstance(payload, list):
        raise ClassApiError("Unexpected response format from classes endpoint.")
    return payload


def get_students_by_class(class_id: int, teacher_id: str = DEFAULT_TEACHER_ID) -> list[dict[str, Any]]:
    response = _request("GET", f"/classes/{class_id}/students", teacher_id=teacher_id)
    payload = response.json()
    if not isinstance(payload, list):
        raise ClassApiError("Unexpected response format from students endpoint.")
    return payload


def create_class(
    class_name: str,
    grade_age_group: str,
    schedule_days: list[str],
    description: str = "",
    teacher_id: str = DEFAULT_TEACHER_ID,
) -> dict[str, Any]:
    payload = {
        "class_name": class_name,
        "grade_age_group": grade_age_group,
        "schedule_days": schedule_days,
        "description": description or None,
    }
    response = _request("POST", "/classes", teacher_id=teacher_id, json=payload)
    data = response.json()
    if not isinstance(data, dict):
        raise ClassApiError("Unexpected response format from create class endpoint.")
    return data


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def build_classes_dashboard(teacher_id: str = DEFAULT_TEACHER_ID) -> dict[str, Any]:
    classes = get_classes(teacher_id=teacher_id)

    total_students = 0
    latest_activity_at: datetime | None = None
    enriched_classes: list[dict[str, Any]] = []

    for classroom in classes:
        class_copy = dict(classroom)
        student_count = 0
        class_id = class_copy.get("id")

        if isinstance(class_id, int):
            try:
                student_count = len(get_students_by_class(class_id=class_id, teacher_id=teacher_id))
            except ClassApiError:
                student_count = 0

        class_copy["student_count"] = student_count
        total_students += student_count

        updated_at = _parse_timestamp(class_copy.get("updated_at"))
        created_at = _parse_timestamp(class_copy.get("created_at"))
        candidate = updated_at or created_at
        if candidate and (latest_activity_at is None or candidate > latest_activity_at):
            latest_activity_at = candidate

        enriched_classes.append(class_copy)

    enriched_classes.sort(
        key=lambda item: (_parse_timestamp(item.get("updated_at")) or _parse_timestamp(item.get("created_at")) or datetime.min),
        reverse=True,
    )

    return {
        "classes": enriched_classes,
        "total_classes": len(enriched_classes),
        "total_students": total_students,
        "last_analysis_at": latest_activity_at,
    }