import os
from datetime import datetime, timezone
from typing import Any

import requests


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_TEACHER_ID = os.getenv("TEACHER_ID", "dev-teacher")


class StudentApiError(Exception):
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


def _request(method: str, path: str, teacher_id: str | None = None, token: str | None = None, **kwargs: Any) -> requests.Response:
    headers = kwargs.pop("headers", {})
    if token:
        headers = {**headers, "Authorization": f"Bearer {token}"}
    elif teacher_id:
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
        raise StudentApiError(f"Failed to connect to backend: {exc}") from exc

    if response.status_code >= 400:
        raise StudentApiError(_parse_error_message(response))

    return response


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
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


def _relative_time_label(value: datetime | None) -> str:
    if value is None:
        return "No predictions yet"

    now = datetime.now(timezone.utc)
    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    delta = now - dt
    seconds = max(0, int(delta.total_seconds()))

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def list_students(class_id: int, teacher_id: str = DEFAULT_TEACHER_ID, token: str | None = None) -> list[dict[str, Any]]:
    response = _request("GET", f"/classes/{class_id}/students", teacher_id=teacher_id, token=token)
    payload = response.json()
    if not isinstance(payload, list):
        raise StudentApiError("Unexpected response format from students endpoint.")

    enriched: list[dict[str, Any]] = []
    for raw in payload:
        item = dict(raw)
        item.setdefault("last_predicted_mood", None)
        item.setdefault("last_predicted_at", None)
        item.setdefault("total_analyses", 0)

        student_id = item.get("id")
        if isinstance(student_id, int):
            try:
                history = list_saved_analyses(student_id=student_id, teacher_id=teacher_id, token=token)
            except StudentApiError:
                history = []

            if history:
                item["total_analyses"] = len(history)
                item["last_predicted_mood"] = history[0].get("mood")
                item["last_predicted_at"] = history[0].get("saved_at")

        last_predicted_at = _parse_timestamp(item.get("last_predicted_at") or item.get("joined_at"))
        item["last_predicted_at"] = last_predicted_at.isoformat() if last_predicted_at else None
        item["last_predicted_label"] = _relative_time_label(last_predicted_at)
        enriched.append(item)

    return enriched


def get_student(student_id: int, teacher_id: str = DEFAULT_TEACHER_ID, token: str | None = None) -> dict[str, Any]:
    response = _request("GET", f"/students/{student_id}", teacher_id=teacher_id, token=token)
    payload = response.json()
    if not isinstance(payload, dict):
        raise StudentApiError("Unexpected response format from student endpoint.")
    return dict(payload)


def get_student_in_class(class_id: int, student_id: int, teacher_id: str = DEFAULT_TEACHER_ID, token: str | None = None) -> dict[str, Any]:
    response = _request("GET", f"/classes/{class_id}/students/{student_id}", teacher_id=teacher_id, token=token)
    payload = response.json()
    if not isinstance(payload, dict):
        raise StudentApiError("Unexpected response format from student detail endpoint.")
    return dict(payload)


def create_student(
    class_id: int,
    name: str,
    age_group: str,
    teacher_id: str = DEFAULT_TEACHER_ID,
    token: str | None = None,
) -> dict[str, Any]:
    payload = {"name": name, "age_group": age_group}
    response = _request("POST", f"/classes/{class_id}/students", teacher_id=teacher_id, token=token, json=payload)
    data = response.json()
    if not isinstance(data, dict):
        raise StudentApiError("Unexpected response format from create student endpoint.")
    return data


def update_student(
    student_id: int,
    name: str,
    age_group: str,
    teacher_id: str = DEFAULT_TEACHER_ID,
    token: str | None = None,
) -> dict[str, Any]:
    payload = {"name": name, "age_group": age_group}
    response = _request("PATCH", f"/students/{student_id}", teacher_id=teacher_id, token=token, json=payload)
    data = response.json()
    if not isinstance(data, dict):
        raise StudentApiError("Unexpected response format from update student endpoint.")
    return data


def delete_student(student_id: int, teacher_id: str = DEFAULT_TEACHER_ID, token: str | None = None) -> None:
    _request("DELETE", f"/students/{student_id}", teacher_id=teacher_id, token=token)


def save_analysis_to_student(student_id: int, job_id: str, teacher_id: str = DEFAULT_TEACHER_ID, token: str | None = None) -> dict[str, Any]:
    response = _request(
        "POST",
        f"/students/{student_id}/saved-analyses",
        teacher_id=teacher_id,
        token=token,
        json={"job_id": job_id},
    )
    data = response.json()
    if not isinstance(data, dict):
        raise StudentApiError("Unexpected response format from save analysis endpoint.")
    return data


def list_saved_analyses(student_id: int, teacher_id: str = DEFAULT_TEACHER_ID, token: str | None = None) -> list[dict[str, Any]]:
    response = _request("GET", f"/students/{student_id}/saved-analyses", teacher_id=teacher_id, token=token)
    payload = response.json()
    if not isinstance(payload, list):
        raise StudentApiError("Unexpected response format from saved analyses endpoint.")
    return [dict(item) for item in payload]
