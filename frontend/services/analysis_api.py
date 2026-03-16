import os
import time
from typing import Any

import requests


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 300


class AnalysisApiError(Exception):
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


def validate_image(image_name: str, image_bytes: bytes, content_type: str | None) -> dict[str, Any]:
    files = {
        "image": (image_name, image_bytes, content_type or "application/octet-stream")
    }

    try:
        response = requests.post(
            f"{BACKEND_BASE_URL}/validate_image",
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise AnalysisApiError(f"Failed to connect to backend: {exc}") from exc

    if response.status_code >= 400:
        raise AnalysisApiError(_parse_error_message(response))

    return response.json()


def upload_job(image_name: str, image_bytes: bytes, content_type: str | None, description: str) -> str:
    files = {
        "image": (image_name, image_bytes, content_type or "application/octet-stream")
    }
    data = {"description": description}

    try:
        response = requests.post(
            f"{BACKEND_BASE_URL}/upload",
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise AnalysisApiError(f"Failed to connect to backend: {exc}") from exc

    if response.status_code >= 400:
        raise AnalysisApiError(_parse_error_message(response))

    payload = response.json()
    job_id = payload.get("job_id")
    if not job_id:
        raise AnalysisApiError("Upload succeeded but no job_id was returned by backend.")

    return str(job_id)


def poll_job_status(job_id: str, timeout_seconds: int = POLL_TIMEOUT_SECONDS) -> dict[str, Any]:
    start = time.time()

    while time.time() - start < timeout_seconds:
        try:
            response = requests.get(
                f"{BACKEND_BASE_URL}/job_status/{job_id}",
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raise AnalysisApiError(f"Failed while polling job status: {exc}") from exc

        if response.status_code >= 400:
            raise AnalysisApiError(_parse_error_message(response))

        payload = response.json()
        status = payload.get("status")

        if status in {"done", "failed"}:
            return payload

        time.sleep(POLL_INTERVAL_SECONDS)

    raise AnalysisApiError(
        f"Job is still running after {timeout_seconds} seconds. Please try again."
    )
