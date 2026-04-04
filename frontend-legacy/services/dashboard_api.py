import os
from typing import Any

import requests


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30


class DashboardApiError(Exception):
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


def get_dashboard_overview(token: str) -> dict[str, Any]:
    if not token:
        raise DashboardApiError("Missing authentication token.")

    try:
        response = requests.get(
            f"{BACKEND_BASE_URL}/dashboard/overview",
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise DashboardApiError(f"Failed to connect to backend: {exc}") from exc

    if response.status_code >= 400:
        raise DashboardApiError(_parse_error_message(response))

    payload = response.json()
    if not isinstance(payload, dict):
        raise DashboardApiError("Unexpected response format from dashboard endpoint.")

    return payload
