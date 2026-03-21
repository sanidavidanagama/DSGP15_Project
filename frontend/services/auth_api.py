import os
from typing import Any

import requests


BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 30


class AuthApiError(Exception):
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


def login(username: str, password: str) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{BACKEND_BASE_URL}/auth/token",
            data={"username": username, "password": password},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise AuthApiError(f"Failed to connect to backend: {exc}") from exc

    if response.status_code >= 400:
        raise AuthApiError(_parse_error_message(response))

    payload = response.json()
    if not isinstance(payload, dict) or "access_token" not in payload:
        raise AuthApiError("Invalid login response")

    return payload


def get_current_profile(token: str) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{BACKEND_BASE_URL}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise AuthApiError(f"Failed to connect to backend: {exc}") from exc

    if response.status_code >= 400:
        raise AuthApiError(_parse_error_message(response))

    payload = response.json()
    if not isinstance(payload, dict):
        raise AuthApiError("Invalid profile response")

    return payload
