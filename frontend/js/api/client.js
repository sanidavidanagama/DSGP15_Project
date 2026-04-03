// Base API client for INKIND frontend
// Handles base URL, JSON parsing, and auth header injection.

import { getAuth } from '../utils/state.js';

const BASE_URL = 'http://localhost:8000';

export async function apiFetch(path, options = {}) {
	const url = `${BASE_URL}${path}`;
	const { method = 'GET', headers = {}, body } = options;

	const auth = getAuth();
	const finalHeaders = { ...headers };
	if (auth && auth.token) {
		finalHeaders.Authorization = `Bearer ${auth.token}`;
	}

	// Set Content-Type to application/json when sending JSON body
	if (body && typeof body === 'string' && !finalHeaders['Content-Type']) {
		finalHeaders['Content-Type'] = 'application/json';
	}

	const response = await fetch(url, {
		method,
		headers: finalHeaders,
		body,
		credentials: 'omit',
	});

	if (!response.ok) {
		let errorMessage = `Request failed with status ${response.status}`;
		try {
			const errorData = await response.json();
			if (errorData && (errorData.detail || errorData.message)) {
				errorMessage = errorData.detail || errorData.message;
			}
		} catch (err) {
			// Ignore JSON parse errors, fall back to generic message
		}
		const error = new Error(errorMessage);
		error.status = response.status;
		throw error;
	}

	// Handle empty responses (204, etc.)
	if (response.status === 204) {
		return null;
	}

	return response.json();
}

