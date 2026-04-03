// Auth API helpers: register, login, getProfile

import { apiFetch } from './client.js';

export async function registerUser({ username, email, password }) {
	return apiFetch('/auth/register', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({ username, email, password }),
	});
}

export async function loginUser({ username, password }) {
	// OAuth2PasswordRequestForm expects form-data, not JSON
	const formData = new FormData();
	formData.append('username', username);
	formData.append('password', password);

	return apiFetch('/auth/token', {
		method: 'POST',
		body: formData,
		// Intentionally no Content-Type header; browser will set it with boundary
	});
}

export async function getProfile() {
	return apiFetch('/auth/me', {
		method: 'GET',
	});
}

