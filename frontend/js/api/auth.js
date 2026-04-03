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

