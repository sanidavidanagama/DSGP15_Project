import { apiFetch } from './client.js';

export async function fetchMySettingsProfile() {
	return apiFetch('/settings/me', { method: 'GET' });
}

export async function updateMyProfile(payload) {
	return apiFetch('/settings/profile', {
		method: 'PATCH',
		body: JSON.stringify(payload),
	});
}

export async function changeMyPassword(payload) {
	return apiFetch('/settings/password', {
		method: 'PATCH',
		body: JSON.stringify(payload),
	});
}

export async function deleteMyData(payload) {
	return apiFetch('/settings/data', {
		method: 'DELETE',
		body: JSON.stringify(payload),
	});
}

export async function deleteMyAccount(payload) {
	return apiFetch('/settings/account', {
		method: 'DELETE',
		body: JSON.stringify(payload),
	});
}
