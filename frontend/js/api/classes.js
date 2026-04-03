import { apiFetch } from './client.js';

export async function fetchClasses() {
	return apiFetch('/classes', { method: 'GET' });
}

export async function fetchClassById(classId) {
	return apiFetch(`/classes/${classId}`, { method: 'GET' });
}

export async function createClass(payload) {
	return apiFetch('/classes', {
		method: 'POST',
		body: JSON.stringify(payload),
	});
}

export async function updateClass(classId, payload) {
	return apiFetch(`/classes/${classId}`, {
		method: 'PATCH',
		body: JSON.stringify(payload),
	});
}

export async function deleteClass(classId) {
	return apiFetch(`/classes/${classId}`, {
		method: 'DELETE',
	});
}
