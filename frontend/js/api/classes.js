import { apiFetch } from './client.js';

export async function fetchClasses() {
	return apiFetch('/classes', { method: 'GET' });
}

export async function fetchClassById(classId) {
	return apiFetch(`/classes/${classId}`, { method: 'GET' });
}
