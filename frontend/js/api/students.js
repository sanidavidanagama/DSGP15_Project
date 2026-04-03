import { apiFetch } from './client.js';

export async function fetchStudents() {
	return apiFetch('/students', { method: 'GET' });
}

export async function fetchStudentDetail(classId, studentId) {
	return apiFetch(`/classes/${classId}/students/${studentId}`, { method: 'GET' });
}

export async function fetchStudentsByClass(classId) {
	return apiFetch(`/classes/${classId}/students`, { method: 'GET' });
}

export async function fetchSavedReport(classId, studentId, jobId) {
	return apiFetch(`/classes/${classId}/students/${studentId}/report/${jobId}`, { method: 'GET' });
}

export async function saveAnalysisToStudent(studentId, jobId) {
	return apiFetch(`/students/${studentId}/saved-analyses`, {
		method: 'POST',
		body: JSON.stringify({ job_id: jobId }),
	});
}

export async function updateStudent(studentId, payload) {
	return apiFetch(`/students/${studentId}`, {
		method: 'PATCH',
		body: JSON.stringify(payload),
	});
}

export async function deleteStudent(studentId) {
	return apiFetch(`/students/${studentId}`, {
		method: 'DELETE',
	});
}
