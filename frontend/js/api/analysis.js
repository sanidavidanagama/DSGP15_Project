import { apiFetch } from './client.js';

const ACTIVE_ANALYSIS_JOB_KEY = 'inkind_active_analysis_job';
const LAST_ANALYSIS_RESULT_KEY = 'inkind_last_analysis_result';

export async function validateAnalysisImage(imageFile) {
	const formData = new FormData();
	formData.append('image', imageFile);

	return apiFetch('/validate_image', {
		method: 'POST',
		body: formData,
	});
}

export async function uploadAnalysisJob({ imageFile, description }) {
	const formData = new FormData();
	formData.append('image', imageFile);
	formData.append('description', description);

	return apiFetch('/upload', {
		method: 'POST',
		body: formData,
	});
}

export async function fetchAnalysisJobStatus(jobId) {
	return apiFetch(`/job_status/${jobId}`, {
		method: 'GET',
	});
}

export function setActiveAnalysisJob(job) {
	window.sessionStorage.setItem(ACTIVE_ANALYSIS_JOB_KEY, JSON.stringify(job));
}

export function getActiveAnalysisJob() {
	try {
		const raw = window.sessionStorage.getItem(ACTIVE_ANALYSIS_JOB_KEY);
		return raw ? JSON.parse(raw) : null;
	} catch (error) {
		console.error('Failed to parse active analysis job', error);
		return null;
	}
}

export function clearActiveAnalysisJob() {
	window.sessionStorage.removeItem(ACTIVE_ANALYSIS_JOB_KEY);
}

export function setLatestAnalysisResult(payload) {
	window.sessionStorage.setItem(LAST_ANALYSIS_RESULT_KEY, JSON.stringify(payload));
}

export function getLatestAnalysisResult() {
	try {
		const raw = window.sessionStorage.getItem(LAST_ANALYSIS_RESULT_KEY);
		return raw ? JSON.parse(raw) : null;
	} catch (error) {
		console.error('Failed to parse latest analysis result', error);
		return null;
	}
}
