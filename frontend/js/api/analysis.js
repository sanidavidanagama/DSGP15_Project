import { apiFetch } from './client.js';

const ACTIVE_ANALYSIS_JOB_KEY = 'inkind_active_analysis_job';
const LAST_ANALYSIS_RESULT_KEY = 'inkind_last_analysis_result';
const ANALYSIS_REPORT_MODE_KEY = 'inkind_analysis_report_mode';

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

export function setAnalysisReportMode(mode) {
	window.sessionStorage.setItem(ANALYSIS_REPORT_MODE_KEY, String(mode || 'new'));
}

export function getAnalysisReportMode() {
	try {
		return window.sessionStorage.getItem(ANALYSIS_REPORT_MODE_KEY) || 'new';
	} catch (error) {
		console.error('Failed to read analysis report mode', error);
		return 'new';
	}
}

export function clearAnalysisReportMode() {
	window.sessionStorage.removeItem(ANALYSIS_REPORT_MODE_KEY);
}
