import { createShell, attachShellHandlers } from '../components/shell.js';
import { showToast } from '../components/toast.js';
import {
	getActiveAnalysisJob,
	fetchAnalysisJobStatus,
	setLatestAnalysisResult,
	clearActiveAnalysisJob,
} from '../api/analysis.js';

const POLL_INTERVAL_MS = 2000;

export async function renderPage() {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	const activeJob = getActiveAnalysisJob();
	if (!activeJob?.jobId) {
		showToast('error', 'No active analysis job found.');
		window.location.hash = '#/analysis';
		return;
	}

	const pageContent = `
		<div class="analysis-loading-page">
			<div class="progress-card">
				<h2 style="margin-bottom: 6px;">Analyzing Drawing</h2>
				<p class="text-muted" style="margin-bottom: 24px;">Job ID: ${activeJob.jobId}</p>

				<div class="progress-bar-wrapper">
					<div class="progress-bar-header">
						<span class="progress-bar-label">Overall Progress</span>
						<span id="progress-pct" class="progress-bar-pct">5%</span>
					</div>
					<div class="progress-bar-track">
						<div id="progress-fill" class="progress-bar-fill" style="width: 5%;"></div>
					</div>
				</div>

				<div class="timeline-steps">
					<div class="timeline-step">
						<div id="step-upload-indicator" class="step-indicator active"><div class="step-dot"></div></div>
						<div class="step-content">
							<div id="step-upload-title" class="step-title-text active">Upload received</div>
							<div class="step-caption">Image accepted and queued</div>
						</div>
					</div>

					<div class="timeline-step">
						<div id="step-image-indicator" class="step-indicator pending"></div>
						<div class="step-content">
							<div id="step-image-title" class="step-title-text pending">Image processing</div>
							<div class="step-caption">Pre-processing drawing and extracting features</div>
						</div>
					</div>

					<div class="timeline-step">
						<div id="step-ai-indicator" class="step-indicator pending"></div>
						<div class="step-content">
							<div id="step-ai-title" class="step-title-text pending">AI analysis</div>
							<div class="step-caption">Mood + drawing insights in progress</div>
						</div>
					</div>

					<div class="timeline-step">
						<div id="step-reco-indicator" class="step-indicator pending"></div>
						<div class="step-content">
							<div id="step-reco-title" class="step-title-text pending">Recommendations</div>
							<div class="step-caption">Preparing final report package</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Analysis In Progress' });
	attachShellHandlers();
	await pollJobUntilComplete(activeJob.jobId);

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

async function pollJobUntilComplete(jobId) {
	const maxAttempts = 180;

	for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
		if (window.location.hash !== '#/analysis/loading') return;

		try {
			const statusPayload = await fetchAnalysisJobStatus(jobId);
			const status = String(statusPayload.status || '').toLowerCase();

			updateProgressByStatus(status, attempt);

			if (status === 'done') {
				setLatestAnalysisResult(statusPayload);
				clearActiveAnalysisJob();
				showToast('success', 'Analysis completed successfully.');
				window.location.hash = '#/analysis/report';
				return;
			}

			if (status === 'failed' || status === 'error') {
				showToast('error', 'Analysis failed. Please try again.');
				window.location.hash = '#/analysis';
				return;
			}
		} catch (error) {
			if (attempt % 3 === 0) {
				showToast('error', error.message || 'Unable to fetch analysis status.');
			}
		}

		await delay(POLL_INTERVAL_MS);
	}

	showToast('error', 'Analysis is taking longer than expected. Please check again.');
	window.location.hash = '#/analysis';
}

function updateProgressByStatus(status, attempt) {
	let progress = Math.min(90, 5 + attempt * 2);

	if (status === 'processing') {
		progress = Math.max(progress, 20);
		setStepState('upload', 'done');
		setStepState('image', 'active');
	}

	if (status === 'image_processed') {
		progress = Math.max(progress, 45);
		setStepState('upload', 'done');
		setStepState('image', 'done');
		setStepState('ai', 'active');
	}

	if (status === 'mood_predicted' || status === 'drawing_insights_ready' || status === 'analysis_ready') {
		progress = Math.max(progress, 72);
		setStepState('upload', 'done');
		setStepState('image', 'done');
		setStepState('ai', 'done');
		setStepState('reco', 'active');
	}

	if (status === 'recommendation_ready') {
		progress = Math.max(progress, 88);
		setStepState('upload', 'done');
		setStepState('image', 'done');
		setStepState('ai', 'done');
		setStepState('reco', 'active');
	}

	if (status === 'done') {
		progress = 100;
		setStepState('upload', 'done');
		setStepState('image', 'done');
		setStepState('ai', 'done');
		setStepState('reco', 'done');
	}

	const fill = document.getElementById('progress-fill');
	const pct = document.getElementById('progress-pct');
	if (fill) fill.style.width = `${progress}%`;
	if (pct) pct.textContent = `${progress}%`;
}

function setStepState(stepKey, state) {
	const indicator = document.getElementById(`step-${stepKey}-indicator`);
	const title = document.getElementById(`step-${stepKey}-title`);
	if (!indicator || !title) return;

	indicator.classList.remove('pending', 'active', 'done');
	indicator.classList.add(state);
	if (state === 'done') {
		indicator.innerHTML = '<i data-lucide="check" style="width:14px;height:14px;"></i>';
	} else if (state === 'active') {
		indicator.innerHTML = '<div class="step-dot"></div>';
	} else {
		indicator.innerHTML = '';
	}

	title.classList.remove('pending', 'active', 'done');
	title.classList.add(state);

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function delay(ms) {
	return new Promise((resolve) => window.setTimeout(resolve, ms));
}
