import { createShell, attachShellHandlers } from '../components/shell.js';
import { showToast } from '../components/toast.js';
import {
	validateAnalysisImage,
	uploadAnalysisJob,
	setActiveAnalysisJob,
} from '../api/analysis.js';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
const MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024;

let selectedImageFile = null;

export async function renderPage() {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	const pageContent = `
		<div class="dashboard-container">
			<section class="card">
				<h2>Upload Analysis</h2>
				<p class="text-muted" style="margin-top: 8px;">
					Upload a child drawing and optional teacher notes to generate a full emotional and drawing-pattern analysis.
				</p>
			</section>

			<div class="analysis-layout">
				<section class="card">
					<input id="analysis-image-input" type="file" accept="image/*" hidden />

					<div id="upload-zone" class="upload-zone" role="button" tabindex="0" aria-label="Upload drawing image">
						<i data-lucide="upload-cloud" class="upload-zone-icon"></i>
						<div class="upload-zone-title">Drop drawing image here or click to browse</div>
						<div class="upload-zone-hint">Accepted: JPG, PNG, WEBP, BMP (max 20MB)</div>
					</div>

					<div id="image-preview-card" class="image-preview-card card" style="padding: 16px; margin-bottom: 16px;">
						<img id="image-preview-thumb" class="image-preview-thumb" alt="Selected drawing preview" />
						<div class="image-preview-info">
							<span id="image-preview-name" class="image-preview-name"></span>
							<span id="image-preview-size" class="image-preview-size"></span>
						</div>
						<button id="remove-image-btn" type="button" class="btn btn-ghost" style="align-self: flex-start; padding-left: 0;">
							Remove image
						</button>
					</div>

					<div class="form-group">
						<label class="form-label" for="analysis-description">Teacher Context</label>
						<textarea
							id="analysis-description"
							class="form-textarea"
							placeholder="Add context: Student's explaination about the drawing..."
						></textarea>
					</div>

					<button id="submit-analysis-btn" type="button" class="btn btn-primary">
						<i data-lucide="sparkles"></i>
						Start Analysis
					</button>
				</section>

				<aside class="card steps-card">
					<h3 style="margin-bottom: 12px;">What happens next?</h3>
					<ol>
						<li class="step-list-item">Upload a drawing image from your device.</li>
						<li class="step-list-item">Optionally add teacher notes for context.</li>
						<li class="step-list-item">The system processes image, mood, and drawing indicators.</li>
						<li class="step-list-item">Review the final report with recommendations.</li>
					</ol>
				</aside>
			</div>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Analysis Upload' });
	attachShellHandlers();
	attachUploadHandlers();

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function attachUploadHandlers() {
	const fileInput = document.getElementById('analysis-image-input');
	const uploadZone = document.getElementById('upload-zone');
	const removeButton = document.getElementById('remove-image-btn');
	const submitButton = document.getElementById('submit-analysis-btn');

	if (!fileInput || !uploadZone || !removeButton || !submitButton) return;

	uploadZone.addEventListener('click', () => fileInput.click());
	uploadZone.addEventListener('keydown', (event) => {
		if (event.key === 'Enter' || event.key === ' ') {
			event.preventDefault();
			fileInput.click();
		}
	});

	uploadZone.addEventListener('dragover', (event) => {
		event.preventDefault();
		uploadZone.classList.add('drag-over');
	});

	uploadZone.addEventListener('dragleave', () => {
		uploadZone.classList.remove('drag-over');
	});

	uploadZone.addEventListener('drop', async (event) => {
		event.preventDefault();
		uploadZone.classList.remove('drag-over');
		const file = event.dataTransfer?.files?.[0];
		if (file) {
			await processSelectedFile(file);
		}
	});

	fileInput.addEventListener('change', async (event) => {
		const file = event.target.files?.[0];
		if (file) {
			await processSelectedFile(file);
		}
	});

	removeButton.addEventListener('click', () => {
		selectedImageFile = null;
		fileInput.value = '';
		renderImagePreview(null);
	});

	submitButton.addEventListener('click', handleSubmitAnalysis);
}

async function processSelectedFile(file) {
	if (!ACCEPTED_TYPES.includes(file.type)) {
		showToast('error', 'Unsupported file type. Please upload JPG, PNG, WEBP, or BMP.');
		return;
	}

	if (file.size > MAX_FILE_SIZE_BYTES) {
		showToast('error', 'Image is too large. Maximum allowed size is 20MB.');
		return;
	}

	try {
		await validateAnalysisImage(file);
		selectedImageFile = file;
		renderImagePreview(file);
		showToast('success', 'Image validated successfully.');
	} catch (error) {
		showToast('error', error.message || 'Image validation failed.');
	}
}

function renderImagePreview(file) {
	const previewCard = document.getElementById('image-preview-card');
	const previewThumb = document.getElementById('image-preview-thumb');
	const previewName = document.getElementById('image-preview-name');
	const previewSize = document.getElementById('image-preview-size');

	if (!previewCard || !previewThumb || !previewName || !previewSize) return;

	if (!file) {
		previewCard.classList.remove('visible');
		previewThumb.removeAttribute('src');
		previewName.textContent = '';
		previewSize.textContent = '';
		return;
	}

	previewCard.classList.add('visible');
	previewThumb.src = URL.createObjectURL(file);
	previewName.textContent = file.name;
	previewSize.textContent = formatBytes(file.size);
}

async function handleSubmitAnalysis() {
	if (!selectedImageFile) {
		showToast('error', 'Please select an image before starting analysis.');
		return;
	}

	const submitButton = document.getElementById('submit-analysis-btn');
	const descriptionInput = document.getElementById('analysis-description');
	if (!submitButton || !descriptionInput) return;

	const description = descriptionInput.value.trim();
	submitButton.disabled = true;
	submitButton.classList.add('btn-submit-loading');
	submitButton.textContent = 'Starting analysis...';

	try {
		const uploadResponse = await uploadAnalysisJob({
			imageFile: selectedImageFile,
			description,
		});

		setActiveAnalysisJob({
			jobId: uploadResponse.job_id,
			description,
			imageName: selectedImageFile.name,
			submittedAt: new Date().toISOString(),
		});

		showToast('success', 'Analysis started. Tracking progress...');
		window.location.hash = '#/analysis/loading';
	} catch (error) {
		submitButton.disabled = false;
		submitButton.classList.remove('btn-submit-loading');
		submitButton.innerHTML = '<i data-lucide="sparkles"></i>Start Analysis';
		if (window.lucide && typeof window.lucide.createIcons === 'function') {
			window.lucide.createIcons();
		}
		showToast('error', error.message || 'Failed to start analysis.');
	}
}

function formatBytes(bytes) {
	if (!bytes) return '0 B';
	const units = ['B', 'KB', 'MB', 'GB'];
	const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
	const value = bytes / Math.pow(1024, index);
	return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}
