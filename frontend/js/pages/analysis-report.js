import { createShell, attachShellHandlers } from '../components/shell.js';
import { getLatestAnalysisResult } from '../api/analysis.js';
import { showToast } from '../components/toast.js';

const API_BASE_URL = 'http://localhost:8000';

export async function renderPage() {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	const resultPayload = getLatestAnalysisResult();
	if (!resultPayload?.result) {
		showToast('error', 'No completed analysis found.');
		window.location.hash = '#/analysis';
		return;
	}

	const result = resultPayload.result || {};
	const imageMetadata = result.image || {};
	const emotion = result.emotion || {};
	const dia = result.dia || {};
	const recommendation = result.recommendation || {};
	const patterns = recommendation.DetectedPatterns || {};

	const happyScore = toPercent(emotion.happy_score);
	const sadScore = Math.max(0, 100 - happyScore);
	const predictedMood = capitalize(emotion.predicted_mood || emotion.emotion || 'Unknown');

	const pageContent = `
		<div class="dashboard-container">
			<section class="card">
				<h2>Analysis Report</h2>
				<p class="text-muted" style="margin-top: 8px;">Job ID: ${safeText(resultPayload.job_id)}}</p>
				<div class="report-chips">
					<span class="badge badge-teal">Status: ${safeText(resultPayload.status || 'done')}</span>
                    <span class="badge badge-gray">Processed: ${formatDateShort(resultPayload.analysis_finished_at)}</span>
				</div>
			</section>

			<section class="report-images-grid">
				<article class="report-image-card">
					<div class="report-image-label">Original Drawing</div>
					${renderImageMarkup(resultPayload.raw_image_path, 'Original drawing')}
				</article>

				<article class="report-image-card">
					<div class="report-image-label">Processed Drawing</div>
					${renderImageMarkup(imageMetadata.processed_image_path, 'Processed drawing')}
				</article>
			</section>

			<section class="report-metrics-row">
				<article class="glass-card report-metric-card">
					<div class="report-metric-icon" style="background: rgba(0,128,128,0.12); color: #008080;">😊</div>
					<div class="report-metric-value">${safeText(predictedMood)}</div>
					<div class="report-metric-label">Predicted Mood</div>
				</article>

				<article class="glass-card report-metric-card">
					<div class="report-metric-icon" style="background: rgba(251,191,36,0.2); color: #d97706;">%</div>
					<div class="report-metric-value">${happyScore}%</div>
					<div class="report-metric-label">Happy Score</div>
				</article>

				<article class="glass-card report-metric-card">
					<div class="report-metric-icon" style="background: rgba(147,51,234,0.15); color: #7e22ce;">⏱</div>
					<div class="report-metric-value">${formatDuration(resultPayload.analysis_duration_seconds)}</div>
					<div class="report-metric-label">Duration</div>
				</article>
			</section>

			<section class="card">
				<h3>Mood Distribution</h3>
				<div class="mood-bar-container">
					<div class="mood-bar-track">
						<div class="mood-bar-happy" style="width: ${happyScore}%;"></div>
						<div class="mood-bar-sad" style="width: ${sadScore}%;"></div>
					</div>
					<div class="mood-bar-labels">
						<span class="mood-bar-label happy">Happy ${happyScore}%</span>
						<span class="mood-bar-label sad">Sad ${sadScore}%</span>
					</div>
				</div>
			</section>

			<section class="card">
				<h3>Drawing Indicators</h3>
				<div class="indicators-grid" style="margin-top: 16px;">
					${renderIndicator('Line Pressure', dia.line_pressure)}
					${renderIndicator('Shading Intensity', dia.shading_intensity)}
					${renderIndicator('Overall Tone', dia.overall_tone)}
					${renderIndicator('Page Usage', dia.page_usage)}
					${renderIndicator('Figure Size', dia.figure_size)}
					${renderIndicator('Placement', dia.placement)}
					${renderIndicator('Figures', dia.number_of_figures)}
					${renderIndicator('Facial Features', dia.facial_features)}
				</div>
			</section>

			<section class="card">
				<h3>Interpretation</h3>
				<div class="interpretation-list" style="margin-top: 16px;">
					${renderInterpretationList(dia.interpretation)}
				</div>
			</section>

			<section class="card recommendation-card">
				<h3>Recommendation</h3>
				<div class="patterns-grid" style="margin-top: 16px;">
					<div class="pattern-item">
						<div class="pattern-label">Emotional Pattern</div>
						<div class="pattern-value">${safeText(patterns.emotional || 'N/A')}</div>
					</div>
					<div class="pattern-item">
						<div class="pattern-label">Spatial Pattern</div>
						<div class="pattern-value">${safeText(patterns.spatial || 'N/A')}</div>
					</div>
				</div>

				<div class="recommendation-text-box">
					${safeText(recommendation.RecommendationText || 'No recommendation generated.')}
				</div>
			</section>

			<section class="card">
				<h3>Next Steps</h3>
				<div style="margin-top: 16px; display: flex; gap: 12px; flex-wrap: wrap;">
					<a href="#/analysis" class="btn btn-secondary">
						<i data-lucide="plus"></i>
						Run New Analysis
					</a>
					<a href="#/dashboard" class="btn btn-primary">Back to Dashboard</a>
				</div>
			</section>

			<section class="card">
				<h3>Save Analysis to Student</h3>
				<p class="text-muted" style="margin-top: 8px; margin-bottom: 16px;">
					Attach this analysis to a student profile to build their longitudinal record.
				</p>
				<div style="display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap;">
					<div style="flex: 1; min-width: 250px;">
						<label class="form-label" for="student-select">Select Student</label>
						<select id="student-select" class="form-input">
							<option value="">Loading students...</option>
						</select>
					</div>
					<button id="save-to-student-btn" class="btn btn-primary">
						<i data-lucide="save"></i>
						Save
					</button>
				</div>
				<p id="save-status-msg" class="text-muted" style="margin-top: 8px; display: none;"></p>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Analysis Report' });
	attachShellHandlers();

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function renderIndicator(label, value) {
	return `
		<div class="indicator-tile">
			<div class="indicator-label">${safeText(label)}</div>
			<div class="indicator-value">${safeText(value || 'N/A')}</div>
		</div>
	`;
}

function renderInterpretationList(items) {
	if (!Array.isArray(items) || items.length === 0) {
		return '<div class="interpretation-item">No interpretation notes available.</div>';
	}

	return items
		.map((item) => `<div class="interpretation-item">${safeText(item)}</div>`)
		.join('');
}

function renderImageMarkup(imagePath, altText) {
	const imageUrl = resolveImageUrl(imagePath);
	if (!imageUrl) {
		return `<div class="report-image-placeholder"><i data-lucide="image"></i><span>No image available</span></div>`;
	}

	return `<img src="${imageUrl}" alt="${safeText(altText)}" />`;
}

function resolveImageUrl(imagePath) {
	if (!imagePath || typeof imagePath !== 'string') return null;
	if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) {
		return encodeURI(imagePath);
	}

	const normalizedPath = imagePath.replaceAll('\\', '/');
	const uploadsIndex = normalizedPath.toLowerCase().lastIndexOf('/uploads/');
	if (uploadsIndex !== -1) {
		return encodeURI(`${API_BASE_URL}${normalizedPath.slice(uploadsIndex)}`);
	}

	const uploadsRelativeIndex = normalizedPath.toLowerCase().indexOf('uploads/');
	if (uploadsRelativeIndex !== -1) {
		return encodeURI(`${API_BASE_URL}/${normalizedPath.slice(uploadsRelativeIndex)}`);
	}

	return encodeURI(`${API_BASE_URL}/uploads/${normalizedPath.split('/').pop()}`);
}

function capitalize(text) {
	if (!text || typeof text !== 'string') return 'Unknown';
	return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
}

function formatDateShort(isoString) {
	if (!isoString) return 'N/A';
	const date = new Date(isoString);
	if (Number.isNaN(date.getTime())) return 'N/A';
	const day = String(date.getDate()).padStart(2, '0');
	const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
	const month = months[date.getMonth()];
	const year = String(date.getFullYear()).slice(-2);
	return `${day}-${month}-${year}`;
}

function toPercent(value) {
	const numeric = Number(value);
	if (Number.isNaN(numeric)) return 50;
	if (numeric >= 0 && numeric <= 1) return Math.round(numeric * 100);
	return Math.max(0, Math.min(100, Math.round(numeric)));
}

function formatDuration(seconds) {
	if (typeof seconds !== 'number' || Number.isNaN(seconds)) return 'N/A';
	if (seconds < 60) return `${Math.round(seconds)}s`;
	return `${(seconds / 60).toFixed(1)}m`;
}

function formatDate(value) {
	if (!value) return 'N/A';
	const date = new Date(value);
	if (Number.isNaN(date.getTime())) return 'N/A';
	return date.toLocaleString();
}

function safeText(value) {
	const text = value == null ? '' : String(value);
	return text
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;')
		.replaceAll("'", '&#39;');
}
