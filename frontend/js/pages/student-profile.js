import { createShell, attachShellHandlers } from '../components/shell.js';
import { fetchClassById } from '../api/classes.js';
import {
	fetchStudentDetail,
	fetchSavedReport,
	updateStudent,
	deleteStudent,
} from '../api/students.js';
import {
	setLatestAnalysisResult,
	setAnalysisReportMode,
} from '../api/analysis.js';
import { openModal, closeModal } from '../components/modal.js';
import { showToast } from '../components/toast.js';

export async function renderPage({ params } = {}) {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	const classId = params?.classId;
	const studentId = params?.studentId;
	if (!classId || !studentId) {
		showToast('error', 'Student profile not found.');
		window.location.hash = '#/classes';
		return;
	}

	try {
		const [classroom, student] = await Promise.all([
			fetchClassById(classId),
			fetchStudentDetail(classId, studentId),
		]);
		renderStudentProfile(appElement, classroom, student);
	} catch (error) {
		showToast('error', error.message || 'Failed to load student profile.');
		renderErrorState(appElement, error, classId);
	}
}

function renderStudentProfile(appElement, classroom, student) {
	const history = Array.isArray(student.history) ? student.history : [];
	const chartPoints = buildChartPoints(history);

	const pageContent = `
		<div class="dashboard-container student-profile-page">
			<section class="student-hero card">
				<div class="student-hero-top">
					<div class="student-hero-identity">
						<div class="student-hero-back-wrap">
							<a href="#/classes/${classroom.id}" class="btn btn-ghost btn-sm">
								<i data-lucide="arrow-left"></i>
								Back
							</a>
						</div>
						<div class="student-avatar-circle student-hero-avatar">${safeText(getInitials(student.name))}</div>
						<div class="student-hero-info">
							<h2>${safeText(student.name || 'Student')}</h2>
							<p class="student-hero-meta">${safeText(student.gender || 'N/A')} · ${safeText(classroom.class_name || 'Class')}</p>
							<div class="student-hero-chips">
								${student.last_predicted_mood ? `<span class="badge ${getMoodBadgeClass(student.last_predicted_mood)}">${safeText(student.last_predicted_mood)}</span>` : '<span class="badge badge-gray">No mood yet</span>'}
								<span class="badge badge-gray">Last updated ${student.last_predicted_at ? formatDateShort(student.last_predicted_at) : 'N/A'}</span>
								<span class="badge badge-teal">${Number(student.total_analyses || 0)} analyses</span>
							</div>
						</div>
					</div>
					<div class="student-hero-actions">
						<button id="edit-student-btn" class="btn btn-secondary" type="button">
							<i data-lucide="pencil"></i>
							Edit Student
						</button>
						<button id="delete-student-btn" class="btn btn-danger-ghost" type="button">
							<i data-lucide="trash-2"></i>
							Delete Student
						</button>
					</div>
				</div>
			</section>

			<section class="classes-summary-grid student-summary-grid">
				<article class="glass-card stat-card">
					<div class="stat-card-icon-row"><div class="stat-card-icon"><i data-lucide="scan-eye"></i></div></div>
					<div class="stat-card-value">${Number(student.total_analyses || 0)}</div>
					<div class="stat-card-label">Total Analyses</div>
				</article>

				<article class="glass-card stat-card">
					<div class="stat-card-icon-row"><div class="stat-card-icon"><i data-lucide="smile"></i></div></div>
					<div class="stat-card-value">${safeText(student.last_predicted_mood || 'N/A')}</div>
					<div class="stat-card-label">Last Predicted Mood</div>
				</article>

				<article class="glass-card stat-card">
					<div class="stat-card-icon-row"><div class="stat-card-icon"><i data-lucide="clock-3"></i></div></div>
					<div class="stat-card-value">${student.last_predicted_at ? formatRelativeTime(student.last_predicted_at) : 'N/A'}</div>
					<div class="stat-card-label">Last Updated</div>
				</article>
			</section>

			<section class="student-content-columns">
				<div class="chart-card">
					<h3>Mood Trend</h3>
					${chartPoints.length >= 2 ? `<div class="chart-container"><canvas id="mood-chart-canvas"></canvas><div id="mood-chart-tooltip" class="chart-tooltip" hidden></div></div>` : `<div class="chart-no-data"><i data-lucide="activity"></i><span>Not enough data points for a trend chart yet.</span></div>`}
				</div>

				<div class="history-card">
					<div class="history-card-header">
						<h3>Saved Analyses</h3>
					</div>
					<div class="history-list" id="history-list">
						${renderHistoryList(classroom.id, student.id, history)}
					</div>
				</div>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: safeText(student.name || 'Student Profile') });
	attachShellHandlers();
	bindEditStudentAction(classroom, student);
	bindDeleteStudentAction(classroom, student);
	bindSavedReportActions(classroom.id, student.id);
	if (chartPoints.length >= 2) {
		renderMoodChart(chartPoints);
	}

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function renderHistoryList(classId, studentId, history) {
	if (!history.length) {
		return `
			<div class="history-empty-state">
				<i data-lucide="file-text"></i>
				<h4>No saved analyses yet.</h4>
				<p class="text-muted">Saved analyses will appear here after you attach a report to this student.</p>
			</div>
		`;
	}

	return history
		.map((item) => `
			<article class="history-item" data-job-id="${safeText(item.job_id)}">
				<div class="history-item-top">
					<span class="badge ${getMoodBadgeClass(item.mood)}">${safeText(item.mood || 'N/A')}</span>
					<span class="badge badge-gray">${item.confidence || '—'}</span>
				</div>
				<p class="history-item-desc">${safeText(item.drawing_description || item.summary || 'No description provided.')}</p>
				<p class="history-item-date">Saved ${formatDateLong(item.saved_at)}</p>
				<div class="history-item-actions">
					<button type="button" class="btn btn-secondary btn-sm view-full-report-btn"
						data-class-id="${classId}"
						data-student-id="${studentId}"
						data-job-id="${safeText(item.job_id)}">
						View Full Report
					</button>
				</div>
			</article>
		`)
		.join('');
}

function bindSavedReportActions(classId, studentId) {
	const buttons = document.querySelectorAll('.view-full-report-btn');
	buttons.forEach((button) => {
		button.addEventListener('click', async () => {
			const jobId = button.dataset.jobId;
			if (!jobId) return;
			button.disabled = true;
			button.textContent = 'Opening...';
			try {
				const report = await fetchSavedReport(classId, studentId, jobId);
				setLatestAnalysisResult(report);
				setAnalysisReportMode('saved');
				window.location.hash = '#/analysis/report';
			} catch (error) {
				button.disabled = false;
				button.textContent = 'View Full Report';
				showToast('error', error.message || 'Failed to open saved report.');
			}
		});
	});
}

function bindEditStudentAction(classroom, student) {
	const button = document.getElementById('edit-student-btn');
	if (!button) return;

	button.addEventListener('click', () => {
		openModal(renderEditStudentModal(student), {
			onClose: () => {
				if (window.lucide && typeof window.lucide.createIcons === 'function') {
					window.lucide.createIcons();
				}
			},
		});

		const form = document.getElementById('edit-student-form');
		const cancelButton = document.getElementById('cancel-edit-student-btn');
		const submitButton = document.getElementById('submit-edit-student-btn');
		if (!form || !cancelButton || !submitButton) return;

		cancelButton.addEventListener('click', () => closeModal());
		form.addEventListener('submit', async (event) => {
			event.preventDefault();
			const formData = new FormData(form);
			const payload = {
				name: String(formData.get('name') || '').trim(),
				gender: String(formData.get('gender') || '').trim(),
			};
			try {
				submitButton.disabled = true;
				await updateStudent(student.id, payload);
				showToast('success', 'Student updated successfully.');
				closeModal();
				window.location.reload();
			} catch (error) {
				submitButton.disabled = false;
				showToast('error', error.message || 'Failed to update student.');
			}
		});
	});
}

function bindDeleteStudentAction(classroom, student) {
	const button = document.getElementById('delete-student-btn');
	if (!button) return;

	button.addEventListener('click', () => {
		openModal(`
			<div class="modal-confirm-body">
				<h3>Delete ${safeText(student.name)}?</h3>
				<p class="text-muted">This action cannot be undone.</p>
				<div class="modal-actions" style="margin-top: 20px;">
					<button id="cancel-delete-student-btn" class="btn btn-ghost" type="button">Cancel</button>
					<button id="confirm-delete-student-btn" class="btn btn-danger" type="button">Delete</button>
				</div>
			</div>
		`);

		const cancelButton = document.getElementById('cancel-delete-student-btn');
		const confirmButton = document.getElementById('confirm-delete-student-btn');
		cancelButton?.addEventListener('click', () => closeModal());
		confirmButton?.addEventListener('click', async () => {
			try {
				confirmButton.disabled = true;
				await deleteStudent(student.id);
				showToast('success', 'Student deleted successfully.');
				closeModal();
				window.location.hash = `#/classes/${classroom.id}`;
			} catch (error) {
				confirmButton.disabled = false;
				showToast('error', error.message || 'Failed to delete student.');
			}
		});
	});
}

function renderEditStudentModal(student) {
	return `
		<form id="edit-student-form" class="modal-form">
			<h3 style="margin-bottom: 16px;">Edit Student</h3>
			<div class="form-group">
				<label class="form-label" for="student-name-input">Student Name</label>
				<input id="student-name-input" name="name" class="form-input" value="${safeText(student.name || '')}" required />
			</div>
			<div class="form-group">
				<label class="form-label" for="student-gender-input">Gender</label>
				<select id="student-gender-input" name="gender" class="form-select" required>
					${renderGenderOptions(student.gender)}
				</select>
			</div>
			<div class="modal-actions">
				<button id="cancel-edit-student-btn" class="btn btn-ghost" type="button">Cancel</button>
				<button id="submit-edit-student-btn" class="btn btn-primary" type="submit">Save Changes</button>
			</div>
		</form>
	`;
}

function renderGenderOptions(selectedGender) {
	const options = ['Female', 'Male', 'Non-binary', 'Prefer not to say'];
	return options
		.map((option) => `<option value="${safeText(option)}" ${option === selectedGender ? 'selected' : ''}>${safeText(option)}</option>`)
		.join('');
}

function renderMoodChart(points) {
	const canvas = document.getElementById('mood-chart-canvas');
	const tooltip = document.getElementById('mood-chart-tooltip');
	if (!canvas || !tooltip) return;

	const context = canvas.getContext('2d');
	if (!context) return;

	const resizeAndDraw = () => {
		const rect = canvas.getBoundingClientRect();
		const dpr = window.devicePixelRatio || 1;
		canvas.width = rect.width * dpr;
		canvas.height = rect.height * dpr;
		context.setTransform(dpr, 0, 0, dpr, 0, 0);
		drawChart(context, rect.width, rect.height, points);
	};

	resizeAndDraw();
	window.addEventListener('resize', resizeAndDraw, { once: true });

	canvas.addEventListener('mousemove', (event) => {
		const rect = canvas.getBoundingClientRect();
		const x = event.clientX - rect.left;
		const nearest = getNearestPoint(points, rect.width, x);
		if (!nearest) {
			tooltip.hidden = true;
			return;
		}

		tooltip.hidden = false;
		tooltip.textContent = `${formatDateShort(nearest.saved_at)} · Happy ${Math.round(nearest.happyScore)}%`;
		tooltip.style.left = `${nearest.x}px`;
		tooltip.style.top = `${nearest.y - 14}px`;
	});

	canvas.addEventListener('mouseleave', () => {
		tooltip.hidden = true;
	});
}

function drawChart(context, width, height, points) {
	context.clearRect(0, 0, width, height);

	const padding = { top: 18, right: 20, bottom: 36, left: 28 };
	const chartWidth = width - padding.left - padding.right;
	const chartHeight = height - padding.top - padding.bottom;
	const zoneToY = (score) => padding.top + chartHeight - (score / 100) * chartHeight;

	drawZones(context, width, height, zoneToY);
	drawThresholdLine(context, padding.left, width - padding.right, zoneToY(40));
	drawThresholdLine(context, padding.left, width - padding.right, zoneToY(60));

	context.strokeStyle = '#008080';
	context.lineWidth = 3;
	context.beginPath();

	points.forEach((point, index) => {
		const x = padding.left + (chartWidth * index) / Math.max(points.length - 1, 1);
		const y = zoneToY(point.happyScore);
		point.x = x;
		point.y = y;
		if (index === 0) context.moveTo(x, y);
		else context.lineTo(x, y);
	});
	context.stroke();

	points.forEach((point) => {
		context.beginPath();
		context.fillStyle = '#008080';
		context.arc(point.x, point.y, 4, 0, Math.PI * 2);
		context.fill();
	});

	context.fillStyle = '#6B7280';
	context.font = '12px DM Sans, sans-serif';
	context.fillText('0%', 2, height - 8);
	context.fillText('100%', 2, 16);
}

function drawZones(context, width, height, zoneToY) {
	context.fillStyle = 'rgba(220, 38, 38, 0.05)';
	context.fillRect(0, zoneToY(40), width, height - zoneToY(40));
	context.fillStyle = 'rgba(217, 119, 6, 0.05)';
	context.fillRect(0, zoneToY(60), width, zoneToY(40) - zoneToY(60));
	context.fillStyle = 'rgba(5, 150, 105, 0.05)';
	context.fillRect(0, 0, width, zoneToY(60));
}

function drawThresholdLine(context, left, right, y) {
	context.strokeStyle = 'rgba(107, 114, 128, 0.35)';
	context.setLineDash([6, 6]);
	context.beginPath();
	context.moveTo(left, y);
	context.lineTo(right, y);
	context.stroke();
	context.setLineDash([]);
}

function getNearestPoint(points, width, mouseX) {
	if (!points.length) return null;
	let nearest = points[0];
	let nearestDistance = Number.POSITIVE_INFINITY;

	points.forEach((point, index) => {
		const x = point.x ?? (width * index) / Math.max(points.length - 1, 1);
		const distance = Math.abs(mouseX - x);
		if (distance < nearestDistance) {
			nearest = point;
			nearestDistance = distance;
		}
	});

	return nearest;
}

function buildChartPoints(history) {
	return [...history]
		.sort((a, b) => new Date(a.saved_at || 0) - new Date(b.saved_at || 0))
		.map((item) => ({
			saved_at: item.saved_at,
			happyScore: normalizeHappyScore(item.happy_score),
		}));
}

function normalizeHappyScore(value) {
	const numeric = Number(value);
	if (Number.isNaN(numeric)) return 50;
	if (numeric >= 0 && numeric <= 1) return numeric * 100;
	return Math.max(0, Math.min(100, numeric));
}

function getMoodBadgeClass(mood) {
	const normalized = String(mood || '').toLowerCase();
	if (normalized.includes('happy')) return 'badge-success';
	if (normalized.includes('sad')) return 'badge-warning';
	return 'badge-info';
}

function getInitials(name) {
	const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
	if (!parts.length) return '?';
	return parts.slice(0, 2).map((part) => part.charAt(0).toUpperCase()).join('');
}

function formatDateShort(value) {
	if (!value) return 'N/A';
	const date = new Date(value);
	if (Number.isNaN(date.getTime())) return 'N/A';
	const day = String(date.getDate()).padStart(2, '0');
	const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
	return `${day}-${months[date.getMonth()]}-${String(date.getFullYear()).slice(-2)}`;
}

function formatDateLong(value) {
	if (!value) return 'N/A';
	const date = new Date(value);
	if (Number.isNaN(date.getTime())) return 'N/A';
	return date.toLocaleString();
}

function formatRelativeTime(value) {
	if (!value) return 'N/A';
	const timestamp = new Date(value).getTime();
	if (Number.isNaN(timestamp)) return 'N/A';
	const delta = Date.now() - timestamp;
	const minute = 60 * 1000;
	const hour = 60 * minute;
	const day = 24 * hour;
	if (delta < minute) return 'Just now';
	if (delta < hour) return `${Math.floor(delta / minute)}m ago`;
	if (delta < day) return `${Math.floor(delta / hour)}h ago`;
	return `${Math.floor(delta / day)}d ago`;
}

function renderErrorState(appElement, error, classId) {
	const pageContent = `
		<div class="dashboard-container">
			<section class="card classes-empty-state">
				<i data-lucide="alert-triangle"></i>
				<h3>Unable to load student profile</h3>
				<p class="text-muted">${safeText(error.message || 'Please try again in a moment.')}</p>
				<a href="#/classes/${classId}" class="btn btn-secondary">Back to Class</a>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Student Profile' });
	attachShellHandlers();
	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
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