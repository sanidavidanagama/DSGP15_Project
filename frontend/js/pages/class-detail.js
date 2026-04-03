import { createShell, attachShellHandlers } from '../components/shell.js';
import { fetchClassById } from '../api/classes.js';
import { deleteClass } from '../api/classes.js';
import { showToast } from '../components/toast.js';

export async function renderPage({ params } = {}) {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	const classId = params?.id;
	if (!classId) {
		showToast('error', 'Class not found.');
		window.location.hash = '#/classes';
		return;
	}

	try {
		const classroom = await fetchClassById(classId);
		renderClassDetail(appElement, classroom);
	} catch (error) {
		showToast('error', error.message || 'Failed to load class details.');
		renderErrorState(appElement, error);
	}
}

function renderClassDetail(appElement, classroom) {
	const students = Array.isArray(classroom.students) ? classroom.students : [];
	const totalAnalyses = students.reduce((sum, student) => sum + Number(student.total_analyses || 0), 0);
	const recentUpdates = students.filter((student) => Boolean(student.last_predicted_at)).length;

	const pageContent = `
		<div class="dashboard-container class-detail-page">
			<section class="class-detail-hero card">
				<div class="class-detail-hero-top">
					<div class="class-detail-hero-identity">
						<div class="class-detail-back-link-wrap">
							<a href="#/classes" class="btn btn-ghost btn-sm">
								<i data-lucide="arrow-left"></i>
								Back
							</a>
						</div>
						<div class="class-detail-hero-info">
							<h2>${safeText(classroom.class_name || 'Class')}</h2>
							<p class="class-detail-meta">${safeText(classroom.grade_age_group || 'N/A')}${classroom.description ? ` · ${safeText(classroom.description)}` : ''}</p>
							<div class="class-detail-chips">
								${renderScheduleDays(classroom.schedule_days)}
							</div>
						</div>
					</div>
					<div class="class-detail-actions">
						<a href="#/classes/${classroom.id}/edit" class="btn btn-secondary">
							<i data-lucide="pencil"></i>
							Edit Class
						</a>
						<button id="delete-class-btn" class="btn btn-danger-ghost" type="button">
							<i data-lucide="trash-2"></i>
							Delete Class
						</button>
					</div>
				</div>
			</section>

			<section class="classes-summary-grid class-detail-summary-grid">
				<article class="glass-card stat-card">
					<div class="stat-card-icon-row">
						<div class="stat-card-icon"><i data-lucide="users"></i></div>
					</div>
					<div class="stat-card-value">${students.length}</div>
					<div class="stat-card-label">Total Students</div>
				</article>

				<article class="glass-card stat-card">
					<div class="stat-card-icon-row">
						<div class="stat-card-icon"><i data-lucide="scan-eye"></i></div>
					</div>
					<div class="stat-card-value">${totalAnalyses}</div>
					<div class="stat-card-label">Total Analyses</div>
				</article>

				<article class="glass-card stat-card">
					<div class="stat-card-icon-row">
						<div class="stat-card-icon"><i data-lucide="clock-3"></i></div>
					</div>
					<div class="stat-card-value">${recentUpdates}</div>
					<div class="stat-card-label">Recently Updated</div>
				</article>
			</section>

			<section class="card class-detail-students-section">
				<div class="classes-page-header-row">
					<div>
						<h3>Students (${students.length})</h3>
						<p class="text-muted" style="margin-top: 6px;">Browse student profiles and their latest analysis summary.</p>
					</div>
					<a href="#/classes/${classroom.id}/students/new" class="btn btn-primary btn-sm">
						<i data-lucide="plus"></i>
						Add Student
					</a>
				</div>

				<div class="classes-toolbar class-detail-toolbar">
					<div class="search-wrapper classes-search-wrapper">
						<i data-lucide="search"></i>
						<input id="students-search" type="search" class="form-input" placeholder="Search students by name or mood" />
					</div>
				</div>

				<div id="students-grid" class="class-detail-students-grid">
					${renderStudentCards(classroom.id, students)}
				</div>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: safeText(classroom.class_name || 'Class Detail') });
	attachShellHandlers();
	bindStudentSearch(classroom.id, students);
	bindDeleteAction(classroom);

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function bindStudentSearch(classId, students) {
	const searchInput = document.getElementById('students-search');
	const studentsGrid = document.getElementById('students-grid');
	if (!searchInput || !studentsGrid) return;

	searchInput.addEventListener('input', () => {
		const query = searchInput.value.trim().toLowerCase();
		const filteredStudents = students.filter((student) => {
			const name = String(student.name || '').toLowerCase();
			const mood = String(student.last_predicted_mood || '').toLowerCase();
			return name.includes(query) || mood.includes(query);
		});

		studentsGrid.innerHTML = renderStudentCards(classId, filteredStudents);
		if (window.lucide && typeof window.lucide.createIcons === 'function') {
			window.lucide.createIcons();
		}
	});
}

function bindDeleteAction(classroom) {
	const deleteButton = document.getElementById('delete-class-btn');
	if (!deleteButton) return;

	deleteButton.addEventListener('click', async () => {
		openModal(`
			<div class="modal-confirm-body">
				<h3>Delete ${safeText(classroom.class_name)}?</h3>
				<p class="text-muted">This will remove the class and keep its students unavailable in the current teacher workspace.</p>
				<div class="modal-actions" style="margin-top: 20px;">
					<button id="cancel-delete-class-btn" class="btn btn-ghost" type="button">Cancel</button>
					<button id="confirm-delete-class-btn" class="btn btn-danger" type="button">Delete</button>
				</div>
			</div>
		`);
	});
		const cancelButton = document.getElementById('cancel-delete-class-btn');
		const confirmButton = document.getElementById('confirm-delete-class-btn');
		cancelButton?.addEventListener('click', () => closeModal());
		confirmButton?.addEventListener('click', async () => {
			try {
				confirmButton.disabled = true;
				await deleteClass(classroom.id);
				showToast('success', 'Class deleted successfully.');
				closeModal();
				window.location.hash = '#/classes';
			} catch (error) {
				confirmButton.disabled = false;
				showToast('error', error.message || 'Failed to delete class.');
			}
		});
}

function renderStudentCards(classId, students) {
	if (!students.length) {
		return `
			<div class="classes-empty-state card">
				<i data-lucide="user"></i>
				<h3>No students yet</h3>
				<p class="text-muted">Add the first student to this class to start tracking analyses.</p>
			</div>
		`;
	}

	return students
		.map((student) => {
			const initials = getInitials(student.name);
			const moodBadgeClass = getMoodBadgeClass(student.last_predicted_mood);
			return `
				<a href="#/classes/${classId}/students/${student.id}" class="card card-clickable class-student-card">
					<div class="class-student-card-top">
						<div class="student-avatar-circle">${safeText(initials)}</div>
						<div class="class-student-card-info">
							<h4>${safeText(student.name || 'Unnamed student')}</h4>
							<p class="text-muted text-sm">Joined ${formatDateShort(student.joined_at)}</p>
						</div>
					</div>

					<div class="class-student-card-meta">
						${student.last_predicted_mood ? `<span class="badge ${moodBadgeClass}">${safeText(student.last_predicted_mood)}</span>` : '<span class="badge badge-gray">No mood yet</span>'}
						<span class="badge badge-gray">${Number(student.total_analyses || 0)} analyses</span>
					</div>

					<div class="class-student-card-footer">
						<span class="text-sm text-muted">${student.last_predicted_at ? `Updated ${formatRelativeTime(student.last_predicted_at)}` : 'No updates yet'}</span>
						<span class="class-card-open">Open Profile <i data-lucide="arrow-right"></i></span>
					</div>
				</a>
			`;
		})
		.join('');
}

function renderScheduleDays(scheduleDays) {
	const days = Array.isArray(scheduleDays) ? scheduleDays : [];
	if (!days.length) {
		return '<span class="badge badge-gray">No schedule assigned</span>';
	}

	return days.map((day) => `<span class="badge badge-teal">${safeText(day)}</span>`).join('');
}

function renderErrorState(appElement, error) {
	const pageContent = `
		<div class="dashboard-container class-detail-page">
			<section class="card classes-empty-state">
				<i data-lucide="alert-triangle"></i>
				<h3>Unable to load class</h3>
				<p class="text-muted">${safeText(error.message || 'Please try again in a moment.')}</p>
				<a href="#/classes" class="btn btn-secondary">Back to Classes</a>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Class Detail' });
	attachShellHandlers();

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
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

function safeText(value) {
	const text = value == null ? '' : String(value);
	return text
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;')
		.replaceAll("'", '&#39;');
}