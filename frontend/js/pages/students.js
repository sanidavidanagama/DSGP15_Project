import { createShell, attachShellHandlers } from '../components/shell.js';
import { fetchStudents } from '../api/students.js';
import { showToast } from '../components/toast.js';

export async function renderPage() {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	try {
		const students = await fetchStudents();
		renderStudentsView(appElement, Array.isArray(students) ? students : []);
	} catch (error) {
		showToast('error', error.message || 'Failed to load students.');
		renderErrorState(appElement, error);
	}
}

function renderStudentsView(appElement, students) {
	const totalStudents = students.length;
	const totalClasses = new Set(students.map((student) => student.class_id)).size;
	const totalAnalyses = students.reduce((sum, student) => sum + Number(student.total_analyses || 0), 0);
	const recentUpdates = students.filter((student) => Boolean(student.last_predicted_at)).length;

	const pageContent = `
		<div class="dashboard-container students-page">
			<section class="card students-page-header">
				<div class="students-page-header-row">
					<div>
						<h2>Students</h2>
						<p class="text-muted" style="margin-top: 8px;">Browse every student across your classes, track recent analyses, and jump straight into profiles.</p>
					</div>
					<a href="#/classes" class="btn btn-primary">
						<i data-lucide="school"></i>
						Open Classes
					</a>
				</div>
			</section>

			<section class="students-summary-grid">
				<article class="glass-card stat-card">
					<div class="stat-card-icon-row"><div class="stat-card-icon"><i data-lucide="users"></i></div></div>
					<div class="stat-card-value">${totalStudents}</div>
					<div class="stat-card-label">Total Students</div>
				</article>

				<article class="glass-card stat-card">
					<div class="stat-card-icon-row"><div class="stat-card-icon"><i data-lucide="book-open"></i></div></div>
					<div class="stat-card-value">${totalClasses}</div>
					<div class="stat-card-label">Classes With Students</div>
				</article>

				<article class="glass-card stat-card">
					<div class="stat-card-icon-row"><div class="stat-card-icon"><i data-lucide="scan-eye"></i></div></div>
					<div class="stat-card-value">${totalAnalyses}</div>
					<div class="stat-card-label">Total Analyses</div>
				</article>

				<article class="glass-card stat-card">
					<div class="stat-card-icon-row"><div class="stat-card-icon"><i data-lucide="clock-3"></i></div></div>
					<div class="stat-card-value">${recentUpdates}</div>
					<div class="stat-card-label">Updated Profiles</div>
				</article>
			</section>

			<section class="card students-toolbar-card">
				<div class="students-toolbar">
					<div class="search-wrapper students-search-wrapper">
						<i data-lucide="search"></i>
						<input id="students-search" type="search" class="form-input" placeholder="Search by name, class, mood, or gender" />
					</div>
					<select id="students-class-filter" class="form-select students-filter-select">
						<option value="">All classes</option>
					</select>
				</div>
				<p class="text-muted text-sm" id="students-count-label">${totalStudents} students</p>
			</section>

			<section>
				<div id="students-grid" class="students-grid">
					${renderStudentCards(students)}
				</div>
			</section>

			<section class="card students-help-card">
				<div>
					<h3>Add Students</h3>
					<p class="text-muted" style="margin-top: 8px;">Open a class to create or manage students. The class detail page includes the add-student entry point.</p>
				</div>
				<a href="#/classes" class="btn btn-secondary">Go to Classes</a>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Students' });
	attachShellHandlers();
	bindStudentsFiltering(students);
	populateClassFilter(students);

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function bindStudentsFiltering(allStudents) {
	const searchInput = document.getElementById('students-search');
	const classFilter = document.getElementById('students-class-filter');
	const studentsGrid = document.getElementById('students-grid');
	const countLabel = document.getElementById('students-count-label');
	if (!searchInput || !classFilter || !studentsGrid || !countLabel) return;

	const renderFilteredStudents = () => {
		const query = searchInput.value.trim().toLowerCase();
		const classIdFilter = classFilter.value;
		const filtered = allStudents.filter((student) => {
			const haystack = [
				student.name,
				student.class_name,
				student.grade_age_group,
				student.last_predicted_mood,
				student.gender,
			].map((value) => String(value || '').toLowerCase());

			const matchesQuery = !query || haystack.some((value) => value.includes(query));
			const matchesClass = !classIdFilter || String(student.class_id) === classIdFilter;
			return matchesQuery && matchesClass;
		});

		countLabel.textContent = `${filtered.length} student${filtered.length === 1 ? '' : 's'}`;
		studentsGrid.innerHTML = renderStudentCards(filtered);

		if (window.lucide && typeof window.lucide.createIcons === 'function') {
			window.lucide.createIcons();
		}
	};

	searchInput.addEventListener('input', renderFilteredStudents);
	classFilter.addEventListener('change', renderFilteredStudents);
}

function populateClassFilter(students) {
	const classFilter = document.getElementById('students-class-filter');
	if (!classFilter) return;

	const classes = new Map();
	students.forEach((student) => {
		if (!student.class_id || classes.has(String(student.class_id))) return;
		classes.set(String(student.class_id), student.class_name || `Class ${student.class_id}`);
	});

	classFilter.innerHTML = ['<option value="">All classes</option>']
		.concat(Array.from(classes.entries()).map(([id, name]) => `<option value="${id}">${safeText(name)}</option>`))
		.join('');
}

function renderStudentCards(students) {
	if (!students.length) {
		return `
			<div class="card students-empty-state">
				<i data-lucide="users"></i>
				<h3>No students found</h3>
				<p class="text-muted">Try another search or open a class to add the first student.</p>
				<a href="#/classes" class="btn btn-secondary">Open Classes</a>
			</div>
		`;
	}

	return students.map((student) => {
		const initials = getInitials(student.name);
		const moodBadgeClass = getMoodBadgeClass(student.last_predicted_mood);
		return `
			<a href="#/classes/${student.class_id}/students/${student.id}" class="card card-clickable student-directory-card">
				<div class="student-directory-top">
					<div class="student-directory-avatar">${safeText(initials)}</div>
					<div class="student-directory-info">
						<h3>${safeText(student.name || 'Unnamed student')}</h3>
						<p class="text-muted">${safeText(student.class_name || 'Class')} · ${safeText(student.grade_age_group || 'N/A')}</p>
					</div>
				</div>

				<div class="student-directory-badges">
					${student.last_predicted_mood ? `<span class="badge ${moodBadgeClass}">${safeText(student.last_predicted_mood)}</span>` : '<span class="badge badge-gray">No mood yet</span>'}
					<span class="badge badge-gray">${Number(student.total_analyses || 0)} analyses</span>
				</div>

				<div class="student-directory-footer">
					<span class="text-sm text-muted">${student.last_predicted_at ? `Updated ${formatRelativeTime(student.last_predicted_at)}` : 'No updates yet'}</span>
					<span class="class-card-open">Open Profile <i data-lucide="arrow-right"></i></span>
				</div>
			</a>
		`;
	}).join('');
}

function renderErrorState(appElement, error) {
	const pageContent = `
		<div class="dashboard-container students-page">
			<section class="card students-empty-state">
				<i data-lucide="alert-triangle"></i>
				<h3>Unable to load students</h3>
				<p class="text-muted">${safeText(error.message || 'Please try again in a moment.')}</p>
				<a href="#/classes" class="btn btn-secondary">Back to Classes</a>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Students' });
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

function formatRelativeTime(value) {
	if (!value) return 'No updates yet';
	const timestamp = new Date(value).getTime();
	if (Number.isNaN(timestamp)) return 'No updates yet';

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