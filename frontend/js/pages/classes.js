import { createShell, attachShellHandlers } from '../components/shell.js';
import { fetchClasses } from '../api/classes.js';
import { showToast } from '../components/toast.js';

export async function renderPage() {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	try {
		const classes = await fetchClasses();
		renderClassesView(appElement, classes || []);
	} catch (error) {
		showToast('error', error.message || 'Failed to load classes.');
		renderErrorState(appElement, error);
	}
}

function renderClassesView(appElement, classes) {
	const totalClasses = classes.length;
	const totalStudents = classes.reduce((sum, classroom) => sum + Number(classroom.student_count || 0), 0);
	const latestUpdatedClass = [...classes]
		.sort((a, b) => new Date(b.updated_at || 0).getTime() - new Date(a.updated_at || 0).getTime())[0];
	const latestUpdateText = latestUpdatedClass?.updated_at
		? formatRelativeTime(latestUpdatedClass.updated_at)
		: 'No updates yet';

	const pageContent = `
		<div class="dashboard-container">
			<section class="card classes-page-header">
				<div class="classes-page-header-row">
					<div>
						<h2>My Classes</h2>
						<p class="text-muted" style="margin-top: 8px;">Manage class groups, track student coverage, and jump into details.</p>
					</div>
					<a href="#/classes/new" class="btn btn-primary">
						<i data-lucide="plus"></i>
						Add New Class
					</a>
				</div>
			</section>

			<section class="classes-summary-grid">
				<article class="glass-card stat-card">
					<div class="stat-card-icon-row">
						<div class="stat-card-icon"><i data-lucide="school"></i></div>
					</div>
					<div class="stat-card-value">${totalClasses}</div>
					<div class="stat-card-label">Total Classes</div>
				</article>

				<article class="glass-card stat-card">
					<div class="stat-card-icon-row">
						<div class="stat-card-icon"><i data-lucide="users"></i></div>
					</div>
					<div class="stat-card-value">${totalStudents}</div>
					<div class="stat-card-label">Total Students</div>
				</article>

				<article class="glass-card stat-card">
					<div class="stat-card-icon-row">
						<div class="stat-card-icon"><i data-lucide="clock-3"></i></div>
					</div>
					<div class="stat-card-value classes-last-updated">${safeText(latestUpdateText)}</div>
					<div class="stat-card-label">Last Analysis</div>
				</article>
			</section>

			<section class="classes-toolbar card">
				<div class="search-wrapper classes-search-wrapper">
					<i data-lucide="search"></i>
					<input id="classes-search" type="search" class="form-input" placeholder="Search classes by name or grade" />
				</div>
				<span class="text-sm text-muted" id="classes-count-label">${totalClasses} classes</span>
			</section>

			<section>
				<div id="classes-grid" class="classes-grid">
					${renderClassCards(classes)}
					${renderAddClassCard()}
				</div>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'My Classes' });
	attachShellHandlers();

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}

	bindClassesFiltering(classes);
}

function bindClassesFiltering(allClasses) {
	const searchInput = document.getElementById('classes-search');
	const classesGrid = document.getElementById('classes-grid');
	const countLabel = document.getElementById('classes-count-label');
	if (!searchInput || !classesGrid || !countLabel) return;

	searchInput.addEventListener('input', () => {
		const query = searchInput.value.trim().toLowerCase();
		const filtered = allClasses.filter((classroom) => {
			const className = String(classroom.class_name || '').toLowerCase();
			const grade = String(classroom.grade_age_group || '').toLowerCase();
			return className.includes(query) || grade.includes(query);
		});

		countLabel.textContent = `${filtered.length} class${filtered.length === 1 ? '' : 'es'}`;
		classesGrid.innerHTML = `
			${renderClassCards(filtered)}
			${renderAddClassCard()}
		`;

		if (window.lucide && typeof window.lucide.createIcons === 'function') {
			window.lucide.createIcons();
		}
	});
}

function renderClassCards(classes) {
	if (!classes.length) {
		return `
			<div class="card classes-empty-state">
				<i data-lucide="school"></i>
				<h3>No matching classes found</h3>
				<p class="text-muted">Try another search term or create a new class.</p>
			</div>
		`;
	}

	return classes
		.map((classroom) => {
			const scheduleDays = Array.isArray(classroom.schedule_days) ? classroom.schedule_days : [];
			return `
				<a href="#/classes/${classroom.id}" class="card card-clickable class-card class-card-link" aria-label="Open ${safeText(classroom.class_name || 'class')}">
					<div class="class-card-body">
						<h3 class="class-card-name">${safeText(classroom.class_name || 'Untitled class')}</h3>
						<p class="class-card-grade">${safeText(classroom.grade_age_group || 'N/A')}</p>

						<div class="class-card-schedule">
							${scheduleDays.length
								? scheduleDays.map((day) => `<span class="badge badge-teal">${safeText(day)}</span>`).join('')
								: '<span class="badge badge-gray">No schedule</span>'}
						</div>

						<p class="class-card-desc">${safeText(classroom.description || 'No class description provided yet.')}</p>
					</div>

					<div class="class-card-footer">
						<div class="class-card-student-count">
							<i data-lucide="users"></i>
							<span>${Number(classroom.student_count || 0)} students</span>
						</div>
						<div class="class-card-open">
							Open class
							<i data-lucide="arrow-right"></i>
						</div>
					</div>
				</a>
			`;
		})
		.join('');
}

function renderAddClassCard() {
	return `
		<a href="#/classes/new" class="card-dashed classes-add-card">
			<i data-lucide="plus" class="card-dashed-icon"></i>
			<h3>Add Class</h3>
			<p class="text-muted text-sm">Create a new class group</p>
		</a>
	`;
}

function renderErrorState(appElement, error) {
	const pageContent = `
		<div class="dashboard-container">
			<section class="card classes-empty-state">
				<i data-lucide="alert-triangle"></i>
				<h3>Unable to load classes</h3>
				<p class="text-muted">${safeText(error.message || 'Please try again in a moment.')}</p>
				<a href="#/dashboard" class="btn btn-secondary">Back to Dashboard</a>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'My Classes' });
	attachShellHandlers();

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
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
