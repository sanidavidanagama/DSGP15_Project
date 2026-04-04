import { createShell, attachShellHandlers } from '../components/shell.js';
import { fetchClassById, updateClass } from '../api/classes.js';
import { showToast } from '../components/toast.js';

const DAY_OPTIONS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

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
		renderEditClassView(appElement, classroom);
	} catch (error) {
		showToast('error', error.message || 'Failed to load class details.');
		renderErrorState(appElement, error);
	}
}

function renderEditClassView(appElement, classroom) {
	const pageContent = `
		<div class="dashboard-container class-form-page">
			<section class="card class-form-card">
				<div class="class-form-intro">
					<h2>Edit Class</h2>
					<p class="text-muted">Update the details for ${safeText(classroom.class_name || 'this class')}.</p>
				</div>
			</section>

			<div class="class-form-grid">
				<section class="card class-form-card">
					<form id="edit-class-form">
						<div class="form-group">
							<label class="form-label" for="class-name-input">Class Name</label>
							<input id="class-name-input" name="class_name" class="form-input" type="text" value="${safeAttr(classroom.class_name || '')}" required />
						</div>

						<div class="form-group">
							<label class="form-label" for="grade-age-group-input">Grade / Age Group</label>
							<input id="grade-age-group-input" name="grade_age_group" class="form-input" type="text" value="${safeAttr(classroom.grade_age_group || '')}" required />
						</div>

						<div class="form-group">
							<label class="form-label">Schedule Days</label>
							<p class="text-muted text-sm" style="margin-bottom: 12px;">Pick at least one day the class meets.</p>
							<div class="schedule-days-grid">
								${renderScheduleOptions(classroom.schedule_days)}
							</div>
						</div>

						<div class="form-group">
							<label class="form-label" for="class-description-input">Description</label>
							<textarea id="class-description-input" name="description" class="form-textarea" rows="5">${safeText(classroom.description || '')}</textarea>
						</div>

						<p id="class-form-status" class="text-muted class-form-status" aria-live="polite"></p>

						<div class="class-form-actions">
							<a href="#/classes/${classroom.id}" class="btn btn-secondary">Cancel</a>
							<button id="submit-class-btn" class="btn btn-primary" type="submit">
								<i data-lucide="save"></i>
								Save Changes
							</button>
						</div>
					</form>
				</section>

				<aside class="class-form-side">
					<section class="card class-form-card">
						<h3 style="margin-bottom: 12px;">Editing tips</h3>
						<ul class="class-form-tips">
							<li>Keep the class name recognizable for teachers.</li>
							<li>Update the schedule if the meeting pattern changes.</li>
							<li>Descriptions can stay short and practical.</li>
						</ul>
					</section>
				</aside>
			</div>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Edit Class' });
	attachShellHandlers();
	bindEditClassForm(classroom.id);

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function bindEditClassForm(classId) {
	const form = document.getElementById('edit-class-form');
	const submitButton = document.getElementById('submit-class-btn');
	const statusMessage = document.getElementById('class-form-status');
	if (!form || !submitButton || !statusMessage) return;

	const setStatusMessage = (message, visible = true) => {
		statusMessage.textContent = message;
		statusMessage.style.display = visible ? 'block' : 'none';
	};

	const setSubmitting = (isSubmitting) => {
		submitButton.disabled = isSubmitting;
		submitButton.innerHTML = isSubmitting
			? '<i data-lucide="loader-2"></i>Saving...'
			: '<i data-lucide="save"></i>Save Changes';
		if (window.lucide && typeof window.lucide.createIcons === 'function') {
			window.lucide.createIcons();
		}
	};

	form.addEventListener('submit', async (event) => {
		event.preventDefault();

		const formData = new FormData(form);
		const className = String(formData.get('class_name') || '').trim();
		const gradeAgeGroup = String(formData.get('grade_age_group') || '').trim();
		const description = String(formData.get('description') || '').trim();
		const scheduleDays = Array.from(form.querySelectorAll('input[name="schedule_days"]:checked')).map((input) => input.value);

		if (!className || !gradeAgeGroup) {
			setStatusMessage('Please complete the class name and grade / age group fields.');
			return;
		}

		if (!scheduleDays.length) {
			setStatusMessage('Please select at least one schedule day.');
			return;
		}

		setSubmitting(true);
		setStatusMessage('Saving changes...', true);

		try {
			await updateClass(classId, {
				class_name: className,
				grade_age_group: gradeAgeGroup,
				schedule_days: scheduleDays,
				description: description || undefined,
			});

			showToast('success', 'Class updated successfully.');
			window.location.hash = `#/classes/${classId}`;
		} catch (error) {
			setSubmitting(false);
			setStatusMessage(error.message || 'Failed to update class.');
			showToast('error', error.message || 'Failed to update class.');
		}
	});
}

function renderScheduleOptions(selectedDays) {
	const days = Array.isArray(selectedDays) ? selectedDays : [];
	return DAY_OPTIONS.map((day) => `
		<label class="schedule-day-option">
			<input type="checkbox" name="schedule_days" value="${safeAttr(day)}" ${days.includes(day) ? 'checked' : ''} />
			<span>${safeText(day)}</span>
		</label>
	`).join('');
}

function renderErrorState(appElement, error) {
	const pageContent = `
		<div class="dashboard-container">
			<section class="card classes-empty-state">
				<i data-lucide="alert-triangle"></i>
				<h3>Unable to load class</h3>
				<p class="text-muted">${safeText(error.message || 'Please try again in a moment.')}</p>
				<a href="#/classes" class="btn btn-secondary">Back to Classes</a>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Edit Class' });
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

function safeAttr(value) {
	return safeText(value).replaceAll('\n', ' ');
}