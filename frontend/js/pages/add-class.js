import { createShell, attachShellHandlers } from '../components/shell.js';
import { createClass } from '../api/classes.js';
import { showToast } from '../components/toast.js';

const DAY_OPTIONS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export async function renderPage() {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	const pageContent = `
		<div class="dashboard-container class-form-page">
			<section class="card class-form-card">
				<div class="class-form-intro">
					<h2>Add New Class</h2>
					<p class="text-muted">Create a class group so you can organize students and save analysis reports to the right profile.</p>
				</div>
			</section>

			<div class="class-form-grid">
				<section class="card class-form-card">
					<form id="add-class-form">
						<div class="form-group">
							<label class="form-label" for="class-name-input">Class Name</label>
							<input id="class-name-input" name="class_name" class="form-input" type="text" placeholder="E.g. Rainbow Cubs" required />
						</div>

						<div class="form-group">
							<label class="form-label" for="grade-age-group-input">Grade / Age Group</label>
							<input id="grade-age-group-input" name="grade_age_group" class="form-input" type="text" placeholder="E.g. Kindergarten, Ages 5-6" required />
						</div>

						<div class="form-group">
							<label class="form-label">Schedule Days</label>
							<p class="text-muted text-sm" style="margin-bottom: 12px;">Pick at least one day the class meets.</p>
							<div class="schedule-days-grid">
								${renderScheduleOptions()}
							</div>
						</div>

						<div class="form-group">
							<label class="form-label" for="class-description-input">Description</label>
							<textarea id="class-description-input" name="description" class="form-textarea" rows="5" placeholder="Optional notes about the class, structure, or support needs."></textarea>
						</div>

						<p id="class-form-status" class="text-muted class-form-status" aria-live="polite"></p>

						<div class="class-form-actions">
							<a href="#/classes" class="btn btn-secondary">Cancel</a>
							<button id="submit-class-btn" class="btn btn-primary" type="submit">
								<i data-lucide="plus"></i>
								Create Class
							</button>
						</div>
					</form>
				</section>

				<aside class="class-form-side">
					<section class="card class-form-card">
						<h3 style="margin-bottom: 12px;">What happens next?</h3>
						<div class="class-form-note">
							<p class="text-muted">After you create the class, you can open it from the Classes page, add students, and save analysis results directly to their profiles.</p>
						</div>
					</section>

					<section class="card class-form-card">
						<h3 style="margin-bottom: 12px;">Tips</h3>
						<ul class="class-form-tips">
							<li>Use a clear class name that teachers will recognize quickly.</li>
							<li>Choose all days that the class regularly meets.</li>
							<li>Keep the description short and practical.</li>
						</ul>
					</section>
				</aside>
			</div>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Add Class' });
	attachShellHandlers();
	bindAddClassForm();

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function bindAddClassForm() {
	const form = document.getElementById('add-class-form');
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
			? '<i data-lucide="loader-2"></i>Creating...'
			: '<i data-lucide="plus"></i>Create Class';
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
		setStatusMessage('Creating class...', true);

		try {
			const createdClass = await createClass({
				class_name: className,
				grade_age_group: gradeAgeGroup,
				schedule_days: scheduleDays,
				description: description || undefined,
			});

			showToast('success', 'Class created successfully.');
			window.location.hash = `#/classes/${createdClass.id}`;
		} catch (error) {
			setSubmitting(false);
			setStatusMessage(error.message || 'Failed to create class.');
			showToast('error', error.message || 'Failed to create class.');
		}
	});
}

function renderScheduleOptions() {
	return DAY_OPTIONS.map((day) => `
		<label class="schedule-day-option">
			<input type="checkbox" name="schedule_days" value="${day}" />
			<span>${day}</span>
		</label>
	`).join('');
}
