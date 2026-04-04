import { createShell, attachShellHandlers } from '../components/shell.js';
import { fetchClassById } from '../api/classes.js';
import { createStudent } from '../api/students.js';
import { showToast } from '../components/toast.js';

const GENDER_OPTIONS = [
	'Male',
	'Female',
	'Non-binary',
	'Prefer not to say',
];

export async function renderPage({ params } = {}) {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	const classId = params?.classId;
	if (!classId) {
		showToast('error', 'Class not found.');
		window.location.hash = '#/classes';
		return;
	}

	try {
		const classroom = await fetchClassById(classId);
		renderAddStudentView(appElement, classroom);
	} catch (error) {
		showToast('error', error.message || 'Failed to load class details.');
		renderErrorState(appElement, error);
	}
}

function renderAddStudentView(appElement, classroom) {
	const pageContent = `
		<div class="dashboard-container class-form-page student-form-page">
			<section class="card class-form-card">
				<div class="class-form-intro">
					<h2>Add Student</h2>
					<p class="text-muted">Add a student to ${safeText(classroom.class_name || 'this class')} so you can save future analysis reports to their profile.</p>
				</div>
			</section>

			<div class="class-form-grid">
				<section class="card class-form-card">
					<div class="class-form-note" style="margin-bottom: 20px;">
						<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
							<span class="badge badge-teal">${safeText(classroom.class_name || 'Class')}</span>
							<span class="badge badge-gray">${safeText(classroom.grade_age_group || 'N/A')}</span>
						</div>
						<p class="text-muted" style="margin: 0;">Students added here will appear in the class roster and the wider students directory.</p>
					</div>

					<form id="add-student-form">
						<div class="form-group">
							<label class="form-label" for="student-name-input">Student Name</label>
							<input id="student-name-input" name="name" class="form-input" type="text" placeholder="E.g. Amina Yusuf" required />
						</div>

						<div class="form-group">
							<label class="form-label" for="student-gender-input">Gender</label>
							<select id="student-gender-input" name="gender" class="form-select" required>
								<option value="" selected disabled>Select gender</option>
								${GENDER_OPTIONS.map((gender) => `<option value="${safeText(gender)}">${safeText(gender)}</option>`).join('')}
							</select>
						</div>

						<p id="student-form-status" class="text-muted class-form-status" aria-live="polite"></p>

						<div class="class-form-actions">
							<a href="#/classes/${classroom.id}" class="btn btn-secondary">Cancel</a>
							<button id="submit-student-btn" class="btn btn-primary" type="submit">
								<i data-lucide="user-plus"></i>
								Add Student
							</button>
						</div>
					</form>
				</section>

				<aside class="class-form-side">
					<section class="card class-form-card">
						<h3 style="margin-bottom: 12px;">What happens next?</h3>
						<div class="class-form-note">
							<p class="text-muted">Once the student is created, you can open their profile, edit their details, and attach saved analysis reports.</p>
						</div>
					</section>

					<section class="card class-form-card">
						<h3 style="margin-bottom: 12px;">Tips</h3>
						<ul class="class-form-tips">
							<li>Use the student’s preferred display name.</li>
							<li>Pick the closest matching gender label for your records.</li>
							<li>You can edit the student later if anything changes.</li>
						</ul>
					</section>
				</aside>
			</div>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Add Student' });
	attachShellHandlers();
	bindAddStudentForm(classroom.id);

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function bindAddStudentForm(classId) {
	const form = document.getElementById('add-student-form');
	const submitButton = document.getElementById('submit-student-btn');
	const statusMessage = document.getElementById('student-form-status');
	if (!form || !submitButton || !statusMessage) return;

	const setStatusMessage = (message, visible = true) => {
		statusMessage.textContent = message;
		statusMessage.style.display = visible ? 'block' : 'none';
	};

	const setSubmitting = (isSubmitting) => {
		submitButton.disabled = isSubmitting;
		submitButton.innerHTML = isSubmitting
			? '<i data-lucide="loader-2"></i>Adding...'
			: '<i data-lucide="user-plus"></i>Add Student';
		if (window.lucide && typeof window.lucide.createIcons === 'function') {
			window.lucide.createIcons();
		}
	};

	form.addEventListener('submit', async (event) => {
		event.preventDefault();

		const formData = new FormData(form);
		const name = String(formData.get('name') || '').trim();
		const gender = String(formData.get('gender') || '').trim();

		if (!name || !gender) {
			setStatusMessage('Please complete the student name and gender fields.');
			return;
		}

		setSubmitting(true);
		setStatusMessage('Adding student...', true);

		try {
			const createdStudent = await createStudent(classId, { name, gender });
			showToast('success', 'Student added successfully.');
			window.location.hash = `#/classes/${classId}/students/${createdStudent.id}`;
		} catch (error) {
			setSubmitting(false);
			setStatusMessage(error.message || 'Failed to add student.');
			showToast('error', error.message || 'Failed to add student.');
		}
	});
}

function renderErrorState(appElement, error) {
	const pageContent = `
		<div class="dashboard-container">
			<section class="card classes-empty-state">
				<i data-lucide="alert-triangle"></i>
				<h3>Unable to load student form</h3>
				<p class="text-muted">${safeText(error.message || 'Please try again in a moment.')}</p>
				<a href="#/classes" class="btn btn-secondary">Back to Classes</a>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Add Student' });
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