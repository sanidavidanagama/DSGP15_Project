import { createShell, attachShellHandlers } from '../components/shell.js';
import { showToast } from '../components/toast.js';
import {
	fetchMySettingsProfile,
	updateMyProfile,
	changeMyPassword,
	deleteMyData,
	deleteMyAccount,
} from '../api/settings.js';
import { getAuth, setAuth, clearAuth } from '../utils/state.js';

export async function renderPage() {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	try {
		const profile = await fetchMySettingsProfile();
		renderSettingsView(appElement, profile || {});
	} catch (error) {
		showToast('error', error.message || 'Failed to load settings.');
		renderErrorState(appElement, error);
	}
}

function renderSettingsView(appElement, profile) {
	const email = safeText(profile.email || '');
	const username = safeAttr(profile.username || '');

	const pageContent = `
		<div class="dashboard-container settings-page">
			<section class="card settings-header-card">
				<h2>Settings</h2>
				<p class="text-muted" style="margin-top: 8px;">Manage your profile, password, and account data controls.</p>
			</section>

			<div class="settings-grid">
				<section class="card settings-card">
					<div class="settings-card-head">
						<h3>Edit Profile</h3>
						<p class="text-muted text-sm">Change your username. Email is read-only.</p>
					</div>
					<form id="settings-profile-form" class="settings-form">
						<div class="form-group">
							<label class="form-label" for="settings-email">Email</label>
							<input id="settings-email" class="form-input" type="email" value="${email}" disabled />
						</div>
						<div class="form-group">
							<label class="form-label" for="settings-username">Username</label>
							<input id="settings-username" name="username" class="form-input" type="text" value="${username}" required />
						</div>
						<p id="settings-profile-status" class="text-muted class-form-status" aria-live="polite"></p>
						<div class="settings-actions">
							<button id="settings-profile-submit" class="btn btn-primary" type="submit">
								<i data-lucide="save"></i>
								Save Profile
							</button>
						</div>
					</form>
				</section>

				<section class="card settings-card">
					<div class="settings-card-head">
						<h3>Change Password</h3>
						<p class="text-muted text-sm">Use your current password to set a new one.</p>
					</div>
					<form id="settings-password-form" class="settings-form">
						<div class="form-group">
							<label class="form-label" for="settings-current-password">Current Password</label>
							<input id="settings-current-password" name="current_password" class="form-input" type="password" required />
						</div>
						<div class="form-group">
							<label class="form-label" for="settings-new-password">New Password</label>
							<input id="settings-new-password" name="new_password" class="form-input" type="password" minlength="8" required />
						</div>
						<div class="form-group">
							<label class="form-label" for="settings-confirm-password">Confirm New Password</label>
							<input id="settings-confirm-password" name="confirm_password" class="form-input" type="password" minlength="8" required />
						</div>
						<p id="settings-password-status" class="text-muted class-form-status" aria-live="polite"></p>
						<div class="settings-actions">
							<button id="settings-password-submit" class="btn btn-primary" type="submit">
								<i data-lucide="key-round"></i>
								Update Password
							</button>
						</div>
					</form>
				</section>
			</div>

			<section class="card settings-card settings-danger-zone">
				<div class="settings-card-head">
					<h3>Delete Data & Account</h3>
					<p class="text-muted text-sm">Danger zone actions are permanent. Please confirm with your password.</p>
				</div>

				<div class="settings-danger-grid">
					<form id="settings-delete-data-form" class="settings-danger-item">
						<h4>Delete All My Data</h4>
						<p class="text-muted text-sm">Deletes your classes, students, and saved analyses. Your account remains.</p>
						<div class="form-group">
							<label class="form-label" for="settings-delete-data-password">Current Password</label>
							<input id="settings-delete-data-password" name="current_password" class="form-input" type="password" required />
						</div>
						<p id="settings-delete-data-status" class="text-muted class-form-status" aria-live="polite"></p>
						<div class="settings-actions">
							<button id="settings-delete-data-submit" class="btn btn-danger" type="submit">
								<i data-lucide="database-zap"></i>
								Delete My Data
							</button>
						</div>
					</form>

					<form id="settings-delete-account-form" class="settings-danger-item">
						<h4>Delete Profile</h4>
						<p class="text-muted text-sm">Deletes your account and all associated data permanently.</p>
						<div class="form-group">
							<label class="form-label" for="settings-delete-account-password">Current Password</label>
							<input id="settings-delete-account-password" name="current_password" class="form-input" type="password" required />
						</div>
						<p id="settings-delete-account-status" class="text-muted class-form-status" aria-live="polite"></p>
						<div class="settings-actions">
							<button id="settings-delete-account-submit" class="btn btn-danger" type="submit">
								<i data-lucide="user-x"></i>
								Delete Profile
							</button>
						</div>
					</form>
				</div>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Settings' });
	attachShellHandlers();
	bindSettingsActions();

	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function bindSettingsActions() {
	bindProfileForm();
	bindPasswordForm();
	bindDeleteDataForm();
	bindDeleteAccountForm();
}

function bindProfileForm() {
	const form = document.getElementById('settings-profile-form');
	const submitButton = document.getElementById('settings-profile-submit');
	const statusMessage = document.getElementById('settings-profile-status');
	if (!form || !submitButton || !statusMessage) return;

	form.addEventListener('submit', async (event) => {
		event.preventDefault();
		const username = String(new FormData(form).get('username') || '').trim();

		if (!username) {
			setStatus(statusMessage, 'Username is required.');
			return;
		}

		setLoading(submitButton, true, '<i data-lucide="save"></i>Save Profile', '<i data-lucide="loader-2"></i>Saving...');
		setStatus(statusMessage, 'Saving profile...');

		try {
			const updated = await updateMyProfile({ username });
			const auth = getAuth();
			if (auth?.token) {
				setAuth(auth.token, auth.teacherId || updated.email, updated.username || username);
			}
			setStatus(statusMessage, 'Profile updated successfully.');
			showToast('success', 'Profile updated successfully.');
		} catch (error) {
			setStatus(statusMessage, error.message || 'Failed to update profile.');
			showToast('error', error.message || 'Failed to update profile.');
		} finally {
			setLoading(submitButton, false, '<i data-lucide="save"></i>Save Profile', '<i data-lucide="loader-2"></i>Saving...');
		}
	});
}

function bindPasswordForm() {
	const form = document.getElementById('settings-password-form');
	const submitButton = document.getElementById('settings-password-submit');
	const statusMessage = document.getElementById('settings-password-status');
	if (!form || !submitButton || !statusMessage) return;

	form.addEventListener('submit', async (event) => {
		event.preventDefault();
		const formData = new FormData(form);
		const currentPassword = String(formData.get('current_password') || '');
		const newPassword = String(formData.get('new_password') || '');
		const confirmPassword = String(formData.get('confirm_password') || '');

		if (!currentPassword || !newPassword || !confirmPassword) {
			setStatus(statusMessage, 'Please fill in all password fields.');
			return;
		}

		if (newPassword.length < 8) {
			setStatus(statusMessage, 'New password must be at least 8 characters.');
			return;
		}

		if (newPassword !== confirmPassword) {
			setStatus(statusMessage, 'New password and confirmation do not match.');
			return;
		}

		setLoading(submitButton, true, '<i data-lucide="key-round"></i>Update Password', '<i data-lucide="loader-2"></i>Updating...');
		setStatus(statusMessage, 'Updating password...');

		try {
			await changeMyPassword({
				current_password: currentPassword,
				new_password: newPassword,
			});
			form.reset();
			setStatus(statusMessage, 'Password updated successfully.');
			showToast('success', 'Password updated successfully.');
		} catch (error) {
			setStatus(statusMessage, error.message || 'Failed to update password.');
			showToast('error', error.message || 'Failed to update password.');
		} finally {
			setLoading(submitButton, false, '<i data-lucide="key-round"></i>Update Password', '<i data-lucide="loader-2"></i>Updating...');
		}
	});
}

function bindDeleteDataForm() {
	const form = document.getElementById('settings-delete-data-form');
	const submitButton = document.getElementById('settings-delete-data-submit');
	const statusMessage = document.getElementById('settings-delete-data-status');
	if (!form || !submitButton || !statusMessage) return;

	form.addEventListener('submit', async (event) => {
		event.preventDefault();
		const password = String(new FormData(form).get('current_password') || '');

		if (!password) {
			setStatus(statusMessage, 'Current password is required.');
			return;
		}

		const confirmed = window.confirm('Delete ALL your classes, students, and saved analyses? This action cannot be undone.');
		if (!confirmed) return;

		setLoading(submitButton, true, '<i data-lucide="database-zap"></i>Delete My Data', '<i data-lucide="loader-2"></i>Deleting...');
		setStatus(statusMessage, 'Deleting your data...');

		try {
			const result = await deleteMyData({ current_password: password });
			form.reset();
			const summary = `Deleted ${result.deleted_classes} classes, ${result.deleted_students} students, and ${result.deleted_saved_analyses} analyses.`;
			setStatus(statusMessage, summary);
			showToast('success', 'All data deleted successfully.');
		} catch (error) {
			setStatus(statusMessage, error.message || 'Failed to delete data.');
			showToast('error', error.message || 'Failed to delete data.');
		} finally {
			setLoading(submitButton, false, '<i data-lucide="database-zap"></i>Delete My Data', '<i data-lucide="loader-2"></i>Deleting...');
		}
	});
}

function bindDeleteAccountForm() {
	const form = document.getElementById('settings-delete-account-form');
	const submitButton = document.getElementById('settings-delete-account-submit');
	const statusMessage = document.getElementById('settings-delete-account-status');
	if (!form || !submitButton || !statusMessage) return;

	form.addEventListener('submit', async (event) => {
		event.preventDefault();
		const password = String(new FormData(form).get('current_password') || '');

		if (!password) {
			setStatus(statusMessage, 'Current password is required.');
			return;
		}

		const confirmed = window.confirm('Delete your profile and all data permanently? This action cannot be undone.');
		if (!confirmed) return;

		setLoading(submitButton, true, '<i data-lucide="user-x"></i>Delete Profile', '<i data-lucide="loader-2"></i>Deleting...');
		setStatus(statusMessage, 'Deleting profile...');

		try {
			await deleteMyAccount({ current_password: password });
			showToast('success', 'Profile deleted successfully.');
			clearAuth();
			window.location.hash = '#/login';
		} catch (error) {
			setStatus(statusMessage, error.message || 'Failed to delete profile.');
			showToast('error', error.message || 'Failed to delete profile.');
			setLoading(submitButton, false, '<i data-lucide="user-x"></i>Delete Profile', '<i data-lucide="loader-2"></i>Deleting...');
		}
	});
}

function setStatus(element, message) {
	element.textContent = message;
	element.style.display = message ? 'block' : 'none';
}

function setLoading(button, loading, idleHtml, loadingHtml) {
	button.disabled = loading;
	button.innerHTML = loading ? loadingHtml : idleHtml;
	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

function renderErrorState(appElement, error) {
	const pageContent = `
		<div class="dashboard-container">
			<section class="card classes-empty-state">
				<i data-lucide="alert-triangle"></i>
				<h3>Unable to load settings</h3>
				<p class="text-muted">${safeText(error.message || 'Please try again in a moment.')}</p>
				<a href="#/dashboard" class="btn btn-secondary">Back to Dashboard</a>
			</section>
		</div>
	`;

	appElement.innerHTML = createShell(pageContent, { topbarTitle: 'Settings' });
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
