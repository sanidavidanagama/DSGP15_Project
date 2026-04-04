// Login page — teacher authentication
import { loginUser, getProfile } from '../api/auth.js';
import { setAuth } from '../utils/state.js';
import { showToast } from '../components/toast.js';
import { isNonEmpty } from '../utils/validators.js';

export async function renderPage() {
	const root = document.getElementById('page-root');
	if (!root) return;

	root.innerHTML = `
		<div class="auth-page">
			<div class="auth-wallpaper" aria-hidden="true">
				<img
					src="https://images.unsplash.com/photo-1513364776144-60967b0f800f?w=800"
					alt="Colorful art supplies"
				/>
				<div class="auth-wallpaper-overlay"></div>
				<div class="auth-wallpaper-text">
					<p class="auth-wallpaper-quote">
						"Every drawing is a sentence in a child's unspoken story."
					</p>
					<p class="auth-wallpaper-sub">
						INKIND helps you listen more closely — with evidence, not guesswork.
					</p>
				</div>
			</div>

			<div class="auth-form-panel">
				<div class="auth-form-inner">
					<a href="#/" class="auth-brand" aria-label="INKIND home">
						<img src="assets/inkind_logo.svg" alt="INKIND" class="auth-brand-logo" />
					</a>

					<h1 class="auth-heading">Welcome back</h1>
					<p class="auth-subheading">
						Sign in to access your dashboard and analyses.
					</p>

					<form class="auth-form" id="login-form" novalidate>
						<div class="form-group">
							<label class="form-label" for="login-username">Username</label>
							<input id="login-username" name="username" class="form-input" type="text" autocomplete="username" />
						</div>

						<div class="form-group">
							<label class="form-label" for="login-password">Password</label>
							<input id="login-password" name="password" class="form-input" type="password" autocomplete="current-password" />
							<p class="form-error text-sm" data-error-for="login-password" style="display:none;"></p>
						</div>

						<button type="submit" class="btn btn-primary btn-full">Sign In</button>
					</form>

					<p class="auth-subheading" style="margin-top: 16px; font-size: 0.875rem;">
						Don't have an account?
						<a href="#/signup" class="text-teal">Create one</a>
					</p>
				</div>
			</div>
		</div>
	`;

	// Attach form handler
	const form = document.getElementById('login-form');
	if (form) {
		form.addEventListener('submit', handleLoginSubmit);
	}
}

async function handleLoginSubmit(e) {
	e.preventDefault();

	const username = document.getElementById('login-username').value.trim();
	const password = document.getElementById('login-password').value.trim();

	// Validate inputs
	if (!isNonEmpty(username)) {
		showToast('error', 'Username is required');
		return;
	}

	if (!isNonEmpty(password)) {
		showToast('error', 'Password is required');
		return;
	}

	try {
		// Login and get token
		const loginResponse = await loginUser({ username, password });

		if (!loginResponse.access_token) {
			showToast('error', loginResponse.detail || 'Login failed');
			return;
		}

		// Store token immediately so getProfile() can use it
		setAuth(loginResponse.access_token, null, username);

		// Get profile info
		const profileResponse = await getProfile();

		// Update with full profile info
		setAuth(loginResponse.access_token, profileResponse.id, profileResponse.username);

		// Success
		showToast('success', 'Login successful!');

		// Redirect to dashboard after brief delay
		setTimeout(() => {
			window.location.hash = '#/dashboard';
		}, 300);
	} catch (error) {
		const errorMsg = error.message || 'An error occurred during login';
		showToast('error', errorMsg);
	}
}

