// Signup page — teacher registration

import { registerUser } from '../api/auth.js';
import { showToast } from '../components/toast.js';
import { isNonEmpty, isValidEmail, isValidPassword } from '../utils/validators.js';

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
					<div class="auth-brand">
						<img src="assets/logo.svg" alt="INKIND" class="auth-brand-logo" />
						<div class="auth-brand-text">
							<span class="auth-brand-name">INKIND</span>
							<span class="auth-brand-sub">Teacher Portal</span>
						</div>
					</div>

					<h1 class="auth-heading">Create your account</h1>
					<p class="auth-subheading">
						Set up your teacher portal to begin analyzing drawings.
					</p>

					<form class="auth-form" id="signup-form" novalidate>
						<div class="form-group">
							<label class="form-label" for="signup-username">Username</label>
							<input id="signup-username" name="username" class="form-input" type="text" autocomplete="username" />
						</div>

						<div class="form-group">
							<label class="form-label" for="signup-email">Email Address</label>
							<input id="signup-email" name="email" class="form-input" type="email" autocomplete="email" />
						</div>

						<div class="form-group">
							<label class="form-label" for="signup-password">Password</label>
							<input id="signup-password" name="password" class="form-input" type="password" autocomplete="new-password" />
							<p class="form-error text-sm" data-error-for="signup-password" style="display:none;"></p>
						</div>

						<button type="submit" class="btn btn-primary btn-full">Create Account</button>
					</form>

					<p class="auth-subheading" style="margin-top: 16px; font-size: 0.875rem;">
						Already have an account?
						<a href="#/login" class="text-teal">Back to login</a>
					</p>
				</div>
			</div>
		</div>
	`;

	const form = document.getElementById('signup-form');
	if (!form) return;

	form.addEventListener('submit', async (event) => {
		event.preventDefault();

		const usernameInput = document.getElementById('signup-username');
		const emailInput = document.getElementById('signup-email');
		const passwordInput = document.getElementById('signup-password');
		const passwordError = form.querySelector('[data-error-for="signup-password"]');

		if (!usernameInput || !emailInput || !passwordInput) return;

		const username = usernameInput.value;
		const email = emailInput.value;
		const password = passwordInput.value;

		// Reset error state
		if (passwordError) {
			passwordError.style.display = 'none';
			passwordError.textContent = '';
		}

		if (!isNonEmpty(username) || !isValidEmail(email) || !isValidPassword(password)) {
			if (!isValidPassword(password) && passwordError) {
				passwordError.textContent = 'Password must be at least 8 characters long.';
				passwordError.style.display = 'block';
			}
			showToast('error', 'Please provide a username, valid email, and strong password.');
			return;
		}

		try {
			await registerUser({ username: username.trim(), email: email.trim(), password });
			showToast('success', 'Account created successfully. You can now sign in.');
			window.location.hash = '#/login';
		} catch (error) {
			showToast('error', error.message || 'Failed to create account. Please try again.');
		}
	});
}


