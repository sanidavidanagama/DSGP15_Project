// Signup page — teacher registration

export async function renderPage() {
	const root = document.getElementById('page-root');
	if (!root) return;

	root.innerHTML = `
		<div class="auth-page">
			<div class="auth-grid">
				<div class="auth-visual" aria-hidden="true">
					<div class="auth-visual-overlay"></div>
					<img
						src="https://images.unsplash.com/photo-1513364776144-60967b0f800f?w=800"
						alt="Colorful art supplies"
						class="auth-visual-image"
					/>
				</div>

				<div class="auth-panel">
					<div class="auth-panel-inner">
						<div class="auth-logo">
							<div class="landing-logo-mark">
								<span class="landing-logo-icon">IN</span>
							</div>
							<span class="landing-logo-title">INKIND</span>
						</div>
						<h1>Create your account</h1>
						<p class="text-muted text-sm">Set up your teacher portal to begin analyzing drawings.</p>

						<form class="auth-form" id="signup-form">
							<div class="form-group">
								<label class="form-label" for="signup-username">Username</label>
								<input id="signup-username" name="username" class="form-input" type="text" autocomplete="username" required />
							</div>

							<div class="form-group">
								<label class="form-label" for="signup-email">Email Address</label>
								<input id="signup-email" name="email" class="form-input" type="email" autocomplete="email" required />
							</div>

							<div class="form-group">
								<label class="form-label" for="signup-password">Password</label>
								<input id="signup-password" name="password" class="form-input" type="password" autocomplete="new-password" required />
							</div>

							<button type="submit" class="btn btn-primary btn-full">Create Account</button>
						</form>

						<p class="auth-switch text-sm">
							Already have an account?
							<a href="#/login" class="text-teal">Back to login</a>
						</p>
					</div>
				</div>
			</div>
		</div>
	`;
}

