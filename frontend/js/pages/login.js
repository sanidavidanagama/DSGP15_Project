// Login page — teacher authentication

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
						<h1>Welcome back</h1>
						<p class="text-muted text-sm">Sign in to access your dashboard and analyses.</p>

						<form class="auth-form" id="login-form">
							<div class="form-group">
								<label class="form-label" for="login-username">Username</label>
								<input id="login-username" name="username" class="form-input" type="text" autocomplete="username" required />
							</div>

							<div class="form-group">
								<label class="form-label" for="login-password">Password</label>
								<input id="login-password" name="password" class="form-input" type="password" autocomplete="current-password" required />
							</div>

							<button type="submit" class="btn btn-primary btn-full">Sign In</button>
						</form>

						<p class="auth-switch text-sm">
							Don't have an account?
							<a href="#/signup" class="text-teal">Create one</a>
						</p>
					</div>
				</div>
			</div>
		</div>
	`;
}

