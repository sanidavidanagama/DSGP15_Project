// Landing page — public marketing page (no app shell)

export async function renderPage() {
	const root = document.getElementById('page-root');
	if (!root) return;

	root.innerHTML = `
		<div class="landing-page">
			<header class="landing-nav">
				<div class="landing-nav-left">
					<div class="landing-logo-mark">
						<span class="landing-logo-icon">IN</span>
					</div>
					<div class="landing-logo-text">
						<span class="landing-logo-title">INKIND</span>
					</div>
				</div>
				<nav class="landing-nav-links">
					<a href="#/" class="landing-nav-link" data-scroll="features">Features</a>
					<a href="#/" class="landing-nav-link" data-scroll="how-it-works">How It Works</a>
					<a href="#/" class="landing-nav-link" data-scroll="team">Team</a>
					<a href="#/license" class="landing-nav-link">License</a>
				</nav>
				<div class="landing-nav-actions">
					<a href="#/login" class="btn btn-ghost">Sign In</a>
					<a href="#/signup" class="btn btn-primary">Get Started</a>
				</div>
			</header>

			<main class="landing-main">
				<!-- Hero Section -->
				<section class="landing-hero" id="hero">
					<div class="landing-hero-content">
						<div class="badge badge-teal text-xs">AI-Powered Drawing Analysis</div>
						<h1>See what children's drawings quietly reveal.</h1>
						<p class="landing-hero-subtitle">
							INKIND helps teachers interpret children's drawings through multimodal AI,
							surfacing emotional and developmental indicators in minutes.
						</p>
						<div class="landing-hero-actions">
							<a href="#/signup" class="btn btn-primary">Get Started</a>
							<a href="#/login" class="btn btn-secondary">Sign In</a>
						</div>
						<p class="landing-hero-trust text-sm text-muted">
							Used by educators · Non-diagnostic · Privacy-first
						</p>
					</div>
					<div class="landing-hero-visual">
						<div class="landing-hero-image-wrapper">
							<img
								src="https://images.unsplash.com/photo-1503454537195-1dcabb73ffb9?w=1200"
								alt="Child drawing with crayons"
								class="landing-hero-image"
							/>
							<div class="glass-card landing-hero-sample-card">
								<div class="landing-hero-sample-header">
									<span class="text-sm text-muted">Sample Analysis</span>
									<span class="badge badge-success text-xs">Balanced Mood</span>
								</div>
								<div class="landing-hero-sample-mood">
									<span class="text-xs text-muted">Mood Balance</span>
									<div class="landing-hero-mood-bar">
										<div class="landing-hero-mood-happy" style="width: 68%"></div>
										<div class="landing-hero-mood-sad" style="width: 32%"></div>
									</div>
									<div class="landing-hero-mood-labels text-xs">
										<span>Happy 68%</span>
										<span>Sad 32%</span>
									</div>
								</div>
							</div>
						</div>
					</div>
				</section>

				<!-- Placeholder sections to be expanded in later steps -->
				<section id="features" class="landing-section">
					<h2>What INKIND Offers</h2>
					<p class="text-muted">
						Detailed feature cards will be implemented next so you can see the full
						storytelling of the product.
					</p>
				</section>

				<section id="how-it-works" class="landing-section">
					<h2>How It Works</h2>
					<p class="text-muted">We will add the full 4-step visual flow in a follow-up step.</p>
				</section>

				<section id="team" class="landing-section">
					<h2>Built by Group 15</h2>
					<p class="text-muted">
						Team member cards and supervisor details will be added in detail shortly.
					</p>
				</section>
			</main>

			<footer class="landing-footer">
				<span class="text-sm text-muted">
					© 2026 INKIND — DSGP 15 | IIT × Robert Gordon University
				</span>
				<div class="landing-footer-links text-sm">
					<a href="#/license">License</a>
					<a href="https://github.com/sanidavidanagama/DSGP15_Project" target="_blank" rel="noopener noreferrer">GitHub</a>
				</div>
			</footer>
		</div>
	`;
}

