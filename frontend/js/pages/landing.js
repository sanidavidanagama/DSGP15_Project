// Landing page — public marketing page (no app shell)

export async function renderPage() {
	const root = document.getElementById('page-root');
	if (!root) return;

	root.innerHTML = `
		<div class="landing-page">
			<header class="landing-nav" id="landing-nav">
				<a href="#/" class="landing-nav-brand" aria-label="INKIND Teacher Portal">
					<img src="assets/inkind_logo.svg" alt="INKIND" class="landing-logo-img" />
				</a>
				<nav class="landing-nav-links">
					<button class="landing-nav-link" data-scroll-target="features">Features</button>
					<button class="landing-nav-link" data-scroll-target="how-it-works">How It Works</button>
					<button class="landing-nav-link" data-scroll-target="team">Team</button>
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
					<div class="landing-hero-inner">
						<div class="landing-hero-left">
							<div class="hero-badge">
								<i data-lucide="scan-eye" class="hero-badge-icon"></i>
								<span>AI-Powered Drawing Analysis</span>
							</div>
							<h1 class="hero-title">See what children's drawings quietly reveal.</h1>
							<p class="hero-subtitle">
								INKIND helps teachers interpret children's drawings through multimodal AI,
								surfacing emotional and developmental indicators in minutes.
							</p>
							<div class="hero-actions">
								<a href="#/signup" class="btn btn-primary">Get Started</a>
								<a href="#/login" class="btn btn-secondary">Sign In</a>
							</div>
							<p class="hero-trust">
								Used by educators · Non-diagnostic · Privacy-first
							</p>
						</div>

						<div class="landing-hero-right">
							<div class="hero-image-wrapper">
								<div class="hero-image-container">
									<img
										src="https://images.unsplash.com/photo-1503454537195-1dcabb73ffb9?w=1200"
										alt="Child drawing with crayons"
									/>
								</div>
								<div class="hero-sample-card">
									<div class="hero-sample-label">Sample analysis</div>
									<div class="hero-sample-bar">
										<div class="hero-sample-happy" style="width: 68%"></div>
										<div class="hero-sample-sad" style="width: 32%"></div>
									</div>
									<div class="hero-sample-labels">
										<span>Happy 68%</span>
										<span>Sad 32%</span>
									</div>
								</div>
							</div>
						</div>
					</div>
				</section>

				<!-- Features Section -->
				<section id="features" class="landing-section">
					<h2 class="landing-section-title">What INKIND Offers</h2>
					<p class="landing-section-subtitle">
						A clinical yet compassionate toolkit to make sense of children's drawings.
					</p>

					<div class="features-grid">
						<article class="feature-card">
							<div class="feature-icon-wrap">
								<i data-lucide="scan-eye"></i>
							</div>
							<h3 class="feature-title">Mood Detection</h3>
							<p class="feature-desc">
								Multimodal analysis combining drawing imagery and teacher-provided context to identify emotional patterns.
							</p>
						</article>

						<article class="feature-card">
							<div class="feature-icon-wrap">
								<i data-lucide="trending-up"></i>
							</div>
							<h3 class="feature-title">Longitudinal View</h3>
							<p class="feature-desc">
								Track mood and drawing indicators across time to spot meaningful shifts, not one-off anomalies.
							</p>
						</article>

						<article class="feature-card">
							<div class="feature-icon-wrap">
								<i data-lucide="users"></i>
							</div>
							<h3 class="feature-title">Classroom-Level Insights</h3>
							<p class="feature-desc">
								See patterns at the class level so you can adapt group activities and environments with confidence.
							</p>
						</article>

						<article class="feature-card">
							<div class="feature-icon-wrap">
								<i data-lucide="check-circle"></i>
							</div>
							<h3 class="feature-title">Teacher-Friendly Reports</h3>
							<p class="feature-desc">
								Clear, non-technical language that highlights what to notice, what to celebrate, and what to monitor.
							</p>
						</article>

						<article class="feature-card">
							<div class="feature-icon-wrap">
								<i data-lucide="shield"></i>
							</div>
							<h3 class="feature-title">Non-Diagnostic Framing</h3>
							<p class="feature-desc">
								Built to support, not label. INKIND offers indicators and suggestions, never clinical diagnoses.
							</p>
						</article>

						<article class="feature-card">
							<div class="feature-icon-wrap">
								<i data-lucide="lock"></i>
							</div>
							<h3 class="feature-title">Privacy-First</h3>
							<p class="feature-desc">
								No PII collected. Drawings processed securely. Results framed as supportive, never diagnostic.
							</p>
						</article>
					</div>
				</section>

				<!-- How It Works Section -->
				<section id="how-it-works" class="how-section">
					<div class="how-section-inner">
						<h2 class="landing-section-title">How It Works</h2>
						<p class="landing-section-subtitle">
							A four-step, teacher-centered workflow that fits into your day.
						</p>

						<div class="steps-row">
							<div class="step-item">
								<div class="step-circle">1</div>
								<h3 class="step-title">Upload</h3>
								<p class="step-desc">
									Teacher uploads a drawing plus optional classroom context notes.
								</p>
							</div>
							<div class="step-connector"></div>
							<div class="step-item">
								<div class="step-circle">2</div>
								<h3 class="step-title">Analyze</h3>
								<p class="step-desc">
									AI models evaluate colors, shapes, spatial layout, and textual cues.
								</p>
							</div>
							<div class="step-connector"></div>
							<div class="step-item">
								<div class="step-circle">3</div>
								<h3 class="step-title">Interpret</h3>
								<p class="step-desc">
									Insights are translated into teacher-friendly language with clear guardrails.
								</p>
							</div>
							<div class="step-connector"></div>
							<div class="step-item">
								<div class="step-circle">4</div>
								<h3 class="step-title">Report</h3>
								<p class="step-desc">
									Integrated report with indicators, mood summary, and recommendations.
								</p>
							</div>
						</div>
					</div>
				</section>

				<!-- Team Section -->
				<section id="team" class="landing-section">
					<h2 class="landing-section-title">Built by Group 15</h2>
					<p class="landing-section-subtitle">
						A Data Science Group Project at IIT in collaboration with Robert Gordon University.
					</p>

					<div class="team-cards">
						<article class="team-card">
							<div class="team-avatar">SV</div>
							<div class="team-name">Sanida Vidanagama</div>
							<div class="team-role">Project Manager, AI Engineer & Full Stack Developer</div>
						</article>

						<article class="team-card">
							<div class="team-avatar">LR</div>
							<div class="team-name">Lidiya Rajapaksha</div>
							<div class="team-role">Full Stack Developer & ML Engineer</div>
						</article>

						<article class="team-card">
							<div class="team-avatar">SD</div>
							<div class="team-name">Sanuli Dhanuge</div>
							<div class="team-role">Data Engineer & Backend Developer</div>
						</article>

						<article class="team-card">
							<div class="team-avatar">KR</div>
							<div class="team-name">Kaviyan Ratneswaran</div>
							<div class="team-role">QA Engineer & Logic Developer</div>
						</article>
					</div>

					<p class="team-supervisor">Supervised by Mr. Prashan Rathnayaka</p>
				</section>

				<!-- CTA Section -->
				<section class="cta-section">
					<h2>Ready to see what children's drawings reveal?</h2>
					<p>
						Start with a single drawing and build a longitudinal picture of your classroom's emotional world.
					</p>
					<a href="#/signup" class="btn-cta">
						<span>Get Started</span>
						<i data-lucide="arrow-right"></i>
					</a>
				</section>
			</main>

			<footer class="landing-footer">
				<span class="landing-footer-copy">
					© 2026 INKIND — DSGP 15 | IIT × Robert Gordon University
				</span>
				<div class="landing-footer-links">
					<a href="#/license">License</a>
					<a href="https://github.com/sanidavidanagama/DSGP15_Project" target="_blank" rel="noopener noreferrer">GitHub</a>
				</div>
			</footer>
		</div>
	`;

	// Scroll behavior for navbar background
	const nav = document.getElementById('landing-nav');
	const onScroll = () => {
		if (!nav) return;
		const threshold = 50;
		if (window.scrollY > threshold) {
			nav.classList.add('scrolled');
		} else {
			nav.classList.remove('scrolled');
		}
	};

	window.addEventListener('scroll', onScroll, { passive: true });
	onScroll();

	// Smooth scroll for in-page nav links
	const scrollButtons = root.querySelectorAll('[data-scroll-target]');
	scrollButtons.forEach((btn) => {
		btn.addEventListener('click', (event) => {
			event.preventDefault();
			const targetId = btn.getAttribute('data-scroll-target');
			const section = targetId ? document.getElementById(targetId) : null;
			if (section) {
				const offset = nav ? nav.offsetHeight + 16 : 80;
				const rect = section.getBoundingClientRect();
				const top = rect.top + window.scrollY - offset;
				window.scrollTo({ top, behavior: 'smooth' });
			}
		});
	});
}

