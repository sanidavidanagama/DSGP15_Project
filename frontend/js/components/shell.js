// Authenticated app shell — sidebar + topbar + main content
import { getAuth, clearAuth } from '../utils/state.js';
import { showToast } from './toast.js';

export function createShell(pageContent, options = {}) {
	const auth = getAuth();
	if (!auth?.token) {
		window.location.hash = '#/login';
		return '';
	}

	const teacherName = auth.teacherName || 'Teacher';
	const topbarTitle = options.topbarTitle || 'Dashboard';

	return `
		<div class="app-shell">
			<!-- Sidebar -->
			<aside class="sidebar">
				<div class="sidebar-header">
					<a href="#/dashboard" class="sidebar-brand">
						<img src="assets/inkind_logo.svg" alt="INKIND" class="sidebar-logo" />
					</a>
				</div>

				<nav class="sidebar-nav">
					<a href="#/dashboard" class="sidebar-nav-item" data-route="dashboard">
						<i data-lucide="home" class="sidebar-nav-icon"></i>
						<span>Dashboard</span>
					</a>
					<a href="#/classes" class="sidebar-nav-item" data-route="classes">
						<i data-lucide="book-open" class="sidebar-nav-icon"></i>
						<span>Classes</span>
					</a>
					<a href="#/analysis" class="sidebar-nav-item" data-route="analysis">
						<i data-lucide="image" class="sidebar-nav-icon"></i>
						<span>Analysis</span>
					</a>
					<a href="#/students" class="sidebar-nav-item" data-route="students">
						<i data-lucide="users" class="sidebar-nav-icon"></i>
						<span>Students</span>
					</a>
					<a href="#/settings" class="sidebar-nav-item" data-route="settings">
						<i data-lucide="settings" class="sidebar-nav-icon"></i>
						<span>Settings</span>
					</a>
				</nav>

				<div class="sidebar-footer">
					<button class="sidebar-logout-btn" id="logout-btn">
						<i data-lucide="log-out" class="sidebar-nav-icon"></i>
						<span>Log Out</span>
					</button>
				</div>
			</aside>

			<!-- Main Content -->
			<main class="main-content">
				<!-- Topbar -->
				<header class="topbar">
					<div class="topbar-left">
						<h1 class="topbar-title">${topbarTitle}</h1>
					</div>
					<div class="topbar-right">
						<div class="topbar-user">
							<div class="topbar-user-avatar">${teacherName.charAt(0)}</div>
							<span class="topbar-user-name">${teacherName}</span>
						</div>
					</div>
				</header>

				<!-- Page Content -->
				<div class="page-content">
					${pageContent}
				</div>
			</main>
		</div>
	`;
}

// Attach shell event handlers
export function attachShellHandlers() {
	const logoutBtn = document.getElementById('logout-btn');
	if (logoutBtn) {
		logoutBtn.addEventListener('click', () => {
			clearAuth();
			showToast('success', 'Logged out successfully');
			setTimeout(() => {
				window.location.hash = '#/login';
			}, 300);
		});
	}

	// Highlight active nav item based on current route
	const currentHash = window.location.hash.slice(2).split('/')[0];
	document.querySelectorAll('.sidebar-nav-item').forEach((item) => {
		const itemRoute = item.dataset.route;
		if (itemRoute === currentHash || (currentHash === '' && itemRoute === 'dashboard')) {
			item.classList.add('active');
		} else {
			item.classList.remove('active');
		}
	});
}
