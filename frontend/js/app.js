// Main application bootstrap for INKIND
// - Defines the route table
// - Wires the Router
// - Handles simple auth-guard logic

import { Router } from './router.js';
import { isAuthenticated, setCurrentRoute } from './utils/state.js';

const ROUTES = {
	'/':                    { page: 'landing',           auth: false, shell: false },
	'/login':               { page: 'login',             auth: false, shell: false },
	'/signup':              { page: 'signup',            auth: false, shell: false },
	'/license':             { page: 'license',           auth: false, shell: false },

	'/dashboard':           { page: 'dashboard',         auth: true,  shell: true },
	'/classes':             { page: 'classes',           auth: true,  shell: true },
	'/students':            { page: 'students',          auth: true,  shell: true },
	'/settings':            { page: 'settings',          auth: true,  shell: true },
	'/classes/new':         { page: 'add-class',         auth: true,  shell: true },
	'/classes/:classId/students/new': { page: 'add-student', auth: true, shell: true },
	'/classes/:id':         { page: 'class-detail',      auth: true,  shell: true },
	'/classes/:id/edit':    { page: 'edit-class',        auth: true,  shell: true },

	'/classes/:classId/students/:studentId': {
		page: 'student-profile',
		auth: true,
		shell: true,
	},

	'/analysis':            { page: 'analysis',          auth: true,  shell: true },
	'/analysis/loading':    { page: 'analysis-loading',  auth: true,  shell: true },
	'/analysis/report':     { page: 'analysis-report',   auth: true,  shell: true },
};

const appElement = document.getElementById('app');

function renderPageContainer({ shell }) {
	if (!appElement) return;

	// For now, we render public pages as full-page layouts without the app shell.
	// The authenticated shell (sidebar + content) will be wired in a later step.
	if (!shell) {
		appElement.innerHTML = '<div class="full-page" id="page-root"></div>';
	} else {
		appElement.innerHTML = '<div class="full-page" id="page-root"></div>';
	}
}

async function loadPageModule(pageName) {
	try {
		const module = await import(`./pages/${pageName}.js`);
		if (module && typeof module.renderPage === 'function') {
			return module.renderPage;
		}
	} catch (err) {
		console.error(`Failed to load page module: ${pageName}`, err);
	}
	return null;
}

async function handleRouteChange(route, params, path) {
	const authed = isAuthenticated();

	// Auth guard: redirect to login if route requires auth
	if (route.auth && !authed) {
		window.location.hash = '#/login';
		return;
	}

	// If authenticated user hits login/signup, send them to dashboard
	if (!route.auth && authed && (path === '/login' || path === '/signup')) {
		window.location.hash = '#/dashboard';
		return;
	}

	setCurrentRoute(path);
	renderPageContainer({ shell: route.shell });

	const pageRoot = document.getElementById('page-root');
	if (!pageRoot) return;

	pageRoot.classList.add('page-enter');

	const renderPage = await loadPageModule(route.page);
	if (renderPage) {
		await renderPage({ params, route, path });
	} else {
		pageRoot.innerHTML = `<div style="padding:24px">Missing view for <code>${route.page}</code></div>`;
	}

	// Refresh Lucide icons after each route render
	if (window.lucide && typeof window.lucide.createIcons === 'function') {
		window.lucide.createIcons();
	}
}

// Initialize router
const router = new Router(ROUTES, handleRouteChange);

// Expose for debugging in the console
window.__inkindRouter = router;

