// Simple application state utilities for INKIND
// Auth state is persisted in localStorage as specified in the frontend spec.

const AUTH_KEY = 'inkind_auth';

export function getAuth() {
	try {
		const raw = window.localStorage.getItem(AUTH_KEY);
		return raw ? JSON.parse(raw) : null;
	} catch (err) {
		console.error('Failed to read auth from localStorage', err);
		return null;
	}
}

export function setAuth(token, teacherId, teacherName) {
	const payload = { token, teacherId, teacherName };
	try {
		window.localStorage.setItem(AUTH_KEY, JSON.stringify(payload));
	} catch (err) {
		console.error('Failed to write auth to localStorage', err);
	}
}

export function clearAuth() {
	try {
		window.localStorage.removeItem(AUTH_KEY);
	} catch (err) {
		console.error('Failed to clear auth from localStorage', err);
	}
}

export function isAuthenticated() {
	const auth = getAuth();
	return Boolean(auth && auth.token);
}

// Basic in-memory UI state (can be extended as needed)
const uiState = {
	currentRoute: '/',
};

export function setCurrentRoute(path) {
	uiState.currentRoute = path;
}

export function getCurrentRoute() {
	return uiState.currentRoute;
}
