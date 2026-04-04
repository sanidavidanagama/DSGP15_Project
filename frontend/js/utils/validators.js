// Basic form validation helpers

export function isNonEmpty(value) {
	return typeof value === 'string' && value.trim().length > 0;
}

export function isValidEmail(value) {
	if (!isNonEmpty(value)) return false;
	// Simple email pattern sufficient for frontend validation
	const pattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
	return pattern.test(value.trim());
}

export function isValidPassword(value) {
	if (!isNonEmpty(value)) return false;
	// Minimum 8 characters for basic strength
	return value.trim().length >= 8;
}

