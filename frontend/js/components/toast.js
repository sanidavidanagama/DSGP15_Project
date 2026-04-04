// Simple toast notification system

const TOAST_DURATION_MS = 4000;

export function showToast(type, message) {
	const container = getToastContainer();
	const toast = document.createElement('div');
	toast.className = `toast toast-${type}`;

	const messageEl = document.createElement('div');
	messageEl.className = 'toast-message';
	messageEl.textContent = message;

	toast.appendChild(messageEl);
	container.appendChild(toast);

	setTimeout(() => {
		toast.classList.add('toast-exit');
		toast.addEventListener('animationend', () => {
			toast.remove();
		});
	}, TOAST_DURATION_MS);
}

function getToastContainer() {
	let container = document.getElementById('toast-container');
	if (!container) {
		container = document.createElement('div');
		container.id = 'toast-container';
		document.body.appendChild(container);
	}
	return container;
}

