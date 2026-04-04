let activeModal = null;

export function openModal(content, { onClose } = {}) {
	closeModal();

	const backdrop = document.createElement('div');
	backdrop.className = 'modal-backdrop';
	backdrop.innerHTML = `
		<div class="modal" role="dialog" aria-modal="true">
			<div class="modal-header">
				<button type="button" class="modal-close" aria-label="Close modal">×</button>
			</div>
			<div class="modal-body">${content}</div>
		</div>
	`;

	const closeButton = backdrop.querySelector('.modal-close');
	const modal = backdrop.querySelector('.modal');

	const handleBackdropClick = (event) => {
		if (event.target === backdrop) {
			closeModal(onClose);
		}
	};

	const handleClose = () => closeModal(onClose);

	closeButton?.addEventListener('click', handleClose);
	backdrop.addEventListener('click', handleBackdropClick);
	document.addEventListener('keydown', handleEscape);
	document.body.appendChild(backdrop);
	activeModal = { backdrop, onClose, handleEscape };
	modal?.focus?.();
}

function handleEscape(event) {
	if (event.key === 'Escape') {
		closeModal();
	}
}

export function closeModal(onCloseOverride) {
	if (!activeModal) return;
	const { backdrop, onClose, handleEscape } = activeModal;
	document.removeEventListener('keydown', handleEscape);
	backdrop.remove();
	activeModal = null;
	if (typeof (onCloseOverride || onClose) === 'function') {
		(onCloseOverride || onClose)();
	}
}
