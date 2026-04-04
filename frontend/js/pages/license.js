// License page — public legal information

const PROJECT_METADATA = {
	name: 'INKIND',
	version: '0.9.0',
	description: "AI powered end-to-end system to analyze children's drawings and a full web application stack.",
	requiresPython: '3.11+',
};

const PROJECT_DEPENDENCIES = [
	'albumentations 2.0.8',
	'chromadb 1.4.1',
	'contourpy 1.3.3',
	'fastapi 0.122.0',
	'ipykernel 7.1.0',
	'jupyterlab 4.5.0',
	'kagglehub 0.3.13',
	'langchain-community 0.4.1',
	'langchain-google-genai 4.2.0',
	'langchain-text-splitters 1.1.0',
	'matplotlib 3.10.7',
	'numpy 2.3.5',
	'opencv-python 4.11.0.86',
	'pandas 2.3.3',
	'pillow 12.0.0',
	'pydantic 2.12.5',
	'pypdf 6.6.2',
	'python-dotenv 1.2.1',
	'python-multipart 0.0.20',
	'requests 2.32.5',
	'scikit-image 0.25.2',
	'scikit-learn 1.7.2',
	'scipy 1.16.3',
	'seaborn 0.13.2',
	'sentence-transformers 5.2.2',
	'shapely 2.1.2',
	'tensorflow 2.20.0',
	'tf-keras 2.20.1',
	'torch 2.2.0',
	'torchvision 0.17.0',
	'torchaudio 2.2.0',
	'tqdm 4.67.1',
	'transformers[torch] 4.40.0',
	'ultralytics 8.1,<9.0',
	'uvicorn 0.38.0',
	'sqlalchemy 2.0.48',
	'segment-anything 1.0',
	'pydantic-settings 2.13.1',
	'streamlit 1.55.0',
	'python-jose[cryptography] 3.5.0',
	'passlib[bcrypt] 1.7.4',
	'email-validator 2.3.0',
];

export async function renderPage() {
	const root = document.getElementById('page-root');
	if (!root) return;

	root.innerHTML = `
		<div class="full-page license-page">
			<div class="license-inner">
				<div class="license-back">
					<button type="button" class="btn btn-ghost text-sm" onclick="window.location.hash = '#/'">
						← Back to landing
					</button>
				</div>

				<section class="license-body">
					<h1>License</h1>
					<p class="license-date">Updated April 2026</p>

					<p>
						© 2026 INKIND · DSGP 15 | IIT · Robert Gordon University. All rights reserved.
						The copyright for this project and all its associated products resides with
						Informatics Institute of Technology.
					</p>

					<h2>Usage Terms</h2>
					<p>
						Permission is granted, free of charge, to use this software and associated
						documentation for academic review and demonstration purposes only.
					</p>
					<ul class="license-list">
						<li>Copyright and permission notices must be included in all copies or substantial portions of the software.</li>
						<li>Commercial use requires explicit written permission from Informatics Institute of Technology.</li>
					</ul>

					<h2>Warranty Disclaimer</h2>
					<p>
						The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability.
					</p>

					<h2>Project Metadata</h2>
					<div class="license-meta-grid">
						<div><span class="license-meta-label">Name</span><span class="license-meta-value">${safeText(PROJECT_METADATA.name)}</span></div>
						<div><span class="license-meta-label">Version</span><span class="license-meta-value">${safeText(PROJECT_METADATA.version)}</span></div>
						<div><span class="license-meta-label">Requires Python</span><span class="license-meta-value">${safeText(PROJECT_METADATA.requiresPython)}</span></div>
						<div><span class="license-meta-label">Description</span><span class="license-meta-value">${safeText(PROJECT_METADATA.description)}</span></div>
					</div>

					<h2>Project Dependencies (${PROJECT_DEPENDENCIES.length})</h2>
					<p class="text-sm text-muted" style="margin-bottom: 12px;">Below is the dependency list declared for this project.</p>
					<div class="license-deps-grid">
						${PROJECT_DEPENDENCIES.map((dep) => `<span class="license-dep-pill">${safeText(dep)}</span>`).join('')}
					</div>
				</section>
			</div>
		</div>
	`;
}

function safeText(value) {
	const text = value == null ? '' : String(value);
	return text
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;')
		.replaceAll("'", '&#39;');
}

