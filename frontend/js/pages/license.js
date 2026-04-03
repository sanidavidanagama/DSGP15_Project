// License page — public legal information

export async function renderPage() {
	const root = document.getElementById('page-root');
	if (!root) return;

	root.innerHTML = `
		<div class="full-page license-page">
			<div class="license-container card">
				<button type="button" class="btn btn-ghost text-sm" onclick="window.location.hash = '#/'">
					← Back to landing
				</button>
				<h1>License</h1>
				<p class="text-sm text-muted" style="margin-bottom: 16px;">
					© 2026 INKIND — DSGP 15 | IIT × Robert Gordon University. All rights reserved.
				</p>
				<p class="text-sm text-muted" style="margin-bottom: 24px;">
					The copyright for this project and all its associated products resides with
					Informatics Institute of Technology.
				</p>
				<div class="license-body text-sm">
					<p>
						Permission is hereby granted, free of charge, to any person obtaining a copy of
						this software and associated documentation files (the "Software"), to use the
						Software for academic review and demonstration purposes only, subject to the
						following conditions:
					</p>
					<ul style="margin-top: 12px; margin-left: 18px; list-style: disc;">
						<li>The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.</li>
						<li>The Software may not be used for commercial purposes without explicit written permission from Informatics Institute of Technology.</li>
					</ul>
					<p style="margin-top: 16px;">
						THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
						IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
						FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
						AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
						LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
						OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
						SOFTWARE.
					</p>
				</div>
			</div>
		</div>
	`;
}

