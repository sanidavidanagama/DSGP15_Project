// Dashboard page — overview metrics + recent activity
import { createShell, attachShellHandlers } from '../components/shell.js';
import { apiFetch } from '../api/client.js';

export async function renderPage() {
	const appElement = document.getElementById('app');
	if (!appElement) return;

	try {
		// Fetch dashboard data
		const dashboardData = await apiFetch('/dashboard/overview', { method: 'GET' });

		const {
			total_students = 0,
			total_classes = 0,
			total_analyses = 0,
			recent_activity = [],
		} = dashboardData || {};

		const pageContent = `
			<div class="dashboard-container">
				<!-- Metrics Grid -->
				<div class="metrics-grid">
					<div class="metric-card glass">
						<div class="metric-icon" style="color: #008080;">
							<i data-lucide="users"></i>
						</div>
						<div class="metric-content">
							<p class="metric-label">Total Students</p>
							<p class="metric-value">${total_students}</p>
						</div>
					</div>

					<div class="metric-card glass">
						<div class="metric-icon" style="color: #008080;">
							<i data-lucide="book-open"></i>
						</div>
						<div class="metric-content">
							<p class="metric-label">Total Classes</p>
							<p class="metric-value">${total_classes}</p>
						</div>
					</div>

					<div class="metric-card glass">
						<div class="metric-icon" style="color: #008080;">
							<i data-lucide="image"></i>
						</div>
						<div class="metric-content">
							<p class="metric-label">Total Analyses</p>
							<p class="metric-value">${total_analyses}</p>
						</div>
					</div>

					<div class="metric-card glass">
						<div class="metric-icon" style="color: #008080;">
							<i data-lucide="zap"></i>
						</div>
						<div class="metric-content">
							<p class="metric-label">This Week</p>
							<p class="metric-value">${recent_activity.length}</p>
						</div>
					</div>
				</div>

				<!-- Quick Actions -->
				<div class="dashboard-section">
					<h2 class="dashboard-section-title">Quick Actions</h2>
					<div class="quick-actions-grid">
						<a href="#/classes" class="quick-action-btn glass">
							<i data-lucide="plus-circle"></i>
							<span>New Class</span>
						</a>
						<a href="#/students" class="quick-action-btn glass">
							<i data-lucide="plus-circle"></i>
							<span>Add Student</span>
						</a>
						<a href="#/analysis" class="quick-action-btn glass">
							<i data-lucide="upload"></i>
							<span>Upload Analysis</span>
						</a>
						<a href="#/settings" class="quick-action-btn glass">
							<i data-lucide="settings"></i>
							<span>Settings</span>
						</a>
					</div>
				</div>

				<!-- Recent Activity -->
				<div class="dashboard-section">
					<h2 class="dashboard-section-title">Recent Activity</h2>
					<div class="activity-feed">
						${
							recent_activity.length > 0
								? recent_activity
										.map(
											(activity) => `
							<div class="activity-item">
								<div class="activity-icon">
									<i data-lucide="${activity.icon || 'check-circle'}"></i>
								</div>
								<div class="activity-content">
									<p class="activity-text">${activity.description}</p>
									<p class="activity-time">${activity.timestamp || 'Just now'}</p>
								</div>
							</div>
						`
										)
										.join('')
								: '<p class="text-muted">No recent activity</p>'
						}
					</div>
				</div>
			</div>
		`;

		appElement.innerHTML = createShell(pageContent);
		attachShellHandlers();

		// Refresh Lucide icons
		if (window.lucide) {
			lucide.createIcons();
		}
	} catch (error) {
		const errorContent = `
			<div class="dashboard-container">
				<div class="error-card glass">
					<i data-lucide="alert-circle"></i>
					<p>Unable to load dashboard data</p>
					<p class="text-sm text-muted">${error.message}</p>
				</div>
			</div>
		`;

		appElement.innerHTML = createShell(errorContent);
		attachShellHandlers();

		if (window.lucide) {
			lucide.createIcons();
		}
	}
}
