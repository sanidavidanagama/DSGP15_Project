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
			active_classes = 0,
			total_analyses = 0,
			analyses_this_week = 0,
			recent_activity = [],
		} = dashboardData || {};

		const renderedRecentActivity = Array.isArray(recent_activity)
			? recent_activity
				.map((activity) => {
					const studentName = escapeHtml(activity.student_name || 'Unknown student');
					const emotion = escapeHtml(activity.emotion || 'unknown');
					const timeAgo = escapeHtml(activity.time_ago || 'just now');
					const classId = Number(activity.class_id);
					const studentId = Number(activity.student_id);
					const canOpenProfile = Number.isInteger(classId) && Number.isInteger(studentId);

					if (!canOpenProfile) {
						return `
							<div class="activity-item" role="group" aria-label="Recent activity item">
								<div class="activity-icon">
									<i data-lucide="smile"></i>
								</div>
								<div class="activity-content">
									<p class="activity-text"><strong>${studentName}</strong> - emotion analyzed: <strong>${emotion}</strong></p>
									<p class="activity-time">${timeAgo}</p>
								</div>
							</div>
						`;
					}

					return `
						<a class="activity-item activity-item-link" href="#/classes/${classId}/students/${studentId}" aria-label="Open ${studentName} profile">
							<div class="activity-icon">
								<i data-lucide="smile"></i>
							</div>
							<div class="activity-content">
								<p class="activity-text"><strong>${studentName}</strong> - emotion analyzed: <strong>${emotion}</strong></p>
								<p class="activity-time">${timeAgo}</p>
							</div>
						</a>
					`;
				})
				.join('')
			: '';

		const pageContent = `
			<div class="dashboard-container dashboard-page-frame">
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
							<p class="metric-value">${active_classes}</p>
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
							<p class="metric-value">${analyses_this_week}</p>
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
								? renderedRecentActivity
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
			<div class="dashboard-container dashboard-page-frame">
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

function escapeHtml(value) {
	return String(value)
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;')
		.replaceAll("'", '&#39;');
}
