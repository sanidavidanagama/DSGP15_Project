// Hash-based client-side router for INKIND
// Supports static paths ("/dashboard") and simple dynamic segments ("/classes/:id").

export class Router {
	constructor(routes, onRouteChange) {
		this.routes = routes;
		this.onRouteChange = onRouteChange;

		this.handleRouteChange = this.handleRouteChange.bind(this);

		window.addEventListener('hashchange', this.handleRouteChange);
		window.addEventListener('load', this.handleRouteChange);
	}

	getCurrentPath() {
		const hash = window.location.hash || '#/';
		const path = hash.startsWith('#') ? hash.slice(1) : hash;
		return path || '/';
	}

	matchRoute(path) {
		const pathSegments = path.split('/').filter(Boolean);

		let matched = null;
		let params = {};

		Object.entries(this.routes).forEach(([routePath, routeDef]) => {
			if (matched) return;

			const routeSegments = routePath.split('/').filter(Boolean);
			if (routeSegments.length !== pathSegments.length) return;

			const tempParams = {};
			let isMatch = true;

			for (let i = 0; i < routeSegments.length; i += 1) {
				const routeSegment = routeSegments[i];
				const pathSegment = pathSegments[i];

				if (routeSegment.startsWith(':')) {
					const key = routeSegment.slice(1);
					tempParams[key] = decodeURIComponent(pathSegment);
				} else if (routeSegment !== pathSegment) {
					isMatch = false;
					break;
				}
			}

			if (isMatch) {
				matched = routeDef;
				params = tempParams;
			}
		});

		return { route: matched, params };
	}

	async handleRouteChange() {
		const path = this.getCurrentPath();
		const { route, params } = this.matchRoute(path);

		if (!route) {
			// Fallback: redirect to landing
			this.navigate('/');
			return;
		}

		if (typeof this.onRouteChange === 'function') {
			await this.onRouteChange(route, params, path);
		}
	}

	navigate(path) {
		const target = path.startsWith('#') ? path : `#${path}`;
		if (window.location.hash === target) {
			// Force handling even if hash is the same
			this.handleRouteChange();
		} else {
			window.location.hash = target;
		}
	}

	destroy() {
		window.removeEventListener('hashchange', this.handleRouteChange);
		window.removeEventListener('load', this.handleRouteChange);
	}
}

