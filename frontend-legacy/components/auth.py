import base64
from pathlib import Path

import streamlit as st

from services.auth_api import AuthApiError, get_current_profile, login as auth_login, register as auth_register


# Frontend asset directory (relative to the working directory where app.py runs)
ASSETS_BASE_PATH = "assets"
LOGO_BLUE_PATH = f"{ASSETS_BASE_PATH}/logo-blue.png"
LOGO_WHITE_PATH = f"{ASSETS_BASE_PATH}/logo-white.png"
WALLPAPER_PATH = f"{ASSETS_BASE_PATH}/wallpaper.jpg"


def _set_auth_page(page: str) -> None:
    st.session_state.auth_page = page


def _inject_wallpaper_style() -> None:
    """Override auth wallpaper background with inline base64 image."""

    try:
        data = Path(WALLPAPER_PATH).read_bytes()
    except OSError:
        return

    encoded = base64.b64encode(data).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .auth-visual {{
            background-image: url("data:image/jpeg;base64,{encoded}");
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_auth_branding(subtitle: str) -> None:
    """Show fully centered logo, INKIND name and expanded tagline."""
    try:
        logo_bytes = Path(LOGO_BLUE_PATH).read_bytes()
        logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")
        logo_src = f"data:image/png;base64,{logo_b64}"
    except OSError:
        logo_src = ""

    img_html = ""
    if logo_src:
        img_html = f'<img src="{logo_src}" alt="INKIND" class="auth-brand-logo" />'

    st.markdown(
        f"""
        <div class="auth-brand">
            {img_html}
            <div class="auth-brand-name">INKIND</div>
            <div class="auth-brand-tagline">Interpreting Nonverbal Knowledge in Drawings</div>
            <div class="auth-brand-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_login_form() -> None:
    """Render the teacher login view with split wallpaper + card layout."""
    _inject_wallpaper_style()
    _render_auth_branding("Sign in to your teacher portal")

    visual_col, form_col = st.columns([1.3, 1])

    with visual_col:
        st.markdown(
            """
            <div class="auth-visual">
                <div class="auth-visual-overlay">
                    <h2>See the story in every rainbow.</h2>
                    <p>
                        Upload student artwork and let INKIND surface
                        gentle emotional patterns you can act on in the classroom.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        st.markdown(
            """
            <div class="card" style="margin-top: 4px;">
                <h3 style="margin:0 0 0.4rem;">Sign in</h3>
                <p style="margin:0 0 1.1rem; color: var(--ink-muted); font-size:0.9rem;">
                    Welcome back. Use your school email or username to access your classes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            identifier = st.text_input("Email or Username")
            password = st.text_input("Password", type="password")

            if st.form_submit_button("Sign In"):
                if not identifier or not password:
                    st.error("Enter email/username and password")
                else:
                    try:
                        token_data = auth_login(identifier, password)
                        token = token_data.get("access_token")
                        profile = get_current_profile(token) if token else {}

                        st.session_state.auth = True
                        st.session_state.auth_token = token
                        st.session_state.teacher_id = (
                            profile.get("teacher_id")
                            or profile.get("email")
                            or identifier
                        )
                        st.session_state.teacher_name = profile.get("username") or ""
                        st.session_state.page = "Dashboard"
                        st.success("Logged in successfully")
                        if hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()
                        else:
                            st.session_state.page = "Dashboard"
                            st.stop()
                    except AuthApiError as exc:
                        st.error(str(exc))

        c1, c2, c3 = st.columns([1.5, 2, 1.5])
        with c2:
            st.caption("Don\'t have an account yet?")
            if st.button("Create Account", key="goto_signup", use_container_width=True):
                _set_auth_page("signup")
                st.rerun()


def _render_signup_form() -> None:
    """Render the account creation view with matching split layout."""
    _inject_wallpaper_style()
    _render_auth_branding("Create your teacher account and INKIND classroom")

    visual_col, form_col = st.columns([1.3, 1])

    with visual_col:
        st.markdown(
            """
            <div class="auth-visual">
                <div class="auth-visual-overlay">
                    <h2>Welcome to a kinder data story.</h2>
                    <p>
                        INKIND helps you notice trends in mood and
                        wellbeing early, using the artwork your
                        students already love to create.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with form_col:
        st.markdown(
            """
            <div class="card" style="margin-top: 4px;">
                <h3 style="margin:0 0 0.4rem;">Create account</h3>
                <p style="margin:0 0 1.1rem; color: var(--ink-muted); font-size:0.9rem;">
                    Just a few details so we can personalise your teacher dashboard.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("signup_form"):
            username = st.text_input("Username")
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")

            if st.form_submit_button("Create Account"):
                if not username or not email or not password:
                    st.error("Enter username, email, and password")
                else:
                    try:
                        auth_register(
                            username=username.strip(),
                            email=email.strip(),
                            password=password,
                        )
                        st.success("Account created successfully. Please sign in.")
                        _set_auth_page("login")
                        st.session_state.login_identifier = email.strip()
                        st.rerun()
                    except AuthApiError as exc:
                        st.error(str(exc))

        c1, c2, c3 = st.columns([1.5, 2, 1.5])
        with c2:
            st.caption("Already have an account?")
            if st.button("Back to Login", key="goto_login", use_container_width=True):
                _set_auth_page("login")
                st.rerun()


def login():
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"

    if st.session_state.auth_page == "signup":
        _render_signup_form()
    else:
        _render_login_form()