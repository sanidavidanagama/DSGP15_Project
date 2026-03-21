import streamlit as st

from services.auth_api import AuthApiError, get_current_profile, login as auth_login, register as auth_register


def _set_auth_page(page: str) -> None:
    st.session_state.auth_page = page


def _render_login_form() -> None:

    st.markdown("<h1 style='text-align:center'>💙</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center;color:#1e3a8a'>Welcome to INKIND</h2>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#2563eb'>Sign in to start tracking emotions</p>", unsafe_allow_html=True)

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
                    st.session_state.teacher_id = profile.get("teacher_id") or profile.get("email") or identifier
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

    c1, c2, c3 = st.columns([2, 3, 2])
    with c2:
        st.caption("Don\'t have an account yet?")
        if st.button("Create Account", key="goto_signup", use_container_width=True):
            _set_auth_page("signup")
            st.rerun()


def _render_signup_form() -> None:
    st.markdown("<h1 style='text-align:center'>💙</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:#1e3a8a'>Create your INKIND account</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#2563eb'>Sign up with username, email, and password</p>", unsafe_allow_html=True)

    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")

        if st.form_submit_button("Create Account"):
            if not username or not email or not password:
                st.error("Enter username, email, and password")
            else:
                try:
                    auth_register(username=username.strip(), email=email.strip(), password=password)
                    st.success("Account created successfully. Please sign in.")
                    _set_auth_page("login")
                    st.session_state.login_identifier = email.strip()
                    st.rerun()
                except AuthApiError as exc:
                    st.error(str(exc))

    c1, c2, c3 = st.columns([2, 3, 2])
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

    st.markdown("Track emotional growth through children's art")
    st.markdown("AI-powered emotion analysis")
    st.markdown("Manage multiple classes")