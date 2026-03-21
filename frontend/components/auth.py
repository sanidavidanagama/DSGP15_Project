import streamlit as st

from services.auth_api import login as auth_login, AuthApiError


def login():

    st.markdown("<h1 style='text-align:center'>💙</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center;color:#1e3a8a'>Welcome to INKIND</h2>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#2563eb'>Sign in to start tracking emotions</p>", unsafe_allow_html=True)

    with st.form("login_form"):

        email = st.text_input("Email Address")
        password = st.text_input("Password", type="password")

        if st.form_submit_button("Sign In"):
            if not email or not password:
                st.error("Enter email and password")
            else:
                try:
                    token_data = auth_login(email, password)
                    st.session_state.auth = True
                    st.session_state.auth_token = token_data.get("access_token")
                    st.session_state.teacher_id = email
                    st.success("Logged in successfully")
                    if hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()
                    else:
                        st.session_state.page = "Dashboard"
                        st.stop()
                except AuthApiError as exc:
                    st.error(str(exc))

    st.markdown("Track emotional growth through children's art")
    st.markdown("AI-powered emotion analysis")
    st.markdown("Manage multiple classes")