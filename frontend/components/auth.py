import streamlit as st

def login():

    st.markdown("<h1 style='text-align:center'>💙</h1>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center;color:#1e3a8a'>Welcome to INKIND</h2>", unsafe_allow_html=True)

    st.markdown("<p style='text-align:center;color:#2563eb'>Create your account to start tracking emotions</p>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")

    if st.button("Create Account"):

        if email and password:
            st.session_state.auth = True
            st.rerun()
        else:
            st.error("Enter email and password")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("Track emotional growth through children's art")
    st.markdown("AI-powered emotion analysis")
    st.markdown("Manage multiple classes")