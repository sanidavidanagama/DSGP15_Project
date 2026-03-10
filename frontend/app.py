import streamlit as st

from components.auth import login
from components.sidebar import sidebar

from pages.dashboard import dashboard
from pages.classes import classes_page
from pages.analysis import analysis

from utils.styles import apply_styles

st.set_page_config(
    page_title="INKIND",
    layout="wide"
)

apply_styles()

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:

    login()

else:

    page = sidebar()

    if page == "Dashboard":
        dashboard()

    if page == "My Classes":
        classes_page()

    if page == "New Analysis":
        analysis()