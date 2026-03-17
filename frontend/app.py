import streamlit as st

from components.auth import login
from components.sidebar import sidebar

from pages.dashboard import dashboard
from pages.classes import classes_page
from pages.class_detail import class_detail_page
from pages.analysis import analysis

from utils.styles import apply_styles


st.set_page_config(
    page_title="INKIND",
    layout="wide"
)

apply_styles()

# -------------------------
# Auth session
# -------------------------

if "auth" not in st.session_state:
    st.session_state.auth = False

# -------------------------
# Page session
# -------------------------

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "sidebar_radio" not in st.session_state:
    st.session_state.sidebar_radio = "Dashboard"

# -------------------------
# Login screen
# -------------------------

if not st.session_state.auth:

    login()

# -------------------------
# Main App
# -------------------------

else:

    sidebar()

    # -------------------------
    # Page Routing
    # -------------------------

    if st.session_state.page == "Dashboard":
        dashboard()

    elif st.session_state.page == "My Classes":
        classes_page()

    elif st.session_state.page == "class_detail":
        class_detail_page()

    elif st.session_state.page == "New Analysis":
        analysis()