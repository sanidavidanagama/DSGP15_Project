import streamlit as st

def sidebar():

    def set_page():
        st.session_state.page = st.session_state.sidebar_radio

    st.sidebar.markdown("## 💙 INKIND")
    st.sidebar.write("Teacher Portal")

    page = st.sidebar.radio(
        "",
        [
            "Dashboard",
            "My Classes",
            "New Analysis"
        ],
        key="sidebar_radio",
        on_change=set_page
    )

    st.sidebar.markdown("---")

    st.sidebar.write("Teacher Name")
    st.sidebar.write("teacher@school.edu")

    if st.sidebar.button("Logout"):
        st.session_state.auth = False
        st.rerun()

    return page