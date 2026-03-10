import streamlit as st

def sidebar():

    st.sidebar.markdown("## 💙 INKIND")
    st.sidebar.write("Teacher Portal")

    page = st.sidebar.radio(
        "",
        [
            "Dashboard",
            "My Classes",
            "New Analysis"
        ]
    )

    st.sidebar.markdown("---")

    st.sidebar.write("Teacher Name")
    st.sidebar.write("teacher@school.edu")

    if st.sidebar.button("Logout"):
        st.session_state.auth = False
        st.rerun()

    return page