import streamlit as st
from data.mock_data import classes

def classes_page():

    st.title("My Classes")

    cols = st.columns(3)

    for i,c in enumerate(classes):

        with cols[i%3]:

            st.markdown(f"""
            <div class="card">
            <h3>{c['name']}</h3>
            <p>{c['grade']}</p>
            <p>{c['students']} students</p>
            <p>{c['schedule']}</p>
            </div>
            """,unsafe_allow_html=True)

            if st.button("Open",key=c["id"]):
                st.session_state.selected_class = c["id"]