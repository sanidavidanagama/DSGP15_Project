import streamlit as st
from data.mock_data import students

def class_detail():

    st.title("Class Details")

    st.subheader("Students")

    for s in students:

        col1,col2,col3 = st.columns([2,1,1])

        col1.write(s["name"])
        col2.write(s["emotion"])
        col3.write(s["trend"])

        if st.button("Profile",key=s["name"]):

            st.session_state.child = s["name"]