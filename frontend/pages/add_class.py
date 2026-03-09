import streamlit as st

def add_class():

    st.title("Add New Class")

    name = st.text_input("Class Name")
    grade = st.text_input("Grade")
    schedule = st.text_input("Schedule")

    desc = st.text_area("Description")

    if st.button("Create Class"):

        st.success("Class Created")