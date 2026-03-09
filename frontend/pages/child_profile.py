import streamlit as st

def child_profile():

    st.title("Child Profile")

    st.metric("Total Analyses","12")

    st.subheader("Emotion Distribution")

    st.progress(0.42)
    st.write("Happy 42%")

    st.progress(0.25)
    st.write("Excited 25%")

    st.progress(0.17)
    st.write("Calm 17%")

    st.progress(0.16)
    st.write("Curious 16%")

    st.subheader("History")

    history = [
        ("Dec 23","Happy"),
        ("Dec 20","Excited"),
        ("Dec 18","Calm"),
        ("Dec 15","Curious")
    ]

    for h in history:
        st.write(h)