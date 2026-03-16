import streamlit as st


def class_detail_page():

    cls = st.session_state.get("selected_class")

    if not cls:
        st.warning("No class selected")
        return

    st.title(cls["name"])
    st.write(cls["grade"])

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("Total Students")
        st.header("6")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("Average Analyses")
        st.header("13")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("Active Today")
        st.header("2")
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.text_input("Search students...", "")

    st.subheader("Students (6)")

    students = [
        {"name": "Emma Watson", "emotion": "Happy", "analyses": 12, "time": "2 hours ago"},
        {"name": "Liam Chen", "emotion": "Calm", "analyses": 15, "time": "1 day ago"},
        {"name": "Sophia Ahmed", "emotion": "Excited", "analyses": 9, "time": "3 days ago"},
    ]

    cols = st.columns(3)

    for i, student in enumerate(students):

        with cols[i % 3]:

            st.markdown("<div class='card'>", unsafe_allow_html=True)

            st.subheader(student["name"])
            st.write(student["emotion"])

            st.divider()

            st.write(student["time"])
            st.write(f"{student['analyses']} analyses")

            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("← Back to Classes"):
        st.session_state.page = "classes"
        st.rerun()