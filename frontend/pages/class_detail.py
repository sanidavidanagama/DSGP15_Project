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
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:18px;font-weight:600;'>Total Students</div>
                <div style='font-size:36px;font-weight:700;margin-top:12px;'>{cls['students']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class='card'>
                <div style='font-size:18px;font-weight:600;'>Average Analyses</div>
                <div style='font-size:36px;font-weight:700;margin-top:12px;'>13</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class='card'>
                <div style='font-size:18px;font-weight:600;'>Active Today</div>
                <div style='font-size:36px;font-weight:700;margin-top:12px;'>2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    search = st.text_input("Search students")

    students = [
        {"name": "Emma Watson", "emotion": "Happy", "analyses": 12, "time": "2 hours ago"},
        {"name": "Liam Chen", "emotion": "Calm", "analyses": 15, "time": "1 day ago"},
        {"name": "Sophia Ahmed", "emotion": "Excited", "analyses": 9, "time": "3 days ago"},
        {"name": "Noah Silva", "emotion": "Happy", "analyses": 7, "time": "1 week ago"},
        {"name": "Ava Brown", "emotion": "Curious", "analyses": 5, "time": "2 days ago"},
        {"name": "Ethan Lee", "emotion": "Calm", "analyses": 4, "time": "3 days ago"},
    ]

    if search:
        students = [s for s in students if search.lower() in s["name"].lower()]

    st.subheader("Students")

    for student in students:
        with st.expander(f"{student['name']} — {student['emotion']}"):
            st.markdown(
                f"""
                <div class='card'>
                    <div style='margin-bottom:10px; font-weight:600;'>Last active: {student['time']}</div>
                    <div style='margin-bottom:10px; font-weight:600;'>Analyses: {student['analyses']}</div>
                    <div style='color:#475569;'>You can add more student-specific insights here as needed.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("← Back to Classes"):
        st.session_state.page = "My Classes"
        st.rerun()