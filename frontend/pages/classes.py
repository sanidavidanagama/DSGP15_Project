import streamlit as st

def classes_page():

    st.title("My Classes")
    st.write("Manage students & track emotional growth")

    classes = [
        {"name": "Class 3A", "grade": "Grade 3", "students": 24, "days": "Mon, Wed, Fri"},
        {"name": "Class 3B", "grade": "Grade 3", "students": 22, "days": "Tue, Thu"},
        {"name": "Class 4A", "grade": "Grade 4", "students": 26, "days": "Mon, Wed"},
        {"name": "Art Club", "grade": "Mixed Ages", "students": 15, "days": "Friday"},
    ]

    cols = st.columns(3)

    for i, cls in enumerate(classes):

        with cols[i % 3]:
            st.markdown('<div class="class-card-container">', unsafe_allow_html=True)
            if st.button(
                f"{cls['name']}\n\n{cls['grade']}\n\n👥 {cls['students']} students • 📅 {cls['days']}",
                key=f"view_{cls['name']}",
                use_container_width=True
            ):
                st.session_state.selected_class = cls
                st.session_state.page = "class_detail"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)