import streamlit as st

from services.class_api import ClassApiError, get_students_by_class


def _safe_text(value: object, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback

def class_detail_page():

    cls = st.session_state.get("selected_class")

    if not cls:
        st.warning("No class selected")
        return

    st.title(_safe_text(cls.get("class_name")))
    st.write(_safe_text(cls.get("grade_age_group")))

    schedule_days = cls.get("schedule_days") or []
    if schedule_days:
        chips = " ".join(f"`{_safe_text(day)}`" for day in schedule_days)
        st.caption(f"Schedule: {chips}")

    class_id = cls.get("id")
    students = []
    if isinstance(class_id, int):
        try:
            students = get_students_by_class(class_id)
        except ClassApiError as exc:
            st.warning(f"Could not load students from backend: {exc}")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:18px;font-weight:600;'>Total Students</div>
                <div style='font-size:36px;font-weight:700;margin-top:12px;'>{len(students)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class='card'>
                <div style='font-size:18px;font-weight:600;'>Average Analyses</div>
                <div style='font-size:36px;font-weight:700;margin-top:12px;'>N/A</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class='card'>
                <div style='font-size:18px;font-weight:600;'>Active Today</div>
                <div style='font-size:36px;font-weight:700;margin-top:12px;'>N/A</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    search = st.text_input("Search students")

    if search:
        students = [s for s in students if search.lower() in _safe_text(s.get("name"), fallback="").lower()]

    st.subheader("Students")

    if not students:
        st.caption("No students found for this class yet.")

    for student in students:
        with st.expander(f"{_safe_text(student.get('name'))} — {_safe_text(student.get('age_group'))}"):
            st.markdown(
                f"""
                <div class='card'>
                    <div style='margin-bottom:10px; font-weight:600;'>Student ID: {_safe_text(student.get('id'))}</div>
                    <div style='margin-bottom:10px; font-weight:600;'>Age Group: {_safe_text(student.get('age_group'))}</div>
                    <div style='color:#475569;'>Joined At: {_safe_text(student.get('joined_at'))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if st.button("← Back to Classes"):
        st.session_state.page = "My Classes"
        st.rerun()