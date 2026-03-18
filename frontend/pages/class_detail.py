import streamlit as st

from services.class_api import ClassApiError, delete_class, get_class, get_students_by_class


WEEKDAY_LABELS = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "weds": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}


def _safe_text(value: object, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _display_day_label(day: object) -> str:
    normalized = _safe_text(day, fallback="").strip()
    if not normalized:
        return ""
    return WEEKDAY_LABELS.get(normalized.lower(), normalized)


def _render_header(classroom: dict) -> None:
    schedule_days = classroom.get("schedule_days") or []
    schedule_chips = []
    for day in schedule_days:
        label = _display_day_label(day)
        if label:
            schedule_chips.append(f"<span class='analysis-chip'>{label}</span>")

    if not schedule_chips:
        schedule_chips = ["<span class='analysis-chip'>No schedule set</span>"]

    subtitle = f"{_safe_text(classroom.get('grade_age_group'))} • {_safe_text(classroom.get('description'), fallback='No class description added yet.')}"

    st.markdown(
        f"""
        <div class='analysis-hero'>
            <h2 style='margin:0'>{_safe_text(classroom.get('class_name'))}</h2>
            <p class='analysis-subtitle'>{subtitle}</p>
            <div class='analysis-chip-row'>
                {''.join(schedule_chips)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def class_detail_page():

    cls = st.session_state.get("selected_class")

    if not cls:
        st.warning("No class selected")
        return

    class_id = cls.get("id")
    if not isinstance(class_id, int):
        st.error("Selected class is missing a valid id.")
        return

    try:
        cls = get_class(class_id)
        st.session_state.selected_class = cls
    except ClassApiError as exc:
        st.warning(f"Could not refresh class details from backend: {exc}")

    _render_header(cls)

    action_col1, action_col2, action_col3 = st.columns([1.1, 1.1, 4.8])
    with action_col1:
        if st.button("Back to Classes", key="detail_back_to_classes", use_container_width=True):
            st.session_state.page = "My Classes"
            st.rerun()

    with action_col2:
        if st.button("Edit Class", key="detail_edit_class", use_container_width=True):
            st.session_state.page = "edit_class"
            st.rerun()

    with action_col3:
        if st.button("Delete Class", key="detail_delete_class", use_container_width=False):
            st.session_state.delete_confirm_class_id = class_id

    if st.session_state.get("delete_confirm_class_id") == class_id:
        st.warning("Delete this class permanently? This action cannot be undone.")
        confirm_col, cancel_col = st.columns([1, 1])
        with confirm_col:
            if st.button("Yes, Delete", key="detail_delete_confirm", use_container_width=True):
                try:
                    delete_class(class_id)
                except ClassApiError as exc:
                    st.error(f"Failed to delete class: {exc}")
                else:
                    st.session_state.delete_confirm_class_id = None
                    st.session_state.selected_class = None
                    st.session_state.page = "My Classes"
                    st.success("Class deleted successfully.")
                    st.rerun()
        with cancel_col:
            if st.button("Cancel", key="detail_delete_cancel", use_container_width=True):
                st.session_state.delete_confirm_class_id = None
                st.rerun()

    students = []
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