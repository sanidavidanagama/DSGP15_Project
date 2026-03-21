import streamlit as st

from services.class_api import ClassApiError, delete_class, get_class
from services.student_api import StudentApiError, create_student, list_students


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


def _format_mood_label(value: object) -> str:
    mood = _safe_text(value, fallback="No mood yet")
    if mood.lower() in {"n/a", "no mood yet"}:
        return mood
    return mood[:1].upper() + mood[1:].lower()


def _render_metric_tile(label: str, value: str) -> None:
    st.markdown(
        f"<div class='analysis-metric'><div class='analysis-metric-label'>{label}</div><div class='analysis-metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


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


def _student_card(student: dict) -> bool:
    student_id = student.get("id")
    mood = _format_mood_label(student.get("last_predicted_mood"))
    last_predicted = _safe_text(student.get("last_predicted_label"), fallback="No predictions yet")
    total_analyses = _safe_text(student.get("total_analyses"), fallback="0")

    st.markdown(
        f"""
        <div class='student-click-wrap'>
            <div class='student-grid-card'>
                <div class='student-grid-head'>
                    <div class='student-avatar'>👧</div>
                    <div class='student-head-copy'>
                        <div class='student-grid-name'>{_safe_text(student.get('name'))}</div>
                        <div class='student-grid-mood'>{mood}</div>
                    </div>
                </div>
                <div class='student-grid-footer'>
                    <span class='student-grid-time'>{last_predicted}</span>
                    <span class='student-grid-analyses'>{total_analyses} analyses</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Open Student Profile", key=f"student_open_{student_id}", use_container_width=True):
        st.session_state.selected_student = student
        st.session_state.page = "student_profile"
        st.rerun()
        return True

    return False

def class_detail_page():
    token = st.session_state.get("auth_token")

    cls = st.session_state.get("selected_class")

    if not cls:
        st.warning("No class selected")
        return

    class_id = cls.get("id")
    if not isinstance(class_id, int):
        st.error("Selected class is missing a valid id.")
        return

    try:
        cls = get_class(class_id, token=token)
        st.session_state.selected_class = cls
    except ClassApiError as exc:
        st.warning(f"Could not refresh class details from backend: {exc}")

    _render_header(cls)

    st.markdown("<h4 class='analysis-section-title'>Class Actions</h4>", unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if st.button("Back to Classes", key="detail_back_to_classes", use_container_width=True):
            st.session_state.page = "My Classes"
            st.rerun()

    with action_col2:
        if st.button("Edit Class", key="detail_edit_class", use_container_width=True):
            st.session_state.page = "edit_class"
            st.rerun()

    with action_col3:
        if st.button("Delete Class", key="detail_delete_class", use_container_width=True):
            st.session_state.delete_confirm_class_id = class_id

    if st.session_state.get("delete_confirm_class_id") == class_id:
        st.warning("Delete this class permanently? This action cannot be undone.")
        confirm_col, cancel_col = st.columns([1, 1])
        with confirm_col:
            if st.button("Yes, Delete", key="detail_delete_confirm", use_container_width=True):
                try:
                    delete_class(class_id, token=token)
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

    students: list[dict] = []
    try:
        students = list_students(class_id, token=token)
    except StudentApiError as exc:
        st.warning(f"Could not load students from backend: {exc}")

    st.divider()

    col1, col2, col3 = st.columns(3)
    total_analyses = sum(int(s.get("total_analyses") or 0) for s in students)

    with col1:
        _render_metric_tile("Total Students", str(len(students)))

    with col2:
        _render_metric_tile("Total Analyses", str(total_analyses))

    with col3:
        recent_count = len([s for s in students if _safe_text(s.get("last_predicted_label")) in {"just now"}])
        _render_metric_tile("Updated Recently", str(recent_count))

    st.divider()

    header_col, action_col = st.columns([3.8, 1.2])
    with header_col:
        st.subheader(f"Students ({len(students)})")
    with action_col:
        if st.button("Add New Student", key="detail_add_student_toggle", use_container_width=True):
            st.session_state.show_add_student_form = not st.session_state.get("show_add_student_form", False)

    if st.session_state.get("show_add_student_form", False):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        with st.form("add_student_form", clear_on_submit=True):
            student_name = st.text_input("Student Name", placeholder="e.g. Emma Watson")
            student_age_group = st.text_input("Age Group", placeholder="e.g. Grade 4")
            submitted = st.form_submit_button("Create Student", type="primary")

        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            if not student_name.strip() or not student_age_group.strip():
                st.error("Student name and age group are required.")
            else:
                try:
                    create_student(
                        class_id=class_id,
                        name=student_name.strip(),
                        age_group=student_age_group.strip(),
                        token=token,
                    )
                except StudentApiError as exc:
                    st.error(f"Failed to create student: {exc}")
                else:
                    st.success("Student added successfully.")
                    st.session_state.show_add_student_form = False
                    st.rerun()

    search = st.text_input("Search students", placeholder="Type student name...")

    if search:
        students = [s for s in students if search.lower() in _safe_text(s.get("name"), fallback="").lower()]

    if not students:
        st.caption("No students found for this class yet.")
        return

    columns = st.columns(3, gap="medium")
    for index, student in enumerate(students):
        with columns[index % 3]:
            _student_card(student)