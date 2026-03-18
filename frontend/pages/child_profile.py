import streamlit as st

from services.student_api import StudentApiError, delete_student, get_student, list_saved_analyses, update_student


def _safe_text(value: object, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _render_hero(student: dict, class_name: str) -> None:
    st.markdown(
        f"""
        <div class='analysis-hero'>
            <h2 style='margin:0'>{_safe_text(student.get('name'))}</h2>
            <p class='analysis-subtitle'>{_safe_text(student.get('age_group'))} • {class_name}</p>
            <div class='analysis-chip-row'>
                <span class='analysis-chip'>Last Mood: {_safe_text(student.get('last_predicted_mood'), fallback='No mood yet')}</span>
                <span class='analysis-chip'>Last Update: {_safe_text(student.get('last_predicted_label'), fallback='No predictions yet')}</span>
                <span class='analysis-chip'>Total Analyses: {_safe_text(student.get('total_analyses'), fallback='0')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def child_profile():

    student = st.session_state.get("selected_student")
    selected_class = st.session_state.get("selected_class") or {}

    if not student:
        st.warning("No student selected")
        if st.button("Back to Class", key="student_profile_back_missing"):
            st.session_state.page = "class_detail"
            st.rerun()
        return

    student_id = student.get("id")
    if not isinstance(student_id, int):
        st.error("Selected student is missing a valid id.")
        return

    try:
        student = get_student(student_id)
        st.session_state.selected_student = student
    except StudentApiError as exc:
        st.warning(f"Could not refresh student details: {exc}")

    try:
        history = list_saved_analyses(student_id)
    except StudentApiError as exc:
        st.warning(f"Could not load saved analysis history: {exc}")
        history = []

    class_name = _safe_text(selected_class.get("class_name"), fallback="Class")
    _render_hero(student, class_name)

    action_col1, action_col2, action_col3 = st.columns([1.2, 1.2, 4.6])
    with action_col1:
        if st.button("Back to Class", key="student_profile_back", use_container_width=True):
            st.session_state.page = "class_detail"
            st.rerun()

    with action_col2:
        if st.button("Edit Student", key="student_profile_edit", use_container_width=True):
            st.session_state.show_student_edit_form = not st.session_state.get("show_student_edit_form", False)

    with action_col3:
        if st.button("Delete Student", key="student_profile_delete", use_container_width=False):
            st.session_state.delete_confirm_student_id = student_id

    if st.session_state.get("delete_confirm_student_id") == student_id:
        st.warning("Delete this student profile? This action cannot be undone.")
        confirm_col, cancel_col = st.columns([1, 1])
        with confirm_col:
            if st.button("Yes, Delete", key="student_profile_delete_confirm", use_container_width=True):
                try:
                    delete_student(student_id)
                except StudentApiError as exc:
                    st.error(f"Failed to delete student: {exc}")
                else:
                    st.session_state.delete_confirm_student_id = None
                    st.session_state.selected_student = None
                    st.session_state.page = "class_detail"
                    st.success("Student deleted successfully.")
                    st.rerun()

        with cancel_col:
            if st.button("Cancel", key="student_profile_delete_cancel", use_container_width=True):
                st.session_state.delete_confirm_student_id = None
                st.rerun()

    if st.session_state.get("show_student_edit_form", False):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        with st.form("edit_student_form", clear_on_submit=False):
            name = st.text_input("Student Name", value=_safe_text(student.get("name"), fallback=""))
            age_group = st.text_input("Age Group", value=_safe_text(student.get("age_group"), fallback=""))
            submitted = st.form_submit_button("Save Changes", type="primary")

        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            if not name.strip() or not age_group.strip():
                st.error("Student name and age group are required.")
            else:
                try:
                    updated = update_student(student_id, name=name.strip(), age_group=age_group.strip())
                except StudentApiError as exc:
                    st.error(f"Failed to update student: {exc}")
                else:
                    st.session_state.selected_student = updated
                    st.session_state.show_student_edit_form = False
                    st.success("Student updated successfully.")
                    st.rerun()

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:18px;font-weight:600;'>Total Analyses</div>
                <div style='font-size:36px;font-weight:700;margin-top:12px;'>{_safe_text(student.get('total_analyses'), fallback='0')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:18px;font-weight:600;'>Last Predicted Mood</div>
                <div style='font-size:36px;font-weight:700;margin-top:12px;'>{_safe_text(student.get('last_predicted_mood'), fallback='N/A')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class='card'>
                <div style='font-size:18px;font-weight:600;'>Last Updated</div>
                <div style='font-size:36px;font-weight:700;margin-top:12px;'>{_safe_text(student.get('last_predicted_label'), fallback='N/A')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Analysis History")
    if not history:
        st.caption("No saved analyses yet. Once teachers save analysis results to this student, they will appear here.")
        return

    for item in history:
        job_id = _safe_text(item.get("job_id"), fallback="")
        st.markdown(
            f"""
            <div class='analysis-list-card' style='margin-bottom:10px'>
                <div class='analysis-kv'>
                    <div class='analysis-kv-item'>
                        <div class='analysis-kv-key'>Mood</div>
                        <div class='analysis-kv-value'>{_safe_text(item.get('mood'))}</div>
                    </div>
                    <div class='analysis-kv-item'>
                        <div class='analysis-kv-key'>Confidence</div>
                        <div class='analysis-kv-value'>{_safe_text(item.get('confidence'))}</div>
                    </div>
                    <div class='analysis-kv-item'>
                        <div class='analysis-kv-key'>Saved At</div>
                        <div class='analysis-kv-value'>{_safe_text(item.get('saved_at'))}</div>
                    </div>
                    <div class='analysis-kv-item'>
                        <div class='analysis-kv-key'>Job Id</div>
                        <div class='analysis-kv-value'>{job_id}</div>
                    </div>
                    <div class='analysis-kv-item'>
                        <div class='analysis-kv-key'>Summary</div>
                        <div class='analysis-kv-value'>{_safe_text(item.get('summary'))}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Open Full Metrics", key=f"open_saved_metrics_{_safe_text(item.get('id'))}", use_container_width=False):
            if not job_id:
                st.error("Job id not available for this saved item.")
            else:
                st.session_state.job_id = job_id
                st.session_state.analysis_page = "loading"
                st.session_state.page = "New Analysis"
                st.rerun()