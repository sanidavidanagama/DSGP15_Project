import streamlit as st
import altair as alt
import pandas as pd
from datetime import datetime

from services.student_api import StudentApiError, delete_student, get_student, list_saved_analyses, update_student


def _safe_text(value: object, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _format_mood_label(value: object) -> str:
    mood = _safe_text(value, fallback="No mood yet")
    if mood.lower() in {"n/a", "no mood yet"}:
        return mood
    return mood[:1].upper() + mood[1:].lower()


def _confidence_to_percent(value: object) -> float:
    text = _safe_text(value, fallback="0").replace("%", "").strip()
    try:
        parsed = float(text)
    except ValueError:
        return 0.0
    return min(100.0, max(0.0, parsed))


def _happy_score_to_percent(value: object) -> float | None:
    if value is None:
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        text = _safe_text(value, fallback="").replace("%", "").strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None

    if 0.0 <= parsed <= 1.0:
        parsed *= 100.0

    return min(100.0, max(0.0, parsed))


def _parse_datetime(value: object) -> datetime | None:
    text = _safe_text(value, fallback="").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _format_saved_at(value: object) -> str:
    dt = _parse_datetime(value)
    if not dt:
        return "N/A"
    return dt.strftime("%d %b %y | %H:%M")


def _render_metric_tile(label: str, value: str) -> None:
    st.markdown(
        f"<div class='analysis-metric'><div class='analysis-metric-label'>{label}</div><div class='analysis-metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def _render_hero(student: dict, class_name: str) -> None:
    st.markdown(
        f"""
        <div class='analysis-hero'>
            <h2 style='margin:0'>{_safe_text(student.get('name'))}</h2>
            <p class='analysis-subtitle'>{_safe_text(student.get('age_group'))} • {class_name}</p>
            <div class='analysis-chip-row'>
                <span class='analysis-chip'>Last Mood: {_format_mood_label(student.get('last_predicted_mood'))}</span>
                <span class='analysis-chip'>Last Update: {_safe_text(student.get('last_predicted_label'), fallback='No predictions yet')}</span>
                <span class='analysis-chip'>Total Analyses: {_safe_text(student.get('total_analyses'), fallback='0')}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def child_profile():
    token = st.session_state.get("auth_token")

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
        student = get_student(student_id, token=token)
        st.session_state.selected_student = student
    except StudentApiError as exc:
        st.warning(f"Could not refresh student details: {exc}")

    try:
        history = list_saved_analyses(student_id, token=token)
    except StudentApiError as exc:
        st.warning(f"Could not load saved analysis history: {exc}")
        history = []

    class_name = _safe_text(selected_class.get("class_name"), fallback="Class")
    _render_hero(student, class_name)

    st.markdown("<h4 class='analysis-section-title'>Student Actions</h4>", unsafe_allow_html=True)
    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if st.button("Back to Class", key="student_profile_back", use_container_width=True):
            st.session_state.page = "class_detail"
            st.rerun()

    with action_col2:
        if st.button("Edit Student", key="student_profile_edit", use_container_width=True):
            st.session_state.show_student_edit_form = not st.session_state.get("show_student_edit_form", False)

    with action_col3:
        if st.button("Delete Student", key="student_profile_delete", use_container_width=True):
            st.session_state.delete_confirm_student_id = student_id

    if st.session_state.get("delete_confirm_student_id") == student_id:
        st.warning("Delete this student profile? This action cannot be undone.")
        confirm_col, cancel_col = st.columns([1, 1])
        with confirm_col:
            if st.button("Yes, Delete", key="student_profile_delete_confirm", use_container_width=True):
                try:
                    delete_student(student_id, token=token)
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
                    updated = update_student(student_id, name=name.strip(), age_group=age_group.strip(), token=token)
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
        _render_metric_tile("Total Analyses", _safe_text(student.get("total_analyses"), fallback="0"))

    with col2:
        _render_metric_tile("Last Predicted Mood", _format_mood_label(student.get("last_predicted_mood")))

    with col3:
        _render_metric_tile("Last Updated", _safe_text(student.get("last_predicted_label"), fallback="N/A"))

    st.subheader("Analysis History")
    if not history:
        st.caption("No saved analyses yet. Once teachers save analysis results to this student, they will appear here.")
        return

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        chart_points = []
        for item in history:
            saved_at = _parse_datetime(item.get("saved_at"))
            happy_percent = _happy_score_to_percent(item.get("happy_score"))
            if not saved_at:
                continue
            if happy_percent is None:
                continue
            chart_points.append(
                {
                    "saved_at": saved_at,
                    "predicted_mood": happy_percent,
                }
            )

        st.markdown("<h4 class='analysis-section-title' style='margin-top:2px'>Mood Trend</h4>", unsafe_allow_html=True)
        if chart_points:
            chart_df = pd.DataFrame(chart_points)
            chart_df = chart_df.dropna(subset=["saved_at"]).sort_values("saved_at")

            if not chart_df.empty:
                start_at = chart_df["saved_at"].min()
                end_at = chart_df["saved_at"].max()

                if start_at == end_at:
                    # Ensure visible horizontal range for a single-point timeline.
                    start_at = start_at.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_at = end_at.replace(hour=23, minute=59, second=59, microsecond=0)

                bands_df = pd.DataFrame(
                    [
                        {"x0": start_at, "x1": end_at, "y0": 0, "y1": 40, "zone": "Low"},
                        {"x0": start_at, "x1": end_at, "y0": 40, "y1": 60, "zone": "Mid"},
                        {"x0": start_at, "x1": end_at, "y0": 60, "y1": 100, "zone": "High"},
                    ]
                )

                zone_colors = alt.Scale(
                    domain=["Low", "Mid", "High"],
                    range=["#ef4444", "#facc15", "#22c55e"],
                )

                y_encoding = alt.Y(
                    "predicted_mood:Q",
                    scale=alt.Scale(domain=[0, 100], nice=False),
                    title="Predicted Mood",
                )
                x_encoding = alt.X(
                    "saved_at:T",
                    axis=alt.Axis(title="Timeline", format="%d %b %y"),
                )

                zones = alt.Chart(bands_df).mark_rect(opacity=0.09).encode(
                    x="x0:T",
                    x2="x1:T",
                    y="y0:Q",
                    y2="y1:Q",
                    color=alt.Color("zone:N", scale=zone_colors, legend=None),
                )

                thresholds = alt.Chart(pd.DataFrame({"threshold": [40, 60]})).mark_rule(
                    color="#7A6F63",
                    strokeDash=[6, 4],
                ).encode(y="threshold:Q")

                trend = alt.Chart(chart_df).mark_line(point=True, color="#2A7F8F", strokeWidth=3).encode(
                    x=x_encoding,
                    y=y_encoding,
                    tooltip=[
                        alt.Tooltip("saved_at:T", title="Timeline", format="%d %b %y"),
                        alt.Tooltip("predicted_mood:Q", title="Predicted Mood", format=".0f"),
                    ],
                )

                chart = (
                    alt.layer(zones, thresholds, trend)
                    .properties(height=320)
                    .configure(background="#e7e1d1")
                    .configure_view(fill="#e7e1d1", strokeOpacity=0)
                    .configure_axis(
                        labelColor="#3D3730",
                        titleColor="#3D3730",
                        domainColor="#9D8F7D",
                        tickColor="#9D8F7D",
                        gridColor="rgba(124, 108, 91, 0.28)",
                    )
                )

                st.altair_chart(chart, use_container_width=True, theme=None)
            else:
                st.caption("Not enough timestamped history to render the mood trend yet.")
        else:
            st.caption("No happy score history available yet.")

    with right_col:
        st.markdown("<h4 class='analysis-section-title' style='margin-top:2px'>Saved Analyses</h4>", unsafe_allow_html=True)
        for item in history:
            job_id = _safe_text(item.get("job_id"), fallback="")
            mood = _format_mood_label(item.get("mood"))
            saved_at = _format_saved_at(item.get("saved_at"))
            description = _safe_text(item.get("drawing_description"), fallback="No drawing description available.")

            st.markdown(
                f"""
                <div class='analysis-list-card student-history-card'>
                    <div class='analysis-kv-item student-history-description'>
                        <div class='analysis-kv-key'>Drawing Description</div>
                        <div class='analysis-kv-value'>{description}</div>
                    </div>
                    <div class='student-history-meta-row'>
                        <div class='analysis-kv-item'>
                            <div class='analysis-kv-key'>Predicted Mood</div>
                            <div class='analysis-kv-value'>{mood}</div>
                        </div>
                        <div class='analysis-kv-item'>
                            <div class='analysis-kv-key'>Saved At</div>
                            <div class='analysis-kv-value'>{saved_at}</div>
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