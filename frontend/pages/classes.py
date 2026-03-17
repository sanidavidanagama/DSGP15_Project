from datetime import datetime, timezone

import streamlit as st

from services.class_api import ClassApiError, build_classes_dashboard


def _safe_text(value: object, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _relative_time_label(value: datetime | None) -> str:
    if value is None:
        return "No analysis yet"

    now = datetime.now(timezone.utc)
    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    delta = now - dt
    seconds = max(0, int(delta.total_seconds()))

    if seconds < 60:
        return "just now"
    if seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m ago"
    if seconds < 86400:
        hours = seconds // 3600
        return f"{hours}h ago"
    days = seconds // 86400
    return f"{days}d ago"


def _render_hero() -> None:
    st.markdown(
        """
        <div class='analysis-hero'>
            <h2 style='margin:0'>My Classes</h2>
            <p class='analysis-subtitle'>Manage class groups, track student coverage, and jump into details quickly.</p>
            <div class='analysis-chip-row'>
                <span class='analysis-chip'>Backend Synced</span>
                <span class='analysis-chip'>Teacher Workspace</span>
                <span class='analysis-chip'>Analysis-Aligned UI</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_tile(label: str, value: str) -> None:
    st.markdown(
        f"<div class='analysis-metric'><div class='analysis-metric-label'>{label}</div><div class='analysis-metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def _render_schedule_chips(schedule_days: list[str]) -> str:
    if not schedule_days:
        return "<span class='analysis-chip'>No schedule set</span>"
    return "".join(f"<span class='analysis-chip'>{_safe_text(day)}</span>" for day in schedule_days)


def _class_card(classroom: dict) -> None:
    schedule_html = _render_schedule_chips(classroom.get("schedule_days") or [])
    st.markdown("<div class='class-click-wrap'>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class='class-grid-card'>
            <div class='class-grid-title'>{_safe_text(classroom.get('class_name'))}</div>
            <div class='class-grid-subtitle'>{_safe_text(classroom.get('grade_age_group'))}</div>
            <div class='analysis-chip-row class-chip-row'>{schedule_html}</div>
            <p class='class-grid-description'>{_safe_text(classroom.get('description'), fallback='No description provided yet.')}</p>
            <p class='class-grid-meta'>Students: {_safe_text(classroom.get('student_count'), fallback='0')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    class_id = classroom.get("id")
    if st.button("Open class", key=f"open_class_{class_id}", use_container_width=True):
        st.session_state.selected_class = classroom
        st.session_state.page = "class_detail"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def _add_class_card() -> None:
    st.markdown("<div class='class-click-wrap'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='class-grid-card class-grid-add'>
            <div class='class-grid-plus'>+</div>
            <div class='class-grid-title'>Add Class</div>
            <p class='class-grid-description'>Create a new class and define schedule days.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Add new class", key="add_class_cta", use_container_width=True):
        st.session_state.page = "add_class"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def classes_page():
    _render_hero()

    try:
        dashboard = build_classes_dashboard()
    except ClassApiError as exc:
        st.error(f"Unable to load classes from backend: {exc}")
        st.caption("Make sure backend is running and reachable from frontend.")
        return

    classes = dashboard.get("classes", [])
    total_classes = dashboard.get("total_classes", 0)
    total_students = dashboard.get("total_students", 0)
    last_analysis_label = _relative_time_label(dashboard.get("last_analysis_at"))

    metric_cols = st.columns(3)
    with metric_cols[0]:
        _render_metric_tile("Total Classes", str(total_classes))
    with metric_cols[1]:
        _render_metric_tile("Total Students", str(total_students))
    with metric_cols[2]:
        _render_metric_tile("Last Analysis", last_analysis_label)

    search = st.text_input("Search classes", placeholder="Type class name or age group...")
    if search:
        search_value = search.lower().strip()
        classes = [
            classroom
            for classroom in classes
            if search_value in _safe_text(classroom.get("class_name"), fallback="").lower()
            or search_value in _safe_text(classroom.get("grade_age_group"), fallback="").lower()
        ]

    st.markdown("<div class='class-grid-wrapper'>", unsafe_allow_html=True)
    columns = st.columns(2, gap="large")

    for index, classroom in enumerate(classes):
        with columns[index % 2]:
            _class_card(classroom)

    with columns[len(classes) % 2]:
        _add_class_card()

    st.markdown("</div>", unsafe_allow_html=True)