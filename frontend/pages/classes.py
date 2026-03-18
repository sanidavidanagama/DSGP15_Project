from datetime import datetime, timezone
from html import escape
import streamlit as st

from services.class_api import ClassApiError, build_classes_dashboard


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
    mapped = WEEKDAY_LABELS.get(normalized.lower())
    return mapped if mapped else normalized


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
    chips: list[str] = []
    for day in schedule_days:
        label = _display_day_label(day)
        if label:
            chips.append(f"<span class='analysis-chip'>{escape(label)}</span>")
    if not chips:
        return "<span class='analysis-chip'>No schedule set</span>"
    return "".join(chips)


def _class_card(classroom: dict) -> None:
    schedule_html = _render_schedule_chips(classroom.get("schedule_days") or [])
    class_name = escape(_safe_text(classroom.get("class_name")))
    grade_age_group = escape(_safe_text(classroom.get("grade_age_group")))
    description = escape(_safe_text(classroom.get("description"), fallback="No description provided yet."))
    student_count = escape(_safe_text(classroom.get("student_count"), fallback="0"))
    class_id = classroom.get("id")

    card_class = "class-grid-link"
    if not isinstance(class_id, int):
        card_class += " is-disabled"

    st.markdown(
        f"""
        <div class='class-click-wrap {card_class}'>
            <div class='class-grid-card'>
                <div class='class-grid-head'>
                    <div class='class-grid-title'>{class_name}</div>
                    <span class='class-grid-arrow' aria-hidden='true'>→</span>
                </div>
                <div class='class-grid-subtitle'>{grade_age_group}</div>
                <div class='analysis-chip-row class-chip-row'>{schedule_html}</div>
                <p class='class-grid-description'>{description}</p>
                <div class='class-grid-footer'>
                    <p class='class-grid-meta'>Students: {student_count}</p>
                    <span class='class-grid-open'>Open class</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_clickable = isinstance(class_id, int)
    label = "Open class" if is_clickable else "Unavailable"
    if st.button(label, key=f"open_class_{class_id}", use_container_width=True, disabled=not is_clickable):
        st.session_state.selected_class = classroom
        st.session_state.page = "class_detail"
        st.rerun()


def _add_class_card() -> None:
    st.markdown(
        """
        <div class='class-click-wrap class-grid-link class-grid-add-link'>
            <div class='class-grid-card class-grid-add'>
                <div class='class-grid-plus'>+</div>
                <div class='class-grid-title'>Add Class</div>
                <p class='class-grid-description'>Create a new class and define schedule days.</p>
                <span class='class-grid-open'>Create class</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Add class", key="add_class_card", use_container_width=True):
        st.session_state.page = "add_class"
        st.rerun()


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
    columns = st.columns(3, gap="small")

    for index, classroom in enumerate(classes):
        with columns[index % 3]:
            _class_card(classroom)

    with columns[len(classes) % 3]:
        _add_class_card()

    st.markdown("</div>", unsafe_allow_html=True)