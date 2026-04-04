import streamlit as st

from services.class_api import ClassApiError, update_class


WEEKDAY_OPTIONS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

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


def _safe_text(value: object, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _normalize_schedule_day(value: object) -> str:
    normalized = _safe_text(value, fallback="")
    if not normalized:
        return ""
    return WEEKDAY_LABELS.get(normalized.lower(), normalized)


def _render_hero(class_name: str) -> None:
    st.markdown(
        f"""
        <div class='analysis-hero'>
            <h2 style='margin:0'>Edit Class</h2>
            <p class='analysis-subtitle'>Update details for {class_name} and keep schedules up to date.</p>
            <div class='analysis-chip-row'>
                <span class='analysis-chip'>Class Settings</span>
                <span class='analysis-chip'>Schedule Update</span>
                <span class='analysis-chip'>Save Changes</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def edit_class_page() -> None:
    token = st.session_state.get("auth_token")

    cls = st.session_state.get("selected_class")
    if not cls:
        st.warning("No class selected")
        if st.button("Back to My Classes", key="edit_back_without_class"):
            st.session_state.page = "My Classes"
            st.rerun()
        return

    class_id = cls.get("id")
    if not isinstance(class_id, int):
        st.error("Selected class is missing a valid id.")
        if st.button("Back to My Classes", key="edit_back_invalid_class"):
            st.session_state.page = "My Classes"
            st.rerun()
        return

    class_name_default = _safe_text(cls.get("class_name"))
    grade_default = _safe_text(cls.get("grade_age_group"))
    description_default = _safe_text(cls.get("description"))

    selected_schedule = [
        normalized
        for day in (cls.get("schedule_days") or [])
        for normalized in [_normalize_schedule_day(day)]
        if normalized in WEEKDAY_OPTIONS
    ]

    _render_hero(class_name_default or "this class")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.form("edit_class_form", clear_on_submit=False):
        class_name = st.text_input("Class Name", value=class_name_default, placeholder="e.g. Grade 4A")
        grade_age_group = st.text_input("Grade / Age Group", value=grade_default, placeholder="e.g. Grade 4")
        schedule_days = st.multiselect(
            "Schedule Days",
            options=WEEKDAY_OPTIONS,
            default=selected_schedule,
            placeholder="Pick one or more days",
        )
        description = st.text_area(
            "Description",
            value=description_default,
            placeholder="Optional class notes or focus areas.",
            height=100,
        )

        submitted = st.form_submit_button("Save Changes", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    action_col1, action_col2 = st.columns([1.2, 4.8])
    with action_col1:
        if st.button("Cancel", key="edit_cancel", use_container_width=True):
            st.session_state.page = "class_detail"
            st.rerun()

    if submitted:
        if not class_name.strip() or not grade_age_group.strip() or not schedule_days:
            st.error("Class name, grade/age group, and at least one schedule day are required.")
            return

        try:
            updated = update_class(
                class_id=class_id,
                class_name=class_name.strip(),
                grade_age_group=grade_age_group.strip(),
                schedule_days=schedule_days,
                description=description.strip(),
                token=token,
            )
        except ClassApiError as exc:
            st.error(f"Failed to update class: {exc}")
            return

        st.session_state.selected_class = updated
        st.success("Class updated successfully.")
        st.session_state.page = "class_detail"
        st.rerun()
