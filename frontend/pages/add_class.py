import streamlit as st

from services.class_api import ClassApiError, create_class


def _render_hero() -> None:
    st.markdown(
        """
        <div class='analysis-hero'>
            <h2 style='margin:0'>Add New Class</h2>
            <p class='analysis-subtitle'>Create a class profile and sync it directly with the backend.</p>
            <div class='analysis-chip-row'>
                <span class='analysis-chip'>Backend Write</span>
                <span class='analysis-chip'>Teacher Scoped</span>
                <span class='analysis-chip'>Schedule-Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def add_class():
    _render_hero()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with st.form("add_class_form", clear_on_submit=False):
        class_name = st.text_input("Class Name", placeholder="e.g. Grade 4A")
        grade_age_group = st.text_input("Grade / Age Group", placeholder="e.g. Grade 4")
        schedule_days = st.multiselect(
            "Schedule Days",
            options=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            placeholder="Pick one or more days",
        )
        description = st.text_area(
            "Description",
            placeholder="Optional class notes or focus areas.",
            height=100,
        )

        submitted = st.form_submit_button("Create Class", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        if not class_name.strip() or not grade_age_group.strip() or not schedule_days:
            st.error("Class name, grade/age group, and at least one schedule day are required.")
            return

        try:
            create_class(
                class_name=class_name.strip(),
                grade_age_group=grade_age_group.strip(),
                schedule_days=schedule_days,
                description=description.strip(),
                token=st.session_state.get("auth_token"),
            )
        except ClassApiError as exc:
            st.error(f"Failed to create class: {exc}")
            return

        st.success("Class created successfully.")
        st.session_state.page = "My Classes"
        st.rerun()

    if st.button("Back to My Classes"):
        st.session_state.page = "My Classes"
        st.rerun()