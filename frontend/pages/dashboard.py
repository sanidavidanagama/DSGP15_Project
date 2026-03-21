import streamlit as st
from html import escape

from services.dashboard_api import DashboardApiError, get_dashboard_overview


def _format_emotion(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return "Unknown"
    return text[:1].upper() + text[1:].lower()

def dashboard():

    token = st.session_state.get("auth_token")
    overview = {
        "total_students": 0,
        "total_analyses": 0,
        "active_classes": 0,
        "analyses_this_week": 0,
        "recent_activity": [],
    }

    try:
        if token:
            overview = get_dashboard_overview(token=token)
        else:
            st.warning("You are not authenticated. Showing empty dashboard values.")
    except DashboardApiError as exc:
        st.warning(f"Unable to load dashboard data from backend: {exc}")

    activity = overview.get("recent_activity") if isinstance(overview.get("recent_activity"), list) else []

    st.title("Dashboard")
    st.write("Welcome back! Here's your overview")

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.markdown(f"<div class='metric-card'><h1>{int(overview.get('total_students') or 0)}</h1>Total Students</div>",unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='metric-card'><h1>{int(overview.get('total_analyses') or 0)}</h1>Total Analyses</div>",unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='metric-card'><h1>{int(overview.get('active_classes') or 0)}</h1>Active Classes</div>",unsafe_allow_html=True)

    with col4:
        st.markdown(f"<div class='metric-card'><h1>{int(overview.get('analyses_this_week') or 0)}</h1>This Week</div>",unsafe_allow_html=True)

    st.markdown("### Recent Activity")

    if not activity:
        st.caption("No analysis activity yet.")

    for item in activity[:5]:
        name = escape(str(item.get("student_name") or "Unknown Student"))
        emotion = escape(_format_emotion(item.get("emotion")))
        time = escape(str(item.get("time_ago") or "just now"))

        st.markdown(f"""
        <div class="card">
        <b>{name}</b><br>
        Emotion: {emotion}<br>
        <small>{time}</small>
        </div>
        """,unsafe_allow_html=True)