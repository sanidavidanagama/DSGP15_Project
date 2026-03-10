import streamlit as st

def dashboard():

    st.title("Dashboard")
    st.write("Welcome back! Here's your overview")

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.markdown("<div class='metric-card'><h1>48</h1>Total Students</div>",unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='metric-card'><h1>127</h1>Total Analyses</div>",unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='metric-card'><h1>4</h1>Active Classes</div>",unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='metric-card'><h1>12</h1>This Week</div>",unsafe_allow_html=True)

    st.markdown("### Recent Activity")

    activity = [
        ("Emma Watson","Happy","2 hours ago"),
        ("Liam Chen","Curious","4 hours ago"),
        ("Sophia Ahmed","Excited","1 day ago"),
        ("Noah Rodriguez","Calm","1 day ago"),
    ]

    for name,emotion,time in activity:

        st.markdown(f"""
        <div class="card">
        <b>{name}</b><br>
        Emotion: {emotion}<br>
        <small>{time}</small>
        </div>
        """,unsafe_allow_html=True)