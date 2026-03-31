import streamlit as st


def sidebar():
    with st.sidebar:

        # ── Logo & Branding ──────────────────────────────
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; padding:20px 0 2px;">
                <span style="font-size:1.6rem; line-height:1;">💙</span>
                <span style="font-size:1.3rem; font-weight:800; color:#e0e0f0;
                             letter-spacing:0.06em;">INKIND</span>
            </div>
            <p style="color:#7a7a9a; font-size:0.82rem; margin:2px 0 18px 2px;">
                Teacher Portal
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Navigation ───────────────────────────────────
        for page_name in ["Dashboard", "My Classes", "New Analysis"]:
            is_active = st.session_state.get("page") == page_name

            if is_active:
                st.markdown(
                    f'<div class="nav-item-active">{page_name}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(page_name, key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.page = page_name
                    st.rerun()

        st.divider()

        # ── Teacher Info ─────────────────────────────────
        teacher_name = st.session_state.get("teacher_name")
        teacher_id = st.session_state.get("teacher_id")

        if teacher_name and teacher_id:
            display_line = f"{teacher_name} ({teacher_id})"
        elif teacher_name:
            display_line = teacher_name
        elif teacher_id:
            display_line = teacher_id
        else:
            display_line = "teacher@school.edu"

        st.markdown(
            f"""
            <p class="sidebar-label">Teacher Name</p>
            <p class="sidebar-email">{display_line}</p>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Logout", key="logout_btn"):
            st.session_state.auth = False
            st.session_state.auth_token = None
            st.session_state.teacher_id = None
            st.session_state.teacher_name = None
            st.session_state.selected_class = None
            st.session_state.selected_student = None
            st.session_state.page = "Dashboard"
            st.rerun()