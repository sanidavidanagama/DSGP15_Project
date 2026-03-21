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
        st.markdown(
            """
            <p class="sidebar-label">Teacher Name</p>
            <p class="sidebar-email">teacher@school.edu</p>
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