import streamlit as st

def apply_styles():

    st.markdown("""
    <style>
    :root {
        /* Backgrounds */
        --ink-bg-1: #F7F2EA;
        --ink-bg-2: #EDE3D0;
        --ink-bg-3: #C9DDE8;

        /* Surfaces */
        --ink-card: rgba(247, 242, 234, 0.72);
        --ink-card-warm: rgba(237, 227, 208, 0.52);
        --ink-border: rgba(160, 130, 100, 0.2);
        --ink-soft-shadow: 0 10px 32px rgba(60, 40, 20, 0.07);

        /* Text */
        --ink-text: #1C1A17;
        --ink-title: #3D3730;
        --ink-muted: #6B6158;

        /* Accents */
        --ink-teal: #2A7F8F;
        --ink-teal-dark: #1F6070;
        --ink-amber: #C47A3A;
        --ink-sage: #5A8A6A;
        --ink-green: #4A9460;
    }

    .stApp {
        background:
            radial-gradient(circle at 18% 8%, rgba(196, 122, 58, 0.14), transparent 36%),
            radial-gradient(circle at 84% 20%, rgba(42, 127, 143, 0.16), transparent 40%),
            linear-gradient(125deg, var(--ink-bg-1), var(--ink-bg-2) 42%, var(--ink-bg-3));
        color: var(--ink-text);
    }

    /* ── Sidebar ──────────────────────────────────────── */

    /* Hide auto-generated Streamlit multi-page nav */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Dark sidebar background */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #1e1e2e !important;
    }

    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 8px 0 !important;
    }

    /* Nav buttons – transparent, left-aligned, muted */
    section[data-testid="stSidebar"] div.stButton > button {
        background: transparent !important;
        color: #9090aa !important;
        border: none !important;
        border-radius: 8px !important;
        text-align: left !important;
        padding: 10px 14px !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: background 0.15s ease, color 0.15s ease;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] div.stButton > button:hover,
    section[data-testid="stSidebar"] div.stButton > button:focus {
        background: rgba(255, 255, 255, 0.07) !important;
        color: #d0d0e8 !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Active nav item */
    .nav-item-active {
        padding: 10px 14px 10px 11px;
        color: #60c4e8;
        font-weight: 600;
        font-size: 0.95rem;
        border-left: 3px solid #60c4e8;
        background: rgba(96, 196, 232, 0.10);
        border-radius: 0 8px 8px 0;
        margin-bottom: 4px;
    }

    /* Teacher info labels */
    .sidebar-label {
        color: #6a6a8a;
        font-size: 0.8rem;
        margin: 4px 0 2px;
    }

    .sidebar-email {
        color: #5eadd4;
        font-size: 0.88rem;
        margin: 0 0 12px;
    }

    /* -- Suppress Streamlit's white widget pill backgrounds -- */
    .stTextInput > div > div,
    .stTextArea > div > div,
    [data-baseweb="input"],
    [data-baseweb="textarea"] {
        background: rgba(237, 227, 208, 0.45) !important;
        border-color: var(--ink-border) !important;
        border-radius: 10px !important;
    }

    /* Keep form copy consistently readable on light surfaces. */
    .stTextInput label,
    .stTextArea label,
    .stSelectbox label,
    .stMultiSelect label,
    .stNumberInput label,
    .stDateInput label,
    .stTimeInput label,
    .stRadio label,
    .stCheckbox label,
    .stFileUploader label {
        color: var(--ink-title) !important;
    }

    .stTextInput input,
    .stTextArea textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        color: var(--ink-text) !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder,
    [data-baseweb="input"] input::placeholder,
    [data-baseweb="textarea"] textarea::placeholder {
        color: #7A6F63 !important;
        opacity: 1 !important;
    }

    /* Remove white backgrounds from Streamlit metric & info widgets */
    [data-testid="stMetric"],
    [data-testid="metric-container"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .card {
        background: var(--ink-card);
        border: 1px solid var(--ink-border);
        padding: 25px;
        border-radius: 20px;
        margin-bottom: 20px;
        color: var(--ink-text);
        box-shadow: var(--ink-soft-shadow);
        backdrop-filter: blur(6px);
    }

    .upload-box {
        background: rgba(42, 127, 143, 0.07);
        border: 2px dashed var(--ink-teal);
        padding: 34px;
        border-radius: 20px;
        text-align: center;
        color: var(--ink-teal-dark);
        font-weight: 600;
    }

    .loading-box {
        background: var(--ink-card-warm);
        border: 1px solid var(--ink-border);
        border-radius: 20px;
        padding: 24px;
        box-shadow: var(--ink-soft-shadow);
    }

    /* -- Buttons -- */
    div.stButton > button {
        background: var(--ink-teal);
        color: #143944;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: background 0.18s ease;
    }

    div.stButton > button:hover,
    div.stButton > button:focus {
        background: var(--ink-teal-dark) !important;
        color: #E3EFEF !important;
        border: none !important;
    }

    /* -- Typography -- */
    h1, h2, h3, h4 {
        color: var(--ink-title);
    }

    /* -- Hero banner -- */
    .analysis-hero {
        border-radius: 24px;
        padding: 24px 28px;
        background: linear-gradient(125deg, rgba(42, 127, 143, 0.13), rgba(196, 122, 58, 0.17));
        border: 1px solid rgba(42, 127, 143, 0.2);
        margin-bottom: 18px;
        box-shadow: var(--ink-soft-shadow);
    }

    .analysis-subtitle {
        color: var(--ink-muted);
        font-size: 0.97rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* -- Chips -- */
    .analysis-chip-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
    }

    .analysis-chip {
        background: rgba(237, 227, 208, 0.7);
        border: 1px solid rgba(160, 130, 100, 0.3);
        color: var(--ink-title);
        border-radius: 999px;
        padding: 4px 13px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .analysis-section-title {
        margin-top: 0.2rem;
        margin-bottom: 0.75rem;
        color: var(--ink-title);
    }

    /* -- Metric tiles -- */
    .analysis-metric {
        border-radius: 16px;
        padding: 16px 18px;
        background: rgba(237, 227, 208, 0.5);
        border: 1px solid var(--ink-border);
        box-shadow: 0 3px 10px rgba(60, 40, 20, 0.05);
    }

    .analysis-metric-label {
        font-size: 0.8rem;
        color: var(--ink-muted);
        margin-bottom: 4px;
    }

    .analysis-metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--ink-title);
    }

    /* -- Loading timeline steps -- */
    .analysis-step {
        border-radius: 14px;
        padding: 12px 14px;
        border: 1px solid var(--ink-border);
        background: rgba(237, 227, 208, 0.36);
        margin-bottom: 8px;
    }

    .analysis-step.active {
        border-color: rgba(42, 127, 143, 0.48);
        background: rgba(42, 127, 143, 0.09);
    }

    .analysis-step.done {
        border-color: rgba(90, 138, 106, 0.38);
        background: rgba(90, 138, 106, 0.09);
    }

    .analysis-step-title {
        font-size: 0.92rem;
        font-weight: 700;
        margin: 0;
        color: var(--ink-title);
    }

    .analysis-step-caption {
        font-size: 0.82rem;
        margin: 2px 0 0;
        color: var(--ink-muted);
    }

    /* -- Mood bar -- */
    .analysis-mood-stack {
        width: 100%;
        height: 18px;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid var(--ink-border);
        background: rgba(237, 227, 208, 0.6);
        display: flex;
        margin: 8px 0;
    }

    .analysis-mood-positive {
        background: var(--ink-green);
        height: 100%;
    }

    .analysis-mood-support {
        background: var(--ink-amber);
        height: 100%;
    }

    /* -- List cards and KV grid -- */
    .analysis-list-card {
        border-radius: 16px;
        padding: 16px;
        border: 1px solid var(--ink-border);
        background: rgba(237, 227, 208, 0.4);
        height: 100%;
    }

    .analysis-kv {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }

    .analysis-kv-item {
        border-radius: 12px;
        padding: 10px 12px;
        background: rgba(247, 242, 234, 0.68);
        border: 1px solid rgba(160, 130, 100, 0.18);
    }

    .analysis-kv-key {
        font-size: 0.78rem;
        color: var(--ink-muted);
        margin-bottom: 3px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .analysis-kv-value {
        font-size: 0.92rem;
        color: var(--ink-title);
        font-weight: 600;
        word-break: break-word;
    }

    @media (max-width: 900px) {
        .analysis-kv {
            grid-template-columns: 1fr;
        }
    }

    /* -- Classes page -- */
    .class-grid-wrapper {
        margin-top: 8px;
    }

    /* Make the widget immediately following a card act as a full-card click layer. */
    div[data-testid="stElementContainer"]:has(.class-click-wrap) {
        margin-bottom: 0;
    }

    div[data-testid="stElementContainer"]:has(.class-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) {
        margin-top: -228px !important;
        height: 228px;
        margin-bottom: 10px !important;
        position: relative;
        z-index: 3;
    }

    div[data-testid="stElementContainer"]:has(.class-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) .stButton {
        height: 100%;
    }

    div[data-testid="stElementContainer"]:has(.class-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) .stButton > button {
        width: 100%;
        height: 100%;
        min-height: 228px;
        opacity: 0;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
        cursor: pointer;
    }

    div[data-testid="stElementContainer"]:has(.class-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) .stButton > button:focus,
    div[data-testid="stElementContainer"]:has(.class-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) .stButton > button:hover {
        border: none !important;
        outline: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: transparent !important;
    }

    .class-grid-link {
        display: block;
        margin-bottom: 0;
        text-decoration: none !important;
        color: inherit;
    }

    .class-grid-link,
    .class-grid-link * {
        text-decoration: none !important;
    }

    .class-grid-link:focus-visible {
        outline: 2px solid rgba(42, 127, 143, 0.65);
        outline-offset: 4px;
        border-radius: 20px;
    }

    .class-grid-link.is-disabled {
        pointer-events: none;
        opacity: 0.75;
    }

    .class-grid-link:hover .class-grid-card {
        border-color: rgba(42, 127, 143, 0.42);
        box-shadow: 0 16px 34px rgba(60, 40, 20, 0.12);
        transform: translateY(-2px);
    }

    .class-grid-link:hover .class-grid-arrow {
        transform: translateX(3px);
        opacity: 1;
    }

    .class-grid-card {
        min-height: 228px;
        border-radius: 20px;
        padding: 18px;
        border: 1px solid var(--ink-border);
        background: rgba(247, 242, 234, 0.72);
        box-shadow: var(--ink-soft-shadow);
        margin-bottom: 0;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .class-grid-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 10px;
    }

    .class-grid-title {
        font-size: 1.16rem;
        font-weight: 700;
        color: var(--ink-title);
        margin: 0;
    }

    .class-grid-subtitle {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--ink-muted);
        margin: -4px 0 0;
    }

    .class-grid-arrow {
        color: var(--ink-teal);
        font-size: 1.15rem;
        line-height: 1;
        opacity: 0.78;
        transition: transform 0.2s ease, opacity 0.2s ease;
    }

    .class-chip-row {
        margin-top: 2px;
        margin-bottom: 2px;
    }

    .class-grid-description {
        font-size: 0.92rem;
        line-height: 1.46;
        color: var(--ink-text);
        margin: 0;
        min-height: 56px;
    }

    .class-grid-footer {
        margin-top: auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding-top: 8px;
        border-top: 1px solid rgba(160, 130, 100, 0.22);
    }

    .class-grid-meta {
        font-size: 0.84rem;
        font-weight: 600;
        color: var(--ink-teal-dark);
        margin: 0;
    }

    .class-grid-open {
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--ink-teal);
        letter-spacing: 0.01em;
    }

    .class-grid-add {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        border: 1.5px dashed rgba(42, 127, 143, 0.45);
        background: rgba(42, 127, 143, 0.09);
        min-height: 236px;
    }

    .class-grid-plus {
        width: 58px;
        height: 58px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(42, 127, 143, 0.18);
        color: var(--ink-teal-dark);
        font-size: 1.9rem;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .class-grid-add .class-grid-description {
        min-height: 0;
        max-width: 320px;
        margin-bottom: 2px;
    }

    /* -- Student cards in class detail -- */
    div[data-testid="stElementContainer"]:has(.student-click-wrap) {
        margin-bottom: 0;
    }

    div[data-testid="stElementContainer"]:has(.student-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) {
        margin-top: -184px !important;
        height: 184px;
        margin-bottom: 12px !important;
        position: relative;
        z-index: 3;
    }

    div[data-testid="stElementContainer"]:has(.student-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) .stButton {
        height: 100%;
    }

    div[data-testid="stElementContainer"]:has(.student-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) .stButton > button {
        width: 100%;
        height: 100%;
        min-height: 184px;
        opacity: 0;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
        cursor: pointer;
    }

    div[data-testid="stElementContainer"]:has(.student-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) .stButton > button:focus,
    div[data-testid="stElementContainer"]:has(.student-click-wrap)
    + div[data-testid="stElementContainer"]:has(.stButton) .stButton > button:hover {
        border: none !important;
        outline: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: transparent !important;
    }

    .student-grid-card {
        min-height: 184px;
        border-radius: 18px;
        padding: 14px;
        border: 1px solid rgba(42, 127, 143, 0.22);
        background: rgba(247, 242, 234, 0.78);
        box-shadow: var(--ink-soft-shadow);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
    }

    .student-click-wrap:hover .student-grid-card {
        border-color: rgba(42, 127, 143, 0.42);
        box-shadow: 0 14px 28px rgba(60, 40, 20, 0.12);
        transform: translateY(-2px);
    }

    .student-grid-head {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .student-avatar {
        width: 58px;
        height: 58px;
        border-radius: 16px;
        border: 1px solid rgba(42, 127, 143, 0.3);
        background: rgba(42, 127, 143, 0.12);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.9rem;
        line-height: 1;
    }

    .student-head-copy {
        min-width: 0;
    }

    .student-grid-name {
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--ink-title);
        margin-bottom: 2px;
    }

    .student-grid-mood {
        font-size: 0.96rem;
        color: var(--ink-teal-dark);
        font-weight: 600;
    }

    .student-grid-footer {
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px solid rgba(160, 130, 100, 0.2);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
    }

    .student-grid-time {
        font-size: 0.9rem;
        color: var(--ink-teal);
        font-weight: 600;
    }

    .student-grid-analyses {
        border: 1px solid rgba(42, 127, 143, 0.24);
        background: rgba(42, 127, 143, 0.1);
        color: var(--ink-teal-dark);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .student-history-panel {
        min-height: 360px;
    }

    .student-history-card {
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .student-history-description {
        background: rgba(247, 242, 234, 0.65);
    }

    .student-history-meta-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }

    @media (max-width: 768px) {
        div[data-testid="stElementContainer"]:has(.class-click-wrap) {
            margin-bottom: 0;
        }

        div[data-testid="stElementContainer"]:has(.class-click-wrap)
        + div[data-testid="stElementContainer"]:has(.stButton) {
            margin-top: -214px !important;
            height: 214px;
            margin-bottom: 12px !important;
        }

        div[data-testid="stElementContainer"]:has(.class-click-wrap)
        + div[data-testid="stElementContainer"]:has(.stButton) .stButton > button {
            min-height: 214px;
        }

        div[data-testid="stElementContainer"]:has(.student-click-wrap)
        + div[data-testid="stElementContainer"]:has(.stButton) {
            margin-top: -170px !important;
            height: 170px;
            margin-bottom: 12px !important;
        }

        div[data-testid="stElementContainer"]:has(.student-click-wrap)
        + div[data-testid="stElementContainer"]:has(.stButton) .stButton > button {
            min-height: 170px;
        }

        .student-grid-card {
            min-height: 170px;
            padding: 12px;
        }

        .student-grid-name {
            font-size: 0.98rem;
        }

        .student-grid-mood,
        .student-grid-time {
            font-size: 0.86rem;
        }

        .class-grid-link {
            margin-bottom: 0;
        }

        .class-grid-card,
        .class-grid-add {
            min-height: 214px;
            padding: 16px;
        }

        .class-grid-title {
            font-size: 1.06rem;
        }

        .class-grid-description {
            min-height: 0;
            font-size: 0.88rem;
        }

        .student-history-meta-row {
            grid-template-columns: 1fr;
        }
    }

    .stForm{
        background: linear-gradient(145deg, rgba(247, 242, 234, 0.62), rgba(237, 227, 208, 0.46));
        border: 1px solid rgba(42, 127, 143, 0.22);
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 18px;
        color: var(--ink-text);
        box-shadow: 0 10px 22px rgba(60, 40, 20, 0.08);
        backdrop-filter: blur(5px);
    }

    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, var(--ink-teal), #237187) !important;
        color: #103640 !important;
        border: 1px solid rgba(31, 96, 112, 0.55) !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        box-shadow: 0 8px 18px rgba(31, 96, 112, 0.22);
    }

    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:focus {
        background: linear-gradient(135deg, var(--ink-teal-dark), #1a5462) !important;
        color: #b7d5db !important;
        border-color: rgba(20, 73, 85, 0.7) !important;
    }

    /* -- Auth wallpaper layout -- */
    .auth-visual {
        position: relative;
        width: 100%;
        min-height: 380px;
        border-radius: 24px;
        overflow: hidden;
        border: 1px solid rgba(160, 130, 100, 0.32);
        box-shadow: var(--ink-soft-shadow);
        background-image: url('../assets/wallpaper.jpg');
        background-size: cover;
        background-position: center;
    }

    .auth-visual-overlay {
        position: absolute;
        inset: 0;
        padding: 22px 22px 20px;
        background: radial-gradient(circle at 10% 0%, rgba(0,0,0,0.4), transparent 55%),
                    linear-gradient(145deg, rgba(7, 24, 46, 0.76), rgba(9, 44, 68, 0.15));
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        color: #f6fbff;
    }

    .auth-visual-overlay h2 {
        margin: 0 0 4px;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .auth-visual-overlay p {
        margin: 0;
        font-size: 0.9rem;
        max-width: 360px;
        line-height: 1.5;
        opacity: 0.94;
    }

    @media (max-width: 900px) {
        .auth-visual {
            min-height: 220px;
            margin-bottom: 14px;
        }
    }

    .auth-brand {
        text-align: center;
        margin: 4px 0 22px;
    }

    .auth-brand-logo {
        width: 90px;
        height: auto;
        display: block;
        margin: 0 auto 8px;
        filter: drop-shadow(0 6px 12px rgba(0, 0, 0, 0.18));
    }

    .auth-brand-name {
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: var(--ink-title);
        margin-bottom: 2px;
    }

    .auth-brand-tagline {
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ink-muted);
        margin-bottom: 4px;
    }

    .auth-brand-subtitle {
        font-size: 0.9rem;
        color: var(--ink-muted);
    }

    </style>
    """, unsafe_allow_html=True)