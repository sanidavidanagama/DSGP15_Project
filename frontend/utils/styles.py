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

    /* -- Suppress Streamlit's white widget pill backgrounds -- */
    .stTextInput > div > div,
    .stTextArea > div > div,
    [data-baseweb="input"],
    [data-baseweb="textarea"] {
        background: rgba(237, 227, 208, 0.45) !important;
        border-color: var(--ink-border) !important;
        border-radius: 10px !important;
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
        color: #F7F2EA;
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
        color: #F7F2EA !important;
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

    </style>
    """, unsafe_allow_html=True)