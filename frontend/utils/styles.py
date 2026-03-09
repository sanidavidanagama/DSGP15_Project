import streamlit as st

def apply_styles():

    st.markdown("""
    <style>

    .stApp{
        background: linear-gradient(135deg,#eff6ff,#e0f2fe,#ecfeff);
    }

    .block-container{
        padding-top:2rem;
        padding-bottom:2rem;
        padding-left:3rem;
        padding-right:3rem;
    }

    /* cards */

    .card{
        background:white;
        padding:25px;
        border-radius:18px;
        border:1px solid #dbeafe;
        box-shadow:0 8px 20px rgba(0,0,0,0.05);
    }

    /* metric cards */

    .metric-card{
        background:white;
        padding:20px;
        border-radius:18px;
        border:1px solid #dbeafe;
        box-shadow:0 4px 12px rgba(0,0,0,0.05);
        text-align:center;
    }

    /* gradient buttons */

    .primary-btn{
        background: linear-gradient(90deg,#2563eb,#06b6d4);
        color:white;
        padding:14px 28px;
        border-radius:14px;
        border:none;
        font-weight:600;
        text-align:center;
    }

    /* upload box */

    .upload-box{
        border:2px dashed #93c5fd;
        border-radius:18px;
        padding:60px;
        text-align:center;
        color:#2563eb;
        background:#eff6ff;
    }

    /* sidebar */

    section[data-testid="stSidebar"]{
        background:white;
        border-right:1px solid #dbeafe;
    }

    </style>
    """, unsafe_allow_html=True)