import streamlit as st

def apply_styles():

    st.markdown("""
    <style>

    .stApp{
        background:linear-gradient(135deg,#dbeafe,#bfdbfe,#93c5fd);
        color:#1e293b;
    }

    .card{
        background:#e0f2fe;
        border:1px solid #60a5fa;
        padding:25px;
        border-radius:16px;
        margin-bottom:20px;
        color:#1e293b;
        box-shadow:0 6px 16px rgba(0,0,0,0.06);
    }

    .upload-box{
        background:#e0f2fe;
        border:2px dashed #2563eb;
        padding:60px;
        border-radius:16px;
        text-align:center;
        color:#1e3a8a;
        font-weight:600;
    }

    .loading-box{
        background:#e0f2fe;
        border:1px solid #60a5fa;
        border-radius:16px;
        padding:50px;
        text-align:center;
        font-size:20px;
        color:#1e3a8a;
    }

    div.stButton > button{
        background:#2563eb;
        color:white;
        border-radius:10px;
        padding:10px 22px;
        border:none;
        font-weight:600;
    }

    div.stButton > button:hover{
        background:#1d4ed8;
    }

    h1,h2,h3{
        color:#1e293b;
    }

    </style>
    """, unsafe_allow_html=True)