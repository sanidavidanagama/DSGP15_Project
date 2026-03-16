import streamlit as st

def apply_styles():

    st.markdown("""
    <style>

    .stApp{
        background:linear-gradient(135deg,#dbeafe,#bfdbfe,#93c5fd);
        color:#1e293b;
    }

    .card{
        background:#f8fafc;
        border:1px solid rgba(96, 165, 250, 0.5);
        padding:20px;
        border-radius:16px;
        margin-bottom:18px;
        color:#1e293b;
        box-shadow:0 4px 12px rgba(0,0,0,0.08);
        transition:transform 0.15s ease, box-shadow 0.15s ease;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow:0 8px 20px rgba(0,0,0,0.12);
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
        background:#f8fafc;
        color:#1e293b;
        border:1px solid rgba(96, 165, 250, 0.5);
        border-radius:16px;
        padding:10px 22px;
        border:none;
        font-weight:600;
    }

    div.stButton > button:hover{
        background:#e0f2fe;
    }

    .class-card-container button{
        background:#f8fafc;
        color:#1e293b;
        border:1px solid rgba(96, 165, 250, 0.5);
        border-radius:16px;
        padding:30px;
        border:none;
        font-weight:600;
        font-size:18px;
        box-shadow:0 4px 12px rgba(0,0,0,0.08);
        transition:transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom:24px;
        height:240px;
        display:flex;
        flex-direction:column;
        justify-content:space-between;
        text-align:left;
        white-space:normal;
        overflow:hidden;
    }

    .class-card-container button::first-line{
        font-size:24px;
        font-weight:700;
    }

    .class-card-container button:hover{
        background:#f8fafc;
        transform: translateY(-2px);
        box-shadow:0 8px 20px rgba(0,0,0,0.12);
    }

    h1,h2,h3{
        color:#1e293b;
    }

    .stForm{
        background:#f8fafc;
        border:1px solid rgba(96, 165, 250, 0.5);
        padding:20px;
        border-radius:16px;
        margin-bottom:18px;
        color:#1e293b;
        box-shadow:0 4px 12px rgba(0,0,0,0.08);
    }

    </style>
    """, unsafe_allow_html=True)