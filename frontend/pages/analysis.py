import streamlit as st
from PIL import Image
import time

def analysis():

    st.title("Emotion Analysis")
    st.write("Upload a drawing and add optional description")

    col1,col2 = st.columns([1,1])

    with col1:

        st.markdown("<div class='upload-box'>Upload Drawing</div>",unsafe_allow_html=True)

        img = st.file_uploader("",type=["png","jpg","jpeg"])

        desc = st.text_area("Description (Optional)")

        if img:

            image = Image.open(img)
            st.image(image,use_container_width=True)

            if st.button("Analyze Emotions"):

                with st.spinner("Analyzing..."):
                    time.sleep(2)

                st.session_state.result = True

    with col2:

        st.markdown("<div class='card'>",unsafe_allow_html=True)

        if "result" not in st.session_state:

            st.write("Upload a drawing to see analysis results")

        else:

            st.subheader("Primary Emotion Detected")
            st.success("Happy")

            st.progress(0.87)
            st.write("Confidence 87%")

            st.subheader("Emotion Breakdown")

            st.write("Happy — 87%")
            st.progress(0.87)

            st.write("Excited — 65%")
            st.progress(0.65)

            st.write("Curious — 45%")
            st.progress(0.45)

            st.write("Calm — 30%")
            st.progress(0.30)

        st.markdown("</div>",unsafe_allow_html=True)