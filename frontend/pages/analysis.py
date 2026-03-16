import streamlit as st
from PIL import Image
import time


def analysis():

    if "page" not in st.session_state:
        st.session_state.page = "upload"

    if st.session_state.page == "upload":
        upload_page()

    elif st.session_state.page == "loading":
        loading_page()

    elif st.session_state.page == "results":
        results_page()

def upload_page():

    st.title("Upload Drawing")

    st.markdown("<div class='upload-box'>Upload Child Drawing</div>", unsafe_allow_html=True)

    image = st.file_uploader("Drawing Image", type=["png","jpg","jpeg"])
    description = st.text_area("Drawing Description")

    if image:

        img = Image.open(image)
        st.image(img, width=300)

        if st.button("Start Analysis"):

            st.session_state.raw_image = img
            st.session_state.description = description

            st.session_state.page = "loading"
            st.rerun()

def loading_page():

    st.title("Analyzing Drawing")

    st.markdown(
        "<div class='loading-box'>Analyzing drawing... Please wait</div>",
        unsafe_allow_html=True
    )

    progress = st.progress(0)

    for i in range(100):
        time.sleep(0.02)
        progress.progress(i+1)

    generate_results()

    st.session_state.page = "results"
    st.rerun()

def generate_results():

    st.session_state.results = {

        "image":{
            "raw": st.session_state.raw_image,
            "processed": st.session_state.raw_image,
            "caption": "Child drew two smiling figures with bright colors"
        },

        "mood":{
            "mood":"Happy",
            "mood_probability":0.87
        },

        "dia":{
            "line_pressure":"Medium",
            "shading_intensity":"Low",
            "overall_tone":"Positive",
            "page_usage":"Full",
            "figure_size":"Large",
            "placement":"Center",
            "human_figure_present":"Yes",
            "missing_body_parts":"None",
            "facial_features":"Smiling",
            "number_of_figures":"2",
            "distance_between_figures":"Close",
            "self_positioning":"Center",
            "interpretation":[
                "Positive emotional expression",
                "Comfortable social representation",
                "Healthy emotional state"
            ]
        },

        "recommendations":{
            "detected_patterns":{
                "emotional":"Positive mood indicators",
                "spatial":"Balanced page usage",
                "structural":"Complete figures drawn"
            },

            "recommendation_category":"Encouragement",

            "recommendation_text":
            "Encourage the child to continue expressing emotions through art."
        }
    }

def results_page():

    st.title("Emotion Analysis Results")

    data = st.session_state.results

    # images
    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Raw Image")
        st.image(data["image"]["raw"])

    with col2:
        st.subheader("Processed Image")
        st.image(data["image"]["processed"])

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Caption")
    st.write(data["image"]["caption"])

    st.subheader("Mood Detection")

    st.write("Mood:", data["mood"]["mood"])
    st.progress(data["mood"]["mood_probability"])

    dia = data["dia"]

    st.subheader("Drawing Analysis")

    st.write("Line Pressure:", dia["line_pressure"])
    st.write("Shading Intensity:", dia["shading_intensity"])
    st.write("Overall Tone:", dia["overall_tone"])
    st.write("Page Usage:", dia["page_usage"])
    st.write("Figure Size:", dia["figure_size"])
    st.write("Placement:", dia["placement"])
    st.write("Human Figure:", dia["human_figure_present"])
    st.write("Missing Body Parts:", dia["missing_body_parts"])
    st.write("Facial Features:", dia["facial_features"])
    st.write("Number of Figures:", dia["number_of_figures"])
    st.write("Distance Between Figures:", dia["distance_between_figures"])
    st.write("Self Positioning:", dia["self_positioning"])

    st.subheader("Interpretation")

    for item in dia["interpretation"]:
        st.write("•", item)

    rec = data["recommendations"]

    st.subheader("Detected Patterns")

    st.write("Emotional:", rec["detected_patterns"]["emotional"])
    st.write("Spatial:", rec["detected_patterns"]["spatial"])
    st.write("Structural:", rec["detected_patterns"]["structural"])

    st.subheader("Recommendation")

    st.write("Category:", rec["recommendation_category"])
    st.info(rec["recommendation_text"])

    st.markdown("</div>", unsafe_allow_html=True)

    
                                