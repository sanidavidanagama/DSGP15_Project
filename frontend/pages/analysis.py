from pathlib import Path

import streamlit as st
from PIL import Image

from services.analysis_api import AnalysisApiError, upload_job, validate_image, poll_job_status


def _resolve_image_path(path_value: str | None) -> str | None:
    if not path_value:
        return None

    candidate = Path(path_value)
    if candidate.is_file():
        return str(candidate.resolve())

    # Backend may return paths relative to project root (e.g., uploads/processed/...).
    frontend_root = Path(__file__).resolve().parents[1]
    project_root = frontend_root.parent

    backend_root = project_root / "backend"

    for base in (frontend_root, project_root, backend_root):
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return str(resolved)

    return None


def _normalize_result_paths(status_response: dict) -> dict:
    result = status_response.get("result", {}) or {}
    image_data = result.get("image", {}) or {}

    processed_image_path = image_data.get("processed_image_path")
    resolved_processed_image_path = _resolve_image_path(processed_image_path)
    if resolved_processed_image_path:
        image_data["processed_image_path"] = resolved_processed_image_path

    result["image"] = image_data
    status_response["result"] = result

    raw_image_path = status_response.get("raw_image_path")
    resolved_raw_image_path = _resolve_image_path(raw_image_path)
    if resolved_raw_image_path:
        status_response["raw_image_path"] = resolved_raw_image_path

    return status_response


def analysis():

    if "analysis_page" not in st.session_state:
        st.session_state.analysis_page = "upload"

    if st.session_state.analysis_page == "upload":
        upload_page()

    elif st.session_state.analysis_page == "loading":
        loading_page()

    elif st.session_state.analysis_page == "results":
        results_page()

def upload_page():

    st.title("Upload Drawing")

    st.markdown("<div class='upload-box'>Upload Child Drawing</div>", unsafe_allow_html=True)

    image = st.file_uploader("Drawing Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    description = st.text_area("Drawing Description")

    if image:

        st.image(image, width=300)

        if st.button("Start Analysis"):

            image_bytes = image.getvalue()

            if not image_bytes:
                st.error("Uploaded file is empty. Please choose a valid image.")
                return

            with st.spinner("Validating image..."):
                try:
                    validation_response = validate_image(
                        image_name=image.name,
                        image_bytes=image_bytes,
                        content_type=image.type,
                    )
                except AnalysisApiError as exc:
                    st.error(f"Image validation failed: {exc}")
                    return

            if not validation_response.get("valid", False):
                st.error(validation_response.get("message", "Image was rejected by backend."))
                return

            st.success(validation_response.get("message", "Image accepted."))

            with st.spinner("Submitting job..."):
                try:
                    job_id = upload_job(
                        image_name=image.name,
                        image_bytes=image_bytes,
                        content_type=image.type,
                        description=description,
                    )
                except AnalysisApiError as exc:
                    st.error(f"Failed to start analysis job: {exc}")
                    return

            st.session_state.raw_image_bytes = image_bytes
            st.session_state.raw_image_name = image.name
            st.session_state.job_id = job_id
            st.session_state.description = description

            st.session_state.analysis_page = "loading"
            st.rerun()

def loading_page():

    st.title("Analyzing Drawing")

    st.markdown(
        "<div class='loading-box'>Analyzing drawing... Please wait</div>",
        unsafe_allow_html=True
    )

    job_id = st.session_state.get("job_id")

    if not job_id:
        st.error("No active job found. Please upload an image first.")
        if st.button("Back to Upload"):
            st.session_state.analysis_page = "upload"
            st.rerun()
        return

    with st.spinner("Waiting for backend results..."):
        try:
            status_response = poll_job_status(job_id)
        except AnalysisApiError as exc:
            st.error(f"Could not fetch job status: {exc}")
            if st.button("Back to Upload"):
                st.session_state.analysis_page = "upload"
                st.rerun()
            return

    status = status_response.get("status")

    if status == "failed":
        failure_message = "Processing failed."
        result = status_response.get("result")
        if isinstance(result, dict):
            failure_message = result.get("error", failure_message)

        st.error(failure_message)
        if st.button("Try Another Image"):
            st.session_state.analysis_page = "upload"
            st.rerun()
        return

    status_response = _normalize_result_paths(status_response)

    st.session_state.results = status_response.get("result", {})
    st.session_state.raw_image_path = status_response.get("raw_image_path")
    st.session_state.analysis_page = "results"
    st.rerun()

def results_page():

    st.title("Emotion Analysis Results")

    data = st.session_state.get("results", {})

    if not data:
        st.warning("No analysis results available yet.")
        if st.button("Back to Upload"):
            st.session_state.analysis_page = "upload"
            st.rerun()
        return

    image_data = data.get("image") or {}
    emotion_data = data.get("emotion") or {}
    dia = data.get("dia") or {}
    rec = data.get("recommendation") or {}

    # images
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Image")
        if st.session_state.get("raw_image_bytes"):
            st.image(st.session_state.raw_image_bytes)
        elif st.session_state.get("raw_image_path"):
            raw_image_path = st.session_state.raw_image_path
            if raw_image_path:
                try:
                    with Image.open(raw_image_path) as raw_img:
                        st.image(raw_img.copy())
                except Exception:
                    st.write("Raw image file could not be opened.")
            else:
                st.write("Raw image path is unavailable from current frontend working directory.")
        else:
            st.write("Raw image not available.")

    with col2:
        st.subheader("Processed Image")
        processed_image_path = image_data.get("processed_image_path")
        if processed_image_path:
            try:
                with Image.open(processed_image_path) as processed_img:
                    st.image(processed_img.copy())
            except Exception:
                st.write("Processed image file could not be opened.")
        else:
            st.write("Processed image not available.")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("Image Metadata")
    st.write("File Type:", image_data.get("file_type", "N/A"))
    st.write("Size (bytes):", image_data.get("size", "N/A"))
    st.write("Dimensions:", image_data.get("dimensions", "N/A"))
    st.write("Processed At:", image_data.get("created_at", "N/A"))

    st.subheader("Mood Detection")
    mood_value = emotion_data.get("predicted_mood") or emotion_data.get("emotion") or "N/A"
    st.write("Mood:", mood_value)

    happy_score = emotion_data.get("happy_score")
    if isinstance(happy_score, (int, float)):
        st.write("Happy Score:", round(float(happy_score), 3))
        st.progress(min(max(float(happy_score), 0.0), 1.0))

    probabilities = emotion_data.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        st.write("Emotion Probabilities:")
        st.json(probabilities)

    st.subheader("Drawing Analysis")
    st.write("Line Pressure:", dia.get("line_pressure", "N/A"))
    st.write("Shading Intensity:", dia.get("shading_intensity", "N/A"))
    st.write("Overall Tone:", dia.get("overall_tone", "N/A"))
    st.write("Page Usage:", dia.get("page_usage", "N/A"))
    st.write("Figure Size:", dia.get("figure_size", "N/A"))
    st.write("Placement:", dia.get("placement", "N/A"))
    st.write("Human Figure:", dia.get("human_figure_present", "N/A"))
    st.write("Missing Body Parts:", dia.get("missing_body_parts", "N/A"))
    st.write("Facial Features:", dia.get("facial_features", "N/A"))
    st.write("Number of Figures:", dia.get("number_of_figures", "N/A"))
    st.write("Distance Between Figures:", dia.get("distance_between_figures", "N/A"))
    st.write("Self Positioning:", dia.get("self_positioning", "N/A"))

    st.subheader("Interpretation")

    for item in dia.get("interpretation", []):
        st.write("•", item)

    detected_patterns = rec.get("DetectedPatterns") or {}

    st.subheader("Detected Patterns")
    st.write("Emotional:", detected_patterns.get("emotional", "N/A"))
    st.write("Spatial:", detected_patterns.get("spatial", "N/A"))

    st.subheader("Recommendation")
    st.write("Category:", rec.get("RecommendationCategory", "N/A"))
    st.info(rec.get("RecommendationText", "No recommendation text available."))

    if st.button("Analyze Another Drawing"):
        st.session_state.analysis_page = "upload"
        st.session_state.pop("results", None)
        st.session_state.pop("job_id", None)
        st.session_state.pop("raw_image_bytes", None)
        st.session_state.pop("raw_image_path", None)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


