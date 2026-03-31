from pathlib import Path
import time
import html
import streamlit as st
from PIL import Image

from services.analysis_api import (
    AnalysisApiError,
    get_job_status,
    get_saved_report,
    upload_job,
    validate_image,
)
from services.class_api import ClassApiError, get_classes
from services.student_api import StudentApiError, list_students, save_analysis_to_student


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


def _render_hero(title: str, subtitle: str, chips: list[str]) -> None:
    chips_html = "".join(f"<span class='analysis-chip'>{chip}</span>" for chip in chips)
    st.markdown(
        f"""
        <div class='analysis-hero'>
            <h2 style='margin:0'>{title}</h2>
            <p class='analysis-subtitle'>{subtitle}</p>
            <div class='analysis-chip-row'>{chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _safe_text(value: object, fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp_percentage(value: float) -> float:
    return max(0.0, min(100.0, value))


def _render_mood_stack(left_label: str, left_pct: float, right_label: str, right_pct: float) -> None:
    st.markdown(
        f"""
        <div class='analysis-mood-stack'>
            <div class='analysis-mood-positive' style='width:{left_pct:.2f}%'></div>
            <div class='analysis-mood-support' style='width:{right_pct:.2f}%'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_left, col_right = st.columns(2)
    with col_left:
        st.caption(f"{left_label}: {left_pct:.1f}%")
    with col_right:
        st.caption(f"{right_label}: {right_pct:.1f}%")


def _render_kv_grid(items: list[tuple[str, object]]) -> None:
    html_items = []
    for key, value in items:
        safe_key = html.escape(str(key))
        safe_value = html.escape(_safe_text(value))
        html_items.append(
            "<div class='analysis-kv-item'>"
            f"<div class='analysis-kv-key'>{key}</div>"
            f"<div class='analysis-kv-value'>{_safe_text(value)}</div>"
            "</div>"
        )

    st.markdown(
        f"<div class='analysis-kv'>{''.join(html_items)}</div>",
        unsafe_allow_html=True,
    )


def _normalize_score(raw_score: float) -> float:
    # Support score formats in either [0, 1] or [0, 100].
    if raw_score > 1.0:
        return _clamp_percentage(raw_score)
    return _clamp_percentage(raw_score * 100.0)


def _compute_happy_sad_split(emotion_data: dict) -> tuple[float, float] | None:
    probabilities = emotion_data.get("probabilities")

    happy_raw = None
    sad_raw = None
    if isinstance(probabilities, dict):
        happy_raw = _to_float(probabilities.get("happy"))
        sad_raw = _to_float(probabilities.get("sad"))

    if happy_raw is None:
        happy_score = _to_float(emotion_data.get("happy_score"))
        if happy_score is not None:
            happy_raw = happy_score

    if happy_raw is None and sad_raw is None:
        return None

    if happy_raw is not None and sad_raw is None:
        happy_pct = _normalize_score(happy_raw)
        sad_pct = _clamp_percentage(100.0 - happy_pct)
        return happy_pct, sad_pct

    if happy_raw is None and sad_raw is not None:
        sad_pct = _normalize_score(sad_raw)
        happy_pct = _clamp_percentage(100.0 - sad_pct)
        return happy_pct, sad_pct

    happy_pct = _normalize_score(happy_raw or 0.0)
    sad_pct = _normalize_score(sad_raw or 0.0)

    total = happy_pct + sad_pct
    if total <= 0:
        return None

    happy_norm = _clamp_percentage((happy_pct / total) * 100.0)
    sad_norm = _clamp_percentage(100.0 - happy_norm)
    return happy_norm, sad_norm


def _format_processing_duration(duration_seconds: object) -> str:
    duration = _to_float(duration_seconds)
    if duration is None:
        return "N/A"
    if duration < 60:
        return f"{duration:.1f}s"
    minutes = int(duration // 60)
    seconds = duration % 60
    return f"{minutes}m {seconds:.1f}s"


def _loading_step_state(status: str, image_processed_elapsed: float) -> tuple[set[str], str]:
    if status == "processing":
        return set(), "processing"

    if status == "image_processed":
        if image_processed_elapsed < 3:
            return {"processing", "image_processed"}, "predicting_mood"
        if image_processed_elapsed < 6:
            return {"processing", "image_processed", "predicting_mood"}, "drawing_insights"
        return {
            "processing",
            "image_processed",
            "predicting_mood",
            "drawing_insights",
        }, "suggesting_recommendations"

    if status == "mood_predicted":
        return {"processing", "image_processed", "predicting_mood"}, "drawing_insights"

    if status == "drawing_insights_ready":
        return {"processing", "image_processed", "drawing_insights"}, "predicting_mood"

    if status == "analysis_ready":
        return {
            "processing",
            "image_processed",
            "predicting_mood",
            "drawing_insights",
        }, "suggesting_recommendations"

    if status == "recommendation_ready":
        return {
            "processing",
            "image_processed",
            "predicting_mood",
            "drawing_insights",
            "suggesting_recommendations",
        }, "done"

    if status == "done":
        return {
            "processing",
            "image_processed",
            "predicting_mood",
            "drawing_insights",
            "suggesting_recommendations",
            "done",
        }, "done"

    return set(), "processing"


def _calculate_progress_percentage(status: str, image_processed_elapsed: float, started_at: float) -> int:
    elapsed = max(0.0, time.time() - started_at)

    if status == "processing":
        return min(28, int(8 + elapsed * 2.4))

    if status == "image_processed":
        return min(58, int(34 + image_processed_elapsed * 6.5))

    if status == "mood_predicted":
        return 68

    if status == "drawing_insights_ready":
        return 74

    if status == "analysis_ready":
        return 84

    if status == "recommendation_ready":
        return 94

    if status == "done":
        return 100

    return min(20, int(6 + elapsed * 1.8))


def _render_loading_timeline(status: str, image_processed_elapsed: float) -> None:
    steps = [
        ("processing", "Getting your image ready", "We are checking and preparing your drawing."),
        ("image_processed", "Image is ready", "Your drawing has been cleaned up for a clearer read."),
        ("predicting_mood", "Understanding mood", "We are identifying the overall mood in the drawing."),
        ("drawing_insights", "Reviewing drawing details", "We are looking at key visual signs in the drawing."),
        ("suggesting_recommendations", "Preparing guidance", "We are creating personalized support suggestions."),
        ("done", "Finishing up", "Your report is being prepared for display."),
    ]

    done_steps, active_step = _loading_step_state(status, image_processed_elapsed)

    for step_id, title, caption in steps:
        css_class = "analysis-step"
        if step_id in done_steps:
            css_class += " done"
        elif step_id == active_step:
            css_class += " active"

        st.markdown(
            f"""
            <div class='{css_class}'>
                <p class='analysis-step-title'>{title}</p>
                <p class='analysis-step-caption'>{caption}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
    _render_hero(
        "Drawing Analysis Workspace",
        "Upload a child's drawing and generate an integrated report combining mood signals, drawing indicators, and actionable recommendations.",
        ["Guided Workflow", "Parallel AI Analysis", "Teacher-Friendly Output"],
    )

    col_left, col_right = st.columns([1.35, 1], gap="large")

    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='analysis-section-title'>Upload Child Drawing</h3>", unsafe_allow_html=True)
        st.markdown(
            "<div class='upload-box'>Drop a drawing here or browse from your device<br/><small>Supported: PNG, JPG, JPEG, WEBP, BMP</small></div>",
            unsafe_allow_html=True,
        )

        image = st.file_uploader(
            "Drawing Image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            help="Use clear and complete photos/scans for better analysis quality.",
        )

        description = st.text_area(
            "Context Notes",
            placeholder="Optional: classroom context, observed behavior, prompt used for drawing, or anything relevant.",
            height=120,
        )

        start_disabled = image is None
        if st.button("Start Full Analysis", type="primary", disabled=start_disabled):
            # New analysis run, not viewing a saved report
            st.session_state.viewing_saved_report = False
            st.session_state.pop("saved_report_context", None)
            image_bytes = image.getvalue() if image else b""

            if not image_bytes:
                st.error("Uploaded file is empty. Please choose a valid image.")
                st.markdown("</div>", unsafe_allow_html=True)
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
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

            if not validation_response.get("valid", False):
                st.error(validation_response.get("message", "Image was rejected by backend."))
                st.markdown("</div>", unsafe_allow_html=True)
                return

            st.success(validation_response.get("message", "Image accepted."))

            with st.spinner("Submitting analysis job..."):
                try:
                    job_id = upload_job(
                        image_name=image.name,
                        image_bytes=image_bytes,
                        content_type=image.type,
                        description=description,
                    )
                except AnalysisApiError as exc:
                    st.error(f"Failed to start analysis job: {exc}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

            st.session_state.raw_image_bytes = image_bytes
            st.session_state.raw_image_name = image.name
            st.session_state.job_id = job_id
            st.session_state.description = description
            st.session_state.analysis_started_at = time.time()
            st.session_state.image_processed_at = None

            st.session_state.analysis_page = "loading"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='analysis-list-card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='analysis-section-title'>What Happens Next</h3>", unsafe_allow_html=True)
        st.caption("Your upload passes through the following stages automatically.")
        st.markdown("1. Image preprocessing and quality checks")
        st.markdown("2. Mood prediction from processed drawing")
        st.markdown("3. Drawing indicator interpretation")
        st.markdown("4. Recommendation synthesis")
        st.markdown("5. Structured report generation")
        st.markdown("</div>", unsafe_allow_html=True)

        if image:
            st.markdown("<div class='analysis-list-card' style='margin-top:12px'>", unsafe_allow_html=True)
            st.markdown("<h3 class='analysis-section-title'>Preview</h3>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.caption(_safe_text(image.name))
            st.markdown("</div>", unsafe_allow_html=True)

def loading_page():
    _render_hero(
        "Analysis In Progress",
        "We are reviewing your drawing and preparing your report.",
        ["Progress Updates", "Working in the Background", "Your Report Is Coming Soon"],
    )

    job_id = st.session_state.get("job_id")
    if "analysis_started_at" not in st.session_state:
        st.session_state.analysis_started_at = time.time()

    if not job_id:
        st.error("No active job found. Please upload an image first.")
        if st.button("Back to Upload"):
            st.session_state.analysis_page = "upload"
            st.rerun()
        return

    # When viewing a previously saved report from a student profile,
    # fetch the finished report directly from the scoped report endpoint
    if st.session_state.get("viewing_saved_report"):
        ctx = st.session_state.get("saved_report_context") or {}
        class_id = ctx.get("class_id")
        student_id = ctx.get("student_id")
        if not isinstance(class_id, int) or not isinstance(student_id, int):
            st.error("Missing class or student context for saved report.")
            if st.button("Back to Upload"):
                st.session_state.analysis_page = "upload"
                st.rerun()
            return

        token = st.session_state.get("auth_token")
        try:
            status_response = get_saved_report(class_id=class_id, student_id=student_id, job_id=str(job_id), token=token)
        except AnalysisApiError as exc:
            st.error(f"Could not fetch saved report: {exc}")
            if st.button("Back to Upload"):
                st.session_state.analysis_page = "upload"
                st.rerun()
            return

        status_response = _normalize_result_paths(status_response)

        st.session_state.results = status_response.get("result", {})
        st.session_state.raw_image_path = status_response.get("raw_image_path")
        st.session_state.analysis_started_at_backend = status_response.get("analysis_started_at")
        st.session_state.analysis_finished_at_backend = status_response.get("analysis_finished_at")
        st.session_state.analysis_duration_seconds = status_response.get("analysis_duration_seconds")
        st.session_state.analysis_page = "results"
        st.rerun()
        return

    try:
        status_response = get_job_status(job_id)
    except AnalysisApiError as exc:
        st.error(f"Could not fetch job status: {exc}")
        if st.button("Back to Upload"):
            st.session_state.analysis_page = "upload"
            st.rerun()
        return

    status = status_response.get("status")
    if status == "image_processed" and st.session_state.get("image_processed_at") is None:
        st.session_state.image_processed_at = time.time()

    image_processed_at = st.session_state.get("image_processed_at")
    image_processed_elapsed = 0.0
    if image_processed_at is not None:
        image_processed_elapsed = max(0.0, time.time() - image_processed_at)

    progress_percentage = _calculate_progress_percentage(
        _safe_text(status, fallback="processing"),
        image_processed_elapsed,
        st.session_state.analysis_started_at,
    )

    st.markdown("<div class='loading-box'>", unsafe_allow_html=True)
    st.subheader("Current Progress")
    st.progress(progress_percentage / 100.0, text=f"{progress_percentage}% complete")
    _render_loading_timeline(_safe_text(status, fallback="processing"), image_processed_elapsed)
    st.markdown("</div>", unsafe_allow_html=True)

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

    if status != "done":
        time.sleep(1.2)
        st.rerun()
        return

    status_response = _normalize_result_paths(status_response)

    st.session_state.results = status_response.get("result", {})
    st.session_state.raw_image_path = status_response.get("raw_image_path")
    st.session_state.analysis_started_at_backend = status_response.get("analysis_started_at")
    st.session_state.analysis_finished_at_backend = status_response.get("analysis_finished_at")
    st.session_state.analysis_duration_seconds = status_response.get("analysis_duration_seconds")
    st.session_state.analysis_page = "results"
    st.rerun()

def results_page():
    _render_hero(
        "Integrated Analysis Report",
        "Review emotional signals, drawing indicators, and evidence-based recommendations in one view.",
        ["Image Insights", "Mood Summary", "Drawing Indicators", "Recommendation"],
    )

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
    analysis_started_at = st.session_state.get("analysis_started_at_backend")
    analysis_finished_at = st.session_state.get("analysis_finished_at_backend")
    analysis_duration_seconds = st.session_state.get("analysis_duration_seconds")

    happy_sad = _compute_happy_sad_split(emotion_data)
    if happy_sad:
        happy_pct, sad_pct = happy_sad
        mood_value = "Happy" if happy_pct >= sad_pct else "Sad"
    else:
        mood_value = emotion_data.get("predicted_mood") or emotion_data.get("emotion") or "N/A"

    top_metrics = st.columns(3)
    with top_metrics[0]:
        st.markdown(
            f"<div class='analysis-metric'><div class='analysis-metric-label'>Predicted Mood</div><div class='analysis-metric-value'>{_safe_text(mood_value)}</div></div>",
            unsafe_allow_html=True,
        )

    with top_metrics[1]:
        st.markdown(
            f"<div class='analysis-metric'><div class='analysis-metric-label'>Analysis Duration</div><div class='analysis-metric-value'>{_format_processing_duration(analysis_duration_seconds)}</div></div>",
            unsafe_allow_html=True,
        )

    with top_metrics[2]:
        st.markdown(
            f"<div class='analysis-metric'><div class='analysis-metric-label'>Recommendation Category</div><div class='analysis-metric-value'>{_safe_text(rec.get('RecommendationCategory'))}</div></div>",
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw Drawing")
        if st.session_state.get("raw_image_bytes"):
            st.image(st.session_state.raw_image_bytes, use_container_width=True)
        elif st.session_state.get("raw_image_path"):
            raw_image_path = st.session_state.raw_image_path
            if raw_image_path:
                try:
                    with Image.open(raw_image_path) as raw_img:
                        st.image(raw_img.copy(), use_container_width=True)
                except Exception:
                    st.write("Raw image file could not be opened.")
            else:
                st.write("Raw image path is unavailable from current frontend working directory.")
        else:
            st.write("Raw image not available.")

    with col2:
        st.subheader("Processed Drawing")
        processed_image_path = image_data.get("processed_image_path")
        if processed_image_path:
            try:
                with Image.open(processed_image_path) as processed_img:
                    st.image(processed_img.copy(), use_container_width=True)
            except Exception:
                st.write("Processed image file could not be opened.")
        else:
            st.write("Processed image not available.")

    st.subheader("Mood Detection")
    st.write(f"Predicted Mood: {_safe_text(mood_value)}")
    if happy_sad:
        happy_pct, sad_pct = happy_sad
        _render_mood_stack("Happy", happy_pct, "Sad", sad_pct)
        st.caption(f"Happy {happy_pct:.1f}% | Sad {sad_pct:.1f}%")

    time_cols = st.columns(2)
    with time_cols[0]:
        st.caption(f"Started: {_safe_text(analysis_started_at)}")
    with time_cols[1]:
        st.caption(f"Finished: {_safe_text(analysis_finished_at)}")

    st.subheader("Drawing Analysis")
    _render_kv_grid(
        [
            ("Line Pressure", dia.get("line_pressure")),
            ("Shading Intensity", dia.get("shading_intensity")),
            ("Overall Tone", dia.get("overall_tone")),
            ("Page Usage", dia.get("page_usage")),
            ("Figure Size", dia.get("figure_size")),
            ("Placement", dia.get("placement")),
            ("Human Figure", dia.get("human_figure_present")),
            ("Missing Body Parts", dia.get("missing_body_parts")),
            ("Facial Features", dia.get("facial_features")),
            ("Number of Figures", dia.get("number_of_figures")),
            ("Distance Between Figures", dia.get("distance_between_figures")),
            ("Self Positioning", dia.get("self_positioning")),
        ]
    )

    st.subheader("Interpretation")
    interpretation = dia.get("interpretation", [])
    if isinstance(interpretation, list) and interpretation:
        for item in interpretation:
            st.markdown(f"- {_safe_text(item)}")
    else:
        st.caption("No interpretation details were provided by the backend.")

    detected_patterns = rec.get("DetectedPatterns") or {}

    st.subheader("Detected Patterns")
    pattern_cols = st.columns(2)
    with pattern_cols[0]:
        st.caption("Emotional")
        st.write(_safe_text(detected_patterns.get("emotional")))
    with pattern_cols[1]:
        st.caption("Spatial")
        st.write(_safe_text(detected_patterns.get("spatial")))

    st.subheader("Recommendation")
    st.write("Category:", _safe_text(rec.get("RecommendationCategory")))
    st.info(rec.get("RecommendationText", "No recommendation text available."))

    # Only show "Save To Student Profile" for fresh analyses, not when
    # revisiting a saved report from a student profile.
    if not st.session_state.get("viewing_saved_report", False):
        st.subheader("Save To Student Profile")
        save_col_left, save_col_right = st.columns([1.4, 1.6])

        classes: list[dict] = []
        class_options: dict[str, int] = {}
        student_options: dict[str, int] = {}

        token = st.session_state.get("auth_token")

        try:
            classes = get_classes(token=token)
            class_options = {
                f"{_safe_text(item.get('class_name'))} ({_safe_text(item.get('grade_age_group'))})": int(item["id"])
                for item in classes
                if isinstance(item.get("id"), int)
            }
        except ClassApiError as exc:
            st.caption(f"Classes unavailable for save flow: {exc}")

        selected_class_id: int | None = None
        selected_student_id: int | None = None

        with save_col_left:
            if class_options:
                selected_class_label = st.selectbox("Select Class", options=list(class_options.keys()))
                selected_class_id = class_options.get(selected_class_label)
            else:
                st.caption("No class available.")

        with save_col_right:
            if selected_class_id is not None:
                try:
                    class_students = list_students(selected_class_id, token=token)
                except StudentApiError as exc:
                    st.caption(f"Students unavailable: {exc}")
                    class_students = []

                student_options = {
                    _safe_text(student.get("name")): int(student["id"])
                    for student in class_students
                    if isinstance(student.get("id"), int)
                }

                if student_options:
                    selected_student_label = st.selectbox("Select Student", options=list(student_options.keys()))
                    selected_student_id = student_options.get(selected_student_label)
                else:
                    st.caption("No students in this class yet.")

        if st.button("Save Result To Student", key="save_analysis_to_student", disabled=selected_student_id is None):
            if selected_student_id is None:
                st.error("Please select a student before saving.")
            else:
                current_job_id = st.session_state.get("job_id")
                if not current_job_id:
                    st.error("No job id found for this analysis result.")
                else:
                    try:
                        save_analysis_to_student(student_id=selected_student_id, job_id=str(current_job_id), token=token)
                    except StudentApiError as exc:
                        st.error(f"Failed to save analysis to student: {exc}")
                    else:
                        st.success("Analysis saved to student profile.")

    if st.button("Analyze Another Drawing"):
        st.session_state.analysis_page = "upload"
        st.session_state.pop("results", None)
        st.session_state.pop("job_id", None)
        st.session_state.pop("raw_image_bytes", None)
        st.session_state.pop("raw_image_path", None)
        st.session_state.pop("analysis_started_at", None)
        st.session_state.pop("image_processed_at", None)
        st.session_state.pop("analysis_started_at_backend", None)
        st.session_state.pop("analysis_finished_at_backend", None)
        st.session_state.pop("analysis_duration_seconds", None)
        st.session_state.pop("viewing_saved_report", None)
        st.session_state.pop("saved_report_context", None)
        st.rerun()


