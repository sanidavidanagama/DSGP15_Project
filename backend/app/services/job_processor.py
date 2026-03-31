from sqlalchemy.orm import Session
from app.database.crud_job import update_job_status_and_result
from app.services.image_service import run_image_processor, build_image_metadata

from app.services.emotion_service import run_emotion_pipeline
from app.core.config import settings
import threading
import json
import time
import logging

# Import DIARagPipeline from the correct path


from app.ml.dia_model.dia_rag_pipeline import DrawingIndicatorAnalyser
from app.ml.dia_model.config import RagConfig

# Recommendation engine imports
from app.ml.recommendation_model.recommendations_engine import RecommendationEngine
from app.utils.recommendation_input_builder import RecommendationInputBuilder

logger = logging.getLogger(__name__)

_dia_pipeline = None
_dia_pipeline_lock = threading.Lock()


def _normalize_dia_result(candidate: dict):
    """Normalize DIA result keys without inventing interpretation text.

    We map legacy key names but leave the model's `interpretation` untouched
    (aside from basic type/length safety). This ensures all narrative content
    comes only from the DIA RAG / VLM.
    """

    base: dict[str, object] = {}

    # Accept either snake_case or legacy PascalCase keys.
    key_map = {
        "line_pressure": ["line_pressure", "LinePressure"],
        "shading_intensity": ["shading_intensity", "ShadingIntensity"],
        "overall_tone": ["overall_tone", "OverallTone"],
        "page_usage": ["page_usage", "PageUsage"],
        "figure_size": ["figure_size", "FigureSize"],
        "placement": ["placement", "Placement"],
        "human_figure_present": ["human_figure_present", "HumanFigurePresent"],
        "missing_body_parts": ["missing_body_parts", "MissingBodyParts"],
        "facial_features": ["facial_features", "FacialFeatures"],
        "number_of_figures": ["number_of_figures", "NumberOfFigures"],
        "distance_between_figures": ["distance_between_figures", "DistanceBetweenFigures"],
        "self_positioning": ["self_positioning", "SelfPositioning"],
        "interpretation": ["interpretation", "Interpretation"],
    }

    out = dict(base)
    for target_key, aliases in key_map.items():
        for alias in aliases:
            if alias in candidate and candidate[alias] is not None:
                out[target_key] = candidate[alias]
                break

    # If any required structural keys are missing, treat result as unusable.
    required_keys = [
        "line_pressure",
        "shading_intensity",
        "overall_tone",
        "page_usage",
        "figure_size",
        "placement",
        "human_figure_present",
        "missing_body_parts",
        "facial_features",
        "number_of_figures",
        "distance_between_figures",
        "self_positioning",
    ]
    missing = [k for k in required_keys if k not in out]
    if missing:
        logger.warning(
            "DIA JSON missing required keys %s; discarding partial result.",
            ", ".join(missing),
        )
        return None

    # Keep the model's interpretation as-is, only enforcing list + max length.
    interp = out.get("interpretation")
    if not isinstance(interp, list):
        interp = [str(interp)] if interp is not None else []
    interp = [str(x) for x in interp][:5]
    out["interpretation"] = interp

    # Ensure string fields remain strings.
    for k, v in out.items():
        if k == "interpretation":
            continue
        out[k] = str(v)

    return out


def _parse_dia_json(raw: str):
    def _repair_json_candidate(s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        if not s:
            return ""

        # Drop markdown fences if present.
        if s.startswith("```"):
            s = s.strip("`").strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()

        # Keep from first object start.
        start = s.find("{")
        if start == -1:
            return ""
        s = s[start:]

        # Keep up to last object close if present.
        end = s.rfind("}")
        if end != -1:
            s = s[: end + 1]

        # If braces are unbalanced, close missing object braces.
        open_count = s.count("{")
        close_count = s.count("}")
        if open_count > close_count:
            s += "}" * (open_count - close_count)

        # Remove trailing commas before object/array close.
        s = s.replace(",}", "}").replace(",]", "]")
        return s.strip()

    # Try direct parse first.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return _normalize_dia_result(obj)
    except json.JSONDecodeError:
        pass

    # Try best-effort extraction/repair of JSON object block.
    repaired = _repair_json_candidate(raw or "")
    if repaired:
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return _normalize_dia_result(obj)
        except json.JSONDecodeError:
            pass

    logger.warning("DIA JSON parse failed; treating DIA result as missing. raw_prefix=%s", (raw or "")[:200])
    return None


def get_dia_pipeline() -> DrawingIndicatorAnalyser:
    global _dia_pipeline
    if _dia_pipeline is not None:
        return _dia_pipeline

    with _dia_pipeline_lock:
        if _dia_pipeline is None:
            rag_config = RagConfig.from_settings()
            _dia_pipeline = DrawingIndicatorAnalyser(rag_config)
    return _dia_pipeline

def process_job(job_id: str, image_path: str, description: str, db: Session):
    # Clear status at start
    update_job_status_and_result(db, job_id, status="processing", result=None, processed_image_path=None)

    # Run image processor for downstream models that need it
    processed_image_path = run_image_processor(image_path)
    if not processed_image_path:
        update_job_status_and_result(db, job_id, status="failed", result={"error": "Image processing failed"})
        return

    update_job_status_and_result(db, job_id, status="image_processed", processed_image_path=processed_image_path)

    # Prepare results containers
    emotion_result = {}
    dia_result = {}
    recommendation_result = {}
    mood_done_event = threading.Event()
    dia_done_event = threading.Event()

    # Threaded tasks
    def emotion_task():
        nonlocal emotion_result
        emotion_result = run_emotion_pipeline(processed_image_path, description)
        mood_done_event.set()

    def dia_task():
        nonlocal dia_result
        try:
            pipeline = get_dia_pipeline()
            # Use the original uploaded image for DIA RAG analysis
            raw_result = pipeline.run(image_path, description)
            dia_result = _parse_dia_json(raw_result)
        except Exception as exc:
            logger.exception("DIA task failed for job_id=%s: %s", job_id, exc)
            dia_result = None
        finally:
            dia_done_event.set()

    # Run emotion and DIA in parallel first
    threads = [
        threading.Thread(target=emotion_task),
        threading.Thread(target=dia_task)
    ]
    for t in threads:
        t.start()

    mood_status_reported = False
    dia_status_reported = False

    # Report intermediate completion in near real-time while both threads run.
    while any(t.is_alive() for t in threads):
        if mood_done_event.is_set() and not mood_status_reported:
            update_job_status_and_result(db, job_id, status="mood_predicted")
            mood_status_reported = True

        if dia_done_event.is_set() and not dia_status_reported:
            update_job_status_and_result(db, job_id, status="drawing_insights_ready")
            dia_status_reported = True

        time.sleep(0.2)

    for t in threads:
        t.join()

    if mood_done_event.is_set() and not mood_status_reported:
        update_job_status_and_result(db, job_id, status="mood_predicted")

    if dia_done_event.is_set() and not dia_status_reported:
        update_job_status_and_result(db, job_id, status="drawing_insights_ready")

    update_job_status_and_result(db, job_id, status="analysis_ready")

    # Now run recommendation engine (can be threaded if needed, but depends on emotion/dia)
    def recommendation_task():
        nonlocal recommendation_result
        try:
            engine = RecommendationEngine()
            mood, data = RecommendationInputBuilder.build(emotion_result, dia_result)
            recommendation_result = engine.generate_recommendation(mood, data)

            if not isinstance(recommendation_result, dict):
                raise ValueError("Recommendation engine returned non-dict payload")

            detected = recommendation_result.get("DetectedPatterns")
            if not isinstance(detected, dict):
                detected = {}
            recommendation_result["DetectedPatterns"] = {
                "emotional": str(detected.get("emotional", "Regulated Expression")),
                "spatial": str(detected.get("spatial", "Constrained Spatial Usage")),
            }
            recommendation_result.setdefault("RecommendationCategory", "Creative Encouragement")
            recommendation_result.setdefault(
                "RecommendationText",
                "Unable to compute specific recommendation with current inputs.",
            )
        except Exception as exc:
            logger.exception("Recommendation generation failed: %s", exc)
            recommendation_result = {
                "DetectedPatterns": {
                    "emotional": "Regulated Expression",
                    "spatial": "Constrained Spatial Usage",
                },
                "RecommendationCategory": "Creative Encouragement",
                "RecommendationText": "Unable to compute specific recommendation with current inputs.",
            }

    rec_thread = threading.Thread(target=recommendation_task)
    rec_thread.start()
    rec_thread.join()

    update_job_status_and_result(db, job_id, status="recommendation_ready")

    # Aggregate results
    image_metadata = build_image_metadata(processed_image_path)

    result = {
        "image": image_metadata,
        "emotion": emotion_result,
        "dia": dia_result,
        "recommendation": recommendation_result
    }
    update_job_status_and_result(db, job_id, status="done", result=result)
    