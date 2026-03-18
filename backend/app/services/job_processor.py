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


def _default_dia_result() -> dict:
    return {
        "line_pressure": "Normal",
        "shading_intensity": "Moderate",
        "overall_tone": "Balanced",
        "page_usage": "Medium",
        "figure_size": "Average",
        "placement": "Center",
        "human_figure_present": "Yes",
        "missing_body_parts": "None",
        "facial_features": "Present",
        "number_of_figures": "2-3",
        "distance_between_figures": "Moderate",
        "self_positioning": "With others",
        "interpretation": [
            "Interpretation unavailable due to incomplete model output.",
            "Please retry the analysis.",
            "Observable indicators were partially recovered from the current output.",
            "Interpretive links were limited by incomplete structured JSON generation.",
            "No reliable additional conclusion can be made from this incomplete output.",
        ],
    }


def _is_fallback_dia_result(dia: dict) -> bool:
    interpretation = dia.get("interpretation")
    if not isinstance(interpretation, list) or not interpretation:
        return True
    first_line = str(interpretation[0]).strip().lower()
    return first_line.startswith("interpretation unavailable")


def _synthesize_interpretation(dia: dict, child_description: str) -> list[str]:
    """Build a concise non-clinical interpretation when model interpretation is missing."""
    overall_tone = str(dia.get("overall_tone", "Balanced")).strip()
    shading = str(dia.get("shading_intensity", "Moderate")).strip()
    page_usage = str(dia.get("page_usage", "Medium")).strip()
    placement = str(dia.get("placement", "Center")).strip()
    num_figures = str(dia.get("number_of_figures", "2")).strip()
    self_positioning = str(dia.get("self_positioning", "With others")).strip()
    distance = str(dia.get("distance_between_figures", "Moderate")).strip()
    facial = str(dia.get("facial_features", "Present")).strip()

    emotional_line = (
        f"The drawing shows a {overall_tone.lower()} visual tone with {shading.lower()} shading, "
        "which may suggest a steady style of expression in this activity."
    )

    spatial_line = (
        f"Spatial organization appears {page_usage.lower()} with {placement.lower()} placement and "
        f"{num_figures} figure(s), indicating a clear and structured scene layout."
    )

    social_line = (
        f"Figure spacing is {distance.lower()} and self-positioning is '{self_positioning}', "
        "which may reflect comfort representing relationships in shared space."
    )

    feature_line = (
        "Facial features are "
        f"{facial.lower()}, supporting readable social-emotional cues in the drawing."
    )

    desc = (child_description or "").strip().replace("\n", " ")
    if len(desc) > 140:
        desc = desc[:137].rstrip() + "..."
    description_line = (
        f"Child description context: {desc}" if desc else "Child description context was limited in this submission."
    )

    lines = [emotional_line, spatial_line, social_line, feature_line, description_line]
    return [str(x)[:220] for x in lines]


def _normalize_dia_result(candidate: dict) -> dict:
    base = _default_dia_result()

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

    interp = out.get("interpretation")
    if not isinstance(interp, list):
        interp = [str(interp)] if interp is not None else []
    interp = [str(x) for x in interp][:5]
    while len(interp) < 5:
        interp.append("Evidence was limited for this interpretation aspect in the model output.")
    interp = [x if x.strip() else "Evidence was limited for this interpretation aspect in the model output." for x in interp]
    out["interpretation"] = interp

    # Ensure string fields remain strings.
    for k, v in out.items():
        if k == "interpretation":
            continue
        out[k] = str(v)

    return out


def _parse_dia_json(raw: str) -> dict:
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

    logger.warning("DIA JSON parse failed; storing fallback result. raw_prefix=%s", (raw or "")[:200])
    return _default_dia_result()


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

    # Run image processor
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
            raw_result = pipeline.run(processed_image_path, description)
            dia_result = _parse_dia_json(raw_result)
            if _is_fallback_dia_result(dia_result):
                dia_result["interpretation"] = _synthesize_interpretation(dia_result, description)
                logger.warning("DIA fallback interpretation synthesized for job_id=%s", job_id)
        except Exception as exc:
            logger.exception("DIA task failed; using fallback for job_id=%s: %s", job_id, exc)
            dia_result = _default_dia_result()
            dia_result["interpretation"] = _synthesize_interpretation(dia_result, description)
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
    