class RecommendationInputBuilder:
    """
    Utility to build the input for RecommendationEngine from emotion and DIA results.
    """
    @staticmethod
    def build(emotion_result: dict, dia_result: dict) -> tuple:
        """
        Returns (mood, data_dict) for RecommendationEngine.
        You may need to adjust the mapping logic based on your actual emotion/DIA outputs.
        """
        # Example: extract mood from emotion_result
        mood = emotion_result.get("predicted_mood") or emotion_result.get("mood") or "Unknown"

        def _pick(*keys, default=None):
            for key in keys:
                if key in dia_result and dia_result.get(key) is not None:
                    return dia_result.get(key)
            return default

        def _num_figures(value) -> int:
            text = str(value or "").strip().lower()
            if text == "many":
                return 4
            if text.isdigit():
                return int(text)
            return 2

        # Example: extract drawing features from dia_result
        # These keys must match what RecommendationEngine expects
        data = {
            "LinePressure": _pick("line_pressure", "LinePressure", default="Normal"),
            "ShadingIntensity": _pick("shading_intensity", "ShadingIntensity", default="Moderate"),
            "OverallTone": _pick("overall_tone", "OverallTone", default="Balanced"),
            "PageUsage": _pick("page_usage", "PageUsage", default="Medium"),
            "FigureSize": _pick("figure_size", "FigureSize", default="Average"),
            "Placement": _pick("placement", "Placement", default="Center"),
            "HumanFigurePresent": _pick("human_figure_present", "HumanFigurePresent", default="Yes"),
            "MissingBodyParts": _pick("missing_body_parts", "MissingBodyParts", default="None"),
            "FacialFeatures": _pick("facial_features", "FacialFeatures", default="Present"),
            "NumberOfFigures": _num_figures(_pick("number_of_figures", "NumberOfFigures", default="2")),
            "DistanceBetweenFigures": _pick("distance_between_figures", "DistanceBetweenFigures", default="Moderate"),
            "SelfPositioning": _pick("self_positioning", "SelfPositioning", default="With others"),
        }
        return mood, data
