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

        # Example: extract drawing features from dia_result
        # These keys must match what RecommendationEngine expects
        data = {
            "LinePressure": dia_result.get("LinePressure", "Normal"),
            "ShadingIntensity": dia_result.get("ShadingIntensity", "Moderate"),
            "OverallTone": dia_result.get("OverallTone", "Balanced"),
            "PageUsage": dia_result.get("PageUsage", "Small"),
            "FigureSize": dia_result.get("FigureSize", "Small"),
            "Placement": dia_result.get("Placement", "Corner"),
            "HumanFigurePresent": dia_result.get("HumanFigurePresent", "Yes"),
            "MissingBodyParts": dia_result.get("MissingBodyParts", "None"),
            "FacialFeatures": dia_result.get("FacialFeatures", "Present"),
            "NumberOfFigures": int(dia_result.get("NumberOfFigures", 1)),
            "DistanceBetweenFigures": dia_result.get("DistanceBetweenFigures", None),
            "SelfPositioning": dia_result.get("SelfPositioning", None)
        }
        return mood, data
