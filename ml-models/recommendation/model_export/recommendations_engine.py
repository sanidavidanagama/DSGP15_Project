class RecommendationEngine:

    def __init__(self):

        self.recommendation_texts = {
            "Emotional Regulation Support":
                """Observed graphic intensity (e.g., strong line pressure,
                heavy shading, darker tonal balance) may reflect elevated
                expressive energy. Supportive emotional regulation
                activities may be beneficial.""",

            "Confidence Building Support":
                """Reduced spatial usage or constrained placement may suggest
                limited expressive expansion. Encouraging open-ended drawing
                and reinforcement may support confidence.""",

            "Social Engagement Support":
                """Relational distancing in drawings may indicate reduced
                social representation. Collaborative creative activities
                may enhance engagement.""",

            "Developmental Monitoring Suggestion":
                """Omission of structural elements may reflect developmental
                simplification. Longitudinal observation is recommended.""",

            "Positive Reinforcement":
                """Balanced structural and expressive indicators suggest
                adaptive representation. Reinforcing positive expression
                is encouraged.""",

            "Expressive Expansion Guidance":
                """Confident spatial engagement may reflect strong expressive
                presence. Encouraging narrative expansion is beneficial.""",

            "Creative Encouragement":
                """Absence of strong constraints suggests thematic exploration.
                Encouraging imaginative creativity is recommended."""
        }

    def detect_patterns(self, data):

        patterns = {}

        # Emotional Indicators
        if (data["LinePressure"] == "High" and
                data["ShadingIntensity"] == "Heavy" and
                data["OverallTone"] == "Dark"):
            patterns["emotional"] = "High Emotional Intensity"

        elif (data["LinePressure"] == "Low" and
              data["ShadingIntensity"] == "None" and
              data["OverallTone"] == "Light"):
            patterns["emotional"] = "Low Expressive Energy"

        elif data["LinePressure"] == "Normal":
            patterns["emotional"] = "Regulated Expression"

        # Spatial Indicators
        if (data["PageUsage"] == "Small" and
                data["Placement"] in ["Corner", "Side"]):
            patterns["spatial"] = "Constrained Spatial Usage"

        elif (data["PageUsage"] == "Large" and
              data["Placement"] == "Center"):
            patterns["spatial"] = "Confident Spatial Engagement"

        # Structural Indicators
        if data["MissingBodyParts"] != "None":
            patterns["structural"] = "Structural Omission"

        # Relational Indicators
        if data["NumberOfFigures"] > 1:
            if data["DistanceBetweenFigures"] == "Far":
                patterns["relational"] = "Relational Distance"

        return patterns


