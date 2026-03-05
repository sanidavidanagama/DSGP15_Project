class RecommendationEngine:

    def __init__(self):

        self.recommendation_texts = {
            "Emotional Regulation Support":
                """High line pressure, heavy shading, and darker tonal balance may
                reflect emotionally intense expressive behaviour. Supportive
                guidance may focus on calming activities and structured
                emotional outlets (Koppitz, 1984; Machover, 1949).""",

            "Confidence Building Support":
                """Small page usage and reduced figure size may indicate
                constrained spatial engagement. Encouraging open-ended drawing
                and positive reinforcement may support expressive confidence
                (Buck, 1948; Koppitz, 1984).""",

            "Social Engagement Support":
                """Greater distance between figures or separate self-positioning
                may reflect relational distancing. Collaborative activities and
                guided social interaction may support engagement
                (Machover, 1949; Malchiodi, 1998).""",

            "Developmental Monitoring Suggestion":
                """Missing body parts or absent facial features may reflect
                developmental or expressive simplification. Observation over
                time is recommended rather than immediate concern
                (Koppitz, 1968).""",

            "Positive Reinforcement":
                """Balanced tone and organized structure may indicate regulated
                expression. Reinforcing positive expressive behaviour is
                recommended (Malchiodi, 2005).""",

            "Expressive Expansion Guidance":
                """Large page usage and confident spatial engagement may indicate
                strong expressive presence. Encouraging narrative expansion
                and creative exploration is beneficial (Betts, 1995).""",

            "Creative Encouragement":
                """Absence of strong emotional or spatial constraints may
                indicate thematic exploration. Encouraging imaginative
                storytelling and symbolic creativity can enhance development
                (Machover, 1949)."""
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

    def generate_recommendation(self, mood, data):

        patterns = self.detect_patterns(data)

        # Priority 1
        if patterns.get("structural") == "Structural Omission":
            category = "Developmental Monitoring Suggestion"

        # Priority 2
        elif (mood == "Sad" and
              patterns.get("emotional") == "High Emotional Intensity"):
            category = "Emotional Regulation Support"

        # Priority 3
        elif (mood == "Sad" and
              patterns.get("spatial") == "Constrained Spatial Usage"):
            category = "Confidence Building Support"

        # Priority 4
        elif patterns.get("relational") == "Relational Distance":
            category = "Social Engagement Support"

        # Positive Patterns
        elif (mood == "Happy" and
              patterns.get("emotional") == "Regulated Expression"):
            category = "Positive Reinforcement"

        elif (mood == "Happy" and
              patterns.get("spatial") == "Confident Spatial Engagement"):
            category = "Expressive Expansion Guidance"

        else:
            category = "Creative Encouragement"

        return {
            "DetectedPatterns": patterns,
            "RecommendationCategory": category,
            "RecommendationText": self.recommendation_texts[category]
        }


