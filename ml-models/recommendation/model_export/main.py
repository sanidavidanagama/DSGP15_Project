from recommendations_engine import RecommendationEngine
import json
# ============================================
# CSV INPUT PARSER
# ============================================uv

def parse_csv_input(input_string):

    values = [v.strip() for v in input_string.split(",")]

    mood = values[0]

    data = {
        "LinePressure": values[1],
        "ShadingIntensity": values[2],
        "OverallTone": values[3],
        "PageUsage": values[4],
        "FigureSize": values[5],
        "Placement": values[6],
        "HumanFigurePresent": values[7],
        "MissingBodyParts": values[8],
        "FacialFeatures": values[9],
        "NumberOfFigures": int(values[10]) if values[10] != "NA" else 1,
        "DistanceBetweenFigures": values[11] if values[11] != "NA" else None,
        "SelfPositioning": values[12] if values[12] != "NA" else None
    }

    return mood, data


# ============================================
# TEST WITH YOUR EXACT INPUT FORMAT
# ============================================

engine = RecommendationEngine()

csv_input = "Sad,Normal,Moderate,Balanced,Small,Small,Corner,Yes,None,Present,1,NA,NA"

mood, data = parse_csv_input(csv_input)

result = engine.generate_recommendation(mood, data)

print(json.dumps(result, indent=4))