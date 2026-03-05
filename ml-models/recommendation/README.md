# Recommendation Engine

A Python-based recommendation engine that analyzes children's drawings 

---

##  Overview

This engine combines **Drawing Indicator Analysis (DIA)** with mood input to detect expressive, spatial, structural, and relational patterns in a child's drawing. 
Based on these patterns, it generates a recommendation category and supporting text grounded in established art therapy and psychological literature.

---

##  How It Works

1. **Input** — A CSV-formatted string containing mood and drawing attributes
2. **Pattern Detection** — The engine analyzes four pattern domains:
   - **Emotional** — Line pressure, shading intensity, overall tone
   - **Spatial** — Page usage and figure placement
   - **Structural** — Missing body parts, presence of human figures
   - **Relational** — Number of figures, distance between figures, self-positioning
3. **Recommendation Generation** — Based on detected patterns and mood, the engine assigns a recommendation category with a clinical rationale

---

##  Usage

### Input Format (CSV String)

```
Mood, LinePressure, ShadingIntensity, OverallTone, PageUsage, FigureSize,
Placement, HumanFigurePresent, MissingBodyParts, FacialFeatures,
NumberOfFigures, DistanceBetweenFigures, SelfPositioning
```

### Example

```python
engine = RecommendationEngine()

csv_input = "Sad,High,Heavy,Dark,Small,Small,Corner,Yes,Hands,Absent,1,NA,NA"

mood, data = parse_csv_input(csv_input)
result = engine.generate_recommendation(mood, data)

print(json.dumps(result, indent=4))
```

### Example Output

```json
{
    "DetectedPatterns": {
        "emotional": "High Emotional Intensity",
        "spatial": "Constrained Spatial Usage",
        "structural": "Developmental Detail Omission"
    },
    "RecommendationCategory": "Developmental Monitoring Suggestion",
    "RecommendationText": "Missing body parts or absent facial features may reflect
        developmental or expressive simplification. Observation over
        time is recommended rather than immediate concern (Koppitz, 1968)."
}
```

---

## 📋 Input Field Reference

| Field | Valid Values |
|---|---|
| Mood | `Happy`, `Sad` |
| LinePressure | `Low`, `Normal`, `High` |
| ShadingIntensity | `None`, `Light`, `Heavy` |
| OverallTone | `Light`, `Neutral`, `Dark` |
| PageUsage | `Small`, `Medium`, `Large` |
| FigureSize | `Small`, `Medium`, `Large` |
| Placement | `Corner`, `Edge`, `Center` |
| HumanFigurePresent | `Yes`, `No` |
| MissingBodyParts | `None`, or body part name (e.g., `Hands`) |
| FacialFeatures | `Present`, `Absent` |
| NumberOfFigures | Integer, or `NA` |
| DistanceBetweenFigures | `Close`, `Far`, or `NA` |
| SelfPositioning | `With others`, `Separate`, or `NA` |

---

##  Recommendation Categories

| Category | Trigger Condition |
|---|---|
| Emotional Regulation Support | Sad mood + High Emotional Intensity |
| Confidence Building Support | Sad mood + Constrained Spatial Usage |
| Social Engagement Support | Sad mood + Relational Distance |
| Developmental Monitoring Suggestion | Missing body parts detected |
| Positive Reinforcement | Happy mood + Regulated Expression |
| Expressive Expansion Guidance | Happy mood + Confident Spatial Engagement |
| Creative Encouragement | Default / no strong pattern detected |

---
