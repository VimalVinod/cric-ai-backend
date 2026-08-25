"""
scoring.py
Turns biomechanical features (from biomechanics.py, evaluated at/near the
release frame) into:
  1. A per-parameter 0-100 score
  2. A weighted overall Technical Score
  3. Risk indicators (rule-based, explicitly NOT a medical claim)
  4. Rule-based recommendations

All reference ranges below are V1 placeholder heuristics based on general
fast-bowling coaching literature (e.g. front knee near-extension at front-foot
contact, minimal lateral trunk lean, high hip-shoulder separation for power).
They are NOT validated against a labelled coach dataset yet — see the
"Dataset & Validation" note in the README this pipeline ships with.
"""

# ---- V1 reference ranges: (ideal_low, ideal_high, unit) ----
REFERENCE_RANGES = {
    "frontKneeAngle": (155, 180),       # near-extension at front-foot contact
    "elbowAngle": (150, 180),           # legal/near-straight arm at release
    "trunkLateralFlexion": (0, 15),     # degrees of side-bend, lower is better
    "trunkForwardFlexion": (15, 40),    # degrees forward lean, moderate is ideal
    "hipShoulderSeparation": (20, 45),  # degrees, higher generally = more power
    "frontFootOffset": (0.0, 0.35),     # normalized, lower = straighter landing
    "headStability": (0.75, 1.0),       # 0-1 scale, higher = more stable
}

WEIGHTS = {
    "headStability": 0.10,
    "frontKneeAngle": 0.15,
    "frontFootOffset": 0.10,
    "trunkLateralFlexion": 0.15,
    "hipShoulderSeparation": 0.15,
    "elbowAngle": 0.15,
    "trunkForwardFlexion": 0.10,
    "backKneeAngle": 0.10,
}


def _score_in_range(value, low, high, higher_is_better_beyond_range=False):
    """Maps a value to 0-100 based on distance from an ideal [low, high] band."""
    if value is None:
        return None
    if low <= value <= high:
        return 100.0
    span = max(high - low, 1e-6)
    if value < low:
        dist = (low - value) / span
    else:
        dist = (value - high) / span
    score = max(0.0, 100.0 - dist * 100.0)
    return round(score, 1)


def score_parameters(release_features):
    """release_features: dict of biomechanical values at/near release frame."""
    param_scores = {}
    for key, (low, high) in REFERENCE_RANGES.items():
        val = release_features.get(key)
        param_scores[key] = {
            "value": None if val is None else round(val, 1),
            "score": _score_in_range(val, low, high),
        }
    return param_scores


def compute_technical_score(param_scores):
    total_weight = 0.0
    weighted_sum = 0.0
    for key, weight in WEIGHTS.items():
        s = param_scores.get(key, {}).get("score")
        if s is not None:
            weighted_sum += s * weight
            total_weight += weight
    if total_weight == 0:
        return None
    # Re-normalize in case some parameters were undetectable (missing frames/limbs)
    return round(weighted_sum / total_weight, 1)


RISK_RULES = [
    # (param, condition_fn, body_area, message)
    ("trunkLateralFlexion", lambda v: v is not None and v > 25,
     "Lower back", "High trunk lateral flexion detected — potential lower-back movement concern."),
    ("frontKneeAngle", lambda v: v is not None and v < 140,
     "Front knee", "Excessive front-knee flexion at landing — potential front-knee loading concern."),
    ("elbowAngle", lambda v: v is not None and v < 140,
     "Bowling elbow", "Significant elbow flexion near release — review for legality and loading concern."),
    ("frontFootOffset", lambda v: v is not None and v > 0.5,
     "Front foot / ankle", "Front foot landing significantly open/closed — potential ankle/knee alignment concern."),
    ("hipShoulderSeparation", lambda v: v is not None and v < 10,
     "Trunk / core", "Low hip-shoulder separation — may indicate reduced trunk rotation efficiency."),
]


def risk_indicators(release_features):
    flags = []
    for param, cond, area, message in RISK_RULES:
        val = release_features.get(param)
        if cond(val):
            flags.append({
                "parameter": param,
                "value": None if val is None else round(val, 1),
                "bodyArea": area,
                "severity": "monitor",
                "message": message,
            })
    return flags


RECOMMENDATION_RULES = [
    ("trunkLateralFlexion", lambda v: v is not None and v > 20,
     "Work on maintaining trunk stability during delivery — reduce side-on lean at release."),
    ("frontKneeAngle", lambda v: v is not None and v < 150,
     "Focus on a firmer, more extended front leg at front-foot contact to brace effectively."),
    ("frontFootOffset", lambda v: v is not None and v > 0.4,
     "Focus on consistent, straighter front-foot placement during the delivery stride."),
    ("headStability", lambda v: v is not None and v < 0.7,
     "Work on maintaining a stable head position through release — avoid excess head movement."),
    ("hipShoulderSeparation", lambda v: v is not None and v < 15,
     "Increase hip-shoulder separation (counter-rotation) to generate more efficient power."),
    ("elbowAngle", lambda v: v is not None and v < 150,
     "Review bowling-arm extension at release with your coach for both power and legality."),
]


def recommendations(release_features):
    recs = []
    for param, cond, message in RECOMMENDATION_RULES:
        val = release_features.get(param)
        if cond(val):
            recs.append(message)
    if not recs:
        recs.append("No significant technical concerns detected in this clip — maintain current technique.")
    return recs


def explain_finding(param, release_features, per_frame_features, release_frame_idx, frame_count):
    """Builds a What/Where/When/Why/Recommendation explanation block for one flagged parameter."""
    val = release_features.get(param)
    pct = None
    if release_frame_idx is not None and frame_count:
        pct = round(100 * release_frame_idx / max(frame_count - 1, 1))
    return {
        "what": param,
        "value": None if val is None else round(val, 1),
        "when_pct_through_action": pct,
        "frame_idx": release_frame_idx,
    }
