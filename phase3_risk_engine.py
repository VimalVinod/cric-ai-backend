"""
phase3_risk_engine.py
=====================

Bowling Biomechanics Analysis System
PHASE 3 - RISK ENGINE

Purpose:
    Convert Phase-2 biomechanical measurements into
    structured injury-risk indicators.

Phase 3 uses:

    - Phase-2 measurements
    - View classification
    - View-aware reliability
    - Existing technical score

Risk levels:

    LOW
    MODERATE
    HIGH
    CRITICAL
"""

import argparse
import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "PHASE-3"

SUPPORTED_VIEWS = [
    "front",
    "rear",
    "side",
    "diagonal",
    "unknown",
]


# ============================================================
# HELPERS
# ============================================================

def clamp(value, minimum=0.0, maximum=1.0):

    return max(
        minimum,
        min(
            maximum,
            float(value)
        )
    )


def severity_rank(severity):

    ranks = {
        "low": 1,
        "moderate": 2,
        "high": 3,
        "critical": 4,
    }

    return ranks.get(
        severity,
        0
    )


def confidence_label(reliability):

    if reliability >= 0.80:
        return "high"

    if reliability >= 0.60:
        return "moderate"

    if reliability >= 0.40:
        return "low"

    return "unreliable"


# ============================================================
# RISK THRESHOLD ENGINE
# ============================================================

def evaluate_measurement(
    parameter,
    value
):

    if value is None:
        return None

    value = float(value)

    # --------------------------------------------------------
    # ELBOW
    # --------------------------------------------------------

    if parameter == "elbowAngle":

        if value < 120:

            return {
                "parameter": parameter,
                "bodyArea": "Bowling elbow",
                "value": round(value, 3),
                "severity": "high",
                "message":
                    "Very low elbow angle detected near the "
                    "delivery phase; review bowling-arm "
                    "mechanics with a qualified coach.",
            }

        elif value < 130:

            return {
                "parameter": parameter,
                "bodyArea": "Bowling elbow",
                "value": round(value, 3),
                "severity": "moderate",
                "message":
                    "Reduced elbow angle detected near the "
                    "delivery phase; monitor bowling-arm "
                    "mechanics.",
            }

        elif value < 140:

            return {
                "parameter": parameter,
                "bodyArea": "Bowling elbow",
                "value": round(value, 3),
                "severity": "low",
                "message":
                    "Some elbow flexion is present; continue "
                    "monitoring consistency.",
            }

        return {
            "parameter": parameter,
            "bodyArea": "Bowling elbow",
            "value": round(value, 3),
            "severity": "low",
            "message":
                "No major elbow-angle warning detected.",
        }

    # --------------------------------------------------------
    # TRUNK LATERAL FLEXION
    # --------------------------------------------------------

    if parameter == "trunkLateralFlexion":

        absolute_value = abs(value)

        if absolute_value >= 35:

            return {
                "parameter": parameter,
                "bodyArea": "Lower back",
                "value": round(value, 3),
                "severity": "high",
                "message":
                    "High lateral trunk flexion detected; "
                    "this may increase lower-back loading.",
            }

        elif absolute_value >= 25:

            return {
                "parameter": parameter,
                "bodyArea": "Lower back",
                "value": round(value, 3),
                "severity": "moderate",
                "message":
                    "Elevated lateral trunk flexion detected; "
                    "monitor trunk stability during delivery.",
            }

        elif absolute_value >= 15:

            return {
                "parameter": parameter,
                "bodyArea": "Lower back",
                "value": round(value, 3),
                "severity": "low",
                "message":
                    "Mild lateral trunk movement detected.",
            }

        return {
            "parameter": parameter,
            "bodyArea": "Lower back",
            "value": round(value, 3),
            "severity": "low",
            "message":
                "No major lateral trunk warning detected.",
        }

    # --------------------------------------------------------
    # TRUNK FORWARD FLEXION
    # --------------------------------------------------------

    if parameter == "trunkForwardFlexion":

        absolute_value = abs(value)

        if absolute_value >= 55:

            return {
                "parameter": parameter,
                "bodyArea": "Lower back",
                "value": round(value, 3),
                "severity": "high",
                "message":
                    "High forward trunk flexion detected; "
                    "review trunk position during delivery.",
            }

        elif absolute_value >= 40:

            return {
                "parameter": parameter,
                "bodyArea": "Lower back",
                "value": round(value, 3),
                "severity": "moderate",
                "message":
                    "Elevated forward trunk flexion detected; "
                    "monitor delivery posture.",
            }

        elif absolute_value >= 25:

            return {
                "parameter": parameter,
                "bodyArea": "Lower back",
                "value": round(value, 3),
                "severity": "low",
                "message":
                    "Moderate forward trunk movement detected.",
            }

        return {
            "parameter": parameter,
            "bodyArea": "Lower back",
            "value": round(value, 3),
            "severity": "low",
            "message":
                "No major forward trunk warning detected.",
        }

    # --------------------------------------------------------
    # FRONT KNEE
    # --------------------------------------------------------

    if parameter == "frontKneeAngle":

        if value < 120:

            return {
                "parameter": parameter,
                "bodyArea": "Front knee",
                "value": round(value, 3),
                "severity": "high",
                "message":
                    "Deep front-knee flexion detected; "
                    "review front-leg loading with a qualified coach.",
            }

        elif value < 135:

            return {
                "parameter": parameter,
                "bodyArea": "Front knee",
                "value": round(value, 3),
                "severity": "moderate",
                "message":
                    "Increased front-knee flexion detected; "
                    "monitor front-leg loading.",
            }

        return {
            "parameter": parameter,
            "bodyArea": "Front knee",
            "value": round(value, 3),
            "severity": "low",
            "message":
                "No major front-knee warning detected.",
        }

    # --------------------------------------------------------
    # BACK KNEE
    # --------------------------------------------------------

    if parameter == "backKneeAngle":

        if value < 120:

            return {
                "parameter": parameter,
                "bodyArea": "Back knee",
                "value": round(value, 3),
                "severity": "high",
                "message":
                    "High back-knee flexion detected; "
                    "monitor lower-limb loading.",
            }

        elif value < 135:

            return {
                "parameter": parameter,
                "bodyArea": "Back knee",
                "value": round(value, 3),
                "severity": "moderate",
                "message":
                    "Increased back-knee flexion detected.",
            }

        return {
            "parameter": parameter,
            "bodyArea": "Back knee",
            "value": round(value, 3),
            "severity": "low",
            "message":
                "No major back-knee warning detected.",
        }

    # --------------------------------------------------------
    # HIP-SHOULDER SEPARATION
    # --------------------------------------------------------

    if parameter == "hipShoulderSeparation":

        if value < 5:

            return {
                "parameter": parameter,
                "bodyArea": "Hip and trunk",
                "value": round(value, 3),
                "severity": "moderate",
                "message":
                    "Low hip-shoulder separation detected; "
                    "review trunk rotation and sequencing.",
            }

        elif value < 10:

            return {
                "parameter": parameter,
                "bodyArea": "Hip and trunk",
                "value": round(value, 3),
                "severity": "low",
                "message":
                    "Limited hip-shoulder separation detected.",
            }

        return {
            "parameter": parameter,
            "bodyArea": "Hip and trunk",
            "value": round(value, 3),
            "severity": "low",
            "message":
                "No major hip-shoulder separation warning detected.",
        }

    # --------------------------------------------------------
    # FRONT FOOT OFFSET
    # --------------------------------------------------------

    if parameter == "frontFootOffset":

        if value > 0.80:

            return {
                "parameter": parameter,
                "bodyArea": "Front foot",
                "value": round(value, 3),
                "severity": "moderate",
                "message":
                    "Large front-foot offset detected; "
                    "review landing stability.",
            }

        return {
            "parameter": parameter,
            "bodyArea": "Front foot",
            "value": round(value, 3),
            "severity": "low",
            "message":
                "No major front-foot offset warning detected.",
        }

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return None


# ============================================================
# CONFIDENCE-AWARE RISK
# ============================================================

def apply_reliability(
    risk,
    reliability
):

    if risk is None:
        return None

    parameter = risk["parameter"]

    rel = float(
        reliability.get(
            parameter,
            0.0
        )
    )

    risk["reliability"] = round(
        rel,
        3
    )

    risk["confidence"] = confidence_label(
        rel
    )

    # --------------------------------------------------------
    # Very low reliability
    # --------------------------------------------------------

    if rel < 0.40:

        risk["severity"] = "low"

        risk["confidence"] = "unreliable"

        risk["message"] += (
            " However, this measurement has insufficient "
            "confidence from the available camera view."
        )

    # --------------------------------------------------------
    # Low reliability
    # --------------------------------------------------------

    elif rel < 0.60:

        if risk["severity"] == "high":
            risk["severity"] = "moderate"

        risk["message"] += (
            " Interpret this finding cautiously because "
            "measurement confidence is limited."
        )

    return risk


# ============================================================
# OVERALL RISK LEVEL
# ============================================================

def calculate_overall_risk(
    risks,
    technical_score
):

    if not risks:

        if technical_score is not None:

            if technical_score < 50:
                return "moderate"

            return "low"

        return "low"

    highest = max(
        severity_rank(
            risk["severity"]
        )
        for risk in risks
    )

    high_count = sum(
        1
        for risk in risks
        if risk["severity"] == "high"
    )

    moderate_count = sum(
        1
        for risk in risks
        if risk["severity"] == "moderate"
    )

    # --------------------------------------------------------
    # Critical combination
    # --------------------------------------------------------

    if high_count >= 2:

        return "critical"

    # --------------------------------------------------------
    # High risk
    # --------------------------------------------------------

    if highest >= 3:

        return "high"

    # --------------------------------------------------------
    # Multiple moderate indicators
    # --------------------------------------------------------

    if moderate_count >= 2:

        return "moderate"

    return "low"


# ============================================================
# RISK SUMMARY
# ============================================================

def build_risk_summary(
    overall_risk,
    risks
):

    if overall_risk == "critical":

        return (
            "Multiple high-risk biomechanical indicators "
            "were detected. Review the delivery with a "
            "qualified cricket coach or sports professional."
        )

    if overall_risk == "high":

        return (
            "One or more significant biomechanical risk "
            "indicators were detected. Further technical "
            "review is recommended."
        )

    if overall_risk == "moderate":

        return (
            "Some biomechanical indicators require "
            "monitoring and technical review."
        )

    if risks:

        return (
            "Minor biomechanical indicators were detected. "
            "Continue monitoring consistency across deliveries."
        )

    return (
        "No major biomechanical risk indicators were detected."
    )


# ============================================================
# PHASE 3 ANALYSIS
# ============================================================

def analyze_phase3(
    phase2_data
):

    if phase2_data.get("status") != "ok":

        raise ValueError(
            "Invalid Phase-2 result."
        )

    measurements = phase2_data.get(
        "measurements",
        {}
    )

    reliability = phase2_data.get(
        "reliability",
        {}
    )

    view_info = phase2_data.get(
        "view",
        {}
    )

    detected_view = view_info.get(
        "detected",
        "unknown"
    )

    technical_score = phase2_data.get(
        "technicalScore"
    )

    # --------------------------------------------------------
    # Evaluate every available measurement
    # --------------------------------------------------------

    risks = []

    for parameter, value in measurements.items():

        risk = evaluate_measurement(
            parameter,
            value
        )

        if risk is None:
            continue

        risk = apply_reliability(
            risk,
            reliability
        )

        # Only retain actual warnings.
        if risk["severity"] != "low":

            risks.append(
                risk
            )

    # --------------------------------------------------------
    # Sort highest risk first
    # --------------------------------------------------------

    risks.sort(
        key=lambda item:
            severity_rank(
                item["severity"]
            ),
        reverse=True
    )

    # --------------------------------------------------------
    # Overall risk
    # --------------------------------------------------------

    overall_risk = calculate_overall_risk(
        risks,
        technical_score
    )

    summary = build_risk_summary(
        overall_risk,
        risks
    )

    # --------------------------------------------------------
    # Risk counts
    # --------------------------------------------------------

    risk_counts = {

        "critical": 0,

        "high": 0,

        "moderate": 0,

        "low": 0,
    }

    for risk in risks:

        severity = risk[
            "severity"
        ]

        if severity in risk_counts:

            risk_counts[
                severity
            ] += 1

    # --------------------------------------------------------
    # Phase 3 result
    # --------------------------------------------------------

    result = {

        "status": "ok",

        "version": VERSION,

        "source": {

            "phase2Version":
                phase2_data.get(
                    "version"
                ),

            "video":
                phase2_data.get(
                    "video",
                    {}
                ).get(
                    "path"
                ),

            "view":
                detected_view,

            "viewQuality":
                view_info.get(
                    "quality"
                ),

            "technicalScore":
                technical_score,
        },

        "overallRisk":
            overall_risk,

        "riskSummary":
            summary,

        "riskCounts":
            risk_counts,

        "riskIndicators":
            risks,

        "recommendations": [],
    }

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendation_set = set()

    for risk in risks:

        parameter = risk[
            "parameter"
        ]

        if parameter == "trunkLateralFlexion":

            recommendation_set.add(
                "Work on maintaining trunk stability "
                "through the delivery and follow-through."
            )

        elif parameter == "trunkForwardFlexion":

            recommendation_set.add(
                "Review trunk posture and forward flexion "
                "during the delivery phase."
            )

        elif parameter == "elbowAngle":

            recommendation_set.add(
                "Review bowling-arm movement and elbow "
                "position with a qualified coach."
            )

        elif parameter in [
            "frontKneeAngle",
            "backKneeAngle"
        ]:

            recommendation_set.add(
                "Review lower-limb loading and knee position "
                "during the delivery stride."
            )

        elif parameter == "hipShoulderSeparation":

            recommendation_set.add(
                "Review hip-shoulder sequencing and rotational "
                "mechanics with a qualified coach."
            )

        elif parameter == "frontFootOffset":

            recommendation_set.add(
                "Review front-foot landing stability and "
                "alignment."
            )

    if not recommendation_set:

        recommendation_set.add(
            "No major biomechanical flags were identified "
            "in this delivery window."
        )

        recommendation_set.add(
            "Continue monitoring consistency across "
            "multiple deliveries."
        )

    result["recommendations"] = list(
        recommendation_set
    )

    return result


# ============================================================
# FILE ANALYSIS
# ============================================================

def analyze_file(
    input_path,
    output_path=None
):

    if not os.path.exists(
        input_path
    ):

        raise FileNotFoundError(
            f"Phase-2 JSON not found: {input_path}"
        )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        phase2_data = json.load(
            file
        )

    result = analyze_phase3(
        phase2_data
    )

    if output_path is None:

        base, _ = os.path.splitext(
            input_path
        )

        output_path = (
            base +
            "_phase3.json"
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 3 - RISK ENGINE")
    print("=" * 60)

    print()

    print(
        f"View          : "
        f"{result['source']['view']}"
    )

    print(
        f"Technical     : "
        f"{result['source']['technicalScore']}"
    )

    print(
        f"Overall risk  : "
        f"{result['overallRisk'].upper()}"
    )

    print()

    print("RISK INDICATORS")
    print("-" * 60)

    if result["riskIndicators"]:

        for risk in result[
            "riskIndicators"
        ]:

            print(
                f"{risk['parameter']:30s} "
                f"{risk['severity'].upper():10s} "
                f"confidence={risk['confidence']}"
            )

            print(
                f"  Value      : "
                f"{risk['value']}"
            )

            print(
                f"  Body area  : "
                f"{risk['bodyArea']}"
            )

            print(
                f"  Reliability: "
                f"{risk['reliability']:.3f}"
            )

            print(
                f"  Message    : "
                f"{risk['message']}"
            )

            print()

    else:

        print(
            "No significant risk indicators detected."
        )

    print()

    print("RECOMMENDATIONS")
    print("-" * 60)

    for recommendation in result[
        "recommendations"
    ]:

        print(
            f"- {recommendation}"
        )

    print()

    print(
        f"Output: {output_path}"
    )

    print()

    print("=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)

    return result


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Bowling Biomechanics Phase 3 Risk Engine"
    )

    parser.add_argument(
        "phase2_json",
        help="Phase-2 JSON result"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Phase-3 output JSON path"
    )

    args = parser.parse_args()

    try:

        analyze_file(
            input_path=args.phase2_json,
            output_path=args.output
        )

    except Exception as exc:

        print()
        print("=" * 60)
        print("PHASE 3 ERROR")
        print("=" * 60)

        print(
            str(exc)
        )

        print("=" * 60)

        raise


if __name__ == "__main__":

    main()