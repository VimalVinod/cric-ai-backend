"""
phase4_coach_engine.py
======================

Bowling Biomechanics Analysis System
PHASE 4 - COACH FEEDBACK ENGINE

Purpose:
    Convert Phase-3 biomechanical risk results into
    structured, coach-style feedback.

Input:
    Phase-3 JSON

Output:
    Phase-4 JSON containing:

        1. Overall assessment
        2. Strengths
        3. Priority issues
        4. Coach feedback
        5. Training focus
        6. Recommendations
        7. Confidence-aware limitations
"""

import argparse
import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "PHASE-4"


# ============================================================
# PARAMETER COACHING RULES
# ============================================================

COACH_RULES = {

    "elbowAngle": {

        "bodyArea": "Bowling elbow",

        "high": {
            "title": "Bowling-arm mechanics",
            "feedback": (
                "The delivery shows substantial elbow flexion "
                "near the release phase. Focus on maintaining "
                "consistent bowling-arm mechanics through the "
                "delivery."
            ),
            "focus": [
                "Review bowling-arm path through the delivery.",
                "Maintain a consistent arm position approaching release.",
                "Compare the bowling arm across multiple deliveries.",
            ],
        },

        "moderate": {
            "title": "Bowling-arm consistency",
            "feedback": (
                "Some variation in elbow position was detected "
                "near delivery. Monitor bowling-arm consistency "
                "across multiple deliveries."
            ),
            "focus": [
                "Monitor elbow position during release.",
                "Work on consistent bowling-arm mechanics.",
            ],
        },
    },

    "trunkLateralFlexion": {

        "bodyArea": "Lower back",

        "high": {
            "title": "Trunk stability",
            "feedback": (
                "The delivery shows considerable lateral trunk "
                "movement. Excessive side bending may increase "
                "loading on the lower back."
            ),
            "focus": [
                "Maintain a more stable trunk through release.",
                "Avoid excessive lateral side bending.",
                "Maintain better head and trunk alignment.",
                "Control trunk position through the follow-through.",
            ],
        },

        "moderate": {
            "title": "Trunk control",
            "feedback": (
                "Moderate lateral trunk movement was detected. "
                "Focus on maintaining controlled trunk movement "
                "through the delivery."
            ),
            "focus": [
                "Improve trunk stability.",
                "Monitor lateral movement during release.",
            ],
        },
    },

    "hipShoulderSeparation": {

        "bodyArea": "Hip and trunk",

        "high": {
            "title": "Hip-shoulder sequencing",
            "feedback": (
                "Very limited hip-shoulder separation was detected. "
                "Review the timing and sequencing of hip and "
                "shoulder rotation."
            ),
            "focus": [
                "Work on coordinated hip and shoulder rotation.",
                "Improve rotational sequencing through delivery.",
                "Compare separation across multiple deliveries.",
            ],
        },

        "moderate": {
            "title": "Hip-shoulder sequencing",
            "feedback": (
                "Reduced hip-shoulder separation was detected. "
                "Review rotational sequencing and coordination."
            ),
            "focus": [
                "Improve hip-shoulder sequencing.",
                "Focus on coordinated rotational movement.",
            ],
        },
    },

    "trunkForwardFlexion": {

        "bodyArea": "Lower back",

        "high": {
            "title": "Forward trunk posture",
            "feedback": (
                "High forward trunk flexion was detected during "
                "the delivery phase. Focus on maintaining "
                "controlled trunk posture."
            ),
            "focus": [
                "Maintain controlled forward trunk position.",
                "Avoid excessive trunk collapse during delivery.",
                "Review posture with a qualified cricket coach.",
            ],
        },

        "moderate": {
            "title": "Forward trunk posture",
            "feedback": (
                "Elevated forward trunk flexion was detected. "
                "Monitor trunk posture during delivery and "
                "follow-through."
            ),
            "focus": [
                "Monitor forward trunk posture.",
                "Maintain controlled trunk movement.",
            ],
        },
    },

    "frontKneeAngle": {

        "bodyArea": "Front leg",

        "high": {
            "title": "Front-leg mechanics",
            "feedback": (
                "The front-leg position requires attention. "
                "Review front-leg mechanics around delivery."
            ),
            "focus": [
                "Monitor front-knee position.",
                "Review front-leg stability with a coach.",
            ],
        },

        "moderate": {
            "title": "Front-leg consistency",
            "feedback": (
                "Some variation was detected in front-leg "
                "position. Monitor consistency across deliveries."
            ),
            "focus": [
                "Maintain consistent front-leg mechanics.",
            ],
        },
    },

    "backKneeAngle": {

        "bodyArea": "Back leg",

        "high": {
            "title": "Back-leg mechanics",
            "feedback": (
                "The back-leg position requires attention during "
                "the delivery sequence."
            ),
            "focus": [
                "Monitor back-knee position.",
                "Review lower-body mechanics with a coach.",
            ],
        },

        "moderate": {
            "title": "Back-leg consistency",
            "feedback": (
                "Some variation was detected in the back-leg "
                "position."
            ),
            "focus": [
                "Work on consistent back-leg mechanics.",
            ],
        },
    },

    "frontFootOffset": {

        "bodyArea": "Front foot",

        "high": {
            "title": "Front-foot position",
            "feedback": (
                "The front-foot position should be monitored "
                "for consistency during delivery."
            ),
            "focus": [
                "Maintain consistent front-foot placement.",
            ],
        },

        "moderate": {
            "title": "Front-foot consistency",
            "feedback": (
                "Some variation in front-foot position was detected."
            ),
            "focus": [
                "Monitor front-foot placement.",
            ],
        },
    },
}


# ============================================================
# HELPERS
# ============================================================

def confidence_label(reliability):

    reliability = float(reliability)

    if reliability >= 0.80:
        return "high"

    if reliability >= 0.60:
        return "moderate"

    if reliability >= 0.40:
        return "low"

    return "unreliable"


def severity_rank(severity):

    ranks = {
        "critical": 4,
        "high": 3,
        "moderate": 2,
        "low": 1,
    }

    return ranks.get(
        severity.lower(),
        0
    )


def confidence_rank(confidence):

    ranks = {
        "high": 3,
        "moderate": 2,
        "low": 1,
        "unreliable": 0,
    }

    return ranks.get(
        confidence.lower(),
        0
    )


# ============================================================
# OVERALL ASSESSMENT
# ============================================================

def build_overall_assessment(data):

    overall_risk = data.get(
        "overallRisk",
        "unknown"
    )

    technical_score = data.get(
        "source",
        {}
    ).get(
        "technicalScore"
    )

    risk_counts = data.get(
        "riskCounts",
        {}
    )

    high_count = risk_counts.get(
        "high",
        0
    )

    moderate_count = risk_counts.get(
        "moderate",
        0
    )

    if overall_risk == "critical":

        message = (
            "This delivery requires focused technical review. "
            "Multiple biomechanical indicators require attention, "
            "with priority given to the highest-confidence high-risk findings."
        )

    elif overall_risk == "high":

        message = (
            "This delivery shows several biomechanical areas "
            "that should be reviewed and monitored during training."
        )

    elif overall_risk == "moderate":

        message = (
            "The delivery is generally workable but contains "
            "some biomechanical areas that should be monitored."
        )

    elif overall_risk == "low":

        message = (
            "The delivery shows relatively few biomechanical "
            "concerns based on the available measurements."
        )

    else:

        message = (
            "The delivery could not be confidently classified."
        )

    return {

        "riskLevel": overall_risk,

        "technicalScore": technical_score,

        "highRiskCount": high_count,

        "moderateRiskCount": moderate_count,

        "summary": message,
    }


# ============================================================
# PRIORITY ISSUES
# ============================================================

def build_priority_issues(risk_indicators):

    sorted_indicators = sorted(
        risk_indicators,
        key=lambda item: (
            severity_rank(
                item.get(
                    "severity",
                    "low"
                )
            ),
            confidence_rank(
                item.get(
                    "confidence",
                    "low"
                )
            ),
            float(
                item.get(
                    "reliability",
                    0.0
                )
            ),
        ),
        reverse=True
    )

    priorities = []

    for index, item in enumerate(
        sorted_indicators,
        start=1
    ):

        parameter = item.get(
            "parameter"
        )

        severity = item.get(
            "severity",
            "low"
        )

        confidence = item.get(
            "confidence",
            "low"
        )

        priorities.append({

            "priority": index,

            "parameter": parameter,

            "bodyArea": item.get(
                "bodyArea"
            ),

            "severity": severity,

            "confidence": confidence,

            "value": item.get(
                "value"
            ),

            "reliability": item.get(
                "reliability"
            ),

            "message": item.get(
                "message"
            ),
        })

    return priorities


# ============================================================
# COACH FEEDBACK
# ============================================================

def build_coach_feedback(risk_indicators):

    feedback = []

    for item in risk_indicators:

        parameter = item.get(
            "parameter"
        )

        severity = item.get(
            "severity",
            "low"
        ).lower()

        reliability = float(
            item.get(
                "reliability",
                0.0
            )
        )

        confidence = item.get(
            "confidence"
        )

        if confidence is None:

            confidence = confidence_label(
                reliability
            )

        rules = COACH_RULES.get(
            parameter
        )

        if rules is None:
            continue

        rule = rules.get(
            severity
        )

        if rule is None:

            if severity == "critical":
                rule = rules.get("high")

            else:
                rule = rules.get("moderate")

        if rule is None:
            continue

        feedback.append({

            "parameter": parameter,

            "bodyArea": item.get(
                "bodyArea"
            ),

            "severity": severity,

            "confidence": confidence,

            "value": item.get(
                "value"
            ),

            "title": rule["title"],

            "feedback": rule["feedback"],

            "trainingFocus": rule["focus"],
        })

    return feedback


# ============================================================
# STRENGTHS
# ============================================================

def build_strengths(data):

    strengths = []

    source = data.get(
        "source",
        {}
    )

    technical_score = source.get(
        "technicalScore"
    )

    if technical_score is not None:

        if technical_score >= 80:

            strengths.append(
                "The overall technical score is strong."
            )

        elif technical_score >= 65:

            strengths.append(
                "The overall technical score indicates several technically acceptable areas."
            )

    risk_counts = data.get(
        "riskCounts",
        {}
    )

    if risk_counts.get(
        "critical",
        0
    ) == 0:

        strengths.append(
            "No individual indicator was classified as critical."
        )

    if not strengths:

        strengths.append(
            "Continue monitoring consistency across multiple deliveries."
        )

    return strengths


# ============================================================
# TRAINING FOCUS
# ============================================================

def build_training_focus(coach_feedback):

    focus = []

    seen = set()

    for item in coach_feedback:

        for exercise in item.get(
            "trainingFocus",
            []
        ):

            if exercise not in seen:

                seen.add(
                    exercise
                )

                focus.append(
                    exercise
                )

    return focus


# ============================================================
# CONFIDENCE LIMITATIONS
# ============================================================

def build_limitations(risk_indicators):

    limitations = []

    for item in risk_indicators:

        confidence = item.get(
            "confidence",
            "low"
        )

        if confidence in [
            "low",
            "unreliable"
        ]:

            limitations.append({

                "parameter": item.get(
                    "parameter"
                ),

                "confidence": confidence,

                "reliability": item.get(
                    "reliability"
                ),

                "message": (
                    f"{item.get('parameter')} should be "
                    f"interpreted cautiously because measurement "
                    f"confidence is {confidence}."
                ),
            })

    return limitations


# ============================================================
# RECOMMENDATIONS
# ============================================================

def build_recommendations(
    data,
    coach_feedback
):

    recommendations = []

    existing = data.get(
        "recommendations",
        []
    )

    for item in existing:

        if item not in recommendations:

            recommendations.append(
                item
            )

    for item in coach_feedback:

        title = item.get(
            "title"
        )

        recommendation = (
            f"Prioritize {title.lower()} during technical review."
        )

        if recommendation not in recommendations:

            recommendations.append(
                recommendation
            )

    if not recommendations:

        recommendations.append(
            "Continue monitoring consistency across multiple deliveries."
        )

    return recommendations


# ============================================================
# MAIN PHASE 4 ANALYSIS
# ============================================================

def analyze_phase3_report(
    input_path,
    output_path=None
):

    print("=" * 60)
    print("PHASE 4 - COACH FEEDBACK ENGINE")
    print("=" * 60)

    if not os.path.exists(
        input_path
    ):

        raise FileNotFoundError(
            f"Phase-3 file not found: {input_path}"
        )

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    if data.get(
        "version"
    ) != "PHASE-3":

        raise ValueError(
            "Input file is not a PHASE-3 report."
        )

    risk_indicators = data.get(
        "riskIndicators",
        []
    )

    # --------------------------------------------------------
    # Build sections
    # --------------------------------------------------------

    overall_assessment = (
        build_overall_assessment(
            data
        )
    )

    priority_issues = (
        build_priority_issues(
            risk_indicators
        )
    )

    coach_feedback = (
        build_coach_feedback(
            risk_indicators
        )
    )

    strengths = (
        build_strengths(
            data
        )
    )

    training_focus = (
        build_training_focus(
            coach_feedback
        )
    )

    limitations = (
        build_limitations(
            risk_indicators
        )
    )

    recommendations = (
        build_recommendations(
            data,
            coach_feedback
        )
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {

        "status": "ok",

        "version": VERSION,

        "source": {

            "phase3Version":
                data.get(
                    "version"
                ),

            "video":
                data.get(
                    "source",
                    {}
                ).get(
                    "video"
                ),

            "view":
                data.get(
                    "source",
                    {}
                ).get(
                    "view"
                ),

            "viewQuality":
                data.get(
                    "source",
                    {}
                ).get(
                    "viewQuality"
                ),

            "technicalScore":
                data.get(
                    "source",
                    {}
                ).get(
                    "technicalScore"
                ),
        },

        "overallAssessment":
            overall_assessment,

        "strengths":
            strengths,

        "priorityIssues":
            priority_issues,

        "coachFeedback":
            coach_feedback,

        "trainingFocus":
            training_focus,

        "recommendations":
            recommendations,

        "confidenceLimitations":
            limitations,
    }

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if output_path is None:

        base, _ = os.path.splitext(
            input_path
        )

        output_path = (
            base +
            "_phase4.json"
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
    print("COACH ASSESSMENT")
    print("-" * 60)

    print(
        f"Risk level       : "
        f"{overall_assessment['riskLevel']}"
    )

    print(
        f"Technical score  : "
        f"{overall_assessment['technicalScore']}"
    )

    print()
    print(
        overall_assessment[
            "summary"
        ]
    )

    print()
    print("PRIORITY ISSUES")
    print("-" * 60)

    for item in priority_issues:

        print(
            f"{item['priority']}. "
            f"{item['parameter']} "
            f"[{item['severity'].upper()}]"
        )

        print(
            f"   Value      : "
            f"{item['value']}"
        )

        print(
            f"   Confidence : "
            f"{item['confidence']}"
        )

        print(
            f"   Area       : "
            f"{item['bodyArea']}"
        )

    print()
    print("COACH FEEDBACK")
    print("-" * 60)

    for item in coach_feedback:

        print(
            f"\n{item['title']}"
        )

        print(
            f"  {item['feedback']}"
        )

        print(
            "  Training focus:"
        )

        for focus in item[
            "trainingFocus"
        ]:

            print(
                f"    - {focus}"
            )

    print()
    print("STRENGTHS")
    print("-" * 60)

    for strength in strengths:

        print(
            f"- {strength}"
        )

    print()
    print("TRAINING FOCUS")
    print("-" * 60)

    for item in training_focus:

        print(
            f"- {item}"
        )

    print()

    if limitations:

        print("CONFIDENCE LIMITATIONS")
        print("-" * 60)

        for item in limitations:

            print(
                f"- {item['message']}"
            )

    else:

        print(
            "Confidence limitations: none"
        )

    print()
    print("RECOMMENDATIONS")
    print("-" * 60)

    for recommendation in recommendations:

        print(
            f"- {recommendation}"
        )

    print()
    print(
        f"Output: {output_path}"
    )

    print()
    print("=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)

    return result


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Bowling Biomechanics Phase 4 "
        "Coach Feedback Engine"
    )

    parser.add_argument(
        "input",
        help="Input PHASE-3 JSON file"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output PHASE-4 JSON file"
    )

    args = parser.parse_args()

    try:

        analyze_phase3_report(
            input_path=args.input,
            output_path=args.output
        )

    except Exception as exc:

        print()
        print("=" * 60)
        print("PHASE 4 ERROR")
        print("=" * 60)

        print(
            str(exc)
        )

        print("=" * 60)

        raise


if __name__ == "__main__":

    main()