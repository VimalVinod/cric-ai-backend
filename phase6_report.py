"""
phase6_report.py

PHASE 1 JSON -> Professional HTML Bowling Biomechanics Report -> PDF

PDF ENGINE:
    Playwright + Chromium

Usage:
    python phase6_report.py "test_data\\side_bowler1_phase1_result.json"

Optional:
    python phase6_report.py "test_data\\side_bowler1_phase1_result.json" --out-name side_report
    python phase6_report.py "test_data\\side_bowler1_phase1_result.json" --out-dir reports

This is a presentation/reporting layer.
It does not modify or reinterpret Phase-1 measurements.
"""

import argparse
import html
import json
import os
import sys
from datetime import datetime


# ============================================================
# PARAMETER DEFINITIONS
# ============================================================

PARAM_LABELS = {
    "frontKneeAngle": "Front Knee Angle",
    "backKneeAngle": "Back Knee Angle",
    "elbowAngle": "Bowling Elbow Angle",
    "trunkLateralFlexion": "Trunk Lateral Flexion",
    "trunkForwardFlexion": "Trunk Forward Flexion",
    "hipShoulderSeparation": "Hip-Shoulder Separation",
    "frontFootOffset": "Front Foot Offset",
    "headOffset": "Head Offset",
    "shoulderLineAngle": "Shoulder Line Angle",
    "hipLineAngle": "Hip Line Angle",
}

PARAM_UNITS = {
    "frontKneeAngle": "°",
    "backKneeAngle": "°",
    "elbowAngle": "°",
    "trunkLateralFlexion": "°",
    "trunkForwardFlexion": "°",
    "hipShoulderSeparation": "°",
    "shoulderLineAngle": "°",
    "hipLineAngle": "°",
    "frontFootOffset": "",
    "headOffset": "",
}

PARAM_ORDER = [
    "elbowAngle",
    "frontKneeAngle",
    "backKneeAngle",
    "hipShoulderSeparation",
    "trunkLateralFlexion",
    "trunkForwardFlexion",
    "frontFootOffset",
    "headOffset",
    "shoulderLineAngle",
    "hipLineAngle",
]


# ============================================================
# FORMATTING HELPERS
# ============================================================

def fmt(value, decimals=2, fallback="—"):
    """Safely format numeric values."""

    if value is None:
        return fallback

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_percent(value, decimals=0, fallback="—"):
    """Convert 0-1 values into percentages."""

    if value is None:
        return fallback

    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return str(value)


def esc(value):
    """HTML-safe escaping."""

    return html.escape(str(value)) if value is not None else ""


# ============================================================
# SCORE CLASSIFICATION
# ============================================================

def score_band(score):

    if score is None:
        return "Measured", "muted"

    try:
        score = float(score)
    except (TypeError, ValueError):
        return "Measured", "muted"

    if score >= 85:
        return "Optimal", "good"

    if score >= 60:
        return "Monitor", "watch"

    return "Attention", "flag"


def risk_band(severity):

    severity = str(severity or "").lower()

    if severity in ("high", "critical", "severe"):
        return "Attention", "flag"

    if severity == "monitor":
        return "Monitor", "watch"

    return "Monitor", "watch"


def overall_band(score, risk_count):

    if score is None:
        base = "Technical score unavailable."

    else:
        try:
            score = float(score)

            if score >= 85:
                base = "Strong technical profile overall."

            elif score >= 70:
                base = (
                    "Sound technical base with areas to monitor."
                )

            elif score >= 50:
                base = (
                    "Several parameters require attention "
                    "against the current Phase-1 reference ranges."
                )

            else:
                base = (
                    "Multiple parameters require attention "
                    "and coach review is recommended."
                )

        except (TypeError, ValueError):
            base = "Technical score unavailable."

    if risk_count == 0:
        flag = (
            "No movement-pattern risk indicators were raised."
        )

    elif risk_count == 1:
        flag = (
            "One movement-pattern indicator was flagged "
            "for coach review."
        )

    else:
        flag = (
            f"{risk_count} movement-pattern indicators "
            "were flagged for coach review."
        )

    return f"{base} {flag}"


# ============================================================
# PARAMETER CARDS
# ============================================================

def build_parameter_cards(
    measurements,
    parameter_scores,
    reliability
):

    cards = []

    for key in PARAM_ORDER:

        if key not in measurements:
            continue

        value = measurements.get(key)

        label = PARAM_LABELS.get(key, key)

        unit = PARAM_UNITS.get(key, "")

        score_data = parameter_scores.get(key)

        rel = reliability.get(key)

        if score_data is not None:

            score_value = score_data.get("score")

            band_label, band_class = score_band(
                score_value
            )

            if score_value is None:
                score_display = "n/a"
            else:
                score_display = f"{float(score_value):.1f}/100"

        else:

            band_label = "Measured"
            band_class = "muted"
            score_display = "n/a"

        value_display = fmt(value)

        unit_html = (
            f'<span class="parameter-unit">'
            f'{esc(unit)}'
            f'</span>'
            if unit
            else ""
        )

        reliability_display = (
            "n/a"
            if rel is None
            else fmt_percent(rel)
        )

        cards.append(
            f"""
            <div class="parameter-card {band_class}">

                <div class="parameter-header">

                    <div class="parameter-name">
                        {esc(label)}
                    </div>

                    <div class="parameter-badge {band_class}">
                        {esc(band_label)}
                    </div>

                </div>

                <div class="parameter-value">

                    {esc(value_display)}

                    {unit_html}

                </div>

                <div class="parameter-meta">

                    <span>
                        SCORE
                        <strong>
                            {esc(score_display)}
                        </strong>
                    </span>

                    <span>
                        RELIABILITY
                        <strong>
                            {esc(reliability_display)}
                        </strong>
                    </span>

                </div>

            </div>
            """
        )

    if not cards:

        return """
        <div class="empty-state">
            No biomechanical measurements were supplied by Phase-1.
        </div>
        """

    return "\n".join(cards)


# ============================================================
# RISK INDICATORS
# ============================================================

def build_risk_items(risks):

    if not risks:

        return """
        <div class="no-risk">

            <div class="no-risk-icon">
                ✓
            </div>

            <div>

                <strong>
                    No movement-pattern flags
                </strong>

                <p>
                    No risk indicators were raised by the Phase-1
                    analysis for this delivery.
                </p>

            </div>

        </div>
        """

    items = []

    for risk in risks:

        severity = risk.get(
            "severity",
            "monitor"
        )

        severity_label, severity_class = risk_band(
            severity
        )

        body_area = risk.get(
            "bodyArea",
            "Unspecified area"
        )

        parameter = risk.get(
            "parameter",
            ""
        )

        value = risk.get("value")

        message = risk.get(
            "message",
            ""
        )

        parameter_text = (
            f"<span>{esc(parameter)}</span>"
            if parameter
            else ""
        )

        items.append(
            f"""
            <div class="risk-item risk-{severity_class}">

                <div class="risk-symbol">
                    △
                </div>

                <div class="risk-content">

                    <div class="risk-title">

                        {esc(body_area)}

                        {parameter_text}

                    </div>

                    <div class="risk-severity">
                        {esc(severity_label)}
                    </div>

                    <div class="risk-value">

                        Observed value:
                        {esc(fmt(value))}

                    </div>

                    <div class="risk-message">

                        {esc(message)}

                    </div>

                </div>

            </div>
            """
        )

    return "\n".join(items)


# ============================================================
# RECOMMENDATIONS
# ============================================================

def build_recommendations(recommendations):

    if not recommendations:

        recommendations = [
            "No specific action items were generated by Phase-1."
        ]

    items = []

    for recommendation in recommendations:

        items.append(
            f"""
            <li>

                <span class="recommendation-arrow">
                    →
                </span>

                <span>
                    {esc(recommendation)}
                </span>

            </li>
            """
        )

    return "\n".join(items)


# ============================================================
# TECHNICAL METADATA
# ============================================================

def build_metadata(data):

    video = data.get("video", {})
    release = data.get("releaseFrame", {})
    window = data.get("analysisWindow", {})
    body_scale = data.get("bodyScale", {})

    rows = [

        (
            "Analysis Version",
            data.get("version", "—")
        ),

        (
            "Bowling Arm",
            str(
                data.get(
                    "bowlingArm",
                    "unknown"
                )
            ).upper()
        ),

        (
            "Video FPS",
            fmt(video.get("fps"), 2)
        ),

        (
            "Resolution",
            f"{video.get('width', '—')} × "
            f"{video.get('height', '—')}"
        ),

        (
            "Release Method",
            release.get(
                "detectionMethod",
                "—"
            )
        ),

        (
            "Release Frame",
            release.get(
                "index",
                "—"
            )
        ),

        (
            "Analysis Window",
            f"Frames "
            f"{window.get('startFrame', '—')} – "
            f"{window.get('endFrame', '—')}"
        ),

        (
            "Frames Used",
            window.get(
                "framesUsed",
                "—"
            )
        ),

        (
            "Median Torso Scale",
            f"{fmt(body_scale.get('medianTorsoPixels'))} px"
        ),

    ]

    html_items = []

    for label, value in rows:

        html_items.append(
            f"""
            <div class="metadata-item">

                <div class="metadata-label">
                    {esc(label)}
                </div>

                <div class="metadata-value">
                    {esc(value)}
                </div>

            </div>
            """
        )

    return "\n".join(html_items)


# ============================================================
# HTML BUILDER
# ============================================================

def build_html(data):

    video = data.get("video", {})

    release = data.get(
        "releaseFrame",
        {}
    )

    measurements = data.get(
        "measurements",
        {}
    )

    parameter_scores = data.get(
        "parameterScores",
        {}
    )

    reliability = data.get(
        "reliability",
        {}
    )

    risks = data.get(
        "riskIndicators",
        []
    )

    recommendations = data.get(
        "recommendations",
        []
    )

    technical_score = data.get(
        "technicalScore"
    )

    detection_rate = data.get(
        "detectionRate"
    )

    bowling_arm = data.get(
        "bowlingArm",
        "unknown"
    )

    window = data.get(
        "analysisWindow",
        {}
    )

    video_path = str(
        video.get(
            "path",
            "unknown"
        )
    ).replace("\\", "/")

    video_name = os.path.basename(
        video_path
    )

    generated_at = datetime.now().strftime(
        "%d %b %Y, %H:%M"
    )

    # --------------------------------------------------------
    # Release percentage
    # --------------------------------------------------------

    release_pct = release.get(
        "percentThroughClip",
        0
    )

    try:
        release_pct = max(
            0,
            min(float(release_pct), 100)
        )
    except (TypeError, ValueError):
        release_pct = 0

    # --------------------------------------------------------
    # Technical score
    # --------------------------------------------------------

    if technical_score is None:

        score_value = None
        technical_score_display = "--"

    else:

        try:
            score_value = max(
                0,
                min(float(technical_score), 100)
            )

            technical_score_display = (
                f"{score_value:.1f}"
            )

        except (TypeError, ValueError):

            score_value = None
            technical_score_display = "--"

    # --------------------------------------------------------
    # Gauge
    # --------------------------------------------------------

    radius = 80

    circumference = (
        2 *
        3.141592653589793 *
        radius
    )

    score_fraction = (
        0
        if score_value is None
        else score_value / 100
    )

    score_dash = (
        circumference *
        score_fraction
    )

    # --------------------------------------------------------
    # Overall state
    # --------------------------------------------------------

    if score_value is None:

        overall_class = "muted"
        overall_label = "Score Unavailable"

    elif risks:

        overall_class = "watch"
        overall_label = "Good — Review Flags"

    elif score_value >= 85:

        overall_class = "good"
        overall_label = "Strong Technical Profile"

    elif score_value >= 70:

        overall_class = "watch"
        overall_label = "Monitor"

    else:

        overall_class = "flag"
        overall_label = "Attention"

    overall_text = overall_band(
        score_value,
        len(risks)
    )

    # --------------------------------------------------------
    # Dynamic sections
    # --------------------------------------------------------

    parameter_cards = build_parameter_cards(
        measurements,
        parameter_scores,
        reliability
    )

    risk_items = build_risk_items(
        risks
    )

    recommendation_items = build_recommendations(
        recommendations
    )

    metadata_items = build_metadata(
        data
    )

    detection_display = fmt_percent(
        detection_rate
    )

    release_frame = release.get(
        "index",
        "—"
    )

    release_time = release.get(
        "timestampSeconds"
    )

    if release_time is None:

        release_time_display = "—"

    else:

        try:
            release_time_display = (
                f"{float(release_time):.3f}s"
            )
        except (TypeError, ValueError):
            release_time_display = str(release_time)

    frame_count = video.get(
        "frameCount",
        "—"
    )

    html_document = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
Bowling Biomechanics Report —
{esc(video_name)}
</title>

<style>

@page {{
    size: A4;
    margin: 12mm;
}}

:root {{

    --navy: #07111F;
    --navy2: #0D192B;
    --panel: #111F34;
    --panel2: #162640;
    --line: #263A58;

    --cyan: #19D6C5;
    --cyan-dark: #0D8E84;

    --amber: #F2AA3B;
    --red: #F05E4C;

    --white: #EAF0F8;
    --muted: #91A1BB;
    --faint: #5D6E8B;
}}

* {{
    box-sizing: border-box;
}}

html,
body {{

    margin: 0;
    padding: 0;

    background: var(--navy);
    color: var(--white);

    font-family:
        Arial,
        Helvetica,
        sans-serif;
}}

body {{

    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}}

.page {{

    max-width: 1000px;

    margin: 0 auto;

    padding:
        34px
        34px
        50px;
}}


/* =====================================================
   HEADER
   ===================================================== */

.eyebrow {{

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 0.18em;

    text-transform: uppercase;

    color: var(--cyan);

    margin-bottom: 7px;
}}

.masthead {{

    display: flex;

    justify-content: space-between;

    align-items: flex-end;

    padding-bottom: 20px;

    border-bottom:
        1px solid var(--line);

    margin-bottom: 24px;
}}

h1 {{

    margin: 0;

    font-size: 34px;

    line-height: 1;

    font-weight: 800;

    letter-spacing: -0.02em;

    text-transform: uppercase;
}}

.header-meta {{

    text-align: right;

    color: var(--muted);

    font-family:
        "Courier New",
        monospace;

    font-size: 10px;

    line-height: 1.7;
}}

.header-meta strong {{
    color: var(--white);
}}


/* =====================================================
   HERO
   ===================================================== */

.hero {{

    display: grid;

    grid-template-columns:
        220px 1fr;

    gap: 30px;

    align-items: center;

    padding: 26px;

    background:
        linear-gradient(
            135deg,
            var(--panel),
            var(--panel2)
        );

    border:
        1px solid var(--line);

    border-radius: 6px;

    margin-bottom: 18px;
}}

.gauge {{

    position: relative;

    width: 190px;
    height: 190px;
}}

.gauge svg {{

    width: 190px;
    height: 190px;
}}

.gauge-center {{

    position: absolute;

    inset: 0;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;
}}

.gauge-score {{

    font-size: 52px;

    font-weight: 800;

    line-height: 1;

    color: var(--cyan);
}}

.gauge-label {{

    margin-top: 5px;

    font-size: 9px;

    letter-spacing: 0.13em;

    text-transform: uppercase;

    color: var(--muted);
}}

.assessment-band {{

    display: inline-block;

    padding: 6px 10px;

    border-radius: 3px;

    font-family:
        "Courier New",
        monospace;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 0.08em;

    text-transform: uppercase;

    margin-bottom: 12px;
}}

.assessment-band.good {{

    color: var(--cyan);

    border:
        1px solid var(--cyan-dark);

    background:
        rgba(
            25,
            214,
            197,
            0.10
        );
}}

.assessment-band.watch {{

    color: var(--amber);

    border:
        1px solid var(--amber);

    background:
        rgba(
            242,
            170,
            59,
            0.10
        );
}}

.assessment-band.flag {{

    color: var(--red);

    border:
        1px solid var(--red);

    background:
        rgba(
            240,
            94,
            76,
            0.10
        );
}}

.assessment-band.muted {{

    color: var(--muted);

    border:
        1px solid var(--line);

    background:
        rgba(
            145,
            161,
            187,
            0.08
        );
}}

.hero-description {{

    margin: 0;

    max-width: 620px;

    font-size: 15px;

    line-height: 1.65;

    color: var(--muted);
}}


/* =====================================================
   STATS
   ===================================================== */

.stat-strip {{

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 1px;

    background: var(--line);

    border:
        1px solid var(--line);

    margin-bottom: 30px;
}}

.stat {{

    background: var(--panel);

    padding: 15px 16px;
}}

.stat-label {{

    color: var(--faint);

    font-size: 9px;

    letter-spacing: 0.12em;

    text-transform: uppercase;

    margin-bottom: 6px;
}}

.stat-value {{

    color: var(--white);

    font-size: 21px;

    font-weight: 800;
}}


/* =====================================================
   SECTIONS
   ===================================================== */

.section {{

    margin-bottom: 30px;
}}

.section-title {{

    display: flex;

    align-items: center;

    gap: 9px;

    font-size: 19px;

    font-weight: 800;

    text-transform: uppercase;

    letter-spacing: 0.01em;

    padding-bottom: 9px;

    border-bottom:
        1px solid var(--line);

    margin-bottom: 15px;
}}

.section-number {{

    color: var(--cyan);

    font-family:
        "Courier New",
        monospace;

    font-size: 12px;
}}


/* =====================================================
   TIMELINE
   ===================================================== */

.timeline {{

    padding:
        18px
        10px
        5px;
}}

.timeline-track {{

    position: relative;

    height: 8px;

    background: var(--panel2);

    border-radius: 8px;
}}

.timeline-fill {{

    position: absolute;

    left: 0;
    top: 0;
    bottom: 0;

    width: {release_pct}%;

    background:
        linear-gradient(
            90deg,
            var(--cyan-dark),
            var(--cyan)
        );

    border-radius: 8px;
}}

.timeline-marker {{

    position: absolute;

    left: {release_pct}%;

    top: -28px;

    transform:
        translateX(-50%);

    color: var(--cyan);

    font-family:
        "Courier New",
        monospace;

    font-size: 9px;

    white-space: nowrap;
}}

.timeline-marker::after {{

    content: "";

    position: absolute;

    left: 50%;

    top: 20px;

    width: 2px;

    height: 18px;

    background: var(--cyan);
}}

.timeline-labels {{

    display: flex;

    justify-content: space-between;

    margin-top: 9px;

    color: var(--faint);

    font-family:
        "Courier New",
        monospace;

    font-size: 8px;

    letter-spacing: 0.08em;
}}


/* =====================================================
   PARAMETERS
   ===================================================== */

.parameter-grid {{

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 11px;
}}

.parameter-card {{

    background: var(--panel);

    border:
        1px solid var(--line);

    border-left:
        3px solid var(--faint);

    padding: 14px 15px;

    border-radius: 4px;
}}

.parameter-card.good {{
    border-left-color: var(--cyan);
}}

.parameter-card.watch {{
    border-left-color: var(--amber);
}}

.parameter-card.flag {{
    border-left-color: var(--red);
}}

.parameter-card.muted {{
    border-left-color: var(--faint);
}}

.parameter-header {{

    display: flex;

    justify-content:
        space-between;

    align-items: center;

    gap: 10px;

    margin-bottom: 6px;
}}

.parameter-name {{

    font-size: 12px;

    color: var(--muted);
}}

.parameter-badge {{

    font-family:
        "Courier New",
        monospace;

    font-size: 8px;

    font-weight: 700;

    letter-spacing: 0.07em;

    text-transform: uppercase;

    padding: 3px 6px;

    border-radius: 2px;
}}

.parameter-badge.good {{

    color: var(--cyan);

    background:
        rgba(
            25,
            214,
            197,
            0.10
        );
}}

.parameter-badge.watch {{

    color: var(--amber);

    background:
        rgba(
            242,
            170,
            59,
            0.10
        );
}}

.parameter-badge.flag {{

    color: var(--red);

    background:
        rgba(
            240,
            94,
            76,
            0.10
        );
}}

.parameter-badge.muted {{

    color: var(--faint);

    background:
        rgba(
            145,
            161,
            187,
            0.08
        );
}}

.parameter-value {{

    font-size: 27px;

    line-height: 1;

    font-weight: 800;

    margin-bottom: 8px;
}}

.parameter-unit {{

    color: var(--muted);

    font-size: 14px;

    margin-left: 2px;
}}

.parameter-meta {{

    display: flex;

    gap: 16px;

    color: var(--faint);

    font-family:
        "Courier New",
        monospace;

    font-size: 8.5px;
}}

.parameter-meta strong {{
    color: var(--muted);
}}


/* =====================================================
   RISK
   ===================================================== */

.risk-item {{

    display: flex;

    gap: 12px;

    padding: 14px 15px;

    border:
        1px solid
        rgba(
            240,
            94,
            76,
            0.35
        );

    border-radius: 4px;

    background:
        rgba(
            240,
            94,
            76,
            0.06
        );

    margin-bottom: 10px;
}}

.risk-item.risk-watch {{

    border-color:
        rgba(
            242,
            170,
            59,
            0.40
        );

    background:
        rgba(
            242,
            170,
            59,
            0.06
        );
}}

.risk-item.risk-flag {{

    border-color:
        rgba(
            240,
            94,
            76,
            0.40
        );
}}

.risk-symbol {{

    font-size: 20px;

    color: var(--red);

    line-height: 1;
}}

.risk-watch .risk-symbol {{
    color: var(--amber);
}}

.risk-content {{
    flex: 1;
}}

.risk-title {{

    font-size: 13px;

    font-weight: 700;

    margin-bottom: 4px;
}}

.risk-title span {{

    color: var(--muted);

    font-family:
        "Courier New",
        monospace;

    font-size: 9px;

    margin-left: 7px;
}}

.risk-severity {{

    display: inline-block;

    color: var(--amber);

    font-family:
        "Courier New",
        monospace;

    font-size: 8px;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.08em;

    margin-bottom: 5px;
}}

.risk-flag .risk-severity {{
    color: var(--red);
}}

.risk-value {{

    color: var(--amber);

    font-family:
        "Courier New",
        monospace;

    font-size: 9px;

    margin-bottom: 5px;
}}

.risk-message {{

    color: var(--muted);

    font-size: 12px;

    line-height: 1.5;
}}

.no-risk {{

    display: flex;

    gap: 12px;

    align-items: flex-start;

    padding: 15px;

    border:
        1px solid
        rgba(
            25,
            214,
            197,
            0.30
        );

    background:
        rgba(
            25,
            214,
            197,
            0.05
        );

    border-radius: 4px;
}}

.no-risk-icon {{

    color: var(--cyan);

    font-size: 20px;
}}

.no-risk strong {{

    color: var(--cyan);

    font-size: 13px;
}}

.no-risk p {{

    color: var(--muted);

    font-size: 12px;

    margin: 4px 0 0;
}}


/* =====================================================
   RECOMMENDATIONS
   ===================================================== */

.recommendations {{

    list-style: none;

    margin: 0;

    padding: 0;
}}

.recommendations li {{

    display: flex;

    gap: 10px;

    padding: 10px 0;

    border-bottom:
        1px solid var(--line);

    color: var(--white);

    font-size: 13px;

    line-height: 1.5;
}}

.recommendation-arrow {{

    color: var(--cyan);

    font-weight: 800;
}}


/* =====================================================
   TECHNICAL DATA
   ===================================================== */

.metadata-grid {{

    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 1px;

    background: var(--line);

    border:
        1px solid var(--line);
}}

.metadata-item {{

    background: var(--panel);

    padding: 12px 14px;
}}

.metadata-label {{

    color: var(--faint);

    font-family:
        "Courier New",
        monospace;

    font-size: 8px;

    letter-spacing: 0.08em;

    text-transform: uppercase;

    margin-bottom: 5px;
}}

.metadata-value {{

    color: var(--white);

    font-size: 12px;

    font-weight: 600;

    word-break: break-word;
}}


/* =====================================================
   OVERALL ASSESSMENT
   ===================================================== */

.overall-box {{

    background:
        linear-gradient(
            135deg,
            var(--panel),
            var(--panel2)
        );

    border:
        1px solid var(--line);

    border-left:
        3px solid var(--cyan);

    padding: 19px;

    border-radius: 4px;
}}

.overall-box p {{

    margin: 0;

    color: var(--white);

    font-size: 14px;

    line-height: 1.65;
}}

.overall-box .secondary {{

    margin-top: 10px;

    color: var(--muted);

    font-size: 11.5px;
}}


/* =====================================================
   EMPTY STATE
   ===================================================== */

.empty-state {{

    padding: 20px;

    border:
        1px solid var(--line);

    background: var(--panel);

    color: var(--muted);

    text-align: center;
}}


/* =====================================================
   DISCLAIMER
   ===================================================== */

.disclaimer {{

    border-top:
        1px solid var(--line);

    padding-top: 17px;

    margin-top: 32px;

    color: var(--faint);

    font-family:
        "Courier New",
        monospace;

    font-size: 8.5px;

    line-height: 1.7;
}}

.disclaimer strong {{
    color: var(--muted);
}}


/* =====================================================
   PRINT
   ===================================================== */

@media print {{

    body {{
        background: var(--navy);
    }}

    .page {{
        padding:
            18px
            20px
            30px;
    }}

    .section {{
        break-inside: avoid;
    }}

    .parameter-card {{
        break-inside: avoid;
    }}

    .risk-item {{
        break-inside: avoid;
    }}
}}

</style>

</head>


<body>

<div class="page">


<!-- =====================================================
     HEADER
     ===================================================== -->

<header class="masthead">

    <div>

        <div class="eyebrow">
            AI Bowling Analysis · Phase 6 Report
        </div>

        <h1>
            Bowling Biomechanics Report
        </h1>

    </div>

    <div class="header-meta">

        GENERATED
        <strong>
            {esc(generated_at)}
        </strong>

        <br>

        SOURCE
        <strong>
            {esc(video_name)}
        </strong>

        <br>

        ARM
        <strong>
            {esc(str(bowling_arm).upper())}
        </strong>

    </div>

</header>


<!-- =====================================================
     HERO
     ===================================================== -->

<section class="hero">

    <div class="gauge">

        <svg
            viewBox="0 0 190 190"
            width="190"
            height="190"
        >

            <circle
                cx="95"
                cy="95"
                r="80"
                fill="none"
                stroke="#1A2941"
                stroke-width="13"
            />

            <circle
                cx="95"
                cy="95"
                r="80"
                fill="none"
                stroke="#19D6C5"
                stroke-width="13"
                stroke-linecap="round"
                stroke-dasharray="{score_dash} {circumference}"
                transform="rotate(-90 95 95)"
            />

        </svg>

        <div class="gauge-center">

            <div class="gauge-score">
                {technical_score_display}
            </div>

            <div class="gauge-label">
                Technical Score
            </div>

        </div>

    </div>


    <div>

        <div class="assessment-band {overall_class}">
            {esc(overall_label)}
        </div>

        <p class="hero-description">
            {esc(overall_text)}
        </p>

    </div>

</section>


<!-- =====================================================
     STATS
     ===================================================== -->

<div class="stat-strip">

    <div class="stat">

        <div class="stat-label">
            Release Frame
        </div>

        <div class="stat-value">
            #{esc(release_frame)}
        </div>

    </div>


    <div class="stat">

        <div class="stat-label">
            Release Time
        </div>

        <div class="stat-value">
            {esc(release_time_display)}
        </div>

    </div>


    <div class="stat">

        <div class="stat-label">
            Detection Rate
        </div>

        <div class="stat-value">
            {esc(detection_display)}
        </div>

    </div>


    <div class="stat">

        <div class="stat-label">
            Clip Frames
        </div>

        <div class="stat-value">
            {esc(frame_count)}
        </div>

    </div>

</div>


<!-- =====================================================
     SECTION 01
     ===================================================== -->

<section class="section">

    <div class="section-title">

        <span class="section-number">
            01
        </span>

        Release Analysis

    </div>


    <div class="timeline">

        <div class="timeline-track">

            <div class="timeline-fill"></div>

            <div class="timeline-marker">

                RELEASE ·
                {fmt(release_pct, 1)}%

            </div>

        </div>


        <div class="timeline-labels">

            <span>
                CLIP START
            </span>

            <span>
                RELEASE
            </span>

            <span>
                CLIP END
            </span>

        </div>

    </div>

</section>


<!-- =====================================================
     SECTION 02
     ===================================================== -->

<section class="section">

    <div class="section-title">

        <span class="section-number">
            02
        </span>

        Biomechanical Measurements

    </div>


    <div class="parameter-grid">

        {parameter_cards}

    </div>

</section>


<!-- =====================================================
     SECTION 03
     ===================================================== -->

<section class="section">

    <div class="section-title">

        <span class="section-number">
            03
        </span>

        Risk Indicators

    </div>


    {risk_items}

</section>


<!-- =====================================================
     SECTION 04
     ===================================================== -->

<section class="section">

    <div class="section-title">

        <span class="section-number">
            04
        </span>

        Coaching Recommendations

    </div>


    <ul class="recommendations">

        {recommendation_items}

    </ul>

</section>


<!-- =====================================================
     SECTION 05
     ===================================================== -->

<section class="section">

    <div class="section-title">

        <span class="section-number">
            05
        </span>

        Technical Analysis Data

    </div>


    <div class="metadata-grid">

        {metadata_items}

    </div>

</section>


<!-- =====================================================
     SECTION 06
     ===================================================== -->

<section class="section">

    <div class="section-title">

        <span class="section-number">
            06
        </span>

        Overall Assessment

    </div>


    <div class="overall-box">

        <p>
            {esc(overall_text)}
        </p>


        <p class="secondary">

            This report is based on a
            <strong>
                single bowling delivery
            </strong>
            analyzed from the supplied video.

            Release-phase measurements were calculated
            using frames
            <strong>
                {esc(window.get("startFrame", "—"))}
                –
                {esc(window.get("endFrame", "—"))}
            </strong>

            with
            <strong>
                {esc(window.get("framesUsed", "—"))}
            </strong>
            frames used in the measurement window.

            Pose detection succeeded on
            <strong>
                {esc(detection_display)}
            </strong>
            of analyzed frames.

        </p>

    </div>

</section>


<!-- =====================================================
     DISCLAIMER
     ===================================================== -->

<div class="disclaimer">

    <strong>
        TECHNICAL DISCLAIMER
    </strong>

    · This Phase 6 report is a presentation layer
    for the Phase 1 bowling-biomechanics engine.

    It does not independently validate or reinterpret
    the underlying biomechanical measurements.

    The reference ranges currently used by the prototype
    are coaching-oriented development ranges and have
    not been established as clinical injury thresholds.

    A flagged parameter represents a movement-pattern
    observation and does not establish injury, injury risk,
    or a medical condition.

    Single-delivery analysis represents one observed
    bowling action and may not represent the athlete's
    normal technique across repeated deliveries.

    Results should therefore be reviewed by a qualified
    cricket coach, sports scientist, or appropriate
    professional before making significant technique or
    training-load decisions.

</div>


</div>

</body>

</html>
"""

    return html_document


# ============================================================
# PLAYWRIGHT PDF
# ============================================================

def generate_pdf(html_path, pdf_path):

    try:

        from playwright.sync_api import sync_playwright

    except ImportError:

        raise RuntimeError(
            "Playwright is not installed in the active "
            "virtual environment."
        )

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page()

        page.goto(
            "file:///"
            + os.path.abspath(html_path)
            .replace("\\", "/"),
            wait_until="networkidle"
        )

        page.pdf(
            path=os.path.abspath(pdf_path),
            format="A4",
            print_background=True,
            prefer_css_page_size=True,
            margin={
                "top": "0",
                "right": "0",
                "bottom": "0",
                "left": "0",
            }
        )

        browser.close()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a professional Phase-6 "
            "bowling biomechanics report "
            "from Phase-1 JSON."
        )
    )

    parser.add_argument(
        "json_path",
        help="Path to Phase-1 JSON result."
    )

    parser.add_argument(
        "--out-name",
        default=None,
        help="Output filename stem without extension."
    )

    parser.add_argument(
        "--out-dir",
        default="output",
        help="Output directory."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not os.path.isfile(args.json_path):

        print(
            f"ERROR: JSON file not found: "
            f"{args.json_path}",
            file=sys.stderr
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    try:

        with open(
            args.json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

    except json.JSONDecodeError as e:

        print(
            f"ERROR: Invalid JSON file: {e}",
            file=sys.stderr
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Basic Phase-1 validation
    # --------------------------------------------------------

    if not isinstance(data, dict):

        print(
            "ERROR: Phase-1 JSON root must be an object.",
            file=sys.stderr
        )

        sys.exit(1)

    if data.get("version") != "PHASE-1":

        print(
            "WARNING: JSON does not identify itself "
            "as version PHASE-1."
        )

    # --------------------------------------------------------
    # Output name
    # --------------------------------------------------------

    if args.out_name:

        stem = args.out_name

    else:

        base = os.path.splitext(
            os.path.basename(
                args.json_path
            )
        )[0]

        stem = base.replace(
            "_phase1_result",
            ""
        )

        stem = stem.rstrip("_")

        stem += "_report"

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.out_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate HTML
    # --------------------------------------------------------

    try:

        html_content = build_html(data)

    except Exception as e:

        print(
            f"ERROR: Failed to build HTML report: {e}",
            file=sys.stderr
        )

        sys.exit(1)

    html_path = os.path.abspath(
        os.path.join(
            args.out_dir,
            f"{stem}.html"
        )
    )

    with open(
        html_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(html_content)

    print(
        f"HTML report written: {html_path}"
    )

    # --------------------------------------------------------
    # Generate PDF with Playwright
    # --------------------------------------------------------

    pdf_path = os.path.abspath(
        os.path.join(
            args.out_dir,
            f"{stem}.pdf"
        )
    )

    try:

        generate_pdf(
            html_path,
            pdf_path
        )

        print(
            f"PDF report written: {pdf_path}"
        )

    except Exception as e:

        print(
            f"ERROR: PDF generation failed: {e}",
            file=sys.stderr
        )

        print(
            "\n"
            "If Playwright Chromium is not installed, run:\n"
            "    python -m playwright install chromium\n"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()