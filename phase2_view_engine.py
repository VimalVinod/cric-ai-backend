
r"""
phase2_view_engine.py
=====================

Bowling Biomechanics Analysis System
PHASE 2 - VIEW ROBUSTNESS ENGINE

Purpose:
    Analyze a bowling video from different camera views and determine:

        1. Camera/view classification
        2. View quality
        3. Body visibility
        4. View-aware measurement reliability
        5. Phase-1 biomechanical measurements
        6. View-aware technical scoring
        7. Risk indicators
        8. Recommendations

Supported view classes:

    FRONT
    REAR
    SIDE
    DIAGONAL
    UNKNOWN

Important:
    This system does NOT claim that every biomechanical measurement
    is equally reliable from every camera angle.

Usage:

    ..\\.venv\\Scripts\\python.exe phase2_view_engine.py "test_data\side_bowler1.mp4" --arm right

    ..\.venv\Scripts\python.exe phase2_view_engine.py "test_data\front_bowler2.mp4" --arm left

    ..\.venv\Scripts\python.exe phase2_view_engine.py "test_data\back_bowler3.mp4" --arm right

Optional:

    --output "test_data\side_phase2.json"
    --model "yolov8n-pose.pt"
"""

import argparse
import json
import math
import os
import statistics

import cv2
import numpy as np

from phase1_engine import (
    PoseDetector,
    BowlingArmTracker,
    calculate_measurements,
    aggregate_measurements,
    calculate_reliability,
    calculate_scores,
    build_risks,
    build_recommendations,
    calculate_detection_rate,
    calculate_body_scale,
    detect_release,
)


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "PHASE-2"

MIN_CONF = 0.30

VIEW_NAMES = [
    "front",
    "rear",
    "side",
    "diagonal",
    "unknown",
]


# ============================================================
# KEYPOINT NAMES
# ============================================================

KEYPOINTS = [
    "nose",

    "left_shoulder",
    "right_shoulder",

    "left_elbow",
    "right_elbow",

    "left_wrist",
    "right_wrist",

    "left_hip",
    "right_hip",

    "left_knee",
    "right_knee",

    "left_ankle",
    "right_ankle",
]


# ============================================================
# BASIC HELPERS
# ============================================================

def point_xy(points, name, min_conf=MIN_CONF):
    """
    Return [x, y] if point exists and confidence is sufficient.
    """

    if points is None:
        return None

    point = points.get(name)

    if point is None:
        return None

    if isinstance(point, dict):

        x = point.get("x")
        y = point.get("y")
        confidence = point.get(
            "confidence",
            point.get("conf", 0.0)
        )

    else:

        if len(point) < 3:
            return None

        x = point[0]
        y = point[1]
        confidence = point[2]

    if x is None or y is None:
        return None

    if float(confidence) < min_conf:
        return None

    return np.array(
        [float(x), float(y)],
        dtype=float
    )


def distance(a, b):

    if a is None or b is None:
        return None

    return float(
        np.linalg.norm(
            np.asarray(a) -
            np.asarray(b)
        )
    )


def angle_between_vectors(v1, v2):

    if v1 is None or v2 is None:
        return None

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return None

    value = np.dot(v1, v2) / (n1 * n2)

    value = np.clip(value, -1.0, 1.0)

    return float(
        np.degrees(
            np.arccos(value)
        )
    )


# ============================================================
# FRAME-LEVEL BODY VISIBILITY
# ============================================================

def calculate_frame_visibility(points):

    if points is None:
        return 0.0

    visible = 0

    for name in KEYPOINTS:

        if point_xy(
            points,
            name,
            MIN_CONF
        ) is not None:

            visible += 1

    return visible / len(KEYPOINTS)


# ============================================================
# LEFT / RIGHT BODY SYMMETRY
# ============================================================

def bilateral_visibility(points):

    if points is None:
        return 0.0

    pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]

    valid_pairs = 0

    for left, right in pairs:

        if (
            point_xy(points, left) is not None
            and
            point_xy(points, right) is not None
        ):
            valid_pairs += 1

    return valid_pairs / len(pairs)


# ============================================================
# BODY ORIENTATION ESTIMATION
# ============================================================
def estimate_body_orientation(points):
    """
    Estimate camera/body orientation using normalized body geometry.

    Returns:
        {
            "width_ratio": ...,
            "hip_ratio": ...,
            "symmetry": ...,
            "nose_center": ...,
            "score": ...
        }

    The result is a heuristic camera-view estimate.
    It is NOT treated as biomechanical ground truth.
    """

    if points is None:
        return None

    ls = point_xy(points, "left_shoulder")
    rs = point_xy(points, "right_shoulder")

    lh = point_xy(points, "left_hip")
    rh = point_xy(points, "right_hip")

    nose = point_xy(points, "nose")

    if ls is None or rs is None:
        return None

    shoulder_mid = (ls + rs) / 2.0

    shoulder_width = distance(
        ls,
        rs
    )

    if shoulder_width is None or shoulder_width <= 1:
        return None

    # --------------------------------------------------------
    # Torso length
    # --------------------------------------------------------

    torso_length = None

    if lh is not None and rh is not None:

        hip_mid = (lh + rh) / 2.0

        torso_length = distance(
            shoulder_mid,
            hip_mid
        )

    if torso_length is None or torso_length <= 1:

        torso_length = shoulder_width

    # --------------------------------------------------------
    # Normalized shoulder / hip width
    # --------------------------------------------------------

    shoulder_ratio = (
        shoulder_width /
        max(torso_length, 1.0)
    )

    if (
        lh is not None
        and
        rh is not None
    ):

        hip_width = distance(
            lh,
            rh
        )

        if hip_width is not None:

            hip_ratio = (
                hip_width /
                max(torso_length, 1.0)
            )

        else:

            hip_ratio = (
                shoulder_ratio * 0.80
            )

    else:

        hip_ratio = (
            shoulder_ratio * 0.80
        )

    # --------------------------------------------------------
    # Left/right shoulder symmetry
    # --------------------------------------------------------

    shoulder_left = distance(
        ls,
        shoulder_mid
    )

    shoulder_right = distance(
        rs,
        shoulder_mid
    )

    if (
        shoulder_left is not None
        and
        shoulder_right is not None
        and
        max(
            shoulder_left,
            shoulder_right
        ) > 0
    ):

        symmetry = min(
            shoulder_left,
            shoulder_right
        ) / max(
            shoulder_left,
            shoulder_right
        )

    else:

        symmetry = 0.0

    # --------------------------------------------------------
    # Nose position relative to shoulder center
    #
    # Useful mainly for distinguishing front/rear.
    # In rear views the nose is usually poorly detected or
    # absent. Therefore this is only a supporting signal.
    # --------------------------------------------------------

    if nose is not None:

        nose_center = (
            abs(
                nose[0] -
                shoulder_mid[0]
            )
            /
            max(
                shoulder_width,
                1.0
            )
        )

    else:

        nose_center = None

    # --------------------------------------------------------
    # Combined orientation score
    #
    # Larger value:
    #     more frontal/rear
    #
    # Smaller value:
    #     more side-on
    # --------------------------------------------------------

    width_component = min(
        shoulder_ratio / 1.2,
        1.0
    )

    hip_component = min(
        hip_ratio / 0.9,
        1.0
    )

    symmetry_component = float(
        max(
            0.0,
            min(
                1.0,
                symmetry
            )
        )
    )

    score = (
        0.50 * width_component +
        0.25 * hip_component +
        0.25 * symmetry_component
    )

    return {
        "width_ratio": float(
            shoulder_ratio
        ),

        "hip_ratio": float(
            hip_ratio
        ),

        "symmetry": float(
            symmetry_component
        ),

        "nose_center": (
            float(nose_center)
            if nose_center is not None
            else None
        ),

        "score": float(
            score
        ),
    }

# ============================================================
# VIEW CLASSIFICATION
# ============================================================
def classify_view(history):
    """
    Classify camera view as:

        front
        rear
        side
        diagonal
        unknown

    Uses:
        - normalized body width
        - body symmetry
        - nose visibility
        - temporal stability

    The classifier is heuristic and should be calibrated
    further using labelled videos.
    """

    samples = []

    for points in history:

        if points is None:
            continue

        orientation = estimate_body_orientation(
            points
        )

        if orientation is None:
            continue

        visibility = calculate_frame_visibility(
            points
        )

        samples.append(
            {
                "orientation": orientation,
                "visibility": visibility
            }
        )

    if not samples:

        return {
            "detected": "unknown",
            "quality": 0.0,
            "orientationScore": None,
        }

    # --------------------------------------------------------
    # Temporal statistics
    # --------------------------------------------------------

    orientation_scores = [
        item["orientation"]["score"]
        for item in samples
    ]

    width_ratios = [
        item["orientation"]["width_ratio"]
        for item in samples
    ]

    hip_ratios = [
        item["orientation"]["hip_ratio"]
        for item in samples
    ]

    symmetries = [
        item["orientation"]["symmetry"]
        for item in samples
    ]

    visibility_values = [
        item["visibility"]
        for item in samples
    ]

    median_score = float(
        statistics.median(
            orientation_scores
        )
    )

    median_width = float(
        statistics.median(
            width_ratios
        )
    )

    median_hip = float(
        statistics.median(
            hip_ratios
        )
    )

    median_symmetry = float(
        statistics.median(
            symmetries
        )
    )

    median_visibility = float(
        statistics.median(
            visibility_values
        )
    )

    # --------------------------------------------------------
    # Nose visibility
    # --------------------------------------------------------

    nose_visible = 0
    nose_total = 0

    for points in history:

        if points is None:
            continue

        nose_total += 1

        if point_xy(
            points,
            "nose",
            MIN_CONF
        ) is not None:

            nose_visible += 1

    if nose_total > 0:

        nose_visibility = (
            nose_visible /
            nose_total
        )

    else:

        nose_visibility = 0.0

    # --------------------------------------------------------
    # Temporal stability
    # --------------------------------------------------------

    if len(orientation_scores) >= 2:

        try:

            orientation_std = float(
                statistics.pstdev(
                    orientation_scores
                )
            )

        except statistics.StatisticsError:

            orientation_std = 0.0

    else:

        orientation_std = 0.0

    stability = max(
        0.0,
        min(
            1.0,
            1.0 -
            (
                orientation_std /
                0.25
            )
        )
    )

    # ========================================================
    # VIEW CLASSIFICATION
    # ========================================================

    # --------------------------------------------------------
    # 1. SIDE
    #
    # Your side video:
    #     orientation ≈ 0.518
    #
    # Anything below 0.58 is treated as side.
    # --------------------------------------------------------

    if median_score < 0.58:

        detected_view = "side"

    # --------------------------------------------------------
    # 2. STRONG FRONT / REAR
    #
    # For wider body projections we use nose visibility.
    #
    # Front:
    #     nose usually visible
    #
    # Rear:
    #     nose usually absent / poorly detected
    # --------------------------------------------------------

    elif median_score >= 0.55:

        if nose_visibility >= 0.60:

            detected_view = "front"

        elif nose_visibility <= 0.35:

            detected_view = "rear"

        else:

            # Nose evidence is ambiguous.
            detected_view = "diagonal"

    # --------------------------------------------------------
    # 3. Additional geometry correction
    # --------------------------------------------------------

    # Strong side geometry should override ambiguous nose data.
    if (
        median_score < 0.60
        and
        median_width < 0.58
    ):

        detected_view = "side"

    # Strong front geometry.
    elif (
        median_score >= 0.55
        and
        nose_visibility >= 0.60
        and
        median_symmetry >= 0.85
    ):

        detected_view = "front"

    # Strong rear geometry.
    elif (
        median_score >= 0.55
        and
        nose_visibility <= 0.35
        and
        median_symmetry >= 0.85
    ):

        detected_view = "rear"

    # --------------------------------------------------------
    # Quality calculation
    # --------------------------------------------------------

    geometry_quality = (
        0.40 *
        min(
            median_score / 0.85,
            1.0
        )
        +
        0.30 *
        median_visibility
        +
        0.30 *
        stability
    )

    quality = float(
        max(
            0.0,
            min(
                1.0,
                geometry_quality
            )
        )
    )

    # --------------------------------------------------------
    # Reduce quality when view is genuinely ambiguous
    # --------------------------------------------------------

    if detected_view == "diagonal":

        quality *= 0.85

    # --------------------------------------------------------
    # Diagnostic output
    # --------------------------------------------------------

    print()
    print("VIEW CLASSIFICATION DETAILS")
    print("-" * 60)

    print(
        f"Median orientation : "
        f"{median_score:.3f}"
    )

    print(
        f"Shoulder ratio     : "
        f"{median_width:.3f}"
    )

    print(
        f"Hip ratio          : "
        f"{median_hip:.3f}"
    )

    print(
        f"Symmetry           : "
        f"{median_symmetry:.3f}"
    )

    print(
        f"Nose visibility    : "
        f"{nose_visibility:.3f}"
    )

    print(
        f"Stability          : "
        f"{stability:.3f}"
    )

    print(
        f"Final view         : "
        f"{detected_view}"
    )

    return {

        "detected":
            detected_view,

        "quality":
            round(
                quality,
                3
            ),

        "orientationScore":
            round(
                median_score,
                3
            ),

        "shoulderRatio":
            round(
                median_width,
                3
            ),

        "hipRatio":
            round(
                median_hip,
                3
            ),

        "symmetry":
            round(
                median_symmetry,
                3
            ),

        "noseVisibility":
            round(
                nose_visibility,
                3
            ),

        "stability":
            round(
                stability,
                3
            ),
    }

    # --------------------------------------------------------
    # Heuristic thresholds
    # --------------------------------------------------------

    # These values are not biomechanical truths.
    # They are camera-view heuristics and can be calibrated later.

    if median_orientation >= 0.95:

        view = "front_or_rear"

    elif median_orientation <= 0.48:

        view = "side"

    else:

        view = "diagonal"

    # --------------------------------------------------------
    # Front vs rear
    # --------------------------------------------------------

    if view == "front_or_rear":

        # Nose visibility helps distinguish front from rear.
        nose_visibility = []

        for points in history:

            if points is None:
                continue

            nose_visibility.append(
                1.0
                if point_xy(
                    points,
                    "nose"
                ) is not None
                else 0.0
            )

        nose_score = (
            float(
                statistics.mean(
                    nose_visibility
                )
            )
            if nose_visibility
            else 0.0
        )

        if nose_score >= 0.45:

            view = "front"

        else:

            view = "rear"

    # --------------------------------------------------------
    # View quality
    # --------------------------------------------------------

    quality = (
        0.60 * median_visibility +
        0.40 * min(
            median_orientation / 1.2,
            1.0
        )
    )

    quality = float(
        max(
            0.0,
            min(
                1.0,
                quality
            )
        )
    )

    return {
        "detected": view,
        "quality": round(
            quality,
            3
        ),
        "orientationScore": round(
            median_orientation,
            3
        ),
    }


# ============================================================
# VIEW-SPECIFIC MEASUREMENT RELIABILITY
# ============================================================

def view_measurement_reliability(
    view,
    measurement_name,
    base_reliability
):

    """
    Apply a view-aware reliability modifier.

    IMPORTANT:

    These modifiers do NOT say that a measurement is impossible.

    They say how confidently the system should interpret
    the measurement from this camera view.

    Later, after collecting more labelled data, these values
    should be calibrated.
    """

    base = float(
        max(
            0.0,
            min(
                1.0,
                base_reliability
            )
        )
    )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    multiplier = 1.0

    # --------------------------------------------------------
    # SIDE VIEW
    # --------------------------------------------------------

    if view == "side":

        if measurement_name in [
            "elbowAngle",
            "frontKneeAngle",
            "backKneeAngle",
            "trunkForwardFlexion",
            "frontFootOffset",
        ]:

            multiplier = 1.0

        elif measurement_name in [
            "hipShoulderSeparation",
            "shoulderLineAngle",
            "hipLineAngle",
        ]:

            multiplier = 0.75

        elif measurement_name == "trunkLateralFlexion":

            multiplier = 0.75

    # --------------------------------------------------------
    # FRONT VIEW
    # --------------------------------------------------------

    elif view == "front":

        if measurement_name in [
            "trunkLateralFlexion",
            "shoulderLineAngle",
            "hipLineAngle",
        ]:

            multiplier = 1.0

        elif measurement_name in [
            "elbowAngle",
            "frontKneeAngle",
        ]:

            multiplier = 0.85

        elif measurement_name == "trunkForwardFlexion":

            multiplier = 0.55

        elif measurement_name == "hipShoulderSeparation":

            multiplier = 0.65

        elif measurement_name == "frontFootOffset":

            multiplier = 0.90

    # --------------------------------------------------------
    # REAR VIEW
    # --------------------------------------------------------

    elif view == "rear":

        if measurement_name in [
            "trunkLateralFlexion",
            "shoulderLineAngle",
            "hipLineAngle",
        ]:

            multiplier = 1.0

        elif measurement_name in [
            "elbowAngle",
            "frontKneeAngle",
        ]:

            multiplier = 0.85

        elif measurement_name == "trunkForwardFlexion":

            multiplier = 0.55

        elif measurement_name == "hipShoulderSeparation":

            multiplier = 0.65

        elif measurement_name == "frontFootOffset":

            multiplier = 0.90

    # --------------------------------------------------------
    # DIAGONAL
    # --------------------------------------------------------

    elif view == "diagonal":

        multiplier = 0.85

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    else:

        multiplier = 0.50

    return float(
        max(
            0.0,
            min(
                1.0,
                base * multiplier
            )
        )
    )


# ============================================================
# VIEW-AWARE RELIABILITY
# ============================================================

def calculate_view_reliability(
    measurements,
    base_reliability,
    view
):

    result = {}

    for name, value in measurements.items():

        base = base_reliability.get(
            name,
            0.0
        )

        result[name] = round(
            view_measurement_reliability(
                view,
                name,
                base
            ),
            3
        )

    return result


# ============================================================
# VIEW-SPECIFIC USABILITY
# ============================================================

def measurement_status(
    measurement_name,
    reliability
):

    if reliability >= 0.80:

        return "high"

    if reliability >= 0.60:

        return "moderate"

    if reliability >= 0.40:

        return "low"

    return "unreliable"


# ============================================================
# VIEW WARNINGS
# ============================================================

def build_view_warnings(
    view,
    reliability
):

    warnings = []

    for name, value in reliability.items():

        status = measurement_status(
            name,
            value
        )

        if status == "unreliable":

            warnings.append(
                f"{name} should not be strongly interpreted from this view."
            )

        elif status == "low":

            warnings.append(
                f"{name} has limited confidence from this view."
            )

    if view in [
        "front",
        "rear"
    ]:

        if reliability.get(
            "trunkForwardFlexion",
            0
        ) < 0.70:

            warnings.append(
                "Forward trunk flexion is strongly view-dependent from this camera position."
            )

        if reliability.get(
            "hipShoulderSeparation",
            0
        ) < 0.70:

            warnings.append(
                "Hip-shoulder separation should be interpreted cautiously from this view."
            )

    return warnings


# ============================================================
# VIEW-AWARE SCORE
# ============================================================

def calculate_view_aware_score(
    parameter_scores,
    reliability
):

    total = 0.0
    weight_total = 0.0

    # Equal weights for Phase 2.
    # Phase 3 will introduce risk-specific weighting.

    for name, data in parameter_scores.items():

        score = data.get(
            "score"
        )

        if score is None:
            continue

        rel = reliability.get(
            name,
            0.0
        )

        if rel <= 0:
            continue

        weight = rel

        total += (
            float(score) *
            weight
        )

        weight_total += weight

    if weight_total <= 0:

        return None

    return float(
        total /
        weight_total
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_video(
    input_path,
    arm="right",
    model_path="yolov8n-pose.pt",
    output_path=None
):

    print("=" * 60)
    print("BOWLING BIOMECHANICS ANALYSIS SYSTEM")
    print("PHASE 2 - VIEW ROBUSTNESS")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not os.path.exists(
        input_path
    ):

        raise FileNotFoundError(
            f"Video not found: {input_path}"
        )

    # --------------------------------------------------------
    # Pose detector
    # --------------------------------------------------------

    detector = PoseDetector(
        model_path=model_path
    )

    tracker = BowlingArmTracker(
        arm=arm
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        raise ValueError(
            f"Could not open video: {input_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print()
    print("VIDEO")
    print("-" * 60)

    print(
        f"Path       : {input_path}"
    )

    print(
        f"FPS        : {fps:.2f}"
    )

    print(
        f"Frames     : {frame_count}"
    )

    print(
        f"Resolution : {width} x {height}"
    )

    print()

    # --------------------------------------------------------
    # Frame processing
    # --------------------------------------------------------

    history = []

    measurements_history = []

    frame_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        points = detector.detect(
            frame
        )

        if points is not None:

            points = tracker.update(
                points
            )

        history.append(
            points
        )

        if points is None:

            measurements_history.append(
                {}
            )

        else:

            measurements = (
                calculate_measurements(
                    points,
                    arm
                )
            )

            measurements_history.append(
                measurements
            )

        frame_index += 1

        if frame_index % 30 == 0:

            print(
                f"Processing: "
                f"{frame_index}/"
                f"{max(frame_count, 1)}"
            )

    cap.release()

    # --------------------------------------------------------
    # View classification
    # --------------------------------------------------------

    view_info = classify_view(
        history
    )

    detected_view = view_info[
        "detected"
    ]

    print()
    print("VIEW")
    print("-" * 60)

    print(
        f"Detected : {detected_view}"
    )

    print(
        f"Quality  : {view_info['quality']}"
    )

    print(
        f"Orientation score : "
        f"{view_info['orientationScore']}"
    )

    # --------------------------------------------------------
    # Release detection
    # --------------------------------------------------------

    release_frame = detect_release(
        history,
        arm
    )

    if release_frame is None:

        print(
            "Release frame: not detected"
        )

        analysis_start = max(
            0,
            len(history) // 2 - 5
        )

        analysis_end = min(
            len(history),
            analysis_start + 11
        )

    else:

        analysis_start = max(
            0,
            release_frame - 5
        )

        analysis_end = min(
            len(history),
            release_frame + 6
        )

    window = (
        measurements_history[
            analysis_start:
            analysis_end
        ]
    )

    # --------------------------------------------------------
    # Aggregate measurements
    # --------------------------------------------------------

    measurements = aggregate_measurements(
        window
    )
    # --------------------------------------------------------
    # Base reliability
    # --------------------------------------------------------
    base_reliability = calculate_reliability(
    history,
    analysis_start,
    analysis_end - 1,
    measurements

    )
    # --------------------------------------------------------
    # View-aware reliability
    # --------------------------------------------------------

    view_reliability = calculate_view_reliability(
        measurements,
        base_reliability,
        detected_view
    )

       # --------------------------------------------------------
    # Existing Phase-1 parameter scores
    # --------------------------------------------------------

    parameter_scores, _ = calculate_scores(
        measurements,
        base_reliability
    )

    # Apply view-aware reliability to parameter scores
    for name in parameter_scores:

        parameter_scores[name]["reliability"] = (
            view_reliability.get(
                name,
                parameter_scores[name].get(
                    "reliability",
                    0.0
                )
            )
        )

    # --------------------------------------------------------
    # View-aware technical score
    # --------------------------------------------------------

    technical_score = calculate_view_aware_score(
        parameter_scores,
        view_reliability
    )

    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------

    risks = build_risks(
        measurements
    )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = build_recommendations(
        measurements,
        risks
    )

    # --------------------------------------------------------
    # View warnings
    # --------------------------------------------------------

    view_warnings = build_view_warnings(
        detected_view,
        view_reliability
    )

    # --------------------------------------------------------
    # Detection rate
    # --------------------------------------------------------

    detection_rate = calculate_detection_rate(
        history
    )

    # --------------------------------------------------------
    # Body scale
    # --------------------------------------------------------

    body_scale = calculate_body_scale(
        history
    )

    # --------------------------------------------------------
    # Output JSON
    # --------------------------------------------------------

    result = {

        "status": "ok",

        "version": VERSION,

        "video": {

            "path": input_path,

            "fps": round(
                float(fps),
                3
            ),

            "frameCount": frame_count,

            "width": width,

            "height": height,
        },

        "bowlingArm": arm,

        "view": {

            "detected": detected_view,

            "quality": view_info[
                "quality"
            ],

            "orientationScore":
                view_info[
                    "orientationScore"
                ],
        },

        "bodyScale": body_scale,

        "releaseFrame": {

            "index": release_frame,

            "timestampSeconds": (
                round(
                    release_frame / fps,
                    3
                )
                if release_frame is not None
                else None
            ),
        },

        "analysisWindow": {

            "startFrame":
                analysis_start,

            "endFrame":
                analysis_end - 1,

            "framesUsed":
                max(
                    0,
                    analysis_end -
                    analysis_start
                ),
        },

        "measurements": measurements,

        "parameterScores":
            parameter_scores,

        "reliability": {

            name: value

            for name, value
            in view_reliability.items()
        },

        "viewWarnings":
            view_warnings,

        "technicalScore":
            (
                round(
                    technical_score,
                    3
                )
                if technical_score is not None
                else None
            ),

        "riskIndicators":
            risks,

        "recommendations":
            recommendations,

        "detectionRate":
            round(
                float(
                    detection_rate
                ),
                3
            ),
    }

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    if output_path is None:

        base, _ = os.path.splitext(
            input_path
        )

        output_path = (
            base +
            "_phase2.json"
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
    print("PHASE 2 RESULT")
    print("=" * 60)

    print()
    print(
        f"View              : "
        f"{detected_view}"
    )

    print(
        f"View quality      : "
        f"{view_info['quality']:.3f}"
    )

    print(
        f"Detection rate    : "
        f"{detection_rate:.3f}"
    )

    print(
        f"Technical score   : "
        f"{technical_score:.2f}"
        if technical_score is not None
        else
        "Technical score   : --"
    )

    print()
    print("VIEW-AWARE RELIABILITY")
    print("-" * 60)

    for name, value in view_reliability.items():

        status = measurement_status(
            name,
            value
        )

        print(
            f"{name:30s} "
            f"{value:.3f} "
            f"({status})"
        )

    print()

    if view_warnings:

        print("VIEW WARNINGS")
        print("-" * 60)

        for warning in view_warnings:

            print(
                f"- {warning}"
            )

    else:

        print(
            "View warnings: none"
        )

    print()
    print(
        f"Output: {output_path}"
    )

    print()
    print("=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)

    return result


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "Bowling Biomechanics Phase 2 "
        "View Robustness Engine"
    )

    parser.add_argument(
        "video",
        help="Input bowling video"
    )

    parser.add_argument(
        "--arm",
        choices=[
            "right",
            "left"
        ],
        default="right",
        help="Bowling arm"
    )

    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="YOLO pose model"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path"
    )

    args = parser.parse_args()

    try:

        analyze_video(
            input_path=args.video,
            arm=args.arm,
            model_path=args.model,
            output_path=args.output
        )

    except Exception as exc:

        print()
        print("=" * 60)
        print("PHASE 2 ERROR")
        print("=" * 60)

        print(
            str(exc)
        )

        print("=" * 60)

        raise


if __name__ == "__main__":

    main()

