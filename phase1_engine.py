
"""
phase1_engine.py
============================================================

CRIC AI - BOWLING BIOMECHANICS ANALYSIS
PHASE 1 - RENDER LIGHTWEIGHT ENGINE

Render Free / CPU / Low-Memory Architecture

IMPORTANT:
This is a biomechanical screening system.
It is NOT a medical diagnosis and does not predict injury
with certainty.

NEW LIGHTWEIGHT FLOW
--------------------

Video
  |
  +--> Read metadata
  |
  +--> Select limited key frames
  |
  +--> Resize frames
  |
  +--> YOLOv8n-Pose
  |
  +--> Extract body keypoints
  |
  +--> Estimate delivery/release
  |
  +--> Biomechanical measurements
  |
  +--> Reliability
  |
  +--> Technical score
  |
  +--> Risk indicators
  |
  +--> Recommendations
  |
  +--> JSON report

DESIGN GOALS
------------

1. Very small CPU workload
2. Very low RAM usage
3. Fixed maximum inference count
4. No full-video pose tracking
5. No large frame history
6. Compatible with Render Free
7. Preserve existing API JSON structure

"""

import argparse
import json
import math
import os
import gc

from statistics import median

import cv2
import numpy as np
import torch

# ============================================================
# CPU / MEMORY CONFIGURATION
# ============================================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from ultralytics import YOLO


# ============================================================
# VERSION
# ============================================================

VERSION = "PHASE-1-RENDER-LITE-V2"


# ============================================================
# RENDER LIMITS
# ============================================================

# Very small YOLO input.
YOLO_IMAGE_SIZE = 256

# Maximum frame dimension before YOLO.
MAX_FRAME_DIMENSION = 384

# HARD LIMIT.
# No matter how long the video is, we never run more than
# this many YOLO inferences.
MAX_INFERENCES = 20

# Minimum confidence.
MIN_CONF = 0.25
POSE_CONF = 0.30

# Number of frames around estimated release used for
# biomechanical aggregation.
MEASUREMENT_WINDOW = 3

# Minimum number of valid samples required for a useful
# release estimation.
MIN_VALID_SAMPLES = 4


# ============================================================
# COCO KEYPOINTS
# ============================================================

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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
# HELPERS
# ============================================================

def safe_float(value):
    """
    Convert a numeric value to a normal JSON-safe float.
    """

    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return round(value, 3)

    except Exception:
        return None


def point_from_dict(points, name, min_conf=MIN_CONF):
    """
    Return numpy [x, y] when the keypoint is reliable.
    """

    if not points:
        return None

    value = points.get(name)

    if value is None or len(value) < 3:
        return None

    try:
        x = float(value[0])
        y = float(value[1])
        confidence = float(value[2])
    except Exception:
        return None

    if confidence < min_conf:
        return None

    return np.array([x, y], dtype=float)


def distance(a, b):
    """
    Euclidean distance between two points.
    """

    if a is None or b is None:
        return None

    return float(
        np.linalg.norm(
            np.asarray(a, dtype=float)
            -
            np.asarray(b, dtype=float)
        )
    )


def angle_3pt(a, b, c):
    """
    Angle ABC in degrees.
    """

    if a is None or b is None or c is None:
        return None

    ba = (
        np.asarray(a, dtype=float)
        -
        np.asarray(b, dtype=float)
    )

    bc = (
        np.asarray(c, dtype=float)
        -
        np.asarray(b, dtype=float)
    )

    denominator = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denominator <= 1e-8:
        return None

    cosine = (
        np.dot(ba, bc)
        /
        denominator
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def line_angle_from_vertical(p1, p2):
    """
    Angle of p1 -> p2 relative to vertical.

    0 = vertical
    Larger = more inclination
    """

    if p1 is None or p2 is None:
        return None

    vector = (
        np.asarray(p2, dtype=float)
        -
        np.asarray(p1, dtype=float)
    )

    length = np.linalg.norm(vector)

    if length <= 1e-8:
        return None

    vertical = np.array(
        [0.0, -1.0],
        dtype=float
    )

    cosine = (
        np.dot(vector, vertical)
        /
        length
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


# ============================================================
# FRAME RESIZE
# ============================================================

def resize_for_inference(frame):
    """
    Resize frame aggressively before YOLO.

    This prevents 1080p / 4K frames from consuming large
    amounts of memory.
    """

    if frame is None:
        return None

    height, width = frame.shape[:2]

    largest = max(
        height,
        width
    )

    if largest <= MAX_FRAME_DIMENSION:
        return frame

    scale = (
        MAX_FRAME_DIMENSION
        /
        largest
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# FRAME SELECTION
# ============================================================

def select_frame_indices(total_frames):
    """
    Select a small number of evenly distributed frames.

    IMPORTANT:
    We intentionally do NOT process the entire video.

    This is the main Render optimization.
    """

    if total_frames <= 0:
        return []

    if total_frames <= MAX_INFERENCES:
        return list(range(total_frames))

    indices = np.linspace(
        0,
        total_frames - 1,
        MAX_INFERENCES,
        dtype=int
    )

    # Remove duplicates.
    indices = sorted(
        set(
            int(x)
            for x in indices
        )
    )

    return indices


# ============================================================
# YOLO POSE DETECTOR
# ============================================================

class PoseDetector:
    """
    Minimal YOLO pose detector.

    Only one model instance is loaded.
    """

    def __init__(self, model_path):

        print()
        print("Loading lightweight YOLO pose model...")
        print(
            f"YOLO size       : {YOLO_IMAGE_SIZE}"
        )
        print(
            f"Max frame size  : {MAX_FRAME_DIMENSION}"
        )
        print(
            f"Max inferences  : {MAX_INFERENCES}"
        )

        self.model = YOLO(
            model_path
        )

        print("YOLO model loaded.")

    def detect(self, frame):

        if frame is None:
            return None

        results = None

        try:

            results = self.model(
                frame,
                verbose=False,
                conf=POSE_CONF,
                imgsz=YOLO_IMAGE_SIZE,
                device="cpu",
                max_det=1,
                classes=[0]
            )

            if not results:
                return None

            result = results[0]

            if result.keypoints is None:
                return None

            data = (
                result
                .keypoints
                .data
                .cpu()
                .numpy()
            )

            if data.ndim != 3:
                return None

            if len(data) == 0:
                return None

            # Because max_det=1 this is normally one person.
            person = data[0]

            if len(person) != len(
                KEYPOINT_NAMES
            ):
                return None

            points = {}

            for index, name in enumerate(
                KEYPOINT_NAMES
            ):

                x = float(
                    person[index][0]
                )

                y = float(
                    person[index][1]
                )

                confidence = float(
                    person[index][2]
                )

                points[name] = (
                    x,
                    y,
                    confidence
                )

            # Require some important body points.
            important = [
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
            ]

            visible = sum(
                1
                for name in important
                if (
                    points.get(name) is not None
                    and
                    points[name][2] >= MIN_CONF
                )
            )

            if visible < 2:
                return None

            return points

        except Exception as error:

            print(
                f"YOLO frame error: {error}"
            )

            return None

        finally:

            del results

            # Do not repeatedly call heavy garbage collection.
            # Python handles temporary objects naturally.


# ============================================================
# BODY SCALE
# ============================================================

def torso_length(points):

    if not points:
        return None

    left_shoulder = point_from_dict(
        points,
        "left_shoulder"
    )

    right_shoulder = point_from_dict(
        points,
        "right_shoulder"
    )

    left_hip = point_from_dict(
        points,
        "left_hip"
    )

    right_hip = point_from_dict(
        points,
        "right_hip"
    )

    if any(
        x is None
        for x in [
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip
        ]
    ):
        return None

    shoulder_mid = (
        left_shoulder
        +
        right_shoulder
    ) / 2.0

    hip_mid = (
        left_hip
        +
        right_hip
    ) / 2.0

    value = distance(
        shoulder_mid,
        hip_mid
    )

    if value is None or value <= 0:
        return None

    return value


# ============================================================
# RELEASE ESTIMATION
# ============================================================

def wrist_position(points, arm):

    name = (
        "right_wrist"
        if arm == "right"
        else "left_wrist"
    )

    return point_from_dict(
        points,
        name
    )


def estimate_release(samples, arm):
    """
    Estimate release from wrist movement.

    Because we use only a limited number of frames,
    this is deliberately simple.

    Returns:
        sample index
        or None
    """

    if len(samples) < MIN_VALID_SAMPLES:
        return None

    speeds = []

    for i in range(1, len(samples)):

        previous = wrist_position(
            samples[i - 1]["points"],
            arm
        )

        current = wrist_position(
            samples[i]["points"],
            arm
        )

        if previous is None or current is None:

            speeds.append(None)

        else:

            speeds.append(
                distance(
                    previous,
                    current
                )
            )

    valid = [
        (i + 1, speed)
        for i, speed in enumerate(speeds)
        if speed is not None
    ]

    if not valid:
        return None

    # Ignore the very first part of the video.
    minimum_index = max(
        1,
        int(len(samples) * 0.15)
    )

    valid = [
        item
        for item in valid
        if item[0] >= minimum_index
    ]

    if not valid:
        return None

    release_index, maximum_speed = max(
        valid,
        key=lambda item: item[1]
    )

    if maximum_speed is None:
        return None

    return int(
        release_index
    )


# ============================================================
# BIOMECHANICAL MEASUREMENTS
# ============================================================

def calculate_measurements(points, arm):

    if points is None:
        return {}

    def p(name):
        return point_from_dict(
            points,
            name
        )

    nose = p("nose")

    left_shoulder = p(
        "left_shoulder"
    )

    right_shoulder = p(
        "right_shoulder"
    )

    left_elbow = p(
        "left_elbow"
    )

    right_elbow = p(
        "right_elbow"
    )

    left_wrist = p(
        "left_wrist"
    )

    right_wrist = p(
        "right_wrist"
    )

    left_hip = p(
        "left_hip"
    )

    right_hip = p(
        "right_hip"
    )

    left_knee = p(
        "left_knee"
    )

    right_knee = p(
        "right_knee"
    )

    left_ankle = p(
        "left_ankle"
    )

    right_ankle = p(
        "right_ankle"
    )

    if arm == "right":

        bowling_shoulder = right_shoulder
        bowling_elbow = right_elbow
        bowling_wrist = right_wrist

        front_hip = left_hip
        front_knee = left_knee
        front_ankle = left_ankle

        back_hip = right_hip
        back_knee = right_knee
        back_ankle = right_ankle

    else:

        bowling_shoulder = left_shoulder
        bowling_elbow = left_elbow
        bowling_wrist = left_wrist

        front_hip = right_hip
        front_knee = right_knee
        front_ankle = right_ankle

        back_hip = left_hip
        back_knee = left_knee
        back_ankle = left_ankle

    measurements = {}

    # --------------------------------------------------------
    # BODY MIDPOINTS
    # --------------------------------------------------------

    mid_shoulder = None
    mid_hip = None

    if (
        left_shoulder is not None
        and
        right_shoulder is not None
    ):

        mid_shoulder = (
            left_shoulder
            +
            right_shoulder
        ) / 2.0

    if (
        left_hip is not None
        and
        right_hip is not None
    ):

        mid_hip = (
            left_hip
            +
            right_hip
        ) / 2.0

    # --------------------------------------------------------
    # ELBOW
    # --------------------------------------------------------

    measurements[
        "elbowAngle"
    ] = angle_3pt(
        bowling_shoulder,
        bowling_elbow,
        bowling_wrist
    )

    # --------------------------------------------------------
    # FRONT KNEE
    # --------------------------------------------------------

    measurements[
        "frontKneeAngle"
    ] = angle_3pt(
        front_hip,
        front_knee,
        front_ankle
    )

    # --------------------------------------------------------
    # BACK KNEE
    # --------------------------------------------------------

    measurements[
        "backKneeAngle"
    ] = angle_3pt(
        back_hip,
        back_knee,
        back_ankle
    )

    # --------------------------------------------------------
    # TRUNK FORWARD FLEXION
    # --------------------------------------------------------

    measurements[
        "trunkForwardFlexion"
    ] = line_angle_from_vertical(
        mid_hip,
        mid_shoulder
    )

    # --------------------------------------------------------
    # TRUNK LATERAL FLEXION
    # --------------------------------------------------------

    if (
        mid_shoulder is not None
        and
        mid_hip is not None
    ):

        torso = distance(
            mid_shoulder,
            mid_hip
        )

        if torso and torso > 0:

            horizontal_offset = abs(
                mid_shoulder[0]
                -
                mid_hip[0]
            )

            measurements[
                "trunkLateralFlexion"
            ] = float(
                np.degrees(
                    np.arctan2(
                        horizontal_offset,
                        torso
                    )
                )
            )

    # --------------------------------------------------------
    # SHOULDER LINE
    # --------------------------------------------------------

    if (
        left_shoulder is not None
        and
        right_shoulder is not None
    ):

        vector = (
            right_shoulder
            -
            left_shoulder
        )

        measurements[
            "shoulderLineAngle"
        ] = float(
            np.degrees(
                np.arctan2(
                    vector[1],
                    vector[0]
                )
            )
        )

    # --------------------------------------------------------
    # HIP LINE
    # --------------------------------------------------------

    if (
        left_hip is not None
        and
        right_hip is not None
    ):

        vector = (
            right_hip
            -
            left_hip
        )

        measurements[
            "hipLineAngle"
        ] = float(
            np.degrees(
                np.arctan2(
                    vector[1],
                    vector[0]
                )
            )
        )

    # --------------------------------------------------------
    # HIP / SHOULDER SEPARATION
    # --------------------------------------------------------

    shoulder_angle = measurements.get(
        "shoulderLineAngle"
    )

    hip_angle = measurements.get(
        "hipLineAngle"
    )

    if (
        shoulder_angle is not None
        and
        hip_angle is not None
    ):

        difference = (
            shoulder_angle
            -
            hip_angle
        )

        difference = (
            difference
            +
            180
        ) % 360 - 180

        measurements[
            "hipShoulderSeparation"
        ] = abs(
            difference
        )

    # --------------------------------------------------------
    # FRONT FOOT OFFSET
    # --------------------------------------------------------

    if (
        front_hip is not None
        and
        front_ankle is not None
    ):

        leg_length = distance(
            front_hip,
            front_ankle
        )

        if leg_length and leg_length > 0:

            measurements[
                "frontFootOffset"
            ] = (
                abs(
                    front_hip[0]
                    -
                    front_ankle[0]
                )
                /
                leg_length
            )

    # --------------------------------------------------------
    # HEAD OFFSET
    # --------------------------------------------------------

    if (
        nose is not None
        and
        mid_shoulder is not None
    ):

        scale = torso_length(
            points
        )

        if scale and scale > 0:

            measurements[
                "headOffset"
            ] = (
                distance(
                    nose,
                    mid_shoulder
                )
                /
                scale
            )

    return {
        key: safe_float(value)
        for key, value in measurements.items()
    }


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_measurements(
    measurement_list
):

    if not measurement_list:
        return {}

    keys = set()

    for item in measurement_list:
        keys.update(
            item.keys()
        )

    result = {}

    for key in keys:

        values = []

        for item in measurement_list:

            value = item.get(
                key
            )

            if value is None:
                continue

            try:

                value = float(value)

                if math.isfinite(value):
                    values.append(value)

            except Exception:
                continue

        if values:

            result[key] = safe_float(
                median(values)
            )

        else:

            result[key] = None

    return result


# ============================================================
# RELIABILITY
# ============================================================

def calculate_reliability(
    samples,
    measurement_names
):

    if not samples:

        return {
            name: 0.0
            for name in measurement_names
        }

    dependencies = {

        "elbowAngle": [
            "right_shoulder",
            "right_elbow",
            "right_wrist"
        ],

        "frontKneeAngle": [
            "left_hip",
            "left_knee",
            "left_ankle"
        ],

        "backKneeAngle": [
            "right_hip",
            "right_knee",
            "right_ankle"
        ],

        "trunkForwardFlexion": [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip"
        ],

        "trunkLateralFlexion": [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip"
        ],

        "hipShoulderSeparation": [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip"
        ],

        "frontFootOffset": [
            "left_hip",
            "left_ankle"
        ],

        "headOffset": [
            "nose",
            "left_shoulder",
            "right_shoulder"
        ],

        "shoulderLineAngle": [
            "left_shoulder",
            "right_shoulder"
        ],

        "hipLineAngle": [
            "left_hip",
            "right_hip"
        ]
    }

    reliability = {}

    for measurement in measurement_names:

        required = dependencies.get(
            measurement,
            []
        )

        if not required:

            reliability[
                measurement
            ] = 0.0

            continue

        valid = 0

        for sample in samples:

            points = sample.get(
                "points"
            )

            if not points:
                continue

            good = True

            for name in required:

                value = points.get(
                    name
                )

                if (
                    value is None
                    or
                    len(value) < 3
                    or
                    float(value[2]) < MIN_CONF
                ):

                    good = False
                    break

            if good:
                valid += 1

        ratio = (
            valid
            /
            max(
                len(samples),
                1
            )
        )

        reliability[
            measurement
        ] = safe_float(
            min(
                1.0,
                ratio
            )
        )

    return reliability


# ============================================================
# TECHNICAL REFERENCE RANGES
# ============================================================

REFERENCE_RANGES = {

    "frontKneeAngle": (
        145.0,
        180.0
    ),

    "elbowAngle": (
        150.0,
        180.0
    ),

    "trunkLateralFlexion": (
        0.0,
        15.0
    ),

    "trunkForwardFlexion": (
        0.0,
        40.0
    ),

    "hipShoulderSeparation": (
        15.0,
        45.0
    ),

    "frontFootOffset": (
        0.0,
        0.55
    )
}


WEIGHTS = {

    "frontKneeAngle": 0.20,

    "elbowAngle": 0.15,

    "trunkLateralFlexion": 0.15,

    "trunkForwardFlexion": 0.10,

    "hipShoulderSeparation": 0.20,

    "frontFootOffset": 0.20
}


# ============================================================
# SCORING
# ============================================================

def parameter_score(
    value,
    low,
    high
):

    if value is None:
        return None

    value = float(value)

    if low <= value <= high:
        return 100.0

    span = max(
        high - low,
        1e-6
    )

    if value < low:
        deviation = low - value
    else:
        deviation = value - high

    score = (
        100.0
        -
        (
            deviation
            /
            span
        )
        *
        100.0
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0
        )
    )


def calculate_scores(
    measurements,
    reliability
):

    parameter_scores = {}

    weighted_total = 0.0
    weight_total = 0.0

    for name, reference in REFERENCE_RANGES.items():

        value = measurements.get(
            name
        )

        score = parameter_score(
            value,
            reference[0],
            reference[1]
        )

        reliability_value = reliability.get(
            name,
            0.0
        )

        parameter_scores[
            name
        ] = {

            "value":
                safe_float(value),

            "score":
                safe_float(score),

            "reliability":
                safe_float(
                    reliability_value
                )
        }

        if (
            score is not None
            and
            reliability_value is not None
            and
            reliability_value > 0
        ):

            weighted_total += (
                score
                *
                WEIGHTS[name]
                *
                reliability_value
            )

            weight_total += (
                WEIGHTS[name]
                *
                reliability_value
            )

    if weight_total > 0:

        technical_score = (
            weighted_total
            /
            weight_total
        )

    else:

        technical_score = None

    return (
        parameter_scores,
        safe_float(
            technical_score
        )
    )


# ============================================================
# RISK ENGINE
# ============================================================

def build_risks(
    measurements
):

    risks = []

    # --------------------------------------------------------
    # TRUNK
    # --------------------------------------------------------

    lateral = measurements.get(
        "trunkLateralFlexion"
    )

    if lateral is not None:

        if lateral >= 30:

            risks.append({

                "parameter":
                    "trunkLateralFlexion",

                "value":
                    safe_float(lateral),

                "bodyArea":
                    "Lower back",

                "severity":
                    "elevated",

                "message":
                    "High lateral trunk flexion detected; this may increase lower-back loading."
            })

        elif lateral >= 20:

            risks.append({

                "parameter":
                    "trunkLateralFlexion",

                "value":
                    safe_float(lateral),

                "bodyArea":
                    "Lower back",

                "severity":
                    "monitor",

                "message":
                    "Moderate lateral trunk flexion detected; monitor trunk control."
            })

    # --------------------------------------------------------
    # ELBOW
    # --------------------------------------------------------

    elbow = measurements.get(
        "elbowAngle"
    )

    if elbow is not None:

        if elbow < 135:

            risks.append({

                "parameter":
                    "elbowAngle",

                "value":
                    safe_float(elbow),

                "bodyArea":
                    "Bowling elbow",

                "severity":
                    "monitor",

                "message":
                    "Significant elbow flexion detected near the delivery phase; review with a qualified coach."
            })

        elif elbow < 145:

            risks.append({

                "parameter":
                    "elbowAngle",

                "value":
                    safe_float(elbow),

                "bodyArea":
                    "Bowling elbow",

                "severity":
                    "monitor",

                "message":
                    "Increased bowling-arm elbow flexion detected."
            })

    # --------------------------------------------------------
    # FRONT KNEE
    # --------------------------------------------------------

    knee = measurements.get(
        "frontKneeAngle"
    )

    if (
        knee is not None
        and
        knee < 135
    ):

        risks.append({

            "parameter":
                "frontKneeAngle",

            "value":
                safe_float(knee),

            "bodyArea":
                "Front knee",

            "severity":
                "monitor",

            "message":
                "Greater front-knee flexion detected during the delivery window."
        })

    # --------------------------------------------------------
    # FRONT FOOT
    # --------------------------------------------------------

    foot = measurements.get(
        "frontFootOffset"
    )

    if foot is not None:

        if foot >= 0.65:

            risks.append({

                "parameter":
                    "frontFootOffset",

                "value":
                    safe_float(foot),

                "bodyArea":
                    "Front foot / ankle",

                "severity":
                    "elevated",

                "message":
                    "Large front-foot alignment deviation detected; review lower-limb alignment."
            })

        elif foot >= 0.55:

            risks.append({

                "parameter":
                    "frontFootOffset",

                "value":
                    safe_float(foot),

                "bodyArea":
                    "Front foot / ankle",

                "severity":
                    "monitor",

                "message":
                    "Front-foot alignment deviation detected."
            })

    return risks


# ============================================================
# RECOMMENDATIONS
# ============================================================

def build_recommendations(
    risks
):

    recommendations = []

    parameters = {
        item["parameter"]
        for item in risks
    }

    if (
        "trunkLateralFlexion"
        in parameters
    ):

        recommendations.append(
            "Work on maintaining trunk stability through the delivery and follow-through."
        )

    if (
        "elbowAngle"
        in parameters
    ):

        recommendations.append(
            "Review bowling-arm movement and elbow position with a qualified coach."
        )

    if (
        "frontKneeAngle"
        in parameters
    ):

        recommendations.append(
            "Work on controlled front-leg mechanics and stable front-knee positioning."
        )

    if (
        "frontFootOffset"
        in parameters
    ):

        recommendations.append(
            "Focus on consistent front-foot placement and lower-limb alignment."
        )

    if not recommendations:

        recommendations.append(
            "No major biomechanical flags were identified in the analyzed delivery window."
        )

        recommendations.append(
            "Continue monitoring consistency across multiple deliveries."
        )

    return recommendations


# ============================================================
# DETECTION RATE
# ============================================================

def calculate_detection_rate(
    samples
):

    if not samples:
        return 0.0

    valid = sum(
        1
        for sample in samples
        if sample.get("points") is not None
    )

    return safe_float(
        valid
        /
        len(samples)
    )


# ============================================================
# BODY SCALE
# ============================================================

def calculate_body_scale(
    samples
):

    values = []

    for sample in samples:

        scale = torso_length(
            sample.get("points")
        )

        if scale is not None and scale > 0:
            values.append(scale)

    if not values:
        return None

    return safe_float(
        median(values)
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_video(
    input_path,
    arm="right",
    output_path=None,
    model_path="yolov8n-pose.pt"
):

    print()
    print("=" * 60)
    print("CRIC AI - PHASE 1")
    print("RENDER LIGHTWEIGHT BOWLING ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not os.path.exists(
        input_path
    ):

        raise FileNotFoundError(
            f"Video not found: {input_path}"
        )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    print()
    print("Input video:")
    print(input_path)

    print()
    print("Loading video metadata...")

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        raise RuntimeError(
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

    duration = (
        frame_count
        /
        fps
        if fps > 0
        else 0
    )

    print()
    print("VIDEO")
    print("-" * 60)
    print(
        f"FPS          : {fps:.2f}"
    )
    print(
        f"Frames       : {frame_count}"
    )
    print(
        f"Resolution   : {width} x {height}"
    )
    print(
        f"Duration     : {duration:.2f}s"
    )

    # --------------------------------------------------------
    # SELECT FRAMES
    # --------------------------------------------------------

    selected_indices = select_frame_indices(
        frame_count
    )

    print()
    print("LIGHTWEIGHT SAMPLING")
    print("-" * 60)
    print(
        f"Selected frames : {len(selected_indices)}"
    )
    print(
        f"Maximum YOLO    : {MAX_INFERENCES}"
    )
    print(
        f"YOLO size       : {YOLO_IMAGE_SIZE}"
    )
    print(
        f"Frame dimension : {MAX_FRAME_DIMENSION}px"
    )

    if not selected_indices:

        cap.release()

        raise RuntimeError(
            "Video contains no readable frames."
        )

    # --------------------------------------------------------
    # LOAD MODEL ONLY AFTER VIDEO VALIDATION
    # --------------------------------------------------------

    detector = PoseDetector(
        model_path
    )

    # --------------------------------------------------------
    # PROCESS SELECTED FRAMES
    # --------------------------------------------------------

    samples = []

    selected_set = set(
        selected_indices
    )

    next_position = 0

    current_frame = 0

    inference_count = 0

    print()
    print("STARTING LIMITED YOLO INFERENCE")
    print("-" * 60)

    while (
        current_frame < frame_count
        and
        inference_count < MAX_INFERENCES
    ):

        success, frame = cap.read()

        if not success:
            break

        if current_frame not in selected_set:

            current_frame += 1
            continue

        # ----------------------------------------------------
        # Resize immediately.
        # ----------------------------------------------------

        frame = resize_for_inference(
            frame
        )

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        points = detector.detect(
            frame
        )

        inference_count += 1

        timestamp = (
            current_frame
            /
            fps
            if fps > 0
            else 0
        )

        samples.append({

            "frameIndex":
                int(current_frame),

            "timestamp":
                safe_float(timestamp),

            "points":
                points
        })

        # ----------------------------------------------------
        # Release frame immediately.
        # ----------------------------------------------------

        del frame

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"Inference "
            f"{inference_count}/"
            f"{len(selected_indices)} "
            f"| frame "
            f"{current_frame}"
        )

        current_frame += 1

    cap.release()

    # --------------------------------------------------------
    # FREE MODEL CACHE
    # --------------------------------------------------------

    try:

        if hasattr(
            detector,
            "model"
        ):

            del detector.model

    except Exception:
        pass

    del detector

    gc.collect()

    # --------------------------------------------------------
    # DETECTION RATE
    # --------------------------------------------------------

    detection_rate = calculate_detection_rate(
        samples
    )

    valid_samples = [
        sample
        for sample in samples
        if sample.get("points") is not None
    ]

    print()
    print(
        f"YOLO inferences : {inference_count}"
    )

    print(
        f"Valid detections: {len(valid_samples)}"
    )

    print(
        f"Detection rate  : "
        f"{detection_rate * 100:.1f}%"
    )

    # --------------------------------------------------------
    # FALLBACK WHEN DETECTION IS TOO LOW
    # --------------------------------------------------------

    if not valid_samples:

        raise RuntimeError(
            "YOLO could not detect a reliable person "
            "in the selected frames."
        )

    # --------------------------------------------------------
    # RELEASE
    # --------------------------------------------------------

    release_sample_index = estimate_release(
        samples,
        arm
    )

    if release_sample_index is None:

        # Use a late-video fallback.
        release_sample_index = max(
            0,
            int(
                len(samples) * 0.70
            )
        )

        release_method = "fallback"

    else:

        release_method = (
            "sampled_wrist_speed_peak"
        )

    release_sample_index = int(
        np.clip(
            release_sample_index,
            0,
            max(
                len(samples) - 1,
                0
            )
        )
    )

    release_sample = samples[
        release_sample_index
    ]

    release_original_frame = int(
        release_sample[
            "frameIndex"
        ]
    )

    release_timestamp = float(
        release_sample[
            "timestamp"
        ]
    )

    # --------------------------------------------------------
    # MEASUREMENT WINDOW
    # --------------------------------------------------------

    window_start = max(
        0,
        release_sample_index
        -
        MEASUREMENT_WINDOW
    )

    window_end = min(
        len(samples) - 1,
        release_sample_index
        +
        MEASUREMENT_WINDOW
    )

    window_samples = samples[
        window_start:
        window_end + 1
    ]

    # --------------------------------------------------------
    # MEASUREMENTS
    # --------------------------------------------------------

    measurement_history = []

    for sample in window_samples:

        measurement_history.append(
            calculate_measurements(
                sample.get("points"),
                arm
            )
        )

    measurements = aggregate_measurements(
        measurement_history
    )

    # --------------------------------------------------------
    # RELIABILITY
    # --------------------------------------------------------

    reliability = calculate_reliability(
        window_samples,
        measurements.keys()
    )

    # --------------------------------------------------------
    # TECHNICAL SCORE
    # --------------------------------------------------------

    (
        parameter_scores,
        technical_score
    ) = calculate_scores(
        measurements,
        reliability
    )

    # --------------------------------------------------------
    # RISKS
    # --------------------------------------------------------

    risks = build_risks(
        measurements
    )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = build_recommendations(
        risks
    )

    # --------------------------------------------------------
    # BODY SCALE
    # --------------------------------------------------------

    body_scale = calculate_body_scale(
        samples
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {

        "status":
            "ok",

        "version":
            VERSION,

        "engine": {

            "architecture":
                "sampled-pose",

            "renderOptimized":
                True,

            "maxInferences":
                MAX_INFERENCES,

            "yoloImageSize":
                YOLO_IMAGE_SIZE,

            "maxFrameDimension":
                MAX_FRAME_DIMENSION
        },

        "video": {

            "path":
                os.path.normpath(
                    input_path
                ),

            "fps":
                safe_float(fps),

            "frameCount":
                int(frame_count),

            "processedFrames":
                int(inference_count),

            "sampledFrames":
                int(
                    len(selected_indices)
                ),

            "width":
                int(width),

            "height":
                int(height),

            "durationSeconds":
                safe_float(duration)
        },

        "bowlingArm":
            arm,

        "bodyScale": {

            "medianTorsoPixels":
                body_scale
        },

        "releaseFrame": {

            "index":
                int(
                    release_original_frame
                ),

            "timestampSeconds":
                safe_float(
                    release_timestamp
                ),

            "percentThroughClip":
                safe_float(
                    (
                        release_original_frame
                        /
                        max(
                            frame_count - 1,
                            1
                        )
                    )
                    *
                    100
                ),

            "sampleIndex":
                int(
                    release_sample_index
                ),

            "detectionMethod":
                release_method
        },

        "analysisWindow": {

            "startFrame":
                int(
                    window_samples[0][
                        "frameIndex"
                    ]
                ),

            "endFrame":
                int(
                    window_samples[-1][
                        "frameIndex"
                    ]
                ),

            "framesUsed":
                int(
                    len(window_samples)
                )
        },

        "measurements":
            measurements,

        "parameterScores":
            parameter_scores,

        "reliability":
            reliability,

        "technicalScore":
            technical_score,

        "riskIndicators":
            risks,

        "recommendations":
            recommendations,

        "detectionRate":
            detection_rate
    }

    # --------------------------------------------------------
    # OUTPUT PATH
    # --------------------------------------------------------

    if output_path is None:

        base, _ = os.path.splitext(
            input_path
        )

        output_path = (
            base
            +
            "_phase1_result.json"
        )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Technical score : "
        f"{technical_score "
        if technical_score is not None :
        else '--'}
    )

    print(
        f"Release frame   : "
        f"{release_original_frame}"
    )

    print(
        f"Release time    : "
        f"{release_timestamp:.3f}s"
    )

    print(
        f"YOLO inferences : "
        f"{inference_count}"
    )

    print(
        f"Detection rate  : "
        f"{detection_rate * 100:.1f}%"
    )

    print(
        f"Risk indicators : "
        f"{len(risks)}"
    )

    print()
    print("MEASUREMENTS")
    print("-" * 60)

    for name, value in measurements.items():

        print(
            f"{name:<30}: "
            f"{value if value is not None else '--'}"
        )

    print()
    print("RISK INDICATORS")
    print("-" * 60)

    if not risks:

        print(
            "No major biomechanical risk indicators."
        )

    else:

        for risk in risks:

            print(
                f"[{risk['severity'].upper()}] "
                f"{risk['bodyArea']}: "
                f"{risk['message']}"
            )

    print()
    print("RECOMMENDATIONS")
    print("-" * 60)

    for recommendation in recommendations:

        print(
            f"- {recommendation}"
        )

    print()
    print("=" * 60)

    print(
        f"JSON report: "
        f"{output_path}"
    )

    print("=" * 60)

    gc.collect()

    return report


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Cric AI - Render Lightweight "
            "Bowling Biomechanics Phase 1"
        )
    )

    parser.add_argument(
        "video",
        help="Path to bowling video"
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
        "--output",
        default=None,
        help="Output JSON path"
    )

    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="YOLO pose model path"
    )

    args = parser.parse_args()

    try:

        analyze_video(
            input_path=args.video,
            arm=args.arm,
            output_path=args.output,
            model_path=args.model
        )

    except KeyboardInterrupt:

        print()
        print(
            "Analysis cancelled."
        )

    except Exception as error:

        print()
        print("=" * 60)
        print("PHASE 1 ERROR")
        print("=" * 60)

        print(
            str(error)
        )

        print("=" * 60)

        raise


if __name__ == "__main__":
    main()
