"""
phase1_engine.py
============================================================

BOWLING BIOMECHANICS ANALYSIS SYSTEM
PHASE 1 - AI ENGINE

Pipeline:
    Video
      -> YOLO Pose Detection
      -> Bowler Selection
      -> Temporal Bowling-Arm Tracking
      -> Release Detection
      -> Delivery Window
      -> Biomechanical Measurements
      -> Reliability
      -> Technical Scores
      -> Risk Indicators
      -> Recommendations
      -> JSON Report

Usage:
    ..\.venv\Scripts\python.exe phase1_engine.py "test_data\my_bowling.mp4" --arm right

Optional:
    --output "phase1_result.json"
    --model "yolov8n-pose.pt"

IMPORTANT:
This system provides biomechanical risk indicators.
It is NOT a medical diagnosis and does not predict injury with certainty.
"""

import argparse
import json
import math
import os
from statistics import median, mean

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "PHASE-1"

MIN_CONF = 0.25
POSE_CONF = 0.30

# Frames around estimated release used for final measurements.
WINDOW_BEFORE = 5
WINDOW_AFTER = 5

# Temporal tracking.
MAX_WRIST_JUMP = 160.0
MAX_ELBOW_JUMP = 120.0
MAX_MISSING_FRAMES = 8

# Release detection smoothing.
RELEASE_MIN_SPEED = 5.0


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
# BASIC HELPERS
# ============================================================

def safe_float(value):
    """Convert numpy/scalar values to normal Python floats."""
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
    Return (x, y) if a keypoint exists and confidence is sufficient.
    """
    if not points:
        return None

    value = points.get(name)

    if value is None or len(value) < 3:
        return None

    x, y, conf = value

    if conf < min_conf:
        return None

    return np.array([float(x), float(y)], dtype=float)


def distance(a, b):
    if a is None or b is None:
        return None

    return float(np.linalg.norm(
        np.array(a, dtype=float) -
        np.array(b, dtype=float)
    ))


def angle_3pt(a, b, c):
    """
    Angle ABC in degrees.
    """
    if a is None or b is None or c is None:
        return None

    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom <= 1e-8:
        return None

    cosine = np.dot(ba, bc) / denom
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def line_angle_from_vertical(p1, p2):
    """
    Angle of p1 -> p2 relative to vertical.

    0 degrees = vertical.
    Larger values = greater forward/backward inclination.
    """
    if p1 is None or p2 is None:
        return None

    vector = np.array(p2, dtype=float) - np.array(p1, dtype=float)

    length = np.linalg.norm(vector)

    if length <= 1e-8:
        return None

    vertical = np.array([0.0, -1.0])

    cosine = np.dot(vector, vertical) / length
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


# ============================================================
# YOLO POSE DETECTOR
# ============================================================

class PoseDetector:

    def __init__(self, model_path):
        print("Loading YOLO pose model...")

        self.model = YOLO(model_path)

        print("Model loaded.")

    def detect(self, frame):
        """
        Detect the strongest person in the frame.

        Returns:
            {
                "nose": (x, y, confidence),
                ...
            }

        or None.
        """

        results = self.model(
            frame,
            verbose=False,
            conf=POSE_CONF
        )

        if not results:
            return None

        result = results[0]

        if result.keypoints is None:
            return None

        if len(result.keypoints.data) == 0:
            return None

        data = result.keypoints.data.cpu().numpy()

        if data.ndim != 3:
            return None

        # ----------------------------------------------------
        # Choose person with highest average visible confidence
        # ----------------------------------------------------

        candidate_scores = []

        for person in data:

            confidences = person[:, 2]

            visible = confidences >= MIN_CONF

            if np.sum(visible) == 0:
                candidate_scores.append(0.0)
            else:
                candidate_scores.append(
                    float(np.mean(confidences[visible]))
                )

        best_index = int(np.argmax(candidate_scores))

        if candidate_scores[best_index] < POSE_CONF:
            return None

        person = data[best_index]

        points = {}

        for index, name in enumerate(KEYPOINT_NAMES):

            x = float(person[index][0])
            y = float(person[index][1])
            conf = float(person[index][2])

            points[name] = (
                x,
                y,
                conf
            )

        return points


# ============================================================
# TEMPORAL BOWLING ARM TRACKER
# ============================================================

class BowlingArmTracker:
    """
    Stabilizes elbow/wrist positions across frames.

    Particularly useful immediately after release, where
    the bowling arm can move rapidly and YOLO may occasionally
    switch to an incorrect wrist position.
    """

    def __init__(self, arm="right"):

        self.arm = arm

        if arm == "right":
            self.elbow = "right_elbow"
            self.wrist = "right_wrist"
        else:
            self.elbow = "left_elbow"
            self.wrist = "left_wrist"

        self.previous_elbow = None
        self.previous_wrist = None

        self.elbow_velocity = np.zeros(2)
        self.wrist_velocity = np.zeros(2)

        self.elbow_missing = 0
        self.wrist_missing = 0

    def extract(self, points, name):

        if not points:
            return None

        value = points.get(name)

        if value is None:
            return None

        x, y, conf = value

        if conf < MIN_CONF:
            return None

        return np.array(
            [float(x), float(y)],
            dtype=float
        )

    def predict(self, previous, velocity):

        if previous is None:
            return None

        return previous + velocity

    def update_point(
        self,
        raw,
        previous,
        velocity,
        missing,
        max_jump
    ):

        if raw is None:

            missing += 1

            predicted = self.predict(
                previous,
                velocity
            )

            if missing <= MAX_MISSING_FRAMES:
                return predicted, velocity, missing

            return None, velocity, missing

        # First valid point.
        if previous is None:

            return (
                raw,
                np.zeros(2),
                0
            )

        jump = np.linalg.norm(
            raw - previous
        )

        # Reject impossible sudden jumps.
        if jump > max_jump:

            missing += 1

            predicted = self.predict(
                previous,
                velocity
            )

            if missing <= MAX_MISSING_FRAMES:
                return predicted, velocity, missing

            return raw, velocity, 0

        # ----------------------------------------------------
        # Light temporal smoothing
        # ----------------------------------------------------

        alpha = 0.70

        smoothed = (
            alpha * raw +
            (1.0 - alpha) * previous
        )

        new_velocity = (
            0.70 * (smoothed - previous) +
            0.30 * velocity
        )

        return (
            smoothed,
            new_velocity,
            0
        )

    def update(self, points):

        if points is None:
            return None

        result = dict(points)

        # ----------------------------------------------------
        # Elbow
        # ----------------------------------------------------

        raw_elbow = self.extract(
            points,
            self.elbow
        )

        elbow, self.elbow_velocity, self.elbow_missing = (
            self.update_point(
                raw_elbow,
                self.previous_elbow,
                self.elbow_velocity,
                self.elbow_missing,
                MAX_ELBOW_JUMP
            )
        )

        if elbow is not None:

            original_conf = points.get(
                self.elbow,
                (0, 0, 0)
            )[2]

            result[self.elbow] = (
                float(elbow[0]),
                float(elbow[1]),
                max(float(original_conf), 0.35)
            )

            self.previous_elbow = elbow

        # ----------------------------------------------------
        # Wrist
        # ----------------------------------------------------

        raw_wrist = self.extract(
            points,
            self.wrist
        )

        wrist, self.wrist_velocity, self.wrist_missing = (
            self.update_point(
                raw_wrist,
                self.previous_wrist,
                self.wrist_velocity,
                self.wrist_missing,
                MAX_WRIST_JUMP
            )
        )

        if wrist is not None:

            original_conf = points.get(
                self.wrist,
                (0, 0, 0)
            )[2]

            result[self.wrist] = (
                float(wrist[0]),
                float(wrist[1]),
                max(float(original_conf), 0.35)
            )

            self.previous_wrist = wrist

        return result


# ============================================================
# BODY SCALE
# ============================================================
def torso_length(points):
    """
    Calculate torso length using the midpoint between
    the two shoulders and the midpoint between the two hips.
    """

    def point_xy(name):
        if points is None:
            return None

        point = points.get(name)

        if point is None:
            return None

        # Expected format: (x, y, confidence)
        if len(point) < 2:
            return None

        try:
            confidence = float(point[2]) if len(point) >= 3 else 1.0
        except (TypeError, ValueError):
            confidence = 1.0

        if confidence < 0.20:
            return None

        return np.array(
            [float(point[0]), float(point[1])],
            dtype=float
        )

    left_shoulder = point_xy("left_shoulder")
    right_shoulder = point_xy("right_shoulder")
    left_hip = point_xy("left_hip")
    right_hip = point_xy("right_hip")

    if any(
        p is None
        for p in (
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip
        )
    ):
        return None

    shoulder_mid = (
        left_shoulder + right_shoulder
    ) / 2.0

    hip_mid = (
        left_hip + right_hip
    ) / 2.0

    torso = np.linalg.norm(
        shoulder_mid - hip_mid
    )

    if torso <= 0:
        return None

    return float(torso)

# ============================================================
# RELEASE DETECTION
# ============================================================

def calculate_wrist_speeds(history, arm):

    wrist_name = (
        "right_wrist"
        if arm == "right"
        else "left_wrist"
    )

    speeds = []

    for i in range(1, len(history)):

        previous = point_from_dict(
            history[i - 1],
            wrist_name
        )

        current = point_from_dict(
            history[i],
            wrist_name
        )

        if previous is None or current is None:
            speeds.append(None)
            continue

        speeds.append(
            distance(
                previous,
                current
            )
        )

    return speeds


def detect_release(history, arm):

    if len(history) < 5:
        return None

    speeds = calculate_wrist_speeds(
        history,
        arm
    )

    valid = [
        (index + 1, speed)
        for index, speed in enumerate(speeds)
        if speed is not None
    ]

    if not valid:
        return None

    # --------------------------------------------------------
    # Ignore very early frames.
    # A release should normally occur after some run-up.
    # --------------------------------------------------------

    minimum_index = max(
        2,
        int(len(history) * 0.15)
    )

    valid = [
        item
        for item in valid
        if item[0] >= minimum_index
    ]

    if not valid:
        return None

    # --------------------------------------------------------
    # Use the highest wrist speed.
    # --------------------------------------------------------

    release_index, maximum_speed = max(
        valid,
        key=lambda x: x[1]
    )

    if maximum_speed < RELEASE_MIN_SPEED:
        return None

    return release_index

def detect_all_releases(history, arm):
    """
    Detect multiple bowling deliveries in a session video.

    Returns:
        List of release frame indices.
    """

    if len(history) < 10:
        return []

    speeds = calculate_wrist_speeds(
        history,
        arm
    )

    valid = [
        (index + 1, speed)
        for index, speed in enumerate(speeds)
        if speed is not None
    ]

    if not valid:
        return []

    # --------------------------------------------------------
    # Ignore very early frames.
    # --------------------------------------------------------

    minimum_index = max(
        5,
        int(len(history) * 0.05)
    )

    valid = [
        item
        for item in valid
        if item[0] >= minimum_index
    ]

    if not valid:
        return []

    # --------------------------------------------------------
    # Parameters for multiple-delivery detection.
    # --------------------------------------------------------

    MIN_SPEED = RELEASE_MIN_SPEED

    # Minimum distance between two deliveries.
    # At 30 FPS, 45 frames = 1.5 seconds.
    MIN_DELIVERY_GAP = 45

    # A peak must be locally maximal.
    PEAK_RADIUS = 3

    candidates = []

    # --------------------------------------------------------
    # Find local wrist-speed peaks.
    # --------------------------------------------------------

    speed_map = {
        index: speed
        for index, speed in valid
    }

    indices = sorted(speed_map.keys())

    for index in indices:

        speed = speed_map[index]

        if speed < MIN_SPEED:
            continue

        is_peak = True

        for offset in range(
            -PEAK_RADIUS,
            PEAK_RADIUS + 1
        ):

            if offset == 0:
                continue

            neighbour = speed_map.get(
                index + offset
            )

            if neighbour is None:
                continue

            if neighbour > speed:
                is_peak = False
                break

        if is_peak:
            candidates.append(
                (index, speed)
            )

    if not candidates:
        return []

    # --------------------------------------------------------
    # Sort strongest peaks first.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    releases = []

    # --------------------------------------------------------
    # Keep strong peaks separated in time.
    # --------------------------------------------------------

    for index, speed in candidates:

        too_close = False

        for existing in releases:

            if abs(index - existing) < MIN_DELIVERY_GAP:
                too_close = True
                break

        if not too_close:
            releases.append(index)

    # --------------------------------------------------------
    # Return chronological order.
    # --------------------------------------------------------

    releases.sort()

    return releases
# ============================================================
# BIOMECHANICAL MEASUREMENTS
# ============================================================

def calculate_measurements(points, arm):

    if points is None:
        return {}

    def p(name):
        return point_from_dict(points, name)

    nose = p("nose")

    left_shoulder = p("left_shoulder")
    right_shoulder = p("right_shoulder")

    left_elbow = p("left_elbow")
    right_elbow = p("right_elbow")

    left_wrist = p("left_wrist")
    right_wrist = p("right_wrist")

    left_hip = p("left_hip")
    right_hip = p("right_hip")

    left_knee = p("left_knee")
    right_knee = p("right_knee")

    left_ankle = p("left_ankle")
    right_ankle = p("right_ankle")

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
    # Midpoints
    # --------------------------------------------------------

    mid_shoulder = None
    mid_hip = None

    if (
        left_shoulder is not None
        and right_shoulder is not None
    ):

        mid_shoulder = (
            left_shoulder +
            right_shoulder
        ) / 2.0

    if (
        left_hip is not None
        and right_hip is not None
    ):

        mid_hip = (
            left_hip +
            right_hip
        ) / 2.0

    # --------------------------------------------------------
    # Elbow angle
    # --------------------------------------------------------

    measurements["elbowAngle"] = angle_3pt(
        bowling_shoulder,
        bowling_elbow,
        bowling_wrist
    )

    # --------------------------------------------------------
    # Front knee
    # --------------------------------------------------------

    measurements["frontKneeAngle"] = angle_3pt(
        front_hip,
        front_knee,
        front_ankle
    )

    # --------------------------------------------------------
    # Back knee
    # --------------------------------------------------------

    measurements["backKneeAngle"] = angle_3pt(
        back_hip,
        back_knee,
        back_ankle
    )

    # --------------------------------------------------------
    # Trunk forward flexion
    # --------------------------------------------------------

    measurements["trunkForwardFlexion"] = (
        line_angle_from_vertical(
            mid_hip,
            mid_shoulder
        )
    )

    # --------------------------------------------------------
    # Trunk lateral flexion
    # --------------------------------------------------------

    if (
        mid_shoulder is not None
        and mid_hip is not None
    ):

        torso = distance(
            mid_shoulder,
            mid_hip
        )

        if torso and torso > 0:

            horizontal_offset = abs(
                mid_shoulder[0] -
                mid_hip[0]
            )

            measurements["trunkLateralFlexion"] = (
                float(
                    np.degrees(
                        np.arctan2(
                            horizontal_offset,
                            torso
                        )
                    )
                )
            )

    # --------------------------------------------------------
    # Shoulder line
    # --------------------------------------------------------

    shoulder_angle = None

    if (
        left_shoulder is not None
        and right_shoulder is not None
    ):

        vector = (
            right_shoulder -
            left_shoulder
        )

        shoulder_angle = float(
            np.degrees(
                np.arctan2(
                    vector[1],
                    vector[0]
                )
            )
        )

    measurements["shoulderLineAngle"] = shoulder_angle

    # --------------------------------------------------------
    # Hip line
    # --------------------------------------------------------

    hip_angle = None

    if (
        left_hip is not None
        and right_hip is not None
    ):

        vector = (
            right_hip -
            left_hip
        )

        hip_angle = float(
            np.degrees(
                np.arctan2(
                    vector[1],
                    vector[0]
                )
            )
        )

    measurements["hipLineAngle"] = hip_angle

    # --------------------------------------------------------
    # Hip / shoulder separation
    # --------------------------------------------------------

    if (
        shoulder_angle is not None
        and hip_angle is not None
    ):

        difference = (
            shoulder_angle -
            hip_angle
        )

        difference = (
            difference + 180
        ) % 360 - 180

        measurements["hipShoulderSeparation"] = abs(
            difference
        )

    # --------------------------------------------------------
    # Front foot offset
    # --------------------------------------------------------

    if (
        front_hip is not None
        and front_ankle is not None
    ):

        leg_length = distance(
            front_hip,
            front_ankle
        )

        if leg_length and leg_length > 0:

            measurements["frontFootOffset"] = (
                abs(
                    front_hip[0] -
                    front_ankle[0]
                )
                / leg_length
            )

    # --------------------------------------------------------
    # Head offset
    #
    # Normalized distance of head from shoulder midpoint.
    # --------------------------------------------------------

    if (
        nose is not None
        and mid_shoulder is not None
    ):

        torso = torso_length(points)

        if torso and torso > 0:

            measurements["headOffset"] = (
                distance(
                    nose,
                    mid_shoulder
                ) / torso
            )

    return {
        key: safe_float(value)
        for key, value in measurements.items()
    }


# ============================================================
# WINDOW AGGREGATION
# ============================================================

def aggregate_measurements(window):

    if not window:
        return {}

    keys = set()

    for item in window:
        keys.update(item.keys())

    result = {}

    for key in keys:

        values = []

        for item in window:

            value = item.get(key)

            if value is None:
                continue

            try:
                if math.isfinite(float(value)):
                    values.append(float(value))
            except Exception:
                pass

        if not values:
            result[key] = None
            continue

        # Median is more robust than a single-frame value.
        result[key] = safe_float(
            median(values)
        )

    return result


# ============================================================
# RELIABILITY
# ============================================================

def calculate_reliability(
    frame_points,
    window_start,
    window_end,
    measurements
):

    relevant = frame_points[
        max(0, window_start):
        min(len(frame_points), window_end + 1)
    ]

    if not relevant:
        return {
            key: 0.0
            for key in measurements
        }

    reliability = {}

    measurement_dependencies = {

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

    for measurement, dependencies in measurement_dependencies.items():

        if measurement not in measurements:
            continue

        valid_frames = 0

        for points in relevant:

            if points is None:
                continue

            good = True

            for name in dependencies:

                value = points.get(name)

                if (
                    value is None
                    or len(value) < 3
                    or value[2] < MIN_CONF
                ):

                    good = False
                    break

            if good:
                valid_frames += 1

        ratio = (
            valid_frames /
            max(len(relevant), 1)
        )

        reliability[measurement] = safe_float(
            min(1.0, ratio)
        )

    return reliability


# ============================================================
# TECHNICAL SCORING
# ============================================================

REFERENCE_RANGES = {

    # These are prototype coaching reference ranges.
    # They must eventually be validated against cricket
    # biomechanics literature and coach/athlete data.

    "frontKneeAngle": (145.0, 180.0),

    "elbowAngle": (150.0, 180.0),

    "trunkLateralFlexion": (0.0, 15.0),

    "trunkForwardFlexion": (0.0, 40.0),

    "hipShoulderSeparation": (15.0, 45.0),

    "frontFootOffset": (0.0, 0.55),
}


WEIGHTS = {

    "frontKneeAngle": 0.20,

    "elbowAngle": 0.15,

    "trunkLateralFlexion": 0.15,

    "trunkForwardFlexion": 0.10,

    "hipShoulderSeparation": 0.20,

    "frontFootOffset": 0.20,
}


def parameter_score(value, low, high):

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
        distance_from_range = low - value
    else:
        distance_from_range = value - high

    score = (
        100.0 -
        (distance_from_range / span) * 100.0
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0
        )
    )


def calculate_scores(measurements, reliability):

    parameter_scores = {}

    weighted_total = 0.0
    weight_total = 0.0

    for name, reference in REFERENCE_RANGES.items():

        value = measurements.get(name)

        score = parameter_score(
            value,
            reference[0],
            reference[1]
        )

        rel = reliability.get(
            name,
            0.0
        )

        parameter_scores[name] = {

            "value": safe_float(value),

            "score": safe_float(score),

            "reliability": safe_float(rel)
        }

        if (
            score is not None
            and rel is not None
            and rel > 0
        ):

            # Reliability influences contribution.
            weighted_total += (
                score *
                WEIGHTS[name] *
                rel
            )

            weight_total += (
                WEIGHTS[name] *
                rel
            )

    if weight_total > 0:

        technical_score = (
            weighted_total /
            weight_total
        )

    else:

        technical_score = None

    return (
        parameter_scores,
        safe_float(technical_score)
    )


# ============================================================
# RISK ENGINE
# ============================================================

def build_risks(measurements):

    risks = []

    # --------------------------------------------------------
    # Lower back
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
    # Bowling elbow
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
    # Front knee
    # --------------------------------------------------------

    knee = measurements.get(
        "frontKneeAngle"
    )

    if knee is not None and knee < 135:

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
    # Front foot / ankle
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
    measurements,
    risks
):

    recommendations = []

    risk_parameters = {
        item["parameter"]
        for item in risks
    }

    if "trunkLateralFlexion" in risk_parameters:

        recommendations.append(
            "Work on maintaining trunk stability through the delivery and follow-through."
        )

    if "elbowAngle" in risk_parameters:

        recommendations.append(
            "Review bowling-arm movement and elbow position with a qualified coach."
        )

    if "frontKneeAngle" in risk_parameters:

        recommendations.append(
            "Work on controlled front-leg mechanics and stable front-knee positioning."
        )

    if "frontFootOffset" in risk_parameters:

        recommendations.append(
            "Focus on consistent front-foot placement and lower-limb alignment."
        )

    # --------------------------------------------------------
    # Positive recommendations when no major issue exists.
    # --------------------------------------------------------

    if not recommendations:

        recommendations.append(
            "No major biomechanical flags were identified in this delivery window."
        )

        recommendations.append(
            "Continue monitoring consistency across multiple deliveries."
        )

    return recommendations


# ============================================================
# DETECTION QUALITY
# ============================================================

def calculate_detection_rate(history):

    if not history:
        return 0.0

    valid = sum(
        1
        for item in history
        if item is not None
    )

    return safe_float(
        valid /
        len(history)
    )


def calculate_body_scale(history):

    scales = []

    for points in history:

        scale = torso_length(points)

        if scale is not None and scale > 0:
            scales.append(scale)

    if not scales:
        return None

    return safe_float(
        median(scales)
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

    print("=" * 60)
    print("BOWLING BIOMECHANICS ANALYSIS SYSTEM")
    print("PHASE 1 - AI ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            f"Video not found: {input_path}"
        )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"YOLO model not found: {model_path}"
        )

    # --------------------------------------------------------
    # Load detector
    # --------------------------------------------------------

    detector = PoseDetector(
        model_path
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

    print()
    print("VIDEO")
    print("-" * 60)
    print(f"Path       : {input_path}")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {frame_count}")
    print(f"Resolution : {width} x {height}")
    print()

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    history = []
    measurements_history = []

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
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

        measurements_history.append(
            calculate_measurements(
                points,
                arm
            )
        )

        frame_index += 1

        if frame_index % 30 == 0:

            percentage = (
                frame_index /
                max(frame_count, 1)
            ) * 100

            print(
                f"Processing: {percentage:.1f}%"
            )

    cap.release()

    # --------------------------------------------------------
    # Detection rate
    # --------------------------------------------------------

    detection_rate = calculate_detection_rate(
        history
    )

    print()
    print(
        f"Detection rate: "
        f"{detection_rate * 100:.1f}%"
    )

    # --------------------------------------------------------
    # Release detection
    # --------------------------------------------------------

    release_frame = detect_release(
        history,
        arm
    )

    if release_frame is None:

        # Fallback:
        # use approximately 70% of clip if wrist speed
        # could not identify a reliable peak.
        release_frame = int(
            len(history) * 0.70
        )

        release_detection_method = (
            "fallback"
        )

    else:

        release_detection_method = (
            "wrist_speed_peak"
        )

    release_frame = int(
        np.clip(
            release_frame,
            0,
            max(len(history) - 1, 0)
        )
    )

    # --------------------------------------------------------
    # Delivery window
    # --------------------------------------------------------

    start_frame = max(
        0,
        release_frame -
        WINDOW_BEFORE
    )

    end_frame = min(
        len(history) - 1,
        release_frame +
        WINDOW_AFTER
    )

    window_measurements = (
        measurements_history[
            start_frame:
            end_frame + 1
        ]
    )

    measurements = aggregate_measurements(
        window_measurements
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    reliability = calculate_reliability(
        history,
        start_frame,
        end_frame,
        measurements
    )

    # --------------------------------------------------------
    # Technical scores
    # --------------------------------------------------------

    parameter_scores, technical_score = (
        calculate_scores(
            measurements,
            reliability
        )
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
    # Body scale
    # --------------------------------------------------------

    body_scale = calculate_body_scale(
        history
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {

        "status": "ok",

        "version": VERSION,

        "video": {

            "path":
                os.path.normpath(
                    input_path
                ),

            "fps":
                safe_float(fps),

            "frameCount":
                int(frame_count),

            "width":
                int(width),

            "height":
                int(height)
        },

        "bowlingArm":
            arm,

        "bodyScale": {

            "medianTorsoPixels":
                body_scale
        },

        "releaseFrame": {

            "index":
                int(release_frame),

            "timestampSeconds":
                safe_float(
                    release_frame / fps
                ),

            "percentThroughClip":
                safe_float(
                    (
                        release_frame /
                        max(frame_count - 1, 1)
                    ) * 100
                ),

            "detectionMethod":
                release_detection_method
        },

        "analysisWindow": {

            "startFrame":
                int(start_frame),

            "endFrame":
                int(end_frame),

            "framesUsed":
                int(
                    end_frame -
                    start_frame +
                    1
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
    # Output
    # --------------------------------------------------------

    if output_path is None:

        base, _ = os.path.splitext(
            input_path
        )

        output_path = (
            base +
            "_phase1_result.json"
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
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Technical score : "
        f"{technical_score if technical_score is not None else '--'}"
    )

    print(
        f"Release frame   : "
        f"{release_frame}"
    )

    print(
        f"Release time    : "
        f"{release_frame / fps:.3f}s"
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

        print("No major biomechanical risk indicators.")

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
        f"JSON report: {output_path}"
    )

    print("=" * 60)

    return report


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Bowling Biomechanics "
            "Analysis System - Phase 1"
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
        help="YOLO pose model"
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
        print("Analysis cancelled.")

    except Exception as error:

        print()
        print("=" * 60)
        print("PHASE 1 ERROR")
        print("=" * 60)
        print(str(error))
        print("=" * 60)

        raise


if __name__ == "__main__":
    main()