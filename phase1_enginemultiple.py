
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
      -> Multi-Delivery Release Detection
      -> Delivery Windows
      -> Biomechanical Measurements
      -> Reliability
      -> Technical Scores
      -> Risk Indicators
      -> Recommendations
      -> JSON Report

Usage:

    ..\.venv\Scripts\python.exe phase1_engine.py ^
        "test_data\side_bowler1.mp4" ^
        --arm right ^
        --camera side

Optional:

    --output "phase1_result.json"
    --model "yolov8n-pose.pt"

Camera modes:

    side
    front
    oblique

IMPORTANT:

This system provides biomechanical movement indicators.
It is NOT a medical diagnosis and does not predict injury
with certainty.

2D camera measurements are highly dependent on camera
position and perspective. Measurements that cannot be
reliably interpreted from the selected camera view are
excluded from technical scoring.
"""

import argparse
import json
import math
import os
from statistics import median

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "PHASE-1-CORRECTED"

MIN_CONF = 0.25
POSE_CONF = 0.30

WINDOW_BEFORE = 5
WINDOW_AFTER = 5

MAX_WRIST_JUMP = 160.0
MAX_ELBOW_JUMP = 120.0
MAX_MISSING_FRAMES = 8

RELEASE_MIN_SPEED = 5.0

MIN_DELIVERY_GAP = 45
PEAK_RADIUS = 3

MIN_RELEASE_CONFIDENCE = 0.45


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
    """Convert numeric values to normal Python floats."""

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
    Return (x, y) if a keypoint exists and confidence
    is sufficient.
    """

    if not points:
        return None

    value = points.get(name)

    if value is None or len(value) < 3:
        return None

    try:
        x, y, conf = value

        if float(conf) < min_conf:
            return None

        return np.array(
            [float(x), float(y)],
            dtype=float
        )

    except Exception:
        return None


def distance(a, b):

    if a is None or b is None:
        return None

    return float(
        np.linalg.norm(
            np.array(a, dtype=float)
            -
            np.array(b, dtype=float)
        )
    )


def angle_3pt(a, b, c):
    """
    Calculate angle ABC in degrees.
    """

    if a is None or b is None or c is None:
        return None

    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)

    denom = (
        np.linalg.norm(ba)
        *
        np.linalg.norm(bc)
    )

    if denom <= 1e-8:
        return None

    cosine = np.dot(ba, bc) / denom
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def midpoint(a, b):

    if a is None or b is None:
        return None

    return (
        np.array(a, dtype=float)
        +
        np.array(b, dtype=float)
    ) / 2.0


def line_angle_from_vertical(p1, p2):
    """
    Angle of p1 -> p2 relative to vertical.

    0 degrees = vertical.
    Larger values = greater inclination.
    """

    if p1 is None or p2 is None:
        return None

    vector = (
        np.array(p2, dtype=float)
        -
        np.array(p1, dtype=float)
    )

    length = np.linalg.norm(vector)

    if length <= 1e-8:
        return None

    vertical = np.array(
        [0.0, -1.0]
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
# YOLO POSE DETECTOR
# ============================================================

class PoseDetector:

    def __init__(self, model_path):

        print("Loading YOLO pose model...")

        self.model = YOLO(
            model_path
        )

        print("Model loaded.")

    def detect(self, frame):
        """
        Detect the strongest visible person.

        Returns:
            Dictionary of COCO pose keypoints.
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

        data = (
            result
            .keypoints
            .data
            .cpu()
            .numpy()
        )

        if data.ndim != 3:
            return None

        candidate_scores = []

        for person in data:

            confidences = person[:, 2]

            visible = (
                confidences >= MIN_CONF
            )

            if np.sum(visible) == 0:

                candidate_scores.append(
                    0.0
                )

            else:

                candidate_scores.append(
                    float(
                        np.mean(
                            confidences[visible]
                        )
                    )
                )

        best_index = int(
            np.argmax(
                candidate_scores
            )
        )

        if (
            candidate_scores[best_index]
            <
            POSE_CONF
        ):
            return None

        person = data[best_index]

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

            conf = float(
                person[index][2]
            )

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

        self.predicted_frames = 0
        self.tracked_frames = 0

    def extract(self, points, name):

        if not points:
            return None

        value = points.get(name)

        if value is None:
            return None

        try:

            x, y, conf = value

            if conf < MIN_CONF:
                return None

            return np.array(
                [float(x), float(y)],
                dtype=float
            )

        except Exception:

            return None

    def predict(
        self,
        previous,
        velocity
    ):

        if previous is None:
            return None

        return (
            previous +
            velocity
        )

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

            if (
                predicted is not None
                and
                missing <= MAX_MISSING_FRAMES
            ):

                return (
                    predicted,
                    velocity,
                    missing,
                    True
                )

            return (
                None,
                velocity,
                missing,
                False
            )

        if previous is None:

            return (
                raw,
                np.zeros(2),
                0,
                False
            )

        jump = np.linalg.norm(
            raw - previous
        )

        if jump > max_jump:

            missing += 1

            predicted = self.predict(
                previous,
                velocity
            )

            if (
                predicted is not None
                and
                missing <= MAX_MISSING_FRAMES
            ):

                return (
                    predicted,
                    velocity,
                    missing,
                    True
                )

            return (
                raw,
                velocity,
                0,
                False
            )

        alpha = 0.70

        smoothed = (
            alpha * raw
            +
            (1.0 - alpha) * previous
        )

        new_velocity = (
            0.70 *
            (smoothed - previous)
            +
            0.30 *
            velocity
        )

        return (
            smoothed,
            new_velocity,
            0,
            False
        )

    def update(self, points):

        if points is None:
            return None

        result = dict(points)

        predicted_this_frame = False

        # ----------------------------------------------------
        # Elbow
        # ----------------------------------------------------

        raw_elbow = self.extract(
            points,
            self.elbow
        )

        (
            elbow,
            self.elbow_velocity,
            self.elbow_missing,
            elbow_predicted
        ) = self.update_point(
            raw_elbow,
            self.previous_elbow,
            self.elbow_velocity,
            self.elbow_missing,
            MAX_ELBOW_JUMP
        )

        if elbow is not None:

            original_conf = float(
                points.get(
                    self.elbow,
                    (0, 0, 0)
                )[2]
            )

            result[self.elbow] = (
                float(elbow[0]),
                float(elbow[1]),
                original_conf
            )

            self.previous_elbow = elbow

            if elbow_predicted:
                predicted_this_frame = True

        # ----------------------------------------------------
        # Wrist
        # ----------------------------------------------------

        raw_wrist = self.extract(
            points,
            self.wrist
        )

        (
            wrist,
            self.wrist_velocity,
            self.wrist_missing,
            wrist_predicted
        ) = self.update_point(
            raw_wrist,
            self.previous_wrist,
            self.wrist_velocity,
            self.wrist_missing,
            MAX_WRIST_JUMP
        )

        if wrist is not None:

            original_conf = float(
                points.get(
                    self.wrist,
                    (0, 0, 0)
                )[2]
            )

            result[self.wrist] = (
                float(wrist[0]),
                float(wrist[1]),
                original_conf
            )

            self.previous_wrist = wrist

            if wrist_predicted:
                predicted_this_frame = True

        if predicted_this_frame:
            self.predicted_frames += 1

        self.tracked_frames += 1

        return result


# ============================================================
# BODY SCALE
# ============================================================

def torso_length(points):

    if points is None:
        return None

    def point_xy(name):

        value = points.get(name)

        if value is None:
            return None

        if len(value) < 2:
            return None

        try:

            confidence = (
                float(value[2])
                if len(value) >= 3
                else 1.0
            )

            if confidence < 0.20:
                return None

            return np.array(
                [
                    float(value[0]),
                    float(value[1])
                ],
                dtype=float
            )

        except Exception:

            return None

    left_shoulder = point_xy(
        "left_shoulder"
    )

    right_shoulder = point_xy(
        "right_shoulder"
    )

    left_hip = point_xy(
        "left_hip"
    )

    right_hip = point_xy(
        "right_hip"
    )

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

    shoulder_mid = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_mid = midpoint(
        left_hip,
        right_hip
    )

    if (
        shoulder_mid is None
        or
        hip_mid is None
    ):
        return None

    torso = distance(
        shoulder_mid,
        hip_mid
    )

    if torso is None or torso <= 0:
        return None

    return float(torso)


# ============================================================
# RELEASE SPEED
# ============================================================

def calculate_wrist_speeds(
    history,
    arm
):

    wrist_name = (
        "right_wrist"
        if arm == "right"
        else "left_wrist"
    )

    speeds = []

    for i in range(
        1,
        len(history)
    ):

        previous = point_from_dict(
            history[i - 1],
            wrist_name
        )

        current = point_from_dict(
            history[i],
            wrist_name
        )

        if (
            previous is None
            or
            current is None
        ):

            speeds.append(None)

            continue

        speeds.append(
            distance(
                previous,
                current
            )
        )

    return speeds


# ============================================================
# RELEASE CONFIDENCE
# ============================================================

def calculate_release_confidence(
    speed_map,
    release_index
):

    if (
        not speed_map
        or
        release_index not in speed_map
    ):
        return 0.0

    peak = float(
        speed_map[release_index]
    )

    nearby = []

    for offset in range(
        -PEAK_RADIUS,
        PEAK_RADIUS + 1
    ):

        index = (
            release_index +
            offset
        )

        if index in speed_map:
            nearby.append(
                speed_map[index]
            )

    if len(nearby) < 3:
        return 0.35

    baseline = float(
        median(nearby)
    )

    if baseline <= 0:
        baseline = 1e-6

    prominence = (
        peak /
        baseline
    )

    # Peak strength component.
    strength_score = np.clip(
        peak / 20.0,
        0.0,
        1.0
    )

    # Peak prominence component.
    prominence_score = np.clip(
        (prominence - 1.0) / 2.0,
        0.0,
        1.0
    )

    confidence = (
        0.55 * strength_score
        +
        0.45 * prominence_score
    )

    return safe_float(
        np.clip(
            confidence,
            0.0,
            1.0
        )
    )


# ============================================================
# SINGLE RELEASE DETECTION
# ============================================================

def detect_release(
    history,
    arm
):

    releases = detect_all_releases(
        history,
        arm
    )

    if not releases:
        return None

    speeds = calculate_wrist_speeds(
        history,
        arm
    )

    speed_map = {
        index + 1: speed
        for index, speed in enumerate(speeds)
        if speed is not None
    }

    best_release = None
    best_speed = -1.0

    for index in releases:

        speed = speed_map.get(
            index
        )

        if speed is None:
            continue

        if speed > best_speed:

            best_speed = speed
            best_release = index

    if best_release is None:
        return None

    confidence = calculate_release_confidence(
        speed_map,
        best_release
    )

    return {
        "frame": int(best_release),
        "confidence": confidence
    }


# ============================================================
# MULTI-DELIVERY RELEASE DETECTION
# ============================================================

def detect_all_releases(
    history,
    arm
):
    """
    Detect multiple bowling deliveries.

    IMPORTANT:
    This function intentionally uses wrist-speed peaks.

    It should NOT be interpreted as absolute ball-release
    ground truth. It is a prototype release estimator.
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

    speed_map = {
        index: speed
        for index, speed in valid
    }

    indices = sorted(
        speed_map.keys()
    )

    candidates = []

    for index in indices:

        speed = speed_map[index]

        if speed < RELEASE_MIN_SPEED:
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

    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    releases = []

    for index, speed in candidates:

        too_close = False

        for existing in releases:

            if (
                abs(index - existing)
                <
                MIN_DELIVERY_GAP
            ):

                too_close = True

                break

        if not too_close:

            releases.append(
                index
            )

    releases.sort()

    return releases


# ============================================================
# MEASUREMENT DEPENDENCIES
# ============================================================

def get_measurement_dependencies(
    arm
):

    bowling_shoulder = (
        "right_shoulder"
        if arm == "right"
        else "left_shoulder"
    )

    bowling_elbow = (
        "right_elbow"
        if arm == "right"
        else "left_elbow"
    )

    bowling_wrist = (
        "right_wrist"
        if arm == "right"
        else "left_wrist"
    )

    front_hip = (
        "left_hip"
        if arm == "right"
        else "right_hip"
    )

    front_knee = (
        "left_knee"
        if arm == "right"
        else "right_knee"
    )

    front_ankle = (
        "left_ankle"
        if arm == "right"
        else "right_ankle"
    )

    back_hip = (
        "right_hip"
        if arm == "right"
        else "left_hip"
    )

    back_knee = (
        "right_knee"
        if arm == "right"
        else "left_knee"
    )

    back_ankle = (
        "right_ankle"
        if arm == "right"
        else "left_ankle"
    )

    return {

        "elbowAngle": [
            bowling_shoulder,
            bowling_elbow,
            bowling_wrist
        ],

        "frontKneeAngle": [
            front_hip,
            front_knee,
            front_ankle
        ],

        "backKneeAngle": [
            back_hip,
            back_knee,
            back_ankle
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

        "frontLegAlignmentOffset": [
            front_hip,
            front_ankle
        ],

        "headHorizontalOffset": [
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


# ============================================================
# BIOMECHANICAL MEASUREMENTS
# ============================================================

def calculate_measurements(
    points,
    arm,
    camera="side"
):

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

    mid_shoulder = midpoint(
        left_shoulder,
        right_shoulder
    )

    mid_hip = midpoint(
        left_hip,
        right_hip
    )

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
    # Torso forward flexion
    #
    # IMPORTANT:
    # This is an image-plane measurement.
    # It is most useful with side/oblique views.
    # --------------------------------------------------------

    measurements["trunkForwardFlexion"] = (
        line_angle_from_vertical(
            mid_hip,
            mid_shoulder
        )
    )

    # --------------------------------------------------------
    # Trunk lateral flexion
    #
    # Only meaningful from a sufficiently frontal/oblique
    # camera. A pure side view cannot reliably distinguish
    # anatomical lateral flexion from camera-plane movement.
    # --------------------------------------------------------

    if camera in (
        "front",
        "oblique"
    ):

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
    # Shoulder line
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
    # Hip line
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
    # Hip / shoulder separation
    #
    # Do not use this as a true 3D separation angle.
    # It is only a 2D image-plane approximation.
    #
    # For side camera, it is excluded because both hip and
    # shoulder rotations are heavily affected by projection.
    # --------------------------------------------------------

    if (
        camera in (
            "front",
            "oblique"
        )
        and
        "shoulderLineAngle" in measurements
        and
        "hipLineAngle" in measurements
    ):

        difference = (
            measurements[
                "shoulderLineAngle"
            ]
            -
            measurements[
                "hipLineAngle"
            ]
        )

        difference = (
            difference + 180
        ) % 360 - 180

        measurements[
            "hipShoulderSeparation"
        ] = abs(difference)

    # --------------------------------------------------------
    # Front-leg image-plane offset
    #
    # This is NOT a medical alignment measure.
    # It is normalized horizontal displacement of front hip
    # relative to front ankle.
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

        if (
            leg_length
            and
            leg_length > 0
        ):

            measurements[
                "frontLegAlignmentOffset"
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
    # Head horizontal offset
    # --------------------------------------------------------

    if (
        nose is not None
        and
        mid_shoulder is not None
    ):

        torso = torso_length(
            points
        )

        if (
            torso
            and
            torso > 0
        ):

            measurements[
                "headHorizontalOffset"
            ] = (
                abs(
                    nose[0]
                    -
                    mid_shoulder[0]
                )
                /
                torso
            )

    return {
        key: safe_float(value)
        for key, value in measurements.items()
    }


# ============================================================
# WINDOW AGGREGATION
# ============================================================

def aggregate_measurements(
    window
):

    if not window:
        return {}

    keys = set()

    for item in window:
        keys.update(
            item.keys()
        )

    result = {}

    for key in keys:

        values = []

        for item in window:

            value = item.get(
                key
            )

            if value is None:
                continue

            try:

                if math.isfinite(
                    float(value)
                ):

                    values.append(
                        float(value)
                    )

            except Exception:
                pass

        if not values:

            result[key] = None

            continue

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
    measurements,
    arm
):

    relevant = frame_points[
        max(0, window_start):
        min(
            len(frame_points),
            window_end + 1
        )
    ]

    if not relevant:

        return {
            key: 0.0
            for key in measurements
        }

    dependencies = (
        get_measurement_dependencies(
            arm
        )
    )

    reliability = {}

    for measurement in measurements:

        measurement_dependencies = (
            dependencies.get(
                measurement
            )
        )

        if not measurement_dependencies:

            reliability[
                measurement
            ] = 0.0

            continue

        valid_frames = 0

        for points in relevant:

            if points is None:
                continue

            good = True

            for name in (
                measurement_dependencies
            ):

                value = points.get(
                    name
                )

                if (
                    value is None
                    or
                    len(value) < 3
                    or
                    float(value[2])
                    < MIN_CONF
                ):

                    good = False

                    break

            if good:
                valid_frames += 1

        ratio = (
            valid_frames
            /
            max(
                len(relevant),
                1
            )
        )

        reliability[
            measurement
        ] = safe_float(
            np.clip(
                ratio,
                0.0,
                1.0
            )
        )

    return reliability


# ============================================================
# CAMERA-SPECIFIC VALID MEASUREMENTS
# ============================================================

def get_scoreable_parameters(
    camera
):

    # --------------------------------------------------------
    # Side camera
    #
    # Strongest measurements:
    #   elbow angle
    #   front knee
    #   back knee
    #   trunk forward flexion
    #   front leg image-plane offset
    # --------------------------------------------------------

    if camera == "side":

        return [
            "elbowAngle",
            "frontKneeAngle",
            "backKneeAngle",
            "trunkForwardFlexion",
            "frontLegAlignmentOffset"
        ]

    # --------------------------------------------------------
    # Front camera
    # --------------------------------------------------------

    if camera == "front":

        return [
            "frontKneeAngle",
            "backKneeAngle",
            "trunkLateralFlexion",
            "hipShoulderSeparation"
        ]

    # --------------------------------------------------------
    # Oblique camera
    # --------------------------------------------------------

    return [
        "elbowAngle",
        "frontKneeAngle",
        "backKneeAngle",
        "trunkForwardFlexion",
        "trunkLateralFlexion",
        "hipShoulderSeparation",
        "frontLegAlignmentOffset"
    ]


# ============================================================
# TECHNICAL SCORING
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

    "backKneeAngle": (
        140.0,
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

    "frontLegAlignmentOffset": (
        0.0,
        0.55
    )
}


WEIGHTS = {

    "frontKneeAngle": 0.20,

    "elbowAngle": 0.15,

    "backKneeAngle": 0.10,

    "trunkLateralFlexion": 0.15,

    "trunkForwardFlexion": 0.15,

    "hipShoulderSeparation": 0.15,

    "frontLegAlignmentOffset": 0.10
}


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

        distance_from_range = (
            low - value
        )

    else:

        distance_from_range = (
            value - high
        )

    score = (
        100.0
        -
        (
            distance_from_range
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
    reliability,
    camera
):

    parameter_scores = {}

    weighted_total = 0.0
    weight_total = 0.0

    scoreable = get_scoreable_parameters(
        camera
    )

    for name in scoreable:

        reference = REFERENCE_RANGES.get(
            name
        )

        if reference is None:
            continue

        value = measurements.get(
            name
        )

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

            "value":
                safe_float(value),

            "score":
                safe_float(score),

            "reliability":
                safe_float(rel)
        }

        if (
            score is not None
            and
            rel is not None
            and
            rel > 0
        ):

            weight = WEIGHTS.get(
                name,
                0.0
            )

            weighted_total += (
                score
                *
                weight
                *
                rel
            )

            weight_total += (
                weight
                *
                rel
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
    measurements,
    camera
):

    risks = []

    # --------------------------------------------------------
    # Trunk lateral flexion
    # --------------------------------------------------------

    lateral = measurements.get(
        "trunkLateralFlexion"
    )

    if (
        camera in (
            "front",
            "oblique"
        )
        and
        lateral is not None
    ):

        if lateral >= 30:

            risks.append({

                "parameter":
                    "trunkLateralFlexion",

                "value":
                    safe_float(lateral),

                "bodyArea":
                    "Lower back / trunk",

                "severity":
                    "elevated",

                "message":
                    "Large 2D lateral trunk inclination detected. Review trunk control and camera alignment with a qualified coach."
            })

        elif lateral >= 20:

            risks.append({

                "parameter":
                    "trunkLateralFlexion",

                "value":
                    safe_float(lateral),

                "bodyArea":
                    "Lower back / trunk",

                "severity":
                    "monitor",

                "message":
                    "Moderate 2D lateral trunk inclination detected. Monitor trunk control."
            })

    # --------------------------------------------------------
    # Elbow
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
                    "Bowling arm",

                "severity":
                    "monitor",

                "message":
                    "Significant bowling-arm elbow flexion was detected in the analysis window. Review the movement with a qualified cricket coach."
            })

        elif elbow < 145:

            risks.append({

                "parameter":
                    "elbowAngle",

                "value":
                    safe_float(elbow),

                "bodyArea":
                    "Bowling arm",

                "severity":
                    "monitor",

                "message":
                    "Increased bowling-arm elbow flexion was detected. Review the movement with a qualified coach."
            })

    # --------------------------------------------------------
    # Front knee
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
                "Front leg",

            "severity":
                "monitor",

            "message":
                "Greater front-knee flexion was detected during the delivery window. Review front-leg mechanics."
        })

    # --------------------------------------------------------
    # Front leg image-plane offset
    # --------------------------------------------------------

    foot = measurements.get(
        "frontLegAlignmentOffset"
    )

    if foot is not None:

        if foot >= 0.65:

            risks.append({

                "parameter":
                    "frontLegAlignmentOffset",

                "value":
                    safe_float(foot),

                "bodyArea":
                    "Front leg",

                "severity":
                    "elevated",

                "message":
                    "Large normalized image-plane hip-to-ankle offset detected. Review front-leg alignment from the selected camera view."
            })

        elif foot >= 0.55:

            risks.append({

                "parameter":
                    "frontLegAlignmentOffset",

                "value":
                    safe_float(foot),

                "bodyArea":
                    "Front leg",

                "severity":
                    "monitor",

                "message":
                    "Increased normalized image-plane hip-to-ankle offset detected. Review lower-limb alignment."
            })

    return risks


# ============================================================
# RECOMMENDATIONS
# ============================================================

def build_recommendations(
    measurements,
    risks,
    camera
):

    recommendations = []

    risk_parameters = {
        item["parameter"]
        for item in risks
    }

    if (
        "trunkLateralFlexion"
        in
        risk_parameters
    ):

        recommendations.append(
            "Work on maintaining controlled trunk alignment through delivery and follow-through."
        )

    if (
        "trunkForwardFlexion"
        in
        measurements
    ):

        recommendations.append(
            "Monitor trunk position consistently across multiple deliveries rather than judging a single frame."
        )

    if (
        "elbowAngle"
        in
        risk_parameters
    ):

        recommendations.append(
            "Review bowling-arm movement and elbow position with a qualified cricket coach."
        )

    if (
        "frontKneeAngle"
        in
        risk_parameters
    ):

        recommendations.append(
            "Work on controlled front-leg mechanics and stable front-knee positioning."
        )

    if (
        "frontLegAlignmentOffset"
        in
        risk_parameters
    ):

        recommendations.append(
            "Focus on consistent front-leg placement and lower-limb alignment."
        )

    if not recommendations:

        recommendations.append(
            "No major prototype biomechanical flags were identified in this delivery window."
        )

    recommendations.append(
        "Compare the same measurements across multiple deliveries before making coaching decisions."
    )

    if camera == "side":

        recommendations.append(
            "Use a front or oblique camera as an additional view when evaluating lateral trunk movement."
        )

    return recommendations


# ============================================================
# DETECTION QUALITY
# ============================================================

def calculate_detection_rate(
    raw_history
):

    if not raw_history:
        return 0.0

    valid = sum(
        1
        for item in raw_history
        if item is not None
    )

    return safe_float(
        valid /
        len(raw_history)
    )


def calculate_tracking_rate(
    tracked_history
):

    if not tracked_history:
        return 0.0

    valid = sum(
        1
        for item in tracked_history
        if item is not None
    )

    return safe_float(
        valid /
        len(tracked_history)
    )


def calculate_body_scale(
    history
):

    scales = []

    for points in history:

        scale = torso_length(
            points
        )

        if (
            scale is not None
            and
            scale > 0
        ):

            scales.append(
                scale
            )

    if not scales:
        return None

    return safe_float(
        median(scales)
    )


# ============================================================
# DELIVERY ANALYSIS
# ============================================================

def analyze_delivery(
    delivery_number,
    release_frame,
    release_confidence,
    history,
    measurements_history,
    fps,
    arm,
    camera
):

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

    reliability = calculate_reliability(
        history,
        start_frame,
        end_frame,
        measurements,
        arm
    )

    parameter_scores, technical_score = (
        calculate_scores(
            measurements,
            reliability,
            camera
        )
    )

    risks = build_risks(
        measurements,
        camera
    )

    recommendations = build_recommendations(
        measurements,
        risks,
        camera
    )

    return {

        "deliveryNumber":
            int(delivery_number),

        "releaseFrame": {

            "index":
                int(release_frame),

            "timestampSeconds":
                safe_float(
                    release_frame /
                    fps
                ),

            "confidence":
                safe_float(
                    release_confidence
                ),

            "detectionMethod":
                "wrist_speed_peak"
        },

        "analysisWindow": {

            "startFrame":
                int(start_frame),

            "endFrame":
                int(end_frame),

            "framesUsed":
                int(
                    end_frame
                    -
                    start_frame
                    +
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
            recommendations
    }


# ============================================================
# SESSION ANALYSIS
# ============================================================

def calculate_session_score(
    deliveries
):

    scores = []

    for delivery in deliveries:

        score = delivery.get(
            "technicalScore"
        )

        if score is not None:

            scores.append(
                float(score)
            )

    if not scores:
        return None

    return safe_float(
        sum(scores)
        /
        len(scores)
    )


def calculate_overall_reliability(
    deliveries
):

    values = []

    for delivery in deliveries:

        reliability = delivery.get(
            "reliability",
            {}
        )

        valid = [
            float(value)
            for value in reliability.values()
            if value is not None
        ]

        if valid:

            values.append(
                sum(valid)
                /
                len(valid)
            )

    if not values:
        return 0.0

    return safe_float(
        sum(values)
        /
        len(values)
    )


def build_session_recommendations(
    deliveries
):

    recommendations = []

    risk_counts = {}

    for delivery in deliveries:

        for risk in delivery.get(
            "riskIndicators",
            []
        ):

            parameter = risk.get(
                "parameter"
            )

            if parameter:

                risk_counts[
                    parameter
                ] = (
                    risk_counts.get(
                        parameter,
                        0
                    )
                    +
                    1
                )

    if risk_counts:

        strongest = sorted(
            risk_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for parameter, count in strongest[:3]:

            if parameter == "elbowAngle":

                recommendations.append(
                    f"Bowling-arm elbow movement was flagged in {count} delivery(s); compare the deliveries with a qualified coach."
                )

            elif parameter == "frontKneeAngle":

                recommendations.append(
                    f"Front-knee mechanics were flagged in {count} delivery(s); review consistency across the spell."
                )

            elif parameter == "trunkLateralFlexion":

                recommendations.append(
                    f"Trunk lateral inclination was flagged in {count} delivery(s); verify this with an appropriate front/oblique camera."
                )

            elif parameter == "frontLegAlignmentOffset":

                recommendations.append(
                    f"Front-leg image-plane alignment was flagged in {count} delivery(s); review consistency from the same camera position."
                )

    if not recommendations:

        recommendations.append(
            "No repeated prototype risk indicators were detected across the analyzed deliveries."
        )

    recommendations.append(
        "Use repeated deliveries and consistent camera placement for meaningful comparison."
    )

    return recommendations


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_video(
    input_path,
    arm="right",
    camera="side",
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

    if camera not in (
        "side",
        "front",
        "oblique"
    ):

        raise ValueError(
            "Camera must be side, front or oblique."
        )

    # --------------------------------------------------------
    # Detector and tracker
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

    print(
        f"Arm        : {arm}"
    )

    print(
        f"Camera     : {camera}"
    )

    print()

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    raw_history = []
    history = []
    measurements_history = []

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        raw_points = detector.detect(
            frame
        )

        raw_history.append(
            raw_points
        )

        tracked_points = None

        if raw_points is not None:

            tracked_points = tracker.update(
                raw_points
            )

        history.append(
            tracked_points
        )

        measurements_history.append(
            calculate_measurements(
                tracked_points,
                arm,
                camera
            )
        )

        frame_index += 1

        if frame_index % 30 == 0:

            percentage = (
                frame_index
                /
                max(
                    frame_count,
                    1
                )
            ) * 100

            print(
                f"Processing: {percentage:.1f}%"
            )

    cap.release()

    # --------------------------------------------------------
    # Detection statistics
    # --------------------------------------------------------

    detection_rate = (
        calculate_detection_rate(
            raw_history
        )
    )

    tracking_rate = (
        calculate_tracking_rate(
            history
        )
    )

    prediction_rate = safe_float(
        tracker.predicted_frames
        /
        max(
            len(history),
            1
        )
    )

    print()

    print(
        f"Raw YOLO detection rate: "
        f"{detection_rate * 100:.1f}%"
    )

    print(
        f"Tracking availability: "
        f"{tracking_rate * 100:.1f}%"
    )

    print(
        f"Predicted-point rate: "
        f"{prediction_rate * 100:.1f}%"
    )

    # --------------------------------------------------------
    # Release detection
    # --------------------------------------------------------

    release_frames = detect_all_releases(
        history,
        arm
    )

    deliveries = []

    speeds = calculate_wrist_speeds(
        history,
        arm
    )

    speed_map = {
        index + 1: speed
        for index, speed in enumerate(
            speeds
        )
        if speed is not None
    }

    for delivery_number, release_frame in enumerate(
        release_frames,
        start=1
    ):

        confidence = (
            calculate_release_confidence(
                speed_map,
                release_frame
            )
        )

        delivery = analyze_delivery(
            delivery_number,
            release_frame,
            confidence,
            history,
            measurements_history,
            fps,
            arm,
            camera
        )

        deliveries.append(
            delivery
        )

    # --------------------------------------------------------
    # No releases
    # --------------------------------------------------------

    if not deliveries:

        print()
        print(
            "WARNING: No reliable delivery release peaks detected."
        )

        print(
            "No artificial fallback release frame will be used."
        )

    # --------------------------------------------------------
    # Body scale
    # --------------------------------------------------------

    body_scale = calculate_body_scale(
        history
    )

    # --------------------------------------------------------
    # Session score
    # --------------------------------------------------------

    session_score = (
        calculate_session_score(
            deliveries
        )
    )

    overall_reliability = (
        calculate_overall_reliability(
            deliveries
        )
    )

    session_recommendations = (
        build_session_recommendations(
            deliveries
        )
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = {

        "status":
            "ok"
            if deliveries
            else "no_releases_detected",

        "version":
            VERSION,

        "analysisType":
            "multi_delivery",

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

        "camera":
            camera,

        "bodyScale": {

            "medianTorsoPixels":
                body_scale
        },

        "detectionQuality": {

            "yoloDetectionRate":
                detection_rate,

            "trackingRate":
                tracking_rate,

            "predictionRate":
                prediction_rate
        },

        "sessionSummary": {

            "deliveriesDetected":
                int(
                    len(deliveries)
                ),

            "technicalScore":
                session_score,

            "overallReliability":
                overall_reliability,

            "recommendations":
                session_recommendations
        },

        "deliveries":
            deliveries
    }

    # --------------------------------------------------------
    # Output
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

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("=" * 60)

    print()

    print(
        f"Deliveries detected     : "
        f"{len(deliveries)}"
    )

    print(
        f"Prototype session score : "
        f"{session_score if session_score is not None else '--'}"
    )

    print(
        f"Overall reliability     : "
        f"{overall_reliability * 100:.1f}%"
    )

    print(
        f"YOLO detection rate     : "
        f"{detection_rate * 100:.1f}%"
    )

    print(
        f"Tracking rate           : "
        f"{tracking_rate * 100:.1f}%"
    )

    print(
        f"Prediction rate         : "
        f"{prediction_rate * 100:.1f}%"
    )

    # --------------------------------------------------------
    # Delivery summaries
    # --------------------------------------------------------

    for delivery in deliveries:

        release = delivery[
            "releaseFrame"
        ]

        print()
        print(
            f"DELIVERY "
            f"{delivery['deliveryNumber']}"
        )

        print("-" * 60)

        print(
            f"Release frame      : "
            f"{release['index']}"
        )

        print(
            f"Release time       : "
            f"{release['timestampSeconds']}s"
        )

        print(
            f"Release confidence : "
            f"{release['confidence']}"
        )

        print(
            f"Technical score    : "
            f"{delivery['technicalScore']}"
        )

        print(
            f"Risk indicators    : "
            f"{len(delivery['riskIndicators'])}"
        )

        print()

        print("MEASUREMENTS")
        print("-" * 60)

        for name, value in (
            delivery[
                "measurements"
            ].items()
        ):

            print(
                f"{name:<32}: "
                f"{value if value is not None else '--'}"
            )

        print()

        print("RELIABILITY")
        print("-" * 60)

        for name, value in (
            delivery[
                "reliability"
            ].items()
        ):

            percentage = (
                value * 100
                if value is not None
                else 0
            )

            print(
                f"{name:<32}: "
                f"{percentage:.1f}%"
            )

        print()

        print("RISK INDICATORS")
        print("-" * 60)

        if not delivery[
            "riskIndicators"
        ]:

            print(
                "No major prototype biomechanical flags."
            )

        else:

            for risk in delivery[
                "riskIndicators"
            ]:

                print(
                    f"["
                    f"{risk['severity'].upper()}"
                    f"] "
                    f"{risk['bodyArea']}: "
                    f"{risk['message']}"
                )

        print()

        print("RECOMMENDATIONS")
        print("-" * 60)

        for recommendation in (
            delivery[
                "recommendations"
            ]
        ):

            print(
                f"- {recommendation}"
            )

    # --------------------------------------------------------
    # Session recommendations
    # --------------------------------------------------------

    print()
    print("SESSION RECOMMENDATIONS")
    print("-" * 60)

    for recommendation in (
        session_recommendations
    ):

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
        "--camera",
        choices=[
            "side",
            "front",
            "oblique"
        ],
        default="side",
        help="Camera orientation"
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
            camera=args.camera,
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
        print(str(error))
        print("=" * 60)

        raise


if __name__ == "__main__":
    main()

