"""
view_biomechanics.py - AI Bowling Analysis V7

REAL VIDEO + VIEW-ROBUST BIOMECHANICS

Usage:
    ..\.venv\Scripts\python.exe view_biomechanics.py "test_data\my_bowling.mp4" --arm right

Output:
    test_data\my_bowling_view_report.json
"""

import argparse
import json
import os
import math

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolov8n-pose.pt"

CONF_THRESHOLD = 0.30
KEYPOINT_CONF = 0.25

# Number of frames used around release
RELEASE_WINDOW = 5

# Maximum frames to inspect if video is extremely long
MAX_ANALYSIS_FRAMES = 1200


# ============================================================
# COCO 17 KEYPOINTS
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

def point(kp, name, min_conf=KEYPOINT_CONF):
    """
    Return [x,y] or None.
    """

    if kp is None:
        return None

    value = kp.get(name)

    if value is None:
        return None

    x, y, conf = value

    if conf < min_conf:
        return None

    return np.array([float(x), float(y)], dtype=float)


def distance(a, b):
    if a is None or b is None:
        return None

    return float(np.linalg.norm(a - b))


def midpoint(a, b):
    if a is None or b is None:
        return None

    return (a + b) / 2.0


def safe_mean(values):
    values = [
        float(v)
        for v in values
        if v is not None and np.isfinite(v)
    ]

    if not values:
        return None

    return float(np.mean(values))


def angle_3pt(a, b, c):
    """
    Angle ABC.
    """

    if a is None or b is None or c is None:
        return None

    ba = a - b
    bc = c - b

    denom = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denom <= 1e-8:
        return None

    cos_value = np.dot(ba, bc) / denom

    cos_value = np.clip(
        cos_value,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(cos_value)
        )
    )


def angle_from_vertical(a, b):
    """
    Angle of a->b relative to vertical.

    0 = vertical.
    """

    if a is None or b is None:
        return None

    v = b - a

    length = np.linalg.norm(v)

    if length <= 1e-8:
        return None

    vertical = np.array(
        [0.0, -1.0]
    )

    cosine = np.dot(v, vertical) / length

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
# YOLO POSE
# ============================================================

class PoseEstimator:

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence=CONF_THRESHOLD
    ):

        print("Loading YOLO pose model...")

        self.model = YOLO(model_path)

        self.confidence = confidence

        print("Model loaded.")

    def detect(self, frame):

        results = self.model(
            frame,
            verbose=False,
            conf=self.confidence
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

        # ----------------------------------------------------
        # Select the most likely bowler.
        #
        # Prefer the person with the largest number of
        # confident body keypoints.
        # ----------------------------------------------------

        best_index = None
        best_score = -1

        for person_index in range(
            len(data)
        ):

            person = data[person_index]

            confidences = person[:, 2]

            valid_count = np.sum(
                confidences >= KEYPOINT_CONF
            )

            mean_conf = np.mean(
                confidences
            )

            score = (
                valid_count * 2
                + mean_conf
            )

            if score > best_score:

                best_score = score
                best_index = person_index

        if best_index is None:
            return None

        person = data[best_index]

        keypoints = {}

        for i, name in enumerate(
            KEYPOINT_NAMES
        ):

            keypoints[name] = (
                float(person[i][0]),
                float(person[i][1]),
                float(person[i][2])
            )

        return keypoints


# ============================================================
# TEMPORAL BOWLING ARM TRACKER
# ============================================================

class BowlingArmTracker:

    def __init__(
        self,
        arm="right"
    ):

        self.arm = arm

        prefix = arm

        self.elbow_name = (
            f"{prefix}_elbow"
        )

        self.wrist_name = (
            f"{prefix}_wrist"
        )

        self.shoulder_name = (
            f"{prefix}_shoulder"
        )

        self.previous_elbow = None
        self.previous_wrist = None

        self.elbow_velocity = None
        self.wrist_velocity = None

        self.missing_elbow = 0
        self.missing_wrist = 0

        self.max_missing = 8

        self.max_elbow_jump = 120
        self.max_wrist_jump = 170

        self.smoothing = 0.70

    def _extract(
        self,
        keypoints,
        name
    ):

        if keypoints is None:
            return None

        value = keypoints.get(name)

        if value is None:
            return None

        x, y, confidence = value

        if confidence < 0.20:
            return None

        return np.array(
            [x, y],
            dtype=float
        )

    def _predict(
        self,
        previous,
        velocity
    ):

        if previous is None:
            return None

        if velocity is None:
            return previous.copy()

        return (
            previous
            + velocity
        )

    def _smooth(
        self,
        current,
        previous
    ):

        if current is None:
            return previous

        if previous is None:
            return current

        return (
            self.smoothing * current
            + (1.0 - self.smoothing)
            * previous
        )

    def _update_point(
        self,
        raw,
        previous,
        velocity,
        max_jump,
        missing
    ):

        current = raw

        if raw is None:

            current = self._predict(
                previous,
                velocity
            )

            missing += 1

        elif previous is not None:

            jump = np.linalg.norm(
                raw - previous
            )

            if jump > max_jump:

                current = self._predict(
                    previous,
                    velocity
                )

                missing += 1

            else:

                missing = 0

        else:

            missing = 0

        if missing > self.max_missing:

            current = raw

        if current is None:

            return (
                None,
                velocity,
                missing
            )

        current = self._smooth(
            current,
            previous
        )

        if previous is not None:

            velocity = (
                current
                - previous
            )

        return (
            current,
            velocity,
            missing
        )

    def update(
        self,
        keypoints
    ):

        if keypoints is None:
            return None

        result = dict(
            keypoints
        )

        raw_elbow = self._extract(
            keypoints,
            self.elbow_name
        )

        raw_wrist = self._extract(
            keypoints,
            self.wrist_name
        )

        (
            elbow,
            self.elbow_velocity,
            self.missing_elbow
        ) = self._update_point(
            raw_elbow,
            self.previous_elbow,
            self.elbow_velocity,
            self.max_elbow_jump,
            self.missing_elbow
        )

        (
            wrist,
            self.wrist_velocity,
            self.missing_wrist
        ) = self._update_point(
            raw_wrist,
            self.previous_wrist,
            self.wrist_velocity,
            self.max_wrist_jump,
            self.missing_wrist
        )

        if elbow is not None:

            old_conf = keypoints.get(
                self.elbow_name,
                (0, 0, 0)
            )[2]

            result[
                self.elbow_name
            ] = (
                float(elbow[0]),
                float(elbow[1]),
                max(
                    float(old_conf),
                    0.35
                )
            )

            self.previous_elbow = (
                elbow.copy()
            )

        if wrist is not None:

            old_conf = keypoints.get(
                self.wrist_name,
                (0, 0, 0)
            )[2]

            result[
                self.wrist_name
            ] = (
                float(wrist[0]),
                float(wrist[1]),
                max(
                    float(old_conf),
                    0.35
                )
            )

            self.previous_wrist = (
                wrist.copy()
            )

        return result


# ============================================================
# VIEW CLASSIFICATION
# ============================================================

def calculate_view_metrics(
    keypoints,
    frame_width,
    frame_height
):

    l_sh = point(
        keypoints,
        "left_shoulder"
    )

    r_sh = point(
        keypoints,
        "right_shoulder"
    )

    l_hip = point(
        keypoints,
        "left_hip"
    )

    r_hip = point(
        keypoints,
        "right_hip"
    )

    l_el = point(
        keypoints,
        "left_elbow"
    )

    r_el = point(
        keypoints,
        "right_elbow"
    )

    if (
        l_sh is None
        or r_sh is None
    ):

        return {
            "view": "unknown",
            "orientation": None,
            "visibility": 0.0
        }

    shoulder_width = distance(
        l_sh,
        r_sh
    )

    if shoulder_width is None:
        shoulder_width = 0

    # --------------------------------------------------------
    # Estimate whether camera sees the person front/rear
    # or side-on.
    #
    # In side view, the two shoulders overlap much more.
    # --------------------------------------------------------

    if (
        l_hip is not None
        and r_hip is not None
    ):

        hip_width = distance(
            l_hip,
            r_hip
        )

    else:

        hip_width = None

    shoulder_ratio = (
        shoulder_width
        / max(frame_width, 1)
    )

    if hip_width is not None:

        hip_ratio = (
            hip_width
            / max(frame_width, 1)
        )

    else:

        hip_ratio = shoulder_ratio

    # --------------------------------------------------------
    # Left/right shoulder depth proxy.
    #
    # When one shoulder moves close to the other, view is
    # more side-on.
    # --------------------------------------------------------

    side_score = 0.0

    if frame_width > 0:

        side_score = np.clip(
            1.0
            - (
                shoulder_width
                / (
                    frame_width
                    * 0.20
                )
            ),
            0.0,
            1.0
        )

    # --------------------------------------------------------
    # Visibility of both elbows.
    # --------------------------------------------------------

    elbow_visibility = 0.0

    if (
        l_el is not None
        and r_el is not None
    ):

        elbow_visibility = 1.0

    elif (
        l_el is not None
        or r_el is not None
    ):

        elbow_visibility = 0.5

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if shoulder_ratio < 0.08:

        view = "side"

    elif shoulder_ratio < 0.13:

        view = "diagonal"

    else:

        view = "front_or_rear"

    # If both sides of body are strongly visible,
    # front/rear is more likely.

    visibility = (
        0.65 * min(
            shoulder_ratio / 0.15,
            1.0
        )
        + 0.35 * elbow_visibility
    )

    return {
        "view": view,
        "orientation": float(
            side_score
        ),
        "visibility": float(
            np.clip(
                visibility,
                0,
                1
            )
        )
    }


# ============================================================
# BODY SCALE NORMALIZATION
# ============================================================

def body_scale(
    keypoints
):

    l_sh = point(
        keypoints,
        "left_shoulder"
    )

    r_sh = point(
        keypoints,
        "right_shoulder"
    )

    l_hip = point(
        keypoints,
        "left_hip"
    )

    r_hip = point(
        keypoints,
        "right_hip"
    )

    shoulder_mid = midpoint(
        l_sh,
        r_sh
    )

    hip_mid = midpoint(
        l_hip,
        r_hip
    )

    if (
        shoulder_mid is None
        or hip_mid is None
    ):

        return None

    torso = distance(
        shoulder_mid,
        hip_mid
    )

    if torso is None or torso <= 1:

        return None

    return float(torso)


# ============================================================
# NORMALIZED BIOMECHANICS
# ============================================================

def calculate_biomechanics(
    keypoints,
    bowling_arm="right"
):

    if keypoints is None:

        return {}

    # --------------------------------------------------------
    # Basic points
    # --------------------------------------------------------

    nose = point(
        keypoints,
        "nose"
    )

    l_sh = point(
        keypoints,
        "left_shoulder"
    )

    r_sh = point(
        keypoints,
        "right_shoulder"
    )

    l_el = point(
        keypoints,
        "left_elbow"
    )

    r_el = point(
        keypoints,
        "right_elbow"
    )

    l_wr = point(
        keypoints,
        "left_wrist"
    )

    r_wr = point(
        keypoints,
        "right_wrist"
    )

    l_hip = point(
        keypoints,
        "left_hip"
    )

    r_hip = point(
        keypoints,
        "right_hip"
    )

    l_kn = point(
        keypoints,
        "left_knee"
    )

    r_kn = point(
        keypoints,
        "right_knee"
    )

    l_an = point(
        keypoints,
        "left_ankle"
    )

    r_an = point(
        keypoints,
        "right_ankle"
    )

    # --------------------------------------------------------
    # Bowling arm / front leg
    # --------------------------------------------------------

    if bowling_arm == "right":

        bowl_sh = r_sh
        bowl_el = r_el
        bowl_wr = r_wr

        front_hip = l_hip
        front_knee = l_kn
        front_an = l_an

        back_hip = r_hip
        back_knee = r_kn
        back_an = r_an

    else:

        bowl_sh = l_sh
        bowl_el = l_el
        bowl_wr = l_wr

        front_hip = r_hip
        front_knee = r_kn
        front_an = r_an

        back_hip = l_hip
        back_knee = l_kn
        back_an = l_an

    # --------------------------------------------------------
    # Midpoints
    # --------------------------------------------------------

    shoulder_mid = midpoint(
        l_sh,
        r_sh
    )

    hip_mid = midpoint(
        l_hip,
        r_hip
    )

    features = {}

    # --------------------------------------------------------
    # Elbow
    # --------------------------------------------------------

    features[
        "elbowAngle"
    ] = angle_3pt(
        bowl_sh,
        bowl_el,
        bowl_wr
    )

    # --------------------------------------------------------
    # Front knee
    # --------------------------------------------------------

    features[
        "frontKneeAngle"
    ] = angle_3pt(
        front_hip,
        front_knee,
        front_an
    )

    # --------------------------------------------------------
    # Back knee
    # --------------------------------------------------------

    features[
        "backKneeAngle"
    ] = angle_3pt(
        back_hip,
        back_knee,
        back_an
    )

    # --------------------------------------------------------
    # Trunk forward flexion
    #
    # Instead of using absolute pixel dimensions, use
    # torso orientation.
    # --------------------------------------------------------

    features[
        "trunkForwardFlexion"
    ] = angle_from_vertical(
        hip_mid,
        shoulder_mid
    )

    # --------------------------------------------------------
    # Trunk lateral flexion
    #
    # Horizontal displacement / torso length.
    # This is scale independent.
    # --------------------------------------------------------

    if (
        shoulder_mid is not None
        and hip_mid is not None
    ):

        torso_length = distance(
            shoulder_mid,
            hip_mid
        )

        if (
            torso_length is not None
            and torso_length > 1
        ):

            horizontal = abs(
                shoulder_mid[0]
                - hip_mid[0]
            )

            features[
                "trunkLateralFlexion"
            ] = float(
                np.degrees(
                    np.arctan2(
                        horizontal,
                        torso_length
                    )
                )
            )

    # --------------------------------------------------------
    # Shoulder rotation proxy
    # --------------------------------------------------------

    if (
        l_sh is not None
        and r_sh is not None
    ):

        dx = r_sh[0] - l_sh[0]
        dy = r_sh[1] - l_sh[1]

        features[
            "shoulderLineAngle"
        ] = float(
            np.degrees(
                np.arctan2(
                    dy,
                    dx
                )
            )
        )

    # --------------------------------------------------------
    # Hip rotation proxy
    # --------------------------------------------------------

    if (
        l_hip is not None
        and r_hip is not None
    ):

        dx = r_hip[0] - l_hip[0]
        dy = r_hip[1] - l_hip[1]

        features[
            "hipLineAngle"
        ] = float(
            np.degrees(
                np.arctan2(
                    dy,
                    dx
                )
            )
        )

    # --------------------------------------------------------
    # Hip-shoulder separation
    # --------------------------------------------------------

    if (
        features.get(
            "shoulderLineAngle"
        ) is not None

        and

        features.get(
            "hipLineAngle"
        ) is not None
    ):

        difference = (
            features[
                "shoulderLineAngle"
            ]
            -
            features[
                "hipLineAngle"
            ]
        )

        difference = (
            difference + 180
        ) % 360 - 180

        features[
            "hipShoulderSeparation"
        ] = abs(
            difference
        )

    # --------------------------------------------------------
    # Front foot offset
    #
    # Normalized by leg length.
    # --------------------------------------------------------

    if (
        front_hip is not None
        and front_an is not None
    ):

        leg_length = distance(
            front_hip,
            front_an
        )

        if (
            leg_length is not None
            and leg_length > 1
        ):

            features[
                "frontFootOffset"
            ] = float(
                abs(
                    front_hip[0]
                    - front_an[0]
                )
                /
                leg_length
            )

    # --------------------------------------------------------
    # Head offset
    #
    # Normalized by torso length.
    # --------------------------------------------------------

    if (
        nose is not None
        and shoulder_mid is not None
    ):

        torso = body_scale(
            keypoints
        )

        if (
            torso is not None
            and torso > 1
        ):

            features[
                "headOffset"
            ] = float(
                distance(
                    nose,
                    shoulder_mid
                )
                / torso
            )

    return features


# ============================================================
# RELEASE DETECTION
# ============================================================

def detect_release(
    history,
    bowling_arm
):

    wrist_name = (
        "right_wrist"
        if bowling_arm == "right"
        else "left_wrist"
    )

    speeds = []

    for i in range(
        1,
        len(history)
    ):

        previous = history[i - 1]
        current = history[i]

        if (
            previous is None
            or current is None
        ):

            continue

        p1 = point(
            previous,
            wrist_name
        )

        p2 = point(
            current,
            wrist_name
        )

        if (
            p1 is None
            or p2 is None
        ):

            continue

        speed = distance(
            p1,
            p2
        )

        if speed is not None:

            speeds.append(
                (
                    i,
                    speed
                )
            )

    if not speeds:

        return None

    return max(
        speeds,
        key=lambda x: x[1]
    )[0]


# ============================================================
# AGGREGATE FEATURES
# ============================================================

def aggregate_features(
    frame_features
):

    result = {}

    if not frame_features:

        return result

    keys = set()

    for item in frame_features:

        keys.update(
            item.keys()
        )

    for key in keys:

        values = []

        for item in frame_features:

            value = item.get(key)

            if (
                value is not None
                and np.isfinite(value)
            ):

                values.append(
                    value
                )

        if values:

            # Median is deliberately used.
            #
            # Median is more resistant to a single bad
            # YOLO detection than a simple mean.

            result[key] = float(
                np.median(
                    values
                )
            )

    return result


# ============================================================
# RELIABILITY
# ============================================================

def calculate_reliability(
    feature_frames,
    total_window
):

    reliability = {}

    if total_window <= 0:
        return reliability

    keys = set()

    for item in feature_frames:

        keys.update(
            item.keys()
        )

    for key in keys:

        valid = 0

        for item in feature_frames:

            value = item.get(key)

            if (
                value is not None
                and np.isfinite(value)
            ):

                valid += 1

        reliability[key] = round(
            min(
                valid
                / total_window,
                1.0
            ),
            3
        )

    return reliability


# ============================================================
# VIEW-ROBUST SCORE
# ============================================================

REFERENCE_RANGES = {

    "frontKneeAngle": (
        150,
        180
    ),

    "elbowAngle": (
        150,
        180
    ),

    "trunkLateralFlexion": (
        0,
        20
    ),

    "trunkForwardFlexion": (
        10,
        45
    ),

    "hipShoulderSeparation": (
        15,
        50
    ),

    "frontFootOffset": (
        0,
        0.40
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


def score_parameter(
    value,
    low,
    high
):

    if value is None:

        return None

    if low <= value <= high:

        return 100.0

    if value < low:

        distance = low - value

    else:

        distance = value - high

    span = max(
        high - low,
        1
    )

    score = (
        100
        - (
            distance
            / span
        ) * 100
    )

    return float(
        np.clip(
            score,
            0,
            100
        )
    )


def calculate_scores(
    features
):

    scores = {}

    weighted_total = 0
    weight_total = 0

    for name, limits in (
        REFERENCE_RANGES.items()
    ):

        value = features.get(
            name
        )

        score = score_parameter(
            value,
            limits[0],
            limits[1]
        )

        scores[name] = {
            "value": value,
            "score": score
        }

        if score is not None:

            weight = WEIGHTS.get(
                name,
                0
            )

            weighted_total += (
                score
                * weight
            )

            weight_total += weight

    if weight_total > 0:

        technical_score = (
            weighted_total
            / weight_total
        )

    else:

        technical_score = None

    return (
        scores,
        technical_score
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

def calculate_risks(
    features
):

    risks = []

    lateral = features.get(
        "trunkLateralFlexion"
    )

    if (
        lateral is not None
        and lateral > 25
    ):

        risks.append({
            "parameter":
                "trunkLateralFlexion",

            "value":
                round(lateral, 1),

            "bodyArea":
                "Lower back",

            "severity":
                "monitor",

            "message":
                "High trunk lateral flexion detected."
        })

    elbow = features.get(
        "elbowAngle"
    )

    if (
        elbow is not None
        and elbow < 140
    ):

        risks.append({
            "parameter":
                "elbowAngle",

            "value":
                round(elbow, 1),

            "bodyArea":
                "Bowling elbow",

            "severity":
                "monitor",

            "message":
                "Significant elbow flexion near release."
        })

    knee = features.get(
        "frontKneeAngle"
    )

    if (
        knee is not None
        and knee < 140
    ):

        risks.append({
            "parameter":
                "frontKneeAngle",

            "value":
                round(knee, 1),

            "bodyArea":
                "Front knee",

            "severity":
                "monitor",

            "message":
                "Excessive front-knee flexion detected."
        })

    foot = features.get(
        "frontFootOffset"
    )

    if (
        foot is not None
        and foot > 0.50
    ):

        risks.append({
            "parameter":
                "frontFootOffset",

            "value":
                round(foot, 3),

            "bodyArea":
                "Front foot / ankle",

            "severity":
                "monitor",

            "message":
                "Large front-foot alignment deviation detected."
        })

    return risks


# ============================================================
# MAIN VIDEO ANALYSIS
# ============================================================

def analyze_video(
    video_path,
    bowling_arm="right",
    model_path=MODEL_PATH
):

    print("=" * 60)

    print(
        "AI BOWLING ANALYSIS - V7"
    )

    print(
        "REAL VIDEO VIEW-ROBUST BIOMECHANICS"
    )

    print("=" * 60)

    print()

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise ValueError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_video_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Video: {video_path}"
    )

    print(
        f"Resolution: "
        f"{frame_width} x {frame_height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {total_video_frames}"
    )

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    estimator = PoseEstimator(
        model_path=model_path
    )

    tracker = BowlingArmTracker(
        arm=bowling_arm
    )

    history = []

    all_features = []

    views = []

    scales = []

    frame_index = 0

    print()
    print(
        "Processing video..."
    )

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if (
            frame_index
            >= MAX_ANALYSIS_FRAMES
        ):

            break

        keypoints = estimator.detect(
            frame
        )

        if keypoints is not None:

            keypoints = tracker.update(
                keypoints
            )

        history.append(
            keypoints
        )

        # ----------------------------------------------------
        # View
        # ----------------------------------------------------

        if keypoints is not None:

            view_info = (
                calculate_view_metrics(
                    keypoints,
                    frame_width,
                    frame_height
                )
            )

            views.append(
                view_info["view"]
            )

            if (
                view_info.get(
                    "visibility"
                ) is not None
            ):

                scales.append(
                    body_scale(
                        keypoints
                    )
                )

            features = (
                calculate_biomechanics(
                    keypoints,
                    bowling_arm
                )
            )

            all_features.append(
                features
            )

        frame_index += 1

        if (
            frame_index % 30
            == 0
        ):

            if total_video_frames > 0:

                progress = (
                    frame_index
                    /
                    total_video_frames
                    * 100
                )

                print(
                    f"Processing: "
                    f"{progress:.1f}%"
                )

            else:

                print(
                    f"Processed "
                    f"{frame_index} frames"
                )

    cap.release()

    print()

    if not history:

        raise RuntimeError(
            "No frames were read from video."
        )

    # ========================================================
    # RELEASE
    # ========================================================

    release_frame = detect_release(
        history,
        bowling_arm
    )

    if release_frame is None:

        # Fallback: middle-late part of clip
        release_frame = int(
            len(history)
            * 0.80
        )

    release_start = max(
        0,
        release_frame
        - RELEASE_WINDOW
    )

    release_end = min(
        len(history),
        release_frame
        + RELEASE_WINDOW
        + 1
    )

    # --------------------------------------------------------
    # Recalculate features only around release.
    #
    # This is important because a bowling video contains
    # setup, run-up and recovery. We don't want those to
    # dominate the biomechanics report.
    # --------------------------------------------------------

    release_features_per_frame = []

    for i in range(
        release_start,
        release_end
    ):

        if i >= len(history):
            continue

        kp = history[i]

        if kp is None:
            continue

        features = (
            calculate_biomechanics(
                kp,
                bowling_arm
            )
        )

        release_features_per_frame.append(
            features
        )

    release_features = aggregate_features(
        release_features_per_frame
    )

    reliability = calculate_reliability(
        release_features_per_frame,
        release_end - release_start
    )

    # ========================================================
    # VIEW
    # ========================================================

    if views:

        from collections import Counter

        counter = Counter(
            views
        )

        detected_view = (
            counter
            .most_common(1)[0][0]
        )

    else:

        detected_view = "unknown"

    # ========================================================
    # SCALE
    # ========================================================

    valid_scales = [
        s
        for s in scales
        if s is not None
    ]

    if valid_scales:

        median_scale = float(
            np.median(
                valid_scales
            )
        )

    else:

        median_scale = None

    # ========================================================
    # SCORES
    # ========================================================

    scores, technical_score = (
        calculate_scores(
            release_features
        )
    )

    # ========================================================
    # RISKS
    # ========================================================

    risks = calculate_risks(
        release_features
    )

    # ========================================================
    # DETECTION RATE
    # ========================================================

    valid_frames = sum(
        1
        for kp in history
        if kp is not None
    )

    detection_rate = (
        valid_frames
        /
        max(
            len(history),
            1
        )
    )

    # ========================================================
    # VIEW QUALITY
    # ========================================================

    view_quality = (
        detection_rate
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {

        "status":
            "ok",

        "version":
            "V7",

        "video": {

            "path":
                video_path,

            "fps":
                round(
                    fps,
                    3
                ),

            "frameCount":
                len(history),

            "width":
                frame_width,

            "height":
                frame_height
        },

        "bowlingArm":
            bowling_arm,

        "view": {

            "detected":
                detected_view,

            "quality":
                round(
                    view_quality,
                    3
                )
        },

        "bodyScale": {

            "medianTorsoPixels":
                (
                    round(
                        median_scale,
                        3
                    )
                    if median_scale
                    is not None
                    else None
                )
        },

        "releaseFrame": {

            "index":
                release_frame,

            "timestampSeconds":
                round(
                    release_frame
                    / fps,
                    3
                ),

            "percentThroughClip":
                round(
                    release_frame
                    /
                    max(
                        len(history),
                        1
                    )
                    * 100,
                    1
                )
        },

        "measurements":
            {
                key:
                    (
                        round(
                            value,
                            3
                        )
                        if isinstance(
                            value,
                            (float, int)
                        )
                        else value
                    )
                for key, value
                in release_features.items()
            },

        "parameterScores":
            scores,

        "reliability":
            reliability,

        "technicalScore":
            (
                round(
                    technical_score,
                    1
                )
                if technical_score
                is not None
                else None
            ),

        "riskIndicators":
            risks,

        "detectionRate":
            round(
                detection_rate,
                3
            ),

        "analysisWindow": {

            "startFrame":
                release_start,

            "endFrame":
                release_end - 1,

            "framesUsed":
                len(
                    release_features_per_frame
                )
        }
    }

    return report


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    report,
    video_path
):

    base, _ = os.path.splitext(
        video_path
    )

    output_path = (
        base
        + "_view_report.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    report
):

    print()
    print("=" * 60)

    print(
        "V7 ANALYSIS RESULT"
    )

    print("=" * 60)

    print()

    print(
        "Status:"
    )

    print(
        report["status"]
    )

    print()

    print(
        "View:"
    )

    print(
        report["view"]["detected"]
    )

    print()

    print(
        "View quality:"
    )

    print(
        report["view"]["quality"]
    )

    print()

    print(
        "Detection rate:"
    )

    print(
        report["detectionRate"]
    )

    print()

    print(
        "Release frame:"
    )

    print(
        report[
            "releaseFrame"
        ]["index"]
    )

    print()

    print(
        "Release time:"
    )

    print(
        report[
            "releaseFrame"
        ]["timestampSeconds"],
        "seconds"
    )

    print()

    print(
        "-" * 60
    )

    print(
        "MEASUREMENTS"
    )

    print(
        "-" * 60
    )

    for name, value in (
        report[
            "measurements"
        ].items()
    ):

        if isinstance(
            value,
            (int, float)
        ):

            print(
                f"{name:<30}: "
                f"{value:.3f}"
            )

        else:

            print(
                f"{name:<30}: "
                f"{value}"
            )

    print()

    print(
        "-" * 60
    )

    print(
        "RELIABILITY"
    )

    print(
        "-" * 60
    )

    for name, value in (
        report[
            "reliability"
        ].items()
    ):

        print(
            f"{name:<30}: "
            f"{value:.3f}"
        )

    print()

    print(
        "-" * 60
    )

    print(
        "PARAMETER SCORES"
    )

    print(
        "-" * 60
    )

    for name, data in (
        report[
            "parameterScores"
        ].items()
    ):

        value = data.get(
            "value"
        )

        score = data.get(
            "score"
        )

        value_text = (
            f"{value:.2f}"
            if value is not None
            else "--"
        )

        score_text = (
            f"{score:.1f}"
            if score is not None
            else "--"
        )

        print(
            f"{name:<30}: "
            f"value={value_text:<8} "
            f"score={score_text}"
        )

    print()

    print(
        "-" * 60
    )

    print(
        "TECHNICAL SCORE"
    )

    print(
        "-" * 60
    )

    print(
        report[
            "technicalScore"
        ]
    )

    print()

    print(
        "-" * 60
    )

    print(
        "RISK INDICATORS"
    )

    print(
        "-" * 60
    )

    if not report[
        "riskIndicators"
    ]:

        print(
            "No major flags."
        )

    else:

        for risk in report[
            "riskIndicators"
        ]:

            print(
                f"[{risk['severity']}] "
                f"{risk['bodyArea']}: "
                f"{risk['message']}"
            )

    print()

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "AI Bowling Analysis V7 "
            "View-Robust Biomechanics"
        )
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
        default=MODEL_PATH,
        help="YOLO pose model"
    )

    args = parser.parse_args()

    report = analyze_video(
        video_path=args.video,
        bowling_arm=args.arm,
        model_path=args.model
    )

    output_path = save_report(
        report,
        args.video
    )

    print_report(
        report
    )

    print()

    print(
        f"JSON report saved to:"
    )

    print(
        output_path
    )

    print()


if __name__ == "__main__":

    main()