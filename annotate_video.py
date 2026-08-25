"""
annotate_video.py - AI Bowling Analysis V4

V4 goals:
- Keep the complete original video visible.
- Draw pose skeleton over the video.
- Put analysis information in a separate right-side panel.
- Show technical score and parameter scores without covering the bowler.
- Highlight risk indicators.
- Show release-frame marker.
- Save an annotated MP4.

Usage:
    ..\.venv\Scripts\python.exe annotate_video.py "test_data\my_bowling.mp4" --arm right

Optional:
    --output "test_data\my_bowling_v4.mp4"
    --model "yolov8n-pose.pt"
"""

import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

class BowlingArmTracker:
    """
    V5 temporal tracker for the bowling arm.

    YOLO can occasionally lose the wrist/elbow immediately after release.
    This tracker:
      - prefers detections close to the previous position
      - rejects sudden unrealistic jumps
      - predicts short gaps
      - smooths the arm movement
      - gradually trusts YOLO again after tracking recovers
    """

    def __init__(self, arm="right"):
        self.arm = arm

        if arm == "right":
            self.elbow_name = "right_elbow"
            self.wrist_name = "right_wrist"
            self.shoulder_name = "right_shoulder"
        else:
            self.elbow_name = "left_elbow"
            self.wrist_name = "left_wrist"
            self.shoulder_name = "left_shoulder"

        self.prev_elbow = None
        self.prev_wrist = None

        self.prev_elbow_velocity = None
        self.prev_wrist_velocity = None

        self.missing_elbow = 0
        self.missing_wrist = 0

        self.MAX_MISSING_FRAMES = 10

        # Maximum allowed movement per frame.
        # This is intentionally generous because the bowling arm is fast.
        self.MAX_WRIST_JUMP = 220.0
        self.MAX_ELBOW_JUMP = 150.0

        self.SMOOTHING = 0.65

    def _distance(self, a, b):
        if a is None or b is None:
            return float("inf")

        return float(np.linalg.norm(np.array(a) - np.array(b)))

    def _smooth(self, current, previous):
        if current is None:
            return previous

        if previous is None:
            return current

        return (
            self.SMOOTHING * np.array(current)
            + (1.0 - self.SMOOTHING) * np.array(previous)
        )

    def _predict(self, previous, velocity):
        if previous is None:
            return None

        if velocity is None:
            return previous

        return np.array(previous) + np.array(velocity)

    def _extract_point(self, keypoints, name):
        if keypoints is None:
            return None

        point = keypoints.get(name)

        if point is None:
            return None

        x, y, confidence = point

        if confidence < 0.20:
            return None

        return np.array([float(x), float(y)])

    def update(self, keypoints):
        """
        Input:
            keypoints = {
                "right_elbow": (x, y, conf),
                "right_wrist": (x, y, conf),
                ...
            }

        Returns:
            corrected keypoints dictionary.
        """

        if keypoints is None:
            return None

        result = dict(keypoints)

        raw_elbow = self._extract_point(
            keypoints,
            self.elbow_name
        )

        raw_wrist = self._extract_point(
            keypoints,
            self.wrist_name
        )

        # ==========================================================
        # ELBOW TRACKING
        # ==========================================================

        elbow = raw_elbow

        if raw_elbow is not None and self.prev_elbow is not None:

            jump = self._distance(
                raw_elbow,
                self.prev_elbow
            )

            if jump > self.MAX_ELBOW_JUMP:

                # YOLO probably jumped to an incorrect point.
                elbow = self._predict(
                    self.prev_elbow,
                    self.prev_elbow_velocity
                )

                self.missing_elbow += 1

            else:
                self.missing_elbow = 0

        elif raw_elbow is None:

            elbow = self._predict(
                self.prev_elbow,
                self.prev_elbow_velocity
            )

            self.missing_elbow += 1

        else:
            self.missing_elbow = 0

        # Stop prediction after too many missing frames.
        if self.missing_elbow > self.MAX_MISSING_FRAMES:
            elbow = raw_elbow

        if elbow is not None:

            elbow = self._smooth(
                elbow,
                self.prev_elbow
            )

            if self.prev_elbow is not None:
                self.prev_elbow_velocity = (
                    elbow - self.prev_elbow
                )

            self.prev_elbow = np.array(elbow)

            old_conf = keypoints.get(
                self.elbow_name,
                (0, 0, 0)
            )[2]

            result[self.elbow_name] = (
                float(elbow[0]),
                float(elbow[1]),
                max(float(old_conf), 0.35)
            )

        # ==========================================================
        # WRIST TRACKING
        # ==========================================================

        wrist = raw_wrist

        if raw_wrist is not None and self.prev_wrist is not None:

            jump = self._distance(
                raw_wrist,
                self.prev_wrist
            )

            if jump > self.MAX_WRIST_JUMP:

                # Large sudden jump.
                # Use predicted position instead.
                wrist = self._predict(
                    self.prev_wrist,
                    self.prev_wrist_velocity
                )

                self.missing_wrist += 1

            else:
                self.missing_wrist = 0

        elif raw_wrist is None:

            wrist = self._predict(
                self.prev_wrist,
                self.prev_wrist_velocity
            )

            self.missing_wrist += 1

        else:
            self.missing_wrist = 0

        if self.missing_wrist > self.MAX_MISSING_FRAMES:
            wrist = raw_wrist

        if wrist is not None:

            wrist = self._smooth(
                wrist,
                self.prev_wrist
            )

            if self.prev_wrist is not None:
                self.prev_wrist_velocity = (
                    wrist - self.prev_wrist
                )

            self.prev_wrist = np.array(wrist)

            old_conf = keypoints.get(
                self.wrist_name,
                (0, 0, 0)
            )[2]

            result[self.wrist_name] = (
                float(wrist[0]),
                float(wrist[1]),
                max(float(old_conf), 0.35)
            )

        return result
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

# Skeleton connections
SKELETON = [
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),

    ("left_shoulder", "right_shoulder"),

    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),

    ("left_hip", "right_hip"),

    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),

    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


# ============================================================
# DISPLAY SETTINGS
# ============================================================

CONF_THRESHOLD = 0.30

PANEL_WIDTH = 390

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================
# POSE ESTIMATOR
# ============================================================

class PoseEstimator:

    def __init__(self, model_path="yolov8n-pose.pt", conf_threshold=0.30):

        print("Loading YOLO pose model...")

        self.model = YOLO(model_path)

        self.conf_threshold = conf_threshold

        print("Model loaded.")

    def detect(self, frame):

        results = self.model(frame, verbose=False)

        if not results:
            return None

        result = results[0]

        if result.keypoints is None:
            return None

        if len(result.keypoints.data) == 0:
            return None

        keypoints = result.keypoints.data.cpu().numpy()

        # ----------------------------------------------------
        # Select person with highest average confidence
        # ----------------------------------------------------

        mean_conf = keypoints[:, :, 2].mean(axis=1)

        best_index = int(np.argmax(mean_conf))

        if mean_conf[best_index] < self.conf_threshold:
            return None

        person = keypoints[best_index]

        points = {}

        for i, name in enumerate(KEYPOINT_NAMES):

            x = float(person[i][0])
            y = float(person[i][1])
            conf = float(person[i][2])

            points[name] = (x, y, conf)

        return points


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def valid_point(points, name):

    if points is None:
        return False

    if name not in points:
        return False

    x, y, confidence = points[name]

    return confidence >= CONF_THRESHOLD


def get_point(points, name):

    if not valid_point(points, name):
        return None

    x, y, confidence = points[name]

    return int(round(x)), int(round(y))


def angle_3pt(a, b, c):

    if a is None or b is None or c is None:
        return None

    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom == 0:
        return None

    cos_angle = np.dot(ba, bc) / denom

    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


def line_angle_vertical(p1, p2):

    if p1 is None or p2 is None:
        return None

    v = np.array(p2, dtype=float) - np.array(p1, dtype=float)

    vertical = np.array([0, -1], dtype=float)

    denom = np.linalg.norm(v)

    if denom == 0:
        return None

    cos_angle = np.dot(v, vertical) / denom

    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_angle)))


# ============================================================
# BIOMECHANICAL FEATURES
# ============================================================

def calculate_features(points, bowling_arm="right"):

    if points is None:
        return {}

    def p(name):
        return get_point(points, name)

    nose = p("nose")

    l_sh = p("left_shoulder")
    r_sh = p("right_shoulder")

    l_el = p("left_elbow")
    r_el = p("right_elbow")

    l_wr = p("left_wrist")
    r_wr = p("right_wrist")

    l_hip = p("left_hip")
    r_hip = p("right_hip")

    l_kn = p("left_knee")
    r_kn = p("right_knee")

    l_an = p("left_ankle")
    r_an = p("right_ankle")

    if bowling_arm == "right":

        bowling_shoulder = r_sh
        bowling_elbow = r_el
        bowling_wrist = r_wr

        front_hip = l_hip
        front_knee = l_kn
        front_ankle = l_an

        back_hip = r_hip
        back_knee = r_kn
        back_ankle = r_an

    else:

        bowling_shoulder = l_sh
        bowling_elbow = l_el
        bowling_wrist = l_wr

        front_hip = r_hip
        front_knee = r_kn
        front_ankle = r_an

        back_hip = l_hip
        back_knee = l_kn
        back_ankle = l_an

    # --------------------------------------------------------
    # Midpoints
    # --------------------------------------------------------

    mid_shoulder = None
    mid_hip = None

    if l_sh and r_sh:
        mid_shoulder = (
            (l_sh[0] + r_sh[0]) / 2,
            (l_sh[1] + r_sh[1]) / 2
        )

    if l_hip and r_hip:
        mid_hip = (
            (l_hip[0] + r_hip[0]) / 2,
            (l_hip[1] + r_hip[1]) / 2
        )

    features = {}

    # --------------------------------------------------------
    # Elbow
    # --------------------------------------------------------

    features["elbowAngle"] = angle_3pt(
        bowling_shoulder,
        bowling_elbow,
        bowling_wrist
    )

    # --------------------------------------------------------
    # Knees
    # --------------------------------------------------------

    features["frontKneeAngle"] = angle_3pt(
        front_hip,
        front_knee,
        front_ankle
    )

    features["backKneeAngle"] = angle_3pt(
        back_hip,
        back_knee,
        back_ankle
    )

    # --------------------------------------------------------
    # Trunk forward flexion
    # --------------------------------------------------------

    features["trunkForwardFlexion"] = line_angle_vertical(
        mid_hip,
        mid_shoulder
    )

    # --------------------------------------------------------
    # Trunk lateral flexion
    # --------------------------------------------------------

    if mid_shoulder and mid_hip:

        torso_length = np.linalg.norm(
            np.array(mid_shoulder) - np.array(mid_hip)
        )

        if torso_length > 0:

            horizontal_offset = abs(
                mid_shoulder[0] - mid_hip[0]
            )

            features["trunkLateralFlexion"] = float(
                np.degrees(
                    np.arctan2(
                        horizontal_offset,
                        torso_length
                    )
                )
            )

    # --------------------------------------------------------
    # Shoulder line
    # --------------------------------------------------------

    if l_sh and r_sh:

        dx = r_sh[0] - l_sh[0]
        dy = r_sh[1] - l_sh[1]

        features["shoulderLineAngle"] = float(
            np.degrees(np.arctan2(dy, dx))
        )

    # --------------------------------------------------------
    # Hip line
    # --------------------------------------------------------

    if l_hip and r_hip:

        dx = r_hip[0] - l_hip[0]
        dy = r_hip[1] - l_hip[1]

        features["hipLineAngle"] = float(
            np.degrees(np.arctan2(dy, dx))
        )

    # --------------------------------------------------------
    # Hip shoulder separation
    # --------------------------------------------------------

    if (
        features.get("shoulderLineAngle") is not None
        and features.get("hipLineAngle") is not None
    ):

        difference = (
            features["shoulderLineAngle"]
            - features["hipLineAngle"]
        )

        difference = (difference + 180) % 360 - 180

        features["hipShoulderSeparation"] = abs(difference)

    # --------------------------------------------------------
    # Front foot alignment
    # --------------------------------------------------------

    if front_hip and front_ankle:

        leg_length = np.linalg.norm(
            np.array(front_hip) -
            np.array(front_ankle)
        )

        if leg_length > 0:

            features["frontFootOffset"] = (
                abs(front_hip[0] - front_ankle[0])
                / leg_length
            )

    return features


# ============================================================
# SKELETON DRAWING
# ============================================================

def draw_skeleton(frame, points):

    if points is None:
        return

    # --------------------------------------------------------
    # Draw bones
    # --------------------------------------------------------

    for a, b in SKELETON:

        pa = get_point(points, a)
        pb = get_point(points, b)

        if pa is None or pb is None:
            continue

        cv2.line(
            frame,
            pa,
            pb,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Draw joints
    # --------------------------------------------------------

    for name in KEYPOINT_NAMES:

        point = get_point(points, name)

        if point is None:
            continue

        cv2.circle(
            frame,
            point,
            5,
            (0, 140, 255),
            -1,
            cv2.LINE_AA
        )


# ============================================================
# RELEASE DETECTION
# ============================================================

def detect_release(history, bowling_arm):

    wrist_name = (
        "right_wrist"
        if bowling_arm == "right"
        else "left_wrist"
    )

    speeds = []

    for i in range(1, len(history)):

        previous = history[i - 1]
        current = history[i]

        if previous is None or current is None:
            continue

        p1 = get_point(previous, wrist_name)
        p2 = get_point(current, wrist_name)

        if p1 is None or p2 is None:
            continue

        speed = np.linalg.norm(
            np.array(p2, dtype=float)
            - np.array(p1, dtype=float)
        )

        speeds.append((i, speed))

    if not speeds:
        return None

    return max(
        speeds,
        key=lambda x: x[1]
    )[0]


# ============================================================
# TECHNICAL SCORE
# ============================================================

REFERENCE_RANGES = {

    "frontKneeAngle": (155, 180),

    "elbowAngle": (150, 180),

    "trunkLateralFlexion": (0, 15),

    "trunkForwardFlexion": (15, 40),

    "hipShoulderSeparation": (20, 45),

    "frontFootOffset": (0.0, 0.35),
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

    if low <= value <= high:
        return 100.0

    span = max(high - low, 1)

    if value < low:
        distance = low - value

    else:
        distance = value - high

    score = 100 - (distance / span) * 100

    return max(0, min(100, score))


def calculate_scores(features):

    scores = {}

    for name, (low, high) in REFERENCE_RANGES.items():

        value = features.get(name)

        score = parameter_score(
            value,
            low,
            high
        )

        scores[name] = {
            "value": value,
            "score": score
        }

    total = 0
    weight_total = 0

    for name, weight in WEIGHTS.items():

        score = scores[name]["score"]

        if score is None:
            continue

        total += score * weight
        weight_total += weight

    technical_score = (
        total / weight_total
        if weight_total > 0
        else None
    )

    return scores, technical_score


# ============================================================
# RISK ANALYSIS
# ============================================================

def calculate_risks(features):

    risks = []

    lateral = features.get("trunkLateralFlexion")

    if lateral is not None and lateral > 25:

        risks.append(
            "High trunk lateral movement"
        )

    knee = features.get("frontKneeAngle")

    if knee is not None and knee < 140:

        risks.append(
            "Excessive front-knee flexion"
        )

    elbow = features.get("elbowAngle")

    if elbow is not None and elbow < 140:

        risks.append(
            "Bowling arm flexion near release"
        )

    foot = features.get("frontFootOffset")

    if foot is not None and foot > 0.5:

        risks.append(
            "Large front-foot alignment deviation"
        )

    return risks


# ============================================================
# PANEL DRAWING
# ============================================================

def draw_panel(
    panel,
    frame_number,
    total_frames,
    fps,
    scores,
    technical_score,
    risks,
    release_frame
):

    panel[:] = (24, 24, 24)

    x = 25

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    cv2.putText(
        panel,
        "AI BOWLING ANALYSIS",
        (x, 40),
        FONT,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        "V4",
        (x, 68),
        FONT,
        0.55,
        (160, 160, 160),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Technical score
    # --------------------------------------------------------

    score_text = (
        "--"
        if technical_score is None
        else f"{technical_score:.1f}"
    )

    cv2.putText(
        panel,
        "TECHNICAL SCORE",
        (x, 115),
        FONT,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        score_text,
        (x, 165),
        FONT,
        1.7,
        (0, 220, 255),
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        panel,
        "/ 100",
        (x + 105, 163),
        FONT,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if total_frames > 0:

        progress = frame_number / total_frames

        cv2.rectangle(
            panel,
            (x, 185),
            (PANEL_WIDTH - 25, 195),
            (70, 70, 70),
            -1
        )

        cv2.rectangle(
            panel,
            (x, 185),
            (
                x + int(
                    (PANEL_WIDTH - 50)
                    * progress
                ),
                195
            ),
            (0, 200, 255),
            -1
        )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    y = 235

    cv2.putText(
        panel,
        "TECHNICAL PARAMETERS",
        (x, y),
        FONT,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    y += 30

    display_names = {

        "frontKneeAngle": "Front Knee",

        "elbowAngle": "Elbow",

        "trunkLateralFlexion": "Trunk Lateral",

        "trunkForwardFlexion": "Trunk Forward",

        "hipShoulderSeparation": "Hip-Shoulder",

        "frontFootOffset": "Front Foot",
    }

    for key, display_name in display_names.items():

        data = scores.get(key)

        if data is None:
            continue

        value = data.get("value")
        score = data.get("score")

        if value is None or score is None:

            value_text = "--"
            score_text = "--"

        else:

            value_text = f"{value:.1f}"
            score_text = f"{score:.0f}"

        cv2.putText(
            panel,
            display_name,
            (x, y),
            FONT,
            0.48,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            panel,
            value_text,
            (225, y),
            FONT,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            panel,
            score_text,
            (320, y),
            FONT,
            0.48,
            (0, 220, 255),
            1,
            cv2.LINE_AA
        )

        y += 30

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    y += 10

    cv2.putText(
        panel,
        "DELIVERY",
        (x, y),
        FONT,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    y += 30

    current_time = frame_number / fps

    cv2.putText(
        panel,
        f"Time: {current_time:.2f}s",
        (x, y),
        FONT,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    y += 25

    if release_frame is not None:

        release_time = release_frame / fps

        cv2.putText(
            panel,
            f"Release: {release_time:.2f}s",
            (x, y),
            FONT,
            0.48,
            (0, 220, 255),
            1,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            panel,
            "Release: detecting...",
            (x, y),
            FONT,
            0.48,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Risks
    # --------------------------------------------------------

    y += 50

    cv2.putText(
        panel,
        "MOVEMENT FLAGS",
        (x, y),
        FONT,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    y += 28

    if not risks:

        cv2.putText(
            panel,
            "No major flags",
            (x, y),
            FONT,
            0.48,
            (100, 220, 120),
            1,
            cv2.LINE_AA
        )

    else:

        for risk in risks[:4]:

            # Word wrap
            words = risk.split()

            line = ""

            for word in words:

                test_line = (
                    line + " " + word
                ).strip()

                if len(test_line) > 31:

                    cv2.putText(
                        panel,
                        "• " + line,
                        (x, y),
                        FONT,
                        0.42,
                        (0, 170, 255),
                        1,
                        cv2.LINE_AA
                    )

                    y += 22

                    line = word

                else:

                    line = test_line

            if line:

                cv2.putText(
                    panel,
                    "• " + line,
                    (x, y),
                    FONT,
                    0.42,
                    (0, 170, 255),
                    1,
                    cv2.LINE_AA
                )

                y += 25


# ============================================================
# MAIN ANNOTATION FUNCTION
# ============================================================

def annotate_video(
    input_path,
    output_path=None,
    bowling_arm="right",
    model_path="yolov8n-pose.pt"
    
):

    print("=" * 60)
    print("AI BOWLING ANALYSIS - V4")
    print("=" * 60)

       # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    estimator = PoseEstimator(
        model_path=model_path,
        conf_threshold=CONF_THRESHOLD
    )

    # --------------------------------------------------------
    # V5 bowling-arm temporal tracker
    # --------------------------------------------------------

    arm_tracker = BowlingArmTracker(
        arm=bowling_arm
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():

        raise ValueError(
            f"Could not open video: {input_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    if output_path is None:

        base, ext = os.path.splitext(
            input_path
        )

        output_path = (
            base +
            "_v4_annotated.mp4"
        )

    # --------------------------------------------------------
    # Output dimensions
    # --------------------------------------------------------

    output_width = width + PANEL_WIDTH

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (
            output_width,
            height
        )
    )

    if not writer.isOpened():

        raise ValueError(
            "Could not create output video."
        )

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    history = []

    all_features = []

    frame_index = 0

    print()
    print(
        f"Input:  {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Resolution: {width} x {height}"
    )

    print(
        f"Frames: {total_frames}"
    )

    print()

    while True:

        ret, frame = cap.read()

        if not ret:
            break
        # ----------------------------------------------------
        # Pose detection
        # ----------------------------------------------------

        points = estimator.detect(frame)

        # V5 bowling-arm temporal tracking
        if points is not None:
            points = arm_tracker.update(points)

        history.append(points)

        # ----------------------------------------------------
        # Draw skeleton
        # ----------------------------------------------------

        draw_skeleton(
            frame,
            points
        )
        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        features = calculate_features(
            points,
            bowling_arm
        )

        all_features.append(
            features
        )

        # ----------------------------------------------------
        # Determine release from frames processed so far
        # ----------------------------------------------------

        release_frame = detect_release(
            history,
            bowling_arm
        )

        # ----------------------------------------------------
        # Use release features
        # ----------------------------------------------------

        release_features = {}

        if (
            release_frame is not None
            and release_frame < len(all_features)
        ):

            start = max(
                0,
                release_frame - 1
            )

            end = min(
                len(all_features),
                release_frame + 2
            )

            window = [
                f
                for f in all_features[start:end]
                if f
            ]

            if window:

                keys = set()

                for item in window:
                    keys.update(
                        item.keys()
                    )

                for key in keys:

                    values = [
                        item[key]
                        for item in window
                        if item.get(key) is not None
                    ]

                    if values:

                        release_features[key] = (
                            sum(values)
                            / len(values)
                        )

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        scores, technical_score = (
            calculate_scores(
                release_features
            )
        )

        risks = calculate_risks(
            release_features
        )

        # ----------------------------------------------------
        # Create panel
        # ----------------------------------------------------

        panel = np.zeros(
            (
                height,
                PANEL_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        draw_panel(
            panel=panel,
            frame_number=frame_index,
            total_frames=total_frames,
            fps=fps,
            scores=scores,
            technical_score=technical_score,
            risks=risks,
            release_frame=release_frame
        )

        # ----------------------------------------------------
        # Combine video + panel
        # ----------------------------------------------------

        combined = np.hstack(
            (
                frame,
                panel
            )
        )

        writer.write(
            combined
        )

        frame_index += 1

        if frame_index % 30 == 0:

            percent = (
                frame_index
                / max(total_frames, 1)
                * 100
            )

            print(
                f"Processing: {percent:.1f}%"
            )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    writer.release()

    print()
    print("=" * 60)
    print("V4 ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Output: {output_path}"
    )

    print(
        f"Processed frames: {frame_index}"
    )

    print("=" * 60)


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="AI Bowling Analysis V4"
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
        "--output",
        default=None,
        help="Output annotated video path"
    )

    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="YOLO pose model path"
    )

    args = parser.parse_args()

    annotate_video(
        input_path=args.video,
        output_path=args.output,
        bowling_arm=args.arm,
        model_path=args.model
    )