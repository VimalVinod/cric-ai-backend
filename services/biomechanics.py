"""
biomechanics.py - V2

Converts pose landmarks into bowling biomechanics features.

V2 improvements:
- Body-scale normalized head stability
- More robust release-frame estimation
- Wrist velocity + acceleration signals
- Bowling phase estimation
- More camera-independent normalized measurements
- Defensive handling of missing keypoints

IMPORTANT:
These measurements are movement-analysis features, not medical diagnoses
or validated injury predictions.
"""

import numpy as np

MIN_CONF = 0.3


# ============================================================
# BASIC UTILITIES
# ============================================================

def _pt(keypoints, name):
    if keypoints is None or name not in keypoints:
        return None

    x, y, c = keypoints[name]

    if c < MIN_CONF:
        return None

    return np.array([x, y], dtype=float)


def _distance(a, b):
    if a is None or b is None:
        return None

    return float(np.linalg.norm(a - b))


def angle_3pt(a, b, c):
    """Angle at point b formed by a-b-c."""

    if a is None or b is None or c is None:
        return None

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denom == 0:
        return None

    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosang)))


def line_angle_from_vertical(p1, p2):
    """
    Angle of p1 -> p2 relative to vertical.

    0 degrees = vertical.
    """

    if p1 is None or p2 is None:
        return None

    v = p2 - p1

    vertical = np.array([0.0, -1.0])

    denom = np.linalg.norm(v)

    if denom == 0:
        return None

    cosang = np.clip(np.dot(v, vertical) / denom, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosang)))


# ============================================================
# FRAME BIOMECHANICS
# ============================================================

def frame_features(keypoints, bowling_arm="right"):

    nose = _pt(keypoints, "nose")

    l_sh = _pt(keypoints, "left_shoulder")
    r_sh = _pt(keypoints, "right_shoulder")

    l_el = _pt(keypoints, "left_elbow")
    r_el = _pt(keypoints, "right_elbow")

    l_wr = _pt(keypoints, "left_wrist")
    r_wr = _pt(keypoints, "right_wrist")

    l_hip = _pt(keypoints, "left_hip")
    r_hip = _pt(keypoints, "right_hip")

    l_kn = _pt(keypoints, "left_knee")
    r_kn = _pt(keypoints, "right_knee")

    l_an = _pt(keypoints, "left_ankle")
    r_an = _pt(keypoints, "right_ankle")


    # --------------------------------------------------------
    # BOWLING SIDE
    # --------------------------------------------------------

    if bowling_arm == "right":

        bowl_sh = r_sh
        bowl_el = r_el
        bowl_wr = r_wr

        front_kn = l_kn
        front_hip = l_hip
        front_an = l_an

        back_kn = r_kn
        back_hip = r_hip
        back_an = r_an

    else:

        bowl_sh = l_sh
        bowl_el = l_el
        bowl_wr = l_wr

        front_kn = r_kn
        front_hip = r_hip
        front_an = r_an

        back_kn = l_kn
        back_hip = l_hip
        back_an = l_an


    # --------------------------------------------------------
    # BODY CENTRES
    # --------------------------------------------------------

    mid_shoulder = None

    if l_sh is not None and r_sh is not None:
        mid_shoulder = (l_sh + r_sh) / 2


    mid_hip = None

    if l_hip is not None and r_hip is not None:
        mid_hip = (l_hip + r_hip) / 2


    features = {}


    # ========================================================
    # ARM
    # ========================================================

    features["elbowAngle"] = angle_3pt(
        bowl_sh,
        bowl_el,
        bowl_wr
    )


    # ========================================================
    # KNEES
    # ========================================================

    features["frontKneeAngle"] = angle_3pt(
        front_hip,
        front_kn,
        front_an
    )

    features["backKneeAngle"] = angle_3pt(
        back_hip,
        back_kn,
        back_an
    )


    # ========================================================
    # TRUNK
    # ========================================================

    features["trunkForwardFlexion"] = line_angle_from_vertical(
        mid_hip,
        mid_shoulder
    )


    # --------------------------------------------------------
    # LATERAL TRUNK LEAN
    #
    # Instead of using raw image coordinates, calculate the
    # shoulder/hip displacement relative to torso length.
    # --------------------------------------------------------

    if mid_shoulder is not None and mid_hip is not None:

        torso_length = np.linalg.norm(
            mid_shoulder - mid_hip
        )

        if torso_length > 1e-6:

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

        else:
            features["trunkLateralFlexion"] = None

    else:
        features["trunkLateralFlexion"] = None


    # ========================================================
    # SHOULDER ORIENTATION
    # ========================================================

    if l_sh is not None and r_sh is not None:

        d = r_sh - l_sh

        features["shoulderLineAngle"] = float(
            np.degrees(
                np.arctan2(d[1], d[0])
            )
        )

    else:
        features["shoulderLineAngle"] = None


    # ========================================================
    # HIP ORIENTATION
    # ========================================================

    if l_hip is not None and r_hip is not None:

        d = r_hip - l_hip

        features["hipLineAngle"] = float(
            np.degrees(
                np.arctan2(d[1], d[0])
            )
        )

    else:
        features["hipLineAngle"] = None


    # ========================================================
    # HIP / SHOULDER SEPARATION
    # ========================================================

    if (
        features["shoulderLineAngle"] is not None
        and
        features["hipLineAngle"] is not None
    ):

        diff = (
            features["shoulderLineAngle"]
            -
            features["hipLineAngle"]
        )

        diff = (diff + 180) % 360 - 180

        features["hipShoulderSeparation"] = abs(diff)

    else:
        features["hipShoulderSeparation"] = None


    # ========================================================
    # HEAD POSITION
    # ========================================================

    features["_headPos"] = (
        None
        if nose is None
        else (float(nose[0]), float(nose[1]))
    )

    features["_midShoulder"] = (
        None
        if mid_shoulder is None
        else (
            float(mid_shoulder[0]),
            float(mid_shoulder[1])
        )
    )


    # ========================================================
    # BODY SCALE
    # ========================================================

    # Shoulder width is useful for scale normalization.
    shoulder_width = _distance(
        l_sh,
        r_sh
    )

    features["_shoulderWidth"] = shoulder_width


    # ========================================================
    # FRONT FOOT ALIGNMENT
    # ========================================================

    if front_hip is not None and front_an is not None:

        leg_length = np.linalg.norm(
            front_hip - front_an
        )

        if leg_length > 1e-6:

            horizontal_offset = abs(
                front_hip[0] - front_an[0]
            )

            features["frontFootOffset"] = float(
                horizontal_offset / leg_length
            )

        else:
            features["frontFootOffset"] = None

    else:
        features["frontFootOffset"] = None


    return features


# ============================================================
# SEQUENCE ANALYSIS
# ============================================================

def sequence_features(frames_data, bowling_arm="right"):

    per_frame = []


    # --------------------------------------------------------
    # Calculate frame-by-frame biomechanics
    # --------------------------------------------------------

    for fd in frames_data:

        feats = frame_features(
            fd["keypoints"],
            bowling_arm=bowling_arm
        )

        feats["frame_idx"] = fd["frame_idx"]
        feats["timestamp_s"] = fd["timestamp_s"]

        per_frame.append(feats)


    # ========================================================
    # HEAD STABILITY
    # ========================================================

    head_stability = _calculate_head_stability(
        per_frame
    )


    # ========================================================
    # WRIST MOTION
    # ========================================================

    wrist_motion = _calculate_wrist_motion(
        frames_data,
        bowling_arm
    )


    # ========================================================
    # RELEASE FRAME
    # ========================================================

    release_frame_idx = _estimate_release_frame(
        frames_data,
        bowling_arm
    )


    # ========================================================
    # BOWLING PHASES
    # ========================================================

    phases = _estimate_phases(
        frames_data,
        bowling_arm,
        release_frame_idx
    )


    summary = {

        "headStability": head_stability,

        "releaseFrameIdx": release_frame_idx,

        "frameCount": len(frames_data),

        "wristPeakSpeed": wrist_motion["peakSpeed"],

        "wristPeakAcceleration": wrist_motion[
            "peakAcceleration"
        ],

        "phases": phases,
    }


    return per_frame, summary


# ============================================================
# HEAD STABILITY
# ============================================================

def _calculate_head_stability(per_frame):

    offsets = []
    scales = []


    for f in per_frame:

        if (
            f.get("_headPos") is not None
            and
            f.get("_midShoulder") is not None
            and
            f.get("_shoulderWidth") is not None
        ):

            head = np.array(
                f["_headPos"],
                dtype=float
            )

            shoulder = np.array(
                f["_midShoulder"],
                dtype=float
            )

            shoulder_width = f["_shoulderWidth"]

            if shoulder_width > 1e-6:

                normalized_offset = (
                    head - shoulder
                ) / shoulder_width

                offsets.append(
                    normalized_offset
                )

                scales.append(
                    shoulder_width
                )


    if len(offsets) < 3:
        return None


    offsets = np.array(offsets)


    # Movement of head relative to torso.
    std = np.std(
        offsets,
        axis=0
    )


    movement = float(
        np.linalg.norm(std)
    )


    # Convert movement into a 0-1 stability score.
    #
    # Smaller movement = better stability.
    #
    # This threshold is a V2 engineering heuristic and
    # should later be calibrated with real bowling data.

    stability = 1.0 - (
        movement / 0.20
    )


    stability = float(
        np.clip(
            stability,
            0.0,
            1.0
        )
    )


    return round(
        stability,
        3
    )


# ============================================================
# WRIST MOTION
# ============================================================

def _calculate_wrist_motion(
    frames_data,
    bowling_arm
):

    wrist_name = (
        "right_wrist"
        if bowling_arm == "right"
        else "left_wrist"
    )


    positions = []

    for fd in frames_data:

        kp = fd["keypoints"]

        if (
            kp
            and wrist_name in kp
            and kp[wrist_name][2] >= MIN_CONF
        ):

            positions.append(
                np.array(
                    kp[wrist_name][:2],
                    dtype=float
                )
            )

        else:

            positions.append(None)


    speeds = []
    speed_frames = []


    for i in range(1, len(positions)):

        current = positions[i]
        previous = positions[i - 1]


        if current is not None and previous is not None:

            speed = float(
                np.linalg.norm(
                    current - previous
                )
            )

            speeds.append(speed)
            speed_frames.append(i)


    if not speeds:

        return {
            "peakSpeed": None,
            "peakAcceleration": None
        }


    # --------------------------------------------------------
    # Acceleration = change in wrist speed
    # --------------------------------------------------------

    accelerations = []

    for i in range(1, len(speeds)):

        acceleration = (
            speeds[i] -
            speeds[i - 1]
        )

        accelerations.append(
            abs(float(acceleration))
        )


    peak_speed = max(speeds)

    peak_acceleration = (
        max(accelerations)
        if accelerations
        else None
    )


    return {

        "peakSpeed": round(
            peak_speed,
            3
        ),

        "peakAcceleration": (
            None
            if peak_acceleration is None
            else round(
                peak_acceleration,
                3
            )
        ),
    }


# ============================================================
# RELEASE DETECTION
# ============================================================

def _estimate_release_frame(
    frames_data,
    bowling_arm
):

    wrist_name = (
        "right_wrist"
        if bowling_arm == "right"
        else "left_wrist"
    )


    positions = []


    for fd in frames_data:

        kp = fd["keypoints"]


        if (
            kp
            and wrist_name in kp
            and kp[wrist_name][2] >= MIN_CONF
        ):

            positions.append(
                np.array(
                    kp[wrist_name][:2],
                    dtype=float
                )
            )

        else:

            positions.append(None)


    if len(positions) < 5:
        return None


    # --------------------------------------------------------
    # Calculate wrist speed
    # --------------------------------------------------------

    speeds = [None]


    for i in range(1, len(positions)):

        if (
            positions[i] is not None
            and
            positions[i - 1] is not None
        ):

            speed = np.linalg.norm(
                positions[i] -
                positions[i - 1]
            )

            speeds.append(
                float(speed)
            )

        else:

            speeds.append(None)


    valid_speeds = [
        s for s in speeds
        if s is not None
    ]


    if not valid_speeds:
        return None


    # --------------------------------------------------------
    # Ignore first 20% of video.
    #
    # The bowler shouldn't release during the early
    # run-up.
    # --------------------------------------------------------

    start_frame = int(
        len(frames_data) * 0.20
    )


    candidates = []

    for i in range(
        max(1, start_frame),
        len(speeds)
    ):

        if speeds[i] is not None:

            candidates.append(
                (i, speeds[i])
            )


    if not candidates:
        return None


    # --------------------------------------------------------
    # Find maximum wrist speed.
    # --------------------------------------------------------

    peak_frame, peak_speed = max(
        candidates,
        key=lambda x: x[1]
    )


    # --------------------------------------------------------
    # Look around the peak.
    #
    # Release normally occurs around the peak acceleration /
    # high-speed portion rather than necessarily at one exact
    # frame.
    # --------------------------------------------------------

    search_start = max(
        start_frame,
        peak_frame - 4
    )

    search_end = min(
        len(speeds) - 1,
        peak_frame + 4
    )


    local_candidates = [
        (i, speeds[i])
        for i in range(
            search_start,
            search_end + 1
        )
        if speeds[i] is not None
    ]


    if not local_candidates:
        return peak_frame


    # For V2 we use the peak speed inside the local
    # acceleration window.

    release_frame = max(
        local_candidates,
        key=lambda x: x[1]
    )[0]


    return int(release_frame)


# ============================================================
# BOWLING PHASE ESTIMATION
# ============================================================

def _estimate_phases(
    frames_data,
    bowling_arm,
    release_frame
):

    total = len(frames_data)


    if total == 0:
        return {}


    if release_frame is None:

        return {

            "runUp": {
                "start": 0,
                "end": int(total * 0.45)
            },

            "gather": {
                "start": int(total * 0.45),
                "end": int(total * 0.65)
            },

            "delivery": {
                "start": int(total * 0.65),
                "end": int(total * 0.85)
            },

            "followThrough": {
                "start": int(total * 0.85),
                "end": total - 1
            },
        }


    # --------------------------------------------------------
    # Use release as the main temporal anchor.
    #
    # These percentages are V2 initial estimates and should
    # later be replaced with learned phase detection.
    # --------------------------------------------------------

    release = int(release_frame)


    gather_start = max(
        0,
        int(release * 0.55)
    )

    delivery_start = max(
        gather_start,
        int(release * 0.75)
    )

    follow_start = min(
        total - 1,
        release + int(total * 0.08)
    )


    return {

        "runUp": {

            "start": 0,

            "end": max(
                0,
                gather_start - 1
            )
        },

        "gather": {

            "start": gather_start,

            "end": max(
                gather_start,
                delivery_start - 1
            )
        },

        "delivery": {

            "start": delivery_start,

            "end": release
        },

        "release": {

            "frame": release
        },

        "followThrough": {

            "start": follow_start,

            "end": total - 1
        },
    }