"""
view_normalization.py
V6.1 - View-Invariant Body Coordinate System

Converts YOLO 2D pose keypoints from pixel coordinates into
body-relative normalized coordinates.

Goals:
    - Reduce dependency on video resolution
    - Reduce dependency on player size in frame
    - Center coordinates around the body
    - Normalize by body scale
    - Prepare the pose for perspective-aware analysis

This module does NOT attempt to reconstruct full 3D pose.
"""

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MIN_CONF = 0.30


# ============================================================
# BASIC HELPERS
# ============================================================

def valid_point(keypoints, name, min_conf=MIN_CONF):
    """
    Check whether a keypoint exists and has sufficient confidence.
    """

    if keypoints is None:
        return False

    if name not in keypoints:
        return False

    x, y, confidence = keypoints[name]

    return confidence >= min_conf


def get_point(keypoints, name, min_conf=MIN_CONF):
    """
    Return a keypoint as numpy [x, y].
    """

    if not valid_point(keypoints, name, min_conf):
        return None

    x, y, confidence = keypoints[name]

    return np.array(
        [float(x), float(y)],
        dtype=np.float32
    )


# ============================================================
# BODY LANDMARKS
# ============================================================

def body_center(keypoints):
    """
    Estimate the center of the body using shoulder and hip centers.

    Returns:
        np.array([x, y]) or None
    """

    left_shoulder = get_point(
        keypoints,
        "left_shoulder"
    )

    right_shoulder = get_point(
        keypoints,
        "right_shoulder"
    )

    left_hip = get_point(
        keypoints,
        "left_hip"
    )

    right_hip = get_point(
        keypoints,
        "right_hip"
    )

    points = []

    if left_shoulder is not None:
        points.append(left_shoulder)

    if right_shoulder is not None:
        points.append(right_shoulder)

    if left_hip is not None:
        points.append(left_hip)

    if right_hip is not None:
        points.append(right_hip)

    if len(points) < 2:
        return None

    return np.mean(points, axis=0)


def shoulder_center(keypoints):
    """
    Center between left and right shoulders.
    """

    left = get_point(
        keypoints,
        "left_shoulder"
    )

    right = get_point(
        keypoints,
        "right_shoulder"
    )

    if left is None or right is None:
        return None

    return (left + right) / 2.0


def hip_center(keypoints):
    """
    Center between left and right hips.
    """

    left = get_point(
        keypoints,
        "left_hip"
    )

    right = get_point(
        keypoints,
        "right_hip"
    )

    if left is None or right is None:
        return None

    return (left + right) / 2.0


# ============================================================
# BODY SCALE
# ============================================================

def estimate_body_scale(keypoints):
    """
    Estimate body scale.

    Priority:
        shoulder width
        hip width
        shoulder-to-hip distance

    Returns:
        positive float or None
    """

    left_shoulder = get_point(
        keypoints,
        "left_shoulder"
    )

    right_shoulder = get_point(
        keypoints,
        "right_shoulder"
    )

    left_hip = get_point(
        keypoints,
        "left_hip"
    )

    right_hip = get_point(
        keypoints,
        "right_hip"
    )

    scales = []

    # Shoulder width
    if (
        left_shoulder is not None
        and right_shoulder is not None
    ):

        width = np.linalg.norm(
            right_shoulder - left_shoulder
        )

        if width > 1:
            scales.append(width)

    # Hip width
    if (
        left_hip is not None
        and right_hip is not None
    ):

        width = np.linalg.norm(
            right_hip - left_hip
        )

        if width > 1:
            scales.append(width)

    # Shoulder-to-hip length
    sh_center = shoulder_center(keypoints)
    h_center = hip_center(keypoints)

    if (
        sh_center is not None
        and h_center is not None
    ):

        torso = np.linalg.norm(
            sh_center - h_center
        )

        if torso > 1:
            scales.append(torso)

    if not scales:
        return None

    # Median is more robust than mean.
    return float(np.median(scales))


# ============================================================
# NORMALIZED POSE
# ============================================================

def normalize_pose(keypoints):
    """
    Convert raw pixel coordinates into body-relative coordinates.

    Output format:

        {
            "nose": {
                "x": ...,
                "y": ...,
                "confidence": ...
            },
            ...
        }

    Coordinates are:
        x = horizontal body-relative position
        y = vertical body-relative position

    The body center becomes approximately (0, 0).
    """

    if keypoints is None:
        return None

    center = body_center(keypoints)

    scale = estimate_body_scale(keypoints)

    if center is None or scale is None:
        return None

    normalized = {}

    for name, value in keypoints.items():

        if value is None:
            continue

        x, y, confidence = value

        if confidence < MIN_CONF:
            continue

        point = np.array(
            [float(x), float(y)],
            dtype=np.float32
        )

        relative = (
            point - center
        ) / scale

        normalized[name] = {
            "x": float(relative[0]),
            "y": float(relative[1]),
            "confidence": float(confidence)
        }

    return {
        "points": normalized,
        "center": (
            float(center[0]),
            float(center[1])
        ),
        "scale": float(scale)
    }


# ============================================================
# NORMALIZED DISTANCE
# ============================================================

def normalized_distance(
    normalized_pose,
    point_a,
    point_b
):
    """
    Calculate distance between two normalized keypoints.
    """

    if normalized_pose is None:
        return None

    points = normalized_pose["points"]

    if (
        point_a not in points
        or point_b not in points
    ):
        return None

    a = np.array([
        points[point_a]["x"],
        points[point_a]["y"]
    ])

    b = np.array([
        points[point_b]["x"],
        points[point_b]["y"]
    ])

    return float(
        np.linalg.norm(a - b)
    )


# ============================================================
# NORMALIZED VECTOR
# ============================================================

def normalized_vector(
    normalized_pose,
    point_a,
    point_b
):
    """
    Vector from point_a to point_b
    in normalized body coordinates.
    """

    if normalized_pose is None:
        return None

    points = normalized_pose["points"]

    if (
        point_a not in points
        or point_b not in points
    ):
        return None

    a = np.array([
        points[point_a]["x"],
        points[point_a]["y"]
    ])

    b = np.array([
        points[point_b]["x"],
        points[point_b]["y"]
    ])

    return b - a


# ============================================================
# NORMALIZED ANGLE
# ============================================================

def normalized_angle(
    normalized_pose,
    point_a,
    vertex,
    point_c
):
    """
    Calculate a 3-point angle using normalized coordinates.

    Because normalization only changes translation and scale,
    the angle remains geometrically meaningful.
    """

    if normalized_pose is None:
        return None

    points = normalized_pose["points"]

    if not all(
        name in points
        for name in (
            point_a,
            vertex,
            point_c
        )
    ):
        return None

    a = np.array([
        points[point_a]["x"],
        points[point_a]["y"]
    ])

    b = np.array([
        points[vertex]["x"],
        points[vertex]["y"]
    ])

    c = np.array([
        points[point_c]["x"],
        points[point_c]["y"]
    ])

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator <= 1e-8:
        return None

    cosine = (
        np.dot(ba, bc)
        / denominator
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
# BODY ORIENTATION
# ============================================================

def estimate_body_orientation(keypoints):
    """
    Estimate the visible body orientation from the shoulder line.

    Returns angle in degrees.

    This is a preliminary 2D orientation estimate.
    It will be improved in later V6 versions.
    """

    left = get_point(
        keypoints,
        "left_shoulder"
    )

    right = get_point(
        keypoints,
        "right_shoulder"
    )

    if left is None or right is None:
        return None

    vector = right - left

    angle = np.degrees(
        np.arctan2(
            vector[1],
            vector[0]
        )
    )

    return float(angle)


# ============================================================
# VIEW QUALITY
# ============================================================

def estimate_view_quality(keypoints):
    """
    Estimate how clearly the body is visible.

    This is NOT yet a front/side/rear classifier.

    It simply measures how many important landmarks
    are confidently detected.
    """

    important_points = [
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

    detected = 0

    for name in important_points:

        if valid_point(
            keypoints,
            name
        ):
            detected += 1

    return float(
        detected / len(important_points)
    )


# ============================================================
# COMPLETE NORMALIZATION
# ============================================================

def process_pose(keypoints):
    """
    Main V6.1 entry point.

    Returns all normalized information needed by
    future view-invariant biomechanics modules.
    """

    normalized = normalize_pose(
        keypoints
    )

    orientation = estimate_body_orientation(
        keypoints
    )

    quality = estimate_view_quality(
        keypoints
    )

    return {
        "normalized_pose": normalized,

        "body_orientation": orientation,

        "view_quality": quality,

        "body_center": (
            None
            if normalized is None
            else normalized["center"]
        ),

        "body_scale": (
            None
            if normalized is None
            else normalized["scale"]
        )
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("V6.1 VIEW NORMALIZATION MODULE")
    print("=" * 60)

    test_pose = {

        "left_shoulder": (
            300,
            300,
            0.95
        ),

        "right_shoulder": (
            400,
            300,
            0.95
        ),

        "left_hip": (
            320,
            500,
            0.95
        ),

        "right_hip": (
            380,
            500,
            0.95
        ),

        "nose": (
            350,
            220,
            0.95
        ),
    }

    result = process_pose(
        test_pose
    )

    print()
    print("Body center:")
    print(result["body_center"])

    print()
    print("Body scale:")
    print(result["body_scale"])

    print()
    print("Body orientation:")
    print(result["body_orientation"])

    print()
    print("View quality:")
    print(result["view_quality"])

    print()
    print("Normalized pose:")
    print(result["normalized_pose"])

    print()
    print("=" * 60)
    print("V6.1 TEST COMPLETE")
    print("=" * 60)