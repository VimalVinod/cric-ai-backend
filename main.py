"""
main.py
The Day-1 deliverable from the spec:

    analyze_bowling(video)
            |
      technical score
            +
        parameters
            +
      risk indicators
            +
     recommendations

Usage:
    python3 main.py path/to/video.mp4 [--arm right|left] [--annotate]
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from services.pose_estimator import PoseEstimator
from services.biomechanics import sequence_features
from services.scoring import (
    score_parameters, compute_technical_score, risk_indicators, recommendations
)


def analyze_bowling(video_path, bowling_arm="right", model_path="yolov8n-pose.pt"):
    estimator = PoseEstimator(model_path=model_path)
    frames_data, meta = estimator.extract_video_landmarks(video_path)

    detected_frames = sum(1 for f in frames_data if f["keypoints"] is not None)
    if detected_frames == 0:
        return {
            "status": "error",
            "message": "Unable to analyze this video. No person detected in any frame. "
                       "Please upload a side-view video where the full body remains visible.",
        }
    if detected_frames / max(len(frames_data), 1) < 0.5:
        quality_warning = ("Player was only reliably detected in "
                            f"{detected_frames}/{len(frames_data)} frames. "
                            "Results may be less reliable — check lighting, framing, and camera stability.")
    else:
        quality_warning = None

    per_frame, summary = sequence_features(frames_data, bowling_arm=bowling_arm)

    release_idx = summary["releaseFrameIdx"]
    if release_idx is None:
        release_features = {}
    else:
        # average a small window around the estimated release frame for stability
        window = [f for f in per_frame if abs(f["frame_idx"] - release_idx) <= 1]
        release_features = _average_features(window)
    release_features["headStability"] = summary["headStability"]

    param_scores = score_parameters(release_features)
    technical_score = compute_technical_score(param_scores)
    risks = risk_indicators(release_features)
    recs = recommendations(release_features)

    result = {
        "status": "ok",
        "qualityWarning": quality_warning,
        "video": {
            "path": video_path,
            "fps": meta["fps"],
            "frameCount": meta["frame_count"],
            "width": meta["width"],
            "height": meta["height"],
        },
        "releaseFrame": {
            "index": release_idx,
            "timestampSeconds": None if release_idx is None else round(release_idx / meta["fps"], 3),
            "percentThroughClip": None if release_idx is None else round(
                100 * release_idx / max(len(frames_data) - 1, 1)
            ),
        },
        "technicalScore": technical_score,
        "parameterScores": param_scores,
        "riskIndicators": risks,
        "recommendations": recs,
        "detectionRate": round(detected_frames / max(len(frames_data), 1), 3),
    }
    return result


def _average_features(window_frames):
    if not window_frames:
        return {}
    keys = [k for k in window_frames[0].keys() if not k.startswith("_") and k not in ("frame_idx", "timestamp_s")]
    avg = {}
    for k in keys:
        vals = [f[k] for f in window_frames if f.get(k) is not None]
        avg[k] = sum(vals) / len(vals) if vals else None
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to bowling video (3-5s clip)")
    parser.add_argument("--arm", choices=["right", "left"], default="right",
                         help="Bowling arm (default: right)")
    parser.add_argument("--out", default=None, help="Path to write JSON output")
    args = parser.parse_args()

    result = analyze_bowling(args.video, bowling_arm=args.arm)
    output = json.dumps(result, indent=2)
    print(output)

    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
