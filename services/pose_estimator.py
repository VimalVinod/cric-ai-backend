"""
pose_estimator.py
Extracts per-frame body keypoints from a bowling video using YOLOv8-pose.

Why YOLOv8-pose instead of MediaPipe Pose:
- MediaPipe's newer Task API requires downloading a .task model file from
  Google's storage servers at runtime.
- YOLOv8-pose ships its weights via GitHub releases (ultralytics/assets),
  is COCO-17-keypoint (sufficient for all V1 biomechanical parameters),
  and is a drop-in swap for MediaPipe later if you want higher-fidelity
  landmarks (33 points, better occlusion handling).

COCO-17 keypoint order (index -> name):
0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
"""

from ultralytics import YOLO
import cv2
import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KP_INDEX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt", conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def extract_video_landmarks(self, video_path, max_frames=None):
        """
        Runs pose estimation on every frame of the video.

        Returns:
            frames_data: list of dicts, one per frame:
                {
                    "frame_idx": int,
                    "timestamp_s": float,
                    "keypoints": {name: (x, y, confidence)} or None if no
                                 person detected in that frame,
                    "frame_w": int, "frame_h": int
                }
            meta: {"fps": float, "frame_count": int, "width": int, "height": int}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames_data = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and idx >= max_frames:
                break

            results = self.model(frame, verbose=False)
            keypoints = self._best_person_keypoints(results, width, height)

            frames_data.append({
                "frame_idx": idx,
                "timestamp_s": idx / fps,
                "keypoints": keypoints,
                "frame_w": width,
                "frame_h": height,
            })
            idx += 1

        cap.release()
        meta = {"fps": fps, "frame_count": frame_count, "width": width, "height": height}
        return frames_data, meta

    def _best_person_keypoints(self, results, width, height):
        """Picks the highest-confidence detected person and returns a
        name->(x,y,conf) dict, or None if nobody was detected."""
        r = results[0]
        if r.keypoints is None or len(r.keypoints.data) == 0:
            return None

        # Choose the detection with the highest mean keypoint confidence
        kpt_tensor = r.keypoints.data.cpu().numpy()  # (num_people, 17, 3)
        mean_conf = kpt_tensor[:, :, 2].mean(axis=1)
        best_idx = int(np.argmax(mean_conf))
        best = kpt_tensor[best_idx]  # (17, 3) -> x, y, conf

        if mean_conf[best_idx] < self.conf_threshold:
            return None

        return {
            KEYPOINT_NAMES[i]: (float(best[i][0]), float(best[i][1]), float(best[i][2]))
            for i in range(17)
        }
