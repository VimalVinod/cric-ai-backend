"""
pose_visualizer.py

V3 - Bowling pose visualization.

Reads a bowling video, runs YOLOv8-pose on every frame,
draws the detected skeleton, frame information, and
estimated release frame.

This module is for VISUAL VALIDATION only.
It does not calculate injury probability or technical scores.
"""

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# COCO-17 KEYPOINT NAMES
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
# COCO SKELETON CONNECTIONS
# ============================================================

SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),

    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),
]


# ============================================================
# MAIN VISUALIZER
# ============================================================

class PoseVisualizer:

    def __init__(
        self,
        model_path="yolov8n-pose.pt",
        confidence=0.3
    ):

        self.model = YOLO(model_path)

        self.confidence = confidence


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    def annotate_video(
        self,
        input_path,
        output_path,
        bowling_arm="right"
    ):

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():

            raise ValueError(
                f"Could not open video: {input_path}"
            )


        fps = cap.get(
            cv2.CAP_PROP_FPS
        ) or 30.0

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

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        # ----------------------------------------------------
        # Video writer
        # ----------------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )


        # ----------------------------------------------------
        # First pass:
        # detect poses and save wrist positions
        # ----------------------------------------------------

        frames = []

        wrist_positions = []


        wrist_name = (
            "right_wrist"
            if bowling_arm == "right"
            else "left_wrist"
        )

        wrist_index = KEYPOINT_NAMES.index(
            wrist_name
        )


        frame_index = 0


        print()
        print("======================================")
        print("V3 BOWLING POSE VISUALIZATION")
        print("======================================")
        print()
        print(
            f"Video: {input_path}"
        )
        print(
            f"Frames: {total_frames}"
        )
        print(
            f"FPS: {fps:.2f}"
        )
        print()


        while True:

            ret, frame = cap.read()

            if not ret:
                break


            results = self.model(
                frame,
                verbose=False
            )


            result = results[0]


            keypoints = None


            if (
                result.keypoints is not None
                and
                len(result.keypoints.data) > 0
            ):

                data = (
                    result.keypoints
                    .data
                    .cpu()
                    .numpy()
                )


                # --------------------------------------------
                # Select strongest person
                # --------------------------------------------

                mean_conf = (
                    data[:, :, 2]
                    .mean(axis=1)
                )


                best_index = int(
                    np.argmax(mean_conf)
                )


                if (
                    mean_conf[best_index]
                    >= self.confidence
                ):

                    keypoints = data[
                        best_index
                    ]


            frames.append(
                (
                    frame,
                    keypoints
                )
            )


            # ------------------------------------------------
            # Save wrist position
            # ------------------------------------------------

            if keypoints is not None:

                x = float(
                    keypoints[
                        wrist_index
                    ][0]
                )

                y = float(
                    keypoints[
                        wrist_index
                    ][1]
                )

                conf = float(
                    keypoints[
                        wrist_index
                    ][2]
                )


                if conf >= self.confidence:

                    wrist_positions.append(
                        np.array(
                            [x, y],
                            dtype=float
                        )
                    )

                else:

                    wrist_positions.append(
                        None
                    )

            else:

                wrist_positions.append(
                    None
                )


            frame_index += 1


            if frame_index % 20 == 0:

                percent = (
                    frame_index /
                    max(total_frames, 1)
                    * 100
                )

                print(
                    f"Processing: "
                    f"{frame_index}/"
                    f"{total_frames} "
                    f"({percent:.1f}%)"
                )


        cap.release()


        # ====================================================
        # ESTIMATE RELEASE
        # ====================================================

        release_frame = self._estimate_release(
            wrist_positions
        )


        print()
        print(
            f"Estimated release frame: "
            f"{release_frame}"
        )


        if release_frame is not None:

            print(
                f"Release time: "
                f"{release_frame / fps:.3f}s"
            )


        # ====================================================
        # SECOND PASS
        # DRAW VIDEO
        # ====================================================

        print()
        print("Rendering annotated video...")


        for i, (
            frame,
            keypoints
        ) in enumerate(frames):


            # ----------------------------------------------
            # Draw skeleton
            # ----------------------------------------------

            if keypoints is not None:

                self._draw_skeleton(
                    frame,
                    keypoints
                )


            # ----------------------------------------------
            # Header
            # ----------------------------------------------

            self._draw_header(
                frame,
                i,
                fps,
                total_frames
            )


            # ----------------------------------------------
            # Bowling information
            # ----------------------------------------------

            self._draw_bowling_info(
                frame,
                bowling_arm
            )


            # ----------------------------------------------
            # Release marker
            # ----------------------------------------------

            if release_frame is not None:

                distance = abs(
                    i - release_frame
                )


                if i == release_frame:

                    self._draw_release(
                        frame,
                        "RELEASE FRAME"
                    )

                elif distance <= 3:

                    self._draw_release(
                        frame,
                        "RELEASE WINDOW"
                    )


            # ----------------------------------------------
            # Write frame
            # ----------------------------------------------

            writer.write(frame)


        writer.release()


        print()
        print("======================================")
        print("DONE")
        print("======================================")
        print()
        print(
            f"Output: {output_path}"
        )
        print()


        return {
            "output": output_path,
            "frames": len(frames),
            "fps": fps,
            "release_frame": release_frame,
        }


    # ========================================================
    # DRAW SKELETON
    # ========================================================

    def _draw_skeleton(
        self,
        frame,
        keypoints
    ):


        # ----------------------------------------------------
        # Draw bones
        # ----------------------------------------------------

        for a, b in SKELETON:

            if (
                keypoints[a][2]
                >= self.confidence
                and
                keypoints[b][2]
                >= self.confidence
            ):

                x1 = int(
                    keypoints[a][0]
                )

                y1 = int(
                    keypoints[a][1]
                )

                x2 = int(
                    keypoints[b][0]
                )

                y2 = int(
                    keypoints[b][1]
                )


                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )


        # ----------------------------------------------------
        # Draw joints
        # ----------------------------------------------------

        for i in range(
            len(keypoints)
        ):

            x = int(
                keypoints[i][0]
            )

            y = int(
                keypoints[i][1]
            )

            conf = float(
                keypoints[i][2]
            )


            if conf >= self.confidence:

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 255, 255),
                    -1
                )


    # ========================================================
    # HEADER
    # ========================================================

    def _draw_header(
        self,
        frame,
        frame_index,
        fps,
        total_frames
    ):

        timestamp = (
            frame_index / fps
        )


        text1 = (
            f"BOWLING AI | "
            f"Frame {frame_index}/{total_frames}"
        )

        text2 = (
            f"Time: {timestamp:.3f}s"
        )


        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 70),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            text1,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            text2,
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )


    # ========================================================
    # BOWLING INFO
    # ========================================================

    def _draw_bowling_info(
        self,
        frame,
        bowling_arm
    ):

        text = (
            f"Bowling arm: "
            f"{bowling_arm.upper()}"
        )


        cv2.putText(
            frame,
            text,
            (
                15,
                frame.shape[0] - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    # ========================================================
    # RELEASE MARKER
    # ========================================================

    def _draw_release(
        self,
        frame,
        text
    ):

        box_height = 55


        cv2.rectangle(
            frame,
            (
                0,
                frame.shape[0] // 2
                - box_height
            ),
            (
                frame.shape[1],
                frame.shape[0] // 2
                + box_height
            ),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            text,
            (
                20,
                frame.shape[0] // 2 + 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3
        )


    # ========================================================
    # RELEASE ESTIMATION
    # ========================================================

    def _estimate_release(
        self,
        positions
    ):

        if len(positions) < 5:

            return None


        speeds = []


        for i in range(
            1,
            len(positions)
        ):

            current = positions[i]

            previous = positions[i - 1]


            if (
                current is not None
                and
                previous is not None
            ):

                speed = np.linalg.norm(
                    current - previous
                )

                speeds.append(
                    (
                        i,
                        float(speed)
                    )
                )


        if not speeds:

            return None


        # Ignore first 20% of clip.
        min_frame = int(
            len(positions) * 0.20
        )


        candidates = [
            item
            for item in speeds
            if item[0] >= min_frame
        ]


        if not candidates:

            return None


        release_frame = max(
            candidates,
            key=lambda x: x[1]
        )[0]


        return int(
            release_frame
        )


# ============================================================
# COMMAND LINE ENTRY
# ============================================================

if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser(
        description=
        "V3 Bowling Pose Visualizer"
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
        default="right"
    )


    parser.add_argument(
        "--out",
        default=
        "test_data/my_bowling_annotated.mp4"
    )


    parser.add_argument(
        "--model",
        default=
        "yolov8n-pose.pt"
    )


    args = parser.parse_args()


    visualizer = PoseVisualizer(
        model_path=args.model
    )


    visualizer.annotate_video(
        input_path=args.video,
        output_path=args.out,
        bowling_arm=args.arm
    )