import argparse
import json
import cv2
import numpy as np

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

def draw_release_frame(video_path, json_path, output_path, arm):
    # 1. Load analysis data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    release_idx = data["releaseFrame"]["index"]
    tech_score = data.get("technicalScore", 0)
    risks = data.get("riskIndicators", [])
    
    # 2. Get the exact frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, release_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read release frame")
        return

    # 3. Run YOLO just ONCE on this single frame
    from ultralytics import YOLO
    model = YOLO("yolov8n-pose.pt")
    results = model(frame, verbose=False, conf=0.3, imgsz=256, device="cpu", max_det=1)
    
    points = {}
    if results and results[0].keypoints is not None:
        kpts = results[0].keypoints.data.cpu().numpy()[0]
        for i, name in enumerate(KEYPOINT_NAMES):
            points[name] = (int(kpts[i][0]), int(kpts[i][1]), float(kpts[i][2]))

    # 4. Identify At-Risk Joints based on JSON data
    risk_joints = set()
    
    # Map JSON risk parameters to specific keypoints based on bowling arm
    for risk in risks:
        param = risk.get("parameter")
        if param == "elbowAngle":
            risk_joints.add("right_elbow" if arm == "right" else "left_elbow")
        elif param == "frontKneeAngle":
            risk_joints.add("left_knee" if arm == "right" else "right_knee")
        elif param == "frontFootOffset":
            risk_joints.add("left_ankle" if arm == "right" else "right_ankle")
        elif param in ["trunkLateralFlexion", "trunkForwardFlexion"]:
            # For trunk, we'll highlight the mid-shoulder and mid-hip
            risk_joints.add("mid_shoulder")
            risk_joints.add("mid_hip")

    # Calculate midpoints for trunk highlighting
    if "left_shoulder" in points and "right_shoulder" in points:
        points["mid_shoulder"] = (
            (points["left_shoulder"][0] + points["right_shoulder"][0]) // 2,
            (points["left_shoulder"][1] + points["right_shoulder"][1]) // 2,
            1.0
        )
    if "left_hip" in points and "right_hip" in points:
        points["mid_hip"] = (
            (points["left_hip"][0] + points["right_hip"][0]) // 2,
            (points["left_hip"][1] + points["right_hip"][1]) // 2,
            1.0
        )

    # 5. Draw Skeleton (Green)
    for a, b in SKELETON:
        if a in points and b in points and points[a][2] > 0.3 and points[b][2] > 0.3:
            cv2.line(frame, points[a][:2], points[b][:2], (0, 255, 0), 3, cv2.LINE_AA)
    
    # 6. Draw Joints (Orange normally, RED with halo if at risk)
    for name, (x, y, conf) in points.items():
        if conf > 0.3:
            if name in risk_joints:
                # Draw glowing red halo for injury risk area
                cv2.circle(frame, (x, y), 18, (0, 0, 255), -1) # Outer red halo
                cv2.circle(frame, (x, y), 8, (255, 255, 255), -1) # Inner white dot for contrast
            else:
                # Normal joint
                cv2.circle(frame, (x, y), 6, (0, 165, 255), -1, cv2.LINE_AA)

    # 7. Draw Info Panel (Right side)
    panel_width = 350
    panel = np.zeros((frame.shape[0], panel_width, 3), dtype=np.uint8)
    panel[:] = (24, 24, 24)
    
    cv2.putText(panel, "RELEASE FRAME", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, f"Score: {tech_score}/100", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 2)
    
    # Add Legend
    y = 130
    cv2.putText(panel, "LEGEND:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 25
    cv2.circle(panel, (30, y-5), 5, (0, 165, 255), -1)
    cv2.putText(panel, "Normal Joint", (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    y += 25
    cv2.circle(panel, (30, y-5), 8, (0, 0, 255), -1)
    cv2.putText(panel, "Injury Risk Area", (45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

    y = 200
    cv2.putText(panel, "BIOMECHANICS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    y += 30
    
    metrics = ["elbowAngle", "frontKneeAngle", "trunkForwardFlexion", "hipShoulderSeparation"]
    labels = ["Elbow Angle", "Front Knee", "Trunk Flex", "Hip-Shoulder Sep"]
    
    for label, key in zip(labels, metrics):
        val = data.get("measurements", {}).get(key, "--")
        cv2.putText(panel, f"{label}:", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, str(val), (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        y += 35

    # 8. Combine and Save
    final_image = np.hstack((frame, panel))
    
    # Resize if too large
    h, w = final_image.shape[:2]
    if w > 800:
        scale = 800 / w
        final_image = cv2.resize(final_image, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    print(f"Release image with risk highlights saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video")
    parser.add_argument("json", help="Input JSON")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--arm", default="right")
    args = parser.parse_args()
    
    draw_release_frame(args.video, args.json, args.output, args.arm)
