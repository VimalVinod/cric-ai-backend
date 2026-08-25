from phase1_engine import (
    load_pose_model,
    process_video,
    detect_all_releases
)

VIDEO_PATH = "test_data/bowling_session.mp4"
ARM = "right"

print("=" * 60)
print("MULTI-DELIVERY DETECTION TEST")
print("=" * 60)

print("\nLoading video...")

history = process_video(
    VIDEO_PATH
)

print(f"Frames detected : {len(history)}")

releases = detect_all_releases(
    history,
    ARM
)

print("\nRELEASE FRAMES")
print("-" * 60)

if not releases:
    print("No deliveries detected.")
else:
    print(f"Deliveries detected : {len(releases)}")

    for number, frame in enumerate(
        releases,
        start=1
    ):
        print(
            f"Delivery {number:<3} : "
            f"frame {frame}"
        )

print("\n" + "=" * 60)