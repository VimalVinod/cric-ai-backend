    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("=" * 60)

    print()
    
    score_display = f"{technical_score}" if technical_score is not None else "--"
    print(f"Technical score : {score_display}")

    print(
        f"Release frame   : "
        f"{release_original_frame}"
    )

    print(
        f"Release time    : "
        f"{release_timestamp:.3f}s"
    )

    print(
        f"YOLO inferences : "
        f"{inference_count}"
    )

    print(
        f"Detection rate  : "
        f"{detection_rate * 100:.1f}%"
    )

    print(
        f"Risk indicators : "
        f"{len(risks)}"
    )
