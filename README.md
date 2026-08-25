# AI Bowling Analysis — Day 1 Core Pipeline

This is the working `analyze_bowling(video)` core described in the spec:

```
video → pose landmarks → biomechanical features → technical score
                                                  → risk indicators
                                                  → recommendations
```

Tested end-to-end and confirmed working (see "What was actually tested" below).

## Quick start

```bash
pip install ultralytics opencv-python-headless numpy
python3 main.py path/to/your_bowling_clip.mp4 --arm right --out result.json
```

First run auto-downloads `yolov8n-pose.pt` (~6.5MB) from GitHub. A copy is
already included in `models/` so you don't need internet after that —
just make sure `main.py` points at it, or drop it in the project root.

Output is the structured JSON described in the spec: technical score,
per-parameter scores, risk indicators, recommendations.

## What was actually tested

I don't have one of your real bowling clips, so I validated the pipeline
against a public sample video of a person walking (not bowling) to confirm
every stage runs correctly end-to-end: pose detection worked on 74% of
frames, all 7 biomechanical parameters computed, the weighted score engine
produced 75.6/100, one risk indicator fired correctly (trunk lateral
flexion), and recommendations generated correctly.

**What this proves:** the pipeline mechanics work.
**What it doesn't prove:** whether the numbers are meaningful for an actual
bowling action — for that I need one of your real 3–5s clips.

**Next step:** send me a bowling video (or record one) and I'll run it
through and tune the phase/release detection against real bowling
mechanics.

## Key design decisions (and why)

- **Pose model: YOLOv8-pose, not MediaPipe.** The spec's originally
  implied MediaPipe Task API needs to download its model from Google's
  servers at runtime. YOLOv8-pose's weights are hosted on GitHub, so I
  could actually fetch and test it here. It gives 17 COCO keypoints
  (nose, shoulders, elbows, wrists, hips, knees, ankles) — enough for
  every V1 parameter. If you later want MediaPipe's 33-point model
  (better for subtle head/hand detail), swap it in `pose_estimator.py`
  without touching anything downstream — `biomechanics.py` only needs a
  `{name: (x,y,confidence)}` dict per frame.

- **Release frame detection is a placeholder heuristic** — frame of
  maximum bowling-wrist speed. This is a reasonable first pass but will
  need tuning against real footage; a proper version should also use
  front-foot-contact detection (ankle deceleration) as a second anchor
  point, per the phase list in the spec (Phase 4).

- **Reference ranges in `scoring.py` (`REFERENCE_RANGES`) are V1
  placeholders** based on general fast-bowling coaching heuristics, not a
  labelled dataset. The spec is explicit that this is fine for V1 as long
  as we don't pretend otherwise — flagged clearly in the file's docstring.
  This is exactly where the Coach Review feedback loop from the spec
  (section 27–28) should eventually feed real corrections back in.

- **Risk indicators are movement-pattern flags, not injury predictions** —
  matches the spec's explicit requirement (section 21) to never claim an
  injury probability.

## File layout

```
bowling_ai/
├── main.py                    # analyze_bowling(video) orchestrator
├── services/
│   ├── pose_estimator.py      # video -> per-frame keypoints (YOLOv8-pose)
│   ├── biomechanics.py        # keypoints -> angles/features + head stability
│   └── scoring.py             # features -> score, risk flags, recommendations
├── models/
│   └── yolov8n-pose.pt        # pretrained weights (pre-downloaded)
└── test_data/
    └── test_person.mp4        # generic person video used for pipeline validation
```

## Immediate next steps (fastest path to a usable demo)

1. **Send a real bowling clip.** Biggest unlock — lets me validate/tune
   release detection and reference ranges against your actual action.
2. **Skeleton overlay video export** — draw the keypoints back onto the
   original video (quick addition to `pose_estimator.py`, ~30 min of work).
3. **Wrap this in a minimal web UI** (per spec §31–32: React upload page
   that POSTs to a `/analyze-bowling` endpoint running this exact code) —
   I can scaffold this next once the analysis itself is validated on real
   footage, so you're not debugging two things at once.
4. **Phase segmentation** — right now only release frame is detected;
   run-up/gather/delivery-stride/front-foot-contact segmentation (spec §9)
   is the next biomechanics addition.

I intentionally built the analysis core first rather than the web app —
per your own Day-1 plan, this is the part that's genuinely hard to get
right, and everything else (UI, coach review, history) is comparatively
mechanical integration work once this is solid.
