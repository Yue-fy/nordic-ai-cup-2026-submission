# Drone Flyby

## Method

`solution/serve.py` runs the official protocol; the detection logic is in
`solution/active_detector.py` and `solution/detector.py`.

- **Geometry model.** YOLO11m (`weights/yolo11m_primary.pt`) fine-tuned on
  the official Helsinki frames with augmentation, image size 1280,
  confidence 0.02. Trained with `solution/train.py` (Ultralytics).
- **Class evidence.** YOLO11l (`weights/yolo11l_teacher.pt`) runs on every
  frame as a class teacher: a primary box keeps its class only if a teacher
  box overlaps it (IoU ≥ 0.50, confidence ≥ 0.25); otherwise the class
  confidence is halved (`--teacher-unsupported-scale 0.5`).
- **Size calibration.** Per-class physical size priors
  (`solution/size_priors.json`) rescale confidences of implausibly sized boxes.
- **Active camera.** The drone view is held at resolution Level 1 and reset to
  the full-frame Level 0 view every 8 frames (`--wide-interval 8`) so the
  whole frame is covered while small objects are seen at higher resolution.
  All outputs are converted back to full source-frame coordinates.
- **Causal tracker.** Frame-to-frame motion is estimated from the images by
  phase correlation; detections are associated to tracks, class evidence is
  accumulated per track, and tracks expire after 12 frames
  (`--track-max-age 12`). Only current and past frames are used.

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r solution/requirements.lock.txt
python solution/serve.py --log-dir runs/service --port 9053 \
  --weights weights/yolo11m_primary.pt --imgsz 1280 --confidence 0.02 --calibration \
  --active-camera --wide-interval 8 --track-max-age 12 \
  --teacher-weights weights/yolo11l_teacher.pt --teacher-imgsz 1280 --teacher-interval 1 \
  --teacher-class-only --teacher-unsupported-scale 0.5
```

`official/drone-flyby/dtos.py` is the organiser's protocol definition,
included unchanged so the service imports the authoritative DTOs.
Other camera policies present in the code (scout, flow-edge, hybrid,
acquisition) were experiments and are disabled by default.
