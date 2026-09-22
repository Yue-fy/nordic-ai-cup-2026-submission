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

## Reproducing the training

Two-stage fine-tuning with Ultralytics; the exact `args.yaml` and
`results.csv` of all four runs are in `training/records/`.

1. Clone the organiser's `drone-flyby` package next to `official/` so that
   `official/drone-flyby/src/helsinki/{images,annotations}` exists.
2. `python training/prepare_data.py` builds annotated crops and
   relocated-object compositions (`solution/data/helsinki_aug`).
3. `python training/prepare_masked_data.py` composes masked objects onto
   reviewed target-free backgrounds (`masked_aug`); with `--refined` it adds
   the additional reviewed instances used for stage 2 (`refined_aug`).
4. Stage 1, frozen backbone, from Ultralytics `yolo11m.pt` / `yolo11l.pt`:
   `python solution/train.py --initial external/weights/yolo11m.pt --data solution/data/masked_aug/data.yaml --epochs 50 --freeze 10 --lr 0.001 --batch 8`
   (YOLO11l: 40 epochs).
5. Stage 2 from the stage-1 `best.pt`: 20 epochs on
   `solution/data/refined_aug/data.yaml`, `--freeze 10 --lr 0.0002 --batch 8`;
   augmentation settings as recorded in `training/records/*/args.yaml`.

Random seeds are recorded in each `args.yaml`. GPU nondeterminism means a
retrained model will be close to, not identical with, the shipped weights.

`official/drone-flyby/utils.py` and `local_evaluator.py` are the organiser's
files; the service imports `utils` and the request builder from them.
