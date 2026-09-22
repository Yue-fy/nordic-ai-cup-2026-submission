# YOLO11l capacity interpolation preregistration

## Hypothesis

YOLO11m is the valid public baseline. YOLO11x improved the four train-excluded appearances (`0.7917 -> 0.9583` mean correct-class recall) but lost realtime frames at 1280 and regressed at 960. YOLO11l lies between these capacities and may preserve M-like throughput while retaining part of X's cross-appearance gain. This is a fixed model-capacity test on the existing data pipeline.

## Training

- Initial checkpoint: official Ultralytics v8.3.0 `yolo11l.pt` from the same release used for M and X. Record URL, AGPL-3.0 licence, download date and SHA-256 in `external/yolo11l_manifest.json` before training.
- Stage 1: `solution/data/masked_aug/data.yaml`, 40 epochs, imgsz 960, batch 8, AdamW lr0 .001, weight decay .001, cosine LR, freeze 10, seed 20261019.
- Stage 2: initialize from stage-1 best; `solution/data/refined_aug/data.yaml`, 20 epochs, imgsz 960, batch 8, AdamW lr0 .0002, weight decay .001, cosine LR, freeze 10, seed 20261020.
- No public crop, user label, incomplete frame, Helsinki evaluation result, or previous trained M/X tensor enters training. Do not choose intermediate epochs after evaluation.

## Evaluation and decision

On one GPU, compare frozen M@1280 with L@960 and L@1280 using the same calibration, active L1 camera and tracker. Require zero handler/protocol errors, 25/25 realtime responses, p95 below 250 ms, offline/realtime mAP at least M minus .005, IoU.5 localization at least M minus .01, held-out mean correct recall at least M and no represented held-out class regression, and background FP@.25 no more than M+4. Select the passing L size with higher realtime mAP.

Only a fully passing L candidate may receive a separate owned-public validation preregistration. Promote only above the valid public best `0.42247028200113684`. Roll back to M on any failure. Do not use validation positions, frame numbers, routes, fingerprints, templates or answer maps; final evaluation is forbidden.
