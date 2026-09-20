# Nordic AI Cup 2026 — team Hannah Family (Aalto University)

Code for the three use cases exactly as served during the final evaluation.
Each directory is self-contained: the runtime code, the frozen parameters or
weights, and a README describing the method and how to run the service.

| Use case | Directory | Method in one line | Final evaluation |
|---|---|---|---|
| Survival Simulator | `survival/` | Hand-designed rule policy (88 parameters) tuned with paired-seed simulation | 1220.1 |
| Drone Flyby | `drone/` | YOLO11m geometry + YOLO11l class evidence, size calibration, active camera, causal tracker | 0.3385 |
| Medical Appointment | `medical/` | WhisperX ASR → Qwen3-8B yes/no → RoBERTa span locator with a conservative selector | 0.7352 |

All three services expose the official `POST /predict` protocol and run
fully self-hosted. No component looks up answers by audio hash, filename,
frame index, scene coordinates or any other identity of the evaluation data;
no cloud inference is used at request time.

Large files: `drone/weights/` (77 MB) and `medical/bundle/models/` (469 MB,
the fine-tuned RoBERTa locator). Qwen3-8B and WhisperX weights are downloaded
from Hugging Face by revision (see `medical/README.md`).
