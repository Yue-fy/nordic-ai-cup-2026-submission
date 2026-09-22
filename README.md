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

## What can be reproduced from this repository

- **Running the evaluated services** — yes. Each directory holds the exact
  code, parameters and weights that served the final evaluation, with
  hashes. Qwen3-8B and WhisperX weights are downloaded from Hugging Face at
  the pinned revisions; the fine-tuned medical locator (474 MiB) is attached to
  the GitHub release (see `medical/bundle/models/medical/active-locator/model/DOWNLOAD.md`).
- **Re-deriving the weights and parameters** — the training and tuning code
  is included (`drone/training/`, `medical/training/`, `survival/tuning/`)
  together with the recorded training arguments and seeds. Retraining needs
  the organiser's supplied data and a GPU; results will be statistically
  equivalent rather than bit-identical.
- **How the solutions were developed**, including everything that was tried
  and rejected and how each decision was tested: `docs/DEVELOPMENT_HISTORY.md`.

Weights in this repository: `drone/weights/` (77 MB) and the medical locator
via the release asset.
