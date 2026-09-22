# Decision: reject YOLO11l before public validation

Both preregistered L inference sizes completed sequentially with 25/25 offline and realtime responses, no handler failure and P95 below 93 ms. M@1280 scored `0.786868` mAP and `0.803089` IoU.5 localization.

- L@960 scored `0.759667` mAP and `0.795367` localization. Background FP@.25 improved from 8 to 5 and held-out mean correct recall rose from `0.7917` to `0.8333`, but jet plane regressed from `0.8333` to `0.75`.
- L@1280 scored `0.760314` mAP and `0.772201` localization. Held-out mean correct recall improved to `0.9167` with no represented-class regression, but both mAP and localization failed their gates; background FP rose to 9.

The exact trained checkpoint SHA-256 is `f623097255fb8167d465b0d09261a6727b059ef1f50088d3755b68a4e2beb267`. Reject both sizes and make no public submission. This capacity interpolation preserves realtime throughput and improves isolated cross-appearance classification at 1280, but loses too much complete-sequence localization and ranking. Keep M as rollback; final evaluation was not called.
