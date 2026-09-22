# H: pretrained medium-model feature capacity

Registered before training/public measurement on2026-09-18. E small frozen model improved the full-view public score from D0.07439 to0.21090. Test whether a larger frozen feature extractor generalises further, without adding unreviewed data or any scene information.

- Public COCO YOLO11m initial weights; freeze10,50epochs,seed20260922,batch8. Same original masked_aug data as E, not G's newly augmented data, so this is a capacity/seed/batch control relative to E. All parameters/hashes saved.
- Helsinki diagnostic remains same-instance. Four synthesis-held-out physical instances/48fixtures provide limited4class diagnosis, not novel-mesh evidence. Public validation is development evidence.
- Existing scale rule selects among960,1280,1536,1920 with p95inference<180ms and gain>=.02 over960. Local offline/realtimeAP>=.20,HTTPp95<250ms,max<3333ms,zero errors/timeouts/invalidcommands.
- Public acceptance: exact URL/errors[], score>=E+.02; additionally compare best observed candidates when choosing deployment. Goal>=0.743 remains unchanged. Rollback to E if latency or accuracy fails. No final evaluation.
- Save checkpoints every epoch and resume if scheduled time expires. Public only after full intended training and local gates.
