# Development history

How the three submitted systems were built between 2026-09-17 and 2026-09-20,
what was tried and rejected, and how each decision was tested. Every number
below comes from a recorded experiment; the raw decision records
(pre-registrations, gate results, arena JSONs) are kept by the team and can be
provided on request.

The same working rule applied to all three challenges: a change was accepted
only if it beat the incumbent on held-out data under a pre-registered
criterion, and nothing that identifies the evaluation data (audio hashes,
filenames, frame indices, scene coordinates, stored answers) was allowed into a
candidate.

---

## 1. Survival Simulator — final evaluation 1220.06 (three-game mean)

**Method.** A deterministic rule policy over the official observations
(`survival/cup/survival/rules.py`) with 88 parameters. It was developed
incrementally; every increment was compared against the incumbent on paired
seeds in the official simulator, with bootstrap confidence intervals on the
paired difference. The simulator is nondeterministic across processes (set
iteration order), so single games are never compared; the unit is a paired
seed set of 100 to 3000 games.

| Version | Change | Evidence (paired games vs previous) | Local mean |
|---|---|---|---|
| v2 (09-17) | Spacing between agents, birth interval, hunger threshold | 16 seeds vs v1: 884 → 1004 | ~1000 |
| v3 | **Deathbed reproduction**: detect the old-age energy drain from unexplained energy loss and reproduce immediately, ignoring the population cap | 4 × 200 games: +68, +66, +66, +83; all CIs exclude 0 | 995 |
| v4 | **Selective breeding**: spawn permission in trait-score order, top 75 % only; speed-weighted traits; walk away from a predator once walking speed beats its sprint | 256 shared seeds: v2 937 → v3 995 → v4 1056; +119 over v2, CI [77, 161] | 1056 |
| v5 | Evolution-strategy search winner over continuous parameters | +54, CI [12, 96] | 1105 |
| v6 | Second search winner (two independent 200-game replications +54 / +62) | +63, CI [24, 102] | 1131 |
| v7 | Spawn threshold 150 with a late-game ramp (time-ramped parameters) | +10 over 7000 paired games | |
| v8 (final) | Birth interval 20 on top of v7 | **+26 over v6, z = 4.1, CI [14, 38], n = 3000** | **1145 ± 292 per game** |

**Diagnostics that drove the design.** Lifetime and encounter logging over
whole games showed that after t = 300 s, sprinting and evasion consumed
50 to 60 % of all energy; predator deaths and old-age deaths were about equally
common, and 77 % of predation victims were childless; food was visible 80 to
90 % of the time, so exploration was not the bottleneck. Those three facts
motivated deathbed reproduction, selective breeding and the walk-away rule.
The selective-breeding strength showed a clean dose response (quantile 50: +48,
75: +66, 85: −15, 92: −49), which is what made that mechanism credible.

**Rejected, with the measured effect.** Parameter combinations from a
one-factor sweep (−83), evolution-search snapshots taken without fresh-seed
confirmation (−73, −25), a predator-cone stealth rule (−32 / −54, retested
later after evolved vision exceeded the predator's range: still no gain),
hierarchical PPO over the rule modes (checkpoints 50 / 100 / 200: −26 / −261 /
−7; the shared team reward gives no per-agent credit), tree relay and dead-
reckoned tree memory (−12 to −27), facing or relaying predators (~0), a lower
flee speed (−13 to −32), spawning only near fruit (−33), stacking every
mechanism at once (−40), a decoy role, edge-aware fleeing, skipping
predator-guarded fruit.

**Methodological findings.** (1) An A/A control (a config byte-identical to
the incumbent) returned +2.7 ± 3.8 over 3000 games, so the paired arena is
unbiased. (2) 25.6 % of games diverge between two runs of the same policy on
the same seed, so replay-based counterfactuals are unreliable; a same-process
state-fork A/A also failed (about 20 % divergence, noise s.d. 123), which is
why per-step advantage labels and any learned controller trained on them were
abandoned. (3) A single public validation is one game (s.d. about 292); the
evaluation is a three-game mean (s.e. about 167). Candidates were therefore
selected on local paired arenas, never on public validation draws. The
reproduction scripts are in `survival/tuning/`; the version lineage is in
`survival/params/lineage/`.

---

## 2. Drone Flyby — final evaluation 0.3385

**Method.** YOLO11m fine-tuned on the official Helsinki frames supplies box
geometry; YOLO11l, trained the same way, runs on every frame as an independent
class-evidence model (a box keeps its class only if a teacher box agrees at
IoU ≥ 0.5 with confidence ≥ 0.25, otherwise its confidence is halved). Per-class
physical size priors calibrate confidences. The camera holds the Level-1 view
and resets to the full-frame Level-0 view every 8 frames; a causal tracker with
image-based motion estimation (phase correlation) carries detections between
frames. Outputs are in full source-frame coordinates; only current and past
frames are used.

**Training recipe** (exact Ultralytics `args.yaml` and `results.csv` for all
four runs are in `drone/training/records/`):

1. Data: `prepare_data.py` makes annotated random crops and relocated-object
   compositions from the official Helsinki frames; `prepare_masked_data.py`
   composes masked objects onto reviewed target-free backgrounds
   (`masked_aug`, 1200 synthetic images) and, with `--refined`, adds
   additional reviewed instances (`refined_aug`, 2000 train / 55 val).
2. Stage 1: from Ultralytics `yolo11m.pt` / `yolo11l.pt`, frozen backbone
   (`freeze 10`), image size 960, AdamW, lr 0.001, 50 / 40 epochs on
   `masked_aug`.
3. Stage 2: from the stage-1 checkpoint, 20 epochs on `refined_aug`, lr
   0.0002, mosaic 0.4, rotation ±180°, flips, HSV-V 0.45, scale 0.5.

**How the design was chosen.** Public validation scores of the same detector
under different camera policies: full Level-0 view 0.278, active Level-1
camera with periodic resets 0.380 (accepted). YOLO11m refined replaced the
smaller model at 0.314 full-view against a pre-registered threshold of 0.298.
Adding YOLO11l as class teacher gave 0.4467 on the public validation scene
(re-run of the identical configuration: 0.4427, so the public noise floor is
about 0.004).

**Rejected, with evidence.** A P2 high-resolution head, crop-edge
preservation, corrected-box labels (0.2057 vs 0.2031, no gain), RT-DETR-L
(failed the component and real-time gates), YOLO11x, DINOv2 crop re-labelling,
frozen ConvNeXt / SigLIP2 / OpenCLIP crop classifiers, YOLO-World open-
vocabulary classes, FastSAM object discovery, synthetic objects rendered from
3D assets, consensus and distilled students, classification-head-only
adaptations, hard-negative refresh, median motion fallback, longer track ages,
an object-directed scout camera, a hybrid detail camera, and an appearance
teacher trained on 99 newly reviewed public crops (macro gain 0.006 against a
0.08 gate). Nine further candidates validated on the final night all lost to
the incumbent.

**Why the score is where it is.** A 54-object hand-verified audit set was
built over 246 pixel-exact reconstructed 4K frames of the public validation
sequence and scored with the official evaluator. On the incumbent's recorded
predictions: as submitted 0.110; perfect classes 0.164; zero false positives
0.190; both 0.319, which equals the class-agnostic recall of 0.341 at IoU 0.5.
Recall sets the ceiling; 62 % of ground-truth boxes were outside the camera
view when scored. Confidence thresholds, top-k, NMS and track-age settings all
moved the score by ≤ 0.0001. The final evaluation ran on a different scene
from the public validation scene (frame correlation 0.22), which is why the
evaluation score (0.339) is below the validation score (0.447): the model was
never adapted to either scene.

**Deliberately excluded.** A "validation-map fusion" candidate scored 0.565 on
public validation by remembering object coordinates of the validation scene.
It was declared ineligible and never submitted, because it cannot transfer to
an unseen sequence and is not a detection method.

---

## 3. Medical Appointment — final evaluation 0.7352

**Method.** WhisperX large-v3 with wav2vec2 forced alignment gives word
timestamps; Qwen3-8B (BF16, non-thinking) answers all ten questions of a
conversation with an answer-first prompt that also returns a supporting quote;
the quote is fuzzy-matched to the transcript and expanded to the enclosing
clause; a RoBERTa-base span locator fine-tuned on the supplied positive spans
proposes an alternative span, and a conversation-nested Ridge selector with a
one-standard-error threshold replaces the clause span only when the predicted
gain is large.

**Development, in order.**

| Step | Result on the 39 supplied conversations (grouped cross-validation) |
|---|---|
| Qwen3-8B yes/no, scoring on the model's "confidence" field | accuracy 0.967, score 0.63–0.69; the confidence field was found to be confidence-in-answer, not P(yes) |
| Prompt v1b: answer first, quote a complete clause | accuracy 0.992, zero JSON failures |
| Evidence = quote → words → clause expansion, −0.1 s alignment shift | tIoU 0.40 (word index) → 0.58 (clause); combined 0.743 |
| Qwen3-14B instead of 8B | no improvement |
| Zero-shot RoBERTa QA span | 0.666 (worse than the clause rule) |
| RoBERTa locator, 2 epochs, grouped OOF | 0.741 |
| Locator + nested-CV Ridge selection gate | 0.763 |
| Fully nested, conservative gate, 99th-percentile length prior, 5 epochs | **0.766** (independent V100 repeat 0.773) |
| Enriched single-locator selector / Qwen14 agreement router | 0.791 / 0.794 on development |

**Why the final candidate is not the one with the best development score.**
Public validation of eight candidates showed that development score stopped
predicting public score above the conservative operating point (slope 0.011,
r = 0.03; r = −0.48 over the seven refined candidates). Five candidate
families that beat the incumbent on development all lost on public
validation: the agreement router 0.724, the enriched selector 0.719, a
Qwen3-32B selective-evidence candidate 0.738 (development 0.809), a
containment trim 0.725, a global time shift 0.739. The mechanism is
replacement precision: the conservative selector replaces 16 % of spans at
+0.245 tIoU each, the enriched selector replaces 42 % at −0.007 each.
Gold-span granularity shift and span mis-sizing were tested as explanations
and both falsified (the optimal resize is the identity). Public validation
noise was measured at s.d. 0.011 paired, 0.025 unpaired.

The pre-registered comparison of the top three candidates on public validation
gave conservative-5ep 0.7443, simple-2ep 0.7418, Qwen32-AWQ 0.7378; the first
was frozen. Its evaluation score, 0.7352, is within noise of its validation
score.

**Rejected.** Synthetic consultations and fold-local augmentation (no
improvement), a second-ASR residual selector (development 0.8001 but the
replacement gate failed: one of five folds lost 0.018), quantized Qwen3-14B
and 32B variants, a pretrained segment re-ranker, a structured-tIoU locator
(public 0.739 on the final morning).

**Release discipline.** The served bundle was frozen with a manifest of file
hashes and a preflight that verifies every model byte offline; it was replayed
end to end (38 conversations, live ASR, no transcript cache) on two
independent starts with byte-identical responses before it was exposed, and
re-verified through the public URL after each restart. The same bundle on a
different GPU architecture produced identical answers but slightly different
spans on 6 of 38 conversations, so acceptance hashes were always reproduced
on the GPU family that generated them.
