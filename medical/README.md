# Medical Appointment

`bundle/` is the frozen release that was served, byte for byte
(`bundle/BUILD.json` lists every file with its SHA-256; `bundle/preflight.py`
verifies them and the offline model assets before the service starts).
`medical_codex` is simply the internal package name of the runtime modules.

## Method

1. **ASR.** WhisperX large-v3 with wav2vec2 forced alignment gives word-level
   timestamps (`bundle/cup/medical/asr.py`). A constant −0.1 s shift corrects
   the alignment offset measured on the development set.
2. **Yes/no answers.** Qwen3-8B (BF16, non-thinking) is asked all questions
   with an answer-first prompt that also returns a supporting quote
   (`bundle/cup/medical/qa.py`). A question is answered *yes* when the model's
   probability exceeds a threshold tuned on development data; development
   accuracy 0.992.
3. **Evidence span.** The quote is fuzzy-matched to the transcript words and
   expanded to the enclosing clause (`bundle/cup/medical/triton/localize.py`).
4. **Span refinement.** A RoBERTa-base extractive locator
   (`deepset/roberta-base-squad2`, fine-tuned 5 epochs on the supplied positive
   spans with a 99th-percentile length cap; weights in
   `bundle/models/medical/active-locator/`) proposes an alternative span. A
   conversation-nested Ridge selector (`bundle/medical_codex/runtime/`) with a
   one-standard-error threshold replaces the clause span only when the
   predicted gain is large; otherwise the clause span is kept. This
   conservative replacement rule was the only refinement that transferred from
   development to public validation.

Development score (conversation-grouped, fully nested): 0.766.
Public validation: 0.744.

## Run

Download `Qwen/Qwen3-8B` (revision in `bundle/candidate.json`) and
`Systran/faster-whisper-large-v3` into a local Hugging Face cache, then:

```bash
cd bundle
pip install -r requirements.txt
MEDICAL_QA_MODEL=/path/to/Qwen3-8B HF_HOME=/path/to/hf-cache MEDICAL_PORT=9054 ./run.sh
```

`run.sh` runs the preflight (asset hashes, offline mode), warms the models and
starts `cup.medical.server:app` on the official protocol.

## Reproducing the locator and selector

`training/` contains the scripts that produced the shipped artifacts. Inputs
are the organiser's supplied development set (`upstream/medical-appointment/data/audio`
and `question_train.csv`).

| Script | Purpose |
|---|---|
| `asr_whisperx_cache.py` | Transcribe the development audio once (WhisperX large-v3 + wav2vec2 alignment) into a word-level cache used by all experiments |
| `qa_eval.py` | Development evaluation of the Qwen3-8B answer prompt and clause-based evidence (accuracy, tIoU, combined score) |
| `evidence_oracle.py` | Upper bound of the evidence localization given the transcript |
| `finetune_qa_span_cv.py`, `extractive_qa_span.py`, `model_regularization.py` | Grouped cross-validation fine-tuning of the RoBERTa span locator |
| `nested_locator_selector_cv.py`, `nested_selector_audit.py` | Fully nested locator + selector evaluation (the 0.766 development figure) |
| `train_qa_span_full.py` | Train the production locator on all supplied positives (5 epochs, q99 span cap) → `models/medical/active-locator` |
| `export_conservative_selector.py` | Fit the one-standard-error Ridge gate from out-of-fold locator predictions → `medical_codex/artifacts/active_selector.json` |
| `structured_tiou_locator_cv.py` | Late alternative locator objective; rejected on public validation |

The scripts were run from the team workspace and keep its layout assumptions
(paths under `medical_codex/` and `cup/medical/`); they are provided as the
authoritative record of the procedure.
