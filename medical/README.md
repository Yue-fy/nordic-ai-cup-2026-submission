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
