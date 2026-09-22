"""Offline question answering over an ASR cache with a local Qwen3 model, scored like the official evaluator.

    python cup/medical/triton/qa_eval.py --cache cup/medical/asr_cache/<model> --model Qwen/Qwen3-8B --out runs/triton-<date>-qa/qwen3-8b

Per conversation: one prompt with the word-indexed transcript and its 10 questions; the model returns JSON with
answer, confidence, evidence word indices and a quote for EVERY question (also for "no", so a low-confidence "no"
can be flipped to "yes" with a span). Post-hoc sweeps: yes-threshold, evidence granularity (word span vs segment).
Scores here are offline diagnostics; the live service must reproduce the pipeline end to end.
"""
import argparse, csv, json, os, re, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'hf'))
DATA = ROOT / 'upstream' / 'medical-appointment' / 'data'

SYSTEM_V1 = ("You are a meticulous clinical documentation auditor. You answer yes/no questions about a doctor-patient "
             "conversation strictly from its transcript. A question is 'yes' only if the transcript explicitly states it; "
             "check drug names, doses, units, durations, dates, who said it, negations and the final agreed plan. Near-miss "
             "statements (a different dose, a different drug, a plan that was discussed but rejected) are 'no'.")
SYSTEM_V2 = ("You are a meticulous clinical documentation auditor. You answer yes/no questions about a doctor-patient "
             "conversation strictly from its transcript. A question is 'yes' only if the transcript explicitly states the SAME "
             "fact: same drug, same dose and unit, same duration, same date, same person, same polarity. If the transcript says "
             "the opposite, a different value, or does not mention it, the answer is 'no'. Most questions are traps built from "
             "near-miss changes, so compare the question against the quote word by word before answering.")
SYSTEM_V3 = ("You are an evidence retrieval specialist for clinical conversations. For each yes/no question, search the whole "
             "transcript for every plausible mention, compare drug, dose, unit, duration, date, person, negation and whether a "
             "proposal became the final plan, then select the one passage that most directly proves or refutes the exact claim. "
             "The evidence must be verbatim and self-contained. Never prefer a nearby lexical match over an exact factual match.")
PROMPT = 'v1'
SYSTEM = SYSTEM_V1
FEWSHOT = Path(__file__).with_name('fewshot_examples.json')

def fewshot_block():
    import json as _j
    ex = _j.load(open(FEWSHOT))
    lines = [f'  Q: {e["question"]}\n  evidence quote: "{e["evidence"]}"' for e in ex]
    return ("Examples of the evidence granularity expected (a complete clause, from one punctuation mark to the next, "
            "that states the fact asked about):\n" + '\n'.join(lines) + "\n\n")

def build_prompt(words, questions):
    text = ' '.join(f"[{w['i']}]{w['text']}" for w in words)
    qs = '\n'.join(f'Q{k+1}: {q}' for k, q in enumerate(questions))
    head = f"Transcript with word indices:\n{text}\n\nQuestions:\n{qs}\n\n"
    if PROMPT == 'v2':
        return head + ("For EVERY question return one JSON object with these keys IN THIS ORDER: q (1-based number); "
            "quote (the exact transcript words, 3-15 words, of the single clause most relevant to the question, copied verbatim); "
            "start and end (word indices of the first and last word of that quote); "
            "question_claims (the specific facts the question asserts); transcript_says (what the quote says about them); "
            "verdict (SAME, DIFFERENT_VALUE, OPPOSITE, NOT_MENTIONED); answer (true only if verdict is SAME); "
            f"confidence (0-1 probability that the answer is yes). Output a JSON array of exactly {len(questions)} objects and nothing else.")
    if PROMPT == 'v3':
        return head + ("For EVERY question silently retrieve all plausible mentions before choosing one. Return one JSON object with "
            "these keys IN THIS ORDER: q (1-based number); quote (a verbatim, self-contained 4-20 word passage that most directly "
            "proves or refutes the exact claim); start and end (the exact first and last word indices of quote); matched_facts "
            "(drug/action, value and unit, time, person, polarity, final-plan status); rejected_near_miss (briefly state why the closest "
            "other mention is weaker, or NONE); verdict (SAME, DIFFERENT_VALUE, OPPOSITE, NOT_MENTIONED); answer (true only for SAME); "
            f"confidence (0-1 probability of yes). Output a JSON array of exactly {len(questions)} objects and nothing else.")
    if PROMPT == 'v1c':
        head = fewshot_block() + head
    quote_rule = ("quote (copy verbatim the COMPLETE clause that states the fact, from the previous punctuation mark to the next, "
                  "typically 5-15 words; not a 2-word fragment)" if PROMPT in ('v1b', 'v1c') else
                  "quote (the words from start to end)")
    return head + ("For EVERY question return one JSON object with keys: q (1-based question number), answer (true/false), "
            "confidence (0-1, how confident you are in your answer), start (index of the first word of the shortest passage that "
            "supports or best addresses the question), end (index of its last word, end >= start), " + quote_rule + ". "
            f"Answer with a JSON array of exactly {len(questions)} objects and nothing else.")

def tiou(a, b):
    if a is None or b is None: return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

def parse_json(text):
    text = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.M).strip()
    start = text.find('['); end = text.rfind(']')
    return json.loads(text[start:end + 1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--model', default='Qwen/Qwen3-8B'); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--max-new-tokens', type=int, default=2500)
    ap.add_argument('--prompt', default='v1', choices=['v1', 'v1b', 'v1c', 'v2', 'v3']); ap.add_argument('--load-8bit', action='store_true')
    args = ap.parse_args()
    global PROMPT, SYSTEM
    PROMPT = args.prompt
    SYSTEM = SYSTEM_V3 if PROMPT == 'v3' else SYSTEM_V2 if PROMPT == 'v2' else SYSTEM_V1
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = {}
    for f in Path(args.cache).glob('*.json'):
        d = json.load(open(f)); cache[d['audio_filename']] = d
    rows = list(csv.DictReader(open(DATA / 'question_train.csv')))
    convs = {}
    for r in rows:
        convs.setdefault(r['transcript_id'], []).append(r)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.load_8bit:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map='cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map='cuda')
    print(f'loaded {args.model} in {time.time()-t0:.0f}s, {torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)
    records = []; latencies = []
    for tid, qrows in sorted(convs.items())[: args.limit or None]:
        fname = f'conversation_{tid}.mp3'
        if fname not in cache: print('no cache for', fname); continue
        d = cache[fname]; words = d['words']
        questions = [r['question'] for r in qrows]
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': build_prompt(words, questions)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', enable_thinking=False).to('cuda')
        t1 = time.time()
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True); lat = time.time() - t1; latencies.append(lat)
        try:
            items = parse_json(text); parsed = {int(it['q']): it for it in items}
        except Exception as e:
            print(f'{tid}: JSON parse failed ({e}); raw head: {text[:200]!r}', flush=True); parsed = {}
        for k, r in enumerate(qrows):
            it = parsed.get(k + 1, {})
            span = None; seg_span = None
            try:
                s, e = int(it['start']), int(it['end'])
                if 0 <= s <= e < len(words):
                    span = (words[s]['start'], words[e]['end'])
                    mid = sum(span) / 2
                    segs = [g for g in d.get('segments', []) if g['start'] <= mid <= g['end']]
                    if segs: seg_span = (segs[0]['start'], segs[0]['end'])
            except Exception:
                pass
            conf = it.get('confidence'); ans = it.get('answer')
            if isinstance(ans, str): ans = {'true': True, 'false': False}.get(ans.strip().lower(), None)
            try: conf = float(conf)
            except Exception: conf = None
            # The boolean answer is the primary output. Models read 'confidence' as confidence IN THEIR ANSWER even when
            # asked for P(yes) (2026-09-17 audit: 182 'false' answers carried confidence 1.0), so derive P(yes) from both.
            if conf is None or not (0 <= conf <= 1): conf = 0.9
            if ans is True: p_yes = conf if conf >= 0.5 else 1 - conf
            elif ans is False: p_yes = 1 - conf if conf >= 0.5 else conf
            else: p_yes = 0.5
            gold = (float(r['evidence_start']), float(r['evidence_end'])) if r['question_type'] == 'positive' else None
            records.append(dict(question_id=r['question_id'], transcript_id=tid, question_type=r['question_type'], label=int(r['label']),
                                model_answer=ans if isinstance(ans, bool) else None, p_yes=round(p_yes, 3),
                                pred_start=span[0] if span else None, pred_end=span[1] if span else None,
                                seg_start=seg_span[0] if seg_span else None, seg_end=seg_span[1] if seg_span else None,
                                gold_start=gold[0] if gold else None, gold_end=gold[1] if gold else None,
                                tiou_words=tiou(gold, span) if gold else None, tiou_segment=tiou(gold, seg_span) if gold else None,
                                quote=str(it.get('quote', ''))[:400], verdict=str(it.get('verdict', ''))[:20], latency_s=round(lat, 2)))
        print(f'{tid}: {lat:.1f}s, parsed {len(parsed)}/10', flush=True)
    # scoring sweeps
    pos = [x for x in records if x['label'] == 1]; n = len(records)
    def score(th, gran):
        acc = sum((x['model_answer'] is not None) and ((x['p_yes'] >= th) == bool(x['label'])) for x in records) / n
        ious = [(x[gran] or 0.0) if x['p_yes'] >= th else 0.0 for x in pos]
        return dict(threshold=th, granularity=gran, accuracy=round(acc, 4), mean_tiou=round(sum(ious) / len(pos), 4),
                    score=round(0.4 * acc + 0.6 * sum(ious) / len(pos), 4))
    sweep = [score(th, g) for g in ('tiou_words', 'tiou_segment') for th in (0.5, 0.4, 0.3, 0.25, 0.2, 0.1)]
    by_type = {}
    for t in ('positive', 'hard_negative', 'off_topic'):
        xs = [x for x in records if x['question_type'] == t]
        by_type[t] = round(sum((x['model_answer'] is not None) and (x['model_answer'] == bool(x['label'])) for x in xs) / max(1, len(xs)), 4)
    lat = sorted(latencies)
    summary = dict(model=args.model, prompt=args.prompt, cache=args.cache, conversations=len(latencies), questions=n,
                   by_type_accuracy_bool=by_type, missing_answers=sum(1 for x in records if x['model_answer'] is None), sweep=sweep, best=max(sweep, key=lambda s: s['score']),
                   evidence_given_for_yes=sum(1 for x in records if x['model_answer'] and x['pred_start'] is not None) / max(1, sum(1 for x in records if x['model_answer'])),
                   latency_p50=round(lat[len(lat) // 2], 2) if lat else None, latency_max=round(lat[-1], 2) if lat else None,
                   gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), machine=os.uname().nodename,
                   git_commit=subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                   date=time.strftime('%Y-%m-%d %H:%M'))
    json.dump(summary, open(out / 'medical_eval.json', 'w'), indent=1)
    with open(out / 'per_question.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
    print(json.dumps(summary, indent=1))

if __name__ == '__main__':
    main()
