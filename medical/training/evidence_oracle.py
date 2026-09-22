"""Offline diagnostics on an ASR cache vs the gold evidence spans (upper bounds, NOT model scores).

For every gold positive question it reports the temporal IoU achievable by
  - oracle_words:   the best contiguous word span of the transcript (ceiling set by word-timestamp quality)
  - oracle_segment: the single ASR segment (sentence) with the highest IoU (ceiling of sentence-level evidence)
  - segment_at_center: the segment containing the gold midpoint (what a sentence-level picker would return)
  - words_snapped:  gold span snapped to the nearest word boundaries (timestamp error only)
and coverage: fraction of gold spans that contain at least one transcribed word.

    python cup/medical/triton/evidence_oracle.py --cache cup/medical/asr_cache/<model>
"""
import argparse, csv, hashlib, json, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'upstream' / 'medical-appointment' / 'data'

def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

def best_word_span(words, gold):
    """Max-IoU contiguous span over word boundaries; O(n^2) over words within a 20 s window of the gold span."""
    cand = [w for w in words if w['end'] > gold[0] - 20 and w['start'] < gold[1] + 20]
    best = 0.0
    for i in range(len(cand)):
        for j in range(i, len(cand)):
            v = tiou((cand[i]['start'], cand[j]['end']), gold)
            if v > best: best = v
            if cand[j]['start'] > gold[1] + 5: break
    return best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', required=True); args = ap.parse_args()
    cache = {}
    for f in Path(args.cache).glob('*.json'):
        d = json.load(open(f)); cache[d['audio_filename']] = d
    rows = list(csv.DictReader(open(DATA / 'question_train.csv')))
    stats = {k: [] for k in ('oracle_words', 'oracle_segment', 'segment_at_center', 'words_snapped')}
    covered = 0; n = 0; missing = set(); durations = []
    for r in rows:
        if r['question_type'] != 'positive': continue
        fname = f"conversation_{r['transcript_id']}.mp3"
        if fname not in cache: missing.add(fname); continue
        d = cache[fname]; gold = (float(r['evidence_start']), float(r['evidence_end'])); n += 1; durations.append(gold[1] - gold[0])
        words = [w for w in d['words'] if not w.get('interp')]
        inside = [w for w in words if w['start'] >= gold[0] - 0.05 and w['end'] <= gold[1] + 0.05]
        covered += bool(inside)
        stats['oracle_words'].append(best_word_span(words, gold))
        if inside: stats['words_snapped'].append(tiou((inside[0]['start'], inside[-1]['end']), gold))
        else: stats['words_snapped'].append(0.0)
        segs = d.get('segments') or []
        if segs:
            stats['oracle_segment'].append(max(tiou((s['start'], s['end']), gold) for s in segs))
            mid = sum(gold) / 2
            at = [s for s in segs if s['start'] <= mid <= s['end']]
            stats['segment_at_center'].append(tiou((at[0]['start'], at[0]['end']), gold) if at else 0.0)
    out = dict(cache=args.cache, conversations=len(cache), gold_positives=n, missing_conversations=sorted(missing),
               coverage=covered / max(1, n), gold_duration_median=statistics.median(durations) if durations else None,
               **{k: dict(mean=round(statistics.mean(v), 4), median=round(statistics.median(v), 4)) for k, v in stats.items() if v})
    print(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
