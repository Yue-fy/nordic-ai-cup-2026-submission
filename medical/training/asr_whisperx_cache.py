"""Transcribe the 39 official conversations with WhisperX (faster-whisper large-v3 + wav2vec2 word alignment)
and write the shared ASR cache described in handoff/INTERFACES.md section 2.

    python cup/medical/triton/asr_whisperx_cache.py [--model large-v3] [--limit N]
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('HF_HOME', str(ROOT / 'models' / 'hf')); os.environ.setdefault('TORCH_HOME', str(ROOT / 'models' / 'torch'))
AUDIO = ROOT / 'upstream' / 'medical-appointment' / 'data' / 'audio'

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', default='large-v3'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=16); ap.add_argument('--compute-type', default='float16')
    args = ap.parse_args()
    import torch, whisperx
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tag = f'whisperx-{args.model}+wav2vec2-base-960h'
    out_dir = ROOT / 'cup' / 'medical' / 'asr_cache' / tag; out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    t0 = time.time(); model = whisperx.load_model(args.model, device, compute_type=args.compute_type, language='en')
    align_model, metadata = whisperx.load_align_model(language_code='en', device=device)
    print(f'models loaded in {time.time()-t0:.1f}s on {device}', flush=True)
    files = sorted(AUDIO.glob('*.mp3'))[: args.limit or None]
    timings = []
    for f in files:
        raw = f.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        target = out_dir / f'{sha[:16]}.json'
        if target.exists():
            print('cached', f.name); continue
        t1 = time.time(); audio = whisperx.load_audio(str(f))
        result = model.transcribe(audio, batch_size=args.batch_size, language='en')
        t2 = time.time()
        aligned = whisperx.align(result['segments'], align_model, metadata, audio, device, return_char_alignments=False)
        t3 = time.time()
        words = []; last_end = 0.0
        for seg in aligned['segments']:
            for w in seg.get('words', []):
                rec = {'i': len(words), 'text': w['word'].strip()}
                if 'start' in w and 'end' in w:
                    rec['start'] = round(float(w['start']), 3); rec['end'] = round(float(w['end']), 3); last_end = rec['end']
                else:
                    rec['start'] = rec['end'] = round(last_end, 3); rec['interp'] = True
                if 'score' in w: rec['score'] = round(float(w['score']), 3)
                words.append(rec)
        # interpolate missing timestamps between neighbours
        for k, w in enumerate(words):
            if w.get('interp'):
                nxt = next((x for x in words[k+1:] if not x.get('interp')), None)
                prv = next((x for x in reversed(words[:k]) if not x.get('interp')), None)
                if prv and nxt:
                    w['start'] = round((prv['end'] + nxt['start']) / 2 - 0.05, 3); w['end'] = round(w['start'] + 0.1, 3)
        doc = {'audio_filename': f.name, 'sha256': sha, 'model': tag, 'revision': getattr(whisperx, '__version__', 'unknown'),
               'duration_s': round(len(audio) / 16000, 2), 'language': result.get('language', 'en'),
               'words': words,
               'segments': [{'start': round(float(s['start']), 3), 'end': round(float(s['end']), 3), 'text': s['text'].strip()} for s in aligned['segments']],
               'machine': os.uname().nodename, 'git_commit': commit, 'date': time.strftime('%Y-%m-%d'),
               'timing_s': {'transcribe': round(t2 - t1, 2), 'align': round(t3 - t2, 2)}}
        tmp = target.with_suffix('.json.tmp'); tmp.write_text(json.dumps(doc, ensure_ascii=False)); os.replace(tmp, target)
        timings.append((doc['duration_s'], t3 - t1))
        print(f"{f.name}: {doc['duration_s']:.0f}s audio, {len(words)} words, transcribe {t2-t1:.1f}s align {t3-t2:.1f}s", flush=True)
    if timings:
        print('mean seconds of compute per second of audio:', round(sum(t for _, t in timings) / sum(d for d, _ in timings), 3))

if __name__ == '__main__':
    main()
