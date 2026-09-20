"""Evidence localization: map a model quote to word timestamps and expand to a clause.

Findings (2026-09-17, Qwen3-8B on WhisperX cache, 195 gold positives): model word indices tIoU 0.40;
quote fuzzy-matched to words 0.49; quote expanded to the punctuation-delimited clause 0.58; gold-oracle 0.90.
"""
import difflib, re
_NORM = re.compile(r'[^a-z0-9 ]')
_CLAUSE = re.compile(r'[.,;:?!]$'); _SENT = re.compile(r'[.?!]$')

def norm(s): return _NORM.sub('', s.lower())

def locate_quote(quote, words, hint_index=None, min_ratio=0.6):
    """Return (i, j) word indices best matching the quote, preferring matches near hint_index; None if not found."""
    q = norm(quote).split()
    if not q: return None
    toks = [norm(w['text']) for w in words]; n = len(q); best = None
    for i in range(len(toks)):
        for L in (n - 1, n, n + 1, n + 2):
            if L <= 0 or i + L > len(toks): continue
            r = difflib.SequenceMatcher(None, q, toks[i:i + L]).ratio()
            if r < min_ratio: continue
            key = (r, -abs(i - hint_index) if hint_index is not None else 0)
            if best is None or key > best[0]: best = (key, (i, i + L - 1))
    return best[1] if best else None

def expand(words, i, j, sentence=False):
    """Extend [i, j] to the enclosing clause (punctuation) or sentence."""
    rx = _SENT if sentence else _CLAUSE; a, b = i, j
    while a > 0 and not rx.search(words[a - 1]['text']): a -= 1
    while b < len(words) - 1 and not rx.search(words[b]['text']): b += 1
    return a, b

SHIFT_S = -0.10   # WhisperX word timestamps run ~0.1 s late vs the gold spans: 8B tIoU 0.576->0.595, 14B 0.601->0.623 (39 conv.)

def span_seconds(words, i, j, pad=0.0, shift=SHIFT_S):
    start = max(0.0, words[i]['start'] - pad + shift); end = max(start + 0.05, words[j]['end'] + pad + shift)
    return start, end

def evidence_from_answer(item, words, mode='clause'):
    """item: model JSON object with quote and optional start/end word indices. Returns (start_s, end_s) or None."""
    hint = None
    try: hint = int(item.get('start'))
    except Exception: pass
    span = locate_quote(str(item.get('quote', '')), words, hint)
    if span is None:
        try:
            s, e = int(item['start']), int(item['end'])
            if 0 <= s <= e < len(words): span = (s, e)
        except Exception:
            return None
    i, j = span
    if mode == 'clause': i, j = expand(words, i, j)
    elif mode == 'sentence': i, j = expand(words, i, j, sentence=True)
    return span_seconds(words, i, j)
