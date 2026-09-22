"""Promotion check: candidate vs incumbent on the SAME fresh seeds, paired statistics, explicit verdict.

    python -m cup.survival.confirm --candidate runs/.../best.json [--incumbent params.json] --seeds 91001-91020 --procs 20 --output runs/x.json

Seed tiers (agreed 2026-09-17): search uses its own random exploration seeds; 90001-90020 is the development
diagnostic set (already used, do not tune on it further); 95001-95030 is the FROZEN final check, to be run once
on the frozen submission only. Use fresh confirmation seeds (e.g. 91001-91020, 92001-92020, ...) for promotions.
"""
import argparse, json, os, random, sys, time
import multiprocessing as mp
from pathlib import Path

def main():
    from cup.survival.run_games import play, parse_seeds
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', required=True); ap.add_argument('--incumbent')
    ap.add_argument('--seeds', default='91001-91020'); ap.add_argument('--procs', type=int, default=20)
    ap.add_argument('--output', required=True); ap.add_argument('--bootstrap', type=int, default=20000)
    args = ap.parse_args()
    cand = json.load(open(args.candidate)); inc = json.load(open(args.incumbent)) if args.incumbent else None
    seeds = parse_seeds(args.seeds)
    with mp.get_context('spawn').Pool(min(args.procs, 2 * len(seeds))) as pool:
        res = pool.map(play, [(s, cand, 30001, 0) for s in seeds] + [(s, inc, 30001, 0) for s in seeds])
    c, i = res[:len(seeds)], res[len(seeds):]
    diffs = [a['score'] - b['score'] for a, b in zip(c, i)]
    rng = random.Random(0); n = len(diffs)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(args.bootstrap))
    lo, hi = boots[int(0.025 * args.bootstrap)], boots[int(0.975 * args.bootstrap)]
    wins = sum(d > 0 for d in diffs)
    verdict = 'PROMOTE' if lo > 0 else ('REJECT' if hi < 0 else 'INCONCLUSIVE')
    out = dict(candidate=args.candidate, incumbent=args.incumbent or 'DEFAULT_PARAMS', seeds=seeds,
               candidate_mean=sum(r['score'] for r in c) / n, incumbent_mean=sum(r['score'] for r in i) / n,
               paired_mean_diff=sum(diffs) / n, bootstrap_ci95=[lo, hi], wins=wins, losses=n - wins, verdict=verdict,
               candidate_full=sum(r['sim_time'] >= 3000 for r in c), incumbent_full=sum(r['sim_time'] >= 3000 for r in i),
               per_seed=[dict(seed=s, candidate=a['score'], incumbent=b['score']) for s, a, b in zip(seeds, c, i)],
               machine=os.uname().nodename, date=time.strftime('%Y-%m-%d %H:%M'))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(args.output, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_seed'}))

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); main()
