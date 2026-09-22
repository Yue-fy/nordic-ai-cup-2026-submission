"""(1+lambda) evolution strategy over the rule-policy parameters, using common random seeds.

    python -m cup.survival.search --output runs/search-X --procs 32 --lam 8 --seeds-per-eval 4 --hours 4

Each generation samples lam perturbations of the incumbent, evaluates every candidate on the same K seeds
(full official games), and adopts a candidate if its mean score beats the incumbent's on those seeds.
The seed set rotates every --rotate-every generations; a candidate replaces the incumbent only if it beats it by
--accept-margin on the same seeds (held-out check on 2026-09-17 showed 4 seeds + zero margin overfits badly). Results: generations.jsonl, best.json.
"""
import argparse, json, os, random, sys, time
import multiprocessing as mp
from pathlib import Path

def main():
    from cup.survival.rules import DEFAULT_PARAMS, sample_params
    from cup.survival.run_games import play
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True); ap.add_argument('--init')
    ap.add_argument('--procs', type=int, default=max(1, min(32, os.cpu_count() or 1)))
    ap.add_argument('--lam', type=int, default=8); ap.add_argument('--seeds-per-eval', type=int, default=4)
    ap.add_argument('--sigma', type=float, default=0.3); ap.add_argument('--hours', type=float, default=4.0)
    ap.add_argument('--seed', type=int, default=12345); ap.add_argument('--steps', type=int, default=30001)
    ap.add_argument('--rotate-every', type=int, default=6); ap.add_argument('--accept-margin', type=float, default=40.0)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    incumbent = dict(DEFAULT_PARAMS)
    if args.init:
        incumbent.update(json.load(open(args.init)))
    ctx = mp.get_context('spawn'); pool = ctx.Pool(args.procs)
    deadline = time.time() + args.hours * 3600; gen = 0; inc_score = None; seeds = None
    def evaluate(cands, seeds):
        jobs = [(s, c, args.steps, 0) for c in cands for s in seeds]
        res = pool.map(play, jobs)
        per = []
        for i in range(len(cands)):
            rs = res[i * len(seeds):(i + 1) * len(seeds)]
            per.append(dict(mean_score=sum(r['score'] for r in rs) / len(rs), full=sum(r['sim_time'] >= 3000 for r in rs),
                            times=[round(r['sim_time']) for r in rs]))
        return per
    try:
        while time.time() < deadline:
            if gen % args.rotate_every == 0:
                seeds = [rng.randrange(1, 10**6) for _ in range(args.seeds_per_eval)]; inc_score = None
            cands = [sample_params(rng, incumbent, args.sigma) for _ in range(args.lam)]
            if inc_score is None:
                cands = [incumbent] + cands
            t0 = time.time(); per = evaluate(cands, seeds)
            if inc_score is None:
                inc_score = per[0]['mean_score']; cands, per = cands[1:], per[1:]
            best_i = max(range(len(per)), key=lambda i: per[i]['mean_score'])
            improved = per[best_i]['mean_score'] > inc_score + args.accept_margin
            if improved:
                incumbent = cands[best_i]; inc_score = per[best_i]['mean_score']
            record = dict(generation=gen, seeds=seeds, incumbent_score=inc_score, improved=improved,
                          candidates=[dict(score=p['mean_score'], full=p['full'], times=p['times']) for p in per],
                          wall_s=round(time.time() - t0), incumbent=incumbent)
            with (out / 'generations.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            json.dump(incumbent, open(out / 'best.json', 'w'), indent=1)
            print(json.dumps({k: v for k, v in record.items() if k != 'incumbent'}), flush=True)
            gen += 1
    finally:
        pool.close(); pool.join()

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); main()
