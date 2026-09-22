"""One-factor-at-a-time sweep: baseline vs single-parameter changes, all on the same seeds. Sharded for Slurm.

    python -m cup.survival.sweep --grid configs/sweep1.json --seeds 70001-70016 --procs 32 --shard 0/4 --output runs/x/shard0.json
    python -m cup.survival.sweep --merge runs/x/shard*.json      # prints the ranked table
"""
import argparse, glob, json, os, sys
import multiprocessing as mp
from pathlib import Path

def main():
    from cup.survival.run_games import play, parse_seeds
    from cup.survival.rules import DEFAULT_PARAMS
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid'); ap.add_argument('--base'); ap.add_argument('--seeds', default='70001-70016')
    ap.add_argument('--procs', type=int, default=32); ap.add_argument('--shard', default='0/1'); ap.add_argument('--output')
    ap.add_argument('--merge', nargs='*')
    args = ap.parse_args()
    if args.merge:
        rows = []; base = None
        for f in args.merge:
            d = json.load(open(f)); base = base or d['baseline']; rows += d['settings']
        bmean = sum(base['scores']) / len(base['scores'])
        rows.sort(key=lambda r: -sum(r['scores']) / len(r['scores']))
        print(f"baseline mean {bmean:.0f} on {len(base['scores'])} seeds")
        for r in rows:
            m = sum(r['scores']) / len(r['scores']); wins = sum(a > b for a, b in zip(r['scores'], base['scores']))
            print(f"{r['param']:>18s} = {r['value']:<8} mean {m:6.0f} diff {m - bmean:+6.0f} wins {wins:2d}/{len(r['scores'])} full {r['full']}")
        return
    base = dict(DEFAULT_PARAMS); base.update(json.load(open(args.base)) if args.base else {})
    grid = json.load(open(args.grid)); settings = [(k, v) for k, vals in grid.items() for v in vals if base.get(k) != v]
    i, n = map(int, args.shard.split('/')); mine = settings[i::n]
    seeds = parse_seeds(args.seeds)
    jobs = [(s, base, 30001, 0) for s in seeds] + [(s, {**base, k: v}, 30001, 0) for k, v in mine for s in seeds]
    with mp.get_context('spawn').Pool(args.procs) as pool:
        res = pool.map(play, jobs)
    def block(j): return res[j * len(seeds):(j + 1) * len(seeds)]
    out = dict(seeds=seeds, base=base, baseline=dict(scores=[r['score'] for r in block(0)], full=sum(r['sim_time'] >= 3000 for r in block(0))),
               settings=[dict(param=k, value=v, scores=[r['score'] for r in block(j + 1)], full=sum(r['sim_time'] >= 3000 for r in block(j + 1)))
                         for j, (k, v) in enumerate(mine)])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(args.output, 'w'), indent=1)
    print(json.dumps({'baseline_mean': sum(out['baseline']['scores']) / len(seeds),
                      **{f"{r['param']}={r['value']}": round(sum(r['scores']) / len(seeds)) for r in out['settings']}}, indent=1))

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); main()
