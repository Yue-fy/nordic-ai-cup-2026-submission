"""CMA-ES over the rule parameters, one Slurm array per generation, with common random numbers.

Why this and not the (1+lambda) search: with ~40 continuous parameters and a per-game std of ~274, independent
log-normal perturbations waste most of their samples on directions that do not matter, and an accepted step tells
you nothing about which direction to keep going. CMA-ES estimates the covariance of the useful directions from the
ranking of each generation, so correlated moves (for example population cap together with birth interval) become a
single search direction. Every candidate of a generation plays the SAME seeds (common random numbers), which removes
most of the map-to-map variance from the ranking; only the ranking is used, so absolute noise matters less.
Any winner still has to pass the 200-game paired arena twice before it is deployed.

    python -m cup.survival.cmaes_driver --init configs/survival_v6_deployed.json --out runs/cma-A --popsize 12 --games 96 --gens 40
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

# Search only the continuous parameters; binary mode flags stay at the incumbent's values.
FLAGS = {'spawn_needs_food', 'follow_agents', 'river_exit', 'stealth', 'deathbed', 'deathbed_food', 'deathbed_cap',
         'relay_trees', 'relay_preds', 'face_pred', 'breed_order', 'breed_select', 'breed_deathbed', 'breed_elite',
         'breed_concave', 'spacing_vision', 'walk_vision', 'flee_smart', 'biome_flee', 'edge_flee', 'decoy'}

# A parameter that a switched-off flag never reads is pure noise to the search, so it is excluded.
INERT_IF = {
    'cone_margin': 'stealth', 'hear_margin': 'stealth',
    'pred_avoid_angle': 'pred_mem_steps', 'spacing_max': 'spacing_vision',
    'breed_elite_frac': 'breed_elite', 'breed_elite_energy': 'breed_elite',
    'edge_flee_dist': 'edge_flee', 'biome_flee_rate': 'biome_flee',
    'decoy_hold_dist': 'decoy', 'decoy_min_energy': 'decoy', 'decoy_frac': 'decoy',
    'deathbed_margin': 'deathbed', 'drain_thresh': 'deathbed',
    'breed_quantile': 'breed_select',
}
ZERO_INERT = ('tree_mem_steps', 'pred_mem_steps', 'flee_speed', 'spawn_fruit_dist', 'ripe_if_alone',
              'fruit_pred_gap', 'fruit_first_dist', 'pop_per_tree_seen', 'hungry_rel', 'late_t')

def active_keys(base, bounds):
    """Parameters the incumbent actually reads, given which flags are on."""
    keys = []
    for k in sorted(bounds):
        if k in FLAGS: continue
        g = INERT_IF.get(k)
        if g is not None and base.get(g, 0.0) <= (0.5 if g in FLAGS else 0.0): continue
        if k in ZERO_INERT and base.get(k, 0.0) == 0.0 and k != 'late_t': continue
        if k == 'late_t' and all(base.get(m, 1.0) == 1.0 for m in
                                 ('late_danger', 'late_explore', 'late_hungry', 'late_spawn', 'late_wspeed', 'late_scan')):
            continue
        keys.append(k)
    return keys

def encode(params, keys, bounds):
    """Log scale wherever the bounds are positive, so one CMA-ES step means the same RELATIVE change for every
    parameter. On a plain [0,1] box a single step moves pop_half (150-3000) by half its value and scan_turn by a
    quarter of its, which makes a common step size meaningless."""
    import math
    out = []
    for k in keys:
        lo, hi = bounds[k]; v = min(max(params[k], lo), hi)
        if lo > 0: out.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)))
        else: out.append((v - lo) / (hi - lo))
    return out

def decode(x, keys, bounds, base):
    import math
    p = dict(base)
    for k, xi in zip(keys, x):
        lo, hi = bounds[k]; u = min(max(xi, 0.0), 1.0)
        p[k] = float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if lo > 0 else lo + u * (hi - lo))
    return p

def submit(gd, n, seeds, partition, cpus, mem='48G', tlimit='02:00:00'):
    sp = gd / 'gen.sbatch'
    sp.write_text(f"""#!/bin/bash
#SBATCH --job-name=cma-{gd.name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={tlimit}
#SBATCH --array=0-{n - 1}
#SBATCH --output={gd}/task-%a.out
cd <workspace>/nordic-ai-cup
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m cup.survival.run_games --params {gd}/cand_$SLURM_ARRAY_TASK_ID.json --seeds {seeds} --procs $SLURM_CPUS_PER_TASK --log-every 0 --output {gd}/res_$SLURM_ARRAY_TASK_ID.json
""")
    return subprocess.run(['sbatch', str(sp)], capture_output=True, text=True, check=True).stdout.strip().split()[-1]

def wait(jid, gd, n, poll=45):
    while True:
        done = sum((gd / f'res_{i}.json').exists() for i in range(n))
        q = subprocess.run(['squeue', '-h', '-j', jid], capture_output=True, text=True).stdout.strip()
        if done == n or not q: return done
        time.sleep(poll)

def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cup.survival.rules import DEFAULT_PARAMS, PARAM_BOUNDS
    import cma
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--popsize', type=int, default=12); ap.add_argument('--games', type=int, default=96)
    ap.add_argument('--gens', type=int, default=40); ap.add_argument('--sigma0', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=17); ap.add_argument('--seed-base', type=int, default=700000)
    ap.add_argument('--partition', default='batch-csl,batch-skl'); ap.add_argument('--cpus', type=int, default=32)
    ap.add_argument('--mem', default='48G'); ap.add_argument('--tlimit', default='02:00:00')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    base = dict(DEFAULT_PARAMS); base.update(json.load(open(a.init)))
    keys = active_keys(base, PARAM_BOUNDS)
    print(f'searching {len(keys)} active parameters: {keys}', flush=True)
    x0 = encode(base, keys, PARAM_BOUNDS)
    es = cma.CMAEvolutionStrategy(x0, a.sigma0, {'popsize': a.popsize, 'seed': a.seed, 'bounds': [0, 1], 'verbose': -9})
    json.dump({'keys': keys, 'init': a.init}, open(out / 'space.json', 'w'), indent=1)
    best = (None, -1e18)
    for g in range(a.gens):
        gd = out / f'gen{g:02d}'; gd.mkdir(exist_ok=True)
        xs = es.ask()
        cands = [decode(x, keys, PARAM_BOUNDS, base) for x in xs]
        for i, c in enumerate(cands): json.dump(c, open(gd / f'cand_{i}.json', 'w'), indent=1)
        # the incumbent rides along on the same seeds, so progress can be read across generations despite the
        # seed set changing every generation (the ranking CMA-ES uses is unaffected either way)
        json.dump(base, open(gd / f'cand_{len(cands)}.json', 'w'), indent=1)
        s0 = a.seed_base + g * 1000; seeds = f'{s0}-{s0 + a.games - 1}'     # same seeds for every candidate of this generation
        jid = submit(gd, len(cands) + 1, seeds, a.partition, a.cpus, a.mem, a.tlimit); t0 = time.time()
        print(f'gen {g}: job {jid}, {len(cands)} candidates x {a.games} common seeds', flush=True)
        wait(jid, gd, len(cands) + 1)
        scores = []
        for i in range(len(cands)):
            f = gd / f'res_{i}.json'
            scores.append(json.load(open(f))['summary']['mean_score'] if f.exists() else None)
        rf = gd / f'res_{len(cands)}.json'
        ref = json.load(open(rf))['summary']['mean_score'] if rf.exists() else None
        ok = [(x, s) for x, s in zip(xs, scores) if s is not None]
        if len(ok) < max(4, len(cands) // 2):
            print('too many missing results, stopping', flush=True); break
        med = sorted(s for _, s in ok)[len(ok) // 2]
        es.tell([x for x, _ in ok], [-s for _, s in ok])                    # CMA-ES minimises
        gb = max(ok, key=lambda t: t[1])
        if gb[1] > best[1]: best = (decode(gb[0], keys, PARAM_BOUNDS, base), gb[1]); json.dump(best[0], open(out / 'best.json', 'w'), indent=1)
        rec = dict(gen=g, job=jid, seeds=seeds, n=len(ok), best=round(gb[1]), median=round(med),
                   worst=round(min(s for _, s in ok)), incumbent=None if ref is None else round(ref),
                   best_vs_incumbent=None if ref is None else round(gb[1] - ref),
                   median_vs_incumbent=None if ref is None else round(med - ref),
                   sigma=round(es.sigma, 4), wall_s=round(time.time() - t0), best_ever=round(best[1]))
        with (out / 'generations.jsonl').open('a') as f: f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)
        json.dump(decode(es.result.xfavorite, keys, PARAM_BOUNDS, base), open(out / 'mean.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
