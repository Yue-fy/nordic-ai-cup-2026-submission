"""(1+lambda) evolution strategy over the rule parameters, one Slurm array per generation, many games per candidate.

Every candidate of a generation (and the incumbent) plays the SAME fresh set of K seeds (common random numbers);
a candidate replaces the incumbent only if its mean beats the incumbent's by --margin on those seeds.
Promotion to the deployed policy is NOT done here: the winner must still pass the 200-game paired arena.

    python -m cup.survival.es_driver --init configs/arena/base_db_free.json --out runs/survival-es-20260918 --lam 15 --games 64 --gens 20
"""
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path

FLAGS = {'spawn_needs_food', 'follow_agents', 'river_exit', 'stealth', 'deathbed', 'deathbed_food', 'deathbed_cap', 'relay_trees', 'face_pred', 'relay_preds'}
FROZEN = FLAGS | {'tree_mem_steps', 'pred_mem_steps', 'pred_avoid_angle', 'pop_per_tree_seen', 'cone_margin', 'hear_margin', 'flee_speed', 'spawn_fruit_dist'}

def sample(rng, center, sigma, bounds):
    import math
    out = dict(center)
    for k, v in center.items():
        if k in FROZEN or k not in bounds: continue
        lo, hi = bounds[k]
        if rng.random() < 0.5: continue                       # perturb about half of the parameters per candidate
        out[k] = float(min(hi, max(lo, v * math.exp(rng.gauss(0, sigma)))))
    return out

def submit(gen_dir, n_tasks, seeds, partition, cpus):
    script = f"""#!/bin/bash
#SBATCH --job-name=cup-es-g{gen_dir.name}
#SBATCH --partition={partition}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --array=0-{n_tasks - 1}
#SBATCH --output={gen_dir}/task-%a.out
cd <workspace>/nordic-ai-cup
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m cup.survival.run_games --params {gen_dir}/cand_$SLURM_ARRAY_TASK_ID.json --seeds {seeds} --procs $SLURM_CPUS_PER_TASK --log-every 0 --output {gen_dir}/res_$SLURM_ARRAY_TASK_ID.json
"""
    sp = gen_dir / 'gen.sbatch'; sp.write_text(script)
    out = subprocess.run(['sbatch', str(sp)], capture_output=True, text=True, check=True).stdout
    return out.strip().split()[-1]

def wait(jobid, gen_dir, n_tasks, poll=60):
    while True:
        done = sum((gen_dir / f'res_{i}.json').exists() for i in range(n_tasks))
        q = subprocess.run(['squeue', '-h', '-j', jobid], capture_output=True, text=True).stdout.strip()
        if done == n_tasks or not q: return done
        time.sleep(poll)

def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from cup.survival.rules import DEFAULT_PARAMS, PARAM_BOUNDS
    ap = argparse.ArgumentParser(); ap.add_argument('--init', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--lam', type=int, default=15); ap.add_argument('--games', type=int, default=64); ap.add_argument('--gens', type=int, default=20)
    ap.add_argument('--sigma', type=float, default=0.2); ap.add_argument('--margin', type=float, default=35.0); ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--confirm-games', type=int, default=96); ap.add_argument('--partition', default='batch-csl,batch-skl'); ap.add_argument('--cpus', type=int, default=32); ap.add_argument('--seed-base', type=int, default=200000)
    a = ap.parse_args(); rng = random.Random(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    inc = dict(DEFAULT_PARAMS); inc.update(json.load(open(a.init)))
    hist = out / 'generations.jsonl'
    for g in range(a.gens):
        gd = out / f'gen{g:02d}'; gd.mkdir(exist_ok=True)
        cands = [inc] + [sample(rng, inc, a.sigma, PARAM_BOUNDS) for _ in range(a.lam)]
        for i, c in enumerate(cands): json.dump(c, open(gd / f'cand_{i}.json', 'w'), indent=1)
        s0 = a.seed_base + g * 1000; seeds = f'{s0}-{s0 + a.games - 1}'
        jid = submit(gd, len(cands), seeds, a.partition, a.cpus); t0 = time.time()
        print(f'gen {g}: job {jid}, {len(cands)} candidates x {a.games} games, seeds {seeds}', flush=True)
        done = wait(jid, gd, len(cands))
        res = {}
        for i in range(len(cands)):
            f = gd / f'res_{i}.json'
            if f.exists(): res[i] = json.load(open(f))['summary']['mean_score']
        if 0 not in res: print('incumbent result missing; skipping generation', flush=True); continue
        best = max((i for i in res if i > 0), key=lambda i: res[i], default=None)
        improved = best is not None and res[best] > res[0] + a.margin
        conf = None
        if improved and a.confirm_games > 0:
            # the best of lam noisy candidates is biased upwards (winner's curse): re-test it against the incumbent on
            # fresh seeds and only accept if it still wins by the margin
            cd = gd / 'confirm'; cd.mkdir(exist_ok=True)
            json.dump(inc, open(cd / 'cand_0.json', 'w'), indent=1); json.dump(cands[best], open(cd / 'cand_1.json', 'w'), indent=1)
            s1 = s0 + 500; cseeds = f'{s1}-{s1 + a.confirm_games - 1}'
            cj = submit(cd, 2, cseeds, a.partition, a.cpus); wait(cj, cd, 2)
            try:
                c0 = json.load(open(cd / 'res_0.json'))['summary']['mean_score']; c1 = json.load(open(cd / 'res_1.json'))['summary']['mean_score']
                conf = dict(seeds=cseeds, incumbent=c0, best=c1); improved = c1 > c0 + a.margin
            except Exception:
                conf = 'missing'; improved = False
        rec = dict(gen=g, job=jid, seeds=seeds, incumbent=res[0], best=res.get(best), best_i=best, improved=improved, confirm=conf, n_done=done, wall_s=round(time.time() - t0),
                   scores=sorted(round(v) for v in res.values()))
        if improved:
            inc = cands[best]; json.dump(inc, open(out / 'best.json', 'w'), indent=1)
            rec['changed'] = {k: (round(DEFAULT_PARAMS[k], 3), round(v, 3)) for k, v in inc.items() if abs(v - json.load(open(a.init)).get(k, DEFAULT_PARAMS[k])) > 1e-9}
        with hist.open('a') as f: f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)
    json.dump(inc, open(out / 'final.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
