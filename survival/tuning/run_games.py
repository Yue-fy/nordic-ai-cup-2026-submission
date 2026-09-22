"""Run full official games with the rule policy on many seeds in parallel and summarize survival time / score.

    python -m cup.survival.run_games --seeds 70001-70016 --procs 16 --output runs/x.json [--params best.json]
"""
import argparse, json, os, sys, time
import multiprocessing as mp
from pathlib import Path

def parse_seeds(text):
    seeds = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-'); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds

def play(args):
    seed, params, max_steps, log_every = args
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    from cup.survival.environment import World
    from cup.survival.rules import RulePolicy
    t0 = time.time(); world = World(seed, max_steps)
    if params and params.get('hrl_checkpoint'):
        from cup.survival.hrl import HRLController
        policy = HRLController(params['hrl_checkpoint'], {k: v for k, v in params.items() if k != 'hrl_checkpoint'}, params.get('hrl_k', 10))
    else:
        policy = RulePolicy(params)
    state = world.state; steps = 0; log = []; nxt = 0; peak = 0
    while True:
        actions = policy.act(state['observations'], state['sim_time'], state['num_agents'])
        state, _, terminated, truncated = world.step(actions); steps += 1
        peak = max(peak, state['num_agents'])
        if log_every and state['sim_time'] >= nxt:
            env = world.sim.env
            log.append(dict(t=round(state['sim_time']), alive=state['num_agents'], trees=len(env.trees),
                            fruits=len(env.fruits), predators=len(env.predators))); nxt += log_every
        if terminated or truncated:
            break
    return dict(seed=seed, score=state['score'], sim_time=state['sim_time'], alive=state['num_agents'],
                peak_population=peak, steps=steps, wall_s=round(time.time() - t0, 1), log=log)

def summarize(results):
    import statistics
    scores = sorted(r['score'] for r in results); times = sorted(r['sim_time'] for r in results)
    n = len(results)
    return dict(n=n, mean_score=sum(scores) / n, median_score=statistics.median(scores), min_score=scores[0],
                median_time=statistics.median(times), full_fraction=sum(t >= 3000 for t in times) / n,
                mean_wall_s=sum(r['wall_s'] for r in results) / n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='70001-70008'); ap.add_argument('--params')
    ap.add_argument('--procs', type=int, default=max(1, min(32, os.cpu_count() or 1)))
    ap.add_argument('--steps', type=int, default=30001); ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    params = json.load(open(args.params)) if args.params else None
    seeds = parse_seeds(args.seeds)
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(args.procs, len(seeds))) as pool:
        results = []
        for r in pool.imap_unordered(play, [(s, params, args.steps, args.log_every) for s in seeds]):
            results.append(r); print(json.dumps({k: v for k, v in r.items() if k != 'log'}), flush=True)
    results.sort(key=lambda r: r['seed'])
    summary = summarize(results); print('SUMMARY', json.dumps(summary), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(params=params, summary=summary, results=results), open(args.output, 'w'), indent=1)

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); main()
