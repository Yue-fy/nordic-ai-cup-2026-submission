"""A/A gate for the state-fork path: two branches that differ in nothing must end identically.

The arena's A/A control showed that 25.6% of games diverge between two runs of the same policy on the same seed,
because the official observation list comes from a set and hash randomisation reorders equivalent entries across
processes. Forking copies the live objects inside one process instead of replaying a seed, so it should not be
exposed to that - but "should not" is not evidence, and every counterfactual label rests on it.

This runs the fork exactly as the campaign does: advance to a decision point, fork twice, step both with the SAME
action list, finish both under cloned policies, and compare. It also checks that forking leaves the parent world
untouched, because fork_world detaches and re-attaches shared pygame surfaces.
"""
import argparse, json, os, random, sys
import multiprocessing as mp

from cup.survival.rollout_advantage import finish


def run_seed(args):
    seed, params, per_seed, rng_seed, lo, hi = args
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    sys.path.insert(0, '<workspace>/nordic-ai-cup')
    from cup.survival.environment import World
    from cup.survival.rules import RulePolicy
    from cup.survival.fork import fork_world
    rng = random.Random(rng_seed)
    world = World(seed); pol = RulePolicy(params); st = world.state
    picks = sorted(rng.sample(range(lo, hi), per_seed))
    out = []; k = 0; nxt = 0
    while True:
        if nxt < len(picks) and k == picks[nxt]:
            nxt += 1
            acts = pol.act(st['observations'], st['sim_time'], st['num_agents'])
            scores = []
            for _ in range(2):                      # identical branches, identical actions
                w2 = fork_world(world)
                w2.step([dict(a) for a in acts])
                scores.append(finish(w2, pol, params))
            parent_ok = (world.state['score'] == st['score'] and world.state['sim_time'] == st['sim_time'])
            out.append({'seed': seed, 'step': k, 'a': round(scores[0], 6), 'b': round(scores[1], 6),
                        'identical': scores[0] == scores[1], 'parent_untouched': parent_ok})
        st, _, term, trunc = world.step(pol.act(st['observations'], st['sim_time'], st['num_agents']))
        k += 1
        if term or trunc: break
    return out


def main():
    from cup.survival.run_games import parse_seeds
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='88001-88120'); ap.add_argument('--params', default='configs/survival_v8_deployed.json')
    ap.add_argument('--per-seed', type=int, default=2); ap.add_argument('--pick-lo', type=int, default=300)
    ap.add_argument('--pick-hi', type=int, default=9000); ap.add_argument('--procs', type=int, default=40)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    params = json.load(open(a.params)); seeds = parse_seeds(a.seeds)
    with mp.get_context('spawn').Pool(min(a.procs, len(seeds))) as pool:
        res = pool.map(run_seed, [(s, params, a.per_seed, 9000 + s, a.pick_lo, a.pick_hi) for s in seeds])
    rows = [r for rs in res for r in rs]
    json.dump(rows, open(a.output, 'w'), indent=1)
    same = sum(1 for r in rows if r['identical']); untouched = sum(1 for r in rows if r['parent_untouched'])
    print('A/A fork pairs: %d' % len(rows))
    print('  identical final score: %d (%.2f%%)' % (same, 100.0 * same / max(1, len(rows))))
    print('  parent world untouched by the fork: %d (%.2f%%)' % (untouched, 100.0 * untouched / max(1, len(rows))))
    bad = [r for r in rows if not r['identical']]
    for r in bad[:8]: print('    DIVERGED seed %d step %d: %.6f vs %.6f (delta %.3f)' % (r['seed'], r['step'], r['a'], r['b'], r['b'] - r['a']))
    print('  VERDICT:', 'fork labels are trustworthy' if not bad else 'FORK LABELS ARE NOT TRUSTWORTHY')


if __name__ == '__main__':
    main()
