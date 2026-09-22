"""Paired counterfactual rollouts: what is one reproduction decision actually worth?

At a decision point the state is forked twice; one branch takes the policy's action, the other takes the opposite
(spawn where it would not, or hold back where it would). Both then continue under the same policy to the end of the
game. Because the fork copies the state object, the two branches are identical apart from that single action, and a
pair of identical forks scores bit-for-bit the same (verified), so the difference is the causal effect of the action
plus whatever randomness the branches accumulate afterwards.

The point of measuring advantages rather than values: a value network on public features scored R^2 0.546 against
0.552 for a quadratic in elapsed time alone, i.e. the clock explains nearly everything. An advantage is a difference
taken at one instant, so the clock cancels out and whatever signal remains is what a policy could actually use.

    python -m cup.survival.rollout_advantage --seeds 75001-75016 --per-seed 6 --repeats 2 --procs 32 --output runs/adv.json
"""
import argparse, json, os, random, sys, copy
import multiprocessing as mp
from pathlib import Path

def features(obs, state, env_time):
    kinds = {'Fruit': 0, 'Tree': 0, 'Agent': 0, 'Predator': 0}
    nearest = {'Fruit': 999.0, 'Tree': 999.0, 'Predator': 999.0}
    for o in obs['observations']:
        t = str(o.get('type', '')).capitalize()
        if t in kinds:
            kinds[t] += 1
            if t in nearest and 'distance' in o: nearest[t] = min(nearest[t], o['distance'])
    return dict(t=round(env_time, 1), n_agents=state['num_agents'], energy=round(obs['energy'], 1),
                age=round(obs['age'], 1), max_energy=round(obs['max_energy'], 1),
                speed=round(obs['speed'], 2), vision=round(obs['vision_range'], 1),
                n_fruit=kinds['Fruit'], n_tree=kinds['Tree'], n_agent=kinds['Agent'], n_pred=kinds['Predator'],
                d_fruit=round(nearest['Fruit'], 1), d_tree=round(nearest['Tree'], 1), d_pred=round(nearest['Predator'], 1))

def clone_policy(pol, params):
    from cup.survival.rules import RulePolicy
    q = RulePolicy(params)
    q.memory = copy.deepcopy(pol.memory); q.step = pol.step
    q.last_sim_time = pol.last_sim_time; q.last_birth_step = pol.last_birth_step
    q.decoys = set(getattr(pol, 'decoys', set()))
    return q

def finish(world, pol, params, cap_steps=40000):
    p = clone_policy(pol, params); st = world.state; k = 0
    while k < cap_steps:
        st, _, term, trunc = world.step(p.act(st['observations'], st['sim_time'], st['num_agents']))
        k += 1
        if term or trunc: break
    return st['score']

def run_seed(args):
    seed, params, per_seed, repeats, rng_seed, pick_lo, pick_hi = args
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    sys.path.insert(0, '<workspace>/nordic-ai-cup')
    from cup.survival.environment import World
    from cup.survival.rules import RulePolicy
    from cup.survival.fork import fork_world
    rng = random.Random(rng_seed)
    world = World(seed); pol = RulePolicy(params); st = world.state
    # decision points are spread over the part of the game the policy actually reaches
    picks = sorted(rng.sample(range(pick_lo, pick_hi), per_seed))
    out = []; k = 0; nxt = 0
    while True:
        if nxt < len(picks) and k == picks[nxt]:
            nxt += 1
            acts = pol.act(st['observations'], st['sim_time'], st['num_agents'])
            spawners = [a for a in acts if a['spawn_agent']]
            by_id = {o['agent_id']: o for o in st['observations']}
            if spawners:
                tgt = rng.choice(spawners); flip = False           # it spawns; the counterfactual holds back
            else:
                cand = [a for a in acts if by_id.get(a['agent_id'], {}).get('energy', 0) > 120]
                if not cand: 
                    st, _, term, trunc = world.step(acts); k += 1
                    if term or trunc: break
                    continue
                tgt = rng.choice(cand); flip = True                # it holds back; the counterfactual spawns
            feat = features(by_id[tgt['agent_id']], st, st['sim_time']); feat['flip'] = flip
            base_scores, alt_scores = [], []
            for _ in range(repeats):
                for which, scores in (('base', base_scores), ('alt', alt_scores)):
                    w2 = fork_world(world)
                    a2 = [dict(a) for a in acts]
                    if which == 'alt':
                        for a in a2:
                            if a['agent_id'] == tgt['agent_id']: a['spawn_agent'] = not a['spawn_agent']
                    w2.step(a2)
                    scores.append(finish(w2, pol, params))
            feat['base'] = round(sum(base_scores) / len(base_scores), 1)
            feat['alt'] = round(sum(alt_scores) / len(alt_scores), 1)
            feat['adv'] = round(feat['alt'] - feat['base'], 1)     # value of taking the OPPOSITE action
            feat['seed'] = seed
            out.append(feat)
        st, _, term, trunc = world.step(pol.act(st['observations'], st['sim_time'], st['num_agents']))
        k += 1
        if term or trunc: break
    return out

def main():
    from cup.survival.run_games import parse_seeds
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='75001-75016'); ap.add_argument('--params', default='configs/survival_v6_deployed.json')
    ap.add_argument('--per-seed', type=int, default=6); ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--pick-lo', type=int, default=300); ap.add_argument('--pick-hi', type=int, default=9000)
    ap.add_argument('--procs', type=int, default=32); ap.add_argument('--output', required=True)
    a = ap.parse_args(); params = json.load(open(a.params)); seeds = parse_seeds(a.seeds)
    with mp.get_context('spawn').Pool(min(a.procs, len(seeds))) as pool:
        res = pool.map(run_seed, [(s, params, a.per_seed, a.repeats, 1000 + s, a.pick_lo, a.pick_hi) for s in seeds])
    rows = [r for sub in res for r in sub]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); json.dump(rows, open(a.output, 'w'), indent=1)
    import statistics
    print('decision points:', len(rows))
    for name, sel in (('policy spawned, counterfactual held back', lambda r: not r['flip']),
                      ('policy held back, counterfactual spawned', lambda r: r['flip'])):
        g = [r for r in rows if sel(r)]
        if g:
            adv = [r['adv'] for r in g]
            m = statistics.mean(adv); se = statistics.stdev(adv) / len(adv) ** 0.5 if len(adv) > 1 else 0
            print(f'  {name}: n={len(g):4d}  mean advantage of the opposite action {m:+7.1f} +- {1.96*se:.0f}')
    if len(rows) > 20:
        import math
        for f in ('t', 'energy', 'age', 'n_agents', 'n_pred', 'd_pred', 'n_fruit', 'vision', 'speed'):
            xs = [r[f] for r in rows]; ys = [r['adv'] for r in rows]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
            if sx == 0 or sy == 0: continue
            c = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs) / (sx * sy)
            print(f'    corr(adv, {f:9s}) = {c:+.3f}')

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); main()
