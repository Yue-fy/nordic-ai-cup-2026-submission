"""Predator-encounter episodes under a policy: how they start (heard vs seen, facing or not), how they end, what they cost.

    python -m cup.survival.diag_encounters --seeds 70001-70024 --procs 24 --output runs/x.json [--params p.json]
"""
import argparse, json, os, sys, math, collections
import multiprocessing as mp
from pathlib import Path

def play(args):
    seed, params = args
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    from cup.survival.environment import World
    from cup.survival.rules import RulePolicy
    world = World(seed); pol = RulePolicy(params); state = world.state; env = world.sim.env
    episodes = []; open_ep = {}          # agent_id -> episode
    while True:
        obs_by_id = {o['agent_id']: o for o in state['observations']}
        actions = pol.act(state['observations'], state['sim_time'], state['num_agents'])
        for act in actions:
            aid = act['agent_id']; o = obs_by_id.get(aid)
            if o is None: continue
            preds = sorted([x for x in o['observations'] if str(x.get('type', '')).capitalize() == 'Predator'], key=lambda x: x['distance'])
            ep = open_ep.get(aid)
            if preds:
                q = preds[0]
                if ep is None:
                    ep = open_ep[aid] = dict(t0=round(env.time, 1), d0=round(q['distance'], 1), heard=q['distance'] <= o['hearing_radius'],
                                             facing0=abs(q['angle']) < math.pi / 2, pred_facing_us0=abs(q['rel_dir']) < math.pi / 6 if 'rel_dir' in q else None,
                                             energy0=round(o['energy'], 1), age0=round(o['age'], 1), steps=0, sprint_steps=0, cost=0.0, min_d=q['distance'], outcome=None, gap=0)
                ep['gap'] = 0; ep['min_d'] = min(ep['min_d'], q['distance'])
            elif ep is not None:
                ep['gap'] += 1
                if ep['gap'] > 30:                        # 3 s without a predator in view: episode over, escaped
                    ep['outcome'] = 'escaped'; ep['energy1'] = round(o['energy'], 1); episodes.append(ep); del open_ep[aid]; ep = None
            if ep is not None:
                d = act['move_distance']; sp = o['speed']
                c = 0.05 * d if d <= sp else 0.05 * sp + 0.5 * (min(d, o['sprint_speed']) - sp)
                c += min(math.pi, abs(act['turn_angle'])) / (2 * math.pi)
                ep['steps'] += 1; ep['cost'] += c; ep['sprint_steps'] += 1 if d > sp + 1e-9 else 0
        prev = {a.agent_id: a.energy for a in env.agents}
        state, _, term, trunc = world.step(actions)
        now = {a.agent_id for a in env.agents}
        for aid in set(prev) - now:
            ep = open_ep.pop(aid, None)
            if ep is not None:
                ep['outcome'] = 'killed' if prev[aid] > 3 else 'died_other'; ep['energy1'] = round(prev[aid], 1); episodes.append(ep)
        if term or trunc: break
    for ep in list(open_ep.values()): ep['outcome'] = 'game_end'; episodes.append(ep)
    return dict(seed=seed, score=state['score'], sim_time=state['sim_time'], episodes=episodes)

def summarize(results):
    eps = [e for r in results for e in r['episodes'] if e['outcome'] in ('escaped', 'killed')]
    n = len(eps); games = len(results)
    print(f'games {games}  episodes {n}  per game {n/games:.1f}  killed {sum(e["outcome"]=="killed" for e in eps)/n:.2f}  mean cost {sum(e["cost"] for e in eps)/n:.0f}  total cost/game {sum(e["cost"] for e in eps)/games:.0f}  sprint steps/game {sum(e["sprint_steps"] for e in eps)/games:.0f}')
    def row(name, sel):
        s = [e for e in eps if sel(e)]
        if not s: return
        print(f"  {name:34s} n={len(s):5d} ({len(s)/n:4.2f})  killed {sum(e['outcome']=='killed' for e in s)/len(s):.2f}  cost {sum(e['cost'] for e in s)/len(s):5.0f}  sprint_steps {sum(e['sprint_steps'] for e in s)/len(s):4.1f}  steps {sum(e['steps'] for e in s)/len(s):5.1f}  d0 {sum(e['d0'] for e in s)/len(s):5.0f}  E0 {sum(e['energy0'] for e in s)/len(s):4.0f}")
    row('start: heard (<=hearing radius)', lambda e: e['heard'])
    row('start: seen, we face it', lambda e: not e['heard'] and e['facing0'])
    row('start: seen, behind us', lambda e: not e['heard'] and not e['facing0'])
    row('start d0 < 60', lambda e: e['d0'] < 60)
    row('start 60 <= d0 < 95', lambda e: 60 <= e['d0'] < 95)
    row('start 95 <= d0 < 130', lambda e: 95 <= e['d0'] < 130)
    row('start d0 >= 130', lambda e: e['d0'] >= 130)
    row('energy0 < 100 (cannot sprint)', lambda e: e['energy0'] < 100)
    row('energy0 >= 100', lambda e: e['energy0'] >= 100)
    row('predator facing us at start', lambda e: e['pred_facing_us0'])
    row('predator not facing us at start', lambda e: e['pred_facing_us0'] is False)
    killed = [e for e in eps if e['outcome'] == 'killed']
    if killed: print(f"  killed: mean E0 {sum(e['energy0'] for e in killed)/len(killed):.0f}, mean steps {sum(e['steps'] for e in killed)/len(killed):.1f}, share with sprint_steps==0 {sum(e['sprint_steps']==0 for e in killed)/len(killed):.2f}, heard-start share {sum(e['heard'] for e in killed)/len(killed):.2f}")

def main():
    from cup.survival.run_games import parse_seeds
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='70001-70024'); ap.add_argument('--params'); ap.add_argument('--procs', type=int, default=24); ap.add_argument('--output', required=True)
    a = ap.parse_args(); params = json.load(open(a.params)) if a.params else None; seeds = parse_seeds(a.seeds)
    with mp.get_context('spawn').Pool(min(a.procs, len(seeds))) as pool:
        results = pool.map(play, [(s, params) for s in seeds])
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); json.dump(results, open(a.output, 'w'))
    print('mean score', sum(r['score'] for r in results) / len(results)); summarize(results)

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); main()
