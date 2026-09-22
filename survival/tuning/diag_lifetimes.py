"""Instrumented full games: where does the energy go, who reproduces, what kills, per 100 s and per agent life.

    python -m cup.survival.diag_lifetimes --seeds 70001-70032 --procs 32 --output runs/x.json [--params p.json]
"""
import argparse, json, os, sys, collections, math
import multiprocessing as mp
from pathlib import Path

def play(args):
    seed, params = args
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy'); os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    from cup.survival.environment import World
    from cup.survival.rules import RulePolicy
    world = World(seed); pol = RulePolicy(params); state = world.state; env = world.sim.env
    life = {}                      # agent_id -> record
    per = collections.defaultdict(lambda: collections.Counter())   # 100 s bucket -> counters
    def rec(a):
        return life.setdefault(a.agent_id, dict(birth=round(env.time, 1), max_age=round(a.max_age, 1), children=0, eaten=0.0, n_fruit=0,
                                                 cost=collections.Counter(), steps=collections.Counter(), death=None, cause=None, energy_at_death=None))
    for a in env.agents: rec(a)
    while True:
        obs_by_id = {o['agent_id']: o for o in state['observations']}
        prev = {a.agent_id: (a.energy, a.age, a.max_age) for a in env.agents}
        actions = pol.act(state['observations'], state['sim_time'], state['num_agents'])
        b = int(env.time // 100) * 100
        for act in actions:
            aid = act['agent_id']; o = obs_by_id.get(aid)
            if o is None: continue
            r = rec(env.agents_dict[aid]); mem = pol.memory.get(aid)
            kinds = collections.Counter(str(x.get('type', '')).capitalize() for x in o['observations'])
            d = act['move_distance']; speed = o['speed']
            if d > speed + 1e-9: cat = 'sprint'
            elif kinds['Predator'] and d > 0: cat = 'evade'
            elif kinds['Fruit'] and d > 0: cat = 'to_fruit'
            elif mem is not None and mem.mode == 'wait': cat = 'wait_tree'
            elif kinds['Tree'] and d > 0: cat = 'to_tree'
            elif d > 0: cat = 'explore_walk'
            else: cat = 'explore_scan'
            if d <= speed: c = 0.05 * d
            else: c = 0.05 * speed + 0.5 * (min(d, o['sprint_speed']) - speed)
            c += min(math.pi, abs(act['turn_angle'])) / (2 * math.pi)
            r['cost'][cat] += c; r['steps'][cat] += 1; per[b]['cost_' + cat] += c; per[b]['steps_' + cat] += 1
            per[b]['sees_food'] += 1 if (kinds['Fruit'] or kinds['Tree']) else 0; per[b]['agent_steps'] += 1
            per[b]['energy_sum'] += o['energy']
            if act.get('spawn_agent') and o['energy'] > 100: r['children'] += 1; per[b]['births'] += 1
        e_before = {a.agent_id: a.energy for a in env.agents}
        state, _, term, trunc = world.step(actions)
        now = {a.agent_id for a in env.agents}
        for a in env.agents:
            r = rec(a)
            if a.agent_id in e_before:
                gain = a.energy - e_before[a.agent_id]
                if gain > 5: r['eaten'] += gain; r['n_fruit'] += 1; per[b]['eaten'] += gain
        for aid in set(prev) - now:
            e, age, ma = prev[aid]; cause = 'predator' if e > 3 else ('old_age' if age > ma else 'starved')
            r = life[aid]; r['death'] = round(env.time, 1); r['cause'] = cause; r['energy_at_death'] = round(e, 1)
            per[b]['death_' + cause] += 1
            per[b]['childless_deaths'] += 1 if r['children'] == 0 else 0
        per[b]['alive_sum'] += state['num_agents']; per[b]['ticks'] += 1
        per[b]['trees_sum'] += len(env.trees); per[b]['fruits_sum'] += len(env.fruits); per[b]['pred_sum'] += len(env.predators)
        if term or trunc: break
    for r in life.values(): r['cost'] = dict(r['cost']); r['steps'] = dict(r['steps'])
    return dict(seed=seed, score=state['score'], sim_time=state['sim_time'], per100={str(k): dict(v) for k, v in sorted(per.items())}, life=life)

def summarize(results):
    keys = sorted({k for r in results for k in r['per100']}, key=int)
    print(f"{'t':>5} {'n':>3} {'alive':>5} {'births':>6} {'d_old':>5} {'d_pred':>6} {'d_starv':>7} {'childless':>9} {'E_mean':>6} {'seesfood':>8} {'trees':>5} {'fruits':>6} {'pred':>4} | energy cost share: sprint evade fruit wait tree walk scan | eaten/agent/100s")
    for k in keys:
        rows = [r['per100'][k] for r in results if k in r['per100']]; n = len(rows)
        def m(f, d=1): return sum(x.get(f, 0) for x in rows) / max(1, n)
        ticks = m('ticks'); asteps = m('agent_steps')
        costs = {c: m('cost_' + c) for c in ('sprint', 'evade', 'to_fruit', 'wait_tree', 'to_tree', 'explore_walk', 'explore_scan')}; tot = sum(costs.values()) or 1
        print(f"{k:>5} {n:>3} {m('alive_sum')/max(1,ticks):5.1f} {m('births'):6.1f} {m('death_old_age'):5.1f} {m('death_predator'):6.1f} {m('death_starved'):7.1f} {m('childless_deaths'):9.1f} "
              f"{m('energy_sum')/max(1,asteps):6.0f} {m('sees_food')/max(1,asteps):8.2f} {m('trees_sum')/max(1,ticks):5.1f} {m('fruits_sum')/max(1,ticks):6.1f} {m('pred_sum')/max(1,ticks):4.1f} | "
              + ' '.join(f"{costs[c]/tot:5.2f}" for c in costs) + f" | {m('eaten')/max(1,asteps)*1000:5.0f}")
    lives = [l for r in results for l in r['life'].values() if l['death'] is not None]
    by = collections.defaultdict(list)
    for l in lives: by[l['cause']].append(l)
    print('\nlives:', len(lives))
    for c, ls in by.items():
        ch = sum(l['children'] for l in ls) / len(ls); childless = sum(l['children'] == 0 for l in ls) / len(ls)
        age = sum(l['death'] - l['birth'] for l in ls) / len(ls); ma = sum(l['max_age'] for l in ls) / len(ls)
        print(f"  {c:9s} n={len(ls):4d} mean_children={ch:.2f} childless={childless:.2f} mean_lifespan={age:.1f} mean_max_age={ma:.1f} mean_eaten={sum(l['eaten'] for l in ls)/len(ls):.0f}")
    # reproduction: how many children per agent by max_age tercile
    for lo, hi in ((60, 80), (80, 100), (100, 121)):
        ls = [l for l in lives if lo <= l['max_age'] < hi]
        if ls: print(f"  max_age in [{lo},{hi}): n={len(ls)} mean_children={sum(l['children'] for l in ls)/len(ls):.2f} childless={sum(l['children']==0 for l in ls)/len(ls):.2f} lifespan={sum(l['death']-l['birth'] for l in ls)/len(ls):.1f}")

def main():
    from cup.survival.run_games import parse_seeds
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='70001-70032'); ap.add_argument('--params'); ap.add_argument('--procs', type=int, default=32); ap.add_argument('--output', required=True)
    a = ap.parse_args(); params = json.load(open(a.params)) if a.params else None; seeds = parse_seeds(a.seeds)
    with mp.get_context('spawn').Pool(min(a.procs, len(seeds))) as pool:
        results = pool.map(play, [(s, params) for s in seeds])
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); json.dump(results, open(a.output, 'w'))
    print('scores', sorted(round(r['score']) for r in results), 'mean', sum(r['score'] for r in results) / len(results))
    summarize(results)

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2])); main()
