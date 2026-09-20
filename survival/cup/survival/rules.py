"""Mechanics-informed rule policy for the Survival Simulator, with a small tunable parameter vector.

Facts from the official source that shape these rules (see environment.py / agent.py / predator.py):
- Living costs 1 energy/s; walking at full speed 5/s; sprinting ~55/s; a full 180-degree turn 0.5 per step.
- move_direction is RELATIVE to the heading and costs nothing extra: agents can strafe without turning.
- Agents past max_age (60-120 s, unobservable) lose 0.01*age per STEP, so nobody lives much past ~130 s.
  The species survives only through continuous reproduction (spawn costs 100, child starts with 75).
- Trees spawn less over time (halving every 300 s) and die after 50-100 s: late game is a famine, so the
  population must shrink with time instead of growing.
- Fruits appear within ~60 units of a tree, start at 20 energy and ripen 2/s up to 60, rot after 100 s.
- Predators chase only when behind the agent or closer than ~90 units; otherwise they circle to get behind.
"""
import math

DEFAULT_PARAMS = {
    'spawn_energy': 230.0,     # spawn when energy above this (and population below cap)
    'spawn_late_age': 70.0,    # past this age, spawn as soon as energy > 100 + late_margin (pass energy on before old-age drain)
    'late_margin': 30.0,
    'spawn_needs_food': 1.0,   # >0.5: only spawn when a tree or fruit is currently sensed
    'pop0': 18.0,              # population cap at t=0 ...
    'pop_half': 700.0,         # ... halving every pop_half seconds ...
    'pop_min': 3.0,            # ... but never below this
    'danger_dist': 130.0,      # predator closer than this: face it and back away at walking speed
    'chase_dist': 95.0,        # predator closer than this: sprint away (it chases regardless of facing)
    'tree_hold': 45.0,         # stay within this distance of a tree and wait for fruit
    'scan_turn': 0.12,         # radians per step of idle scanning rotation
    'explore_speed': 0.35,     # fraction of walking speed used when nothing is sensed
    'walk_steps': 25.0,        # explore cycle: walk this many steps ...
    'scan_steps': 55.0,        # ... then stand and scan this many steps
    'fruit_ripe_steps': 120.0, # while waiting at a tree, let a fruit ripen this many steps before eating
    'hungry_energy': 160.0,    # below this energy eat immediately and ignore ripening (sprinting needs > 20% of max_energy)
    'max_fruit_dist': 400.0,
    'follow_agents': 1.0,      # >0.5: with nothing else in sight, walk towards other agents (they are probably near food)
    'spacing_dist': 35.0,      # keep at least this far from the nearest other agent (a predator kill refills its energy: clustered herds get chain-killed)
    'birth_interval': 60.0,    # at most one birth per this many steps across the whole population (staggers generations)
    'sprint_reserve': 0.25,    # sprint from a predator only if energy > this fraction of max_energy (engine forbids sprinting below 0.2)
    'river_exit': 0.0,         # >0.5: in river/swamp (movement penalty, no trees in rivers) with nothing to eat, walk straight out
    'pop_per_tree_seen': 0.0,  # >0: population cap also limited to pop_min + this * (number of agents currently seeing a tree)
    'stealth': 0.0,            # >0.5: predators only see a 60-degree cone (250) and hear 60; react only when inside its cone or
                               #       near its hearing range, and step sideways out of the cone instead of running away
    'cone_margin': 0.2,        # radians added to the predator's half cone (pi/6) when deciding whether we are visible to it
    'hear_margin': 80.0,       # keep at least this far from a predator that cannot see us (its hearing radius is 60)
    'deathbed': 0.0,           # >0.5: detect the old-age drain (energy falls faster than our own actions explain) and spawn at once while energy > 100 + deathbed_margin
    'deathbed_margin': 0.0,
    'deathbed_food': 0.0,      # >0.5: a deathbed spawn still requires a tree or fruit in sight
    'deathbed_cap': 0.0,       # >0.5: a deathbed spawn still respects the population cap and birth interval
    'drain_thresh': 0.35,      # unexplained energy loss per step that counts as old-age drain (true drain is 0.01*age >= 0.6)
    'relay_trees': 0.0,        # >0.5: an agent that sees another agent AND a tree tells it where the tree is (exact, in the other agent's frame)
    'tree_mem_steps': 0.0,     # >0: remember the last seen tree (dead-reckoned in our own frame) and walk back to it for this many steps when nothing is in sight
    'pred_mem_steps': 0.0,     # >0: remember the last seen predator position for this many steps and do not explore/follow towards it
    'pred_avoid_angle': 1.0,   # radians: half-angle around the remembered predator bearing that exploration avoids
    'face_pred': 0.0,          # >0.5: keep turning to face any visible predator (it only chases when we look away or it is within 90); foraging continues by strafing
    'relay_preds': 0.0,        # >0.5: an agent that sees a predator AND other agents tells them where it is (exact, in their frame) so they face it early
    'flee_speed': 0.0,         # >0: sprint from a predator at this absolute speed instead of full sprint_speed (predator sprints 15; matching it keeps the gap at 0.5+0.5*(v-10) per step instead of 5.5)
    'flee_close_dist': 40.0,   # closer than this we always use full sprint_speed
    'spawn_fruit_dist': 0.0,   # >0: a normal spawn also requires a fruit within this distance (the child starts with 75 energy and cannot sprint until it eats)
    # --- selective breeding -------------------------------------------------------------------------------
    # Children inherit the parent's traits, each mutated with probability 0.1 by a factor U(0.5, 1.5), and every
    # trait can double from its baseline (speed 10->20, sprint 20->40, max_energy 500->1000, hearing 50->100,
    # vision 200->400, cone pi/3->pi/2). A game lasts ~25 generations, and all six traits are in the observation,
    # so choosing WHICH agent reproduces compounds: vision radius x cone alone can grow the sensed area ~6x.
    'breed_order': 0.0,        # >0.5: when the population cap limits births, grant them in descending trait order
    'breed_select': 0.0,       # >0.5: a normal spawn also requires the parent's trait score to be >= the population quantile
    'breed_quantile': 0.5,     # 0.5 = only the better half of the living population may reproduce normally
    'breed_deathbed': 0.0,     # >0.5: apply the same gate to deathbed spawns (they would lose that energy anyway)
    'breed_w_vision': 1.0, 'breed_w_hearing': 1.0, 'breed_w_cone': 1.0,
    'breed_w_sprint': 1.0, 'breed_w_energy': 0.0, 'breed_w_speed': 0.5,
    # --- coverage ------------------------------------------------------------------------------------------
    # A standing, scanning agent sees a disc of its own vision radius: 6.5% of the map at the baseline 200, 26% at
    # the cap 400. Food never vanishes (still ~3 energy/s at t=3000) but the few remaining trees must be FOUND, so
    # spreading the population out matters more as vision grows.
    'spacing_vision': 0.0,     # >0.5: scale spacing_dist by the agent's own vision_range / 200
    'spacing_max': 400.0,      # cap for the scaled spacing distance
    'walk_vision': 0.0,        # >0.5: scale walk_steps by vision_range / 200 (walk about one vision diameter between full scans)
    'hungry_rel': 0.0,         # >0: hunger threshold as a fraction of the agent's own max_energy instead of the absolute hungry_energy
    'breed_elite': 0.0,        # >0.5: the best-ranked fraction of the population reproduces at a LOWER energy threshold, so the
                               #       best genome spreads faster (measured: vision reaches 1.74x baseline by t=1400 under selection)
    'breed_elite_frac': 0.25,  # top fraction treated as elite
    'breed_elite_energy': 150.0,  # elite spawn threshold (engine minimum is 100; leave the parent a margin)
    # --- exhausting predators instead of outrunning them ---------------------------------------------------
    # A predator charges only when we look away or it is within ~90; otherwise it PIVOTS at sprint speed, closing
    # about 0.6 units per step against our walk. Pivoting costs it 2.55 energy/step out of a 200 maximum, while
    # facing it and backing away costs us 0.5/step, so ~7 s of that puts it to sleep. Fleeing instead costs us
    # 5.5/step and it recovers 30/s while resting, so the flight is the losing side of the energy war.
    # The predator sees 250; we see 200 at baseline but up to 400 once vision evolves, which is what makes
    # starting this early possible.
    'danger_vision_frac': 0.0, # >0: back away (facing it) from this fraction of our own vision_range instead of the fixed danger_dist
    'flee_smart': 0.0,         # >0.5: flee at WALKING speed once it already beats the predator's sprint (15). Energy per unit of
                               #       gap opened is 0.05*v/(v-15) walking versus ((S-10)*0.5+0.5)/(S-15) sprinting: at speed 20
                               #       that is 0.20 against 1.10, a 5.5x saving, and walking is never blocked by low energy.
    'flee_walk_min': 16.5,     # walking speed above which fleeing on foot beats sprinting
    # --- where to be, and when to wait ---------------------------------------------------------------------
    # Trees only appear where biome.tree_spawn_rate beats a random draw: forest 1.0, swamp 0.9, grassland 0.5,
    # desert 0.1, river 0.0. A desert holds about a tenth of a forest's trees, so being stuck in one is fatal late.
    'biome_flee': 0.0,         # >0.5: in a poor biome with nothing in sight, walk in a straight line instead of the scan/walk cycle
    'biome_flee_rate': 0.45,   # biomes whose tree_spawn_rate is at or below this count as poor (0.45 = desert and river)
    # A fruit starts at 20 energy and ripens 2/s to 60, so waiting triples its value -- but only if no one else eats it.
    'ripe_if_alone': 0.0,      # >0: only wait for ripening when the nearest other agent is farther away than this
    # The observation carries Edge entries (obstacle and boundary segments, already rotated into the agent's frame)
    # that the policy has never used. Fleeing straight away from a predator into a wall is how a cornered agent dies:
    # the engine then rotates the step in 10-degree increments until it clears, which usually turns the escape sideways.
    'edge_flee': 0.0,          # >0.5: when fleeing, slide along a blocking edge instead of running into it
    'edge_flee_dist': 45.0,    # how far ahead an edge counts as blocking
    'fruit_pred_gap': 0.0,     # >0: skip a fruit whose approach would take us within this distance of a visible predator
                               #     (walking into a chase costs far more than the fruit is worth)
    # Extinction happens with ~25 trees and ~58 fruits still on the map: the last agents starve at ~110 energy while
    # spending 57% of their effort backing away from predators. Grabbing a fruit two steps away costs under an energy
    # point and returns 20-60, so a close fruit should outrank a predator that is not yet charging.
    'fruit_first_dist': 0.0,   # >0: take a fruit this close even with a predator in view, as long as it is beyond chase_dist
    'retreat_to_food': 0.0,    # >0.5: when backing away, pick the retreat heading (within a quarter turn of straight away)
                               #       that points most towards the nearest fruit or tree, so retreating also forages
    # --- division of labour ---------------------------------------------------------------------------------
    # A predator chases only the CLOSEST agent it perceives, and being eaten costs energy/100 (about 1.4 points)
    # against a score measured in surviving seconds. Meanwhile the breeding gate means roughly half the population
    # can never reproduce, so the energy those agents gather is lost when they die. Spending them as decoys that
    # keep predators occupied converts that waste into protection for the agents that do carry the genes.
    # --- time conditioning ----------------------------------------------------------------------------------
    # A value network on 43 observable features predicted remaining score with R^2 0.546 on held-out games, while a
    # quadratic in elapsed time alone got 0.552: the clock carries nearly all the usable information. Yet these rules
    # are almost time-invariant. Each multiplier below ramps its parameter from the base value at t=0 to
    # base*multiplier at t=late_t, so the same policy can expand early (food supports ~96 idle agents at t=0) and
    # play safe late (~6 at t=2400, with a predator arriving every ~100 s). A multiplier of 1.0 reproduces v6 exactly.
    'late_t': 1500.0,          # simulated seconds over which every ramp below completes
    'late_danger': 1.0,        # danger_dist: react to predators from further away as they multiply
    'late_explore': 1.0,       # explore_speed: cheaper wandering when food is scarce
    'late_hungry': 1.0,        # hungry_energy: eat sooner rather than waiting for ripeness
    'late_spawn': 1.0,         # spawn_energy: reproduce at a different bar late
    'late_wspeed': 1.0,        # breed_w_speed: shift the selection target from sensing towards escape speed
    'late_scan': 1.0,          # scan_steps: scan longer, walk less
    'late_quantile': 1.0,      # breed_quantile: the q-sweep found a tension -- strong selection evolves traits fast but
                               #                 starves the birth rate, so ramp it instead of picking one value
    # Every agent currently runs the same averaged style. Splitting the population into campers (stay at a tree and
    # wait for fruit to ripen from 20 to 60 energy) and foragers (roam wide for fruit already on the ground) lets both
    # strategies run at once; the split is by a stable hash of the agent id, so it does not interact with breeding.
    'role_split': 0.0,         # >0.5: enable the two styles
    'role_frac': 0.5,          # fraction of agents that are campers
    'role_explore': 0.5,       # camper multiplier on explore_speed
    'role_treehold': 0.8,      # camper multiplier on tree_hold (stay closer to the tree)
    'role_fruitdist': 0.6,     # camper multiplier on max_fruit_dist (do not chase far fruit)
    'role_ripe': 1.6,          # camper multiplier on fruit_ripe_steps (wait longer for ripeness)
    'decoy': 0.0,              # >0.5: agents below the breeding gate hold their ground near a predator instead of fleeing
    'decoy_hold_dist': 60.0,   # a decoy keeps roughly this distance: close enough to stay the nearest target, far enough to live
    'decoy_min_energy': 60.0,  # below this a decoy stops volunteering and feeds itself
    'mpc': 0.0,                # >0.5: choose the escape by rolling the local scene forward against the predator rule
    'mpc_dist': 200.0,         # only plan when the nearest predator is at least this close
    'mpc_steps': 15.0,         # horizon, in simulator steps
    'mpc_dirs': 12.0,          # candidate move directions
    'mpc_preds': 3.0,          # how many of the visible predators to plan against
    'mpc_w_risk': 400.0,       # energy-equivalent price of being caught
    'mpc_w_food': 1.0,         # pull towards the nearest food, per 100 units of final distance
    'decoy_sen': 0.0,          # >0.5: only agents the deathbed detector has flagged as dying act as decoys
    'decoy_frac': 0.5,         # at most this fraction of the population acts as decoys at any time
    'breed_concave': 0.0,      # >0.5: score traits as sqrt(trait/baseline) instead of the raw ratio. The engine clips every
                               #       trait at twice its baseline (cone at 1.5x), so a linear score lets one maxed trait pay
                               #       for the rest; a concave one keeps pushing whichever trait is still furthest behind.
}
TRAIT_BASE = {'vision_range': 200.0, 'hearing_radius': 50.0, 'vision_angle': math.pi / 3, 'sprint_speed': 20.0, 'max_energy': 500.0, 'speed': 10.0}
TRAIT_W = {'vision_range': 'breed_w_vision', 'hearing_radius': 'breed_w_hearing', 'vision_angle': 'breed_w_cone',
           'sprint_speed': 'breed_w_sprint', 'max_energy': 'breed_w_energy', 'speed': 'breed_w_speed'}

def mpc_escape(obs, kinds, p, speed, sprint, energy):
    """Short-horizon rollout against the predator's own rule, using only what this agent can already see.

    The predator charges when the agent's bearing to it exceeds pi/2 or the distance drops below 90, and otherwise
    circles at sprint 15. Its decision variable rel_dir is, by construction, the same angle this agent observes for
    it, so the local scene can be rolled forward with no global state at all: what rules out tree search in this
    task is missing information, not compute, and this particular question needs none.

    The hand rule reacts to the nearest predator only; this scores every candidate move against all of them.
    """
    preds = kinds['Predator'][:int(p['mpc_preds'])]
    if not preds:
        return None
    K = int(p['mpc_steps'])
    qx0 = [q['distance'] * math.cos(q['angle']) for q in preds]
    qy0 = [q['distance'] * math.sin(q['angle']) for q in preds]
    food = (kinds['Fruit'] or kinds['Tree'])
    fx = fy = None
    if food:
        fx = food[0]['distance'] * math.cos(food[0]['angle'])
        fy = food[0]['distance'] * math.sin(food[0]['angle'])
    fast = sprint if energy > obs['max_energy'] * p['sprint_reserve'] else speed
    n_dir = int(p['mpc_dirs'])
    best = None
    for i in range(n_dir):
        move_dir = -math.pi + 2.0 * math.pi * i / n_dir
        ux, uy = math.cos(move_dir), math.sin(move_dir)
        for v in ((speed, fast) if fast > speed else (speed,)):
            step_cost = v * 0.05 if v <= speed else speed * 0.05 + (v - speed) * 0.5
            ax = ay = 0.0
            spent = 0.0
            caught = None
            qx = list(qx0)
            qy = list(qy0)
            for k in range(K):
                ax += v * ux
                ay += v * uy
                spent += step_cost
                for j in range(len(qx)):
                    dx = ax - qx[j]
                    dy = ay - qy[j]
                    d = math.hypot(dx, dy)
                    if d < 20.0:
                        caught = k
                        break
                    bearing = math.atan2(dy, dx)                  # from the predator towards us
                    ang_to_pred = wrap(math.atan2(-dy, -dx))      # our own bearing to it, which is its rel_dir
                    if abs(ang_to_pred) > math.pi / 2 or d < 90.0:
                        step = min(15.0, d)                       # it charges
                        qx[j] += step * math.cos(bearing)
                        qy[j] += step * math.sin(bearing)
                    else:
                        side = 1.0 if ang_to_pred >= 0 else -1.0  # it circles instead of closing
                        c = wrap(bearing + side * math.pi / 4)
                        qx[j] += 15.0 * math.cos(c)
                        qy[j] += 15.0 * math.sin(c)
                if caught is not None:
                    break
            risk = 0.0 if caught is None else float(K - caught) / K
            score = p['mpc_w_risk'] * risk + spent
            if fx is not None:
                score += p['mpc_w_food'] * math.hypot(ax - fx, ay - fy) / 100.0
            if best is None or score < best[0]:
                best = (score, move_dir, v)
    return best


def trait_score(obs, p):
    """Heritable quality of an agent, each trait relative to the species baseline."""
    if p['breed_concave'] > 0.5:
        return sum(p[w] * math.sqrt(float(obs.get(k, TRAIT_BASE[k])) / TRAIT_BASE[k]) for k, w in TRAIT_W.items())
    return sum(p[w] * (float(obs.get(k, TRAIT_BASE[k])) / TRAIT_BASE[k]) for k, w in TRAIT_W.items())
PARAM_BOUNDS = {
    'spawn_energy': (120, 450), 'spawn_late_age': (40, 130), 'late_margin': (0, 120), 'spawn_needs_food': (0, 1),
    'pop0': (4, 60), 'pop_half': (150, 3000), 'pop_min': (1, 12), 'danger_dist': (60, 250), 'chase_dist': (50, 160),
    'tree_hold': (20, 120), 'scan_turn': (0.02, 0.5), 'explore_speed': (0.05, 1.0), 'walk_steps': (5, 150),
    'scan_steps': (0, 150), 'fruit_ripe_steps': (0, 250), 'hungry_energy': (20, 250), 'max_fruit_dist': (100, 600),
    'follow_agents': (0, 1), 'spacing_dist': (0, 120), 'birth_interval': (1, 400), 'sprint_reserve': (0.2, 0.8),
    'river_exit': (0, 1), 'pop_per_tree_seen': (0, 6), 'stealth': (0, 1), 'cone_margin': (0, 0.6), 'hear_margin': (60, 130),
    'deathbed': (0, 1), 'deathbed_margin': (0, 150), 'deathbed_food': (0, 1), 'deathbed_cap': (0, 1), 'drain_thresh': (0.2, 0.6),
    'relay_trees': (0, 1), 'tree_mem_steps': (0, 1500), 'pred_mem_steps': (0, 600), 'pred_avoid_angle': (0.3, 2.0),
    'face_pred': (0, 1), 'relay_preds': (0, 1), 'flee_speed': (0, 40), 'flee_close_dist': (15, 95), 'spawn_fruit_dist': (0, 200),
    'breed_order': (0, 1), 'breed_select': (0, 1), 'breed_quantile': (0, 0.95), 'breed_deathbed': (0, 1),
    'breed_w_vision': (0, 3), 'breed_w_hearing': (0, 3), 'breed_w_cone': (0, 3), 'breed_w_sprint': (0, 3), 'breed_w_energy': (0, 3), 'breed_w_speed': (0, 3),
    'spacing_vision': (0, 1), 'spacing_max': (40, 600), 'walk_vision': (0, 1), 'hungry_rel': (0, 0.8),
    'breed_elite': (0, 1), 'breed_elite_frac': (0.05, 0.8), 'breed_elite_energy': (110, 350), 'danger_vision_frac': (0, 1.0), 'flee_smart': (0, 1), 'flee_walk_min': (15.5, 30), 'biome_flee': (0, 1), 'biome_flee_rate': (0.05, 0.95), 'ripe_if_alone': (0, 300), 'breed_concave': (0, 1),
    'mpc': (0, 1), 'mpc_dist': (80, 300), 'mpc_steps': (5, 40), 'mpc_dirs': (6, 24), 'mpc_preds': (1, 5),
    'mpc_w_risk': (50, 2000), 'mpc_w_food': (0, 10),
    'decoy': (0, 1), 'decoy_sen': (0, 1), 'decoy_hold_dist': (25, 140), 'decoy_min_energy': (30, 200), 'decoy_frac': (0.1, 0.9),
    'role_split': (0, 1), 'role_frac': (0.1, 0.9), 'role_explore': (0.1, 2.0), 'role_treehold': (0.4, 2.0),
    'role_fruitdist': (0.2, 2.0), 'role_ripe': (0.3, 3.0),
    'late_t': (400, 3000), 'late_danger': (0.4, 2.5), 'late_explore': (0.2, 3.0), 'late_hungry': (0.4, 2.5),
    'late_spawn': (0.4, 2.0), 'late_wspeed': (0.2, 4.0), 'late_scan': (0.3, 3.0), 'late_quantile': (0.3, 1.8), 'edge_flee': (0, 1), 'edge_flee_dist': (15, 150), 'fruit_pred_gap': (0, 250), 'fruit_first_dist': (0, 160), 'retreat_to_food': (0, 1),
}

BIOME_PENALTY = {'forest': 1.0, 'grassland': 1.0, 'swamp': 0.5, 'desert': 0.8, 'river': 0.3}

def clear_direction(edges, want, limit):
    """Pick a heading close to `want` that is not blocked by an edge within `limit`.  Edge coords are already in the
    agent's frame, so the closest point of each segment to the origin gives its bearing and range directly."""
    blockers = []
    for e in edges:
        c = e.get('coords')
        if not c: continue
        (x1, y1), (x2, y2) = c
        dx, dy = x2 - x1, y2 - y1; den = dx * dx + dy * dy
        t = 0.0 if den <= 0 else max(0.0, min(1.0, -(x1 * dx + y1 * dy) / den))
        px, py = x1 + t * dx, y1 + t * dy; d = math.hypot(px, py)
        if d < limit: blockers.append((math.atan2(py, px), d))
    if not blockers: return want
    def blocked(th):
        return any(abs(wrap(th - b)) < math.pi / 3 for b, _ in blockers)
    if not blocked(want): return want
    for off in (0.6, -0.6, 1.1, -1.1, 1.6, -1.6, 2.2, -2.2, math.pi):
        if not blocked(wrap(want + off)): return wrap(want + off)
    return want

RAMPED = {'danger_dist': 'late_danger', 'explore_speed': 'late_explore', 'hungry_energy': 'late_hungry',
          'spawn_energy': 'late_spawn', 'breed_w_speed': 'late_wspeed', 'scan_steps': 'late_scan',
          'breed_quantile': 'late_quantile'}

def ramped(p, sim_time):
    """Apply the time ramps once per step; identical to p when every multiplier is 1."""
    if all(p[m] == 1.0 for m in RAMPED.values()): return p
    f = min(1.0, max(0.0, sim_time / max(1.0, p['late_t'])))
    q = dict(p)
    for k, mk in RAMPED.items():
        q[k] = p[k] * (1.0 + (p[mk] - 1.0) * f)
    return q

BIOME_TREE_RATE = {'forest': 1.0, 'grassland': 0.5, 'swamp': 0.9, 'desert': 0.1, 'river': 0.0}

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

class AgentMemory:
    __slots__ = ('mode', 'counter', 'heading', 'fruits', 'last_step', 'last_energy', 'expected_cost', 'drain_count', 'senescent', 'tree_mem', 'pred_mem')
    def __init__(self, step):
        self.mode = 'scan'; self.counter = 0; self.heading = 0.0; self.fruits = []; self.last_step = step
        self.last_energy = None; self.expected_cost = 0.0; self.drain_count = 0; self.senescent = False
        self.tree_mem = None; self.pred_mem = None      # [x, y, step] in the agent's own frame (x forward, y left-handed like the engine)

def advance_point(pt, act, speed, sprint, penalty):
    """Dead-reckon a remembered point (agent frame) through our own action: the engine moves us first by
    distance*penalty along heading+move_direction, then turns us by turn_angle."""
    d = max(0.0, min(act['move_distance'], sprint)) * penalty
    x, y = pt[0] - d * math.cos(act['move_direction']), pt[1] - d * math.sin(act['move_direction'])
    t = -act['turn_angle']; c, s_ = math.cos(t), math.sin(t)
    pt[0], pt[1] = x * c - y * s_, x * s_ + y * c

def relay_tree_for(observer, kinds, kind='Tree'):
    """For every other agent the observer sees, compute the observer's nearest `kind` object in THAT agent's frame.
    rel_dir of an observed agent B = (bearing from B to observer) - B.direction, so B.direction in the observer's
    frame is angle_B + pi - rel_dir.  Returns {agent_id: (distance, angle)}."""
    out = {}
    if not kinds[kind]: return out
    tr = kinds[kind][0]; tx, ty = tr['distance'] * math.cos(tr['angle']), tr['distance'] * math.sin(tr['angle'])
    for b in kinds['Agent']:
        if 'id' not in b or 'rel_dir' not in b: continue
        bx, by = b['distance'] * math.cos(b['angle']), b['distance'] * math.sin(b['angle'])
        heading_b = b['angle'] + math.pi - b['rel_dir']
        vx, vy = tx - bx, ty - by
        out[b['id']] = (math.hypot(vx, vy), wrap(math.atan2(vy, vx) - heading_b))
    return out

def action_cost(act, speed, sprint, spawned):
    """Energy the engine charges for this action (walk 0.05/unit, sprint 0.5/unit above walking speed, turn <= pi over 2 pi, spawn 100)."""
    d = max(0.0, min(act['move_distance'], sprint))
    c = 0.05 * d if d <= speed else 0.05 * speed + 0.5 * (d - speed)
    c += min(math.pi, abs(act['turn_angle'])) / (2 * math.pi)
    return c + (100.0 if spawned else 0.0)

class RulePolicy:
    def __init__(self, params=None):
        self.params = dict(DEFAULT_PARAMS); self.params.update(params or {})
        self.reset()

    def reset(self):
        self.memory = {}; self.step = 0; self.last_sim_time = -1.0; self.last_birth_step = -10**9; self.decoys = set()

    # -------------------------------------------------------------- public API
    def act(self, agent_status, sim_time, n_agents=None, overrides=None):
        """agent_status: list of official ObservationResponse dicts. Returns list of ActionRequest dicts."""
        if sim_time < self.last_sim_time:   # a new game started on the same server
            self.reset()
        self.last_sim_time = sim_time; self.step += 1
        p = ramped(self.params, sim_time)
        alive = n_agents if n_agents is not None else len(agent_status)
        cap = max(p['pop_min'], p['pop0'] * 0.5 ** (sim_time / p['pop_half']))
        if p['pop_per_tree_seen'] > 0:
            seeing = sum(1 for o in agent_status if any(str(x.get('type', '')).capitalize() == 'Tree' for x in o['observations']))
            cap = min(cap, p['pop_min'] + p['pop_per_tree_seen'] * seeing)
        spawned = 0; actions = []; pending = []
        relays = {}; prelays = {}
        if p['relay_trees'] > 0.5 or p['relay_preds'] > 0.5:
            sees_tree = {o['agent_id'] for o in agent_status if any(str(x.get('type', '')).capitalize() in ('Tree', 'Fruit') for x in o['observations'])}
            sees_pred = {o['agent_id'] for o in agent_status if any(str(x.get('type', '')).capitalize() == 'Predator' for x in o['observations'])}
            for o in agent_status:
                k = {'Tree': [], 'Agent': [], 'Predator': []}
                for x in o['observations']:
                    t = str(x.get('type', '')).capitalize()
                    if t in k and 'distance' in x: k[t].append(x)
                if not k['Agent']: continue
                for lst in k.values(): lst.sort(key=lambda z: z['distance'])
                if p['relay_trees'] > 0.5 and k['Tree']:
                    for bid, (d, a) in relay_tree_for(o, k, 'Tree').items():
                        if bid not in sees_tree and (bid not in relays or d < relays[bid][0]): relays[bid] = (d, a)
                if p['relay_preds'] > 0.5 and k['Predator']:
                    for bid, (d, a) in relay_tree_for(o, k, 'Predator').items():
                        if bid not in sees_pred and d < 260 and (bid not in prelays or d < prelays[bid][0]): prelays[bid] = (d, a)
        # role assignment needs the whole population, so rank first when the decoy rule is on
        self.decoys = set()
        if p['decoy_sen'] > 0.5:
            # a senescent agent is dead within seconds whatever it does, so its remaining energy is nearly free to
            # spend on holding a predator's attention; the earlier decoy rule spent healthy low-trait agents instead
            self.decoys = {o['agent_id'] for o in agent_status if o['energy'] > p['decoy_min_energy']
                           and getattr(self.memory.get(o['agent_id']), 'senescent', False)}
        elif p['decoy'] > 0.5 and len(agent_status) > 2:
            ranked = sorted(agent_status, key=lambda o: trait_score(o, p))
            k = int(p['decoy_frac'] * len(ranked))
            self.decoys = {o['agent_id'] for o in ranked[:k] if o['energy'] > p['decoy_min_energy']}
        for obs in agent_status:
            mem = self.memory.get(obs['agent_id'])
            if mem is None:
                mem = self.memory[obs['agent_id']] = AgentMemory(self.step)
            mem.last_step = self.step
            pa = {**p, **overrides[obs['agent_id']]} if overrides and obs['agent_id'] in overrides else p
            if pa['role_split'] > 0.5 and (obs['agent_id'] * 2654435761 % 1000) / 1000.0 < pa['role_frac']:
                pa = {**pa, 'explore_speed': pa['explore_speed'] * pa['role_explore'],
                      'tree_hold': pa['tree_hold'] * pa['role_treehold'],
                      'max_fruit_dist': pa['max_fruit_dist'] * pa['role_fruitdist'],
                      'fruit_ripe_steps': pa['fruit_ripe_steps'] * pa['role_ripe']}
            if obs['agent_id'] in relays or obs['agent_id'] in prelays:
                obs = dict(obs); obs['observations'] = list(obs['observations'])
                if obs['agent_id'] in relays:
                    d, a = relays[obs['agent_id']]; obs['observations'].append({'type': 'Tree', 'distance': d, 'angle': a, 'relayed': True})
                if obs['agent_id'] in prelays:
                    d, a = prelays[obs['agent_id']]; obs['observations'].append({'type': 'Predator', 'distance': d, 'angle': a, 'relayed': True})
            # old-age detection: the engine drains 0.01*age per step once age > max_age (unobservable); we see it as an
            # energy drop our own actions plus the 0.1/step living cost do not explain (eating only adds energy)
            if pa['deathbed'] > 0.5:
                if mem.last_energy is not None:
                    unexplained = (mem.last_energy - obs['energy']) - mem.expected_cost - 0.1
                    mem.drain_count = mem.drain_count + 1 if unexplained > pa['drain_thresh'] else 0
                    if mem.drain_count >= 2 and obs['age'] > 55: mem.senescent = True
                mem.last_energy = obs['energy']
            action, wants_spawn = self._agent(obs, mem, pa)
            deathbed = pa['deathbed'] > 0.5 and mem.senescent and obs['energy'] > 100 + pa['deathbed_margin']
            if deathbed and pa['deathbed_food'] > 0.5 and not any(str(x.get('type', '')).capitalize() in ('Tree', 'Fruit') for x in obs['observations']):
                deathbed = False
            pending.append((obs, mem, pa, action, wants_spawn, deathbed))
            actions.append(action)
        # --- who reproduces (decided across the whole population, not first-come) ----------------------------
        need_scores = p['breed_order'] > 0.5 or p['breed_select'] > 0.5 or p['breed_elite'] > 0.5 or p['decoy'] > 0.5
        scores = {o['agent_id']: trait_score(o, p) for o, *_ in pending} if need_scores else {}
        gate = -1e18; elite_gate = 1e18
        if p['breed_select'] > 0.5 and scores:
            ranked = sorted(scores.values()); gate = ranked[min(len(ranked) - 1, int(p['breed_quantile'] * len(ranked)))]
        if p['breed_elite'] > 0.5 and scores:
            ranked = sorted(scores.values()); elite_gate = ranked[min(len(ranked) - 1, int((1.0 - p['breed_elite_frac']) * len(ranked)))]
        order = sorted(range(len(pending)), key=lambda i: -scores[pending[i][0]['agent_id']]) if p['breed_order'] > 0.5 else range(len(pending))
        for i in order:
            obs, mem, pa, action, wants_spawn, deathbed = pending[i]
            passes = gate <= -1e17 or scores.get(obs['agent_id'], 0.0) >= gate
            if not wants_spawn and elite_gate < 1e17 and scores.get(obs['agent_id'], 0.0) >= elite_gate and obs['energy'] > pa['breed_elite_energy']:
                wants_spawn = pa['spawn_needs_food'] < 0.5 or any(str(x.get('type', '')).capitalize() in ('Tree', 'Fruit') for x in obs['observations'])
            if deathbed and (pa['breed_deathbed'] < 0.5 or passes):
                if pa['deathbed_cap'] < 0.5 or (alive + spawned < cap and self.step - self.last_birth_step >= pa['birth_interval']):
                    action['spawn_agent'] = True; spawned += 1; self.last_birth_step = self.step; continue
            if wants_spawn and passes and alive + spawned < cap and self.step - self.last_birth_step >= pa['birth_interval']:
                action['spawn_agent'] = True; spawned += 1; self.last_birth_step = self.step
        for obs, mem, pa, action, _ws, _db in pending:
            mem.expected_cost = action_cost(action, obs['speed'], obs['sprint_speed'], action['spawn_agent'] and obs['energy'] > 100)
            mem.heading = wrap(mem.heading + action['turn_angle'])
            pen = BIOME_PENALTY.get(str(obs.get('biome', '')).lower(), 1.0)
            for pt in (mem.tree_mem, mem.pred_mem):
                if pt is not None: advance_point(pt, action, obs['speed'], obs['sprint_speed'], pen)
        if self.step % 200 == 0:   # forget dead agents
            self.memory = {k: v for k, v in self.memory.items() if self.step - v.last_step < 5}
        return actions

    # -------------------------------------------------------------- per-agent rules
    def _agent(self, obs, mem, p):
        act, ws = self._agent_inner(obs, mem, p)
        if p['face_pred'] > 0.5 and act['move_distance'] <= obs['speed'] + 1e-9:     # never override a sprint
            preds = [o for o in obs['observations'] if str(o.get('type', '')).capitalize() == 'Predator' and 'angle' in o]
            if preds:
                q = min(preds, key=lambda o: o['distance'])
                if abs(q['angle']) > 0.05:
                    # move_direction is relative to the CURRENT heading (engine moves before turning): keep the intended
                    # world direction of the step, then turn to face the predator
                    act['turn_angle'] = q['angle']
        return act, ws

    def _agent_inner(self, obs, mem, p):
        speed, sprint = obs['speed'], obs['sprint_speed']
        energy, age = obs['energy'], obs['age']
        half_cone = obs['vision_angle'] / 2; hearing = obs['hearing_radius']
        kinds = {'Fruit': [], 'Predator': [], 'Tree': [], 'Agent': []}
        edges = []
        for o in obs['observations']:
            t = str(o.get('type', '')).capitalize()      # official simulator sends 'Fruit'; be tolerant of 'fruit'
            if t == 'Edge':
                edges.append(o)
            elif t in kinds and 'distance' in o and 'angle' in o:
                kinds[t].append(o)
        for lst in kinds.values():
            lst.sort(key=lambda o: o['distance'])
        act = dict(agent_id=obs['agent_id'], move_distance=0.0, move_direction=0.0, turn_angle=0.0, spawn_agent=False)
        # memories: real sightings overwrite; a remembered tree we should be able to hear but cannot is gone
        real_trees = [t for t in kinds['Tree'] if not t.get('relayed')]
        if p['tree_mem_steps'] > 0:
            if real_trees:
                t0 = real_trees[0]; mem.tree_mem = [t0['distance'] * math.cos(t0['angle']), t0['distance'] * math.sin(t0['angle']), self.step]
            elif mem.tree_mem is not None:
                dm = math.hypot(mem.tree_mem[0], mem.tree_mem[1])
                if dm < hearing * 0.9 or self.step - mem.tree_mem[2] > p['tree_mem_steps']: mem.tree_mem = None
                elif not kinds['Tree'] and not kinds['Fruit'] and not kinds['Predator']:
                    kinds['Tree'].append({'type': 'Tree', 'distance': dm, 'angle': math.atan2(mem.tree_mem[1], mem.tree_mem[0]), 'remembered': True})
        if p['pred_mem_steps'] > 0:
            if kinds['Predator']:
                q = kinds['Predator'][0]; mem.pred_mem = [q['distance'] * math.cos(q['angle']), q['distance'] * math.sin(q['angle']), self.step]
            elif mem.pred_mem is not None and self.step - mem.pred_mem[2] > p['pred_mem_steps']:
                mem.pred_mem = None

        wants_spawn = (energy > p['spawn_energy']) or (age > p['spawn_late_age'] and energy > 100 + p['late_margin'])
        if p['spawn_needs_food'] > 0.5 and not (kinds['Tree'] or kinds['Fruit']):
            wants_spawn = False
        if p['spawn_fruit_dist'] > 0 and energy <= p['spawn_energy'] * 1.3 and not (kinds['Fruit'] and kinds['Fruit'][0]['distance'] < p['spawn_fruit_dist']):
            wants_spawn = False                           # (very rich parents may still spawn anywhere)

        # 1. predators
        if kinds['Predator']:
            pr = kinds['Predator'][0]; d, a = pr['distance'], pr['angle']
            if obs['agent_id'] in self.decoys and energy > p['decoy_min_energy']:
                # stay the nearest target: face it and hold near decoy_hold_dist, only sprinting if it closes to the kill range
                hold = p['decoy_hold_dist']
                if d < p['chase_dist'] * 0.55:
                    act['move_distance'] = sprint if energy > obs['max_energy'] * p['sprint_reserve'] else speed
                    act['move_direction'] = wrap(a + math.pi)
                elif d > hold * 1.6:
                    act['move_distance'] = speed * 0.6; act['move_direction'] = 0.0
                act['turn_angle'] = a
                mem.fruits = []
                return act, wants_spawn and p['decoy_sen'] > 0.5   # a doomed decoy must still make its last child
            near_fruit = kinds['Fruit'][0] if kinds['Fruit'] else None
            if (p['fruit_first_dist'] > 0 and near_fruit is not None and d >= p['chase_dist']
                    and near_fruit['distance'] <= p['fruit_first_dist']):
                f = near_fruit                              # two steps for 20-60 energy beats retreating on an empty tank
                act['move_distance'] = min(speed, max(0.0, f['distance'] - 4.0)); act['move_direction'] = f['angle']
                if abs(f['angle']) > half_cone: act['turn_angle'] = f['angle']
                mem.fruits = []; mem.mode = 'scan'; mem.counter = 0
                return act, wants_spawn
            if p['mpc'] > 0.5 and d < p['mpc_dist']:
                plan = mpc_escape(obs, kinds, p, speed, sprint, energy)
                if plan is not None:
                    act['move_distance'] = plan[2]; act['move_direction'] = plan[1]; act['turn_angle'] = a
                    mem.fruits = []; mem.mode = 'scan'; mem.counter = 0
                    return act, False
            if d < p['chase_dist']:                       # it chases anything this close: run, facing it
                v = sprint if (p['flee_speed'] <= 0 or d < p['flee_close_dist']) else min(sprint, max(speed, p['flee_speed']))
                if p['flee_smart'] > 0.5 and speed >= p['flee_walk_min'] and d >= p['flee_close_dist']:
                    v = speed                             # walking already outpaces its sprint, at a fraction of the cost
                    away = wrap(a + math.pi)
                    if p['edge_flee'] > 0.5 and edges: away = clear_direction(edges, away, p['edge_flee_dist'])
                    act['move_distance'] = v; act['move_direction'] = away; act['turn_angle'] = a
                    mem.fruits = []; mem.mode = 'scan'; mem.counter = 0
                    return act, False
                act['move_distance'] = v if energy > obs['max_energy'] * p['sprint_reserve'] else speed
                away = wrap(a + math.pi)
                if p['edge_flee'] > 0.5 and edges: away = clear_direction(edges, away, p['edge_flee_dist'])
                act['move_direction'] = away; act['turn_angle'] = a
                mem.fruits = []; mem.mode = 'scan'; mem.counter = 0
                return act, False
            if p['stealth'] > 0.5 and 'rel_dir' in pr:
                r = pr['rel_dir']                          # our bearing as seen from the predator, relative to its heading
                in_cone = abs(r) < math.pi / 6 + p['cone_margin'] and d < 260
                if in_cone:                                # step sideways out of its cone (perpendicular to its heading), keep facing it
                    heading = wrap(a + math.pi - r)        # predator heading expressed in our frame
                    side = 1.0 if r >= 0 else -1.0
                    act['move_distance'] = speed; act['move_direction'] = wrap(heading + side * math.pi / 2); act['turn_angle'] = a
                    mem.fruits = []
                    return act, False
                if d < p['hear_margin']:                   # it cannot see us but could hear us: drift away quietly
                    act['move_distance'] = speed * 0.5; act['move_direction'] = wrap(a + math.pi)
                    mem.fruits = []
                    return act, False
                # otherwise it cannot perceive us: keep foraging
            elif d < (p['danger_vision_frac'] * float(obs.get('vision_range', 200.0)) if p['danger_vision_frac'] > 0 else p['danger_dist']):
                away = wrap(a + math.pi)
                if p['retreat_to_food'] > 0.5:
                    food = (kinds['Fruit'] or kinds['Tree'])
                    if food:
                        off = wrap(food[0]['angle'] - away)
                        away = wrap(away + max(-math.pi / 4, min(math.pi / 4, off)))
                act['move_distance'] = speed; act['move_direction'] = away; act['turn_angle'] = a
                mem.fruits = []
                return act, False

        # 2. spacing: drift away from a too-close neighbour (cheap: half walking speed, no turn)
        spacing = p['spacing_dist']
        if p['spacing_vision'] > 0.5:
            spacing = min(p['spacing_max'], spacing * float(obs.get('vision_range', 200.0)) / 200.0)
        if kinds['Agent'] and kinds['Agent'][0]['distance'] < spacing and energy > (p['hungry_rel'] * obs['max_energy'] if p['hungry_rel'] > 0 else p['hungry_energy']) * 0.5:
            ag = kinds['Agent'][0]
            act['move_distance'] = speed * 0.5; act['move_direction'] = wrap(ag['angle'] + math.pi)
            mem.fruits = []
            return act, wants_spawn

        # 3. fruit
        fruits = [f for f in kinds['Fruit'] if f['distance'] < p['max_fruit_dist']]
        if p['fruit_pred_gap'] > 0 and kinds['Predator'] and fruits:
            safe = []
            for f in fruits:
                fx, fy = f['distance'] * math.cos(f['angle']), f['distance'] * math.sin(f['angle'])
                ok = True
                for q in kinds['Predator']:
                    qx, qy = q['distance'] * math.cos(q['angle']), q['distance'] * math.sin(q['angle'])
                    if math.hypot(fx - qx, fy - qy) < p['fruit_pred_gap']: ok = False; break
                if ok: safe.append(f)
            if safe: fruits = safe                          # only give the fruit up when a safe one exists
        target = None
        hungry_at = p['hungry_rel'] * obs['max_energy'] if p['hungry_rel'] > 0 else p['hungry_energy']
        if fruits:
            hungry = energy < hungry_at
            if p['ripe_if_alone'] > 0 and kinds['Agent'] and kinds['Agent'][0]['distance'] < p['ripe_if_alone']:
                hungry = True                              # a neighbour is close enough to take it: eat now, unripe
            if hungry or mem.mode != 'wait':
                target = fruits[0]
            else:
                target = self._ripe_fruit(fruits, mem, p)
        if target is not None:
            d, a = target['distance'], target['angle']
            act['move_distance'] = min(speed, max(0.0, d - 4.0)); act['move_direction'] = a
            if abs(a) > half_cone and d > hearing * 0.8:
                act['turn_angle'] = a
            if act['move_distance'] > 0:
                mem.fruits = []; mem.mode = 'scan'; mem.counter = 0
            return act, wants_spawn

        # 4. trees: approach, then wait and scan
        if kinds['Tree']:
            tr = kinds['Tree'][0]; d, a = tr['distance'], tr['angle']
            if d > p['tree_hold']:
                act['move_distance'] = min(speed, d - p['tree_hold'] * 0.8); act['move_direction'] = a
                if abs(a) > half_cone:
                    act['turn_angle'] = a
                mem.fruits = []; mem.mode = 'scan'; mem.counter = 0
                return act, wants_spawn
            if mem.mode != 'wait':
                mem.mode = 'wait'; mem.fruits = []
            act['turn_angle'] = p['scan_turn']
            self._remember_fruits(fruits, mem)
            return act, wants_spawn

        # 5. nothing useful in sight: leave bad biomes, follow the herd or explore cheaply
        mem.fruits = []
        if p['biome_flee'] > 0.5 and BIOME_TREE_RATE.get(str(obs.get('biome', '')).lower(), 1.0) <= p['biome_flee_rate']:
            act['move_distance'] = speed * p['explore_speed']   # straight line out of a barren biome, no scanning detour
            mem.mode = 'walk'; mem.counter = 0
            return self._avoid_pred(act, mem, p), wants_spawn
        if p['river_exit'] > 0.5 and str(obs.get('biome', '')).lower() in ('river', 'swamp'):
            act['move_distance'] = speed; mem.mode = 'scan'; mem.counter = 0
            return act, wants_spawn
        if p['follow_agents'] > 0.5 and kinds['Agent']:
            ag = kinds['Agent'][0]
            if ag['distance'] > 30:
                act['move_distance'] = speed * p['explore_speed']; act['move_direction'] = ag['angle']
                act['turn_angle'] = ag['angle'] if abs(ag['angle']) > half_cone else 0.0
                mem.mode = 'scan'; mem.counter = 0
                return self._avoid_pred(act, mem, p), wants_spawn
        if mem.mode == 'wait':
            mem.mode = 'scan'; mem.counter = 0
        if mem.mode == 'scan':
            act['turn_angle'] = p['scan_turn']; mem.counter += 1
            if mem.counter >= p['scan_steps']:
                mem.mode = 'walk'; mem.counter = 0
        else:
            act['move_distance'] = speed * p['explore_speed']; mem.counter += 1
            walk_steps = p['walk_steps'] * (float(obs.get('vision_range', 200.0)) / 200.0 if p['walk_vision'] > 0.5 else 1.0)
            if mem.counter >= walk_steps:
                mem.mode = 'scan'; mem.counter = 0
        return self._avoid_pred(act, mem, p), wants_spawn

    def _avoid_pred(self, act, mem, p):
        """Exploration/following must not head towards a remembered predator: mirror the direction away from it."""
        if mem.pred_mem is None or act['move_distance'] <= 0: return act
        bearing = math.atan2(mem.pred_mem[1], mem.pred_mem[0])
        if abs(wrap(act['move_direction'] - bearing)) < p['pred_avoid_angle']:
            act['move_direction'] = wrap(bearing + math.pi); act['turn_angle'] = 0.0
        return act

    # -------------------------------------------------------------- fruit ripening memory (only while standing still)
    def _remember_fruits(self, fruits, mem):
        seen = []
        for f in fruits:
            ang = wrap(mem.heading + f['angle']); d = f['distance']
            for rec in mem.fruits:
                if abs(wrap(rec[0] - ang)) < 0.08 and abs(rec[1] - d) < 6:
                    seen.append(rec); break
            else:
                seen.append([ang, d, self.step])
        mem.fruits = seen

    def _ripe_fruit(self, fruits, mem, p):
        self._remember_fruits(fruits, mem)
        best = None
        for f in fruits:
            ang = wrap(mem.heading + f['angle'])
            for rec in mem.fruits:
                if abs(wrap(rec[0] - ang)) < 0.08 and abs(rec[1] - f['distance']) < 6:
                    if self.step - rec[2] >= p['fruit_ripe_steps'] and (best is None or f['distance'] < best['distance']):
                        best = f
                    break
        return best

def sample_params(rng, center, sigma=0.3):
    """Log-normal perturbation of every parameter inside its bounds; binary flags flip with probability sigma/2."""
    out = {}
    for k, v in center.items():
        lo, hi = PARAM_BOUNDS[k]
        if (lo, hi) == (0, 1):
            out[k] = float(1 - v) if rng.random() < sigma / 2 else float(v)
        else:
            x = v * math.exp(rng.gauss(0, sigma)); out[k] = float(min(hi, max(lo, x)))
    return out
