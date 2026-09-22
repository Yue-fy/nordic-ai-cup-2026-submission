# Survival Simulator

## Method

A deterministic rule policy (`cup/survival/rules.py`, class `RulePolicy`)
controls every agent from the official observation list only. The 88
parameters that define the submitted policy are in `params/survival_v8.json`.

Main mechanisms:

- **Foraging.** Agents alternate scan and walk modes at a reduced cruising
  speed (`explore_speed` ≈ 0.34 of max) because movement energy dominates the
  budget. Ripe fruit is approached; unripe trees are remembered for
  `fruit_ripe_steps` and revisited.
- **Predator avoidance.** Threats are detected by vision and hearing
  (`danger_dist`, `hear_margin`). Escape uses a short-horizon rollout
  (`mpc_escape`) over candidate headings, sprinting only when the energy
  reserve allows (`sprint_reserve`), and otherwise walking away once walking
  speed exceeds the predator's.
- **Population control.** Reproduction requires energy above `spawn_energy`
  (150) and a per-agent `birth_interval` (20 steps); the population target
  decays with game time (`pop0`, `pop_half`).
- **Deathbed spawning.** Old-age energy drain is detected from unexplained
  energy loss (`drain_thresh`); a senescent agent reproduces immediately,
  ignoring the population cap, so lineage energy is not lost.
- **Selective breeding.** Spawn permission is granted in trait-score order
  (vision, hearing, cone, sprint, speed weights `breed_w_*`); only the top
  `breed_quantile` ≈ 48 % of agents may reproduce.
- **Time ramps.** Selected thresholds are multiplied linearly over the first
  `late_t` = 1500 simulated seconds (`late_*` multipliers), e.g. the spawn
  threshold doubles late in the game when trees become scarce.

## How it was tuned

Every change was compared against the incumbent on paired seeds in the
official simulator (200 to 3000 games per comparison, bootstrap confidence
interval on the paired difference). Continuous parameters were searched with
an evolution strategy on 64 common seeds and confirmed on fresh seeds.
Local mean per game: 1145 (std 292 over 3000 games).

## Run

```bash
pip install -r requirements.txt
SURVIVAL_PARAMS=params/survival_v8.json uvicorn cup.survival.server:app --host 0.0.0.0 --port 9052 --workers 1
```

The server resets policy state whenever `sim_time` goes backwards, so it can
serve consecutive games on one endpoint.

## Reproducing the tuning

`tuning/` holds the harness that produced the parameters. It expects the
official simulator checked out at `../upstream/survival-simulator` (see
`cup/common.py`) and Python 3.10+ with `numpy` (and `cma` for `cmaes_driver.py`).

| Script | Purpose |
|---|---|
| `run_games.py` | Play whole games on a seed range with a parameter file (multiprocessing) |
| `confirm.py` | Paired comparison candidate vs incumbent on shared seeds with a bootstrap CI; the promotion test used for every version |
| `search.py`, `sweep.py` | Evolution-strategy search and one-factor sweeps over continuous parameters |
| `es_driver.py`, `cmaes_driver.py` | Generation drivers with common random numbers and fresh-seed confirmation; they submit Slurm jobs and need the partition names adapted |
| `diag_lifetimes.py`, `diag_encounters.py` | Death-cause and predator-encounter diagnostics that motivated the deathbed and breeding rules |
| `fork.py`, `fork_aa.py`, `rollout_advantage.py` | State-fork counterfactual tooling and the A/A control that showed it to be unreliable |

`params/lineage/` contains every deployed version from v2 to v8. Example
promotion test:

```bash
python -m tuning.confirm --candidate params/survival_v8.json --incumbent params/lineage/survival_v6_deployed.json --seeds 90001-90200 --procs 40
```
