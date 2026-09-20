"""Agent endpoint for the Survival Simulator: rule policy, never raises, resets between games.

    SURVIVAL_PARAMS=path/to/best.json uvicorn cup.survival.server:app --host 0.0.0.0 --port 9052
"""
import json, logging, os, threading, time
from fastapi import FastAPI, Body
from .rules import RulePolicy

log = logging.getLogger('survival')
params = json.load(open(os.environ['SURVIVAL_PARAMS'])) if os.environ.get('SURVIVAL_PARAMS') else None
policy = RulePolicy(params); lock = threading.Lock()
# SURVIVAL_LOG=<path> appends one line per request: wall clock, sim_time, agents, score, whether we reset, and the
# handling time. The official request carries no game id, so this is the only way to tell whether the three games of
# an attempt are replayed one after another or concurrently against the same endpoint.
REQLOG = os.environ.get('SURVIVAL_LOG')
app = FastAPI(title='Survival agent (rules)')

@app.get('/')
def index():
    return {'message': 'Agent endpoint running!', 'policy': 'rules', 'params': policy.params}

@app.post('/predict')
def predict(step: dict = Body(...)):
    t0 = time.perf_counter(); status = step.get('agent_status') or []
    st = float(step.get('sim_time', 0.0))
    if not status:
        with lock:
            was = policy.last_sim_time; policy.reset()      # first (empty) request of a game
        if REQLOG:
            with open(REQLOG, 'a') as f: f.write(f'{time.time():.4f} {st:.1f} 0 empty prev={was:.1f} {1000*(time.perf_counter()-t0):.2f}\n')
        return {'actions': []}
    try:
        with lock:
            prev = policy.last_sim_time
            actions = policy.act(status, st, step.get('n_agents'))
        if REQLOG:
            with open(REQLOG, 'a') as f:
                f.write(f'{time.time():.4f} {st:.1f} {len(status)} ok prev={prev:.1f} {1000*(time.perf_counter()-t0):.2f}\n')
        return {'actions': actions}
    except Exception:
        log.exception('policy failed; returning idle actions')
        return {'actions': [dict(agent_id=o['agent_id'], move_distance=0.0, move_direction=0.0, turn_angle=0.0,
                                 spawn_agent=False) for o in status if 'agent_id' in o]}
