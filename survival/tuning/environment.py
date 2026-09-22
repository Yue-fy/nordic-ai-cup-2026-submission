"""Use the official simulator unchanged, including seeded rendering RNG draws."""
import os
import sys
from cup.common import UPSTREAM
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, str(UPSTREAM / 'survival-simulator'))
from src.core import SimulationCore
from src.utils.DTOs import ActionRequest

class World:
    def __init__(self, seed, max_steps=30001):
        self.seed = seed
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.sim = SimulationCore(seed=self.seed)
        self.seed += 1
        self.steps = 0
        # Official HTTP loop starts with no observations, then an empty step.
        self.state = self.sim.step([])
        return self.state

    def step(self, actions):
        previous = self.sim.env.score
        self.state = self.sim.step([(a['agent_id'], ActionRequest(**a)) for a in actions])
        self.steps += 1
        terminated = not self.state['num_agents'] or self.state['sim_time'] > 3000
        truncated = self.steps >= self.max_steps and not terminated
        return self.state, self.state['score'] - previous, terminated, truncated

def worker(connection, seed, max_steps):
    try:
        world = World(seed, max_steps)
        connection.send(world.state)
        while True:
            command, payload = connection.recv()
            if command == 'close':
                break
            if command == 'step':
                connection.send(world.step(payload))
            elif command == 'reset':
                connection.send(world.reset())
    except BaseException:
        import traceback
        connection.send({'worker_error': traceback.format_exc()})
    finally:
        connection.close()
