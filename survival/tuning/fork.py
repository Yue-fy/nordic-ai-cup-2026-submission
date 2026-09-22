"""Fork a live simulator state so the same situation can be replayed under different actions.

Earlier counterfactual work replayed from the same seed and found that identical policies could still diverge by
hundreds of points, because the engine iterates over sets whose order depends on object identity. Forking the state
object avoids that entirely: both branches continue from the same objects, each with its own copy of the RNG.

The environment carries pygame Surfaces that cannot be deep-copied and are only used for rendering, so they are
detached during the copy and shared afterwards (nothing in the headless step path draws to them).
"""
import copy

SURFACES = ('biome_surface', 'static_surface', 'shadow_surface', 'obstacle_surface', 'world_surface',
            'vision_screen', 'leaf_screen')

def fork_world(world):
    """Return an independent copy of a cup.survival.environment.World that can be stepped separately."""
    env = world.sim.env
    stash = {name: getattr(env, name, None) for name in SURFACES}
    for name in stash: 
        if hasattr(env, name): setattr(env, name, None)
    try:
        clone = copy.deepcopy(world)
    finally:
        for name, surf in stash.items():
            if surf is not None: setattr(env, name, surf)
    cenv = clone.sim.env
    for name, surf in stash.items():
        if surf is not None: setattr(cenv, name, surf)     # shared, render-only
    return clone
