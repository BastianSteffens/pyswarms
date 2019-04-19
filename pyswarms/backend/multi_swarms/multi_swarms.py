from pyswarms.backend.generators import generate_swarm, generate_velocity
from .pareto_front import ParetoFront, is_dominant

def create_multi_swarm(
    n_particles,
    dimensions,
    options={},
    bounds=None,
    center=1.0,
    init_pos=None,
    clamp=None,
):
    position = generate_swarm(
        n_particles,
        dimensions,
        bounds=bounds,
        center=center,
        init_pos=init_pos,
    )

    velocity = generate_velocity(n_particles, dimensions, clamp=clamp)
    archive = ParetoFront(options=options)
    return MultiSwarm(position, velocity, archive=archive, options=options)

# Import modules
import random
import numpy as np
from attr import attrib, attrs
from attr.validators import instance_of
from pyswarms.backend.swarms import Swarm

@attrs
class MultiSwarm(Swarm):
    archive = attrib(type=ParetoFront, default=None)
    
    # Surrogate for utility functions
    best_cost = attrib(
        type=np.ndarray,
        default=np.array([]),
        validator=instance_of(np.ndarray),
    )
    
    def update_archive(self):
        self.archive.insert_all(zip(self.position, self.current_cost))
    
    def update_personal_best(self):
        for i in range(self.n_particles):
            if(is_dominant(self.pbest_cost[i],self.current_cost[i])):
                self.pbest_cost[i] = self.current_cost[i]
                self.pbest_pos[i] = self.position[i]
                
    def generate_global_best(self):
        return self.archive.get_random_item()
