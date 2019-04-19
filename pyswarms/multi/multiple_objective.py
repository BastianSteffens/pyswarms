# -*- coding: utf-8 -*-

# https://ieeexplore.ieee.org/abstract/document/1304847
# https://www.sciencedirect.com/science/article/pii/S1568494618303557
import logging
import numpy as np
from time import sleep

from ..base import SwarmOptimizer
from ..utils import Reporter
from ..backend.handlers import BoundaryHandler, VelocityHandler
from ..backend.generators import create_swarm
from ..backend.operators import compute_velocity, compute_position
from ..backend.multi_swarms import create_multi_swarm

class MOPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        bh_strategy="periodic",
        velocity_clamp=None,
        vh_strategy="unmodified",
        center=1.00,
        ftol=-np.inf,
        init_pos=None,
    ):
        super(MOPSO, self).__init__(
            n_particles=n_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds,
            velocity_clamp=velocity_clamp,
            center=center,
            ftol=ftol,
            init_pos=init_pos,
        )

        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Initialize the resettable attributes
        self.reset()
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.name = __name__
        
    def optimize(self, objective_func, iters, fast=False, **kwargs):
        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=logging.INFO,
        )
        
        self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
        self.swarm.update_archive()
        self.swarm.pbest_pos = self.swarm.position
        self.swarm.pbest_cost = self.swarm.current_cost
        for _ in self.rep.pbar(iters, self.name):
            if not fast:
                sleep(0.01)
            # fmt: off
            self.swarm.current_cost = objective_func(self.swarm.position, **kwargs)
            self.swarm.update_personal_best()
            self.swarm.update_archive()
            self.swarm.best_pos, self.swarm.best_cost = self.swarm.generate_global_best()
            # fmt: on
            self.rep.hook(best_cost=self.swarm.archive.get_best_cost(), mean_cost=self.swarm.archive.aggregate(np.mean))
            # Save to history TODO
#             hist = self.ToHistory(
#                 best_cost=self.swarm.best_cost,
#                 mean_pbest_cost=np.mean(self.swarm.pbest_cost),
#                 mean_neighbor_cost=self.swarm.best_cost,
#                 position=self.swarm.position,
#                 velocity=self.swarm.velocity,
#             )
#             self._populate_history(hist)
#             # Verify stop criteria based on the relative acceptable cost ftol
#             relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
#             if (
#                 np.abs(self.swarm.best_cost - best_cost_yet_found)
#                 < relative_measure
#             ):
#                 break
            # Perform velocity and position updates
            self.swarm.velocity = compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = compute_position(
                self.swarm, self.bounds, self.bh
            )
        front = self.swarm.archive.get_front()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | front size: {}".format(
                len(front)
            ),
            lvl=logging.INFO,
        )
        return front
    
    def reset(self):
        """Reset the attributes of the optimizer

        All variables/atributes that will be re-initialized when this
        method is defined here. Note that this method
        can be called twice: (1) during initialization, and (2) when
        this is called from an instance.

        It is good practice to keep the number of resettable
        attributes at a minimum. This is to prevent spamming the same
        object instance with various swarm definitions.

        Normally, swarm definitions are as atomic as possible, where
        each type of swarm is contained in its own instance. Thus, the
        following attributes are the only ones recommended to be
        resettable:

        * Swarm position matrix (self.pos)
        * Velocity matrix (self.pos)
        * Best scores and positions (gbest_cost, gbest_pos, etc.)

        Otherwise, consider using positional arguments.
        """
        # Initialize history lists
        self.cost_history = []
        self.mean_pbest_history = []
        self.mean_neighbor_history = []
        self.pos_history = []
        self.velocity_history = []
    
        # Initialize the swarm
        self.swarm = create_multi_swarm( # To be changed 
            n_particles=self.n_particles,
            dimensions=self.dimensions,
            bounds=self.bounds,
            center=self.center,
            init_pos=self.init_pos,
            clamp=self.velocity_clamp,
            options=self.options,
        )