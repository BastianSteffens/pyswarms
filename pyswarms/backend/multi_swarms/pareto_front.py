import random
import numpy as np

def is_dominant(left,right):
    """
    Returns if the right vector dominates the left vector
    
    Params
    ------    
    left : numpy.ndarray
        Vector to test against for Pareto dominance :code:`(n_dimensions)`
    right : numpy.ndarray
        Vector to test for Pareto dominance :code:`(n_dimensions)`
    
    Returns
    -------
    boolean
        Is the right vector dominates the left vector
    """
    return (right <= left).all()


class ParetoFront:
    def _init_grid(self):
        return np.ndarray(shape=tuple([self.grid_size] * self.dimensions),dtype=np.ndarray)
    
    def _grid_insert(self, grid, grid_weight, item):
        grid_position = self.objective_cell(item[1])
        if(grid[grid_position] == None):
            grid[grid_position] = [item]
            grid_weight += self.grid_weight_coef
        else:
            l = len(grid[grid_position])
            grid[grid_position] += [item]
            grid_weight += self.grid_weight_coef / (l+1) - self.grid_weight_coef / l
            
        return grid, grid_weight
        
    
    def __init__(self,options=None):
        self.dimensions = options['obj_dimensions']
        self.grid_size = options['grid_size']
        self.objective_bounds = options['obj_bounds']
        self.grid = self._init_grid()
        self.grid_weight_coef = options['grid_weight_coef']
        self.grid_weight = 0

    def objective_cell(self,objective):
        return tuple(np.array([min(self.grid_size-1,(x[0]/(x[1][1]-x[1][0])-x[1][0])*self.grid_size) for x in zip(objective,self.objective_bounds)], dtype=np.int))
        
    def insert_all(self,items):
        for item in items:
            self.insert(item)
        
    def insert(self,item):
        """
        Params
        ------
        item : tuple(tuple * param_dimensions, tuple * dimensions)
            Position and cost to insert
        """
        item = tuple([np.array(x) for x in item]) # convert to comparable np.array
        new_grid = self._init_grid()
        new_grid_weight = 0
        for index in np.ndindex(self.grid.shape):
            if(self.grid[index] == None):
                continue
            for v in self.grid[index]:
                if(is_dominant(item[1],v[1])):
                    return # not part of Pareto Front
                if(not is_dominant(v[1],item[1])):
                    new_grid, new_grid_weight = self._grid_insert(new_grid,new_grid_weight,v)
        self.grid, self.grid_weight = self._grid_insert(new_grid,new_grid_weight,item)
        
    def get_random_item(self):
        n = random.random() * self.grid_weight
        w = 0
        for index in np.ndindex(self.grid.shape):
            if(self.grid[index] == None):
                continue
            l = len(self.grid[index])
            w += self.grid_weight_coef / l
            if(w >= n):
                # Choose cell, take random item
                return self.grid[index][random.randint(0,l-1)]
        return None
    
    def get_front(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        return flatten([self.grid[index] for index in np.ndindex(self.grid.shape) if self.grid[index] != None])
    
    def get_best_cost(self):
        return self.aggregate(np.min)

    def aggregate(self, fun):
        costs = list(list(zip(*self.get_front()))[1])
        return fun(list(zip(*costs)),axis=1)
                
    