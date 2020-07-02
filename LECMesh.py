#!/usr/bin/env python
# coding: utf-8

import numpy as np
from multiprocessing import Pool
import time
from functools import lru_cache
from queue import PriorityQueue


class LECMesh(object):

    def __init__(self, mesh, max_fuel = 2000, travel_cost_function = None, neighbours_cache_size = None, other_cache_size = None):
        self.mesh = mesh
        self.max_fuel = max_fuel

        if travel_cost_function:
            # Allow the user to define their own cost function
            self.travel_cost_function = travel_cost_function
        else:
            # but if they don't, use the default
            self.travel_cost_function = self.elevation_only

        if not neighbours_cache_size:
            # If no cache size was supplied, make it as big as the mesh
            neighbours_cache_size = self.mesh.point_data['Z'].shape[0]

        if not other_cache_size:
            other_cache_size = neighbours_cache_size

        # Apply a LRU cache to all the hot functions
        self.neighbours_func = lru_cache(maxsize=neighbours_cache_size)(self.graph_neighbours)
        self.travel_cost_function = lru_cache(maxsize=other_cache_size)(self.travel_cost_function)
        self.dist_func = lru_cache(maxsize=other_cache_size)(self.distance)


    def distance(self, current, _next):
        # from https://stackoverflow.com/a/1401828
        if current == _next:
            return 0
        return int(np.linalg.norm(self.mesh.points[current]-self.mesh.points[_next]))
    

    # We need to define a way to calculate cost
    def elevation_only(self, current, _next):
        # Only take into account elevation changes for costs
        if current == _next:
            return 0
        return int(abs(self.mesh.point_data['Z'][current] - self.mesh.point_data['Z'][_next]) + self.dist_func(current, _next)*0.004)


    def graph_neighbours(self, current):
        # Get all the other points from the cells that have the current point in them.
        points = np.unique(self.mesh.cells['triangle'][np.where(self.mesh.cells['triangle']==current)[0]])
        # remove the current point from these results:
        points = points[points != current]
        
        # Get the elevation of those connected points
        elevations = self.mesh.point_data['Z'][points]
        # Return a list of connected points, as long as they are above sea-level
        return points[elevations >= 0]
        
            
    def cost_search(self, start):
        # Some code and inspiration from http://theory.stanford.edu/~amitp/GameProgramming/ImplementationNotes.html

        frontier = PriorityQueue()  # The priority queue means that we can find the least cost path to continue
        frontier.put(start, 0)      # from, along any path, meaning the resulting paths should always be the least-cost
                                    # path to get to that point.
        
        # Setup the data structures. Keys are point indexes. 
        came_from = {}
        cost_so_far = {}
        dist_so_far = {}

        # Init the data structures
        came_from[start] = None
        cost_so_far[start] = 0
        dist_so_far[start] = 0
        
        while not frontier.empty():
            current = frontier.get()
            for _next in self.neighbours_func(current):
                # Calculate the cost of going to this new point.
                new_cost = cost_so_far[current] + self.travel_cost_function(current, _next)
                # Calculate the eulerian distance to this new point.
                new_dist = dist_so_far[current] + self.dist_func(current, _next)

                # The max_distance check tells the algorithm to stop once we start getting too far away from the starting point.
                if (_next not in cost_so_far or new_cost < cost_so_far[_next]) and new_cost <= self.max_fuel:
                    cost_so_far[_next] = new_cost
                    dist_so_far[_next] = new_dist
                    priority = new_cost
                    frontier.put(_next, priority)
                    came_from[_next] = current

        return came_from, cost_so_far, dist_so_far


    def get_total_distance_for_all_paths_to_point(self, start):
        came_from, cost_so_far, dist_so_far = self.cost_search(start)
        
        # Find the edge nodes, and add up their costs to get the total
        total_dist = 0
        for k in came_from.keys():             # For all the points we've visited,
            if k not in came_from.values():    # Find all the points that haven't been 'came_from' (meaning they're the end of a path)
                total_dist += dist_so_far[k]
                
        return total_dist


    def get_dist_from_point(self, point):
        # Return a tuple of (the point id, it's cost)
        return (point, self.get_total_distance_for_all_paths_to_point(point))
