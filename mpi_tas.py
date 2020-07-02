#!/usr/bin/env python3
# coding: utf-8

# run me with: mpiexec -n 4 python3 mpi_tas.py

import meshio
import numpy as np
import time

from mpi4py import MPI

from LECMesh import LECMesh


comm = MPI.COMM_WORLD
rank = comm.Get_rank()


infile = "australia/data/TAS.vtk"
outfile = "outputs/costed_tas_parallel.vtk"
mesh = meshio.read(infile)

max_fuel = 2500 # A smaller value means visiting far fewer nodes, so it speeds things up a lot

points_above_sealevel = np.nonzero(mesh.point_data['Z'] >= 0)[0]
if rank == 0:
    print("Total starting points available: ", points_above_sealevel.shape[0], flush=True)
    
    # The first CPU can figure out how to split up all the data to send out to the others
    point_list_split = np.array_split(points_above_sealevel, comm.size, axis=0)

    # Setup the output file:
    mesh.point_data['cost'] = np.zeros_like(mesh.point_data['Z'])
    
else:
    point_list_split = None
    
# All the CPUs get their list of points to work on
my_points = comm.scatter(point_list_split, root=0)

print("Proc {} will work on {} points".format(rank, my_points.shape[0]), flush=True)


# Setup the LECMesh functions via class instantiation. Each CPU gets their own object
lm = LECMesh(mesh, max_fuel, neighbours_cache_size = points_above_sealevel.shape[0])


def process_points_with_feedback(points):
    """
    This function is overly complicated, only because it's nice to see progress
    """
    all_costs = []
    start = 0
    inc = 50
    stop = inc
    
    while start < points.shape[0]:
        start_time = time.time()
        
        # the real work is being done here
        costs = list(map(lm.get_dist_from_point, points[start:stop]))
        # you could replace this whole function with just:
        #     return list(map(lm.get_dist_from_point, points))
        # I think...
        
        print("{}: points: {: 7d} to {: 7d} took {:.3f} s. % complete: {:.3f}".format(rank, 
                                                                                 start, 
                                                                                 stop, 
                                                                                 time.time() - start_time, 
                                                                                 (stop/(len(points)-1)) * 100), flush=True)

        # Save the data
        all_costs.extend(costs)
        
        # move to the next chunk of data
        start += inc
        stop += inc
        if stop >= len(points):
            stop = len(points)-1
            
    return all_costs


my_costs = process_points_with_feedback(my_points)

# Sync all the processors up
cost_list = comm.gather(my_costs)

# The first CPU can now write out all the data
if comm.rank == 0:
    for costs in cost_list:
        for i in costs:
            mesh.point_data['cost'][i[0]] = i[1]
    meshio.write(outfile, mesh)
