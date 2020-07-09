#!/usr/bin/env python3
# coding: utf-8

# run me with: mpiexec -n $CPUS python3 -m mpi4py.futures mpi_tas.py

import meshio
import numpy as np
import time

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from LECMesh import LECMesh

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

infile = "australia/data/TAS.vtk"
outfile = "outputs/costed_tas_parallel.vtk"

def parprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

def init(point):
    global lm
    try:
        lm
    except NameError:
        print("{}: creating lm obj".format(rank))
        lm = LECMesh(mesh, max_fuel, travel_cost_function = elevation_only, neighbours_cache_size = points_above_sealevel.shape[0])
    # Setup the LECMesh functions via class instantiation. 

    return lm.get_dist_from_point(point)

def elevation_only(self, current, _next):
    # Only take into account elevation changes for costs
    if current == _next:
        return 0
    return int(abs(self.mesh.point_data['Z'][current] - self.mesh.point_data['Z'][_next]) )

mesh = meshio.read(infile)
points_above_sealevel = np.nonzero(mesh.point_data['Z'] >= 0)[0]
parprint("Total starting points available: ", points_above_sealevel.shape[0], flush=True)

# Setup the output file:
mesh.point_data['cost'] = np.zeros_like(mesh.point_data['Z'])

max_fuel = 2500 # A smaller value means visiting far fewer nodes, so it speeds things up a lot

    
if __name__ == "__main__":

    all_costs = []
    with MPIPoolExecutor() as ex:
        for res in ex.map(init, points_above_sealevel, chunksize=20):
            all_costs.append(res)
            parprint('Progress: {: 4.3f} %'.format((len(all_costs)/points_above_sealevel.shape[0]) * 100))

    parprint(len(all_costs))

    # The first CPU can now write out all the data
    if rank == 0:
        for i in all_costs:
            mesh.point_data['cost'][i[0]] = i[1]
        meshio.write(outfile, mesh)
