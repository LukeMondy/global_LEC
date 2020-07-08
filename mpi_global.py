#!/usr/bin/env python3
# coding: utf-8

# run me with: mpiexec -n $CPUS python3 -m mpi4py.futures mpi_global.py

import meshio
import numpy as np
import time
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

from LECMesh import LECMesh

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#############################################################################
# We need these variables to to be in the global scope, so all CPUs have them
infile = "earth/data/globe.vtk"
outfile = "outputs/costed_globe.vtk"

mesh = meshio.read(infile)
points_above_sealevel = np.nonzero(mesh.point_data['Z'] >= 0)[0]
parprint("Total starting points available: ", points_above_sealevel.shape[0], flush=True)

# Setup the output file:
mesh.point_data['cost'] = np.zeros_like(mesh.point_data['Z'])

max_fuel = 2500 # A smaller value means visiting far fewer nodes, so it speeds things up a lot
#############################################################################


def parprint(*args, **kwargs):
    """
    Print if on CPU 0
    """
    if rank == 0:
        print(*args, **kwargs)


def init(point):
    """
    This function sets up an LECMesh object for each CPU.
    From: https://groups.google.com/d/msg/mpi4py/k7Hc6raaWgY/PSbrygDTAQAJ
    """
    global lm
    try:
        lm
    except NameError:
        print("{}: creating lm obj".format(rank))
        # Setup the LECMesh functions via class instantiation. 
        lm = LECMesh(mesh, max_fuel, neighbours_cache_size = points_above_sealevel.shape[0])
    return lm.get_dist_from_point(point)


if __name__ == "__main__":

    all_costs = []
    with MPIPoolExecutor() as ex:
        for res in ex.map(init, points_above_sealevel, chunksize=20):
            all_costs.append(res)
            parprint('Progress: {: 4.3f} %'.format((len(all_costs)/points_above_sealevel.shape[0]) * 100))

    # The first CPU can now write out all the data
    parprint("Writing out the data...")
    if rank == 0:
        for i in all_costs:
            mesh.point_data['cost'][i[0]] = i[1]
        meshio.write(outfile, mesh)
    parprint("Done")
