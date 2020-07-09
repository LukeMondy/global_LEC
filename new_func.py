#!/usr/bin/env python3
# coding: utf-8

# run me with: mpiexec -n $CPUS python3 -m mpi4py.futures mpi_tas.py

import meshio
import numpy as np
import time

from LECMesh import LECMesh

infile = "australia/data/TAS.vtk"
outfile = "outputs/costed_tas_parallel.vtk"

def elevation_only(self, current, _next):
    # Only take into account elevation changes for costs
    if current == _next:
        return 0
    return int(abs(self.mesh.point_data['Z'][current] - self.mesh.point_data['Z'][_next]) )


mesh = meshio.read(infile)
points_above_sealevel = np.nonzero(mesh.point_data['Z'] >= 0)[0]
print("Total starting points available: ", points_above_sealevel.shape[0], flush=True)

# Setup the output file:
mesh.point_data['cost'] = np.zeros_like(mesh.point_data['Z'])

max_fuel = 2500 # A smaller value means visiting far fewer nodes, so it speeds things up a lot

    
if __name__ == "__main__":
    lm = LECMesh(mesh, max_fuel, neighbours_cache_size = points_above_sealevel.shape[0])

    all_costs = []
    for res in map(lm.get_dist_from_point, points_above_sealevel):
        all_costs.append(res)
        print('Progress: {: 4.3f} %'.format((len(all_costs)/points_above_sealevel.shape[0]) * 100))

    print(len(all_costs))

    for i in all_costs:
        mesh.point_data['cost'][i[0]] = i[1]
    meshio.write(outfile, mesh)
