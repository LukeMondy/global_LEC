# global_LEC

This is a repo for looking how to efficiently perform global LEC analysis.

## Things to look at

- single_point_analysis.ipynb is a nice way to explore how the algorithm works when looking at a single starting point. It visualises the paths taken, and the costs it calculates.

- australia.ipynb shows how to use the same code to perform an LEC analysis on the scale of Australia, using parallel map functions for speedup.

- ugs.py has all the code for doing the slightly modified Dijkstra's algorithm

## Data origins

The data was taken from the escape models notebooks. 

## How to run:

Checkout this repo, and `cd` into it. Then run:

```
docker run -it -w /host -v $PWD:/host geodels/escape-docker
```

and then connect to the running JuPyter Notebook server.
