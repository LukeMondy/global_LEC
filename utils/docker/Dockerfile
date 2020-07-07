FROM python:3.7-buster

RUN apt update -q \
    && apt-get install -y --no-install-recommends \
        libopenmpi-dev \
        openmpi-bin \
    && apt-get clean autoclean \
    && apt-get autoremove --purge --yes \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        methodtools \
        meshio \
        mpi4py 


