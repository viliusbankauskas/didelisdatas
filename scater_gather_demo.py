#!/usr/bin/env python
"""
scather/gather demo
"""

from mpi4py import MPI
import sys

import numpy as np
comm = MPI.COMM_WORLD

name = MPI.Get_processor_name()
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

root_rank = 0
if rank == root_rank: # use one process for communication
  data = np.random.randint(1,42,size)
  print("data at %d rank" % rank,data)
else:
  data = None

comm.barrier() # not necessary just for nice printing

print("%03d/%03d data before scater: %s" % (rank,size,data))

if rank == 0:
    print("**********")
comm.barrier()

data = comm.scatter(data,root=root_rank)

random_number = np.random.randint(0,10)
print("%03d/%03d data after scater = %s, random number %d (sum %d)" % (rank,size,data,random_number,data+random_number))

comm.barrier()
if rank == 0:
  print("******")
  print("Performing gather to root_rank");

comm.barrier()


data = comm.gather(data + random_number,root_rank)


print("%03d/%03d data after gather: %s" % (rank,size,data))