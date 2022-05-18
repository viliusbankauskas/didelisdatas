from cluster import *
from urllib3.connectionpool import xrange
from scrape import *
from functools import reduce
from mpi4py import MPI
import sys
import numpy as np
import pandas as pd

def chunkify(lst,n):
    return [lst[i::n] for i in xrange(n)]

comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
node_rank = MPI.COMM_WORLD.Get_rank()
num_nodes = MPI.COMM_WORLD.Get_size()
root_rank = 0

# load data and distribute to all nodes
csv_data_full = []
csv_data_clean = []
if node_rank == root_rank:
    csv_data_full, csv_data_clean = load_csv_data("./cluster_data.csv")

# asd = chunkify(csv_data_full, 100)
data_full_node_chunk = comm.scatter(chunkify(csv_data_full, num_nodes), root=root_rank)
data_clean_node_chunk = comm.scatter(chunkify(csv_data_clean, num_nodes), root=root_rank)
data_normal_node_chunk = normalize_data(data_clean_node_chunk)
# pandas.DataFrame.dropna(axis = 0, how ='any', thresh = None, subset = None, inplace=False)

# knn
query_qwikiid = "Q5366980"

data_row_full = 0
data_row_normal = 0
if sum(data_full_node_chunk["qwikidata_id"] == query_qwikiid) >= 1:
    print(f"{node_rank}/{num_nodes} I have this object")

    row_index = data_full_node_chunk['qwikidata_id'].tolist().index(query_qwikiid)
    data_row_full = data_full_node_chunk.iloc[[row_index]]
    data_row_normal = data_normal_node_chunk[row_index]

data_row_normal_arr = comm.gather((data_row_full, data_row_normal), root=root_rank)
if node_rank == root_rank:
    for e in data_row_normal_arr:
        if type(e) is tuple and e[0] is not 0:
            data_row_full, data_row_normal = e
data_row_full, data_row_normal = comm.bcast((data_row_full, data_row_normal), root=root_rank)

if node_rank == root_rank:
    print(f"{node_rank}/{num_nodes} Searching for similar rows to this:")
    print(data_row_full.to_markdown())

# skaičiuojamas atstumas nuo šio taško iki visų kitų taškų
node_pointDistancesToRow = calc_dist_bulk(data_normal_node_chunk, None, data_row_normal)

data_full_node_chunk.index = np.arange(0, len(data_full_node_chunk))
node_knn_data = pd.concat([pd.DataFrame(node_pointDistancesToRow, columns = ["distance"]), data_full_node_chunk], axis = 1)
node_knn_data = node_knn_data.sort_values('distance')
node_knn_data = node_knn_data[0:5]

all_nodes_knn_data_arr = comm.gather(node_knn_data, root=root_rank)
if node_rank == root_rank:
    all_nodes_knn_data = pd.concat(all_nodes_knn_data_arr, axis=0)
    all_nodes_knn_data = all_nodes_knn_data.sort_values('distance')
    print("Search found these results:")
    print(all_nodes_knn_data.to_markdown())