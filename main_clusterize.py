from cluster import *
from urllib3.connectionpool import xrange
from scrape import *
from mpi4py import MPI
import sys
import numpy as np
import pandas as pd
import io
import os

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
    csv_data_full, csv_data_clean = load_csv_data("./ktu_didziuju_ld2_final.csv")

data_full_node_chunk = comm.scatter(chunkify(csv_data_full, num_nodes), root=root_rank)
data_clean_node_chunk = comm.scatter(chunkify(csv_data_clean, num_nodes), root=root_rank)
data_normal_node_chunk = normalize_data(data_clean_node_chunk)
# chunks_with_ids = []
# for i in range(0, len(chunks)):
#     object = {"id": i, "chunk": chunks[i]}
#     chunks_with_ids.append(object)

# chunks_normal = chunkify(data_normal_full, num_nodes)
# data_normal_node_chunk = comm.scatter(chunks_normal, root=root_rank)
# data_normal_node_chunk = data_normal_node_chunk_object["chunk"]
# node_chunk_id = data_normal_node_chunk_object["id"]

comm.barrier()
print(f"{node_rank}/{num_nodes} have data with {len(data_normal_node_chunk)} rows")

rows_per_node, num_columns = data_normal_node_chunk.shape

# skaičiuojame pradinius klasterių centro pozicijų taškus
begin_time = time.time()
cluster_coords = dict()
numberOfClusters = 7
numberOfAttempts = 100
best_point_cluster_assgn = []
best_sum_error = sys.maxsize
for attempt in range(0, numberOfAttempts):
    for i in range(0, numberOfClusters):
        cluster_coords[i] = [None] * num_columns

        random_node_rank = random.randint(0, num_nodes)
        random_node_rank = comm.bcast(random_node_rank, root=root_rank)
        num_index = random.randint(0, rows_per_node - 1)
        cluster_coord = data_normal_node_chunk[num_index]
        allnodes_cluster_coord = comm.gather(cluster_coord, root=root_rank)
        if node_rank == root_rank:
            cluster_coord = allnodes_cluster_coord[0]
            cluster_coords[i] = cluster_coord

    comm.barrier()
    cluster_coords = comm.bcast(cluster_coords, root=root_rank)

    # pradžioje visi taškai priskirti klasteriui 0
    node_point_cluster_assgn = np.zeros((rows_per_node, 1))
    node_point_cluster_dist = calc_dist_bulk(data_normal_node_chunk, None, cluster_coords[0])

    while True:
        node_num_point_changes = 0
        for k in range(0, numberOfClusters):
            cluster_coord = cluster_coords[k]

            # skaičiuojame kurie taškai yra artimesnis šiam klasteriui lyginant su tuo klasteriu, kuriam jie yra priskirti
            # ir atnaujinam taškų priskirimus klasteriams
            pointDistancesToCluster = calc_dist_bulk(data_normal_node_chunk, None, cluster_coord)
            mask = pointDistancesToCluster < node_point_cluster_dist
            node_point_cluster_dist[mask] = pointDistancesToCluster[mask]
            node_point_cluster_assgn[mask] = k
            # print(f"{node_rank}/{num_nodes} assigned {len(node_point_cluster_assgn[mask])} points to cluster {k}")
            node_num_point_changes += len(node_point_cluster_assgn[mask])

        # skaičiuojam naujas klasterių centrų pozicijas
        for k in range(0, numberOfClusters):
            mask = list(np.concatenate(node_point_cluster_assgn == k).flat)
            new_cluster_coords = []
            for dimension in range(0, len(cluster_coords[k])):
                cluster_points = data_normal_node_chunk[mask, dimension]
                if len(cluster_points) > 0:
                    new_cluster_coords.append(cluster_points.mean())
                else:
                    new_cluster_coords.append(0.0)
            cluster_coords[k] = new_cluster_coords

        all_nodes_cluster_coords = comm.gather(cluster_coords, root=root_rank)
        if node_rank == root_rank:
            for k in range(0, numberOfClusters):
                new_cluster_coords = [0] * num_columns
                for node_cluster_coords in all_nodes_cluster_coords:
                    coords = node_cluster_coords[k]
                    for col in range(0, num_columns):
                        new_cluster_coords[col] += coords[col] / num_nodes
                cluster_coords[k] = new_cluster_coords
        cluster_coords = comm.bcast(cluster_coords, root=root_rank)

        all_nodes_num_point_changes_arr = comm.gather(node_num_point_changes, root=root_rank)
        all_node_num_point_changes = None
        if node_rank == root_rank:
            all_node_num_point_changes = sum(all_nodes_num_point_changes_arr)
            # print(f"{node_rank}/{num_nodes} num point changes {all_node_num_point_changes}")

        everybody_num_point_changes = comm.bcast(all_node_num_point_changes, root=root_rank)
        if everybody_num_point_changes == 0:
            break

    # skaičiuojam bendrą paklaidą
    node_sum_error = 0
    for k in range(0, numberOfClusters):
        mask = list(np.concatenate(node_point_cluster_assgn == k).flat)
        if len(data_normal_node_chunk[mask]) == 0:
            continue
        pointDistancesToCluster = calc_dist_bulk(data_normal_node_chunk, mask, cluster_coords[k])
        node_sum_error += sum(pointDistancesToCluster)


    all_nodes_sum_error_arr = comm.gather(node_sum_error, root=root_rank)
    all_nodes_sum_error = None
    if node_rank == root_rank:
        all_nodes_sum_error = sum(all_nodes_sum_error_arr)
    all_nodes_sum_error = comm.bcast(all_nodes_sum_error, root=root_rank)

    # jei šio mėginimo rezultatas atvedė į mažesnę paklaidą, išsaugom kaip geriausią
    if all_nodes_sum_error < best_sum_error:
        previous_best_sum_error = best_sum_error
        best_sum_error = all_nodes_sum_error
        best_point_cluster_assgn = node_point_cluster_assgn
        # all_nodes_point_cluster_assgn_arr = comm.gather(node_point_cluster_assgn, root=root_rank)
        if node_rank == root_rank:
            # all_nodes_point_cluster_assgn = [item for sublist in all_nodes_point_cluster_assgn_arr for item in sublist]  # flatten
            # best_point_cluster_assgn = all_nodes_point_cluster_assgn
            print(f"{node_rank}/{num_nodes} Updated best model from error {previous_best_sum_error} to error {all_nodes_sum_error}")


if node_rank == root_rank:
    print(f"{node_rank}/{num_nodes} k means finished, saving to file")
    if os.path.exists("./cluster_data.csv"):
        os.remove("./cluster_data.csv")
    file = open("./cluster_data.csv", "w+")
    file.close()

for rank in range(0, num_nodes):
    comm.barrier()
    if rank == node_rank:
        # write to file
        # print(f"{len(best_point_cluster_assgn)} ir {len(data_full_node_chunk)}")
        # clusterized_data = pd.DataFrame(list(filter(lambda e: e["numberOfClusters"] == 7, best_point_cluster_assgn))[0]["cluster_assgn"], columns=["cluster"])
        clusterized_data = pd.DataFrame(best_point_cluster_assgn, columns=["cluster"])
        data_full_node_chunk.index = np.arange(0, len(data_full_node_chunk))
        csv_data_full2 = pd.concat([clusterized_data, data_full_node_chunk], axis=1)

        tmp_file = io.StringIO()
        csv_data_full2.to_csv(tmp_file, header=(node_rank == 0), sep=";")
        node_final_str = tmp_file.getvalue()

        file = open("./cluster_data.csv", "a")
        file.write(node_final_str)
        file.write("\n")
        file.close()

end_time = time.time()
duration = end_time - begin_time
if node_rank == root_rank:
    print(f"Total took {duration} seconds")