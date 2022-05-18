from urllib3.connectionpool import xrange
import gc
from scrape import *
from mpi4py import MPI
import sys
import numpy as np
import time
# import ctypes
# libc = ctypes.CDLL("libc.so.6")
def chunkify(lst,n):
    return [lst[i::n] for i in xrange(n)]

osmIdsCityNamesMap = {
    # 11.3k objects
    "44915": "Italy Milan",
    "271110": "Netherlands Amsterdam",
    "439840": "Czechia Prague",
    "407489": "Luxembourg Luxembourg",
    "41485": "Italy Rome",
    "1529146": "Lithuania Vilnius",
    "2220322": "Monaco Monaco",
    "111968": "US San Francisco",
    "112382": "US Stockton",
    "175905": "US New York",
    "122604": "US Chicago",
    "2688911": "US Houston",
    "111257": "US Phoenix",
    "188022": "US Philadelphia",
    "253556": "US San Antonio",
    "253832": "US San Diego",
    "6571629": "US Dallas",
    "112143": "US San Jose",
    "10264792": "Hong Kong",
    "92277": "Thailand Bangkok",
    "536780": "Singapore",
    "3766483": "UAE Dubai",
    "223474": "Turkey Istanbul",
    "435514": "Czechia Republic Prague",
    "2297418": "South Korea Seoul"
}

comm = MPI.COMM_WORLD
name = MPI.Get_processor_name()
node_rank = MPI.COMM_WORLD.Get_rank()
num_nodes = MPI.COMM_WORLD.Get_size()
root_rank = 0
start_timestamp = time.time()
comm.barrier()
#######################################
# load city polygons
taskOsmIdsChunks = chunkify(list(osmIdsCityNamesMap.keys()), num_nodes)
nodeOsmIdPolygonTasksChunk = comm.scatter(taskOsmIdsChunks, root=root_rank)
print(f"{node_rank}/{num_nodes} will load city polygon: {nodeOsmIdPolygonTasksChunk}")

polygon_result = None
for osmIdTask in nodeOsmIdPolygonTasksChunk:
    polygon_result = {osmIdTask: get_polygon_for_city_osmid(osmIdTask)}

osmid_polygon_results_map = {}
if node_rank == root_rank:
    for osmid in osmIdsCityNamesMap:
        osmid_polygon_results_map[osmid] = get_polygon_for_city_osmid(osmid)


osmid_polygon_results = comm.allgather(osmid_polygon_results_map)[0]

# print(osmid_polygon_results)
# osmid_polygon_results_map = {}
# for result in osmid_polygon_results or {}:
#     if result is not None and len(result) > 0:
#         key = list(result.keys())[0]
#         osmid_polygon_results_map[key] = result[key]

#######################################
# scrape
scrape_tasks = []
for city_osmid in osmIdsCityNamesMap.keys():
    #for i in range(0, 10*10):
    for i in range(0, 3*3):
        task = {
            "osmid": city_osmid,
            "block_index": i
        }
        scrape_tasks.append(task)

import random
random.shuffle(scrape_tasks)


task_chunks = chunkify(scrape_tasks, num_nodes)

node_scrape_tasks = comm.scatter(task_chunks, root=root_rank)

print(f"{node_rank}/{num_nodes} will perform {len(node_scrape_tasks)} scrape tasks")

node_scraped_objects = []
i = 1
for scrape_task in node_scrape_tasks:
    city_polygon_shape = osmid_polygon_results[scrape_task["osmid"]]
    city_name = osmIdsCityNamesMap[scrape_task["osmid"]]
    overpass_raw = get_raw_overpass_for_city_block(scrape_task["osmid"], city_polygon_shape, scrape_task["block_index"])
    objects = process_overpass_document(overpass_raw, city_name, city_polygon_shape)
    for obj in objects:
        node_scraped_objects.append(obj)
    print(f"{node_rank}/{num_nodes} finished scraping task {i}/{len(node_scrape_tasks)}, obtained {len(objects)} objects")
    i += 1
    del overpass_raw
    del objects
    gc.collect()
    # libc.malloc_trim(0)

print(f"{node_rank}/{num_nodes} finished scraping, obtained {len(node_scraped_objects)} objects")
#######################################
# reduce objects from all nodes into one array and format into one csv file
comm.barrier()

def reduce_objects_list(a, b, _):
    res = []
    for e in a:
        if type(e) is list:
            for ee in e:
                res.append(ee)
        else:
            res.append(e)

    for e in b:
        if type(e) is list:
            for ee in e:
                res.append(ee)
        else:
            res.append(e)

    res = list(dict.fromkeys(res))
    return res

op_reduce_objects_list = MPI.Op.Create(reduce_objects_list, commute=True)
all_objects = comm.reduce(node_scraped_objects, op_reduce_objects_list, root=root_rank)
if node_rank == root_rank:
    print(f"{node_rank}/{num_nodes} have total {len(all_objects)} objects from all nodes")
    with open('./ktu_didziuju_ld2_final.csv', 'w') as f:
        header = "qwikidata_id;city_name;lat;lng;is_shop;is_tourism;is_leisure;is_other;instance_of;pageview_count;registered_contributors_count;anonymous_contributors_count;num_wikipedia_lang_pages;has_image;description;image_files"
        f.write(header)
        f.write("\n")
        for obj in all_objects:
            f.write(obj)
            f.write("\n")

    end_timestamp = time.time()
    total_duration = end_timestamp - start_timestamp
    print(f"{node_rank}/{num_nodes} total time was {total_duration} seconds")