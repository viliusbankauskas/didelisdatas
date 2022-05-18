#!pip install plotly ipympl
#!pip install gower
import pandas as pd
import math
from pandas import DataFrame
import random
import numpy as np
import sys

def load_csv_data(filepath):
    csv_data = pd.read_csv(filepath, sep = ';', dtype = {'instance_of': str, 'pageview_count': float, 'registered_contributors_count': float, 'anonymous_contributors_count': float, 'num_wikipedia_lang_pages': float, 'is_shop': int, 'is_tourism': int, 'is_leisure': int, 'is_other': int, 'has_image': int})
    csv_data2 = csv_data[['pageview_count', 'registered_contributors_count', 'anonymous_contributors_count', 'num_wikipedia_lang_pages', 'instance_of', 'is_shop', 'is_tourism', 'is_leisure', 'is_other', 'has_image']]

    count = {}
    instance_of_types = []
    for row in csv_data2['instance_of']:
        for instance in row.split(','):
            if instance in count:
                count[instance] += 1
            else:
                count[instance] = 1
            instance_of_types.append(instance)

    instance_of_types = list(set(instance_of_types))
    instance_of_types.sort(key=lambda a: count[a], reverse=True)
    instance_of_types = instance_of_types[0:30]

    # for instance in instance_of_types:
    #     one_hot = csv_data2['instance_of'].str.contains(instance)
    #     csv_data2 = csv_data2 + one_hot

    instance_one_hot = csv_data2['instance_of'].str.get_dummies(sep=",")
    for column_name in instance_one_hot.columns:
        ok = False
        for type in instance_of_types:
            if type == column_name:
                ok = True
                break
        if not ok:
            del instance_one_hot[column_name]
    instance_one_hot = instance_one_hot.add_prefix('instance_')
    csv_data2 = pd.concat([csv_data2, instance_one_hot], axis = 1)
    csv_data2 = csv_data2.drop(columns = ["instance_of"])
    return csv_data, csv_data2

def normalize_data(data):
    # np.seterr(divide='ignore', invalid='ignore')
    data_normal = data.to_numpy()
    num_rows, num_columns = data_normal.shape
    for col in range(0, num_columns):
        stdval = np.std(data_normal[:,col])
        if stdval != 0.0:
            data_normal[:,col]=(data_normal[:,col] - np.mean(data_normal[:,col])) / stdval
        else:
            data_normal[:, col] = np.zeros((1, num_rows))
    return data_normal

def calc_dist_bulk(data, mask, centroid):
    sum = 0
    for col in range(0, len(centroid)):
        if mask is not None:
            sum += (data[mask, col] - centroid[col])**2
        else:
            sum += (data[:, col] - centroid[col])**2
    return np.sqrt(sum)


def perform_kmeans(data_normal, numberOfClusters, numberOfAttempts):
    num_rows, num_columns = data_normal.shape
    best_point_cluster_assgn = []
    best_point_cluster_dist = []
    best_sum_error = sys.maxsize
    for attempts in range(0, numberOfAttempts):
        # skaičiuojame pradinius klasterių centro pozicijų taškus
        cluster_coords = dict()
        for i in range(0, numberOfClusters):
            cluster_coords[i] = [None] * num_columns
            # pusę kartų mėginam vieną klasterių centro pradžios taško priskyrimo metodą, pusę kartų kitą
            if attempts < numberOfAttempts / 2:
                num_index = random.randint(0, num_rows - 1)
                cluster_coords[i] = data_normal[num_index]
            else:
                for j in range(0, num_columns):
                    column_min = np.min(data_normal[:, j])
                    column_max = np.max(data_normal[:, j])
                    coord_val = random.uniform(column_min, column_max)
                    cluster_coords[i][j] = coord_val

        # pradžioje visi taškai priskirti klasteriui 0
        point_cluster_assgn = np.zeros((num_rows, 1))
        point_cluster_dist = calc_dist_bulk(data_normal, None, cluster_coords[0])

        while True:
            num_point_changes = 0
            for k in range(0, numberOfClusters):
                cluster_coord = cluster_coords[k]

                # skaičiuojame kurie taškai yra artimesnis šiam klasteriui lyginant su tuo klasteriu, kuriam jie yra priskirti
                # ir atnaujinam taškų priskirimus klasteriams
                pointDistancesToCluster = calc_dist_bulk(data_normal, None, cluster_coord)
                mask = pointDistancesToCluster < point_cluster_dist
                point_cluster_dist[mask] = pointDistancesToCluster[mask]
                point_cluster_assgn[mask] = k
                # print(f"Assigned {len(point_cluster_assgn[mask])} points to cluster {k}")
                num_point_changes += len(point_cluster_assgn[mask])

            # skaičiuojam naujas klasterių centrų pozicijas
            for k in range(0, numberOfClusters):
                mask = list(np.concatenate(point_cluster_assgn == k).flat)
                new_cluster_coords = []
                for dimension in range(0, len(cluster_coords[k])):
                    cluster_points = data_normal[mask, dimension]
                    if len(cluster_points) > 0:
                        new_cluster_coords.append(cluster_points.mean())
                    else:
                        new_cluster_coords.append(0.0)
                cluster_coords[k] = new_cluster_coords

            if num_point_changes == 0:
                break

        # skaičiuojam bendrą paklaidą
        sum_error = 0
        for k in range(0, numberOfClusters):
            mask = list(np.concatenate(point_cluster_assgn == k).flat)
            if len(data_normal[mask]) == 0:
                continue
            pointDistancesToCluster = calc_dist_bulk(data_normal, mask, cluster_coords[k])
            sum_error += sum(pointDistancesToCluster)
        # print(sum_error)

        # jei šio mėginimo rezultatas atvedė į mažesnę paklaidą, išsaugom kaip geriausią
        if sum_error < best_sum_error:
            previous_best_sum_error = best_sum_error
            best_sum_error = sum_error
            best_point_cluster_dist = point_cluster_dist
            best_point_cluster_assgn = point_cluster_assgn
            #print(f"Updated best model from error {previous_best_sum_error} to error {sum_error}")


    #print("k-means finished")
    return best_point_cluster_assgn, best_sum_error
