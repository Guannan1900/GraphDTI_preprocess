# -*- coding: utf-8 -*-
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn import metrics
import argparse
from time import time

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-outpath',
                        required=False,
                        default='results/',
                        help='path of result folder.')
    parser.add_argument('-inpath',
                        required=True,
                        default='processed_data/',
                        help='input file path.')
    parser.add_argument('-op',
                        required=True,
                        default='scaled_PMD',
                        choices=['original_PMD', 'scaled_PMD', 'features'],
                        help="'original_PMD','scaled_PMD','features'")

    return parser.parse_args()

def calculate_silscore(data, klist, option):
    """
    :param data: precomputed distance
    :param klist: number of clusters
    :return: Silhouette Coefficient
    """
    if option == 'scaled_PMD':
        method = 'precomputed'
    elif option == 'original_PMD':
        method = 'precomputed'
    elif option == 'features':
        method = 'euclidean'
    print(method)
    SSE = []
    sils = []
    for k in klist:
        kmedoids = KMedoids(n_clusters=k, init='k-medoids++', metric=method).fit(data)
        pred_clusters = kmedoids.labels_
        # SSE.append(kmedoids.inertia_)
        sil_score = metrics.silhouette_score(data, kmedoids.labels_, metric=method)
        # print(sil_score)
        sils.append(sil_score)

    # return sils, SSE
    return sils


def optimize_k(path, output, op):
    """
    :param path: inputdata path
    :param output: output result
    :param op: options for precomputed matrix: original PMD, scaled PMD, Mol2vec+ProtVec
    :return: Silhouette Coefficient
    """
    s = time()
    if op == 'scaled_PMD':
        distance_ori = np.load(path + 'pmd_total.npy', allow_pickle=True)
        n_size = distance_ori.shape[0]
        # print(n_size)
        print(np.where(distance_ori == np.amax(distance_ori)))
        # distance[distance == np.sqrt(2)] = 1.41420
        a = np.sqrt(2)
        r = a * np.ones((n_size, n_size))
        distance_e = distance_ori / (r-distance_ori)
        print(np.amin(distance_e), np.amax(distance_e))
        print(distance_e.shape)
        distance = distance_e
        name = 'sil_repeat_scaled_pmd.npy'
    elif op == 'original_PMD':
        distance_ori = np.load(path + 'pmd_total.npy', allow_pickle=True)
        distance = distance_ori
        name = 'sil_repeat_original_pmd.npy'
    elif op == 'features':
        distance_ori = np.load(path + 'features.npy', allow_pickle=True)
        distance = distance_ori
        name = 'sil_repeat_features.npy'
    print(distance.shape)
    print('get pmd matrix {:.4f} seconds'.format(time() - s))

    repeat_num = 10
    sil_repeat = []
    k_list = [2,50,100,150,200,250,300,350,400,450,500]
    for j in range(repeat_num):
        sils2 = calculate_silscore(distance, k_list, op)
        sil_repeat.append(sils2)
        # SSE_repeat.append(SSE2)
    sil_repeat = np.array(sil_repeat)
    # SSE_repeat = np.array(SSE_repeat)
    print(sil_repeat)
    print('get sh score {:.4f} seconds'.format(time() - s))

    np.save(output + name, sil_repeat)
    # np.save(output + 'SSE_repeat_pmd.npy', SSE_repeat)
    print('time spend processing pmd cluster {:.4f} seconds'.format(time() - s))


if __name__ == "__main__":

    parse = getArgs()
    input_path = parse.inpath
    output_path = parse.outpath
    op = parse.op
    optimize_k(input_path, output_path, op)







