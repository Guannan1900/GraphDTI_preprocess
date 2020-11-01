import networkx as nx
import argparse
import time as time
import pandas as pd
import os

def read_file(filename):
    f = open(filename, 'r')
    list_ = [i.strip("\n").strip(" ") for i in list(f)]
    f.close()
    return list_

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outFolder',
                        required=False,
                        default='target_node_shortest_path/',
                        help='path of output folder.')
    parser.add_argument('--infile',
                        required=True,
                        default='input_list/',
                        help='input node list.')
    parser.add_argument('--graph',
                        required=True,
                        default='undirected_graph.gexf',
                        help='graph file.')

    return parser.parse_args()

def shortest_path(start_node, G, out_folder):

    '''
    :param start_node: Input node
    :param G: input graph
    :return: short path of the start node in graph
    '''

    length, path = nx.single_source_dijkstra(G, start_node, cutoff=5, weight='weight')
    dest_index = list(length.keys())
    shortest_distance = list(length.values())
    d = {'destination_index': dest_index, 'shortest_path_distance': shortest_distance}
    df_out = pd.DataFrame(data=d)
    df_out.to_csv(out_folder + 'path_dist_index_' + str(start_node) + '.csv', sep=',', index=False)


if __name__ == '__main__':

    start = time.time()
    parse = getArgs()
    output_folder = parse.outFolder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    graph_name = parse.graph
    G = nx.read_gexf('undirected_graph.gexf')
    input_path = parse.infile
    file_name = os.listdir(input_path)
    for file in file_name:
        protein_list = read_file(input_path + file)
        print(protein_list)
        [shortest_path(start_node, G, output_folder) for start_node in protein_list]
    print('end of all')
    del G
    end = time.time()
    print('vector time elapsed :' + str(end - start))
