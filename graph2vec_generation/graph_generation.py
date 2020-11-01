import networkx as nx
import time as time
import pandas as pd
import argparse

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-infile',
                        required=False,
                        default='data/',
                        help='input file path.')

    return parser.parse_args()

def graph_gene_inx(prot_list, prot_name, prot_inx):
    '''
    function: get protein index of the graph
    '''

    target_list = []
    for i in range(len(prot_list)):
        if prot_list[i] in prot_name:
            inx = prot_name.index(prot_list[i])
            target_list.append(prot_inx[inx])

    return target_list

def graph_generation(inpath):
    '''
    :param inpath: input path
    :function: generate graph of human interactions network
    '''

    inputfile = inpath + 'HUMAN_interactions_sample.csv'
    input_inx = inpath + 'node_index_sample.csv'

    network_df = pd.read_csv(inputfile)
    inx_df = pd.read_csv(input_inx)
    print(inx_df)

    prot1_list = network_df['protein1'].tolist()
    prot2_list = network_df['protein2'].tolist()
    prot_name = inx_df['Protein'].tolist()
    prot_inx = inx_df['idx'].tolist()

    protein1 = graph_gene_inx(prot1_list, prot_name, prot_inx)
    protein2 = graph_gene_inx(prot2_list, prot_name, prot_inx)
    combined_score_df = network_df['combined_score'].map(lambda x: 1 / (x))
    weight = combined_score_df.tolist()

    G = nx.Graph()
    G.add_nodes_from(protein1)
    for i in range(len(protein1)):
        temp = (protein1[i], protein2[i])
        G.add_edges_from([temp], weight=weight[i])
    nx.write_gexf(G, 'undirected_graph.gexf')



if __name__ == '__main__':

    start = time.time()
    parse = getArgs()
    input_path = parse.infile
    graph_generation(input_path)
    end = time.time()
    print('vector time elapsed :' + str(end - start))