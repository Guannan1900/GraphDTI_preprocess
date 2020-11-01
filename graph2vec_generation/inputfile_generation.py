import os
import pandas as pd
import statistics
import json
import numpy as np
import time
import argparse

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-inpath',
                        required=False,
                        default='data_sample/',
                        help='input file path.')

    return parser.parse_args()

def get_gene_expression(sig_intance_path, ensp_index_path,\
                        sig_id_path, gene_exp_path, state):

    input_shortpath = 'target_node_shortest_path/'
    index_path = 'data_sample/node_index_new.csv'
    prot_index = pd.read_csv(index_path)
    print(prot_index)
    prot_index_new = prot_index['Protein'].tolist()
    print(prot_index_new[0:10])
    sig_intance = pd.read_csv(sig_intance_path, usecols=['ensp_id', 'sig_id'])
    print(sig_intance)

    sig_id_index = pd.read_csv(sig_id_path)
    ensp_index = pd.read_csv(ensp_index_path)
    gene_exp = np.load(gene_exp_path, allow_pickle=True)
    sig_id = sig_id_index['sig_id'].tolist()
    gene_ensp_id = ensp_index['ensp_id'].tolist()
    # gene_exp_sig = gene_exp[:, 1]
    # print(gene_exp_sig)
    ensp_id = list(set(sig_intance['ensp_id'].tolist()))
    print('ensp number is:', len(ensp_id))
    target_index = [prot_index_new.index(i) for i in ensp_id]

    for i in range(len(target_index)):
        print('protein name is:', ensp_id[i], target_index[i])
        shortpath_df = pd.read_csv(input_shortpath + 'path_dist_index_' + str(target_index[i]) +'.csv', nrows=500)
        target_sig_id_df = sig_intance.loc[sig_intance['ensp_id'] == ensp_id[i]].reset_index(drop=True)
        target_sig_id_list = target_sig_id_df['sig_id'].tolist()
        print('the number of sig id is:', len(target_sig_id_list))
        data = {'sig_id': target_sig_id_list, 'inx': [x for x in range(len(target_sig_id_list))]}
        df_inx = pd.DataFrame(data=data)
        out_path = 'sig_index_new/' + state
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        df_inx_path = out_path + 'index_' + str(target_index[i]) + '.csv'
        df_inx.to_csv(df_inx_path, index=False)

        list_k = list(range(10, 80, 10))
        dic2 = {}
        for k in list_k:
            print('k is :', k)
            prot_neighbor_index = shortpath_df['destination_index'][0:k + 1].tolist()
            prot_neighbor_distance = shortpath_df['shortest_path_distance'][0:k + 1].tolist()
            edge_list = [[target_index[i], neighbor_index] for neighbor_index in prot_neighbor_index]
            print('number of sig_id:', len(target_sig_id_list))
            for j in range(len(target_sig_id_list)):
                # print('target protein is :', target_index[i], 'sig id is: ', target_sig_id_list[j])
                if target_sig_id_list[j] in sig_id:
                    sig_id_inx = sig_id.index(target_sig_id_list[j])
                    gene_exp_sig_list = gene_exp[:, sig_id_inx]
                    neighobr_gene_exp = []
                    for m in range(len(prot_neighbor_index)):
                        index_tmp = prot_index.loc[prot_index['idx'] == prot_neighbor_index[m]].reset_index(drop=True)
                        prot_neghbour_name = index_tmp['Protein'][0]
                        # print('protein neighbor is :', prot_neighbor_index[m],\
                        #       'ensp id is :', prot_neghbour_name)
                        if prot_neghbour_name in gene_ensp_id:
                            ensp_inx = gene_ensp_id.index(prot_neghbour_name)
                            neighbor_gene_tmp = gene_exp_sig_list[ensp_inx]
                            neighobr_gene_exp.append(neighbor_gene_tmp)
                        else:
                            total_list = gene_exp_sig_list
                            media = statistics.median(total_list)
                            neighobr_gene_exp.append(media)
                    # print('length of gene expression: ', len(prot_neighbor_index), len(neighobr_gene_exp),
                    #       len(prot_neighbor_distance))
                    d = {'gene': prot_neighbor_index, 'expression': neighobr_gene_exp,
                         'distance': prot_neighbor_distance}
                    df_features = pd.DataFrame(data=d)
                    dic1 = df_features.set_index('gene').T.to_dict('list')
                    dic2['edges'] = edge_list
                    dic2['features'] = dic1
                    dir = 'input_json/' + str(k) + '/' + state + str(target_index[i]) + '/'
                    # print(dir)
                    if not os.path.exists(dir):
                        os.makedirs(dir)
                    with open(dir + "temp.json", 'w') as jsonfile:
                        json.dump(dic2, jsonfile)  # change a dictionary to json
                    dst = dir + str(j) + ".json"
                    scr = dir + "temp.json"
                    os.rename(scr, dst)
                else:
                    print('sig_id cannot be found: ', target_sig_id_list[j])

if __name__ == "__main__":

    start = time.time()
    parse = getArgs()
    input_path = parse.inpath

    pos_sig_intance_path = input_path + 'pos_sig_instance_new.csv'
    neg_sig_intance_path = input_path + 'neg_sig_instance_new.csv'
    pos_ensp_index = input_path + 'positive_ensp_id.csv'
    neg_ensp_index = input_path + 'negative_ensp_id.csv'
    pos_sig_id = input_path + 'positive_sig_id.csv'
    neg_sig_id = input_path + 'negative_sig_id.csv'
    pos_gene_exp_path = 'positive_gene_exp.npy'
    neg_gene_exp_path = 'negative_gene_exp.npy'

    get_gene_expression(pos_sig_intance_path, pos_ensp_index,\
                        pos_sig_id, pos_gene_exp_path, 'positive/')

    get_gene_expression(neg_sig_intance_path, neg_ensp_index,\
                        neg_sig_id, neg_gene_exp_path, 'negative/')
    end = time.time()
    print('vector time elapsed :' + str(end - start))