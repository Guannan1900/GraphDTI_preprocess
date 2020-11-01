import os
import pandas as pd
import pickle
import numpy as np
import argparse
import time

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-inpath',
                        required=False,
                        default='input_list/',
                        help='input file path.')

    return parser.parse_args()


def get_list(pos_target_path, neg_target_path):

    pos_target_prot = pd.read_csv(pos_target_path)
    neg_target_prot = pd.read_csv(neg_target_path)
    # print(pos_target_prot)
    pos_list = list(set(pos_target_prot['protein'].tolist()))
    neg_list = list(set(neg_target_prot['protein'].tolist()))

    return pos_list, neg_list

def get_prot2vec(prot):

    prot2vec_path = 'ProtVec/'
    prot2vec_file = prot2vec_path + prot + '.protvec'
    with open(prot2vec_file, 'r') as prot_file:
        protvec_data = prot_file.read()
    protvec_string = protvec_data.split(',99\n')[1]
    protvec_tmp = protvec_string.replace(',', ' ').split()
    protvec_list = [float(i) for i in protvec_tmp]

    return protvec_list


def get_bindingsite(protein):

    actual_path = 'input_list/boinoi-AE_list'
    binding_path = 'Bionoi-AE/'
    with open(actual_path, 'r') as bind_file:
        bind_data = bind_file.read()
    prot_list = bind_data.replace('\n',' ').replace(',',' ').split()
    if protein in prot_list:
        target_index = prot_list.index(protein) - 1
        target = prot_list[target_index]
        target_path = binding_path + target + '_blended_XOY+_r0_OO.pickle'
        with open(target_path, 'rb') as target_file:
            target_tmp = pickle.load(target_file)
        # print(target_tmp.shape)
        binding_site = np.hstack(target_tmp)
        binding_site = binding_site.tolist()
    else:
        print('Protein do not have target: ', protein)
        binding_site = []

    return binding_site

def get_mol2vec(protein_name, prot_vec, prot_sig_instance, target_index_path, ensp_name, state):

    target_index = pd.read_csv(target_index_path)
    ensp_index_df = target_index.loc[target_index['Protein'] == ensp_name].reset_index(drop=True)
    ensp_index = ensp_index_df['idx'][0]
    # print(state, ensp_index)
    graph2vec_path = 'Graph2vec/' + state + str(ensp_index) + '.csv'
    graph2vec_df = pd.read_csv(graph2vec_path)
    sig_id_path = 'sig_index/' + state + 'index_' + str(ensp_index) + '.csv'
    sig_id_df = pd.read_csv(sig_id_path)
    outpath = 'training_data/' + state
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # print(prot_sig_instance)
    drug_name_list = list(set(prot_sig_instance['drug'].tolist()))
    drug_feature_path = 'mol2vec/'
    name_list = []
    for drug_name in drug_name_list:
        drug_feature = pd.read_csv(drug_feature_path + drug_name + '.csv')
        mol2vec_tmp = drug_feature.iloc[:, 2:302]
        mol2vec = mol2vec_tmp.iloc[0].values.tolist()
        drug_sig_instance = prot_sig_instance.loc[prot_sig_instance['drug'] == drug_name].reset_index(drop=True)
        # print(drug_sig_instance)
        sig_id_list = list(set(drug_sig_instance['sig_id'].tolist()))
        for sig_id in sig_id_list:
            # print(sig_id)
            sig_id_index_tmp = sig_id_df.loc[sig_id_df['sig_id'] == sig_id].reset_index(drop=True)
            sig_id_index = sig_id_index_tmp['inx'][0]
            graph2vec_list = graph2vec_df.loc[graph2vec_df['type'] == sig_id_index].reset_index(drop=True)
            graph2vec_tmp = graph2vec_list.iloc[:, 1:302]
            graph2vec = graph2vec_tmp.iloc[0].values.tolist()
            tmp = np.hstack((prot_vec, mol2vec, graph2vec))
            # print(len(graph2vec), len(tmp))
            name = protein_name + '_' + drug_name + '_' + ensp_name + '_' + sig_id
            if name in name_list:
                # print(name)
                print('name')
                # continue
            else:
                name_list.append(name)

            pickle_file = open(outpath + name + '.pkl', 'wb')
            pickle.dump(tmp, pickle_file)


def feature_generation(prot_list, instance_path, target_path, state):

    print('Training data label is ', state)
    sig_instance = pd.read_csv(instance_path)

    for prot_name in prot_list:
        print('Target protein is', prot_name)
        prot_protvec = get_prot2vec(prot_name)
        prot_bindingsite = get_bindingsite(prot_name)
        prot_features = prot_protvec + prot_bindingsite
        # print(len(prot_protvec), len(prot_bindingsite), len(prot_features))

        prot_sig_instance = sig_instance.loc[sig_instance['protein'] == prot_name].reset_index(drop=True)
        ensp_id_list = list(set(prot_sig_instance['ensp_id'].tolist()))
        print('ENSP ID of target protein is', ensp_id_list)
        for ensp_id in ensp_id_list:
            get_mol2vec(prot_name, prot_features, prot_sig_instance, target_path, ensp_id, state)


if __name__ == "__main__":

    start = time.time()
    parse = getArgs()
    input_path = parse.inpath

    pos_target_ensp_path = input_path + 'positive_target_protein.csv'
    neg_target_ensp_path = input_path + 'negative_target_protein.csv'

    pos_sig_intance_path = input_path + 'pos_sig_instance_new.csv'
    neg_sig_intance_path = input_path + 'neg_sig_instance_new.csv'

    pos_prot, neg_prot = get_list(pos_sig_intance_path, neg_sig_intance_path)
    print('positive protein is', len(pos_prot), 'negavite protein is', len(neg_prot))

    feature_generation(pos_prot, pos_sig_intance_path, pos_target_ensp_path, 'positive/')
    feature_generation(neg_prot, neg_sig_intance_path, neg_target_ensp_path, 'negative/')
    end = time.time()
    print('vector time elapsed :' + str(end - start))
