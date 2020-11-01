import pandas as pd
import numpy as np
import argparse
import time

def getArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument('-output',
                        required=False,
                        default='processed_data/',
                        help='path of output folder.')
    parser.add_argument('-inpath',
                        required=False,
                        default='inputdata/',
                        help='input file path and name.')

    return parser.parse_args()

def get_prot2vec(prot):

    prot2vec_path = 'protein_pos_protvec/'
    prot2vec_file = prot2vec_path + prot + '.protvec'
    with open(prot2vec_file, 'r') as prot_file:
        protvec_data = prot_file.read()
    protvec_string = protvec_data.split(',99\n')[1]
    protvec_tmp = protvec_string.replace(',', ' ').split()
    protvec_list = [float(i) for i in protvec_tmp]

    return protvec_list

def get_mol2vec(drug_name):

    drug_feature = pd.read_csv('drugs_mol2vec/' + drug_name + '.csv')
    mol2vec_tmp = drug_feature.iloc[:, 2:302]
    mol2vec_ = mol2vec_tmp.iloc[0].values.tolist()

    return mol2vec_


def feature_generation(path, output_path):

    # generate drug and protein list
    instances0 = pd.read_csv(path + 'negative_drug_protein.csv')
    instances1 = pd.read_csv(path + 'positive_drug_protein.csv')
    instances = pd.concat([instances0, instances1], axis=0, ignore_index=True)
    bad_drugs = ['CHEMBL3616754', 'CHEMBL3616756', 'CHEMBL3616758', 'CHEMBL3616761', 'CHEMBL442315']
    instances = instances[~instances['drug'].isin(bad_drugs)]
    # print(instances)

    prot_list = instances['protein'].tolist()
    drug_list = instances['drug'].tolist()
    print(len(prot_list))
    print(len(list(set(prot_list))))  # 579
    print(len(drug_list))
    print(len(list(set(drug_list))))  # 3619
    features = []
    for i in range(len(prot_list)):
        # s = time()
        prot_protvec = get_prot2vec(prot_list[i])
        mol2vec = get_mol2vec(drug_list[i])
        features_tmp = prot_protvec + mol2vec
        features.append(features_tmp)
        # print('time spend 1 time {:.4f} seconds'.format(time() - s))
    features_np = np.array(features)
    print(features_np.shape)
    np.save(output_path + 'features.npy', features_np)


if __name__ == "__main__":

    start = time.time()
    parse = getArgs()
    in_path = parse.inpath
    out_path = parse.output
    feature_generation(in_path, out_path)
    end = time.time()
    print('vector time elapsed :' + str(end - start))
