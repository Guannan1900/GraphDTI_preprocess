from time import time
import argparse
import re
import pandas as pd
import numpy as np

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-outpath',
                        required=False,
                        default='processed_data/',
                        help='path of output folder.')
    parser.add_argument('-inpath',
                        required=False,
                        default='inputdata/',
                        help='input file path.')

    return parser.parse_args()

def pmd_calculation(path, output):

    s = time()
    instances0 = pd.read_csv(path + 'negative_drug_protein.csv')
    instances1 = pd.read_csv(path + 'positive_drug_protein.csv')
    instances = pd.concat([instances0, instances1], axis=0, ignore_index=True)
    # print(len(instances))
    # instances = instances.sample(n=5).reset_index(drop=True)
    bad_drugs = ['CHEMBL3616754', 'CHEMBL3616756', 'CHEMBL3616758', 'CHEMBL3616761', 'CHEMBL442315']
    instances = instances[~instances['drug'].isin(bad_drugs)]
    for i, row in instances.iterrows():
        tmp = row['drug']
        idx = [x.start() for x in re.finditer('C', tmp)]
        if len(idx) > 1:
            instances.at[i, 'drug_new'] = tmp[:idx[1]]
            # print(tmp)
        else:
            instances.at[i, 'drug_new'] = tmp
    print(len(instances))
    # print(instances)
    print('time spend processing instances {:.4f} seconds'.format(time() - s))

    instance_list = [' '.join(x) for x in zip(instances['drug'].tolist(), instances['protein'].tolist())]
    with open(output + 'intances.lst', 'w') as f:
        for item in instance_list:
            f.write(item + '\n')
    drug_list = instances['drug_new'].tolist()
    protein_list = instances['protein'].tolist()

    s = time()
    ds = dict()
    with open(path + 'drug_similarity.dat') as f:
        for item in f.readlines():
            tmp = item.strip('\n').split(' ')
            ds[tmp[0] + '-' + tmp[1]] = float(tmp[2])
            ds[tmp[1] + '-' + tmp[0]] = float(tmp[2])
    for dd in drug_list:
        ds['-'.join([dd, dd])] = 1.0
    print('time spend processing drug similarities {:.4f} seconds'.format(time() - s))

    s = time()
    ps = dict()
    with open(path + 'target_similarity.dat') as f:
        for item in f.readlines():
            tmp = item.strip('\n').split(' ')
            ps[tmp[0] + '-' + tmp[1]] = float(tmp[2])
            ps[tmp[1] + '-' + tmp[0]] = float(tmp[2])
    for pp in protein_list:
        ps['-'.join([pp, pp])] = 1.0
    print('time spend processing protein similarities {:.4f} seconds'.format(time() - s))
    # PMD = np.array((len(instance_list),len(instance_list)))
    pmd_list = []
    for i, row in instances.iterrows():
        drug = row['drug_new']
        protein = row['protein']
        tcs = np.array([ds['-'.join([drug, x])] for x in instances['drug_new'].tolist()])
        tms = np.array([ps['-'.join([protein, x])] for x in instances['protein'].tolist()])
        pmd = np.sqrt((1 - tcs) ** 2 + (1 - tms) ** 2)
        pmd_list.append(pmd)
    pmd_np = np.array(pmd_list)
    np.save(output + 'pmd_total.npy', pmd_np)
    # print(pmd_np)
    print(pmd_np.shape)
    print('time spend processing protein similarities {:.4f} seconds'.format(time() - s))


if __name__ == "__main__":
    parse = getArgs()
    input_path = parse.inpath
    output_path = parse.outpath
    pmd_calculation(input_path, output_path)



