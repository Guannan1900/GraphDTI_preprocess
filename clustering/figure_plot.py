import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def calculation(sil_repeat):

    sil_mean = []
    sil_std = []
    for i in range(15):
        # print(sil_repeat[:,i])
        sil_mean_tmp = np.mean(sil_repeat[:,i])
        sil_std_tmp = np.std(sil_repeat[:,i])
        sil_mean.append(sil_mean_tmp)
        sil_std.append(sil_std_tmp)

    return sil_mean, sil_std

sil_feature = np.load('result/sil_repeat_features.npy', allow_pickle=True)
sil_pmd = np.load('result/sil_repeat_scaled_pmd.npy', allow_pickle=True)
sil_pmd_ori = np.load('result/sil_repeat_original_pmd.npy', allow_pickle=True)

# print(sil_feature)
# print(sil_feature.shape)
# print(sil_pmd)
# print(sil_pmd.shape)

sil_mean_feature, sil_std_feature = calculation(sil_feature)
sil_mean_pmd, sil_std_pmd = calculation(sil_pmd)
sil_mean_ori, sil_std_ori = calculation(sil_pmd_ori)
# print(sil_mean_feature)
# print(sil_mean_pmd)
# print(sil_mean_ori)
# print(sil_mean_pmd[0:12])

k_list = [2,50,100,150,200,250,300,350,400,450,500,600,700,800,1000]
# print(len(k_list))
fig1 = plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
font0 = FontProperties()
font = font0.copy()
font.set_weight('semibold')
font.set_family('sans-serif')
plt.tick_params(labelsize=12)
# dim=[2,100,200,300,400,500,600,700,800]
dim=[2,100,200,300,400,500]
num = 11
plt.errorbar(k_list[0:num], sil_mean_feature[0:num], yerr=sil_std_feature[0:num], fmt='--o', color='b', label='Protvec+Mol2vec') #'#9370DB'
plt.errorbar(k_list[0:num], sil_mean_pmd[0:num], yerr=sil_std_pmd[0:num], fmt='-o', color='g', label='Scaled PMD') #'#90C978'
plt.errorbar(k_list[0:num], sil_mean_ori[0:num], yerr=sil_std_ori[0:num], fmt=':o', color='r', label='Original PMD') #'#FA8072'
plt.legend(loc='upper right',  prop={'size': 13, 'family': 'serif'})
plt.xlabel('Number of clusters', fontsize=13, family='serif')
plt.ylabel('Silhouette Coefficient', fontsize=13, family='serif')
# plt.xticks(dim)
plt.gca().set_aspect(1/plt.gca().get_data_ratio(), adjustable='box')
# plt.grid(axis='x')
fig1.savefig('figures/silhouette_score_k_modify.tif', format='tif', dpi=800, bbox_inches='tight')
# plt.show()
