# clustering

## Description
In order to address the issue of overlapped data and evaluate the generalizability of GraphDTI, we design a cluster-based split protocol for cross-validation. We first calculate three kinds of distance metrics for clustering. The two kinds of distance metrics for drug-target instances are the original **perfect match distance (PMD)** [[1]](#1) and the **scaled PMD**. The other one distance metric is the Euclideanand distance between the [Mol2vec](https://github.com/samoturk/mol2vec) and [ProtVec](https://github.com/kyu999/biovec) features of drug-target instances. Then we use *KMedoids* (https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html) to cluster all instances and use *Silhouette Coefficient* (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) to select the optimal number of clusters.


## Files
- ```train_list.pkl``` and```training_label.pickle``` -> The names and labels for all instances in the training dataset. 
- ```test_list.pkl``` and ```test_label.pickle``` -> The names and labels for all instances in the test dataset.

> "Note: the training and test dataset can be available at: [link]"

## Usage
1. PMD metric calculation

&ensp;&ensp; Download and unzip the inputdata.zip. The ```inputdata/``` contains:
- positive and negative drug-target instance -> ```positive_drug_protein.csv``` and ```negative_drug_protein.csv``` .
- similarities for drugs and targets -> ```drug_similarity.dat``` and ```target_similarity.dat```.

&ensp;&ensp; Output files:
- drug-target instance list -> ```processed_data/intances.lst```.
- PMD matrix for all instances -> ```processed_data/pmd_total.npy```.

```shell
python pmd_calculation.py -inpath 'inputdata/'
```

2. Feature metric calculation

&ensp;&ensp; Required data:
- positive and negative drug-target instance -> ```positive_drug_protein.csv``` and ```negative_drug_protein.csv``` .
- Mol2vec features for drugs and Protvec features for targets -> ```drugs_mol2vec/``` and ```protein_pos_protvec/```.
> "Note: the Mol2vec features and ProtVec features can be available at: [link]". You can download it and put in the same fold with the code.

&ensp;&ensp; Output file:
- feature matrix for all instances -> ```processed_data/features.npy```.

```shell
python feature_calculation.py -inpath 'inputdata/'
```

3. Clustering

&ensp;&ensp; In this section, we use we use *KMedoids* (https://scikit-learn-extra.readthedocs.io/en/latest/generated/sklearn_extra.cluster.KMedoids.html) to cluster all instances with different number of clusters. We provide three options for distance metrics, which are 'original_PMD', 'scaled_PMD' and 'features'. 

&ensp;&ensp; Required data:
- 'original_PMD' and 'scaled_PMD' -> ```processed_data/pmd_total.npy``` .
- 'features' -> ```processed_data/features.npy```.

&ensp;&ensp; Output file:
- Silhouette coefficients for different number of clusters.

```shell
python cluster_optimization.py -inpath 'processed_data/' -op 'scaled_PMD'
```
4. Figures for Silhouette Coefficients for different number of clusters.






## References
<a id="1">[1]</a> 
Naderi, Misagh and Govindaraj, Rajiv Gandhi and Brylinski, Michal. (2018). 
eModel-BDB: a database of comparative structure models of drug-target interactions from the Binding Database. 
Gigascience, 7(8), 1-9.
