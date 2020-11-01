# Feature selection

## Description
In order to diminish the overfitting problem, the feature selection part is used to select the optimal features for GraphDTI. For the MLP model used in GraphDTI, we use *permutation feature importance* (https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html) to calculate the important scores of all 1412 features, which are [ProtVec](https://github.com/kyu999/biovec) (index: 0-299), [Bionoi-AE](https://github.com/CSBG-LSU/BionoiNet) (index: 300-811), [Mol2vec](https://github.com/samoturk/mol2vec) (index: 812-1111) and [Graph2vec](https://github.com/benedekrozemberczki/graph2vec) (index: 1112-1411). 

## Files
- ```train_list.pkl``` and```training_label.pickle``` -> The names and labels for all instances in the training dataset. 
- ```test_list.pkl``` and ```test_label.pickle``` -> The names and labels for all instances in the test dataset.

> "Note: the training and test dataset can be available at: [link]"

## Usage
+ required file -> ```train_list.pkl```, ```training_label.pickle```, ```test_list.pkl``` and ```test_label.pickle```
+ input-> the training data path, ```training_data/```, and the test data path ```test_data/```
+ output -> the indices and important scores file, ```permu_feature_improtance.json```
```shell
python feature_selection.py  -train 'training_data/' -test 'test_data/'
```
