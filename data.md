## Description
The data folder contains 5 subfolders: graph2vec_optimization, feature_integration_training, feature_integration_test, training_data and test_data. The structure of the data is shown as below:
```
    .
    └── GraphDTI      
        ├── graph2vec_optimization
            ├── 10   
            ├── 20
            ├── ...
            └── 70
        ├── feature_integration_training
            ├── Bionoi-AE   
            ├── ProtVec
            ├── Mol2vec
            └── Graph2vec
        ├── feature_integration_test
            ├── Bionoi-AE   
            ├── ProtVec
            ├── Mol2vec
            └── Graph2vec
        ├── training_data
            ├── positive   
            └── negative
        └── test_data
            ├── positive   
            └── negative
```

- graph2vec_optimization -> the data is used to select the Graph2vec features with different M value of connected nodes of the same target protein. The optimized subgraph which contains one target node and its M connected nodes is used to calculate the 
- feature_integration_training -> the original features for the training instances 
- feature_integration_test -> the original features for the test instances
- training_data -> The pickle files representing feature vectors for all instances in the training dataset used for GraphDTI. 
  + name format for the pickle file -> Protein name_Drug name_ENSP ID_Signature ID.pkl
  + sturcture of feature vector -> 300 dimensional ProtVec (index: 0-299), 512 dimensional Bionoi-AE (index: 300-811), 300 dimensional Mol2Vec (index: 812-1111) and 300 dimensional Graph2vec (index: 1112-1411)
- test_data

The training_data fold contains feature vectors for all instances in the training dataset. The feature vector for each instance is represented as a pickle file.
The pickle file is named in the format like below:
Protein name_Drug name_ENSP ID_Signature ID.pkl
The length of each feature vector is 1412.  
The 1412 features for each instance contain: 
300 dimensional ProtVec (index: 0-299), 512 dimensional Bionoi-AE (index: 300-811), 300 dimensional Mol2Vec (index: 812-1111) and 300 dimensional Graph2vec (index: 1112-1411).

The test_data fold contains feature vectors for all instances in the test dataset. The feature vector for each instance is represented as a pickle file.
The format and structure of the pickle file for each instance is the same with the training_data
