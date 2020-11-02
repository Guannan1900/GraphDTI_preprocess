The data file contains 7 subfolders: data_sampled, training_data, test_data and models. The structure of the data is shown as below:
```
    .
    └── GraphDTI      
        ├── graph2vec_optimization
            ├── 10   
            ├── 20
            ├── ...
            └── 70
        ├── feature_integration
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


The training_data fold contains feature vectors for all instances in the training dataset. The feature vector for each instance is represented as a pickle file.
The pickle file is named in the format like below:
Protein name_Drug name_ENSP ID_Signature ID.pkl
The length of each feature vector is 1412.  
The 1412 features for each instance contain: 
300 dimensional ProtVec (index: 0-299), 512 dimensional Bionoi-AE (index: 300-811), 300 dimensional Mol2Vec (index: 812-1111) and 300 dimensional Graph2vec (index: 1112-1411).

The test_data fold contains feature vectors for all instances in the test dataset. The feature vector for each instance is represented as a pickle file.
The format and structure of the pickle file for each instance is the same with the training_data
