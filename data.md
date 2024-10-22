## Description
The data folder for GraphDTI contains 5 subfolders: ```graph2vec_optimization/```, ```feature_integration_training/```, ```feature_integration_test/```, ```training_data/``` and ```test_data/```. The structure of the data is shown as below:
```
    .
    └── GraphDTI 
        ├── graph2vec_generation
            ├── training
                ├── gene_expression   
                ├── target_node_shortest_path
                └── input_file
            └── test
                ├── gene_expression   
                ├── target_node_shortest_path
                └── input_file
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
- graph2vec_generation -> Generate Graph2vec features for training and test. The training and the test folder has the same structure
  + gene_expression -> The gene expressions for protein used as the node feature of the target protein for graph2vec features.
  + target_node_shortest_path -> The distance between connected node and the target node used as the node feature of the target protein for graph2vec features.
  + input_file -> The input json files for generating graph2vec features.
- graph2vec_optimization -> Different Graph2vec features with different number (10, 20,...,70) of the connected nodes of the target protein and the same Mol2vec features for the sampled instances used to optimize Graph2vec features for GraphDTI. 
- feature_integration_training -> the original features including ProtVec, Bionoi-AE, Mol2Vec and Graph2vec for the training instances.
- feature_integration_test -> the original features including ProtVec, Bionoi-AE, Mol2Vec and Graph2vec for the test instances.
- training_data -> The pickle files representing feature vectors for all instances in the training dataset used for GraphDTI. 
  + name format for the pickle file -> Uni-prot ID_Drug ID_ENSP ID_Signature ID.pkl
  + sturcture of the feature vector -> 300 dimensional ProtVec (index: 0-299), 512 dimensional Bionoi-AE (index: 300-811), 300 dimensional Mol2Vec (index: 812-1111) and 300 dimensional Graph2vec (index: 1112-1411)
- test_data -> The pickle files representing feature vectors for all instances in the test dataset used for GraphDTI. The format and structure of the pickle file for each instance is the same with the ```training_data/```

