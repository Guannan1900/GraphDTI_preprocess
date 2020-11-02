# Feature integration

## Description
In this section, we integrate four types of features which are used for GraphDTI project. The four type of features for each instance are  [ProtVec](https://github.com/kyu999/biovec) (index: 0-299), [Bionoi-AE](https://github.com/CSBG-LSU/BionoiNet) (index: 300-811), [Mol2vec](https://github.com/samoturk/mol2vec) (index: 812-1111) and optimized [Graph2vec](https://github.com/benedekrozemberczki/graph2vec) (index: 1112-1411). The procedure for Graph2vec feature generation and optimization is provided in [link] and [link]. 

## Usage
- Required files:
  + drug-tanget instance list: ```pos_sig_instance_new.csv``` and ```neg_sig_instance_new.csv```
  + ENSP ID list for target protein: ```positive_target_protein.csv``` and ```negative_target_protein.csv```
  + Boinoi-AE_list for target protein: ```boinoi-AE_list```
  
> "Note: the required files are in the folder ```input_list/```
