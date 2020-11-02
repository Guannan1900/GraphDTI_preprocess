# Graph2vec optimization
## Description
In this section, we design a processdure to optimize the graph2vec features generating from every target protein and its varying M number of connected nodes in human PPI network. The optimized Graph2vec features are used in the GraphDTI project (). In order to select the optimal M value for Graph2vec features, we randomly sample 20,000 instances (```graph2vec_optimization/```) from the entire dataset and calculate performance of a multilayer perceptron (MLP) used in GraphDTI. The features of these instances are Graph2Vec features with varying M values and the same [Mol2vec](https://github.com/samoturk/mol2vec) features.


## Usage
- Required files: 
```input_list/```: The training and validation list for instances. 
```label_list/```: The labels for data.
- Input data: 
```graph2vec_optimization/``` -> input features for graph2vec optimization.
- Output:
```result/``` -> ROC results of different Graph2vec features with different M vaules.
- Run
```shell
python train_graph2vec.py -data_dir 'data_sampled/'
```

> Note: the input data ```graph2vec_optimization/``` can be available at: [link]

## Result

## Requirements




