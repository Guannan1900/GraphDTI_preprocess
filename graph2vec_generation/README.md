# Graph2vec generalization

## Description
The details of generating the Graph2vec features of the target protein is provided here. Note that the Graph2vec features is used for GraphDTI (https://github.com/Guannan1900/GraphDTI) project.

## Implementation Details
In this section, we generate the Graph2vec features for each target protein and its different number (denoted by M) of connected proteins based on the human Protein-Protein Interaction (PPI) network. First, we calculate the shortest path for each each target node in the graph using the *Dijkstra’s shortest path algorithm* (). Then we generate the inputfile for [Graph2vec](https://github.com/benedekrozemberczki/graph2vec) using expressions of the protein and the distance between target protein and its neighbors as the node features. Finally, we generate the Graph2vec features for each subgraph containing one target protein and its M number of connected neighbors. The flowchart of the procedure is shown below:

## Usage
1. Graph generation: generate the graph based on the human PPI network.
    - Required files -> a edge table ```human_interactions_sample.csv``` and an node index table ```node_index_sample.csv``` for the network.
    - Output -> a graph file ```undirected_graph.gexf``` which can be visualized by ```Gephi``` and processed by ```Networkx``` 
    - Run
    ```shell
    python graph_generation.py -infile 'ori_data/'
    ```
    > Note: these required files are in the fold ```ori_data/```.
2. Shortest path calculation: calculate the shortest path for each target node in the graph.
    - Required files -> a graph file ```undirected_graph.gexf``` and the node index list ```0.in```
    - Output -> ```target_node_shortest_path/``` which contains shortest path for each target node
    - Run
    ```shell
    python shortest_path.py --infile '0.in'
    ```

3. Inputfile generation: generate subgraphs containing each target node and its different number of the connected neighbors using expressions of the target protein and the reciprocal value of the confidence score between two connected protein as the node features.
    - Required files:
      + shortest path for target proteins -> ```target_node_shortest_path/```
      + node index for target proteins -> ```node_index_new.csv```
      + gene expression of the target proteins ->  ```positive_gene_exp.npy``` and ```negative_gene_exp.npy```
      + drug-target instances with their signature ID -> ```pos_sig_instance_new.csv``` and ```neg_sig_instance_new.csv```
      + ENSP ID (target proteins) index for gene expression and signature ID index for gene expression -> ```positive_ensp_id.csv```, ```negative_ensp_id.csv``` and     ```positive_sig_id.csv```, ```negative_sig_id.csv```.
     - Output:
       + input subgraphs (formated in json files) for [Graph2vec](https://github.com/benedekrozemberczki/graph2vec) -> ```input_json/```
     - Run
     ```shell
     python inputfile_generation.py 
     ```
     
    > Note: these required files are in the fold ```data_sample/```.
4. Graph2vec features generation: generate Graph2vec features for the input subgraphs which contain the target protein and different number of connected proteins using [Graph2vec](https://github.com/benedekrozemberczki/graph2vec).
    - Required files -> the input subgraphs ```input_json/```
    - Output files -> Graph2vec features for different subgraph ```graph2vec/``` 
    - Run
    ```shell
    ./graph_vector.sh
    ```

## Requirments

