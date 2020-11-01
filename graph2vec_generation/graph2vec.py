import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
#        print(self.nodes)
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = "_".join([str(self.features[node])] + sorted([str(deg) for deg in degs]))
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for iteration in range(self.iterations):
            self.features = self.do_a_recursion()


def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path.strip(".json").split("/")[-1]
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])
#    print(graph)

    if "features" in data.keys():
        features = data["features"]
    else:
        features = nx.degree(graph)

    features = {int(k): v for k, v, in features.items()}
    return graph, features, name


def feature_extractor(path, rounds):
    """
    Function to extract WL features from a graph.
    :param path: The path to the graph json.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph, features, name = dataset_reader(path)
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])
    return doc


def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = f.split("/")[-1].strip(".json")
        out.append([int(identifier)] + list(model.docvecs["g_" + identifier]))

    out = pd.DataFrame(out, columns=["type"] + ["x_" + str(dimension) for dimension in range(dimensions)])
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)


def main(args):
    """
    Main function to read the graph list, extract features, learn the embedding and save it.
    :param args: Object with the arguments.
    """
    graphs = glob.glob(args.input_path + "*.json")
    print("\nFeature extraction started.\n")
    print(args.wl_iterations)
    document_collections = Parallel(n_jobs=args.workers)(
        delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,
                    size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    iter=args.epochs,
                    alpha=args.learning_rate)

    save_embedding(args.output_path, model, graphs, args.dimensions)


if __name__ == "__main__":
    args = parameter_parser()
    inputpath = 'input_json/'
    files = os.listdir(inputpath)
    k_list = list(range(10, 80, 10))
    for k in k_list:
        pos_input = inputpath + str(k) + '/' + 'positive/'
        pos_files = os.listdir(pos_input)
        for pos_file in pos_files:
            print('positive; k is ', k, pos_file)
            pos_feat_path = pos_input + pos_file + '/'
            args.input_path = pos_feat_path
            pos_output = 'graph2vec/' + str(k) + '/' + 'positive/'
            if not os.path.exists(pos_output):
                os.makedirs(pos_output)
            args.output_path = pos_output + pos_file + '.csv'
            # print(args.output_path)
            main(args)

        neg_input = inputpath + str(k) + '/' + 'negative/'
        neg_files = os.listdir(neg_input)
        for neg_file in neg_files:
            print('negative; k is ', k, neg_file)
            neg_feat_path = neg_input + neg_file + '/'
            args.input_path = neg_feat_path
            neg_output = 'graph2vec/' + str(k) + '/' + 'negative/'
            if not os.path.exists(neg_output):
                os.makedirs(neg_output)
            args.output_path = neg_output + neg_file + '.csv'
            # print(args.output_path)
            main(args)