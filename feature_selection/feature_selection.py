"""
This code is used to select optimal number of features

@author: lgn
"""
import torch
import time
import numpy as np
import os
import json
import random
from mlp_model import mlp_model
from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score
from eli5.permutation_importance import get_score_importances
import argparse
import pickle

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-training',
                        required=True,
                        default='training_data/',
                        help='training dataset')
    parser.add_argument('-test',
                        required=True,
                        default='test_data/',
                        help='test dataset')

    return parser.parse_args()

def load_data(data_list, labels, data_path):

    label_list = []
    features = []

    for i in range(len(data_list)):
        ID = data_list[i]
        y = labels[ID]

        if y == 1:
            folder = data_path + 'positive/'
        elif y == 0:
            folder = data_path + 'negative/'
        label_list.append(y)

        X = np.load(folder + ID, allow_pickle=True)
        features.append(X)
    feature_use = np.array(features)
    print(feature_use.shape)
    feature_use = feature_use.astype(np.float32)
    target_use = np.array(label_list)
    print(target_use.shape)

    return feature_use, target_use

def feature_list_generation(train_data_path, test_data_path):

    input_size = 1412     # original feature size
    # hidden_size = int(input_size/3)
    hidden_size = 300
    output_size = 2
    num_epochs = 50
    # lr = 0.00001
    lr = 0.0001
    batch_size = 32

    ### Load train dataset
    with open('train_list.pkl', 'rb') as f_train:
        train_list = pickle.load(f_train)
    with open('training_label.pickle', 'rb') as f_label_train:
        train_labels = pickle.load(f_label_train)

    train_list = train_list[0:200000]
    print(len(train_list))
    feature_train, target_train = load_data(train_list, train_labels, train_data_path)
    ## Load test dataset
    with open('test_list.pkl', 'rb') as f_test:
        test_list = pickle.load(f_test)
    with open('test_label.pickle', 'rb') as f_label_test:
        test_labels = pickle.load(f_label_test)

    # test_list = test_list[0:10000]
    print(len(test_list))
    feature_test, target_test = load_data(test_list, test_labels, test_data_path)
    # model with skorch
    # convert model based on pytorch to sklearn
    net = NeuralNetClassifier(
        mlp_model(input_size, hidden_size, output_size),
        max_epochs=num_epochs,
        lr=lr,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        batch_size=batch_size,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        optimizer__momentum=0.9,
        optimizer__weight_decay=0.00001
    )
    model = net.fit(feature_train, target_train)
    print(model.score(feature_test, target_test))
    # define a score function. using accuracy
    def score(feature_test, target_test):
        y_pred = net.predict(feature_test)
        return accuracy_score(target_test, y_pred)

    # This function takes only numpy arrays as inputs
    # base_score = score_func(feature_train, target_train)
    base_score, score_decreases = get_score_importances(score, feature_test, target_test, n_iter=10)
    feature_importances = np.mean(score_decreases, axis=0)
    feature_importance_dict = {}
    for i in range(1412):
        feature_importance_dict[str(i)] = feature_importances[i]
    permu_features = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))
    # print(permu_features)

    with open('permu_feature_improtance.json', 'w') as fp:
        json.dump(permu_features, fp)


if __name__ == "__main__":

    start = time.time()
    parse = getArgs()
    feature_list_generation(parse.training, parse.test)

    end = time.time()
    print('vector time elapsed :' + str(end - start))
