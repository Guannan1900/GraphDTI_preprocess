import argparse
import time
import numpy as np
import os
import json
import pickle
import random
import sklearn.metrics as metrics
import torch
from torch.utils import data
from data_generator import Dataset
from mlp_model import mlp_model

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def getArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir', required=True, default='data_sampled/', help='input data path')
    parser.add_argument('-bs', required=False, default=32, help='batch size, normally 2^n')
    parser.add_argument('-lr', required=False, default=0.0001, help='initial learning rate')
    parser.add_argument('-epoch', required=False, default=30, help='number of epochs for taining')

    return parser.parse_args()

def split_train_valid(index):

    input_path = 'input_list/'
    label_path = 'label_list/'

    test_name = str(index) + '_valid_list.pkl'
    with open(input_path + test_name, 'rb') as f:
        valid_list = pickle.load(f)
    print(len(valid_list))
    file_list = os.listdir(input_path)
    file_list.remove(test_name)
    print(file_list)
    train_list_total = []
    for name in file_list:
        with open(input_path + name, 'rb') as f_tmp:
            train_list_tmp = pickle.load(f_tmp)
            # print(len(train_list_tmp))
        train_list_total += train_list_tmp

    train_list = train_list_total
    print(len(train_list), len(valid_list))

    with open(label_path +'10_label.pickle', 'rb') as f_label:
        labels = pickle.load(f_label)

    return train_list, valid_list, labels

def to_onehot(yy):
    yy1 = np.zeros([len(yy), 2])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1

def train_cross_validation(k, datapath, batch_size, lr, num_epochs):

    # hyper parameters
    input_size = 600
    hidden_size = 128
    output_size = 2

    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 4}

    stage_list = [1, 2, 3, 4, 5]
    logs = {}
    best_val_loss_roc = {}
    for stage in stage_list:
        path_train = path_valid = datapath + str(k) + '/'
        new_train_list, new_valid_list, labels = split_train_valid(stage)
        print(len(new_train_list), len(new_valid_list))
        print(path_train, path_valid)
        partition = {"train": new_train_list, "validation": new_valid_list}
        # Generators
        training_set = Dataset(partition['train'], labels, path_train)
        training_generator = data.DataLoader(training_set, **params)
        validation_set = Dataset(partition['validation'], labels, path_valid)
        validation_generator = data.DataLoader(validation_set, **params)
        print('Training data is ready')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mlp = mlp_model(input_size, hidden_size, output_size)
        mlp = mlp.to(device)
        print(mlp.parameters)
        optimizer = torch.optim.SGD(mlp.parameters(), lr=lr, momentum=0.9, weight_decay=0.00001)
        criterion = torch.nn.CrossEntropyLoss()

        key_words = ['train_loss_' + str(stage), 'train_accuracy_' + str(stage), 'val_loss_' + str(stage),
                     'val_accuracy_' + str(stage)]
        logs[key_words[0]] = []
        logs[key_words[1]] = []
        logs[key_words[2]] = []
        logs[key_words[3]] = []

        for epoch in range(num_epochs):
            print(epoch)
            train_acc_sum = 0
            train_loss_sum = 0.0
            val_acc_sum = 0
            val_loss_sum = 0.0
            mlp.train()
            for train_inputs, train_labels in training_generator:
                train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
                train_outputs = mlp(train_inputs.float())
                train_loss = criterion(train_outputs, train_labels)
                optimizer.zero_grad()  # zero the gradient buffer
                train_loss.backward()
                optimizer.step()
                # trained_weight = mlp.fc1.weight.data
                # print(trained_weight)
                _, train_predicted = torch.max(train_outputs, 1)  # find the max of softmax and map the predicted list
                train_loss_sum += train_loss.detach() * train_inputs.size(0)
                train_acc_sum += (train_predicted == train_labels.data).sum() # different from CPU

            train_loss_epoch = train_loss_sum.item() / len(training_set)
            train_acc_epoch = train_acc_sum.item() / len(training_set)
            if (epoch + 1) % 1 == 0:
                print('Epoch [{}/{}],  Training Loss:{}, Training Accuracy:{}'.format
                      (epoch + 1, num_epochs, train_loss_epoch,
                       train_acc_epoch))
                # print(predicted)
                # print(labels)
            logs[key_words[0]].append(train_loss_epoch)
            logs[key_words[1]].append(train_acc_epoch)

            mlp.eval()
            torch.no_grad()
            epoch_outproba_val = np.empty((0, output_size))
            epoch_labels_val = np.empty((0, output_size))
            for val_inputs, val_labels in validation_generator:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = mlp(val_inputs.float())
                val_loss = criterion(val_outputs, val_labels)
                _, val_predicted = torch.max(val_outputs, 1)  # find the max of softmax and map the predicted list
                val_loss_sum += val_loss.detach() * val_inputs.size(0)
                val_acc_sum += (val_predicted == val_labels.data).sum()
                our_labels = val_labels.cpu().numpy()
                # print(our_labels)
                # print(len(our_labels))
                outproba = val_outputs.cpu()
                outproba = outproba.detach().numpy()
                our_target = to_onehot(our_labels)
                epoch_labels_val = np.append(epoch_labels_val, our_target, axis=0)
                # print(epoch_labels_val)
                epoch_outproba_val = np.append(epoch_outproba_val, outproba, axis=0)
            # print(train_loss_sum.item())
            # print(train_acc_sum.item())
            # print(epoch_labels_val.shape)
            # print(epoch_outproba_val.shape)
            val_loss_epoch = val_loss_sum.item() / len(validation_set)
            val_acc_epoch = val_acc_sum.item() / len(validation_set)
            # print(train_loss_epoch)
            # print(train_acc_epoch)
            if (epoch + 1) % 1 == 0:
                print('Epoch [{}/{}],  Validation Loss:{}, Validation Accuracy:{}'.format
                      (epoch + 1, num_epochs, val_loss_epoch,
                       val_acc_epoch))
                # print(predicted)
                # print(labels)
            logs[key_words[2]].append(val_loss_epoch)
            logs[key_words[3]].append(val_acc_epoch)
        # print(len([epoch in range(num_epochs)]))
        # print(len(logs['train_loss']), len(logs['train_loss']))
        model_saved = str(k) + '_mlp_graph2vec_opt_' + str(stage) + '.pt'
        # calculate roc score
        # print to list so that can be saved in .json file
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        thresholds = dict()
        for i in range(output_size):
            y_score = np.array(epoch_outproba_val[:, i])
            # print(y_score)
            y_test = np.array(epoch_labels_val[:, i])
            # print(y_test)
            fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_test, y_score)
            # fpr_t, tpr_t, _t = metrics.roc_curve(y_test, y_score, pos_label=1)
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        print(roc_auc[1])
        best_val_loss_roc['fpr_' + str(stage)] = fpr[1].tolist()
        best_val_loss_roc['tpr_' + str(stage)] = tpr[1].tolist()
        best_val_loss_roc['thresholds_' + str(stage)] = thresholds[1].tolist()
        best_val_loss_roc['auc_' + str(stage)] = roc_auc[1]
        # best_val_loss_roc = {'fpr_' + str(stage): fpr[1].tolist(), 'tpr_'+ str(stage): tpr[1].tolist(),
        #                      'thresholds_'+ str(stage): thresholds[1].tolist(), 'auc_'+ str(stage): roc_auc[1]}
        # print(best_val_loss_roc)
        model_path = 'model/'
        if model_path == None:
            torch.save(mlp, model_path + model_saved)
        else:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            if os.path.exists(model_saved):
                os.remove(model_saved)
            torch.save(mlp, model_path + model_saved)
            mm = torch.load(model_path + model_saved)
            print(mm.parameters)

    log_path = 'results/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(log_path + str(k) + '_logs.json', 'w') as fp:
        json.dump(logs, fp)
    with open(log_path + str(k) + '_roc.json', 'w') as f_roc:
        json.dump(best_val_loss_roc, f_roc)


if __name__ == "__main__":

    start = time.time()
    parse = getArgs()

    k_list = [10, 20, 30, 40, 50, 60, 70]

    for k in k_list:
        print('k is ', k)
        train_cross_validation(k, parse.data_dir, parse.bs, parse.lr, parse.epoch)

    end = time.time()
    print('vector time elapsed :' + str(end - start))
