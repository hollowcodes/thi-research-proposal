""" file to train and evaluate the model """

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import wandb
import argparse
import sklearn
import hashlib

import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from model import ConvLSTM as Model
from utils import create_dataloader, show_progress, onehot_to_string, init_xavier_weights, device, char_indices_to_string, lr_scheduler, write_json, load_json
from test_metrics import validate_accuracy, create_confusion_matrix, recall, precision, f1_score, score_plot, true_positive_rate, plot_true_positive_rates, plot_weight_influence
import xman as xman


torch.manual_seed(0)


class Run:
    def __init__(self, run_name, str="", dataset_name: str="", test_size: float=0.1, model_file: str="", epochs: int=10, batch_size: int=128, 
                lr_schedule: list=[0.001, 0.95, 100], rnn_config: list=[100, 2], cnn_config: list=[2, 3, [32, 64]],
                dropout_chance: float=0.5, embedding_size: int=32, augmentation_chance: float=0.5, continue_: bool=False, loss_weights: list=None):

        self.run_name = run_name

        # dataset
        self.dataset_path = "dataset/" + dataset_name + "/dataset.pickle"
        self.test_size = test_size
        with open("dataset/" + dataset_name + "/nationalities.json", "r") as f: 
            self.classes = json.load(f) 
            self.total_classes = len(self.classes)

        # model file
        self.model_file = "models/" + model_file
        if not os.path.exists(self.model_file):
            open(self.model_file, "w+")

        # hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_chance = dropout_chance
        self.embedding_size = embedding_size
        self.augmentation_chance = augmentation_chance

        # unpack learning-rate parameters (idx 0: current lr, idx 1: decay rate, idx 2: decay intervall in iterations)
        self.lr = lr_schedule[0]
        self.lr_decay_rate = lr_schedule[1]
        self.lr_decay_intervall = lr_schedule[2]

        # rnn parameters (idx 0: hidden-nodes, idx: 1 internal layers)
        self.hidden_size = rnn_config[0]
        self.rnn_layers = rnn_config[1]

        # unpack cnn parameters (idx 0: amount of layers, idx 1: kernel size, idx 2: list of feature map dimensions)
        self.cnn_layers = cnn_config[0]
        self.kernel_size = cnn_config[1]
        self.channels = cnn_config[2]
        assert self.cnn_layers == len(self.channels), "The amount of convolutional layers doesn't match the given amount of channels!"

        # dataloaders for train, test and validation
        self.train_set, self.validation_set, self.test_set = create_dataloader(dataset_path=self.dataset_path, test_size=self.test_size, val_size=self.test_size, \
                                                                               batch_size=self.batch_size, class_amount=self.total_classes, augmentation=self.augmentation_chance)

        # resume training boolean
        self.continue_ = continue_

        # load loss weights if needed
        self.loss_weights = torch.Tensor(loss_weights).type(torch.FloatTensor).to(device=device) if loss_weights != None else None

        # initialize xman experiment manager
        self.xmanager = xman.ExperimentManager(experiment_name=self.run_name, continue_=self.continue_)
        self.xmanager.init(optimizer="Adam", 
                            loss_function="NLLLoss", 
                            epochs=self.epochs, 
                            learning_rate=self.lr, 
                            batch_size=self.batch_size,
                            custom_parameters={})

    def _validate(self, model, dataset, confusion_matrix: bool=False, plot_scores: bool=False):
        validation_dataset = dataset

        criterion = nn.NLLLoss()
        losses = []
        total_targets, total_predictions = [], []

        for names, targets, _ in tqdm(validation_dataset, desc="validating", ncols=150):
            names = names.to(device=device)
            targets = targets.to(device=device)

            predictions = model.eval()(names)
            loss = criterion(predictions, targets.squeeze())
            losses.append(loss.item())

            for i in range(predictions.size()[0]):
                target_index = targets[i].cpu().detach().numpy()[0]

                prediction = predictions[i].cpu().detach().numpy()
                prediction_index = list(prediction).index(max(prediction))

                total_targets.append(target_index)
                total_predictions.append(prediction_index)

        # calculate loss
        loss = np.mean(losses)

        # calculate accuracy
        #accuracy = 100 * sklearn.metrics.accuracy_score(total_targets, total_predictions)
        accuracy = validate_accuracy(total_targets, total_predictions)

        # calculate precision, recall and F1 scores
        #precision_scores = sklearn.metrics.precision_score(total_targets, total_predictions, average=None)
        precision_scores = precision(total_targets, total_predictions, classes=self.total_classes)

        #recall_scores = sklearn.metrics.recall_score(total_targets, total_predictions, average=None)
        recall_scores = recall(total_targets, total_predictions, classes=self.total_classes)

        #f1_scores = sklearn.metrics.f1_score(total_targets, total_predictions, average=None)
        f1_scores = f1_score(precision_scores, recall_scores)

        # calculate true positive rates
        tp_rates = true_positive_rate(total_targets, total_predictions, classes=len(self.classes.keys()))

        # create confusion matrix
        if confusion_matrix:
            create_confusion_matrix(total_targets, total_predictions, classes=list(self.classes.keys()), save="x-manager/" + self.run_name)
        
        if plot_scores:
            score_plot(precision_scores, recall_scores, f1_scores, classes=list(self.classes.keys()), save="x-manager/" + self.run_name)

        return loss, accuracy, (tp_rates, precision_scores, recall_scores, f1_scores)

    def train(self):
        model = Model(class_amount=self.total_classes, hidden_size=self.hidden_size, layers=self.rnn_layers, dropout_chance=self.dropout_chance, \
                      embedding_size=self.embedding_size, kernel_size=self.kernel_size, channels=self.channels).to(device=device)

        if self.continue_:
            model.load_state_dict(torch.load(self.model_file))

        criterion = nn.NLLLoss(weight=self.loss_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        iterations = 0
        for epoch in range(1, (self.epochs + 1)):

            total_train_targets, total_train_predictions = [], []
            epoch_train_loss = []
            for names, targets, _ in tqdm(self.train_set, desc="epoch", ncols=150):
                optimizer.zero_grad()

                names = names.to(device=device)
                targets = targets.to(device=device)
                predictions = model.train()(names)

                loss = criterion(predictions, targets.squeeze())
                loss.backward()

                lr_scheduler(optimizer, iterations, decay_rate=self.lr_decay_rate, decay_intervall=self.lr_decay_intervall)
                optimizer.step()

                # log train loss
                epoch_train_loss.append(loss.item())
                
                # log targets and prediction of every iteration to compute the train accuracy later
                validated_predictions = model.eval()(names)
                for i in range(validated_predictions.size()[0]): 
                    total_train_targets.append(targets[i].cpu().detach().numpy()[0])
                    validated_prediction = validated_predictions[i].cpu().detach().numpy()
                    total_train_predictions.append(list(validated_prediction).index(max(validated_prediction)))
                
                iterations += 1

            # calculate train loss and accuracy of last epoch
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_train_accuracy = validate_accuracy(total_train_targets, total_train_predictions)

            # calculate validation loss and accuracy of last epoch
            epoch_val_loss, epoch_val_accuracy, _ = self._validate(model, self.validation_set)

            # print training stats in pretty format
            show_progress(self.epochs, epoch, epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy)
            print("\nlr: ", optimizer.param_groups[0]["lr"], "\n")

            # save checkpoint of model
            torch.save(model.state_dict(), self.model_file)

            # log epoch results with xman (uncomment if you have the xman libary installed)
            self.xmanager.log_epoch(model, self.lr, self.batch_size, epoch_train_accuracy, epoch_train_loss, epoch_val_accuracy, epoch_val_loss)

        # plot train-history with xman (uncomment if you have the xman libary installed)
        self.xmanager.plot_history(save=True)

    def test(self, pre_true_positive_rates: list=None, print_examples: bool=False):
        model = Model(class_amount=self.total_classes, hidden_size=self.hidden_size, layers=self.rnn_layers, dropout_chance=0.0, \
                      embedding_size=self.embedding_size, kernel_size=self.kernel_size, channels=self.channels).to(device=device)

        model.load_state_dict(torch.load(self.model_file))

        _, accuracy, scores = self._validate(model, self.test_set, confusion_matrix=True, plot_scores=True)

        if print_examples:
            for names, targets, non_padded_names in tqdm(self.test_set, desc="epoch", ncols=150):
                names = names.to(device=device)
                targets = targets.to(device=device)

                predictions = model.eval()(names)
                predictions, targets, names = predictions.cpu().detach().numpy(), targets.cpu().detach().numpy(), names.cpu().detach().numpy()

                try:
                    for idx in range(len(names)):
                        names, prediction, target, non_padded_name = names[idx], predictions[idx], targets[idx], non_padded_names[idx]

                        # convert to one-hot target
                        amount_classes = prediction.shape[0]
                        target_empty = np.zeros((amount_classes))
                        target_empty[target] = 1
                        target = target_empty

                        # convert log-softmax to one-hot
                        prediction = list(np.exp(prediction))
                        certency = np.max(prediction)
                        
                        prediction = [1 if e == certency else 0 for e in prediction]
                        certency = round(certency * 100, 4)

                        target_class = list(target).index(1)
                        target_class = list(self.classes.keys())[list(self.classes.values()).index(target_class)]
                        
                        try:
                            predicted_class = list(prediction).index(1)
                            predicted_class = list(self.classes.keys())[list(self.classes.values()).index(predicted_class)]
                        except:
                            predicted_class = "else"
        
                        names = char_indices_to_string(char_indices=non_padded_name)
    
                        print("\n______________\n")
                        print("name:", names)
        
                        print("predicted as:", predicted_class, "(" + str(certency) + "%)")
                        print("actual target:", target_class)

                except:
                    pass

                break
        
        # print scores
        true_positive_rates, precisions, recalls, f1_scores = scores
        print("\ntest accuracy:", accuracy)
        print("precision of every class:", precisions)
        print("recall of every class:", recalls)
        print("f1-score of every class:", f1_scores)
        print("\n")
        print("true positive rates:", true_positive_rates)
        print("\n")

        if pre_true_positive_rates != None:
            plot_weight_influence(pre_true_positive_rates=pre_true_positive_rates,
                                  post_true_positive_rates=true_positive_rates,
                                  classes=list(self.classes.keys()))

            var = lambda tp_scores: np.mean([pow(tp_score - np.mean(tp_scores), 2) for tp_score in tp_scores])

            print("variance of the pre-trained true-positive scores: {}".format(var(pre_true_positive_rates)))
            print("variance of the weighted-trained true-positive scores: {}\n".format(var(true_positive_rates)))



def run_experiment(pre_true_positive_rates: list=None):

    if isinstance(pre_true_positive_rates, list):
        average = np.mean(pre_true_positive_rates)

        loss_weights = [pow(1 - ((score - average) / score), 4) for score in pre_true_positive_rates]
        # loss_weights = [pow(2 - score, 4) for score in pre_true_positive_rates]
        print("training using weighted loss, weights: {}".format(loss_weights))
    else:
        loss_weights = None
        
    run = Run(
            run_name="test_run",
            dataset_name="preprocessed-dataset",
            test_size=0.1,
            model_file="model1.pt",
            epochs=20,
            batch_size=1024,
            lr_schedule=[0.001, 0.925, 100],
            rnn_config=[200, 2],
            cnn_config=[1, 3, [64]],
            dropout_chance=0.5,
            embedding_size=200,
            augmentation_chance=0.25,
            loss_weights=loss_weights,
            continue_=False)

    # train
    run.train()

    # test
    run.test(pre_true_positive_rates=pre_true_positive_rates, print_examples=False)
        


run_experiment()

# run_experiment(pre_true_positive_rates=[0.6675007819831091, 0.8160847880299252, 0.5745752045311516, 0.8288201160541586, 0.8395534290271133])