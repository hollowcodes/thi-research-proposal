
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import List, Tuple
import sklearn.metrics

import torch
import torch.utils.data
import torch.nn as nn

from model import Model
from utils import device, show_progress, lr_scheduler
from cifarTenDataset import create_dataloader
from test_metrics import validate_accuracy, create_confusion_matrix, recall, precision, f1_score, score_plot, plot_weight_influence, true_positive_rate
import xman


torch.manual_seed(0)


class Run:
    def __init__(self, run_name: str="", model_file: str="", dataset_path: str="", classes: str="", epochs: int=10, lr: float=0.001, lr_decay: Tuple[float]=(100, 0.99), 
                 batch_size: int=32, loss_weights: list=None, continue_: bool=False):

        # name of current run
        self.run_name = run_name

        # dataset and model path
        self.model_file = model_file
        self.dataset_path = dataset_path
        self.classes = classes

        # hyperparameters
        self.epochs = epochs
        self.lr = lr
        self.lr_decay_intervall = lr_decay[0]
        self.lr_decay_rate = lr_decay[1]
        self.batch_size = batch_size
        self.hyperparameter_config = {}

        # load weights if wanted
        self.loss_weights = torch.Tensor(loss_weights).type(torch.FloatTensor).to(device=device) if loss_weights != None else None

        # creator dataloders
        self.train_dataset = create_dataloader(dataset_path=self.dataset_path + "train_dataset.npy", batch_size=self.batch_size, class_amount=len(self.classes.keys()))
        self.validation_dataset = create_dataloader(dataset_path=self.dataset_path + "val_dataset.npy", batch_size=self.batch_size, class_amount=len(self.classes.keys()))
        self.test_dataset = create_dataloader(dataset_path=self.dataset_path + "test_dataset.npy", batch_size=self.batch_size, class_amount=len(self.classes.keys()))

        # resume bool
        self.continue_ = continue_

        # initialize experiment manager
        self.xmanager = xman.ExperimentManager(experiment_name=self.run_name, continue_=self.continue_)
        self.xmanager.init(optimizer="Adam",
                            loss_function="CrossEntropy",
                            epochs=self.epochs,
                            learning_rate=self.lr,
                            batch_size=self.batch_size,
                            custom_parameters=self.hyperparameter_config)

    def _validate(self, model, dataset, confusion_matrix: bool=False, plot_scores: bool=False):
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        total_targets, total_predictions = [], []
        for names, targets in tqdm(dataset, desc="validating", ncols=150):
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
        # accuracy = 100 * sklearn.metrics.accuracy_score(total_targets, total_predictions)
        accuracy = validate_accuracy(total_targets, total_predictions)

        # calculate precision, recall and F1 scores
        # precision_scores = sklearn.metrics.precision_score(total_targets, total_predictions, average=None)
        precision_scores = precision(total_targets, total_predictions, classes=len(self.classes.keys()))

        # recall_scores = sklearn.metrics.recall_score(total_targets, total_predictions, average=None)
        recall_scores = recall(total_targets, total_predictions, classes=len(self.classes.keys()))

        # f1_scores = sklearn.metrics.f1_score(total_targets, total_predictions, average=None)
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
        # load model (resume if continue_=True)
        model = Model(class_amount=len(self.classes.keys())).to(device=device)
        if self.continue_:
            model.load_state_dict(torch.load(self.model_file))
            # self.lr = self.xmanager.get_last_lr()

        # loss and optimizer function
        criterion = nn.CrossEntropyLoss(weight=self.loss_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # model iteration counter
        iterations = 0

        # epoch loops
        for epoch in range(1, (self.epochs + 1)):
            
            # current batch train data targets, predictions and loss for train-set evaluation
            total_train_targets, total_train_predictions = [], []
            epoch_train_loss = []

            # dataset batch loop
            for images, targets in tqdm(self.train_dataset, desc="epoch", ncols=150):
                # clear gradients
                optimizer.zero_grad()

                # feed data through model
                images, targets = images.to(device=device), targets.to(device=device)
                predictions = model.train()(images, print_=False)
                
                # backpropagation
                loss = criterion(predictions, targets.squeeze())
                loss.backward()

                # learning rate schedule step
                lr_scheduler(optimizer, iterations, decay_rate=self.lr_decay_rate, decay_intervall=self.lr_decay_intervall, print_lr_state=False)

                # update model parameters
                optimizer.step()

                # log train loss
                epoch_train_loss.append(loss.item())
                
                # log targets and prediction of every iteration to compute the accuracy later
                validated_predictions = model.eval()(images)
                for i in range(validated_predictions.size()[0]): 
                    total_train_targets.append(targets[i].cpu().detach().numpy()[0])
                    validated_prediction = validated_predictions[i].cpu().detach().numpy()
                    total_train_predictions.append(list(validated_prediction).index(max(validated_prediction)))

                # count iterations                
                iterations += 1

            # calculate train loss and accuracy of last epoch
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_train_accuracy = validate_accuracy(total_train_targets, total_train_predictions)

            # calculate validation loss and accuracy of last epoch
            epoch_val_loss, epoch_val_accuracy, _ = self._validate(model, self.validation_dataset)

            # print training stats in pretty format
            show_progress(self.epochs, epoch, epoch_train_loss, epoch_train_accuracy, epoch_val_loss, epoch_val_accuracy)
            print("\nlr: {} \n".format(optimizer.param_groups[0]["lr"]))

            # save checkpoint of model
            torch.save(model.state_dict(), self.model_file)

            # log epoch results with xman
            self.xmanager.log_epoch(model, optimizer.param_groups[0]["lr"], self.batch_size, epoch_train_accuracy, epoch_train_loss, epoch_val_accuracy, epoch_val_loss)

        # plot train-history with xman
        self.xmanager.plot_history(save=True)

    def test(self, pre_true_positive_rates: list=None, print_examples: bool=True):
        # load model from last training checkpoint
        model = Model(class_amount=len(self.classes.keys())).to(device=device)
        model.load_state_dict(torch.load(self.model_file))

        # validate test-set and print accuracy
        loss, accuracy, scores = self._validate(model, self.test_dataset, confusion_matrix=True, plot_scores=True)

        # print single images and predictions of the test-set if wanted
        if print_examples:
            for images, targets in tqdm(self.test_dataset, desc="epoch", ncols=150):
                images, targets = images.to(device=device), targets.to(device=device)

                # predict whole batch
                predictions = model.eval()(images)

                # convert batch from tensor to list
                predictions, targets, images = predictions.cpu().detach().numpy(), targets.cpu().detach().numpy(), images.cpu().detach().numpy()

                # loop through batch
                for idx in range(len(predictions)):
                    prediction = [round(e, 3) for e in predictions[idx]]
                    one_hot_target = np.zeros((len(self.classes.keys())))
                    one_hot_target[targets[idx][0]] = 1
                    print("prediction: {} actual: {}".format(prediction, one_hot_target))

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



# load runner class
def run_experiment(pre_true_positive_rates: list=None):

    if isinstance(pre_true_positive_rates, list):
        average = np.mean(pre_true_positive_rates)

        # loss_weights = [pow(1 - ((score - average) / score), 2) for score in pre_true_positive_rates]

        #print("training using weighted loss, weights: {}".format(loss_weights))
        loss_weights = [pow(2 - score, 4) for score in pre_true_positive_rates]
        print(loss_weights)
    else:
        loss_weights = None
        
    run = Run(
            run_name="test_run",
            model_file="models/model1.pt",
            dataset_path="../datasets/preprocessed_datasets/cifar-10/",
            classes={"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9},
            epochs=35,
            lr=0.001,
            lr_decay=(100, 0.925),
            batch_size=512,
            loss_weights=loss_weights,
            continue_=False)

    # train
    run.train()

    # test
    run.test(print_examples=False, pre_true_positive_rates=pre_true_positive_rates)


# run_experiment()

run_experiment(pre_true_positive_rates=[0.8363273453093812, 0.9284294234592445, 0.7088846880907372, 0.6540755467196819, 0.7793522267206477, 0.7243589743589743, 0.8722943722943723, 0.8511066398390342, 0.854043392504931, 0.8899253731343284])


# 35 E - start 0.001 - lr 100 0.925

# PRE acc: 81.02%

# good functions: 
#    [pow(2 - score, 3) for score in pre_true_positive_rates]