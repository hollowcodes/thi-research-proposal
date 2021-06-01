
import string
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import json
import random
from cifarTenDataset import CifarTenDataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def show_progress(epochs: int, epoch: int, train_loss: float, train_accuracy: float, val_loss: float, val_accuracy: float):
    """ print training stats
    
    :param int epochs: amount of total epochs
    :param int epoch: current epoch
    :param float train_loss/train_accuracy: train-loss, train-accuracy
    :param float val_loss/val_accuracy: validation accuracy/loss
    :return None
    """

    """ colored not working on anaconda powershell
    epochs = colored(epoch, "cyan", attrs=["bold"]) + colored("/", "cyan", attrs=["bold"]) + colored(epochs, "cyan", attrs=["bold"])
    train_accuracy = colored(round(train_accuracy, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
    train_loss = colored(round(train_loss, 6), "cyan", attrs=["bold"])
    val_accuracy = colored(round(val_accuracy, 4), "cyan", attrs=["bold"]) + colored("%", "cyan", attrs=["bold"])
    val_loss = colored(round(val_loss, 6), "cyan", attrs=["bold"])
    """
    
    epochs = epoch
    train_accuracy = round(train_accuracy, 4)
    train_loss = round(train_loss, 6)
    val_accuracy = round(val_accuracy, 4)
    val_loss = round(val_loss, 6)
    
    print("epoch {} train_loss: {} - train_acc: {} - val_loss: {} - val_acc: {}".format(epochs, train_loss, train_accuracy, val_loss, val_accuracy), "\n")


def lr_scheduler(optimizer: torch.optim, current_iteration: int=0, warmup_iterations: int=0, lr_end: float=0.001, 
                 decay_rate: float=0.99, decay_intervall: int=100, print_lr_state: bool=False) -> None:

    current_iteration += 1
    current_lr = optimizer.param_groups[0]["lr"]

    if decay_intervall == 0:
        return None

    if current_iteration <= warmup_iterations:
        optimizer.param_groups[0]["lr"] = (current_iteration * lr_end) / warmup_iterations
        if print_lr_state: print("-> warumup:", optimizer.param_groups[0]["lr"])

    elif current_iteration > warmup_iterations and current_iteration % decay_intervall == 0:
        optimizer.param_groups[0]["lr"] = current_lr * decay_rate
        if print_lr_state: print("-> warumup:", optimizer.param_groups[0]["lr"])
    else:
        pass







