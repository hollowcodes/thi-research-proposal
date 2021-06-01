
import numpy as np
import matplotlib.pyplot as plt

# sklearn alternatives:
# accuracy = 100 * sklearn.metrics.accuracy_score(total_targets, total_predictions)
# precision_scores = sklearn.metrics.precision_score(total_targets, total_predictions, average=None)
# recall_scores = sklearn.metrics.recall_score(total_targets, total_predictions, average=None)
# f1_scores = sklearn.metrics.f1_score(total_targets, total_predictions, average=None)

plt.rcParams.update({"font.size": 25})
plt.rcParams["figure.figsize"] = 5, 5


def validate_accuracy(y_true: list, y_pred: list) -> float:
    """ calculates the accuracy of predictions
    
    :param list y_true: targets
    :param list y_pred: predictions
    :return float: accuracy
    """

    correct_in_batch = 0
    for idx in range(len(y_true)):
        output, target = y_pred[idx], y_true[idx]

        if target == output:
            correct_in_batch += 1
    
    return round((100 * correct_in_batch / len(y_true)), 5)


def create_confusion_matrix(y_true: list, y_pred: list, classes: list=None, save: str=None) -> None:
    """ creates and plots a confusion matrix given two list (targets and predictions)

    :param list y_true: list of all targets (as indices of one-hot enc. vector)
    :param list y_pred: list of all predictions (as indices of one-hot enc. vector)
    :param list classes: list of class names
    """

    amount_classes = len(classes)

    confusion_matrix = np.zeros((amount_classes, amount_classes))
    for idx in range(len(y_true)):
        target = y_true[idx]
        output = y_pred[idx]

        confusion_matrix[target][output] += 1

    fig, ax = plt.subplots(1)

    ax.matshow(confusion_matrix)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    plt.setp(ax.get_xticklabels(), rotation=15, ha="left", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    if save != None:
        try:
            plt.savefig(save + "/confusion_matrix.png")
        except Exception as e:
            print("\ncouldn't save confusion matrix!")
            print(e)

    plt.show()


def precision(y_true: list, y_pred: list, classes: int=10) -> list:
    """ calculates recall scores of classes (against all other classes)

    :param list y_true: list of all targets (as indices of one-hot enc. vector)
    :param list y_pred: list of all predictions (as indices of one-hot enc. vector)
    :param int classes: amount of classes
    :return list: list of the precision scores of each class
    """

    total_prediction_of_classes, total_true_prediction_of_classes = [0 for i in range(classes)], [0 for i in range(classes)]
    for i in range(len(y_true)):
        output, target = y_pred[i], y_true[i]

        for class_ in range(classes):
            if output == class_:
                total_prediction_of_classes[class_] += 1

                if output == target:
                    total_true_prediction_of_classes[class_] += 1

    all_precisions = [0 for i in range(classes)]
    for i in range(classes):
        if total_prediction_of_classes[i] > 0:
            all_precisions[i] = round((total_true_prediction_of_classes[i] / total_prediction_of_classes[i]), 5)
        else:
            all_precisions[i] = 0

    return all_precisions


def recall(y_true: list, y_pred: list, classes: int=10) -> list:
    """ calculates recall scores of all classes (against all other classes)

    :param list y_true: list of all targets (as indices of one-hot enc. vector)
    :param list y_pred: list of all predictions (as indices of one-hot enc. vector)
    :param int classes: amount of classes
    :return list: list of the recall scores of each class
    """

    total_prediction_of_classes, total_true_of_classes = [0 for i in range(classes)], [0 for i in range(classes)]
    for i in range(len(y_true)):
        output, target = y_pred[i], y_true[i]

        for class_ in range(classes):
            if target == class_:
                total_true_of_classes[class_] += 1

                if output == class_:
                    total_prediction_of_classes[class_] += 1

    all_recalls = [0 for i in range(classes)]
    for i in range(classes):
        if total_true_of_classes[i] > 0:
            all_recalls[i] = round((total_prediction_of_classes[i] / total_true_of_classes[i]), 5)
        else:
            all_recalls[i] = 0

    return all_recalls


def f1_score(precisions: list, recalls: list) -> list:
    """ calculates F1 scores of all classes (against all other classes)

    :param list precisions: list containing the precision of each class
    :param list recalls: list containing the recall of each class
    :return list: list of the F1 score of each class
    """
    
    f1_scores = []
    for i in range(len(precisions)):
        precision_score, recall_score = precisions[i], recalls[i]

        try:
            f1_score = round((2 * ((precision_score * recall_score) / (precision_score + recall_score))), 5)
        except:
            f1_score = 0.0

        f1_scores.append(f1_score)

    return f1_scores


def score_plot(precisions: list, recalls: list, f1_scores: list, classes: dict=None, save: str=None) -> None:
    """ plots the precision-, recall- and F!-score for every class

    :param list precisions: list containing the precision of each class
    :param list recalls: list containing the recall of each class
    :param list f1_scores: list containing the f1-score of each class
    """

    plt.style.use("bmh")

    fig, axs = plt.subplots(1, 3)

    axs[0].bar(classes, precisions, color="steelblue", alpha=0.9)
    axs[0].set_xticklabels(classes, rotation=75)
    axs[0].title.set_text("precision scores")

    axs[1].bar(classes, recalls, color="orange", alpha=0.85)
    axs[1].set_xticklabels(classes, rotation=75)
    axs[1].title.set_text("recall scores")

    axs[2].bar(classes, f1_scores, color="forestgreen", alpha=0.85)
    axs[2].set_xticklabels(classes, rotation=75)
    axs[2].title.set_text("f1 scores")

    if save != None:
        try:
            plt.savefig(save + "/scores.png")
        except Exception as e:
            print("\ncouldn't save score plots!")
            print(e)

    plt.show()



def true_positive_rate(y_true: list, y_pred: list, classes: int=10) -> list:
    """ calculates the true positive rate of every class

    :param list y_true: list of all targets (as indices of one-hot enc. vector)
    :param list y_pred: list of all predictions (as indices of one-hot enc. vector)
    :param list classes: list of class names
    """

    true_positive_list = [0 for _ in range(classes)]
    total_class_count_list = [0 for _ in range(classes)]

    for idx in range(len(y_true)):
        target = y_true[idx]
        output = y_pred[idx]

        if target == output:
           true_positive_list[target] += 1

        total_class_count_list[target] += 1

    return [true_positive_list[i] / total_class_count_list[i] for i in range(classes)]


def plot_true_positive_rates(true_positive_rates: list, classes: dict=None, save: str=None) -> None:
    """ plots the precision-, recall- and F!-score for every class

    :param list precisions: list containing the precision of each class
    :param list recalls: list containing the recall of each class
    :param list f1_scores: list containing the f1-score of each class
    """

    plt.style.use("bmh")

    plt.bar(classes, true_positive_rates, color="lightslategray", alpha=0.9, align="center")
    plt.xticks(classes, rotation=50)
    # plt.xticklabels(classes, rotation=25)
    plt.title("true positive scores")

    """fig, axs = plt.subplots(1, 2)
    
    axs[0].bar(classes, true_positive_rates, color="lightslategray", alpha=0.9, align="center")
    axs[0].set_xticklabels(classes, rotation=75)
    axs[0].title.set_text("true positive scores")

    axs[1].bar(classes, [np.mean(true_positive_rates) for _ in range(len(classes))], color="lightslategray", alpha=0.9, align="center")
    axs[1].set_xticklabels(classes, rotation=75)
    axs[1].title.set_text("desired true positive scores")
    axs[1].set_ylim([0, 1.0])"""

    if save != None:
        try:
            plt.savefig(save + "/scores.png")
        except Exception as e:
            print("\ncouldn't save score plots!")
            print(e)

    plt.show()


def plot_weight_influence(pre_true_positive_rates: list, post_true_positive_rates: list, classes: dict=None, save: str=None) -> None:
    """ plots the precision-, recall- and F!-score for every class

    :param list precisions: list containing the precision of each class
    :param list recalls: list containing the recall of each class
    :param list f1_scores: list containing the f1-score of each class
    """

    plt.style.use("bmh")
    
    nrange = np.arange(len(classes))
    width = 0.35


    plt.bar(nrange - width, pre_true_positive_rates, color="steelblue", width=(width - 0.05), alpha=0.9, align="center")
    plt.bar(nrange, post_true_positive_rates, color="orange", width=(width - 0.05), alpha=0.9, align="center")
    plt.xticks(nrange, classes, rotation=25)
    plt.title("true positive scores")

    if save != None:
        try:
            plt.savefig(save + "/scores.png")
        except Exception as e:
            print("\ncouldn't save score plots!")
            print(e)

    plt.show()

