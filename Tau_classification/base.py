"""

Module which contains helper functions
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from statistics import mean
from constants import extracted_features


def get_threshold(best_params):
    """
    Get all class-specific thresholds from best parameters.
    """
    class_thresholds = []
    for i in best_params:
        t = best_params[i][0]
        class_thresholds.append(t)
    return class_thresholds


def multiclass_PR_curves(n_class, test_y_numeric, predy):
    """
    Create PR curve for each class in multi-classification problem.
    """
    precision = {}
    recall = {}
    thresh = {}

    # calculate PR curve locations
    for i in range(n_class):  # for each class, calculate roc_curve
        precision[i], recall[i], thresh[i] = precision_recall_curve(
            test_y_numeric, predy[:, i], pos_label=i)

    return precision, recall, thresh


def multiclass_roc_curves(n_class, test_y_numeric, predy):
    """
    Create ROC curve for each class in multi-classification problem.
    """
    fpr = {}
    tpr = {}
    thresh = {}

    # calcualte roc curve locations
    for i in range(n_class):  # for each class, calculate roc_curve
        fpr[i], tpr[i], thresh[i] = roc_curve(
            test_y_numeric, predy[:, i], pos_label=i)

    return fpr, tpr, thresh


def best_param_f_score(n_class, precision, recall, thresh):
    """
    Find the best position on the precision-recall curve
    for multi-classification problem.
    """
    # calculate f-score for each threshold
    best_params = {}
    for i in range(n_class):
        p = precision[i]
        r = recall[i]
        nu = (2*p*r)
        de = (p+r)
        f_score = np.divide(nu, de, out=np.zeros_like(nu), where=de != 0)
        t = thresh[i]
        ix = np.argmax(f_score)
        best_params[i] = (t[ix], f_score[ix], p[ix], r[ix])
    return best_params


def prob_thresholding(y_pred_prob, y_pred, threshold):
    """
    A function to perform thresholding to convert
    scores into crisp class label.
    """
    thresholded_class = []
    for i in range(0, len(y_pred_prob)):
        if(max(y_pred_prob[i]) < threshold):
            c = 'Ambiguous'
        else:
            c = y_pred[i]
        thresholded_class.append(c)
    return thresholded_class


def remove_amb_class(t_class, y_test):
    """
    Removing 'ambiguous' class from the thresholded class & y_predict.
    """
    # get indices of instances with no ambiguous label
    x = pd.Series(t_class)
    y_pred_no_amb = x[x != 'Ambiguous']
    y_pred_no_amb_indices = y_pred_no_amb.index

    # extract these instances fom y_pred
    # y_predict_no_amb = y_predict.iloc[pred_no_amb_indices]

    # subset y_test
    y_test_no_amb = y_test.iloc[y_pred_no_amb_indices]

    return (y_pred_no_amb, y_test_no_amb)


def precision_recall_auc_screening(clf, X, Y):
    """
    Calculates precision-recall area under the curve
     of screening stage ('Tau','Non-tau).
    """
    # variables
    pr_score = {}

    # get y prob predictions
    y_prob_pred = clf.predict_proba(X)

    # convert true y name into numerical classes
    y_true_numeric = name_to_numeric_classes_c1(Y)

    # get number of classes
    n_class = list(set(Y))

    # create PR curve using OVR approach
    for i in range(len(n_class)):  # for each class, calculate roc_curve
        p, r, thresh = precision_recall_curve(
            y_true_numeric, y_prob_pred[:, i], pos_label=i)
        pr_score[i] = auc(r, p)  # recall on x axis, precision on y axis

    # combine all pr-scores using 'macro' method
    pr_auc = mean(pr_score.values())
    return pr_auc


def precision_recall_auc_tau(clf, X, Y):
    """
    Calculates precision-recall area under the curve
     of tau hallmark classification ('CB','NFT','Others','TA').
    """
    # variables
    pr_score = {}

    # get y prob predictions
    y_prob_pred = clf.predict_proba(X)

    # convert true y name into numerical classes
    y_true_numeric = name_to_numeric_classes_c2(Y)

    # get number of classes
    n_class = list(set(Y))

    # create PR curve using OVR approach
    for i in range(len(n_class)):  # for each class, calculate roc_curve
        p, r, thresh = precision_recall_curve(
            y_true_numeric, y_prob_pred[:, i], pos_label=i)
        pr_score[i] = auc(r, p)  # recall on x axis, precision on y axis

    # combine all pr-scores using 'macro' method
    pr_auc = mean(pr_score.values())
    return pr_auc


def precision_recall_auc_tau_noTA(clf, X, Y):
    """
    Calculates precision-recall area under the curve
     of tau hallmark classification ('CB','NFT','Others').
    """
    # Variables
    pr_score = {}

    # get y prob predictions
    y_prob_pred = clf.predict_proba(X)

    # Convert true y name into numerical classes
    y_true_numeric = name_to_numeric_classes_c2_noTA(Y)

    # get number of classes
    n_class = list(set(Y))

    # create PR curve using OVR approach
    for i in range(len(n_class)):  # for each class, calculate roc_curve
        p, r, thresh = precision_recall_curve(
            y_true_numeric, y_prob_pred[:, i], pos_label=i)
        pr_score[i] = auc(r, p)  # recall on x axis, precision on y axis

    # combine all pr-scores using 'macro' method
    pr_auc = mean(pr_score.values())
    return pr_auc


def threshold_list_c1(y_pred_prob, best_params):
    """
    Applies class-specific thresholding to each detection
     from classifier 1 (non-tau, tau, ambiguous).
    """
    thresholded_classes = []
    for i in y_pred_prob:  # for each detection

        # get 2 class-specific threshold values:
        thresholds = get_threshold(best_params)

        # calculate predicted probability - threshold
        # = difference for each of the classes
        differences = (i-thresholds)/thresholds

        # count number of positive or equal (0) differences
        count = np.count_nonzero(differences >= 0)

        if (count == 1):  # only assign class when 1 class passes the threshold
            pred_class = np.argmax(differences)
        else:  # otherwise, label as ambiguous
            # (when more than 1 class passes, or when no class passes)
            pred_class = 2

        # putting prediction in a list
        thresholded_classes.append(pred_class)

    thresholded_classes_ = numeric_to_name_classes_c1(thresholded_classes)
    return thresholded_classes_


def threshold_list_c2_noTA(y_pred_prob, best_params):
    """
    Applies class-specific thresholding to each detection
     from classifier 2 (CB, NFT, Others, Ambiguous).
    """
    thresholded_classes = []
    for i in y_pred_prob:  # for each cell (containing 4 class probabilities)

        # Get cell 2 class-specific threshold values:
        thresholds = get_threshold(best_params)

        # Calculate predicted probability - threshold
        #  = difference for each of the 4 classes
        differences = (i-thresholds) / thresholds

        # Count number of positive or equal (0) differences
        count = np.count_nonzero(differences >= 0)

        if (count == 1):  # only assign class when 1 class passes the threshold
            pred_class = np.argmax(differences)
        else:  # Otherwise, label as ambiguous
            # (when more than 1 class passes, or when no class passes)
            pred_class = 3

        # putting prediction in a list
        thresholded_classes.append(pred_class)

    thresholded_classes_ = numeric_to_name_classes_c2_noTA(thresholded_classes)
    return thresholded_classes_


def threshold_list_c2(y_pred_prob, best_params):
    """
    Applies class-specific thresholding to each detection
     from classifier 2 (CB, NFT, Others, TA, Ambiguous).
    """
    thresholded_classes = []
    for i in y_pred_prob:  # for each detection

        # get class-specific threshold values:
        thresholds = get_threshold(best_params)

        # calculate predicted probability - threshold
        # = difference for each of the classes
        differences = (i-thresholds)/thresholds

        # count number of positive or equal (0) differences
        count = np.count_nonzero(differences >= 0)

        if (count == 1):  # only assign class when 1 class passes the threshold
            pred_class = np.argmax(differences)
        else:  # otherwise, label as ambiguous
            # (when more than 1 class passes, or when no class passes)
            pred_class = 4

        # putting prediction in a list
        thresholded_classes.append(pred_class)

    thresholded_classes_ = numeric_to_name_classes_c2(thresholded_classes)
    return thresholded_classes_


def numeric_to_name_classes_c1(numeric_classes):
    """
    Converts numeric to its corresponding name classes for classifier 1.
    """
    code = {0: 'Non_tau', 1: 'Tau', 2: 'Ambiguous'}
    output = [code[i] for i in numeric_classes]
    return output


def numeric_to_name_classes_c2_noTA(numeric_classes):
    """
    Converts numeric to its corresponding name classes
     for classifier 2 (CB, NFT, Others, Ambiguous).
    """
    code = {0: 'CB', 1: 'NFT', 2: 'Others', 3: 'Ambiguous'}
    output = [code[i] for i in numeric_classes]
    return output


def numeric_to_name_classes_c2(numeric_classes):
    """
    Converts numeric to its corresponding name classes
     for classifier 2 (CB, NFT, Others, TA, Ambiguous).
    """
    code = {0: 'CB', 1: 'NFT', 2: 'Others', 3: 'TA', 4: 'Ambiguous'}
    output = [code[i] for i in numeric_classes]
    return output


def name_to_numeric_classes_c1(name_classes):
    """
    Converts name to its corresponding numeric classes for classifier 1.
    """
    code = {'Non_tau': 0, 'Tau': 1, 'Ambiguous': 2}
    output = [code[i] for i in name_classes]
    return output


def name_to_numeric_classes_c2_noTA(name_classes):
    """
    Converts name to its corresponding numeric classes
     for classifier 2 (CB,NFT,Others, Ambiguous).
    """
    code = {'CB': 0, 'NFT': 1, 'Others': 2, 'Ambiguous': 3}
    output = [code[i] for i in name_classes]
    return output


def name_to_numeric_classes_c2(name_classes):
    """
    Converts name to its corresponding numeric classes
     for classifier 2 (CB,NFT,Others, TA Ambiguous).
    """
    code = {'CB': 0, 'NFT': 1, 'Others': 2, 'TA': 3, 'Ambiguous': 4}
    output = [code[i] for i in name_classes]
    return output


# Creating compatible data to be used with our tau classification pipeline
class TauDataBase():
    """
    Reading in file path to annotated detections and formatting them.
    """
    def __init__(self, path, filename):

        # importing annotated detections from slides
        with open(path+filename) as f:
            mylist = f.read().splitlines()

        inputs = []
        for i in mylist:
            data = pd.read_csv(path + i, sep="\t")
            # Changing column names
            # since these names tend to be inconsistent causing problems
            data_ = data[extracted_features]
            data_.columns.values[5] = 'Centroid_X'
            data_.columns.values[6] = 'Centroid_Y'

            inputs.append(data_)

        # combine detections across slides
        labelled_orig = pd.concat(inputs, sort=True)
        dat = labelled_orig.drop(columns=['Name', 'Parent', 'ROI'])

        # Attributes
        self.data = dat
        self.c1_data = None
        self.c2_data = None

    def classifier1_prep(self):
        """
        Data preparation for cleaning data for classifier 1 (tau, non-tau)
        """
        # Put them into: non-tau, tau classes
        dat = self.data

        dat_ = dat[(dat['Class'] == 'TA') |
                   (dat['Class'] == 'CB') |
                   (dat['Class'] == 'NFT') |
                   (dat['Class'] == 'Non_tau') |
                   (dat['Class'] == 'Others')]
        dat_ = dat_.reset_index(drop=True)

        # Pool tau_fragments & non_tau together -> Others class
        class_ = dat_['Class']
        y = ['Tau' if i != 'Non_tau' else i for i in class_]
        data = dat_.copy()
        data.loc[:, 'Class'] = y
        self.c1_data = data

        # Create training set
        X_train = data.drop(
            columns=['Class',
                     'Image',
                     'Centroid_X',
                     'Centroid_Y'])
        Y_train = data['Class']
        self.c1_X_train = X_train
        self.c1_Y_train = Y_train
        self.c1_train_location = data.drop(columns=['Class'])

    def classifier2_prep(self):
        """
        Data preparation for cleaning data for classifier 2 (tau hallmark)
        """
        dat = self.data
        # Only retaining tau hallmarks
        dat_ = dat[(dat['Class'] == 'TA') |
                   (dat['Class'] == 'CB') |
                   (dat['Class'] == 'NFT') |
                   (dat['Class'] == 'Others')]
        dat_ = dat_.reset_index(drop=True)
        self.c2_data = dat_

        # Create training set
        X_train = dat_.drop(columns=['Class',
                                     'Image',
                                     'Centroid_X',
                                     'Centroid_Y'])
        Y_train = dat_['Class']
        self.c2_X_train = X_train
        self.c2_Y_train = Y_train
        self.c2_train_location = dat_.drop(columns=['Class'])
