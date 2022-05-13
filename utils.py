from sklearn.metrics import accuracy_score, roc_auc_score
from ensemble import HashBasedUndersamplingEnsemble
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import math
from math import sqrt
from sklearn.metrics import recall_score, f1_score, confusion_matrix

def prepare(X: np.array, y: np.array, minority=None, verbose: bool = False):
    """Preparing Data for Ensemble
    Make the data binary by minority class in the dataset

    :param X: np.array (n_samples, n_features)
        feature matrix

    :param y: np.array (n_samples,)
        label vecotr

    :param minority: int or str (default = None)
    label of minority class
    if you want to set a specific class to be minority class

    :param verbose: bool (default = False)
        verbosity

    :return: np.array, np.array
        X, y returned
    """

    # Get classes and number of them
    classes, counts = np.unique(y, return_counts=True)

    if minority is None:
        # find minority class
        minority = classes[np.argmin(counts)]

    if minority not in classes:
        raise ValueError("class '{}' does not exist".format(
            minority
        ))

    # set new label for data (1 for minority class and -1 for rest of data)
    y_ = np.where(y == minority, 1, -1)

    if verbose:
        information = "[Preparing]\n" \
                      "+ #classes: {}\n" \
                      "+ classes and counts: {}\n" \
                      "+ Minority class: {}\n" \
                      "+ Size of Minority: {}\n" \
                      "+ Size of Majority: {}\n" \
                      "".format(len(classes),
                                list(zip(classes, counts)),
                                minority,
                                np.sum(y_ == 1),
                                np.sum(y_ != 1),
                                )

        print(information)

    return X, y_


def evaluate(
        name,
        base_classifier,
        X,
        y,
        X_test,
        y_test,
        minority_class=None,
        k: int = 5,
        n_runs: int = 20,
        n_iterations: int = 100,
        random_state: int = None,
        verbose: bool = False,
        **kwargs
):
    """Model Evaluation

    :param name: str
        title of this classifier

    :param base_classifier:
        Base Classifier for Hashing-Based Undersampling Ensemble

    :param X: np.array (n_samples, n_features)
        Feature matrix

    :param y: np.array (n_samples,)
        labels vector

    :param minority_class: int or str (default = None)
        label of minority class
        if you want to set a specific class to be minority class

    :param k: int (default = 5)
        number of Folds (KFold)

    :param n_runs: int (default = 20)
        number of runs

    :param n_iterations: int (default = 50)
        number of iterations for Iterative Quantization of Hashing-Based Undersampling Ensemble

    :param random_state: int (default = None)
        seed of random generator

    :param verbose: bool (default = False)
        verbosity

    :return None
    """

    print()
    print("======[Dataset: {}]======".format(
        name
    ))

    np.random.seed(random_state)

    # Output template
    OUTPUT = "[{}] Bal: {:.4f}, AUC: {:.4f}, F1: {:.4f}, PF: {:.4f}, PD: {:.4f}"

    # Prepate the data (Make it Binary)
    X = np.array(X)
    y = np.array(y)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X, y = prepare(X, y, minority_class, verbose)
    X_test, y_test = prepare(X_test, y_test, minority_class, verbose)

    folds = np.zeros((n_runs, 5))
    for run in tqdm(range(n_runs)):

        # Applying k-Fold (k = 5 due to the paper)
        kFold = StratifiedKFold(n_splits=k, shuffle=True)
        #print(kFold)
        # store metrics in this variable
        metrics = np.zeros((k, 5))
        #i = 0
        #for fold, (trIndexes, tsIndexes) in enumerate(kFold.split(X, y)):
        for fold in range(5):
            # Split data to Train and Test
            #i = i+1
            #print(trIndexes)
            #print(np.shape(trIndexes))
            #Xtr, ytr = X[trIndexes], y[trIndexes]
            #Xts, yts = X[tsIndexes], y[tsIndexes]
            Xtr, ytr = X, y
            Xts, yts = X_test, y_test
            #print(np.shape(Xtr))
            #print(np.shape(Xts))
            # Define Model
            model = HashBasedUndersamplingEnsemble(
                base_estimator=base_classifier,
                n_iterations=n_iterations,
                random_state=random_state,
                **kwargs
            )

            # Fit the training data on the model
            model.fit(Xtr, ytr)

            # Predict the test data
            predicted = model.predict(Xts)

            # AUC evaluation
            AUC = roc_auc_score(yts, predicted)


            #bal evaluation
            recall = recall_score(yts, predicted, average='macro')
            one, two = confusion_matrix(y_true=yts, y_pred=predicted)
            TP = one[0]
            FN = one[1]
            FP = two[0]
            TN = two[1]
            PF = FP / (FP + TN)
            PD = TP / (TP + FN)
            bal = 1 - (sqrt(pow(PF, 2) + pow((1 - recall), 2)) / sqrt(2))


            fm = f1_score(yts, predicted, average='macro')
            # Accuracy evaluation
            accuracy = accuracy_score(yts, predicted)

            # Show result for each step
            metrics[fold, :] = [bal, AUC, fm, PF, PD]

        folds[run, :] = np.mean(metrics, axis=0)

    print(OUTPUT.format(
        "Best",
        *np.max(folds, axis=0)
    ))