# coding=utf-8

import numpy as np
from sklearn.metrics import confusion_matrix


class ConfMat(object):
    """Confusion matrix
    Args:
        y (np.ndarray): 1-d array, actual labels
        pred (np.ndarray): 1-d array, predicted labels
    """
    def __init__(self, y, pred):
        assert len(y) == len(pred)
        confmat = confusion_matrix(y, pred).transpose().astype(int)
        self.n_classes = confmat.shape[0]
        self.confmat = np.zeros((self.n_classes + 1, self.n_classes + 1))
        self.confmat[:-1, :-1] = confmat

        self.accuracy = self.confmat.diagonal().sum() / self.confmat.sum()
        self.precisions = np.zeros(self.n_classes)
        self.recalls = np.zeros(self.n_classes)
        for i in range(self.n_classes):
            row_sum = self.confmat[i, :].sum()
            self.precisions[i] = self.confmat[i, i] / row_sum if row_sum > 0 else 0

            col_sum = self.confmat[:, i].sum()
            self.recalls[i] = self.confmat[i, i] / col_sum if col_sum > 0 else 0

        self.confmat[-1, :-1] = self.recalls
        self.confmat[:-1, -1] = self.precisions
        self.confmat[-1, -1] = self.accuracy

    def detail_str(self):
        confmat_str = ""
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                confmat_str += "{:>9.0f} ".format(self.confmat[i, j])

            # precision
            confmat_str += "| {:.4f} ".format(self.precisions[i])
            # total number of predicted class
            confmat_str += "({:>9.0f})\n".format(self.confmat[i, :-1].sum())

        for _ in range(self.n_classes):
            confmat_str += '----------'

        confmat_str += '+--------------------\n'

        # recalls
        for j in range(self.n_classes):
            confmat_str += "{:>9.4f} ".format(self.recalls[j])

        # total accuracy
        confmat_str += "| {:.4f}\n".format(self.accuracy)

        # total number of actual class
        for j in range(self.n_classes):
            confmat_str += "({:>7.0f}) ".format(self.confmat[:-1, j].sum())

        confmat_str += "| ({:>7.0f})".format(self.confmat[:-1, :-1].sum())

        return confmat_str

    def matrix_str(self):
        confmat_str = ""
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                confmat_str += "{:>7.0f},".format(self.confmat[i, j])

            confmat_str += "{:>7.3f}\n".format(self.precisions[i])

        for j in range(self.n_classes):
            confmat_str += "{:>7.3f},".format(self.recalls[j])

        confmat_str += "{:>7.3f}".format(self.accuracy)
        return confmat_str

    def __str__(self):
        confmat_str = "["
        for i in range(self.n_classes):
            confmat_str += "["
            for j in range(self.n_classes):
                confmat_str += "{:>7.0f},".format(self.confmat[i, j])

            confmat_str += "{:>7.3f}], ".format(self.precisions[i])

        confmat_str += "["
        for j in range(self.n_classes):
            confmat_str += "{:>7.3f},".format(self.recalls[j])

        confmat_str += "{:>7.3f}]".format(self.accuracy)
        confmat_str += "]"
        return confmat_str