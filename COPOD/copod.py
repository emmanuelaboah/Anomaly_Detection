import os
import csv

import pandas as pd
import numpy as np
from pyod.models.copod import COPOD


class CopodModel:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.clf = COPOD()

    def train(self):
        self.clf.fit(self.X_train)
        y_train_scores = self.clf.decision_scores_
        return y_train_scores

    def test(self):
        y_test_scores = self.clf.decision_function(self.X_test)
        return y_test_scores

    def predict(self, test_df=None):
        if test_df is None:
            x_test = self.X_test
        else:
            x_test = test_df

        y_test_labels = self.clf.predict(x_test)
        return y_test_labels
