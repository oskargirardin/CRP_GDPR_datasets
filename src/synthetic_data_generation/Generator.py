import pandas as pd
import torch

from sdv.tabular import CTGAN, GaussianCopula
from sdv.evaluation import evaluate
from table_evaluator import TableEvaluator


class Generator:

    def __init__(self, n_epochs, n_samples,architecture, data, categorical_columns, sensitive_columns):

        """
        :param n_epochs: the number of epochs used for training
        :param n_samples: the number of rows to generate
        :param architecture: the chosen architecture, one of ['CTGAN', 'GaussianCopula']
        :param data: the data that should be trained on, should be in a pandas dataframe
        :param categorical_columns: a list with categorical columns
        :param sensitive_columns: a dict with sensitive columns and what faker category they belong to
        """

        self.n_epochs = n_epochs
        self.n_samples = n_samples
        if architecture in ['CTGAN', 'GaussianCopula']:
            self.architecture = architecture
        else:
            print('The requested architecture is not available')
            raise ValueError

        self.data = data
        self.categorical_columns = categorical_columns
        self.sensitive_columns = sensitive_columns

    def generate(self):
        """
        Based on the chosen architecture, this function returns synthetically generated data
        :return: synthetic data, a pandas dataframe
        """

        # TODO: Add more generators, especially PATEGAN or other differentially private ones synthcity seems to have
        #  implementations of these, but I have not been able to import their library

        if self.architecture == "CTGAN":
            model = CTGAN(epochs=self.n_epochs)
            model.fit(self.data)
            synth_data = model.sample(self.n_samples)

        elif self.architecture == "GaussianCopula":
            model = GaussianCopula()
            model.fit(self.data)
            synth_data = model.sample(self.n_samples)

        return synth_data

    def anonymize(self):
        """
        Tries to anonymize sensitive columns for example with faker, or chatgpt
        :return: the synthetic data with anonymized columns
        """
        for column in self.sensitive_columns:
            pass











