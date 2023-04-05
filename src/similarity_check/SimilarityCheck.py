from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from table_evaluator import TableEvaluator
from sdmetrics.reports.single_table import QualityReport

# TODO: ML model performance and test correlation

#############
# The quality report needs metadata
# sdv can deduce this from the dataframe, but it does not yet work on my computer
#from sdv.metadata import SingleTableMetadata
#metadata = SingleTableMetadata()
#metadata.detect_from_dataframe(data=my_pandas_dataframe)


class SimilarityCheck:

    '''
    Check the quality of the synthetic data both visually and with metrics
    '''

    def __init__(self, real_data, synthetic_data, cat_cols, metadata):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.cat_cols = cat_cols
        self.metadata = metadata
        ############
        # We can immediately call the functions upon initialization
        self.comparison_columns()
        self.visual_comparison_columns()

    #def check_similarity(self):
      #  evaluator = TableEvaluator(real=self.real_data, fake=self.synthetic_data, cat_cols=self.cat_cols)
      #  evaluator.visual_evaluation()

    def comparison_columns(self):
        '''
        :return: the KL divergence for numerical variables...
        basically a numerical performance for the whole synthetic data and for each column
        requires metadata, which is defined in the main for this data!!
        '''
        report = QualityReport()
        print(report.generate(self.real_data, self.synthetic_data, self.metadata))
        print(report.get_details(property_name='Column Shapes')
)

    def visual_comparison_columns(self):
        '''
        Plot data in one of three ways:
        1) numeric columns are plotted using the densities
        2) categorical columns with limited categories are ideally plotted with a bar: find a way to put them
        next to each other, potentially by making one big df and using 'hue'
        3)
        :return:
        '''
        if (self.real_data.columns != self.synthetic_data.columns).all():
            print('Columns in real and synthetic data not the same!')
            return

        fig, ax = plt.subplots(nrows=len(self.real_data.columns), figsize=(100, 100))

        for i, column in enumerate(self.real_data.columns):

            if column not in self.cat_cols:
                sns.kdeplot(self.real_data[column], ax=ax[i], label='Real', fill=True, color='c')
                sns.kdeplot(self.synthetic_data[column], ax=ax[i], label='Synthetic', fill=True, color='m')
            elif len(self.real_data[column].unique()) <= 5:
                sns.histplot(data = self.real_data,x = column, ax=ax[i],bins = len(self.real_data[column].unique()), label='Real', stat = 'density', color='c',discrete = False,element = 'step')
                sns.histplot(data = self.synthetic_data,x = column, ax=ax[i], bins = len(self.real_data[column].unique()),label='Synthetic', stat = "density", color='m',discrete = False ,element = 'step')
            else:
                sns.histplot(data = self.real_data,x = column, ax=ax[i], label='Real', fill = False,stat = 'density', color='c',discrete = False,element = 'step')
                sns.histplot(data = self.synthetic_data,x = column, ax=ax[i], label='Synthetic', fill = False,stat = "density", color='m',discrete = False ,element = 'step')

            ax[i].set_title(f'Comparison of {column}')
            ax[i].autoscale_view()
            ax[i].set_xlabel(column)
            ax[i].legend()
        plt.show()

    def compare_correlations(self):
        '''
        Compare correlation matrices
        :return:
        '''
        fig, ax = plt.subplots(figsize=(20, 15))
        diff_corr = abs(self.real_data.corr() - self.synthetic_data.corr())
        mask = np.tril(np.ones_like(diff_corr, dtype=bool))
        sns.heatmap(diff_corr, mask=mask)

    def compare_model_performance(self, fitted_model_real, fitted_model_synth, X_test, y_test):
        """
        Method that computes how close the scores of a model trained on the real vs. synthetic
        data are.
        """
        score_real = fitted_model_real.score(X_test, y_test)
        score_synth = fitted_model_synth.score(X_test, y_test)
        print(f"Score on real dataset: {score_real}\nScore on synthetic dataset: {score_synth}")
        return score_real, score_synth



