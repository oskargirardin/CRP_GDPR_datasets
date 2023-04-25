from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sdv.evaluation.single_table import evaluate_quality

# TODO: ML model performance


class SimilarityCheck:

    '''
    Check the quality of the synthetic data both visually and with metrics
    '''

    def __init__(self, real_data, synthetic_data, cat_cols, metadata):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        if cat_cols is not None:
          self.cat_cols = cat_cols
        else:
          self.cat_cols = []
        self.metadata = metadata


    def comparison_columns(self):
        '''
        :return: the KL divergence for numerical variables...
        tested and works
        '''
        quality_report = evaluate_quality(
          real_data=self.real_data,
          synthetic_data=self.synthetic_data,
          metadata=self.metadata)
        print(quality_report)
        print(quality_report.get_details(property_name='Column Shapes'))


    def visual_comparison_columns(self):
        '''
        Plot data in one of three ways:
        1) numeric columns are plotted using the densities
        2) categorical columns with limited categories are plotted with bar plots
        3) categorical columns with many categories do not show the values, if performance looks problematic
        we encourage further exploration of these columns
        '''
        if (self.real_data.columns != self.synthetic_data.columns).all():
            print('Columns in real and synthetic data not the same!')
            return

        fig, ax = plt.subplots(nrows=len(self.real_data.columns), figsize=(10, len(self.real_data.columns)*4))
        fig.tight_layout(pad = 5)
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
                ax[i].set_xticklabels([])
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
        fig, ax = plt.subplots(figsize=(len(self.real_data.columns), (len(self.real_data.columns))))
        diff_corr = abs(self.real_data.corr() - self.synthetic_data.corr())
        mask = np.tril(np.ones_like(diff_corr, dtype=bool))
        sns.heatmap(diff_corr, mask=mask)
        ax.set_title('Differences in correlation values real and synthetic data')

    def compare_model_performance(self, fitted_model_real, fitted_model_synth, X_test, y_test):
        """
        Method that computes how close the scores of a model trained on the real vs. synthetic
        data are.
        """
        score_real = fitted_model_real.score(X_test, y_test)
        score_synth = fitted_model_synth.score(X_test, y_test)
        print(f"Score on real dataset: {score_real}\nScore on synthetic dataset: {score_synth}")
        return score_real, score_synth