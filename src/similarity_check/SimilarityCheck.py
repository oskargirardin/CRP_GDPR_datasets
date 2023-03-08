import pandas as pd
import torch

from sdv.tabular import CTGAN, GaussianCopula
from sdv.evaluation import evaluate
from table_evaluator import TableEvaluator


class SimilarityCheck:

    def __init__(self, real_data, synthetic_data, cat_cols):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.cat_cols = cat_cols

    def check_similarity(self):
        evaluator = TableEvaluator(real=self.real_data, fake=self.synthetic_data, cat_cols=self.cat_cols)
        evaluator.visual_evaluation()
    
    def compare_model_performance(self, fitted_model_real, fitted_model_synth, X_test, y_test):
        """
        Method that computes how close the scores of a model trained on the real vs. synthetic
        data are.
        """
        score_real = fitted_model_real.score(X_test, y_test)
        score_synth = fitted_model_synth.score(X_test, y_test)
        print(f"Score on real dataset: {score_real}\nScore on synthetic dataset: {score_synth}")
        return score_real, score_synth



