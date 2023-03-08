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



