import pandas as pd
import numpy as np
from faker import Faker
import random
from collections import OrderedDict
from sdv.tabular import CTGAN, GaussianCopula
from sdv.evaluation import evaluate
from table_evaluator import TableEvaluator
from src.utils import *
import re
from similarity_check.SimilarityCheck import *
from synthetic_data_generation.generator import *

if __name__ == "__main__":

    ##############################
    # Testing area
    ##############################

    # define path to the data you want to test
    path_test_data = "./Subsample_training.csv"

    # take the comment out to see the first 10 rows of your data

    # indicate which columns are categorical, and which are sensitive
    cat_cols = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE', 'Risk_Flag']
    sensitive_cols = ["first_name", "last_name", "email", "gender", "ip_address", "nationality", "city"]

    my_metadata = {
        'fields':
            {
                'Income': {'type': 'numerical', 'subtype': 'integer'},
                'Age': {'type': 'numerical', 'subtype': 'integer'},
                'Experience': {'type': 'numerical', 'subtype': 'integer'},
                'CURRENT_JOB_YRS': {'type': 'numerical', 'subtype': 'integer'},
                'CURRENT_HOUSE_YRS': {'type': 'numerical', 'subtype': 'integer'},
                'Married/Single': {'type': 'categorical'},
                'House_Ownership': {'type': 'categorical'},
                'Car_Ownership': {'type': 'categorical'},
                'Profession': {'type': 'categorical'},
                'CITY': {'type': 'categorical'},
                'STATE': {'type': 'categorical'},
                'Risk_Flag': {'type': 'boolean'}
            },
        'constraints': [],
        'model_kwargs': {},
        'name': None,
        'primary_key': None,
        'sequence_index': None,
        'entity_columns': [],
        'context_columns': []
    }

    data = get_data(path_test_data)
    # checking that it can deal with nan values
    data.iloc[3, 2] = float("nan")
    print(data.head())
    # create object
    generator = Generator(n_epochs=1, n_samples=100, architecture='CTGAN',
                          data=data,
                          categorical_columns=cat_cols,
                          sensitive_columns=sensitive_cols)
    print("Generating data")
    synth_data = generator.generate().iloc[:, 2:]
    anonymized_data = generator.faker_categorical()
    df = pd.concat([anonymized_data, synth_data], axis=1)
    print(df.columns)
    df.drop(['CITY', 'STATE'], inplace=True, axis=1)
    print(df.head())
    #df.to_csv('synth_data.csv')

    similarity_checker = SimilarityCheck(data.iloc[:, 2:], synth_data, cat_cols, my_metadata)
