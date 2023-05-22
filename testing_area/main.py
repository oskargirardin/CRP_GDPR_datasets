"""

"""

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from faker import Faker
import re
from src.similarity_check.SimilarityCheck import *
from src.synthetic_data_generation.generator import *
from src.privacy_check.privacy_check import *


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

    data = pd.read_csv(path_test_data)
    # checking that it can deal with nan values
    data.iloc[3, 2] = float("nan")
    print(data.head())
    # create object
    generator = Generator(num_epochs=50, n_samples=100, architecture='CTGAN',
                          data=data.iloc[:, 2:],
                          categorical_columns=cat_cols,
                          sensitive_columns=sensitive_cols)
    print("Generating data")
    synth_data = generator.generate()
    anonymized_data = generator.faker_categorical()
    df = pd.concat([anonymized_data, synth_data], axis=1)
    df.drop(['CITY', 'STATE'], inplace=True, axis=1)
    #df.to_csv('synth_data.csv')

    similarity_checker = SimilarityCheck(generator.data, synth_data, cat_cols, generator.metadata)
    print(similarity_checker.comparison_columns())
    similarity_checker.visual_comparison_columns()

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
    print(generator.metadata)

    print('Computing the privacy score')
    privacy_check = PrivacyCheck(generator.data, synth_data, generator.metadata.to_dict(), dist_threshold = 0.1)
    privacy_check.generate_privacy_score()
    print(privacy_check.get_privacy_score(k = 10))