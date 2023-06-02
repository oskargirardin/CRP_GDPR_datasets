"""

"""

import sys
import os
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
    # Get the absolute path of the current script file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the data you want to test relative to the script file
    path_test_data = os.path.join(current_dir, "Subsample_training.csv")

    # take the comment out to see the first 10 rows of your data

    # indicate which columns are categorical, and which are sensitive
    cat_cols = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE', 'Risk_Flag']
    sensitive_cols = ["first_name", "last_name", "email", "gender", "ip_address", "nationality", "city"]

    data = pd.read_csv(path_test_data)
    # checking that it can deal with nan values
    data.iloc[3, 2] = float("nan")
    print(data.head())
    # create object
    generator = Generator(num_epochs=100, n_samples=100, architecture='CTGAN',
                          data=data.iloc[:, 2:],
                          categorical_columns=cat_cols,
                          sensitive_columns=sensitive_cols)
    print("Generating data")
    synth_data = generator.generate()
    anonymized_data = generator.faker_categorical()
    df = pd.concat([anonymized_data, synth_data], axis=1)
    df.drop(['CITY', 'STATE'], inplace=True, axis=1)


    similarity_checker = SimilarityCheck(generator.data, synth_data, cat_cols, generator.metadata)
    print(similarity_checker.comparison_columns())
    similarity_checker.visual_comparison_columns()


    print('Computing the privacy score')
    privacy_check = PrivacyCheck(generator.data, synth_data, generator.metadata, dist_threshold = 0.1)
    privacy_check.find_nearest_neighbours()
    print(privacy_check.get_closest_pairs(10, display = True))
