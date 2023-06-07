"""
The main file for the synthetic data generation: it shows the work flow for generating synthetic data, 
comparing it with the original and generating a privacy score. 
"""

import sys
import os
import pandas as pd
import numpy as np
from faker import Faker
import re
sys.path.append('..')
from similarity_check.SimilarityCheck import *
from synthetic_data_generation.generator import *
from privacy_check.privacy_check import *


if __name__ == "__main__":

    ##############################
    # Reading in the data from the data folder
    ##############################

    # Get the absolute path of the current script file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate two levels up from the current directory
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)

    # Define the path to the data folder relative to the grandparent directory
    data_folder_path = os.path.join(grandparent_dir, "data")

    # Define the path to the data you want to test relative to the data folder
    path_test_data = os.path.join(data_folder_path, "Subsample_training.csv")

    ###################
    # Read data and define categorical columns
    ###################

    data = pd.read_csv(path_test_data)

    # indicate which columns are categorical, and which are sensitive
    # The categorical columns will be visualized in a different way when checking the similarity
    cat_cols = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE', 'Risk_Flag']

    
    # Insert a nan value to check that the generator can deal with that
    data.iloc[3, 2] = float("nan")
    print(data.head())

    #################
    # Generate synthetic data
    #################

    # create object
    generator = Generator(num_epochs=100, n_samples=100, architecture='CTGAN',
                          data=data.iloc[:, 2:],
                          categorical_columns=cat_cols)
    print("Generating data")
    synth_data = generator.generate()
    

    #################
    # Check the similarity of the data
    #################
    similarity_checker = SimilarityCheck(generator.data, synth_data, cat_cols, generator.metadata)
    print(similarity_checker.comparison_columns())
    similarity_checker.visual_comparison_columns()


    #################
    # Compute privacy scores for the data
    #################

    print('Computing the privacy score')
    privacy_check = PrivacyCheck(generator.data, synth_data, generator.metadata, dist_threshold = 0.1)
    privacy_check.find_nearest_neighbours()
    print(privacy_check.get_closest_pairs(10, display = True))
