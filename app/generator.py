# This file handles the generation of the synthetic data, it gets the training data from main.py

import main

class gen:
    def __init__(self, architecture, num_Epochs, cat_columns, private_columns):
        self.architecture = architecture
        self.num_Epochs = num_Epochs
        self.train_data = main.csv_data #get the train data from main.py
        self.cat_columns = cat_columns
        self.private_columns = private_columns
