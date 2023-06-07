
"""
Created on Mon May 22 14:59 2023

@author: marinaplt
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from collections import OrderedDict
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
import torch
import re
from sdv.metadata import SingleTableMetadata
from realtabformer import REaLTabFormer

from SingleTableGeneration.synthetic_data_generation.generator import Generator

def augmentation(data, target_col, architecture, p_fraud,  
                           n_epochs, n_boostrap = 500):
    '''

    Parameters
    ----------
    data : pandas dataframe
        Dataframe to be synthesized and augmented.
    target_col : string
        name of the target column.
    architecture : str
        The chosen architecture, one of ['CTGAN', 'GaussianCopula', 'RealTabFormer'].
    p_fraud: float
        size of the minority class (to augment) relative to the size of the majority class (percentage)
    n_epochs : int
        the number of epochs used for training of the fraud subsample.  
    n_boostrap : int
        number of bootstraps for the RealTabFormer, default is 500.


    Returns
    -------
    metadata: 
        dataframe's metadata
    synth : pandas dataframe
        dataset that has been synthetized and augmented.

    '''
    categorical_columns = []

    for col in list(data):
        if data[col].dtypes == 'object':
            categorical_columns.append(col)
    
    fraud_data = data[data[target_col] == 1]
    

    #this line computes the number of samples to generate to have the desired class probability
    n_samples_fraud = round(len(data)*(p_fraud)/(1-p_fraud) - len(fraud_data))
    
    # creating the data generator for fraud
    fraud_generator = Generator(num_epochs = n_epochs,
                                  n_samples = n_samples_fraud,
                                  num_bootstrap = n_boostrap ,
                                  architecture = architecture,
                                  data = fraud_data,
                                  categorical_columns = categorical_columns,
                                  )
    

    # generating the samples for the fraud data
    synth_fraud = fraud_generator.generate()

    
    # stick the new fraud data to the original data (and shuffle the data): dataset is augmented!
    augmented = pd.concat([data,synth_fraud],ignore_index=True, axis = 0)
    augmented = augmented.sample(frac=1).reset_index(drop=True)
    print("------Dataset has been successfully augmented!------")
    
    metadata = fraud_generator.metadata

    return metadata, augmented
