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

from generator import Generator




def generator_augmentation(data, architecture, p_fraud, 
                           n_epochs_nofraud, n_epochs_fraud, n_boostrap, 
                           target_col, sensitive_columns = None):
    '''

    Parameters
    ----------
    data : pandas dataframe
        Dataframe to be synthesized and augmented.
    architecture : str
        The chosen architecture, one of ['CTGAN', 'GaussianCopula', 'RealTabFormer'].
    p_fraud: float
        size of the minority class (to augment) relative to the size of the majority class (percentage)
    n_epochs_nofraud : int
        the number of epochs used for training of the non fraud subsample.  
    n_epochs_fraud : int
        the number of epochs used for training of the fraud subsample
    n_boostrap : int
        number of bootstraps for the RealTabFormer, default is 500.
    target_col : string
        name of the target column (assumin fraud = 1, no fraud = 0).
    sensitive_columns : dict
        a dict with sensitive columns and what  category they belong to.


    Returns
    -------
    metadata: 
        dataframe's metadata
    synth : pandas dataframe
        dataset that has been synthetized and augmented.

    '''
    # extracting the categorical columns of the dataset
    categorical_columns = []

    for col in list(data):
        if data[col].dtypes == 'object':
            categorical_columns.append(col)
    
    #minority class subset
    fraud_data = data[data[target_col] == 1]
    #majority class subset
    nofraud_data = data[data[target_col] == 0]
    
    #Computing the desired size of the minority class for augmentation
    n_samples_no_fraud = len(nofraud_data)
    n_samples_fraud = round(n_samples_no_fraud*(p_fraud/(1 - p_fraud)))
    
    nofraud_generator = Generator(num_epochs=n_epochs_nofraud,
                                  n_samples=n_samples_no_fraud,
                                  num_bootstrap = n_boostrap ,
                                  architecture= architecture,
                                  data=nofraud_data,
                                  categorical_columns=categorical_columns,
                                  sensitive_columns=sensitive_columns)
    
    fraud_generator = Generator(num_epochs=n_epochs_fraud,
                                  n_samples=n_samples_fraud,
                                  num_bootstrap = n_boostrap ,
                                  architecture= architecture,
                                  data=fraud_data,
                                  categorical_columns=categorical_columns,
                                  sensitive_columns=sensitive_columns)
    
    #generating the non fraudulent data
    synth_nofraud = nofraud_generator.generate()
    #generating the fraudulent data
    synth_fraud = fraud_generator.generate()
    
    #Merging the two datasets and shuffling them
    synth = pd.concat([synth_nofraud,synth_fraud],ignore_index=True, axis = 0)
    synth = synth.sample(frac=1).reset_index(drop=True)
    
    metadata = nofraud_generator.metadata

    return metadata, synth


