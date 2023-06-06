"""
Date: [10/5/2023]

Description:
This module contains a class for generating synthetic data using different architectures. The class takes in a pandas dataframe and generates synthetic data based on the chosen architecture (CTGAN, GaussianCopula, or RealTabFormer). It also includes a method for generating Faker data for sensitive columns.

Dependencies:
- pandas
- numpy
- faker
- random
- collections
- sdv
- torch
- re
- realtabformer

Usage:
Instantiate the Generator class with the required arguments, including the data to be trained on, the chosen architecture, the number of samples to generate, and any categorical or sensitive columns. Call the generate() method to generate synthetic data based on the chosen architecture. Call the faker_categorical() method to generate Faker data for sensitive columns.

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


class Generator:

    def __init__(self, data, architecture, n_samples, num_epochs=None, num_bootstrap=None, categorical_columns=None,
                 sensitive_columns=None, verbose = True):

        """
        :param n_epochs: the number of epochs used for training, default is 200
        :param num_bootstraps: number of bootstraps for the RealTabFormer, default is 500
        :param n_samples: the number of rows to generate
        :param architecture: the chosen architecture, one of ['CTGAN', 'GaussianCopula', 'RealTabFormer']
        :param data: the data that should be trained on, should be in a pandas dataframe
        :param categorical_columns: a list with categorical columns
        :param sensitive_columns: a dict with sensitive columns and what  category they belong to
        :param verbose: should the generate function print out the progress or not (only for CTGAN)
        The categories can be found in the faker_categorical function
        The metadata: an sdv metadata object required to call CTGAN and other methods
        Also required for similarity checks
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs
        else:
            # Default value for RealTabFormer and could be enough for CTGAN
            self.num_epochs = 200
        if num_bootstrap is not None:
            self.num_bootstrap = num_bootstrap
        else:
            # Set to default for RealTabFormer
            self.num_bootstrap = 500
        self.n_samples = n_samples
        if architecture in ['CTGAN', 'GaussianCopula', 'RealTabFormer']:
            self.architecture = architecture
        else:
            print('The requested architecture is not available')
            raise ValueError
        print('Retrieving metadata, check with generator.metadata')
        self.data = data
        self.metadata = self.create_metadata()
        self.categorical_columns = categorical_columns
        self.sensitive_columns = sensitive_columns
        self.verbose = verbose
        
        
    def create_metadata(self):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=self.data)
        return metadata

    def generate(self):
        """
        Based on the chosen architecture, this function returns synthetically generated data

        :return: synthetic data, a pandas dataframe
        """

        #  implementations of these, but I have not been able to import their library

        if self.architecture == "CTGAN":
            model = CTGANSynthesizer(metadata=self.metadata, epochs=self.num_epochs, verbose=self.verbose)
            model.fit(self.data)
            synth_data = model.sample(self.n_samples)

        elif self.architecture == "GaussianCopula":
            model = GaussianCopulaSynthesizer(metadata=self.metadata)
            model.fit(self.data)
            synth_data = model.sample(self.n_samples)
        elif self.architecture == "RealTabFormer":
            model = REaLTabFormer(
                model_type="tabular",
                epochs=self.num_epochs,
                gradient_accumulation_steps=4,
                numeric_max_len = 11,
                # Output log each 100 steps
                logging_steps=100)
            model.fit(self.data, num_bootstrap=self.num_bootstrap)
            synth_data = model.sample(n_samples= self.n_samples)

        return synth_data

    def faker_categorical(self, seed=None):
        """
        Instantiates Faker, generates fake data for it
        WARNING: data generated here should not be used for ML models
        :param seed: int, random seed, defaults
        """
        seed = seed or random.seed()
        # We can initialize the faker with multiple locations: can now draw addresses and names from
        # Germany, US, UK, Spain, France, Italy. Either just a list => all equal weights, or an ordered
        # dictionary in which weights can be specified.
        locale_list = ['de_DE', 'en_US', 'en_GB', 'es_ES', 'fr_FR', 'it_IT']
        fake = Faker(locale_list)
        fake.seed_instance(seed)
        # check whether the ID generated is already in use
        used_ids = set()
        output = []
        for i in range(self.n_samples):
            # select a locale at random => will allow us to generate internally consistent city-country pairs
            # or name/email pairs. Problem is that not all countries might be able to generate all of these
            # attributes. For example Belgium can't create IP-addresses
            locale = np.random.choice(locale_list)
            while True:
                new_id = fake.random_int(min=1, max=self.n_samples)
                if new_id not in used_ids:
                    used_ids.add(new_id)
                    break

            gender = np.random.choice(["Male", "Female"], p=[0.5, 0.5])
            # this works, but it's technical debt, talk to Léo about it
            if gender == "male":
                first_name = fake[locale].first_name_male()
            else:
                first_name = fake[locale].first_name_female()
            last_name = fake[locale].last_name()
            row = {
                "id": new_id,
                "first_name": first_name,
                "last_name": last_name,
                # take everything before @, and replace with first name.lastname
                "email": re.sub(r'^(.*?)@', first_name + "." + last_name + "@", fake[locale].free_email()),
                "gender": gender,
                "ip_address": fake[locale].ipv4_private(),
                "nationality": fake[locale].current_country(),
                "city": fake[locale].city()
            }
            output.append(row)

        df = pd.DataFrame(output, columns=self.sensitive_columns)
        return df
