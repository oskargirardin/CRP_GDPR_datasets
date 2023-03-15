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

class Generator:

    def __init__(self, data, architecture, n_samples,n_epochs = None, categorical_columns = None, sensitive_columns = None):

        """
        :param n_epochs: the number of epochs used for training
        :param n_samples: the number of rows to generate
        :param architecture: the chosen architecture, one of ['CTGAN', 'GaussianCopula']
        :param data: the data that should be trained on, should be in a pandas dataframe
        :param categorical_columns: a list with categorical columns
        :param sensitive_columns: a dict with sensitive columns and what  category they belong to
        The categories can be found in the faker_categorical function
        """

        self.n_epochs = n_epochs
        self.n_samples = n_samples
        if architecture in ['CTGAN', 'GaussianCopula']:
            self.architecture = architecture
        else:
            print('The requested architecture is not available')
            raise ValueError

        self.data = data
        self.categorical_columns = categorical_columns
        self.sensitive_columns = sensitive_columns

    def generate(self):
        """
        Based on the chosen architecture, this function returns synthetically generated data
        :return: synthetic data, a pandas dataframe
        """

        # TODO: Add more generators, especially PATEGAN or other differentially private ones synthcity seems to have
        #  implementations of these, but I have not been able to import their library

        if self.architecture == "CTGAN":
            model = CTGAN(epochs=self.n_epochs)
            model.fit(self.data)
            synth_data = model.sample(self.n_samples)

        elif self.architecture == "GaussianCopula":
            model = GaussianCopula()
            model.fit(self.data)
            synth_data = model.sample(self.n_samples)

        return synth_data

    def faker_categorical(self, seed=None):
        """
        Instantiates Faker, generates fake data for it
        WARNING: data generated here should not be used for ML models
        :param seed: int, random seed, defaults
        """

        # TODO: Find a way to make this generalizable, f.e. create many attributes, and return the ones asked

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


##############################
# Testing area
##############################

# define path to the data you want to test
path_test_data="./Subsample_training.csv"

# take the comment out to see the first 10 rows of your data

# indicate which columns are categorical, and which are sensitive 
cat_cols = ['Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', 'STATE', 'Risk_Flag']
sensitive_cols = ["first_name", "last_name","email", "gender", "ip_address", "nationality","city"]
data = get_data(path_test_data)
# checking that it can deal with nan values
data.iloc[3,2] = float("nan")
print(data.head())
# create object
generator = Generator(n_epochs=300, n_samples=100, architecture='CTGAN',
                      data=data,
                      categorical_columns=cat_cols,
                      sensitive_columns= sensitive_cols)

synth_data = generator.generate().iloc[:,2:]
anonymized_data = generator.faker_categorical()
df = pd.concat([anonymized_data, synth_data], axis=1)
print(df.columns)
df.drop(['CITY', 'STATE'], inplace=True, axis=1)
df.to_csv('synth_data.csv')


