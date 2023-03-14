import pandas as pd
import numpy as np
from faker import Faker
import random

from sdv.tabular import CTGAN, GaussianCopula
from sdv.evaluation import evaluate
from table_evaluator import TableEvaluator


class Generator:

    def __init__(self, n_epochs, n_samples,architecture, data, categorical_columns, sensitive_columns):

        """
        :param n_epochs: the number of epochs used for training
        :param n_samples: the number of rows to generate
        :param architecture: the chosen architecture, one of ['CTGAN', 'GaussianCopula']
        :param data: the data that should be trained on, should be in a pandas dataframe
        :param categorical_columns: a list with categorical columns
        :param sensitive_columns: a dict with sensitive columns and what faker category they belong to
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


    def faker_categorical(self, sensitive_col, seed=None, num_points=100):
        """
        Instantiates Faker, generates fake data for it 

        :param sensitive_col: list of str, the column names
        :param seed: int, random seed, defaults
        :param num_points: int, the number of data points to generate. Defaults to 100. 
        """

        # To do find a way to guess the faker data type of new 
        seed=seed or random.seed()
        fake = Faker()
        fake.seed_instance(seed)

        used_ids=set() #check whether the ID generated is already in use
        output = []
        for i in range(num_points):
            while True:
                new_id=fake.random_int(min=1, max=1000)
                if new_id not in used_ids:
                    used_ids.add(new_id)
                    break

            gender = np.random.choice(["Male", "Female"], p=[0.5, 0.5])
            row = { #this works, but it's technical debt, talk to Léo about it 
                "id": new_id,
                "first_name": fake.first_name_male() if gender == "Male" else fake.first_name_female(),
                "last_name": fake.last_name(),
                "email": fake.free_email(),
                "gender": gender,
                "ip_address": fake.ipv4_private(),
                "nationality": fake.country()
            }
            output.append(row)
        
        df = pd.DataFrame(output, columns=sensitive_col)
        return df


############################## Helper functions #####################################################

# data loader + collects key information from the dataset
def get_data(file_path):
    """
    Puts data into a dataframe + 
    """
    data = pd.read_csv(file_path)  
    return data

#define helper function to see your data in terminal 
def print_first_10_rows(file_path):
    """
    Prints the first 10 rows of a csv file

    :param file_path: str, the path of the csv file
    """
    df_test=pd.read_csv(file_path)
    print(df_test.head(10))


############################## Testing area #####################################################
#define path to the data you want to test
path_test_data="/Users/jeannetton/Desktop/DSBA/CRP/fake/CRP_GDPR_datasets/app/data/MOCK_DATA.csv" #change this

#take the comment out to see the first 10 rows of your data
#print_first_10_rows(path_test_data)

# indicate which columns are categorical, and which are sensitive 
cat_col=['id', 'first_name', 'last_name', 'email', 'gender', 'ip_address', 'nationality']
sens_col=['id', 'first_name', 'last_name', 'email', 'gender', 'ip_address', 'nationality']

data = get_data(path_test_data) 

#create object
generator = Generator(n_epochs=100, n_samples=500, architecture='CTGAN',
                      data=data, 
                      categorical_columns=['id, first_name, last_name, email, gender, ip_address, ethnicity'],
                      sensitive_columns={'id': 'random_int', 'first_name': 'first_name','last_name': 'address','email': 'free_email','gender': 'address','ip_address': 'address','Ethnicity': 'address'})

#take the comment out to see the first 10 rows of your data
#print_first_10_rows(path_test_data)

#synth_data = generator.generate()
anonymized_data = generator.faker_categorical(sensitive_col=sens_col, num_points=generator.n_samples)
print(anonymized_data)






