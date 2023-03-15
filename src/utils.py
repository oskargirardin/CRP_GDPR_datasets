import pandas as pd



def get_data(file_path):
    """
    Puts data into a dataframe +
    """
    data = pd.read_csv(file_path)
    return data

def print_first_10_rows(file_path):
    """
    Prints the first 10 rows of a csv file

    :param file_path: str, the path of the csv file
    """
    df_test=pd.read_csv(file_path)
    print(df_test.head(10))