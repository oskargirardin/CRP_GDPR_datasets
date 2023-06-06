from sdv.datasets.local import load_csvs
from data_processing import DataProcessor
from ts_generator import TSGenerator
import os

if __name__ == "__main__":

    ################
    # Define the path to the data
    ################

    # Get the absolute path of the current script file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate two levels up from the current directory
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    # Define the path to the data folder relative to the grandparent directory
    data_folder_path = os.path.join(grandparent_dir, "data")

    #################
    # Read in the data
    #################

    datasets = load_csvs(folder_name=data_folder_path)

    df = datasets["energy_dataset"]

    #################
    # Process the data so that it becomes usable for the PARSynthesizer
    #################

    data_processor = DataProcessor(df, obs_limit=1000, interpolate=True)
    # the name of the time column is time in this df and all columns can be used as identifiers in long format
    data_processor.convert_to_long_format(time_columns='time', verbose=True)
    data_processor.get_metadata_long_df(identifier='variable', time_column='time', datetime_format='%Y-%m-%d %H:%M:%S')
    #print(data_processor.metadata)

    ##################
    # Generate time series data
    ##################

    generator = TSGenerator(data_processor.df_long, data_processor.metadata, verbose=True)
    generator.train(n_epochs=2)

    synthetic_df = generator.sample(n_samples=1)

    print(synthetic_df.head())
