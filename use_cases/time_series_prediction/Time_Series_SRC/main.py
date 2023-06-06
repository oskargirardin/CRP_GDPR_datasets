from sdv.datasets.local import load_csvs
from data_processing import DataProcessor
from ts_generator import TSGenerator

if __name__ == "__main__":
    datasets = load_csvs(folder_name='../data/')
    df = datasets["energy_dataset"]

    data_processor = DataProcessor(df, obs_limit=1000, interpolate=True)
    # the name of the time column is time in this df and all columns can be used as identifiers in long format
    data_processor.convert_to_long_format(time_columns='time', verbose=True)
    data_processor.get_metadata_long_df(identifier='variable', time_column='time', datetime_format='%Y-%m-%d %H:%M:%S')
    #print(data_processor.metadata)

    generator = TSGenerator(data_processor.df_long, data_processor.metadata, verbose=True)
    generator.train(n_epochs=2)

    synthetic_df = generator.sample(n_samples=1)

    print(synthetic_df.head())
