import pandas as pd
from pathlib import Path

import sdv.metadata.errors
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from sdv.metadata import SingleTableMetadata
import matplotlib.dates as md
from sdv.datasets.local import load_csvs


class DataProcessor():

    """
    This class contains the functionality to process the data in such a way that it can be fed into
    the synthetic data generation models.

    Usually, the data will be in 'wide format', so that for each timestamp, there are several columns
    for each sequence: for example [fossiul_fuel_demand, wind_offshore_demand,...]. If your data
    is in this format, it will have to be converted into long format for the PARSynthesizer

    For the PARSynthesizer, the data has to be in 'long' format, so that it has three columns:

    ['Time', 'Variable', 'Value'], in which the time shows the progression of time for the time series
    and in which the Variable is an identifier for the sequence. For example, the Time column can be from
    [t_0 until t_n] for each identifier in the variable column: [fossil_fuel_demand, wind_offshore_demand,...]

    For the DGAN, the data does not have to be in 'long' format, but each individual sequence should be cut up into
    several sequences.
    """

    def __init__(self, df, metadata = None, obs_limit = 1000, interpolate = True, drop_na_cols = True, long = False):
        """
        :param df: the dataframe of time series
        :param metadata: the metadata in 'SingleTableMetadata' format from sdv
        :param obs_limit: the number of observations per sequence
        :param interpolate: interpolating nan values
        :param drop_na_cols: dropping nan values
        :param long: is the df already in long format
        """
        self.obs_limit = obs_limit
        self.interpolate = interpolate
        self.drop_na_cols = drop_na_cols
        self.long = long
        if long:
            # If the dataframe is already in long format, the df_long is the df
            # taking a copy to avoid spillover modifications
            self.df_long = df.copy()
        else:
            # If the dataframe is in wide format, subset it so that it contains at most obs_limit
            # observations for each sequence
            self.df = df.iloc[:obs_limit]
            # How should NaN values be treated? Interpolated or removed?
            if interpolate:
                self.df = self.df.interpolate()
            elif drop_na_cols:
                self.df = self.df.dropna(axis=1, how="all")

        self.metadata = metadata

    def convert_to_long_format(self, time_columns, desired_identifiers=None, verbose = False):
        """
        This function will convert the dataframe into long format if it is not yet the case
        It is similar to the pd.melt() functionality

        :param time_columns: which columns order the observations?
        :param desired_identifiers: a list of the columns you want to include as identifiers, and on which the model
        should be trained. If None, all columns will become an identifier in long format
        :param verbose: if True, print the dataframe
        """
        if self.long:
            print('The DataProcessor was initialized on a long df, if this was not the case, reinitialize the df')
            return

        if desired_identifiers is not None:
            self.df_long = self.df.melt([time_columns], desired_identifiers)
        else:
            self.df_long = self.df.melt([time_columns])

        if verbose:
            print(self.df_long.head())
        
        return self.df_long

    def get_metadata_long_df(self, identifier, time_column, datetime_format=None):
        """
        Obtains the metadata from the df_long, the metadata can be accessed through the metadata attribute,
        it is of type SingleTableMetadata, but can be converted into a dict by calling to_dict()

        :param identifier: the sequence identifier, the columns in wide format (in long format, the Variable column)
        :param time_column: orders the observations for each sequence, should be a numeric or a datetime format
        :param datetime_format: in what format is the date? For example '%Y-%m-%d %H:%M:%S'.
        """
        metadata = SingleTableMetadata()
        if datetime_format:
            # If a datetime format is specified, convert the time column into datetime
            # If it is not specified, it should be a numeric
            self.df_long[time_column] = pd.to_datetime(self.df_long[time_column], utc=True).dt.tz_localize(None)

        metadata.detect_from_dataframe(self.df_long)
        metadata.update_column(
            identifier,
            sdtype='id'
        )
        metadata.set_sequence_key(identifier)

        if datetime_format:
            # if a datetime format was specified, we convert the column into that format
            metadata.update_column(
                column_name=time_column,
                sdtype='datetime',
                datetime_format=datetime_format)
        try:
            metadata.set_sequence_index(column_name=time_column)
        except sdv.metadata.errors.InvalidMetadataError:
            print('Could not set sequence index! Try again, PARSynthesizer will not work.')
            print('The time column must be a numeric (0,...,n) or of the datetime format specified as parameter')
        self.metadata = metadata

        return metadata

    def get_df_long(self):
        return self.df_long
    
    def get_metadata(self):
        if self.metadata is None:
            print("Warning: self.metadata is None. Run get_metadata_long_df first.")
        return self.metadata
