import pandas as pd
import numpy as np
import dtw

class TSSimilarityCheck():

    def __init__(self,df_real, df_synth, metadata):
        """
        :param df_real: the real time series (long format)
        :param df_synth: the synthetic time series generated with PAR (long format)
        :param metadata: the metadata (long format)
        """
        self.df_real = df_real
        self.df_synth = df_synth
        self.metadata = metadata
        self.dist_matrix = None

    def compute_dtw_matrix(self):
        """
        Function that computes the DTW for every combination of original & synthetic sequences.

        --------------

        Returns: DataFrame with distance matrix
        """
        meta_dict = self.metadata.to_dict()
        # Sequence key: what entity does the value belong to
        # Sequence index: the variable that stores the order of values, usually time
        sequence_key, sequence_index = meta_dict["sequence_key"], meta_dict["sequence_index"]
        # Obtain the time series column(s)
        ts_col = list(filter(lambda x: x not in [sequence_key, sequence_index], self.df_synth.columns))[0]
        sequence_key_values_synth = self.df_synth[sequence_key].unique()
        sequence_key_values_original = self.df_real[sequence_key].unique()

        # Initializing the matrix that will store all distances between real and synthetic time series

        ################# real_key 1   real_key 2 ...
        # synthetic key 1
        # synthetic key 2
        # ...

        dist_matrix = pd.DataFrame(np.zeros((len(sequence_key_values_synth), len(sequence_key_values_original))),
                                   columns=sequence_key_values_original,
                                   index=sequence_key_values_synth)

        for i, row in enumerate(dist_matrix.iterrows()):
            synth_key, row = row
            # Retrieve the synthetic time series
            ts_synth = self.df_synth[self.df_synth[sequence_key] == synth_key][ts_col].dropna().values
            if len(ts_synth) == 0:
                # If length is 0, return all nan values
                dist_matrix.iloc[i, :] = np.full(len(sequence_key_values_original), np.nan)
                continue
            for j, original_key in enumerate(sequence_key_values_original):
                # For the synthetic series, loop over all real series to compute the distance
                ts_original = self.df_real[self.df_real[sequence_key] == original_key][ts_col].dropna().values
                if len(ts_original) == 0:
                    # If the real series is of length 0 (all NaN values), distance is NaN
                    dist_matrix.iloc[i, j] = np.nan
                    continue

                # Computing the DTW distance
                alignment = dtw.dtw(ts_synth, ts_original)
                dist = alignment.normalizedDistance
                dist_matrix.iloc[i, j] = dist
        self.dist_matrix = dist_matrix
        return dist_matrix
