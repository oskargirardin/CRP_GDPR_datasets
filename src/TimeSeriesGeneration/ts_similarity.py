import pandas as pd
import numpy as np
import dtw
import matplotlib.pyplot as plt
import math

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

    def compute_distance_matrix(self):
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


    def get_mean_nn_distances(self):
        # Compute distance matrix
        if self.dist_matrix is None:
            self.compute_dtw_matrix()

        # Find the nearest neighbour's names
        nearest_synth_ts = self.dist_matrix.idxmin(axis=0)

        # Init distance_list
        nn_distances = []

        for real_col, synth_name in nearest_synth_ts.items():
            # Skip NAs
            if pd.isna(synth_name):
                continue
            # Get DTW distance of pair
            nn_distance = self.dist_matrix[real_col][synth_name]
            nn_distances.append(nn_distance)

        return np.mean(nn_distances)



    def plot_nearest_neighbours(self, sequence_column = "variable", value_column = "value", time_column = "time", **fig_kw):
        """
        Function that plots the nearest (synthetic) time series for every real time series.

        :param sequence_column: column that identifies different sequences
        :param value_column: column that contains the values of the time series
        :param time_column: column that identifies the time point
        """
        # Compute distance matrix
        if self.dist_matrix is None:
            self.compute_distance_matrix()

        # Find the nearest neighbour's names
        nearest_synth_ts = self.dist_matrix.idxmin(axis=0)

        # Init plotting
        n_plots = len(self.df_real[sequence_column].unique())
        fig, axs = plt.subplots(nrows=math.ceil(n_plots/2), ncols=2, **fig_kw) 
        axs = axs.reshape(-1)

        for i, (real_col, synth_name) in enumerate(nearest_synth_ts.items()):

            if pd.isna(synth_name):
                axs[i].set_title(f"NaN: {real_col}")
                axs[i].axis("off")
                continue

            # Get DTW distance of pair
            nn_distance = self.dist_matrix[real_col][synth_name]

            # Subset real dataset and extract y, x for plotting
            real_df_subset = self.df_real[self.df_real[sequence_column] == real_col]
            y_real = real_df_subset[value_column]
            x_real = real_df_subset[time_column]

            # Subset synthetic dataset and extract y, x for plotting
            synth_df_subset = self.df_synth[self.df_synth[sequence_column] == synth_name]
            y_synth = synth_df_subset[value_column]
            x_synth = synth_df_subset[time_column]

            axs[i].plot(x_real, y_real, label = f"Real ({real_col})")
            axs[i].plot(x_synth, y_synth, label = f"Synthetic ({synth_name})")
            axs[i].tick_params(axis='x', which='both', bottom=False,top=False,labelbottom=False)
            axs[i].set_title(f"Nearest neighbour: {real_col} (DTW-distance: {nn_distance: .2f})")
            axs[i].legend()
        
        
        # If the number of plots is odd, make last plot empty
        if n_plots % 2 == 1:
            axs[-1].axis("off")

        plt.show()