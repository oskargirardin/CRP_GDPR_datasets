from sdmetrics.reports.single_table import DiagnosticReport
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tqdm


class PrivacyCheck(DiagnosticReport):
    """
    Class to generate a report on the synthetic data, that checks 
    how similar it is to the original one, with respect to privacy
    concerns. 
    """
    def __init__(self, original_data, synthetic_data, metadata, dist_threshold=0.2, only_cat=False):
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.metadata = metadata
        self.dist_threshold = dist_threshold
        self.only_cat = only_cat
        super().__init__()
        

    def generate_report(self, verbose = True):
        """
        Generates the privacy report using the NewRowSynthesis metric.

        :param original_data: original pandas dataframe
        :param synthetic_data: synthetic pandas dataframe
        :param metadata: dictionary of the column types
        :param verbose: controls verbosity of report generation
        """
        self.generate(self.original_data, self.synthetic_data, self.metadata, verbose)
    

    def get_visualization(self, property_name):
        """
        Generates a visualization of some properties.

        :param property_name: A string with the name of the property. One of: 'Synthesis', 'Coverage' or 'Boundaries'.
        """
        return super().get_visualization(property_name)
    
    def get_results(self):
        """
        Get results of the report.
        """
        return super().get_results()
    
    def get_details(self, property_name):
        """
        Get additional details about properties of the sythetic data.

        :param property_name: A string with the name of the property. One of: 'Synthesis', 'Coverage' or 'Boundaries'.
        """
        return super().get_details(property_name)
    
    def get_properties(self):
        """
        Returns a dictionary with scores of properties.
        """
        return super().get_properties()
    

    def find_nearest_neighbours(self, verbose = True):
        """
        Function to generate privacy score based on nearest neighbours
        """
        pairs = self._find_nn(verbose=verbose)
        ##self.score = score
        self.pairs = pairs




    def _find_nn(self, verbose = True):
        """
        Function that computes the privacy score of the synthetic data

        :param only_cat: boolean that indicates if we should only consider categorical columns to compute the distance

        return: list of nearest neighbours of all sythentic rows
        """
        # Initialization
        nneighbour_idx = []
        n_samples = len(self.synthetic_data)
        df_real, df_synth = self.original_data.copy(), self.synthetic_data.copy()

        # Separate dataframes into numerical and categorical
        dtypes = {col: col_type["type"] for col, col_type in self.metadata["fields"].items()}
        numeric_cols = [col for col, type in dtypes.items() if type == "numerical"]
        cat_cols = [col for col, type in dtypes.items() if type != "numerical"]
        df_real_num = df_real[numeric_cols]
        df_synth_num = df_synth[numeric_cols]
        df_real_cat = df_real[cat_cols]
        df_synth_cat = df_synth[cat_cols]
        
        # Normalize numeric columns -> equal weights to each col
        scaler = StandardScaler()
        df_real_num = pd.DataFrame(scaler.fit_transform(df_real_num))
        df_synth_num = pd.DataFrame(scaler.transform(df_synth_num))
        
        # Loop over rows in synthetic dataset
        for idx_synth in tqdm.tqdm(range(n_samples), desc='Computing privacy score', disable=(not verbose)):
            # Get numerical elements and categorical elements
            row_num = df_synth_num.iloc[idx_synth]
            row_cat = df_synth_cat.iloc[idx_synth]
            # Compute distance to every real row 
            # TODO: Give more importance to dists_cat, because dists_num > dists_cat in general
            dists_num = np.sum(np.abs(df_real_num - row_num), axis = 1)
            dists_cat = np.sum(df_real_cat != row_cat, axis = 1)
            dists = (dists_num + dists_cat) / (len(numeric_cols) + len(cat_cols))
            # Find minimal distance and append neighbour index
            min_dist = dists.min()
            idx_neighbour = dists.argmin()
            nneighbour_idx += [(idx_synth, idx_neighbour, min_dist)]

        # OUTDATED: 
        # Score is determined only by the closest distance among all nearest neighbours
        # For privacy, we need every synthetic row to be sufficiently different from every real row.
        #score = 1 - min(dists_nn)


        return nneighbour_idx

    def display_k_closest_pairs(self, k):
        """
        Displays the k closest pairs

        :param k (int): number of closest pairs to display 
        """
        pairs_sorted = sorted(self.pairs, key = lambda x: x[2])
        k_closest_pairs = pairs_sorted[:k]
        for i in range(k):
            print(f"############ TOP {k} CLOSEST PAIRS ############")
            idx_synth, idx_neighbour, dist = k_closest_pairs[i]
            df = pd.concat([self.synthetic_data.iloc[idx_synth], self.original_data.iloc[idx_neighbour], ], axis = 1)
            df.columns = [f"Synthetic obs. (idx: {idx_synth})", f"Closest real obs. (idx: {idx_neighbour})"]
            print(f"{i+1}. Closest pair with distance: {dist: .4f}")
            print(df)
