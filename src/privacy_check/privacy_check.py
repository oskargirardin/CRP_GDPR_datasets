from sdmetrics.reports.single_table import DiagnosticReport
from IPython.display import display, Markdown
from sklearn.preprocessing import StandardScaler
import scipy
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
    

    def generate_privacy_score(self, verbose = True):
        """
        Function to generate privacy score based on nearest neighbours
        """
        score, pairs = self._nn_privacy_score(only_cat=self.only_cat, verbose=verbose)
        self.score = score
        self.pairs = pairs

    def get_privacy_score(self, verbose = True, k = 3):
        """
        Get the privacy score that was computed previously.
        """
        if verbose:
            print("#################### SCORE #####################")
            print(f"Privacy score: {self.score*100: .2f}%")
            print(f"############ TOP {k} CLOSEST PAIRS ############")
            self._display_k_closest_pairs(k)
        return self.score, self.pairs
    

    def _dist_metric(arr1, arr2, dtypes, only_cat = False):
        """
        Function that computes a distance between two arrays with data types dtypes

        :param arr1: first array (pd.Series with column names)
        :param arr2: second array (pd.Series with column names)
        :param dtypes: dictionary with column names as keys and type as value
        :param only_cat: boolean that indicates if we should only consider categorical columns to compute the distance

        return: distance (float)
        """
        assert len(arr1) == len(arr2), "Arrays not the same length"
        n = len(arr1)
        dist = 0
        n_num_cols = 0
        col_names = arr1.index
        for idx in range(n):
            if dtypes[col_names[idx]] == "numerical":
                # If numeric: Computes the area between the two points -> min = 0, max = 1
                n_num_cols += 1
                dist += abs(scipy.stats.norm.cdf(arr1[idx])-scipy.stats.norm.cdf(arr2[idx])) if not only_cat else 0
            else:
                # If string: Checks if they're the same or not -> min = 0, max = 1
                dist += int(arr1[idx] != arr2[idx])
        return dist/n if not only_cat else dist/(n-n_num_cols)
    
    def _dist_metric_num(arr1, arr2, only_cat = False):
        """
        Function that computes a distance between two arrays with data types dtypes

        :param arr1: first array (pd.Series with column names)
        :param arr2: second array (pd.Series with column names)
        :param only_cat: boolean that indicates if we should only consider categorical columns to compute the distance

        return: distance (float)
        """
        #assert len(arr1) == len(arr2), "Arrays not the same length"
        
        # Computes the area between the two points -> min = 0, max = 1
        el_dist = abs(scipy.stats.norm.cdf(arr1)-scipy.stats.norm.cdf(arr2))

        return np.sum(el_dist)
    
    def _dist_metric_cat(arr1, arr2, only_cat = False):
        """
        Function that computes a distance between two arrays with data types dtypes

        :param arr1: first array (pd.Series with column names)
        :param arr2: second array (pd.Series with column names)
        :param only_cat: boolean that indicates if we should only consider categorical columns to compute the distance

        return: distance (float)
        """
        #assert len(arr1) == len(arr2), "Arrays not the same length"
        
        # Computes the area between the two points -> min = 0, max = 1
        identical_els = (arr1 == arr2)

        return np.sum(identical_els)



    def _nn_privacy_score(self, only_cat = False, verbose = True):
        """
        Function that computes the privacy score of the synthetic data

        :param only_cat: boolean that indicates if we should only consider categorical columns to compute the distance

        return: score (float), list of nearest neighbours of all sythentic rows
        """
        # Initialization
        nneighbour_idx = []
        dists_nn = []
        n_samples = len(self.synthetic_data)
        df_real, df_synth = self.original_data.copy(), self.synthetic_data.copy()
        # Separate dataframes into numerical and categorical
        dtypes = {col: col_type["sdtype"] for col, col_type in self.metadata["columns"].items()}
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
            dists_num = df_real_num.apply(lambda x: PrivacyCheck._dist_metric_num(row_num, x, only_cat), axis = 1)
            dists_cat = df_real_cat.apply(lambda x: PrivacyCheck._dist_metric_cat(row_cat, x, only_cat), axis = 1)
            dists = dists_num + dists_cat
            # Find minimal distance and append neighbour index
            min_dist = dists.min()
            dists_nn.append(min_dist)
            idx_neighbour = dists.argmin()
            nneighbour_idx += [(idx_synth, idx_neighbour, min_dist)]

        # Score is determined only by the closest distance among all nearest neighbours
        # For privacy, we need every synthetic row to be sufficiently different from every real row.
        score = 1 - min(dists_nn)
        return score, nneighbour_idx

    def _display_k_closest_pairs(self, k):
        """
        Displays the k closest pairs

        :param k (int): number of closest pairs to display 
        """
        pairs_sorted = sorted(self.pairs, key = lambda x: x[2])
        k_closest_pairs = pairs_sorted[:k]
        for i in range(k):
            idx_synth, idx_neighbour, dist = k_closest_pairs[i]
            df = pd.concat([self.synthetic_data.iloc[idx_synth], self.original_data.iloc[idx_neighbour]], axis = 1)
            df.columns = [f"Synthetic obs. (idx: {idx_synth})", f"Closest real obs. (idx: {idx_neighbour})"]
            print(f"{i+1}. Closest pair with distance: {dist: .4f}")
            try:
                display(Markdown(df.to_markdown()))
            except ImportError:
                print(df.head(k))
