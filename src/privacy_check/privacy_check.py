# Import libraries
from sdmetrics.reports.single_table import DiagnosticReport
from IPython.display import display, Markdown
from sklearn.preprocessing import StandardScaler
import scipy
import pandas as pd
import numpy as np

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
    

    def generate_privacy_score(self):
        print("Generating report...", end = "")
        score, pairs = self._nn_privacy_score(dist_threshold=self.dist_threshold, only_cat=self.only_cat)
        print("Done!")
        self.score = score
        self.pairs = pairs

    def get_privacy_score(self, verbose = True, k = 3):
        if verbose:
            critical_pairs = list(filter(lambda x: x[2] < self.dist_threshold, self.pairs))
            n_critical_pairs = len(critical_pairs)
            print("############ SCORE ############")
            print(f"Privacy score: {self.score*100: .2f}%")
            #print(f"Number of critical pairs with threshold {self.dist_threshold}: {n_critical_pairs}/{len(self.synthetic_data)}")
            print(f"############ TOP {k} CLOSEST PAIRS ############")
            self._display_k_closest_pairs(k)
        return self.score, self.pairs
    

    def _dist_metric(arr1, arr2, dtypes, only_cat = False):
        """
        Function that computes a distance between two arrays with data types dtypes
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


    def _nn_privacy_score(self, dist_threshold = 0.1, only_cat = False):
        self.dist_threshold = dist_threshold
        nneighbour_idx = []
        dists_nn = []
        n_samples = len(self.synthetic_data)
        df_r, df_s = self.original_data.copy(), self.synthetic_data.copy()
        dtypes = {col: col_type["type"] for col, col_type in self.metadata["fields"].items()}
        numeric_cols = [col for col, type in dtypes.items() if type == "numerical"]
        scaler = StandardScaler()
        df_r = df_r
        df_r[numeric_cols] = scaler.fit_transform(df_r[numeric_cols])
        df_s[numeric_cols] = scaler.transform(df_s[numeric_cols])
        for idx_synth in range(n_samples):
            row = df_s.iloc[idx_synth]
            dists = df_r.apply(lambda x: PrivacyCheck._dist_metric(row, x, dtypes, only_cat), axis = 1)
            min_dist = dists.min()
            dists_nn.append(min_dist)
            idx_neighbour = dists.argmin()
            nneighbour_idx += [(idx_synth, idx_neighbour, min_dist)]
       
        return 1-min(dists_nn), nneighbour_idx

    def _display_k_closest_pairs(self, k):
        pairs_sorted = sorted(self.pairs, key = lambda x: x[2])
        k_closest_pairs = pairs_sorted[:k]
        for i in range(k):
            idx_synth, idx_neighbour, dist = k_closest_pairs[i]
            df = pd.concat([self.synthetic_data.iloc[idx_synth], self.original_data.iloc[idx_neighbour]], axis = 1)
            df.columns = [f"Synthetic obs. (idx: {idx_synth})", f"Closest real obs. (idx: {idx_neighbour})"]
            print(f"# {i+1}. Closest pair with distance: {dist: .4f}")
            display(Markdown(df.to_markdown()))
