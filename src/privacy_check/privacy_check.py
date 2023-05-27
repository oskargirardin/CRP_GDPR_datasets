from sdmetrics.reports.single_table import DiagnosticReport
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tqdm
from sdv.metadata import SingleTableMetadata

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
    

    def _get_column_types(self):
        """
        Function that gets the column types of the data

        return: numerical columns (list), categorical columns (list)
        """
       
        if isinstance(self.metadata, SingleTableMetadata):
            metadata = self.metadata.to_dict()
            dtypes = {col: col_type["sdtype"] for col, col_type in metadata["columns"].items()}
        else:
            metadata = self.metadata
            dtypes = {col: col_type["type"] for col, col_type in metadata["fields"].items()}
        numeric_cols = [col for col, type in dtypes.items() if type == "numerical"]
        cat_cols = [col for col, type in dtypes.items() if type != "numerical"]
        return numeric_cols, cat_cols


    def _filter_columns(self, columns, *dfs):
        """
        Function that subsets dataframes to only include given columns

        :param columns: columns that should be included
        :param *dfs: dataframes that should be subsetted
        """
        for df in dfs:
            for col in df.columns:
                if col not in columns:
                    df.drop(col, axis = 1, inplace = True)


    def find_nearest_neighbours(self, sensitve_columns = None, verbose = True):
        """
        Function that finds the nearest neighbours of every synthetic row. Rows with NA values have a distance of NA as well.

        :param verbose (optional): boolean that controls the verbosity of the output
        :param sensitve_columns (optional): list to specify columns that should play a role for distance computation

        return: list of nearest neighbours of all sythentic rows
        """
        # Initialization
        neighbour_pairs = []
        n_samples = len(self.synthetic_data)
        df_real, df_synth = self.original_data.copy(), self.synthetic_data.copy()

        # Separate dataframes into numerical and categorical
        numeric_cols, cat_cols = self._get_column_types()
        df_real_num, df_real_cat = df_real[numeric_cols], df_real[cat_cols]
        df_synth_num, df_synth_cat = df_synth[numeric_cols], df_synth[cat_cols]
        
        # Normalize numeric columns to give equal weights to each column
        scaler = StandardScaler()
        df_real_num = pd.DataFrame(scaler.fit_transform(df_real_num), columns=numeric_cols)
        df_synth_num = pd.DataFrame(scaler.transform(df_synth_num), columns=numeric_cols)
        
        if not sensitve_columns is None:
            self._filter_columns(sensitve_columns, df_real_num, df_real_cat, df_synth_num, df_synth_cat)


        # For every synthetic row, find its nearest neighbour
        for idx_synth in tqdm.tqdm(range(n_samples), desc='Finding nearest neighbours', disable=(not verbose)):
            # Get numerical elements and categorical elements
            row_num = df_synth_num.iloc[idx_synth].to_numpy()
            row_cat = df_synth_cat.iloc[idx_synth].to_numpy()
            # Compute distance to every real row 
            # TODO: Give more importance to dists_cat, because dists_num > dists_cat in general
            dists_num = np.sum(np.abs(df_real_num.to_numpy() - row_num), axis = 1)
            dists_cat = np.sum(df_real_cat.to_numpy() != row_cat, axis = 1)
            dists = (dists_num + dists_cat) / (len(numeric_cols) + len(cat_cols))
            # Find minimal distance and append neighbour index
            min_dist = np.min(dists)
            idx_neighbour = np.argmin(dists)
            neighbour_pairs += [(idx_synth, idx_neighbour, min_dist)]

        self.pairs = neighbour_pairs

        return neighbour_pairs

    def display_closest_pairs(self, k):
        """
        Displays the k closest pairs

        :param k (int): number of closest pairs to display 
        """
        assert k > 0, "k must be larger than 0"
        try:
            pairs_sorted = sorted(self.pairs, key = lambda x: x[2])
        except AttributeError:
            raise Exception("self.pairs is not defined, please run function find_nearest_neighbours first.")
        k_closest_pairs = pairs_sorted[:k]
        print(f"############ TOP {k} CLOSEST PAIRS ############")
        for i in range(k):
            idx_synth, idx_neighbour, dist = k_closest_pairs[i]
            df = pd.concat([self.synthetic_data.iloc[idx_synth], self.original_data.iloc[idx_neighbour], ], axis = 1)
            df.columns = [f"Synthetic obs. (idx: {idx_synth})", f"Closest real obs. (idx: {idx_neighbour})"]
            print(f"{i+1}. Closest pair with distance: {dist: .4f}")
            return df

    
    def delete_closest_synthetic_columns(self, k):
        pass
