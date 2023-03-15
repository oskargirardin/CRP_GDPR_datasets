# Import libraries
from sdmetrics.reports.single_table import DiagnosticReport
from sdmetrics.reports.utils import get_column_plot

class PrivacyCheck(DiagnosticReport):
    """
    Class to generate a report on the synthetic data, that checks 
    how similar it is to the original one, with respect to privacy
    concerns. 
    """
    def __init__(self):
        super().__init__()
        

    def generate_report(self, original_data, synthetic_data, metadata, verbose = True):
        """
        Generates the privacy report using the NewRowSynthesis metric.

        :param original_data: original pandas dataframe
        :param synthetic_data: synthetic pandas dataframe
        :param metadata: dictionary of the column types
        :param verbose: controls verbosity of report generation
        """
        self.original_data = original_data
        self.synthetic_data = synthetic_data
        self.metadata = metadata
        self.generate(original_data, synthetic_data, metadata, verbose)
    

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
    
    def get_column_plot(self, column_name):
        """
        Returns a plot of a comparison of the real vs. sythetic column.

        :param column_name: name of the column to plot
        """
        get_column_plot(
            self.original_data,
            self.synthetic_data,
            column_name,
            self.metadata
        )