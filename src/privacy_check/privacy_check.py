# Import libraries
from sdmetrics.reports.single_table import DiagnosticReport


class PrivacyCheck(DiagnosticReport):
    """
    Class to generate a report on the synthetic data, that checks 
    how similar it is to the original one, with respect to privacy
    concerns.

    For the moment the class only inherits of DiagnosticReport.
    
    Attributes:
        - Synthesis, Coverage, Boundaries scores
        - Real data
        - Sythetic Data
        - Datatypes of columns

    Methods:
        - Create Report (-> compute Synthesis, Coverage, Boundaries scores)
        - Get details on report (return summary of report)
        - Get individual scores
        - Visualizations
        - Save report as a file
 
    """
    def __init__(self):
        super().__init__()
        
    def report(self, real_data, synthetic data, metadata):
        """
        Generates a report
        real_data: A pandas.DataFrame containing the real data
        synthetic_data: A pandas.DataFrame containing the synthetic data
        metadata: A dictionary describing the format and types of data. See Single Table Metadata for more details.
        """
        report = DiagnosticReport()
        results_output=report.generate(real_data, synthetica data, metadata)
        properties_output=report.get_properties()
        return results_output, properties
    
    def visualize(self, property_name):
        """
        Plot visualizations based on the property chosen. 
        """
        fig = report.get_visualization(property_name='Coverage')
        viz=fig.show()
        return viz
        
    

