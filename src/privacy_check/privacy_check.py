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
    

