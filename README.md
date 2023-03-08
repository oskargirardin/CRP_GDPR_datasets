# CRP_GDPR_datasets

## Structure of library

### Generator
Attributes:
- Architecture (ctgan, pategan, gaussiancopula,...)
- Number of epochs
- Training data
- List of categorical columns 
- List of columns of privacy concerns

Methods:
- Main method that implements the logic and calls other methods that implement the specific architectures
- Method for each individual architecture
- Method for manual correction of privacy sensitive columns (GAN would never generate new emailaddresses from a column, so privacy concerns would remain, we have to fake them with faker or Chat-GPT,...)


### Similarity Check
Attributes:
- Type of data
- Metrics to check
- Real table
- Synthetic table

Methods:
- Calculating the metrics
- Making visualizations
- Privacy checks

### Privacy check (Based on SDMetrics Diagnostic Report: https://docs.sdv.dev/sdmetrics/reports/diagnostic-report/single-table-api)
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