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

### Main 
This project contains a Python script main.py that generates synthetic data using the CTGAN algorithm and evaluates the similarity between the original and generated data. 

## Usage

*Clone this repository to your local machine.
*Install the required libraries using pip install -r requirements.txt.
*Modify the parameters passed to the Generator class in main.py
- Change the path to your csv (path_test_data)
- Change the list of categorical columns (cat_cols)
- Change the list of sensitive columns (sensitive_cols)
- Change the metadata to specify the type of data you want to generate (my_metadata)
*Modify the path_test_data, cat_cols, and sensitive_cols variables in main.py to match your data.
*Run main.py using python main.py.

The generated synthetic data will be outputted to the console and stored in a CSV file named synth_data.csv if you uncomment #df.to_csv('synth_data.csv'). The similarity between the original and generated data will also be evaluated and printed to the console. 


