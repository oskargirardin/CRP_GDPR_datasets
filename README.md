# CRP_GDPR_datasets

## Structure of library

### Generator

Generates synthetic data using two different generative models, either CTGAN or GaussianCopula. 

Initialize a generator object using the following function call: 

    generator = Generator(data, architecture, n_samples, n_epochs=None, categorical_columns=None, sensitive_columns=None))

Parameters:
  * Real data
  * Architecture (ctgan, gaussiancopula,...)
  * n_samples refers to the number of synthetic samples to generate. 
  * Number of epochs
  * List of categorical columns 
  * List of columns of privacy concerns

Attributes: 

  * n_epochs
  * n_samples
  * architecture
  * metadata
  * data
  * categorical_columns
  * sensitive columns



Methods:
  1. create_metadata(): This function takes in the training dataframe and outputs metadata that can be accessed through the Generator.metadata attribute. It is automatically called upon creation of the generator object but should be checked before calling the generate function described below. 

    generator.create_metadata()
  3. generate(): this method generates synthetic data using the chosen generative model (either CTGAN or GaussianCopula) and returns it as a pandas dataframe.

    generator.generate()
  5. faker_categorical(): this method uses the Faker library to generate fake categorical data. This is not intended for use in machine learning models as correlations with the real data are not maintained.  However, it can be used as an alternative for dropping sensitive data columns. Currently, the following types of data can be faked: 

    * ID: an identifier
    * First name
    * Last name
    * email
    * gender
    * ip_address
    * nationality
    * city
   

We want to stress again that these attributes should not be used in a Machine Learning model and are purely there for anonymization purposes. 

    generator.faker_categorical()



### Similarity Check

The SimilarityCheck class is used to check the quality of synthetic data, both visually and with metrics. It provides methods to compare the real and synthetic data, generate visual comparisons, and compare the performance of machine learning models trained on real and synthetic data.

To initialize an instance of the SimilarityCheck class, the following arguments need to be passed:

 * real_data: a Pandas dataframe containing the real data
 * synthetic_data: a Pandas dataframe containing the synthetic data
 * cat_cols: a list of categorical columns in the data (optional)
 * metadata: metadata for the data (optional), included in the generator object.

       sim_check = SimilarityCheck(real_data=my_real_dataframe,
                            synthetic_data=my_synthetic_dataframe,
                            cat_cols=my_categorical_columns,
                            metadata=metadata)
Methods

#### TODO: make figure of adaptive size based on number of columns and find a way to display categorical columns with limited categories
**1. visual_comparison_columns()**

This method generates visual comparisons between the real and synthetic data. It plots data in one of three ways:

 * Numeric columns are plotted using the densities
 * Categorical columns with limited (less than 5) categories are plotted with a bar plot
 * For categorical columns with more than five categories, it plots a density histogram.

The function can be calles like this: 
    
    sim_check.visual_comparison_columns()


**2. comparison_columns()** 

This method compares the KL divergence for numerical variables.

    sim_check.comparison_columns()

#### Not tested yet
**3. compare_correlations()** 

This method compares correlation matrices between the real and synthetic data.

    sim_check.compare_correlations()

#### Not tested yet
**4. compare_model_performance()**


This method compares the performance of machine learning models trained on real and synthetic data. It takes four arguments:

fitted_model_real: a machine learning model trained on the real data
fitted_model_synth: a machine learning model trained on the synthetic data
X_test: test features
y_test: test targets

    sim_check.compare_model_performance(fitted_model_real=my_fitted_model_real,
                                     fitted_model_synth=my_fitted_model_synth,
                                     X_test=my_test_features,
                                     y_test=my_test_targets)



### Privacy check 
(Based on SDMetrics Diagnostic Report: https://docs.sdv.dev/sdmetrics/reports/diagnostic-report/single-table-api)

Contains the PrivacyCheck class, which generates a report on the similarity between original and synthetic data with respect to privacy concerns. 

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

### Customization 

The PrivacyCheck class uses the NewRowSynthesis metric to generate the privacy report. You can customize the behavior of the report by modifying the parameters passed to this metric. For example, you can change the sensitivity threshold or the privacy model used.

You can also customize the visualizations generated by the get_visualization method by modifying the property_name argument. The available options are 'Synthesis', 'Coverage', and 'Boundaries'.

### Main 
This project contains a Python script main.py that generates synthetic data using the CTGAN algorithm and evaluates the similarity between the original and generated data. 

### Usage

* Clone this repository to your local machine.
* Install the required libraries using pip install -r requirements.txt.
* Modify the parameters passed to the Generator class in main.py
  - Change the path to your csv (path_test_data)
  - Change the list of categorical columns (cat_cols)
  - Change the list of sensitive columns (sensitive_cols)
  - Change the metadata to specify the type of data you want to generate (my_metadata)
* Run main.py using python main.py.

The generated synthetic data will be outputted to the console and stored in a CSV file named synth_data.csv if you uncomment #df.to_csv('synth_data.csv'). The similarity between the original and generated data will also be evaluated and printed to the console. 


