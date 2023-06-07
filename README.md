# CRP_GDPR_datasets

**IMPORTANT: To install the packages required with this library, the python version should not be above 3.10!**

This repository contains the functionality to create synthetic tabular and time series data. To use it, clone the repository locally, 
and install the desired requirements file: pip install -r SingleTableRequirements.txt or pip install -r TsRequirements.txt

Afterwards, the UserGuides folder contains a guide on how to use the library contained in the src folder for both time series and tabular data generation. 
The main.py files in the SingleTableGeneration and TimeSeriesGeneration show a brief version of the workflow. 

The data folder contains data used in the userguide and main.py files.

## Structure of the library

### Single Table Generation

#### Generator


Generates synthetic data using two different generative models, either CTGAN or GaussianCopula. 

Initialize a generator object using the following function call: 

    generator = Generator(data, architecture, n_samples, n_epochs=None, categorical_columns=None, sensitive_columns=None))

Parameters:
  * Real data
  * Architecture (ctgan, gaussiancopula,RealTabFormer)
  * n_samples refers to the number of synthetic samples to generate. 
  * Number of epochs
  * List of categorical columns 
  * List of columns of privacy concerns

Attributes: 

  * num_epochs: number of epochs to train, default = 200
  * num_bootstrap: number of bootstraps for RealTabFormer, default = 500, can be used to speed up process
  * n_samples: number of samples to generate
  * architecture: CTGAN, GaussianCopula or RealTabFormer
The RealTabFormer can overfit the real data and generate equivalent rows if it trains for too long. This can be checked with the privacy check class described below. If too many rows are equivalent to the real data, we suggest limiting the number of epochs. 
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



#### Similarity Check

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


**3. compare_correlations()** 

This method compares correlation matrices between the real and synthetic data.

    sim_check.compare_correlations()



#### Privacy check 
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

#### Customization 

The PrivacyCheck class uses the NewRowSynthesis metric to generate the privacy report. You can customize the behavior of the report by modifying the parameters passed to this metric. For example, you can change the sensitivity threshold or the privacy model used.

You can also customize the visualizations generated by the get_visualization method by modifying the property_name argument. The available options are 'Synthesis', 'Coverage', and 'Boundaries'.

#### Main 
This project contains a Python script main.py that generates synthetic data using the CTGAN algorithm and evaluates the similarity between the original and generated data. 

#### Usage

* Clone this repository to your local machine.
* Install the required libraries using pip install -r SingleTableRequirements.txt
* Run main.py using python main.py, do this either from the SingleTableGeneration 


### Time Series Generation

#### DataProcessor

For the PARSynthesizer that we use, the data has to be in 'long' format. The DataProcessor contains all the necessary methods to easily do this. 

    data_processor = DataProcessor(df, metadata = None, obs_limit = 1000, interpolate = True, drop_na_cols = True, long = False)


Attributes: 
* df: the data to process
* metadata: the metadata of the data to process
* obs_limit: the number of rows to use (last k observations for the time series)
* interpolate: whether to interpolate nan values
* drop_na_cols: whether to drop nan columns

    
Methods: 

**1. convert_to_long_format(time_columns, desired_identifiers, verbose)** 

    data_processor.convert_to_long_format(time_columns, desired_identifiers=None, verbose = False)


Attributes: 
* time_columns: which columns order the observations?
* desired_identifiers: a list of the columns you want to include as identifiers, and on which the model
      should be trained. If None, all columns will become an identifier in long format
* verbose: if True, print the dataframe

**2. get_metadata_long_df(identifier, time_column, datetime_format=None)** 

    data_processor.get_metadata_long_df(identifier, time_column, datetime_format=None)

Attributes: 
* identifier: the sequence identifier, the columns in wide format (in long format, the Variable column)
* time_column: orders the observations for each sequence, should be a numeric or a datetime format
* datetime_format: in what format is the date? For example '%Y-%m-%d %H:%M:%S'.


#### TSGenerator

A class that can generate synthetic time series using the PARSynthesizer method available in the Synthetic Data Vault. 

    generator = TSGenerator(df, metadata, method='PAR', verbose=False, cuda=False)

Attributes: 
* df: the dataframe, which for the PARSynthesizer should be in long format, which can be achieved with the DataProcessor.
* metadata: the metadata corresponding to the long dataframe
* method: the method with which to generate time series
* verbose: whether to print training progress
* cuda: whether a GPU is available


Methods: 

**1. train(n_epochs = 100)**

    generator.train()

This function will train the generator on the data passed to the constructor. 

Attributes: 
* n_epochs: the number of epochs to train for

**2. sample(n_samples = 10, sequence_length = None)**

    generator.sample()

This function will sample a given amount of sequences

Attributes: 
* n_samples: the number of sequences to generate
* sequence_length: the length of each sequence

#### TSSimilarityCheck

A class that will check the similarity for the time series

    sim_checker = TSSimilarityCheck(df_real, df_synth, metadata)
    
Attributes: 
* df_real: the real time series (long format)
* df_synthetic: the synthetic time series (long format)
* metadata: the metadata for the real data

Methods: 

**1. compute_distance_matrix()**

    sim_checker.compute_distance_matrix()
    
Computes a matrix of dynamic time warping distances between each real and synthetic time series. 


**2. get_mean_nn_distances()**

    sim_checker.get_mean_nn_distances()

Get the mean DTW distance for all closest pairs

**3. plot_nearest_neighbours(sequence_column = "variable", value_column = "value", time_column = "time")**

    sim_checker.plot_nearest_neighbours()

Function that plots the nearest (synthetic) time series for every real time series.

Attributes: 

* sequence_column: column that identifies different sequences
* value_column: column that contains the values of the time series
* time_column: column that identifies the time point


#### Usage

An example usage can be found in the UserGuide and in the main.py file in the TimeSeriesGeneration folder. The TsRequirements.txt contains the packages that should be installed. 
