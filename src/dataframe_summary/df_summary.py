#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: [Author Name]
Date: [Date]

Description:
This module contains a function for generating a summary of a pandas dataframe. The function takes in a dataframe and a file name, and generates a summary dataframe with the following information:
- names of the columns
- data types of the columns
- randomly selected example to show the column's formatting

Dependencies:
- pandas
- random

Usage:
Call the dataframe_summary() function with the dataframe to summarize and the desired file name for the summary output. The function returns the summary dataframe and also saves it to a CSV file.

Warning: HERE LUCRE ADDS A COMMENT TO EXPLAIN WHY WE PRINT THE VARIABLES THAT CONTAIN MULTIPLE DATATYPES
"""

import pandas as pd
import random 


def dataframe_summary(df, file_name):
    '''
    Inputs:
    df: dataframe to summarize
    file_name: name of the file the summary should be saved in. Of the format 'file_name.csv'
    
    Returns: summary dataframe with the following information:
        - names of the columns 
        - data types fo the columns
        - randomly selected example to show the column's formatting
    
    '''
    summary = pd.DataFrame()

    summary['column_names']= df.columns
    summary['column_types']= list(df.dtypes)
    summary['example'] = pd.Series(dtype='object')
  
    for i in range(len(df.columns)):
      col_name = summary.loc[i, 'column_names']
      #example is randomly selected among all the occurrences in the column
      x = random.randint(0, len(summary)-1)
      summary.at[i, 'example'] = str(df.loc[x,col_name])
  
    summary.to_csv(file_name)
    return summary
