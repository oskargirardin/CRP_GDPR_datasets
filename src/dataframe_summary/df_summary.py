#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:39:05 2023

@author: lucreziacerto
"""

import pandas as pd
import random 


def dataframe_summary(df, file_name):
    '''
    Inputs:
    df: dataframe to summarize
    file_name: name of the file the summary should be saved in. Of the format 'file_name.csv'
    
    Prints:
        Number of rows of the dataset
        Dataset summary (column names, columns data type, example)
        Columns that have multiple datatypes
        
    Saves:
        Dataset summary as a csv file
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
    print('Number of rows of the dataset:', len(df),'\n')
    print('---- DATAFRAME SUMMARY -----')
    print(summary)
    print('---------')
    
    #Checking if there are columns with multiples data types - it becomes an issue in data generation
    print('\nColumns with different datatypes')
    for col in df.columns:
        unique_types = df[col].apply(type).unique()
        if len(unique_types) > 1:
            print(col, unique_types)
