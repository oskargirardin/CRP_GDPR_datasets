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
