
import pandas as pd
import numpy as np
from numpy.random import uniform

import copy
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

# Plotting libraries
import matplotlib.pyplot as plt
# import matplotlib as plt

import seaborn as sns
# Plotting defaults
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['figure.dpi'] = 80

# sklearn modules
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


def typo_cleaner(col_name, df):
    '''
    Function looks at column of data in pd dataframe, and makes checks 
    for typos. Conditions - all entries that only appear once, that have
    no semicolon in (i.e. only one name) and that are a >90% match to their
    closest match in the list. This doesnt catch all typos but, a good start!

    Dependencies: Requires fuzzywuzzy (from fuzzywuzzy import process)
    Inputs:
        col_name: string of column name you want to check
        df: pd dataframe on which you can do df['col_name']
    Output: none 
    '''
    k = 0 # just a counter
    choices = df[col_name]
    # # For each director in the list print out the matches:
    # for i in range(len(choices)): 
    #     k+=1
    
    #     print(choices[i],' === ',process.extract(choices[i], choices, limit=3))

    matches_limit = 3 # how many matches we want for each director - 3 is arbitrary but sensible

    for i in range(len(choices)):
        # Calculate closest matches to the current string
        matches = process.extract(choices[i], choices, limit = matches_limit)

        # If director only appears one time exactly, its a 90% match to its closest match, and doesnt contain ';':
        if (choices.str.count(choices[i]).sum()) == 1 and matches[1][1] > 90 and ";" not in choices[i]: 
            # print(df[col_name][i], df[col_name].str.count(df[col_name][i]).sum())
            # print('matches found === ', matches)
            # print('old entry == ', choices[i])
            df[col_name][i] = matches[1][0]
            # print (choices[i], '=== ', matches[1][0])
            # print('new entry == ', choices[i])

    print('Typo cleaner finished - if no other output, no changes were made')


def eliminate_sparse(col_name:str):
    director_n = df_split_test.groupby(col_name) \
       .agg({col_name:'size', 'imdb_rating':'mean'}) \
       .rename(columns={col_name:'count','imdb_rating':'mean_rating'}) \
       .reset_index()
    
    multiple_dirs = director_n[director_n["count"]>= 5]
    single_dirs = director_n[director_n["count"]< 5]
    multiple_dirs.sort_values('mean_rating')

    single_dirs[col_name] = col_name +'_' + single_dirs[col_name].astype(str)
    to_elim_dirs = list(single_dirs[col_name])

    return to_elim_dirs



