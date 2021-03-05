import pandas as pd
import numpy as np
from numpy.random import uniform

import copy
from fuzzywuzzy import process
from fuzzywuzzy import fuzz

# Plotting libraries
import matplotlib.pyplot as plt

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


def addNumbers(a, b): 
    print("Sum is ", a + b) 

def addMoreNumbers(a, b, c): 
    print("Sum is ", a + b + c) 
  
def subtractNumbers(a, b): 
    print("Difference is ", a-b) 

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


# def eliminate_sparse(col_name:str):
#     director_n = df_split_test.groupby(col_name) \
#        .agg({col_name:'size', 'imdb_rating':'mean'}) \
#        .rename(columns={col_name:'count','imdb_rating':'mean_rating'}) \
#        .reset_index()
    
#     multiple_dirs = director_n[director_n["count"]>= 5]
#     single_dirs = director_n[director_n["count"]< 5]
#     multiple_dirs.sort_values('mean_rating')

#     single_dirs[col_name] = col_name +'_' + single_dirs[col_name].astype(str)
#     to_elim_dirs = list(single_dirs[col_name])

#     return to_elim_dirs

def date_and_time(df_full, df_split_test, ccc):
    df_full = pd.concat([df_split_test, ccc], axis=1)
    df_full["month"] = df_full["air_date"].str[5:7]
    df_full['day_of_week'] = pd.to_datetime(df_full['air_date']).dt.dayofweek
    days = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}

    df_full['day_of_week'] = df_full['day_of_week'].apply(lambda x: days[x])
    
    return df_full

def get_coefs(m):
    """Returns the model coefficients from a Scikit-learn model object as an array,
    includes the intercept if available.
    """
    
    # If pipeline, use the last step as the model
    if (isinstance(m, sklearn.pipeline.Pipeline)):
        m = m.steps[-1][1]
    
    
    if m.intercept_ is None:
        return m.coef_
    
    return np.concatenate([[m.intercept_], m.coef_])

def model_fit(m, X, y, plot = False):
    """Returns the root mean squared error of a fitted model based on provided X and y values.
    
    Args:
        m: sklearn model object
        X: model matrix to use for prediction
        y: outcome vector to use to calculating rmse and residuals
        plot: boolean value, should fit plots be shown 
    """
    
    y_hat = m.predict(X)
    rmse = mean_squared_error(y, y_hat, squared=False)
    
    res = pd.DataFrame(
        data = {'y': y, 'y_hat': y_hat, 'resid': y - y_hat}
    )
    
    if plot:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        sns.lineplot(x='y', y='y_hat', color="grey", data =  pd.DataFrame(data={'y': [min(y),max(y)], 'y_hat': [min(y),max(y)]}))
        sns.scatterplot(x='y', y='y_hat', data=res).set_title("Fit plot")
        
        plt.subplot(122)
        sns.scatterplot(x='y', y='resid', data=res).set_title("Residual plot")
        
        plt.subplots_adjust(left=0.0)
        
        plt.suptitle("Model rmse = " + str(round(rmse, 4)), fontsize=16)
        plt.show()
    
    return rmse

def dataframe_prep(dataframe,col_predicted:str):
    '''
    Create inputs X dataframe, outputs y dataframe, and split the data using test_train_split from sklearn
    inputs: dataframe = dataframe to be modelled
            col_predicted
    '''
    # Lose the columns we are predicting from inputs X, and write this to outputs y
    X = dataframe.drop(col_predicted, axis = 1)
    y = dataframe[col_predicted]

    # Test train split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_test, y_train, y_test

def elim_writers_directors(df_split_test):
       
       # Eliminating writers and directors that appear less than 5 times

       director_n = df_split_test.groupby('director') \
              .agg({'director':'size', 'imdb_rating':'mean'}) \
              .rename(columns={'director':'count','imdb_rating':'mean_rating'}) \
              .reset_index()

       multiple_dirs = director_n[director_n["count"]>= 5]
       multiple_dirs.sort_values('mean_rating')
       single_dirs = director_n[director_n["count"]< 5]

       single_dirs['director'] = 'director_' + single_dirs['director'].astype(str)
       to_elim_dirs = list(single_dirs['director'])

       writer_n = df_split_test.groupby('writer') \
              .agg({'writer':'size', 'imdb_rating':'mean'}) \
              .rename(columns={'writer':'count','imdb_rating':'mean_rating'}) \
              .reset_index()

       multiple_writers = writer_n[writer_n["count"]>= 5]
       single_writers = writer_n[writer_n["count"]< 5]
       multiple_writers.sort_values('mean_rating')

       single_writers['writer'] = 'writer_' + single_writers['writer'].astype(str)
       to_elim_writers = list(single_writers['writer'])

       return to_elim_dirs, to_elim_writers


# def run_linear_regression(dataframe):
#     '''
#     Run standard linear regression
#     Function immediately drops the column we are predicting (TODO: have this as an input string)
#     input: dataframe on which you wish to run linea regression
#             where all features are dummies, i.e. 0/1
#     '''
    
#     X_train, X_test, y_train, y_test = dataframe_prep(dataframe,'imdb_rating')

#     #lm = LinearRegression().fit(X_train, y_train)
#     #model_fit(lm, X_test, y_test, plot=True)
#     #print("number of coefficients:",len(get_coefs(lm)))

#     first = make_pipeline(
#             LinearRegression(fit_intercept = False)
#         )

#     parameters = {'linearregression__normalize': [True,False]}

#     kf = KFold(n_splits=5, shuffle=True, random_state=0)

#     #this is the name you must change for each dataframe
#     first_grid = GridSearchCV(first,parameters,  cv=kf, scoring="neg_root_mean_squared_error").fit(X_train, y_train)

#     #==Print the results========
#     print("best index: ", first_grid.best_index_) #position of the array of the degree
#     print("best param: ", first_grid.best_params_)
#     print("best neg_root_mean_squared_error (score): ", first_grid.best_score_ *-1)
#     print("number of coefficients:", len(first_grid.best_estimator_.named_steps['linearregression'].coef_))

#     y_hat = first_grid.predict(X_test)
#     model_fit(first_grid, X_test, y_test, plot=True)
#     rmse_test = mean_squared_error(y_test, y_hat, squared=False)
#     rmse_train = mean_squared_error(y_train, y_train, squared=False)

#     print('rmse_test == ', rmse_test)
#     print('rmse_train == ', rmse_train)
#     print(first_grid.best_estimator_.named_steps['linearregression'].coef_)
#     print("intercept == ",first_grid.best_estimator_.named_steps['linearregression'].intercept_)

#     res = pd.DataFrame(
#             data = {'y': y_test, 'y_hat': y_hat, 'resid': round(y_test - y_hat,1)}
#         )
    
#     return res,rmse_train,rmse_test

# def run_linear_regr_standardisation(dataframe):
#     '''
#     Linear regression model with standardisation

#     '''
#     # Lose the columns we are predicting from inputs X, and write this to outputs y
#     X = dataframe.drop('imdb_rating', axis = 1)
#     y = dataframe["imdb_rating"]

#     # Test train split:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
#     first_2 = make_pipeline(
#             StandardScaler(),
#             LinearRegression(fit_intercept = False)
#         )

#     parameters = {'linearregression__normalize': [True,False]}

#     kf = KFold(n_splits=5, shuffle=True, random_state=0)
#     first2_grid = GridSearchCV(first_2,parameters,  cv=kf, scoring="neg_root_mean_squared_error").fit(X_train, y_train)


#     print("best index: ", first2_grid.best_index_) #position of the array of the degree
#     print("best param: ", first2_grid.best_params_)
#     print("best score: ", first2_grid.best_score_ *-1)
#     print("number of coefficients:",len(first2_grid.best_estimator_.named_steps['linearregression'].coef_))

#     y_hat = first2_grid.predict(X_test)
#     model_fit(first2_grid, X_test, y_test, plot=True) #compute over test
#     rmse_test = mean_squared_error(y_test, y_hat, squared=False)
#     rmse_train = mean_squared_error(y_train, y_train, squared=False)

#     print('rmse_test == ', rmse_test)
#     print('rmse_train == ', rmse_train)
#     print(first2_grid.best_estimator_.named_steps['linearregression'].coef_)
#     print("intercept == ",first2_grid.best_estimator_.named_steps['linearregression'].intercept_)

#     res = pd.DataFrame(
#             data = {'y': y_test, 'y_hat': y_hat, 'resid': round(y_test - y_hat,1)}
#         )

#     return res, rmse_train, rmse_test

# def run_polynomial_regression(dataframe):
#     '''
#     Run the polynomial regression model 
#     '''

#     # Lose the columns we are predicting from inputs X, and write this to outputs y
#     X = dataframe.drop('imdb_rating', axis = 1)
#     y = dataframe["imdb_rating"]

#     # Test train split:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#     second = make_pipeline(
#             PolynomialFeatures(),
#             LinearRegression()
#         )

#     parameters = {
#         'polynomialfeatures__degree': np.arange(1,3,1),
#         "linearregression__fit_intercept" : [True,False],
#         'linearregression__normalize': [True,False]
#     }

#     kf = KFold(n_splits=5, shuffle=True, random_state=0)

#     second_grid = GridSearchCV(second, parameters, cv=kf, scoring="neg_root_mean_squared_error").fit(X_train, y_train)
#     print("best index: ", second_grid.best_index_) #position of the array of the degree
#     print("best param: ", second_grid.best_params_)
#     print("best score: ", second_grid.best_score_ *-1)
#     print("number of coefficients:",len(second_grid.best_estimator_.named_steps['linearregression'].coef_))

#     y_hat = second_grid.predict(X_test)
#     model_fit(second_grid, X_test, y_test, plot=True)
#     rmse = mean_squared_error(y_test, y_hat, squared=False)
#     #print(rmse)
#     print(second_grid.best_estimator_.named_steps['linearregression'].coef_)
#     print("intercept == ",second_grid.best_estimator_.named_steps['linearregression'].intercept_)
    
# def run_lasso_(dataframe):
#     '''
#     Run Lasso... 

#     '''
    
#     # Lose the columns we are predicting from inputs X, and write this to outputs y
#     X = dataframe.drop('imdb_rating', axis = 1)
#     y = dataframe["imdb_rating"]

#     # Test train split:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#     alpha_list = np.linspace(0.01, 15, num=100)

#     third = make_pipeline(
#             StandardScaler(),
#             PolynomialFeatures(),
#             Lasso()
#         )

#     parameters = {
#         'polynomialfeatures__degree': np.arange(1,3,1),
#         'lasso__alpha': alpha_list,
#         'polynomialfeatures__include_bias': [True,False]
#     }

#     kf = KFold(n_splits=5, shuffle=True, random_state=0)

#     third_grid = GridSearchCV(third, parameters, cv=kf, scoring="neg_root_mean_squared_error").fit(X_train, y_train)
#     print("best index: ", third_grid.best_index_) #position of the array of the degree
#     print("best param: ", third_grid.best_params_)
#     print("best score: ", third_grid.best_score_ *-1)
#     print("number of coefficients:",len(third_grid.best_estimator_.named_steps["lasso"].coef_))


#     y_hat = third_grid.predict(X_test)
#     model_fit(third_grid, X_test, y_test, plot=True)
#     rmse = mean_squared_error(y_test, y_hat, squared=False)
#     print(rmse)
#     print("coefficients == ", third_grid.best_estimator_.named_steps["lasso"].coef_)
#     print("intercept == ", third_grid.best_estimator_.named_steps["lasso"].intercept_)


# def run_lasso_(dataframe):
#     '''
#     Run Lasso... 

#     '''
    
#     # Lose the columns we are predicting from inputs X, and write this to outputs y
#     X = dataframe.drop('imdb_rating', axis = 1)
#     y = dataframe["imdb_rating"]

#     # Test train split:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#     alpha_list = np.linspace(0.01, 15, num=100)

#     third = make_pipeline(
#             StandardScaler(),
#             PolynomialFeatures(),
#             Lasso()
#         )

#     parameters = {
#         'polynomialfeatures__degree': np.arange(1,3,1),
#         'lasso__alpha': alpha_list,
#         'polynomialfeatures__include_bias': [True,False]
#     }

#     kf = KFold(n_splits=5, shuffle=True, random_state=0)

#     third_grid = GridSearchCV(third, parameters, cv=kf, scoring="neg_root_mean_squared_error").fit(X_train, y_train)
#     print("best index: ", third_grid.best_index_) #position of the array of the degree
#     print("best param: ", third_grid.best_params_)
#     print("best score: ", third_grid.best_score_ *-1)
#     print("number of coefficients:",len(third_grid.best_estimator_.named_steps["lasso"].coef_))


#     y_hat = third_grid.predict(X_test)
#     model_fit(third_grid, X_test, y_test, plot=True)
#     rmse = mean_squared_error(y_test, y_hat, squared=False)
#     print(rmse)
#     print("coefficients == ", third_grid.best_estimator_.named_steps["lasso"].coef_)
#     print("intercept == ", third_grid.best_estimator_.named_steps["lasso"].intercept_)

# def run_ridge_(dataframe):
#     '''
#     Run the polynomial regression model - is this identical to above?
#     '''

#     # Lose the columns we are predicting from inputs X, and write this to outputs y
#     X = dataframe.drop('imdb_rating', axis = 1)
#     y = dataframe["imdb_rating"]

#     # Test train split:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#     alpha_list = np.logspace(-2, 3, num=200)


#     fourth = make_pipeline(
#             StandardScaler(),
#             PolynomialFeatures(),
#             Ridge()   
#         )

#     parameters = {
#         'polynomialfeatures__degree': np.arange(1,3,1),
#         'ridge__alpha': alpha_list,
#         'polynomialfeatures__include_bias': [True,False]
#     }

#     kf = KFold(n_splits=5, shuffle=True, random_state=0)

#     fourth_grid = GridSearchCV(fourth, parameters, cv=kf, scoring="neg_root_mean_squared_error").fit(X_train, y_train)
#     print("best index: ", fourth_grid.best_index_) #position of the array of the degree
#     print("best param: ", fourth_grid.best_params_)
#     print("best score: ", fourth_grid.best_score_ *-1)
#     print("number of coefficients:",len(fourth_grid.best_estimator_.named_steps["ridge"].coef_))

#     y_hat = fourth_grid.predict(X_test)
#     model_fit(fourth_grid, X_test, y_test, plot=True)
#     rmse = mean_squared_error(y_test, y_hat, squared=False)
#     print(rmse)
#     print("coefficients == ", fourth_grid.best_estimator_.named_steps["ridge"].coef_)
#     print("intercept == ", fourth_grid.best_estimator_.named_steps["ridge"].intercept_)

