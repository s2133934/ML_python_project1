# Machine Learning in Python - Project 1

Due Friday, March 6th by 5 pm.

*include contributors names here*

## 0. Setup


```python
# Install required packages
!pip install -q -r requirements.txt
```


```python
# Add any additional libraries or submodules below

# Display plots inline
%matplotlib inline

# Data libraries
import pandas as pd
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting defaults
plt.rcParams['figure.figsize'] = (8,5)
plt.rcParams['figure.dpi'] = 80

# sklearn modules
import sklearn
```


```python
# Load data
d = pd.read_csv("the_office.csv")
d
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>episode</th>
      <th>episode_name</th>
      <th>director</th>
      <th>writer</th>
      <th>imdb_rating</th>
      <th>total_votes</th>
      <th>air_date</th>
      <th>n_lines</th>
      <th>n_directions</th>
      <th>n_words</th>
      <th>n_speak_char</th>
      <th>main_chars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Pilot</td>
      <td>Ken Kwapis</td>
      <td>Ricky Gervais;Stephen Merchant;Greg Daniels</td>
      <td>7.6</td>
      <td>3706</td>
      <td>2005-03-24</td>
      <td>229</td>
      <td>27</td>
      <td>2757</td>
      <td>15</td>
      <td>Angela;Dwight;Jim;Kevin;Michael;Oscar;Pam;Phyl...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>Diversity Day</td>
      <td>Ken Kwapis</td>
      <td>B.J. Novak</td>
      <td>8.3</td>
      <td>3566</td>
      <td>2005-03-29</td>
      <td>203</td>
      <td>20</td>
      <td>2808</td>
      <td>12</td>
      <td>Angela;Dwight;Jim;Kelly;Kevin;Michael;Oscar;Pa...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>Health Care</td>
      <td>Ken Whittingham</td>
      <td>Paul Lieberstein</td>
      <td>7.9</td>
      <td>2983</td>
      <td>2005-04-05</td>
      <td>244</td>
      <td>21</td>
      <td>2769</td>
      <td>13</td>
      <td>Angela;Dwight;Jim;Kevin;Meredith;Michael;Oscar...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>The Alliance</td>
      <td>Bryan Gordon</td>
      <td>Michael Schur</td>
      <td>8.1</td>
      <td>2886</td>
      <td>2005-04-12</td>
      <td>243</td>
      <td>24</td>
      <td>2939</td>
      <td>14</td>
      <td>Angela;Dwight;Jim;Kevin;Meredith;Michael;Oscar...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>Basketball</td>
      <td>Greg Daniels</td>
      <td>Greg Daniels</td>
      <td>8.4</td>
      <td>3179</td>
      <td>2005-04-19</td>
      <td>230</td>
      <td>49</td>
      <td>2437</td>
      <td>18</td>
      <td>Angela;Darryl;Dwight;Jim;Kevin;Michael;Oscar;P...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>181</th>
      <td>9</td>
      <td>19</td>
      <td>Stairmageddon</td>
      <td>Matt Sohn</td>
      <td>Dan Sterling</td>
      <td>8.0</td>
      <td>1484</td>
      <td>2013-04-11</td>
      <td>273</td>
      <td>59</td>
      <td>2965</td>
      <td>24</td>
      <td>Andy;Angela;Creed;Dwight;Erin;Jim;Kevin;Meredi...</td>
    </tr>
    <tr>
      <th>182</th>
      <td>9</td>
      <td>20</td>
      <td>Paper Airplane</td>
      <td>Jesse Peretz</td>
      <td>Halsted Sullivan;Warren Lieberstein</td>
      <td>8.0</td>
      <td>1482</td>
      <td>2013-04-25</td>
      <td>234</td>
      <td>48</td>
      <td>2564</td>
      <td>27</td>
      <td>Andy;Angela;Creed;Darryl;Dwight;Erin;Jim;Kevin...</td>
    </tr>
    <tr>
      <th>183</th>
      <td>9</td>
      <td>21</td>
      <td>Livin' the Dream</td>
      <td>Jeffrey Blitz</td>
      <td>Nicki Schwartz-Wright</td>
      <td>8.9</td>
      <td>2041</td>
      <td>2013-05-02</td>
      <td>382</td>
      <td>33</td>
      <td>4333</td>
      <td>20</td>
      <td>Andy;Angela;Creed;Darryl;Dwight;Erin;Jim;Kevin...</td>
    </tr>
    <tr>
      <th>184</th>
      <td>9</td>
      <td>22</td>
      <td>A.A.R.M</td>
      <td>David Rogers</td>
      <td>Brent Forrester</td>
      <td>9.3</td>
      <td>2860</td>
      <td>2013-05-09</td>
      <td>501</td>
      <td>54</td>
      <td>4965</td>
      <td>30</td>
      <td>Andy;Angela;Creed;Darryl;Dwight;Erin;Jim;Kevin...</td>
    </tr>
    <tr>
      <th>185</th>
      <td>9</td>
      <td>24</td>
      <td>Finale</td>
      <td>Ken Kwapis</td>
      <td>Greg Daniels</td>
      <td>9.7</td>
      <td>7934</td>
      <td>2013-05-16</td>
      <td>522</td>
      <td>107</td>
      <td>5960</td>
      <td>54</td>
      <td>Andy;Angela;Creed;Darryl;Dwight;Erin;Jim;Kelly...</td>
    </tr>
  </tbody>
</table>
<p>186 rows × 13 columns</p>
</div>



## 1. Introduction

*This section should include a brief introduction to the task and the data (assume this is a report you are delivering to a client). If you use any additional data sources, you should introduce them here and discuss why they were included.*

*Briefly outline the approaches being used and the conclusions that you are able to draw.*

## 2. Exploratory Data Analysis and Feature Engineering

### Notes on what to look at:
* As Seasons increase, we see an increased variance in the ratings given, whilst also a increasing then decreasing mean rating for the season (this could be tabulated and expressed in results), but we also see that total votes decreaeses as seasons go on. So it would appear that maybe the ratings are skewed as the show becomes more established/populare. 
* make number of lines & number of words & number of speaking characters into intervals, rather than just having integer values. We should provide ranges of these to help with forming the data from the grouped ()



```python
 d['binned_lines'] = pd.cut(d['n_lines'], bins=50)
```


```python
sns.pairplot(data=d)
```




    <seaborn.axisgrid.PairGrid at 0x7f2bad397190>



*Include a detailed discussion of the data with a particular emphasis on the features of the data that are relevant for the subsequent modeling. Including visualizations of the data is strongly encouraged - all code and plots must also be described in the write up. Think carefully about whether each plot needs to be included in your final draft - your report should include figures but they should be as focused and impactful as possible.*

*Additionally, this section should also implement and describe any preprocessing / feature engineering of the data. Specifically, this should be any code that you use to generate new columns in the data frame `d`. All of this processing is explicitly meant to occur before we split the data in to training and testing subsets. Processing that will be performed as part of an sklearn pipeline can be mentioned here but should be implemented in the following section.*

*All code and figures should be accompanied by text that provides an overview / context to what is being done or presented.*

## 3. Model Fitting and Tuning

*In this section you should detail your choice of model and describe the process used to refine and fit that model. You are strongly encouraged to explore many different modeling methods (e.g. linear regression, regression trees, lasso, etc.) but you should not include a detailed narrative of all of these attempts. At most this section should mention the methods explored and why they were rejected - most of your effort should go into describing the model you are using and your process for tuning and validatin it.*

*For example if you considered a linear regression model, a classification tree, and a lasso model and ultimately settled on the linear regression approach then you should mention that other two approaches were tried but do not include any of the code or any in depth discussion of these models beyond why they were rejected. This section should then detail is the development of the linear regression model in terms of features used, interactions considered, and any additional tuning and validation which ultimately led to your final model.* 

*This section should also include the full implementation of your final model, including all necessary validation. As with figures, any included code must also be addressed in the text of the document.*

## 4. Discussion & Conclusions


*In this section you should provide a general overview of your final model, its performance, and reliability. You should discuss what the implications of your model are in terms of the included features, predictive performance, and anything else you think is relevant.*

*This should be written with a target audience of a NBC Universal executive who is with the show and  university level mathematics but not necessarily someone who has taken a postgraduate statistical modeling course. Your goal should be to convince this audience that your model is both accurate and useful.*

*Finally, you should include concrete recommendations on what NBC Universal should do to make their reunion episode a popular as possible.*

*Keep in mind that a negative result, i.e. a model that does not work well predictively, that is well explained and justified in terms of why it failed will likely receive higher marks than a model with strong predictive performance but with poor or incorrect explinations / justifications.*

## 5. Convert Document


```python
# Run the following to render to PDF
!jupyter nbconvert --to markdown project1.ipynb
```

    [NbConvertApp] Converting notebook project1.ipynb to markdown
    [NbConvertApp] Writing 7103 bytes to project1.md


<a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=8d54bbee-a4bc-40dc-a5f1-a0190c0e14b4' target="_blank">
<img style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
