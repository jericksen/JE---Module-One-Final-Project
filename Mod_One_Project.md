
# Module 1 Final Project

For: Langleer Real Estate Associates

By: Jonathan E Ericksen, JE Consulting

## Background:

Langleer Real Estate Associates (LREA) has contracted with JE Consulting to develop a statistical model to predict housing prices within the greater King County, WA territory. The intent is to provide a tool by which LREA agents can leverage when consulting with clients who are looking to sell their real estate holdings. 

The content herin describes the methodology and technical steps used by JE Consulting to develop and improve the predictive regression model for use by the client. 

## Objective:

LREA has contracted with JE Consulting with the objective to build a multi-variate regression model that accuratley predicts housing prices throughout the King County territory. The resulting model will be used by LEAR associates as a tool to  engage when consulting with prospective sellers on a price to list their property. Further, this model will be used in consultation with future prospective sellers as to meaningful impovements that can be made to existing property, i.e., adding an extra bathroom, that will quantify the increase in home's value.

Upon completion, the model will exist within the confines of LEAR's current technology stack, and later used as an engine driving a mobile application for use by agents in the field. The mobile application will take in existing parameters on a given property, provide an estimated sale price, followed by an explatory workflow whereby agents can run hypothetical scenarios on potential property improvements and the resulting affects on the future sale price.

In addition to developing a predictive model, LREA would like these key questions answered through detailed visualizations from the existing data: 

- How are housing prices (home values) distributed throughout King County? And what factors might explains the concentration of expensive housing towards the northeast corner of the county?
- Does the age of the house have a noticeable impact on value?
- Does King County exhibit any seasonality in terms of housing inventory turnover?


## Methodology:

This project was approached using the industry standard OSEMiN process. The 5 stages in OSEMiN are outlined in the table of contents below with each stage highlighted within the ensuing notebook:

**Table of Contents:**

- **Obtain**: 
    - Sourcing the data

- **Scrub**: 
    - Data processing, cleaning & wrangling

- **Explore**: 
    - Question One, Visual One, Answer One
    - Question Two, Visual Two, Answer Two
    - Question Three, Visual Three, Answer Three
    - EDA Summary
    
- **Model**: 
    - Condition the data by identifying correlations and solving for coliniearity
    - Model the Data
    - Final Model
    - Model Summary
    - Cross Validation

- **Interpret**: 
    - Interpret the final results explaining the implications of the resulting coefficents & predictions


*(non- OSEMiN additions)*
- **Recommendations**
- **Conclusions**
- **Future Work**
------------------------------------------------------------------------------------------------------------------

### Obtaining The Data

Section Highlights:
- Import applicable libraries/packages
- Import data file as a pandas dataframe assigning it to the variable 'df'

------------------------------------------------------------------------------------------------------------------

*Import applicable libraries/packages*


```python
import pandas as pd
import pandas_profiling
import numpy as np

import statsmodels.api as sm
import statsmodels.stats as sts
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns
```

*Importing the data file using pandas and storing it as a datafram assigned to the variable 'df'*


```python
df = pd.read_csv('kc_house_data.csv')
```

### Scrubbing The Data

Section Highlights:
- Data familiarization (five point statistics)
- Removed ambiguous columns (yr_renovated, id)
- Removed rows with null values in the 'view' column 
- Replaced null values in the 'waterfront' column with the column median value
- Replaced the placeholder data ('?') in the 'sqft_basement' feature with the median column value

------------------------------------------------------------------------------------------------------------------

*Inspecting the columns from the dataset*


```python
df.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
           'lat', 'long', 'sqft_living15', 'sqft_lot15'],
          dtype='object')



*Inspecting the dataframe shape*


```python
df.shape
```




    (21597, 21)



*Inspecting the dataframe head*


```python
df.head()
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
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



*Inspect the data types as well as data volumes. Looking for missing values within the given features*


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21597 entries, 0 to 21596
    Data columns (total 21 columns):
    id               21597 non-null int64
    date             21597 non-null object
    price            21597 non-null float64
    bedrooms         21597 non-null int64
    bathrooms        21597 non-null float64
    sqft_living      21597 non-null int64
    sqft_lot         21597 non-null int64
    floors           21597 non-null float64
    waterfront       19221 non-null float64
    view             21534 non-null float64
    condition        21597 non-null int64
    grade            21597 non-null int64
    sqft_above       21597 non-null int64
    sqft_basement    21597 non-null object
    yr_built         21597 non-null int64
    yr_renovated     17755 non-null float64
    zipcode          21597 non-null int64
    lat              21597 non-null float64
    long             21597 non-null float64
    sqft_living15    21597 non-null int64
    sqft_lot15       21597 non-null int64
    dtypes: float64(8), int64(11), object(2)
    memory usage: 3.5+ MB


*Analyzing the five point statistics while rounding resulting output to 2 decimal points*


```python
df.describe().round(2)
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
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.159700e+04</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>19221.00</td>
      <td>21534.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>17755.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
      <td>21597.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.580474e+09</td>
      <td>540296.57</td>
      <td>3.37</td>
      <td>2.12</td>
      <td>2080.32</td>
      <td>15099.41</td>
      <td>1.49</td>
      <td>0.01</td>
      <td>0.23</td>
      <td>3.41</td>
      <td>7.66</td>
      <td>1788.60</td>
      <td>1971.00</td>
      <td>83.64</td>
      <td>98077.95</td>
      <td>47.56</td>
      <td>-122.21</td>
      <td>1986.62</td>
      <td>12758.28</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.876736e+09</td>
      <td>367368.14</td>
      <td>0.93</td>
      <td>0.77</td>
      <td>918.11</td>
      <td>41412.64</td>
      <td>0.54</td>
      <td>0.09</td>
      <td>0.77</td>
      <td>0.65</td>
      <td>1.17</td>
      <td>827.76</td>
      <td>29.38</td>
      <td>399.95</td>
      <td>53.51</td>
      <td>0.14</td>
      <td>0.14</td>
      <td>685.23</td>
      <td>27274.44</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>78000.00</td>
      <td>1.00</td>
      <td>0.50</td>
      <td>370.00</td>
      <td>520.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>370.00</td>
      <td>1900.00</td>
      <td>0.00</td>
      <td>98001.00</td>
      <td>47.16</td>
      <td>-122.52</td>
      <td>399.00</td>
      <td>651.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.123049e+09</td>
      <td>322000.00</td>
      <td>3.00</td>
      <td>1.75</td>
      <td>1430.00</td>
      <td>5040.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>7.00</td>
      <td>1190.00</td>
      <td>1951.00</td>
      <td>0.00</td>
      <td>98033.00</td>
      <td>47.47</td>
      <td>-122.33</td>
      <td>1490.00</td>
      <td>5100.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.904930e+09</td>
      <td>450000.00</td>
      <td>3.00</td>
      <td>2.25</td>
      <td>1910.00</td>
      <td>7618.00</td>
      <td>1.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>7.00</td>
      <td>1560.00</td>
      <td>1975.00</td>
      <td>0.00</td>
      <td>98065.00</td>
      <td>47.57</td>
      <td>-122.23</td>
      <td>1840.00</td>
      <td>7620.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.308900e+09</td>
      <td>645000.00</td>
      <td>4.00</td>
      <td>2.50</td>
      <td>2550.00</td>
      <td>10685.00</td>
      <td>2.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>8.00</td>
      <td>2210.00</td>
      <td>1997.00</td>
      <td>0.00</td>
      <td>98118.00</td>
      <td>47.68</td>
      <td>-122.12</td>
      <td>2360.00</td>
      <td>10083.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>7700000.00</td>
      <td>33.00</td>
      <td>8.00</td>
      <td>13540.00</td>
      <td>1651359.00</td>
      <td>3.50</td>
      <td>1.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>13.00</td>
      <td>9410.00</td>
      <td>2015.00</td>
      <td>2015.00</td>
      <td>98199.00</td>
      <td>47.78</td>
      <td>-121.32</td>
      <td>6210.00</td>
      <td>871200.00</td>
    </tr>
  </tbody>
</table>
</div>



*Beyond using the standard methods for familiarizing myself with the data, I ran a pandas profiling feature that provides an organized report on the dataset*


```python
pandas_profiling.ProfileReport(df)
```




<meta charset="UTF-8">

<style>

        .variablerow {
            border: 1px solid #e1e1e8;
            border-top: hidden;
            padding-top: 2em;
            padding-bottom: 2em;
            padding-left: 1em;
            padding-right: 1em;
        }

        .headerrow {
            border: 1px solid #e1e1e8;
            background-color: #f5f5f5;
            padding: 2em;
        }
        .namecol {
            margin-top: -1em;
            overflow-x: auto;
        }

        .dl-horizontal dt {
            text-align: left;
            padding-right: 1em;
            white-space: normal;
        }

        .dl-horizontal dd {
            margin-left: 0;
        }

        .ignore {
            opacity: 0.4;
        }

        .container.pandas-profiling {
            max-width:975px;
        }

        .col-md-12 {
            padding-left: 2em;
        }

        .indent {
            margin-left: 1em;
        }

        .center-img {
            margin-left: auto !important;
            margin-right: auto !important;
            display: block;
        }

        /* Table example_values */
            table.example_values {
                border: 0;
            }

            .example_values th {
                border: 0;
                padding: 0 ;
                color: #555;
                font-weight: 600;
            }

            .example_values tr, .example_values td{
                border: 0;
                padding: 0;
                color: #555;
            }

        /* STATS */
            table.stats {
                border: 0;
            }

            .stats th {
                border: 0;
                padding: 0 2em 0 0;
                color: #555;
                font-weight: 600;
            }

            .stats tr {
                border: 0;
            }

            .stats td{
                color: #555;
                padding: 1px;
                border: 0;
            }


        /* Sample table */
            table.sample {
                border: 0;
                margin-bottom: 2em;
                margin-left:1em;
            }
            .sample tr {
                border:0;
            }
            .sample td, .sample th{
                padding: 0.5em;
                white-space: nowrap;
                border: none;

            }

            .sample thead {
                border-top: 0;
                border-bottom: 2px solid #ddd;
            }

            .sample td {
                width:100%;
            }


        /* There is no good solution available to make the divs equal height and then center ... */
            .histogram {
                margin-top: 3em;
            }
        /* Freq table */

            table.freq {
                margin-bottom: 2em;
                border: 0;
            }
            table.freq th, table.freq tr, table.freq td {
                border: 0;
                padding: 0;
            }

            .freq thead {
                font-weight: 600;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;

            }

            td.fillremaining{
                width:auto;
                max-width: none;
            }

            td.number, th.number {
                text-align:right ;
            }

        /* Freq mini */
            .freq.mini td{
                width: 50%;
                padding: 1px;
                font-size: 12px;

            }
            table.freq.mini {
                 width:100%;
            }
            .freq.mini th {
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                max-width: 5em;
                font-weight: 400;
                text-align:right;
                padding-right: 0.5em;
            }

            .missing {
                color: #a94442;
            }
            .alert, .alert > th, .alert > td {
                color: #a94442;
            }


        /* Bars in tables */
            .freq .bar{
                float: left;
                width: 0;
                height: 100%;
                line-height: 20px;
                color: #fff;
                text-align: center;
                background-color: #337ab7;
                border-radius: 3px;
                margin-right: 4px;
            }
            .other .bar {
                background-color: #999;
            }
            .missing .bar{
                background-color: #a94442;
            }
            .tooltip-inner {
                width: 100%;
                white-space: nowrap;
                text-align:left;
            }

            .extrapadding{
                padding: 2em;
            }

            .pp-anchor{

            }

</style>

<div class="container pandas-profiling">
    <div class="row headerrow highlight">
        <h1>Overview</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-6 namecol">
        <p class="h4">Dataset info</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Number of variables</th>
                <td>21 </td>
            </tr>
            <tr>
                <th>Number of observations</th>
                <td>21597 </td>
            </tr>
            <tr>
                <th>Total Missing (%)</th>
                <td>1.4% </td>
            </tr>
            <tr>
                <th>Total size in memory</th>
                <td>3.5 MiB </td>
            </tr>
            <tr>
                <th>Average record size in memory</th>
                <td>168.0 B </td>
            </tr>
            </tbody>
        </table>
    </div>
    <div class="col-md-6 namecol">
        <p class="h4">Variables types</p>
        <table class="stats" style="margin-left: 1em;">
            <tbody>
            <tr>
                <th>Numeric</th>
                <td>19 </td>
            </tr>
            <tr>
                <th>Categorical</th>
                <td>2 </td>
            </tr>
            <tr>
                <th>Boolean</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Date</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Text (Unique)</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Rejected</th>
                <td>0 </td>
            </tr>
            <tr>
                <th>Unsupported</th>
                <td>0 </td>
            </tr>
            </tbody>
        </table>
    </div>
    <div class="col-md-12" style="padding-left: 1em;">
        
        <p class="h4">Warnings</p>
        <ul class="list-unstyled"><li><a href="#pp_var_date"><code>date</code></a> has a high cardinality: 372 distinct values  <span class="label label-warning">Warning</span></li><li><a href="#pp_var_sqft_basement"><code>sqft_basement</code></a> has a high cardinality: 304 distinct values  <span class="label label-warning">Warning</span></li><li><a href="#pp_var_view"><code>view</code></a> has 19422 / 89.9% zeros <span class="label label-info">Zeros</span></li><li><a href="#pp_var_waterfront"><code>waterfront</code></a> has 19075 / 88.3% zeros <span class="label label-info">Zeros</span></li><li><a href="#pp_var_waterfront"><code>waterfront</code></a> has 2376 / 11.0% missing values <span class="label label-default">Missing</span></li><li><a href="#pp_var_yr_renovated"><code>yr_renovated</code></a> has 17011 / 78.8% zeros <span class="label label-info">Zeros</span></li><li><a href="#pp_var_yr_renovated"><code>yr_renovated</code></a> has 3842 / 17.8% missing values <span class="label label-default">Missing</span></li> </ul>
    </div>
</div>
    <div class="row headerrow highlight">
        <h1>Variables</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_bathrooms">bathrooms<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>29</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.1%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>2.1158</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0.5</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>8</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-7025008681964841562">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAASRJREFUeJzt3cEJwkAQQFEVS7IIe/JsTxZhT2sD8lFBMsp798BePrPZhGS/1lo74KnD1guAyY5bL2Arp8vt7Wvu1/MXVsJkJggEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIhL/4LtYn37iCV5ggEAQCYdwWy3aJSUwQCAKBIBAIAoEgEAgCgTDumHeyd4%2Bg/ZHq95kgEAQCQSAQBAJBIBAEAkEgEDwH%2BaJPXt337GQWEwSCQCAIBIJ7kGHct8yyX2utrRcBU9liQRAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQHgAysgXZ0m7RPsAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-7025008681964841562,#minihistogram-7025008681964841562"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-7025008681964841562">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-7025008681964841562"
                                                  aria-controls="quantiles-7025008681964841562" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-7025008681964841562" aria-controls="histogram-7025008681964841562"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-7025008681964841562" aria-controls="common-7025008681964841562"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-7025008681964841562" aria-controls="extreme-7025008681964841562"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-7025008681964841562">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0.5</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>1.75</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>2.25</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>2.5</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>3.5</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>8</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>7.5</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>0.75</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>0.76898</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.36344</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>1.2793</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>2.1158</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>0.6146</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.51971</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>45696</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>0.59134</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-7025008681964841562">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3X1QlXX%2B//GXcFAOKIIhdjM2mUCWOkmi5s3aN5RcSqyUYjfG0tnMVZLcKahE13YNlRVHM4tMc/umbouabKvj7e66ebMOaLpazuCIruWuJmflJuRGgcP3j37y25O2YX4Ol%2Bc6z8eM43hd51y833Mgn5xzsHbNzc3NAgAAgDEBVg8AAABgNwQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQQWAACAYQ6rB/AXLle11SMYERDQTl26hKq8vEZud7PV43gNe9qLv%2Bwp%2Bc%2Bu7Gkv3tyza9dORq/XWjyDhWsSENBO7dq1U0BAO6tH8Sr2tBd/2VPyn13Z017suCeBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYBiBBQAAYJjD6gGAG1V89larR7gmW6YPtXoEAMD/wzNYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhvlsYB09elRpaWmKj4/XsGHD9Prrr%2BvSpUuSpE8%2B%2BUTJycnq16%2BfkpKStHPnTo/7Ll%2B%2BXMOHD1e/fv00fvx4nTx5suVcbW2tXn31VQ0aNEj9%2B/dXVlaWampq2nQ3AADg23wysNxutyZPnqxRo0apuLhY69ev1549e7R8%2BXKdOnVK06ZN0wsvvKADBw5o2rRpmj59us6dOydJKiws1KpVq/Tee%2B%2BpqKhIvXv3VkZGhpqbmyVJc%2BbM0dmzZ7Vt2zZt375dZ8%2BeVV5enpXrAgAAH%2BOTgVVVVSWXyyW3290SRgEBAXI6nSosLFR8fLxGjhwph8Ohhx9%2BWAMGDFBBQYEkae3atXrqqacUExOjDh066MUXX9SZM2dUVFSkuro6bdy4URkZGQoPD9dNN92kl156SRs2bFBdXZ2VKwMAAB/isHqAHyIiIkITJkxQbm6ufvOb36ipqUkjRozQhAkTNG3aNMXGxnrcPjo6WiUlJZKk0tJSTZo0qeVcUFCQ7rjjDpWUlCg8PFwNDQ0e9%2B/Zs6fq6%2Bt16tQp3X333a2ar6ysTC6Xy%2BOYwxGiqKioH7ryDSMwMMDjd7vyxf0cjmuf2d8eT7vvKfnPruxpL3bc0ycDy%2B12Kzg4WLNmzVJKSoq%2B%2BOILPf/881qyZIlqamrkdDo9bh8cHKza2lpJ%2Bq/nL1y4IEkKCQlpOXf5ttfyPqyCggItXbrU41h6eroyMjJav%2BQNLizM%2Bf03QpuKiAj9wff1l8fTX/aU/GdX9rQXO%2B3pk4G1Y8cObdu2TVu3bpUkxcTEKD09XTk5ObrvvvtUX1/vcfv6%2BnqFhn7zl4/T6fzO85fDqq6uruX2l18a7NixY6vnS01NVUJCgscxhyNEFRW%2B/2b5wMAAhYU59fXXdWpqcls9jtf44ndRP%2BTzy58eT3/YU/KfXdnTXry55/V883k9fDKwzp492/ITg5c5HA4FBQUpNjZWR48e9ThXWlqqPn36SPomxo4fP64HH3xQktTQ0KBTp04pNjZWPXr0UFBQkEpLS3XvvfdKkk6cONHyMmJrRUVFXfFyoMtVrcZG%2B3xxNDW5bbWPHVzP4%2BEvj6e/7Cn5z67saS922tP3vk2XNGzYMLlcLr3zzjtqamrS6dOnlZ%2Bfr%2BTkZI0ZM0bFxcXavHmzGhsbtXnzZhUXF%2BvRRx%2BVJI0bN06rV69WSUmJLl68qIULFyoyMlLx8fFyOp1KSkpSXl6eysvLVV5erry8PI0ePVrBwcEWbw0AAHyFTz6DFR0drWXLlmnx4sVasWKFOnXqpDFjxig9PV3t27fXW2%2B9pby8PGVnZ%2Bu2227Tm2%2B%2BqR49ekiSUlJSVF1drfT0dJWXl6tv375atmyZgoKCJEmzZ89Wbm6ukpOT1dDQoBEjRmjWrFlWrgsAAHxMu%2BbL/84BvMrlqrZ6BCMcjgBFRISqoqLGNk/jXo3DEaDEvN1Wj3FNtkwfes338afH0x/2lPxnV/a0F2/u2bVrJ6PXay2ffIkQAADgRkZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGOazgVVZWamsrCwNGjRIAwYM0NSpU1VWViZJOnz4sJ544gnFxcUpISFB69at87hvYWGhEhMT1a9fP40dO1aHDh1qOdfU1KTc3FwNGTJEcXFxmjJlSst1AQAAWsNnA2vatGmqra3Vjh07tHPnTgUGBmrWrFmqqqrSc889p8cee0z79%2B9XTk6O5s2bpyNHjkiSioqKNGfOHM2fP1/79%2B/XmDFjNGXKFNXV1UmS8vPztXfvXn300UfavXu3goODNXPmTCtXBQAAPsYnA%2Bvzzz/X4cOHNX/%2BfIWFhaljx46aM2eOXnrpJW3fvl3h4eFKS0uTw%2BHQ4MGDlZycrDVr1kiS1q1bp0ceeUT9%2B/dXUFCQJkyYoIiICG3evLnl/KRJk3TLLbeoY8eOys7O1q5du3T69GkrVwYAAD7EJwPryJEjio6O1tq1a5WYmKhhw4YpNzdXXbt21fHjxxUbG%2Btx%2B%2BjoaJWUlEiSSktLv/N8dXW1vvrqK4/zkZGR6ty5s44dO%2Bb9xQAAgC04rB7gh6iqqtKxY8fUp08fFRYWqr6%2BXllZWXr55ZcVGRkpp9Ppcfvg4GDV1tZKkmpqar7zfE1NjSQpJCTkivOXz7VGWVmZXC6XxzGHI0RRUVGtvsaNKjAwwON3u/LF/RyOa5/Z3x5Pu%2B8p%2Bc%2Bu7GkvdtzTJwOrffv2kqTs7Gx16NBBHTt21PTp0/Xkk09q7Nixqq%2Bv97h9fX29QkNDJUlOp/Oq5yMiIlrC6/L7sa52/9YoKCjQ0qVLPY6lp6crIyOj1de40YWFOb//RmhTERGt/xz9Nn95PP1lT8l/dmVPe7HTnj4ZWNHR0XK73WpoaFCHDh0kSW63W5J0991363e/%2B53H7UtLSxUTEyNJiomJ0fHjx684P3z4cHXu3FndunXzeBnR5XKpsrLyipcV/5vU1FQlJCR4HHM4QlRR0fpnwW5UgYEBCgtz6uuv69TU5LZ6HK/xxe%2Bifsjnlz89nv6wp%2BQ/u7KnvXhzz%2Bv55vN6%2BGRgDRkyRN27d9eMGTM0b948Xbx4UYsWLdLIkSM1evRoLVmyRO%2B//77S0tL06aefauPGjXr77bclSSkpKUpPT1dSUpL69%2B%2BvNWvW6Pz580pMTJQkjR07Vvn5%2Berbt68iIiI0d%2B5cDRw4ULfffnur54uKirri5UCXq1qNjfb54mhqcttqHzu4nsfDXx5Pf9lT8p9d2dNe7LSnTwZWUFCQVq1apfnz52vUqFG6ePGiEhISlJ2drbCwMK1cuVI5OTlasmSJunTpopkzZ%2Br%2B%2B%2B%2BXJA0ePFizZ8/Wa6%2B9pnPnzik6OlrLly9XeHi4pG9eymtsbFRaWppqamo0aNAgLV682Mp1AQCAj2nX3NzcbPUQ/sDlqrZ6BCMcjgBFRISqoqLGNt9lXI3DEaDEvN1Wj3FNtkwfes338afH0x/2lPxnV/a0F2/u2bVrJ6PXay3fe6MJAADADY7AAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMIzAAgAAMMxh9QDwH0mL91o9AgAAbYJnsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAwjsAAAAAzz%2BcBqamrS%2BPHj9corr7Qc%2B%2BSTT5ScnKx%2B/fopKSlJO3fu9LjP8uXLNXz4cPXr10/jx4/XyZMnW87V1tbq1Vdf1aBBg9S/f39lZWWppqamzfYBAAC%2Bz7LAunTpkpHrLF26VAcOHGj586lTpzRt2jS98MILOnDggKZNm6bp06fr3LlzkqTCwkKtWrVK7733noqKitS7d29lZGSoublZkjRnzhydPXtW27Zt0/bt23X27Fnl5eUZmRUAAPiHNg%2BsDz/8UAkJCerXr59Onz6t2bNna%2BnSpT/oWvv27dP27dv10EMPtRwrLCxUfHy8Ro4cKYfDoYcfflgDBgxQQUGBJGnt2rV66qmnFBMTow4dOujFF1/UmTNnVFRUpLq6Om3cuFEZGRkKDw/XTTfdpJdeekkbNmxQXV2dkf0BAID9tWlgbdy4UQsXLtTjjz%2BuoKAgSVLPnj317rvvavny5dd0rfPnzys7O1sLFy6U0%2BlsOV5aWqrY2FiP20ZHR6ukpOSq54OCgnTHHXeopKREX3zxhRoaGjzO9%2BzZU/X19Tp16tS1rgsAAPyUoy0/2MqVK5Wdna3HH39cK1eulCQ9/fTT6tSpk/Lz8zVp0qRWXcftdiszM1MTJ05Ur169PM7V1NR4BJckBQcHq7a29nvPX7hwQZIUEhLScu7yba/lfVhlZWVyuVwexxyOEEVFRbX6GjeqwMAAj99x43A4rv0x8ZfH01/2lPxnV/a0Fzvu2aaB9Y9//EPx8fFXHI%2BPj9dXX33V6ussW7ZM7du31/jx468453Q6VV9f73Gsvr5eoaGh33v%2BcljV1dW13P7yS4MdO3Zs9XwFBQVXvOyZnp6ujIyMVl/jRhcW5vz%2BG6FNRUSE/uD7%2Bsvj6S97Sv6zK3vai532bNPAioyM1MmTJ9W9e3eP4wcPHrymZ3c%2B/vhjlZWVtcTa5WD605/%2BpLS0NB09etTj9qWlperTp48kKSYmRsePH9eDDz4oSWpoaNCpU6cUGxurHj16KCgoSKWlpbr33nslSSdOnGh5GbG1UlNTlZCQ4HHM4QhRRYXv/zRiYGCAwsKc%2BvrrOjU1ua0eB//hh3x%2B%2Bcvj6S97Sv6zK3vaizf3vJ5vPq9HmwZWamqqfvWrX7X8kwonT57U7t279cYbb2jChAmtvs7WrVs9/nz5evPnz9eJEyf029/%2BVps3b9ZDDz2k7du3q7i4WNnZ2ZKkcePG6c0339Tw4cPVo0cPLVq0SJGRkYqPj1dQUJCSkpKUl5enN954Q5KUl5en0aNHKzg4uNXzRUVFXRGMLle1GhvNf3EkLd5r/JrwTdfz%2BdXU5PbK5%2BeNxl/2lPxnV/a0Fzvt2aaBNWnSJFVXVyszM1MXL17U5MmT5XA49JOf/ESTJ0828jF69uypt956S3l5ecrOztZtt92mN998Uz169JAkpaSkqLq6Wunp6SovL1ffvn21bNmyljfdz549W7m5uUpOTlZDQ4NGjBihWbNmGZkNAAD4h3bNl/8BqDZUV1en0tJSNTc3684777ym9zf5Kper2ivX5RksXLZl%2BtBrvo/DEaCIiFBVVNTY5rvGq/GXPSX/2ZU97cWbe3bt2sno9VqrTZ/BuszpdKpv375WfGgAAACvs8/PQwIAANwgCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDvB5Y8%2BbNU21trbc/DAAAwA3D64H1wQcfqK6uzuPYz372M5WVlXn7QwMAAFjC64F1tf/V4cGDB3Xx4kVvf2gAAABL8B4sAAAAwwgsAAAAw9oksNq1a9cWHwYAAOCG4GiLD/L666%2BrQ4cOLX9uaGjQggULFBoa6nG7efPmtcU4AAAAXuX1wBowYIBcLpfHsbi4OFVUVKiiosLbHx4AAKDNeT2wVq1a5e0PAQAAcEPhTe4AAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGEVgAAACGOaweAIAZSYv3Wj1Cq22ZPtTqEQDAq3gGCwAAwDACCwAAwDACCwAAwDACCwAAwDACCwAAwDACCwAAwDACCwAAwDACCwAAwDACCwAAwDACCwAAwDACCwAAwDCfDaySkhJNnDhRAwcO1NChQ5WVlaXy8nJJ0uHDh/XEE08oLi5OCQkJWrduncd9CwsLlZiYqH79%2Bmns2LE6dOhQy7mmpibl5uZqyJAhiouL05QpU1RWVtamuwEAAN/mk4FVX1%2BvZ599VnFxcdqzZ482bdqkyspKzZgxQ1VVVXruuef02GOPaf/%2B/crJydG8efN05MgRSVJRUZHmzJmj%2BfPna//%2B/RozZoymTJmiuro6SVJ%2Bfr727t2rjz76SLt371ZwcLBmzpxp5boAAMDH%2BGRgnTlzRr169VJ6errat2%2BviIgIpaamav/%2B/dq%2BfbvCw8OVlpYmh8OhwYMHKzk5WWvWrJEkrVu3To888oj69%2B%2BvoKAgTZgwQREREdq8eXPL%2BUmTJumWW25Rx44dlZ2drV27dun06dNWrgwAAHyITwbWnXfeqRUrVigwMLDl2LZt29S7d28dP35csbGxHrePjo5WSUmJJKm0tPQ7z1dXV%2Burr77yOB8ZGanOnTvr2LFjXtwIAADYicPqAa5Xc3OzFi9erJ07d2r16tX64IMP5HQ6PW4THBys2tpaSVJNTc13nq%2BpqZEkhYSEXHH%2B8rnWKCsrk8vl8jjmcIQoKiqq1dcA7MzhaNvv7QIDAzx%2BtzN/2ZU97cWOe/p0YF24cEGvvvqqjh49qtWrV%2Buuu%2B6S0%2BlUdXW1x%2B3q6%2BsVGhoqSXI6naqvr7/ifEREREt4XX4/1tXu3xoFBQVaunSpx7H09HRlZGS0%2BhqAnUVEtP7ryaSwMOf338gm/GVX9rQXO%2B3ps4H15ZdfatKkSbr11lu1fv16denSRZIUGxurvXv3ety2tLRUMTExkqSYmBgdP378ivPDhw9X586d1a1bN4%2BXEV0ulyorK694WfG/SU1NVUJCgscxhyNEFRWtfxYMsLO2/loIDAxQWJhTX39dp6Ymd5t%2B7LbmL7uyp714c0%2BrvqHzycCqqqrSM888o/vvv185OTkKCPj/TykmJiZqwYIFev/995WWlqZPP/1UGzdu1Ntvvy1JSklJUXp6upKSktS/f3%2BtWbNG58%2BfV2JioiRp7Nixys/PV9%2B%2BfRUREaG5c%2Bdq4MCBuv3221s9X1RU1BUvB7pc1WpstO8XB3AtrPpaaGpy%2B83Xob/syp72Yqc9fTKwNmzYoDNnzmjLli3aunWrx7lDhw5p5cqVysnJ0ZIlS9SlSxfNnDlT999/vyRp8ODBmj17tl577TWdO3dO0dHRWr58ucLDwyV981JeY2Oj0tLSVFNTo0GDBmnx4sVtviMAAPBd7Zqbm5utHsIfuFzV33%2BjHyBp8d7vvxFwg9kyfWibfjyHI0AREaGqqKixzXfH38VfdmVPe/Hmnl27djJ6vdayz9v1AQAAbhAEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEOqwcA4H%2BSFu%2B1eoRrsmX6UKtHAOBjeAYLAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMAILAADAMIfVAwDAjS5p8V6rR7gmB3J%2BbPUIgN/jGSwAAADDCKyrOH/%2BvKZOnar4%2BHgNGjRIOTk5amxstHosAADgIwisq5g%2BfbpCQkK0e/durV%2B/Xvv27dP7779v9VgAAMBHEFjf8sUXX6i4uFiZmZlyOp3q3r27pk6dqjVr1lg9GgAA8BEE1rccP35c4eHh6tatW8uxnj176syZM/r6668tnAwAAPgKforwW2pqauR0Oj2OXf5zbW2twsLCvvcaZWVlcrlcHsccjhBFRUWZGxQAvkN89larR2i1HS/96AfdLzAwwON3u2JP30VgfUtISIjq6uo8jl3%2Bc2hoaKuuUVBQoKVLl3oce/755zVt2jQzQ/6Htv5x7LKyMhUUFCg1NdXWwcie9uIve0r%2Bs2tZWZn%2B939XsKdN2HFP%2B6SiITExMaqsrNS///3vlmMnTpzQzTffrE6dOrXqGqmpqdqwYYPHr9TUVG%2BN3KZcLpeWLl16xTN0dsOe9uIve0r%2Bsyt72osd9%2BQZrG%2B544471L9/f82dO1e//vWvVVFRobffflspKSmtvkZUVJRtChwAAFw7nsG6iiVLlqixsVEjRozQk08%2BqR/96EeaOnWq1WMBAAAfwTNYVxEZGaklS5ZYPQYAAPBRga%2B99tprVg8B3xIaGqqBAwe2%2Bk3/voo97cVf9pT8Z1f2tBe77dmuubm52eohAAAA7IT3YAEAABhGYAEAABhGYAEAABhGYAEAABhGYAEAABhGYAEAABhGYAEAABhGYAEAABhGYOGalZeXKzExUUVFRVaP4hUlJSWaOHGiBg4cqKFDhyorK0vl5eVWj2Xcvn379MQTT%2Bi%2B%2B%2B7T0KFDNWfOHNXX11s9ltc0NTVp/PjxeuWVV6wexSs2b96se%2B65R3FxcS2/MjMzrR7LuMrKSmVlZWnQoEEaMGCApk6dqrKyMqvHMu6Pf/yjx2MZFxenPn36qE%2BfPlaPZtzRo0eVlpam%2BPh4DRs2TK%2B//rouXbpk9VjXjcDCNfn000%2BVmpqqL7/80upRvKK%2Bvl7PPvus4uLitGfPHm3atEmVlZWaMWOG1aMZVV5ersmTJ%2BunP/2pDhw4oMLCQhUXF%2Bvdd9%2B1ejSvWbp0qQ4cOGD1GF7z2Wef6dFHH9WhQ4dafi1YsMDqsYybNm2aamtrtWPHDu3cuVOBgYGaNWuW1WMZN2bMGI/HcuvWrQoPD1dOTo7Voxnldrs1efJkjRo1SsXFxVq/fr327Nmj5cuXWz3adeN/9oxWKyws1JIlS5SZmalf/OIXVo/jFWfOnFGvXr2Unp6uwMBAtW/fXqmpqcrKyrJ6NKO6dOmiv/3tb%2BrYsaOam5tVWVmpixcvqkuXLlaP5hX79u3T9u3b9dBDD1k9itd89tlnSkpKsnoMr/r88891%2BPDhls9dSZozZ45cLpfFk3lXc3OzMjMz9T//8z969NFHrR7HqKqqKrlcLrndbl3%2BP/cFBATI6XRaPNn14xkstNqwYcO0Y8cOPfzww1aP4jV33nmnVqxYocDAwJZj27ZtU%2B/evS2cyjsu/wX1wAMPKDk5WV27dtXYsWMtnsq88%2BfPKzs7WwsXLrTFf7Svxu126%2BjRo/rrX/%2BqBx98UMOHD9esWbNUVVVl9WhGHTlyRNHR0Vq7dq0SExM1bNgw5ebmqmvXrlaP5lUff/yxSktLbfnydkREhCZMmKDc3Fz17dtXDzzwgO644w5NmDDB6tGuG4GFVuvatascDv950rO5uVmLFi3Szp07lZ2dbfU4XrN9%2B3bt2rVLAQEBysjIsHoco9xutzIzMzVx4kT16tXL6nG8pry8XPfcc49GjRqlzZs36/e//71OnTplu/dgVVVV6dixYzp16pQKCwv1hz/8QefOndPLL79s9Whe43a7lZ%2Bfr5///Oct3xTZidvtVnBwsGbNmqW///3v2rRpk06cOKElS5ZYPdp1I7CAq7hw4YIyMjK0ceNGrV69WnfddZfVI3lNcHCwunXrpsyOeWAdAAACw0lEQVTMTO3evdtWz3osW7ZM7du31/jx460exasiIyO1Zs0apaSkyOl06tZbb1VmZqZ27dqlCxcuWD2eMe3bt5ckZWdnq2PHjoqMjNT06dP1ySefqKamxuLpvKOoqEhlZWVKSUmxehSv2LFjh7Zt26annnpK7du3V0xMjNLT0/Xhhx9aPdp1I7CAb/nyyy81btw4XbhwQevXr7dlXB08eFA//vGPPX5S59KlSwoKCrLVy2gff/yxiouLFR8fr/j4eG3atEmbNm1SfHy81aMZVVJSory8vJb3sEjfPJ4BAQEtUWIH0dHRcrvdamhoaDnmdrslyWN3O9m2bZsSExMVEhJi9Shecfbs2St%2BYtDhcCgoKMiiicwhsID/UFVVpWeeeUb33Xef3nvvPdu%2B6fuuu%2B5SfX29Fi5cqEuXLulf//qXcnNzlZKSYqu/kLdu3aqDBw/qwIEDOnDggEaPHq3Ro0fb7qcJw8PDtWbNGq1YsUKNjY06c%2BaMFixYoMcff9xWj%2BeQIUPUvXt3zZgxQzU1NSovL9eiRYs0cuRIW758Jn3zk9sDBgywegyvGTZsmFwul9555x01NTXp9OnTys/PV3JystWjXTcCC/gPGzZs0JkzZ7Rlyxb179/f49%2BgsZPQ0FCtWLFCx48f19ChQzV%2B/HgNGTLEdv8chb%2B4%2BeabtWzZMv35z3/WwIEDNW7cOPXt21e//OUvrR7NqKCgIK1atUqBgYEaNWqURo0apZtvvllz5861ejSv%2Bec//6moqCirx/Ca6OhoLVu2TH/5y180aNAgPf3000pISLDFT6q3a7br86oAAAAW4RksAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAw/4PFHvfC27tlAAAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-7025008681964841562">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">2.5</td>
        <td class="number">5377</td>
        <td class="number">24.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">3851</td>
        <td class="number">17.8%</td>
        <td>
            <div class="bar" style="width:71%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.75</td>
        <td class="number">3048</td>
        <td class="number">14.1%</td>
        <td>
            <div class="bar" style="width:57%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.25</td>
        <td class="number">2047</td>
        <td class="number">9.5%</td>
        <td>
            <div class="bar" style="width:38%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.0</td>
        <td class="number">1930</td>
        <td class="number">8.9%</td>
        <td>
            <div class="bar" style="width:36%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.5</td>
        <td class="number">1445</td>
        <td class="number">6.7%</td>
        <td>
            <div class="bar" style="width:27%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.75</td>
        <td class="number">1185</td>
        <td class="number">5.5%</td>
        <td>
            <div class="bar" style="width:22%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.0</td>
        <td class="number">753</td>
        <td class="number">3.5%</td>
        <td>
            <div class="bar" style="width:14%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.5</td>
        <td class="number">731</td>
        <td class="number">3.4%</td>
        <td>
            <div class="bar" style="width:14%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.25</td>
        <td class="number">589</td>
        <td class="number">2.7%</td>
        <td>
            <div class="bar" style="width:11%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (19)</td>
        <td class="number">641</td>
        <td class="number">3.0%</td>
        <td>
            <div class="bar" style="width:12%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-7025008681964841562">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.5</td>
        <td class="number">4</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">0.75</td>
        <td class="number">71</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">3851</td>
        <td class="number">17.8%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.25</td>
        <td class="number">9</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.5</td>
        <td class="number">1445</td>
        <td class="number">6.7%</td>
        <td>
            <div class="bar" style="width:38%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">6.5</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6.75</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7.5</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7.75</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8.0</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_bedrooms">bedrooms<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>12</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.1%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>3.3732</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>1</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>33</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram1684772975517577072">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAAQlJREFUeJzt1cEJAlEQBUEVQzIIc/JsTgaxOY13kQbB5YtU3QfepZnjzMwBeOu0egD8svPqAa8ut8fHN9v9usMS8EEgCQSCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIJxXD/iGy%2B3x8c12v%2B6whH/jg0AQCASBQBAIhOPMzOoR8Kt8EAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAhPBQMOkb154NMAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives1684772975517577072,#minihistogram1684772975517577072"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives1684772975517577072">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles1684772975517577072"
                                                  aria-controls="quantiles1684772975517577072" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram1684772975517577072" aria-controls="histogram1684772975517577072"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common1684772975517577072" aria-controls="common1684772975517577072"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme1684772975517577072" aria-controls="extreme1684772975517577072"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles1684772975517577072">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>2</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>3</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>3</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>4</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>5</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>33</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>32</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>1</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>0.9263</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.27461</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>49.822</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>3.3732</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>0.73357</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>2.0236</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>72851</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>0.85803</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram1684772975517577072">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3XtUVOeh/vEHGJQBRVBEmxwbLYKpl0SqEW%2Bl9UKtUdQghlRrTVqT00gkWkUPTVKtFo1HYtRYrddYlbUkGj2pORhtelJjjUVjTWpsiGC8pRhBAeUabvP7Iz%2Bn3aIG9A3DwPez1iw777tn73c/3SvrYWYzeDgcDocAAABgjKerFwAAANDUULAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGE2Vy%2BgucjLK7rlnKenh9q29VN%2BfolqahwNuKrGiTxqIxMr8rAiDyvysGruebRv39olx%2BUdrEbA09NDHh4e8vT0cPVSGgXyqI1MrMjDijysyMOKPFyDggUAAGAYBQsAAMAwChYAAIBhjbZgZWZm6oknnlC/fv00aNAgzZkzR/n5%2BZKkDz/8UBMmTFB4eLiGDh2qHTt2WF67e/duRUVFqXfv3oqJidHx48edc9XV1VqyZIkGDhyo8PBwPf3008rNzXXOX7lyRdOmTVPfvn0VERGh5ORkVVVVNcxJAwCAJqFRFqzy8nJNnTpV4eHh%2Bstf/qI333xThYWF%2BuUvf6mrV6/qqaee0rhx43T06FElJydr8eLF%2Bvvf/y5JysjI0MKFC/Xiiy/q6NGjGjNmjJ5%2B%2BmmVlZVJktasWaNDhw7p9ddf18GDB%2BXj46Pnn3/eeewZM2bI19dXBw8e1M6dO3X48GFt3rzZFTEAAAA31SgLVk5Oju6//37Fx8erRYsWCgwMVFxcnI4ePar9%2B/crICBAkyZNks1m04ABAxQdHa3U1FRJ0o4dOzRq1Cj16dNH3t7eevzxxxUYGKj09HTn/JNPPqlvfOMbatWqlZ577jm9%2B%2B67unDhgs6dO6cjR44oMTFRdrtdnTp10rRp05z7BgAAqItGWbC%2B9a1vacOGDfLy8nKO7du3Tz169FBWVpbCwsIs23ft2lWZmZmSpOzs7FvOFxUV6fPPP7fMBwUFqU2bNvrkk0%2BUlZWlgIAAdejQwTkfEhKinJwcXbt27es4VQAA0AQ1%2Bi8adTgcWr58ud555x1t27ZNW7Zskd1ut2zj4%2BOj0tJSSVJJSckt50tKSiRJvr6%2Bteavz9342uvPS0tL5e/vX6c15%2BbmKi8vzzJms/kqODj4ptt7eXla/m3uyKM2MrEiDyvysCIPK/JwjUZdsIqLi5WUlKSTJ09q27Zt6tatm%2Bx2u4qKrN%2BKXl5eLj8/P0lfFqLy8vJa84GBgc6ydP1%2BrBtf73A4as1df359/3WRlpamVatWWcbi4%2BOVkJBw29f5%2B9tvO9/ckEdtZGJFHlbkYUUeVuTRsBptwTp//ryefPJJ3XPPPdq5c6fatm0rSQoLC9OhQ4cs22ZnZys0NFSSFBoaqqysrFrzkZGRatOmjTp06GD5GDEvL0%2BFhYUKCwtTTU2NCgsLdfnyZQUFBUmSTp8%2BrY4dO6p167p/1X5cXJyGDh1qGbPZfFVQUHLT7b28POXvb9e1a2Wqrq6p83GaKvKojUysyMOKPKzIw6q55xEYWPc3SExqlAXr6tWrmjJlivr376/k5GR5ev7rbc2oqCgtXbpUmzdv1qRJk3Ts2DHt2bNHq1evliTFxsYqPj5eI0eOVJ8%2BfZSamqorV64oKipKkhQTE6M1a9aoV69eCgwM1KJFi9SvXz9985vflCT16dNHixYt0oIFC1RQUKDVq1crNja2XusPDg6u9XFgXl6Rqqpuf2FXV9d85TbNCXnURiZW5GFFHlbkYUUeDatRFqxdu3YpJydHe/fu1VtvvWWZO378uDZt2qTk5GStXLlSbdu21fPPP6/%2B/ftLkgYMGKB58%2BZp/vz5unTpkrp27ar169crICBA0pcf1VVVVWnSpEkqKSlRRESEli9f7tz/ypUrtWDBAg0bNkyenp4aN26cpk2b1nAnDwAA3J6Hw%2BFofn9a2wXy8opuOWezeSow0E8FBSX8dCHyuBkysSIPK/KwIg%2Br5p5H%2B/Z1v8XHpEb5DhbqbuTyQ1%2B9USOxd8YgVy8BAIAGwe9sAgAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDC3LVjp6enq3r27wsPDnY/ExETn/IEDBxQdHa3evXtr5MiReueddyyvX79%2BvSIjI9W7d29NnjxZn376aUOfAgAAaKLctmCdOHFCY8eO1fHjx52PpUuXSpLOnj2r6dOn69lnn9X777%2Bv6dOna8aMGbp06ZIkaffu3dq6das2btyojIwM9ejRQwkJCXI4HK48JQAA0ES4dcHq2bPnTed2796tvn37avjw4bLZbHr44Yf10EMPKS0tTZL02muvaeLEiQoNDVXLli01a9Ys5eTkKCMjoyFPAQAANFFuWbBqamp08uRJ/fnPf9aQIUMUGRmpF154QVevXpUkZWdnKywszPKarl27KjMz86bz3t7e6ty5s3MeAADgbthcvYA7kZ%2Bfr%2B7du2vEiBFauXKlCgoKNHfuXCUmJmrdunUqKSmR3W63vMbHx0elpaWS9JXzdys3N1d5eXmWMZvNV8HBwTfd3svL0/JvU2Wz1e38mkse9UEmVuRhRR5W5GFFHq7hlgUrKChIqampzud2u12JiYl69NFHVVxcLLvdrvLycstrysvL5efn59z%2BdvN3Ky0tTatWrbKMxcfHKyEh4bav8/e333be3QUG1i/fpp7HnSATK/KwIg8r8rAij4bllgUrMzNTb775pmbNmiUPDw9JUkVFhTw9PdWiRQuFhYXp5MmTltdkZ2c779kKDQ1VVlaWhgwZIkmqrKzU2bNna32seKfi4uI0dOhQy5jN5quCgpKbbu/l5Sl/f7uuXStTdXWNkTU0Rrc6/xs1lzzqg0ysyMOKPKzIw6q551HfH%2B5NccuCFRAQoNTUVLVp00ZPPPGEcnNztXTpUj3yyCNq0aKFxowZo1dffVXp6en6wQ9%2BoP379%2BvIkSN67rnnJEnjx4/XK6%2B8osjISHXp0kUvv/yygoKC1LdvXyPrCw4OrvVxYF5ekaqqbn9hV1fXfOU27qy%2B59bU87gTZGJFHlbkYUUeVuTRsNyyYHXs2FFr167VsmXLtGbNGrVs2VKjRo1yfg9WSEiIfvvb3yolJUXPPfec7r33Xr3yyivq0qWLJCk2NlZFRUWKj49Xfn6%2BevXqpbVr18rb29uVpwUAAJoIDwdf/tQg8vKKbjlns3kqMNBPBQUl9f7pYuTyQ3e7tAazd8agOm13N3k0VWRiRR5W5GFFHlbNPY/27Vu75Lj8SgEAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMa/QFKz8/X1FRUcrIyHCOzZs3Tz179lR4eLjzkZaW5pxfv369IiMj1bt3b02ePFmffvqpc660tFRJSUmKiIhQnz59NGfOHJWUlDjnz5w5oylTpig8PFyDBw/W7373u4Y5UQAA0GQ06oJ17NgxxcXF6fz585bxEydOaOHChTp%2B/LjzERcXJ0navXu3tm7dqo0bNyojI0M9evRQQkKCHA6HJGnhwoW6ePGi9u3bp/379%2BvixYtKSUmRJFVWVurnP/%2B5evXqpYyMDK1bt06pqanau3dvw544AABwa422YO3evVuzZ8/WzJkzLeMVFRU6deqUevbsedPXvfbaa5o4caJCQ0PVsmVLzZo1Szk5OcrIyFBZWZn27NmjhIQEBQQEqF27dpo9e7Z27dqlsrIyHT16VLm5uUpISFCLFi3UvXt3TZ48WampqQ1xygAAoIlotAVr8ODB%2BuMf/6iHH37YMp6ZmamqqiqtXLlSAwcO1IgRI7Ru3TrV1NRIkrKzsxUWFubc3tvbW507d1ZmZqbOnTunyspKy3xISIjKy8t19uxZZWVlqUuXLmrRooVzvmvXrsrMzPyazxYAADQlNlcv4Fbat29/0/GioiL169dPkydP1rJly/Txxx8rPj5enp6emjp1qkpKSmS32y2v8fHxUWlpqYqLiyVJvr6%2Bzrnr25aUlNz0tXa7XaWlpfVae25urvLy8ixjNpuvgoODb7q9l5en5d%2Bmymar2/k1lzzqg0ysyMOKPKzIw4o8XKPRFqxbGTRokAYNGuR8/sADD2jKlClKT0/X1KlTZbfbVV5ebnlNeXm5/Pz8nMWqrKxMfn5%2Bzv8tSa1atZKvr6/z%2BXX/vm1dpaWladWqVZax%2BPh4JSQk3PZ1/v722867u8DA%2BuXY1PO4E2RiRR5W5GFFHlbk0bDcrmC9/fbbunz5sh577DHnWEVFhXx8fCRJoaGhysrK0pAhQyR9eeP62bNnFRYWpi5dusjb21vZ2dl68MEHJUmnT592fox45coVnT17VlVVVbLZvowmOztboaGh9VpjXFychg4dahmz2XxVUFBy0%2B29vDzl72/XtWtlqq6uqdex3Mmtzv9GzSWP%2BiATK/KwIg8r8rBq7nnU94d7U9yuYDkcDi1evFj33Xef%2Bvfvrw8%2B%2BEBbtmxRUlKSJGn8%2BPF65ZVXFBkZqS5duujll19WUFCQ%2BvbtK29vb40cOVIpKSlasWKFJCklJUWjR4%2BWj4%2BPIiIiFBgYqJdeekkzZszQmTNntHXr1lo32n%2BV4ODgWh8H5uUVqarq9hd2dXXNV27jzup7bk09jztBJlbkYUUeVuRhRR4Ny%2B0KVlRUlJKSkjR//nxdunRJQUFBmj59usaOHStJio2NVVFRkeLj45Wfn69evXpp7dq18vb2lvTld2gtWbJE0dHRqqys1LBhw/TCCy9Ikmw2mzZt2qQFCxZo0KBB8vX11eTJkxUTE%2BOy8wUAAO7Hw3H9C6LwtcrLK7rlnM3mqcBAPxUUlNT7p4uRyw/d7dIazN4Zg756I91dHk0VmViRhxV5WJGHVXPPo3371i45Lr9SAAAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAxr9AUrPz9fUVFRysjIcI7t27dPY8eO1Xe%2B8x0NHTpUq1atUk1NjXN%2B5MiRevDBBxUeHu58nD59WpJUWlqqpKQkRUREqE%2BfPpozZ45KSkqcrz1z5oymTJmi8PBwDR48WL/73e8a7mQBAECT0KgL1rFjxxQXF6fz5887xz766CPNmTNHM2bM0Pvvv6/169dr165d2rx5sySpuLhYZ86cUXp6uo4fP%2B58hISESJIWLlyoixcvat%2B%2Bfdq/f78uXryolJQUSVJlZaV%2B/vOfq1evXsrIyNC6deuUmpqqvXv3Nvi5AwAA99VoC9bu3bs1e/ZszZw50zL%2Bz3/%2BU4899piGDBkiT09PhYSEKCoqSkePHpX0ZQELCAjQvffeW2ufZWVl2rNnjxISEhQQEKB27dpp9uzZ2rVrl8rKynT06FHl5uYqISFBLVq0UPfu3TV58mSlpqY2yDkDAICmwebqBdzK4MGDFR0dLZvNZilZI0aM0IgRI5zPy8vL9ec//1nR0dGSpBMnTshut%2BvHP/6xsrKydO%2B992r69OkaMmSIzp07p8rKSoWFhTlfHxISovLycp09e1ZZWVnq0qWLWrRo4Zzv2rWr1q1bV6%2B15%2BbmKi8vzzJms/kqODj4ptt7eXla/m2qbLa6nV9zyaM%2ByMSKPKzIw4o8rMjDNYwXrOrqanl5ed31ftq3b/%2BV2xQXF%2BvZZ5%2BVj4%2BPHn/8cUmSh4eHevXqpV/84he655579NZbb2n69Onatm2bqqqqJEm%2Bvr7OfdjtdklSSUmJSkpKnM//fb60tLRea09LS9OqVassY/Hx8UpISLjt6/z97bedd3eBgX712r6p53EnyMSKPKzIw4o8rMijYRkvWJGRkRo7dqxiYmLUtWtX07t3%2BvTTT5WQkKB27dppy5YtatWqlSRp6tSplu3GjBmjN998U/v27XO%2By1VWViY/Pz/n/5akVq1aydfX1/n8un/ftq7i4uI0dOhQy5jN5quCgpKbbu/l5Sl/f7uuXStTdXXNTbdpCm51/jdqLnnUB5lYkYcVeViRh1Vzz6O%2BP9ybYrxgPfPMM3rjjTe0adMm9erVS%2BPHj9eoUaPUunVrY8c4cOCAfvGLX%2BjRRx/VrFmzZLP96zQ2btyo7t27a8CAAc6xiooKtWzZUl26dJG3t7eys7P14IMPSpJOnz4tb29vde7cWVeuXNHZs2dVVVXl3Gd2drZCQ0Prtb7g4OBaHwfm5RWpqur2F3Z1dc1XbuPO6ntuTT2PO0EmVuRhRR5W5GFFHg3L%2BAeyP/rRj7R9%2B3a99dZbGjhwoNavX6/Bgwdr1qxZeu%2B99%2B56/x988IHi4%2BOVlJSkuXPnWsqVJF28eFG//vWvdeHCBVVVVWnnzp06fvy4HnnkEdntdo0cOVIpKSnKz89Xfn6%2BUlJSNHr0aPn4%2BCgiIkKBgYF66aWX9MUXXygzM1Nbt25VbGzsXa8bAAA0H1/bHW%2BdO3fWzJkz9dZbbyk%2BPl5/%2BtOf9LOf/UxDhw7Vq6%2B%2Bqurq6jva7%2B9%2B9ztVVVUpOTnZ8j1X1z8anDNnjiIjIzVx4kT17dtX27dv17p163TfffdJkubNm6fOnTsrOjpaP/zhD/Uf//Ef%2BtWvfiVJstls2rRpk06dOqVBgwbpqaee0uTJkxUTE2MmFAAA0Cx4OBwOx9ex4w8//FD/8z//o/T0dFVUVGj48OGKiYnRpUuXtGLFCoWHh2vZsmVfx6Ebpby8olvO2WyeCgz0U0FBSb3fvh25/NDdLq3B7J0xqE7b3U0eTRWZWJGHFXlYkYdVc8%2BjfXtztyjVh/F7sFavXq033nhD586dU69evTRz5kyNHj3aeRO6JHl5eTnfNQIAAGhqjBesbdu2acyYMYqNjb3lbxGGhIRo9uzZpg8NAADQKBgvWO%2B%2B%2B66Ki4tVWFjoHEtPT9eAAQMUGBgoSerevbu6d%2B9u%2BtAAAACNgvGb3P/xj39oxIgRSktLc44tXbpU0dHROnXqlOnDAQAANDrGC9Z///d/6wc/%2BIHlz9u8/fbbioyM1Isvvmj6cAAAAI2O8YJ18uRJPfXUU5a/5%2Bfl5aWnnnpKH3zwgenDAQAANDrGC1arVq10/vz5WuOff/65fHx8TB8OAACg0TFesEaMGKH58%2BfrvffeU3FxsUpKSvTXv/5VCxYsUFRUlOnDAQAANDrGf4tw1qxZunDhgn7605/Kw8PDOR4VFaU5c%2BaYPhwAAECjY7xg2e12rV27VmfOnNEnn3wib29vhYSEqHPnzqYPBQAA0CgZL1jXdenSRV26dPm6dg8AANBoGS9YZ86c0YIFC3Ts2DFVVlbWmv/4449NHxIAAKBRMV6w5s%2Bfr5ycHM2ePVutW7vmDywCAAC4kvGCdfz4cf3%2B979XeHi46V0DAAC4BeNf0xAYGCg/Pz/TuwUAAHAbxgvW5MmTtWzZMhUVFZneNQAAgFsw/hHhgQMH9MEHHygiIkLt2rWz/MkcSfrTn/5k%2BpAAAACNivGCFRERoYiICNO7BQAAcBvGC9YzzzxjepcAAABuxfg9WJKUmZmppKQkPfbYY7p06ZJSU1OVkZHxdRwKAACg0TFesD766CNNmDBBn332mT766CNVVFTo448/1k9/%2BlO98847pg8HAADQ6BgvWCkpKfrpT3%2BqrVu3ytvbW5L0m9/8Rj/5yU%2B0atUq04cDAABodL6Wd7DGjRtXa/xHP/qRPv30U9OHAwAAaHSMFyxvb28VFxfXGs/JyZHdbjd9OAAAgEbHeMEaPny4XnrpJRUUFDjHTp8%2BreTkZH3/%2B983fTgAAIBGx3jBmjt3rsrLyzVw4ECVlZUpJiZGo0ePls1m05w5c0wfDgAAoNEx/j1YrVq10vbt23X48GH94x//UE1NjcLCwvTd735Xnp5fy7dCAAAANCrGC9Z1AwYM0IABA76u3QMAADRaxt9SGjp0qIYNG3bLR33l5%2BcrKirK8kWlH374oSZMmKDw8HANHTpUO3bssLxm9%2B7dioqKUu/evRUTE6Pjx48756qrq7VkyRINHDhQ4eHhevrpp5Wbm%2Bucv3LliqZNm6a%2BffsqIiJCycnJqqqquoMkAABAc2W8YD3yyCOWx%2BjRo9WrVy8VFBRoypQp9drXsWPHFBcXp/PnzzvHrl69qqeeekrjxo3T0aNHlZycrMWLF%2Bvvf/%2B7JCkjI0MLFy7Uiy%2B%2BqKNHj2rMmDF6%2BumnVVZWJklas2aNDh06pNdff10HDx6Uj4%2BPnn/%2Beef%2BZ8yYIV9fXx08eFA7d%2B7U4cOHtXnz5rsPBgAANBvGPyKcPn36Tce3bdumY8eO6Sc/%2BUmd9rN7926tXLlSiYmJmjlzpnN8//79CggI0KRJkyR9%2BVFkdHS0UlNT9cADD2jHjh0aNWqU%2BvTpI0l6/PHHlZaWpvT0dI0fP147duzQ7Nmz9Y1vfEOS9Nxzz2nw4MG6cOGCampqdOTIEb377ruy2%2B3q1KmTpk2bpqVLl2rq1Kl3EwsAAGhGGuyu8yFDhujAgQN13n7w4MH64x//qIcfftgynpWVpbCwMMtY165dlZmZKUnKzs6%2B5XxRUZE%2B//xzy3xQUJDatGmjTz75RFlZWQoICFCHDh2c8yEhIcrJydG1a9fqvHYAANC8fW03ud/oyJEjatmyZZ23b9%2B%2B/U3HS0pKan1hqY%2BPj0pLS79yvqSkRJLk6%2Btba/763I2vvf68tLRU/v7%2BdVp7bm6u8vLyLGM2m6%2BCg4Nvur2Xl6fl36bKZqvb%2BTWXPOqDTKzIw4o8rMjDijxcw3jBuvEjQIfDoeLiYn3yySd1/njwdux2u4qKiixj5eXl8vPzc86Xl5fXmg8MDHSWpev3Y934eofDUWvu%2BvPr%2B6%2BLtLS0Wn93MT4%2BXgkJCbd9nb9/0/6m%2B8DAumcoNf087gSZWJGHFXlYkYcVeTQs4wXrnnvukYeHh2XM29tbU6ZMUXR09F3vPywsTIcOHbKMZWdnKzQ0VJIUGhqqrKysWvORkZFq06aNOnToYPkYMS8vT4WFhQoLC1NNTY0KCwt1%2BfJlBQUFSfryW%2Bg7duyo1q1b13mNcXFxGjp0qGXMZvNVQUHJTbf38vKUv79d166Vqbq6ps7HcTe3Ov8bNZc86oNMrMjDijysyMOquedR3x/uTTFesF588UXTu7SIiorS0qVLtXnzZk2aNEnHjh3Tnj17tHr1aklSbGys4uPjNXLkSPXp00epqam6cuWKoqKiJEkxMTFas2aNevXqpcDAQC1atEj9%2BvXTN7/5TUlSnz59tGjRIi1YsEAFBQVavXq1YmNj67XG4ODgWh8H5uUVqarq9hd2dXXNV27jzup7bk09jztBJlbkYUUeVuRhRR4Ny3jBOnr0aJ23feihh%2Bq9/8DAQG3atEnJyclauXKl2rZtq%2Beff179%2B/eX9OVvFc6bN0/z58/XpUuX1LVrV61fv14BAQGSvvyorqqqSpMmTVJJSYkiIiK0fPly5/5XrlypBQsWaNiwYfL09NS4ceM0bdq0eq8TAAA0Xx4Oh8Nhcoc9evSQw%2BFwPpwH%2Bv8fG14f8/Dw0Mcff2zy0I1aXl7RLedsNk8FBvqpoKCk3j9djFx%2B6Ks3aiT2zhhUp%2B3uJo%2BmikysyMOKPKzIw6q559G%2Bfd1v8THJ%2BDtYr7zyihYvXqy5c%2Beqf//%2B8vb21ocffqj58%2Bdr4sSJGjJkiOlDAgAANCrGf2dzyZIlmjdvnoYPH65WrVqpZcuW6tevnxYsWKBNmzbp3nvvdT4AAACaIuMFKzc31/kt6f%2BuVatWKigoMH04AACARsd4werdu7eWLVum4uJi51hhYaGWLl2qAQMGmD4cAABAo2P8Hqznn39eU6ZMUWRkpDp37ixJOnPmjNq3b68tW7aYPhwAAECjY7xghYSEKD09XXv27NHp06clSRMnTtSoUaNq/RkaAACApuhr%2BVuE/v7%2BmjBhgj777DN16tRJ0pff5g4AANAcGL8Hy%2BFwKCUlRQ899JBGjx6tzz//XHPnzlVSUpIqKytNHw4AAKDRMV6wtm7dqjfeeEPz5s1TixYtJEnDhw/X//3f/2nFihWmDwcAANDoGC9YaWlp%2BtWvfqWYmBjnt7c//PDDSk5O1v/%2B7/%2BaPhwAAECjY7xgffbZZ/r2t79da7xbt266fPmy6cMBAAA0OsYL1r333qu///3vtcYPHDjgvOEdAACgKTP%2BW4Q/%2B9nP9Otf/1qXLl2Sw%2BHQ4cOHtX37dm3dulVJSUmmDwcAANB8f05fAAAYdklEQVToGC9Y48ePV1VVldasWaPy8nL96le/Urt27TRz5kz96Ec/Mn04AACARsd4wfrDH/6gH/7wh4qLi1N%2Bfr4cDofatWtn%2BjAAAACNlvF7sH7zm984b2Zv27Yt5QoAADQ7xgtW586d9cknn5jeLQAAgNsw/hFhaGioZs%2BerQ0bNqhz585q2bKlZX7x4sWmDwkAANCoGC9Y58%2BfV58%2BfSRJeXl5pncPAADQ6BkpWIsXL9azzz4rX19fbd261cQuAQAA3JaRe7C2bNmisrIyy9jPfvYz5ebmmtg9AACAWzFSsBwOR62xv/3tb/riiy9M7B4AAMCtGP8tQgAAgOaOggUAAGCYsYLl4eFhalcAAABuzdjXNPzmN7%2BxfOdVZWWlli5dKj8/P8t2fA8WAABo6owUrIceeqjWd16Fh4eroKBABQUFJg4BAADgNowULL77CgAA4F%2B4yR0AAMAw438qpyH84Q9/0Lx58yxjlZWVkqSPPvpIU6dOVUZGhmy2f53eihUrFBkZqerqaqWkpOiNN95QWVmZ%2Bvfvr1//%2BtcKDg6WJF25ckUvvPCCjhw5Ii8vL40ZM0Zz58617AsAAOB23PIdrDFjxuj48ePOx1tvvaWAgAAlJydL%2BrJkbdy40bJNZGSkJGnNmjU6dOiQXn/9dR08eFA%2BPj56/vnnnfueMWOGfH19dfDgQe3cuVOHDx/W5s2bXXGaAADATbllwfp3DodDiYmJ%2Bv73v6%2BxY8fqwoULunr1qrp3737T7Xfs2KEnn3xS3/jGN9SqVSs999xzevfdd3XhwgWdO3dOR44cUWJioux2uzp16qRp06YpNTW1gc8KAAC4M7f/3OuNN95Qdna2Vq9eLUk6ceKE/Pz8NHPmTJ04cUJBQUF6/PHHFRsbq6KiIn3%2B%2BecKCwtzvj4oKEht2rTRJ598IkkKCAhQhw4dnPMhISHKycnRtWvX5O/v37AnBwAA3JJbF6yamhqtWbNGP//5z9WqVStJUkVFhXr37q2ZM2cqNDRUGRkZmj59uvz8/BQeHi5J8vX1tezHx8dHJSUlkiS73W6Zu/68tLS0zgUrNze31tdW2Gy%2Bzvu8buTl5Wn5t6my2ep2fs0lj/ogEyvysCIPK/KwIg/XcOuClZGRodzcXMXGxjrHxo0bp3HjxjmfDx48WOPGjdPevXs1cOBASVJZWZllP%2BXl5fLz85PD4ag1d/35jV%2BYejtpaWlatWqVZSw%2BPl4JCQm3fZ2/v/228%2B4uMLDuGUpNP487QSZW5GFFHlbkYUUeDcutC9a%2BffsUFRVleUdq586d8vPz08iRI51jFRUVatmypdq0aaMOHTooOzvb%2BTFhXl6eCgsLFRYWppqaGhUWFury5csKCgqSJJ0%2BfVodO3ZU69at67yuuLg4DR061DJms/mqoKDkptt7eXnK39%2Bua9fKVF1dU%2BfjuJtbnf%2BNmkse9UEmVuRhRR5W5GHV3POo7w/3prh1wTp27Jh%2B8pOfWMaKi4u1bNky3Xfffbr//vv17rvv6s0339TGjRslSTExMVqzZo169eqlwMBALVq0SP369dM3v/lNSVKfPn20aNEiLViwQAUFBVq9erXlHbK6CA4OrvVxYF5ekaqqbn9hV1fXfOU27qy%2B59bU87gTZGJFHlbkYUUeVuTRsNy6YH322We1isyUKVNUWlqqZ555RleuXFGnTp20ZMkS9e3bV9KXH9VVVVVp0qRJKikpUUREhJYvX%2B58/cqVK7VgwQINGzZMnp6eGjdunKZNm9ag5wUAANybh8PhcLh6Ec1BXl7RLedsNk8FBvqpoKCk3j9djFx%2B6G6X1mD2zhhUp%2B3uJo%2BmikysyMOKPKzIw6q559G%2Bfd1v8TGJXykAAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAw9yqYKWnp6t79%2B4KDw93PhITEyVJBw4cUHR0tHr37q2RI0fqnXfesbx2/fr1ioyMVO/evTV58mR9%2BumnzrnS0lIlJSUpIiJCffr00Zw5c1RSUtKg5wYAAJoOtypYJ06c0NixY3X8%2BHHnY%2BnSpTp79qymT5%2BuZ599Vu%2B//76mT5%2BuGTNm6NKlS5Kk3bt3a%2BvWrdq4caMyMjLUo0cPJSQkyOFwSJIWLlyoixcvat%2B%2Bfdq/f78uXryolJQUV54qAABwY25XsHr27FlrfPfu3erbt6%2BGDx8um82mhx9%2BWA899JDS0tIkSa%2B99pomTpyo0NBQtWzZUrNmzVJOTo4yMjJUVlamPXv2KCEhQQEBAWrXrp1mz56tXbt2qaysrKFPEQAANAE2Vy%2BgrmpqanTy5EnZ7XZt2LBB1dXV%2Bt73vqfZs2crOztbYWFhlu27du2qzMxMSVJ2draefPJJ55y3t7c6d%2B6szMxMBQQEqLKy0vL6kJAQlZeX6%2BzZs/r2t79d77Xm5uYqLy/PMmaz%2BSo4OPim23t5eVr%2BbapstrqdX3PJoz7IxIo8rMjDijysyMM13KZg5efnq3v37hoxYoRWrlypgoICzZ07V4mJiaqoqJDdbrds7%2BPjo9LSUklSSUnJLeeLi4slSb6%2Bvs6569ve6X1YaWlpWrVqlWUsPj5eCQkJt32dv7/9tvPuLjDQr17bN/U87gSZWJGHFXlYkYcVeTQstylYQUFBSk1NdT632%2B1KTEzUo48%2BqoiICJWXl1u2Ly8vl5%2Bfn3PbW81fL1ZlZWXO7a9/NNiqVas7WmtcXJyGDh1qGbPZfFVQcPPC5uXlKX9/u65dK1N1dc0dHdMd3Or8b9Rc8qgPMrEiDyvysCIPq%2BaeR31/uDfFbQpWZmam3nzzTc2aNUseHh6SpIqKCnl6euqBBx7Qxx9/bNk%2BOzvbeb9WaGiosrKyNGTIEElSZWWlzp49q7CwMHXp0kXe3t7Kzs7Wgw8%2BKEk6ffq082PEOxEcHFzr48C8vCJVVd3%2Bwq6urvnKbdxZfc%2BtqedxJ8jEijysyMOKPKzIo2G5zQeyAQEBSk1N1YYNG1RVVaWcnBwtXbpUjzzyiMaNG6cjR44oPT1dVVVVSk9P15EjRzR27FhJ0vjx47Vt2zZlZmbqiy%2B%2B0EsvvaSgoCD17dtXdrtdI0eOVEpKivLz85Wfn6%2BUlBSNHj1aPj4%2BLj5rAADgjtzmHayOHTtq7dq1WrZsmdasWaOWLVtq1KhRSkxMVMuWLfXb3/5WKSkpeu6553TvvffqlVdeUZcuXSRJsbGxKioqUnx8vPLz89WrVy%2BtXbtW3t7ekqR58%2BZpyZIlio6OVmVlpYYNG6YXXnjBlacLAADcmIfj%2BpdB4WuVl1d0yzmbzVOBgX4qKCip99u3I5cfutulNZi9MwbVabu7yaOpIhMr8rAiDyvysGruebRv39olx3WbjwgBAADcBQULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAw9y2YGVmZuqJJ55Qv379NGjQIM2ZM0f5%2BfmSpHnz5qlnz54KDw93PtLS0pyvXb9%2BvSIjI9W7d29NnjxZn376qXOutLRUSUlJioiIUJ8%2BfTRnzhyVlJQ0%2BPkBAAD35ZYFq7y8XFOnTlV4eLj%2B8pe/6M0331RhYaF%2B%2BctfSpJOnDihhQsX6vjx485HXFycJGn37t3aunWrNm7cqIyMDPXo0UMJCQlyOBySpIULF%2BrixYvat2%2Bf9u/fr4sXLyolJcVl5woAANyPWxasnJwc3X///YqPj1eLFi0UGBiouLg4HT16VBUVFTp16pR69ux509e%2B9tprmjhxokJDQ9WyZUvNmjVLOTk5ysjIUFlZmfbs2aOEhAQFBASoXbt2mj17tnbt2qWysrIGPksAAOCu3LJgfetb39KGDRvk5eXlHNu3b5969OihzMxMVVVVaeXKlRo4cKBGjBihdevWqaamRpKUnZ2tsLAw5%2Bu8vb3VuXNnZWZm6ty5c6qsrLTMh4SEqLy8XGfPnm2w8wMAAO7N5uoF3C2Hw6Hly5frnXfe0bZt23T58mX169dPkydP1rJly/Txxx8rPj5enp6emjp1qkpKSmS32y378PHxUWlpqYqLiyVJvr6%2Bzrnr29bnPqzc3Fzl5eVZxmw2XwUHB990ey8vT8u/TZXNVrfzay551AeZWJGHFXlYkYcVebiGWxes4uJiJSUl6eTJk9q2bZu6deumbt26adCgQc5tHnjgAU2ZMkXp6emaOnWq7Ha7ysvLLfspLy%2BXn5%2Bfs1iVlZXJz8/P%2Bb8lqVWrVnVeV1pamlatWmUZi4%2BPV0JCwm1f5%2B9vv%2B28uwsM9KvX9k09jztBJlbkYUUeVuRhRR4Ny20L1vnz5/Xkk0/qnnvu0c6dO9W2bVtJ0ttvv63Lly/rsccec25bUVEhHx8fSVJoaKiysrI0ZMgQSVJlZaXOnj2rsLAwdenSRd7e3srOztaDDz4oSTp9%2BrTzY8S6iouL09ChQy1jNpuvCgpu/i6Yl5en/P3tunatTNXVNXU%2Bjru51fnfqLnkUR9kYkUeVuRhRR5WzT2P%2Bv5wb4pbFqyrV69qypQp6t%2B/v5KTk%2BXp%2Ba%2B3PR0OhxYvXqz77rtP/fv31wcffKAtW7YoKSlJkjR%2B/Hi98sorioyMVJcuXfTyyy8rKChIffv2lbe3t0aOHKmUlBStWLFCkpSSkqLRo0c7C1pdBAcH1/o4MC%2BvSFVVt7%2Bwq6trvnIbd1bfc2vqedwJMrEiDyvysCIPK/JoWG5ZsHbt2qWcnBzt3btXb731lmXu%2BPHjSkpK0vz583Xp0iUFBQVp%2BvTpGjt2rCQpNjZWRUVFio%2BPV35%2Bvnr16qW1a9fK29tb0pffobVkyRJFR0ersrJSw4YN0wsvvNDg5wgAANyXh%2BP6F0Dha5WXV3TLOZvNU4GBfiooKKn3Txcjlx%2B626U1mL0zBn31Rrq7PJoqMrEiDyvysCIPq%2BaeR/v2rV1yXH6lAAAAwDAKFgAAgGEULAAAAMMoWAAAAIa55W8Rwj250w35Ut1vygcA4Ea8gwUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMgnUTV65c0bRp09S3b19FREQoOTlZVVVVrl4WAABwExSsm5gxY4Z8fX118OBB7dy5U4cPH9bmzZtdvSwAAOAmKFg3OHfunI4cOaLExETZ7XZ16tRJ06ZNU2pqqquXBgAA3AQF6wZZWVkKCAhQhw4dnGMhISHKycnRtWvXXLgyAADgLmyuXkBjU1JSIrvdbhm7/ry0tFT%2B/v5fuY/c3Fzl5eVZxmw2XwUHB990ey8vT8u/aBxstsbz/wfXiBV5WJGHFXlYkYdrULBu4Ovrq7KyMsvY9ed%2Bfn512kdaWppWrVplGXvmmWc0ffr0m26fm5ur3/9%2Bg%2BLi4m5Zwm7l/eQf1mt7d5Cbm6u0tLQ7yqOpuptrpCkiDyvysCIPK/JwDersDUJDQ1VYWKjLly87x06fPq2OHTuqdevWddpHXFycdu3aZXnExcXdcvu8vDytWrWq1rtezRV51EYmVuRhRR5W5GFFHq7BO1g36Ny5s/r06aNFixZpwYIFKigo0OrVqxUbG1vnfQQHB/NTAgAAzRjvYN3EypUrVVVVpWHDhunRRx/Vd7/7XU2bNs3VywIAAG6Cd7BuIigoSCtXrnT1MgAAgJvymj9//nxXLwJf3kDfr1%2B/Ot9I39SRR21kYkUeVuRhRR5W5NHwPBwOh8PViwAAAGhKuAcLAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFy8WuXLmiadOmqW/fvoqIiFBycrKqqqpcvSyXSU9PV/fu3RUeHu58JCYmunpZDS4/P19RUVHKyMhwjn344YeaMGGCwsPDNXToUO3YscOFK2xYN8ujucrMzNQTTzyhfv36adCgQZozZ47y8/MlNc9r5HZ5zJs3Tz179rT89yQtLc3FK/56HT58WBMmTNB3vvMdDRo0SAsXLlR5ebmk5nl9uJQDLvXjH//YMWvWLEdpaanj/PnzjlGjRjnWr1/v6mW5zIsvvuj4r//6L1cvw6Xef/99x/Dhwx1hYWGOv/71rw6Hw%2BEoLCx09OvXz7Ft2zZHZWWl47333nOEh4c7PvzwQxev9ut3szyaq7KyMsegQYMcK1ascHzxxReO/Px8x5NPPun4z//8z2Z5jdwuD4fD4XjkkUccu3btcvEqG86VK1ccvXr1crz%2B%2BuuO6upqx6VLlxyjR492rFixolleH67GO1gudO7cOR05ckSJiYmy2%2B3q1KmTpk2bptTUVFcvzWVOnDihnj17unoZLrN7927Nnj1bM2fOtIzv379fAQEBmjRpkmw2mwYMGKDo6Ogmf63cKo/mKicnR/fff7/i4%2BPVokULBQYGKi4uTkePHm2W18jt8qioqNCpU6ea1X9P2rZtq/fee08xMTHy8PBQYWGhvvjiC7Vt27ZZXh%2BuRsFyoaysLAUEBKhDhw7OsZCQEOXk5OjatWsuXJlr1NTU6OTJk/rzn/%2BsIUOGKDIyUi%2B88IKuXr3q6qU1mMGDB%2BuPf/yjHn74Yct4VlaWwsLCLGNdu3ZVZmZmQy6vwd0qj%2BbqW9/6ljZs2CAvLy/n2L59%2B9SjR49meY3cLo/MzExVVVVp5cqVGjhwoEaMGKF169appqbGhSv%2B%2BrVq1UqS9L3vfU/R0dFq3769YmJimuX14WoULBcqKSmR3W63jF1/Xlpa6ooluVR%2Bfr66d%2B%2BuESNGKD09Xdu3b9fZs2eb1T1Y7du3l81mqzV%2Bs2vFx8enyV8nt8oDksPh0Msvv6x33nlHzz33XLO9Rq67MY%2BioiL169dPkydP1oEDB7R06VJt3bpVmzZtcvVSG8T%2B/fv17rvvytPTUwkJCc3%2B%2BnAF/svlQr6%2BviorK7OMXX/u5%2BfniiW5VFBQkOXtarvdrsTERD366KMqLi52/mTWHNntdhUVFVnGysvLm%2BV1Aqm4uFhJSUk6efKktm3bpm7dujXra%2BRmeXTr1k2DBg1ybvPAAw9oypQpSk9P19SpU1242obh4%2BMjHx8fJSYmasKECZo8eXKzvT5chXewXCg0NFSFhYW6fPmyc%2Bz06dPq2LGjWrdu7cKVuUZmZqZSUlLkcDicYxUVFfL09FSLFi1cuDLXCwsLU1ZWlmUsOztboaGhLloRXOX8%2BfMaP368iouLtXPnTnXr1k1S871GbpXH22%2B/re3bt1u2raiokI%2BPjyuW2SD%2B9re/6Yc//KEqKiqcYxUVFfL29lbXrl2b5fXhShQsF%2BrcubP69OmjRYsWqbi4WBcuXNDq1asVGxvr6qW5REBAgFJTU7VhwwZVVVUpJydHS5cu1SOPPNLsC1ZUVJQuX76szZs3q7KyUn/961%2B1Z88ejR8/3tVLQwO6evWqpkyZou985zvauHGj2rZt65xrjtfI7fJwOBxavHixDh8%2BLIfDoePHj2vLli2Ki4tz4Yq/Xt26dVN5ebleeuklVVRU6J///KeWLFmi2NhYjRgxotldH67m4fj3twvQ4C5fvqwFCxYoIyNDnp6eGjdunGbPnm25abM5OXLkiJYtW6ZTp06pZcuWGjVqlBITE9WyZUtXL63BdevWTVu2bFFERISkL3/DMjk5WadOnVLbtm01bdo0xcTEuHiVDefGPJqjV199VS%2B%2B%2BKLsdrs8PDwsc8ePH29218hX5bF9%2B3a9%2BuqrunTpkoKCgvTEE09o0qRJLlptw8jOztaiRYt04sQJtW7dWtHR0c7fsmxu14erUbAAAAAM4yNCAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMCw/wdKhZJbDOuX3AAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common1684772975517577072">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">3</td>
        <td class="number">9824</td>
        <td class="number">45.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">6882</td>
        <td class="number">31.9%</td>
        <td>
            <div class="bar" style="width:70%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">2760</td>
        <td class="number">12.8%</td>
        <td>
            <div class="bar" style="width:28%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">1601</td>
        <td class="number">7.4%</td>
        <td>
            <div class="bar" style="width:17%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6</td>
        <td class="number">272</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">196</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7</td>
        <td class="number">38</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8</td>
        <td class="number">13</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9</td>
        <td class="number">6</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (2)</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme1684772975517577072">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1</td>
        <td class="number">196</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">2760</td>
        <td class="number">12.8%</td>
        <td>
            <div class="bar" style="width:28%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">9824</td>
        <td class="number">45.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">6882</td>
        <td class="number">31.9%</td>
        <td>
            <div class="bar" style="width:70%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">1601</td>
        <td class="number">7.4%</td>
        <td>
            <div class="bar" style="width:17%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">8</td>
        <td class="number">13</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9</td>
        <td class="number">6</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:46%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:23%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">11</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:8%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">33</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:8%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_condition">condition<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>5</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>3.4098</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>1</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>5</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram4366645780344090733">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAARJJREFUeJzt3MEJwkAQQNEolmQR9uTZnizCnsazIB89hAR97x6Yy2eysOxhZmYB3jpuPQDs2WnrAXh1vt6//uZxu6wwCctig0ASCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQPA27x/y/u/nbBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKB4LIiq/mFS5E2CASBQBAIBIFA%2BNtD%2Bi8cIFnfYWZm6yFgr/xiQRAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQHgC8eAUleSgPSQAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives4366645780344090733,#minihistogram4366645780344090733"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives4366645780344090733">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles4366645780344090733"
                                                  aria-controls="quantiles4366645780344090733" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram4366645780344090733" aria-controls="histogram4366645780344090733"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common4366645780344090733" aria-controls="common4366645780344090733"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme4366645780344090733" aria-controls="extreme4366645780344090733"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles4366645780344090733">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>3</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>3</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>3</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>4</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>5</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>5</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>4</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>1</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>0.65055</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.19079</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>0.51924</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>3.4098</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>0.56075</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>1.036</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>73642</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>0.42321</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram4366645780344090733">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3Xt0VOWh//9PkglkEowJQrBeukJJohXQYILcsVwicgnSEE0rpcJR64E0MR65GiweMVxOY0VAkIIt3yLnGEFzkDYKdh0K1GJC1GrlNDaBgiiYRBIgt4Fc9u8PDvl1GoSJPMnMbN%2BvtVhZ2fuZvZ9PHjLrM7MnMwGWZVkCAACAMYHengAAAIDdULAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEOb0/gm6Kyssb4MQMDA9S9e5iqqurU0mIZP7632DWXZN9sds0lkc0f2TWXZN9sHZmrZ8%2BrjB7PUzyD5ccCAwMUEBCgwMAAb0/FKLvmkuybza65JLL5I7vmkuybzY65KFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhvl9wWpubtb06dO1YMECt%2B179uxRcnKy4uPjNX78eO3evdtt/4YNGzRy5EjFx8dr%2BvTpOnz4cOu%2B%2Bvp6LVy4UIMGDVJCQoLmzZunurq6TskDAAD8n98XrDVr1qi4uNht25EjR5SRkaFHH31UxcXFysjIUFZWlsrLyyVJ%2Bfn52rx5s1566SUVFhaqb9%2B%2ByszMlGWd//yjJUuW6MSJE9q5c6d27dqlEydOKDc3t9OzAQAA/%2BTXBWv//v3atWuX7rrrLrft%2Bfn5SkxM1NixY%2BVwODRhwgQNHDhQeXl5kqRXX31V999/v2JjY9W1a1c9/vjjOn78uAoLC9XQ0KAdO3YoMzNTERERuuaaazRnzhy9/vrramho8EZMAADgZ/y2YJ08eVLZ2dl69tln5XQ63faVlZUpLi7ObVtMTIxKSkouuj84OFjR0dEqKSnR0aNH1djY6La/T58%2BcrlcOnLkSMcFAgAAtuHw9gS%2BjpaWFs2dO1czZ87UzTff3GZ/XV1dm9IVEhKi%2Bvr6y%2B6vra2VJIWGhrbuuzDW09dhVVRUqLKy0m2bwxGqqKgoj27vqaCgQLevdmHXXJJ9s9k1l3Q%2BU2L2W96eRru8PWeER%2BPsum52zSXZN5sdc/llwVq/fr26dOmi6dOnX3S/0%2BmUy%2BVy2%2BZyuRQWFnbZ/ReKVUNDQ%2Bv4C5cGu3Xr5tH88vLytGbNGrdt6enpyszM9Oj27RUe7rz8ID9k11ySfbPZNZe/iYwMa9d4u66bXXNJ9s1mp1x%2BWbC2b9%2BuiooKJSYmSlJrWfr973%2Bv4uJixcXF6eDBg263KSsrU79%2B/SRJsbGxKi0t1ahRoyRJjY2NOnLkiOLi4tS7d28FBwerrKxMt912myTp0KFDrZcRPZGWlqbRo0e7bXM4QlVdbfYvEYOCAhUe7tSZMw1qbm4xemxvsmsuyb7Z7JpL8s9H1J7e19h13eyaS7Jvto7M1d4HHKb4ZcF66y33p%2BsvvEXD8uXLJUmTJ0/Wr3/9axUUFOiuu%2B7Srl27VFRUpOzsbEnS1KlTtXr1ao0cOVK9e/fWc889px49eigxMVHBwcEaP368cnNz9fzzz0uScnNzNWnSJIWEhHg0v6ioqDaXAysra9TU1DG/DM3NLR12bG%2Byay7JvtnsmsvftHcN7Lpuds0l2TebnXL5ZcG6nD59%2BuiFF15Qbm6usrOzdf3112v16tXq3bu3JCk1NVU1NTVKT09XVVWV%2Bvfvr/Xr1ys4OFiStHjxYq1YsULJyclqbGzUmDFj9OSTT3ozEgAA8CMB1oU3f0KHqqysMX5MhyNQkZFhqq6us03jl%2BybS7JvNrvmks5nS8rd5%2B1ptMubWcM8GmfXdbNrLsm%2B2ToyV8%2BeVxk9nqf878UFAAAAPo6CBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGCYzxesqqoqJSUlqbCwsHXbzp07dc899%2Bj222/X6NGjtWbNGrW0tLTuz8/PV1JSkuLj45WSkqIPPvigdV9zc7NWrFihoUOHasCAAZo1a5YqKipa9588eVKzZ89WYmKiBg0apJycHDU1NXVOWAAAYAs%2BXbDee%2B89paWl6dNPP23d9vHHH2vevHnKyspScXGxNmzYoNdff12bNm2SJBUWFmrJkiVavny5Dhw4oMmTJ2vWrFlqaGiQJK1bt07vvPOOXnvtNe3bt08hISFatGhR6/GzsrIUGhqqffv2adu2bdq/f3/rsQEAADzhswUrPz9fc%2BbM0WOPPea2/fPPP9cPfvADjRo1SoGBgerTp4%2BSkpJ04MABSdLWrVs1ceJEJSQkKDg4WDNmzFBkZKQKCgpa9z/88MP61re%2BpW7duik7O1t79%2B7VsWPHdPToURUVFWnu3LlyOp268cYbNXv2bG3ZsqXT8wMAAP/l8PYEvsrw4cOVnJwsh8PhVrLGjRuncePGtX7vcrn0hz/8QcnJyZKksrIyTZ061e1YMTExKikpUU1Njb744gvFxcW17uvRo4euvvpqffLJJ5KkiIgI9erVq3V/nz59dPz4cZ05c0bh4eEezb2iokKVlZVu2xyOUEVFRXmY3jNBQYFuX%2B3Crrkk%2B2azay7JPzM5HJ7N2a7rZtdckn2z2TGXzxasnj17XnZMbW2tHn30UYWEhGjGjBmSpLq6OjmdTrdxISEhqq%2BvV11dnSQpNDS0zf4L%2B/75the%2Br6%2Bv97hg5eXlac2aNW7b0tPTlZmZ6dHt2ys83Hn5QX7Irrkk%2B2azay5/ExkZ1q7xdl03u%2BaS7JvNTrl8tmBdzuHDh5WZmalrrrlGv/nNb9StWzdJ5wuRy%2BVyG%2BtyuRQZGdlali68Husf94eFhcmyrDb7LnwfFub5HVZaWppGjx7tts3hCFV1dZ3Hx/BEUFCgwsOdOnOmQc3NLZe/gZ%2Bway7Jvtnsmkvyz0fUnt7X2HXd7JpLsm%2B2jszV3gccpvhlwdqzZ4/%2B7d/%2BTffdd58ef/xxORz/f4zY2FiVlpa6jS8rK9PIkSN19dVXq1evXiorK2u9TFhZWalTp04pLi5OLS0tOnXqlL788kv16NFDknTo0CFde%2B21uuqqqzyeX1RUVJvLgZWVNWpq6phfhubmlg47tjfZNZdk32x2zeVv2rsGdl03u%2BaS7JvNTrn87qHZn//8Z6Wnp2vhwoWaP3%2B%2BW7mSpNTUVO3YsUPvvvuuGhsbtWnTJp08eVJJSUmSpJSUFK1bt07Hjh1TbW2tli5dqjvuuEPf/va3FR0drYSEBC1dulS1tbU6duyY1q5dq9TUVG9EBQAAfsrvnsF68cUX1dTUpJycHOXk5LRuT0hI0MaNGzVkyBAtXrxYTz31lMrLyxUTE6MNGzYoIiJC0vnXQjU1NWnatGmqq6vToEGDtHLlytbjrFq1Sk8//bTGjBmjwMBATZkyRbNnz%2B70nAAAwH8FWJZleXsS3wSVlTXGj%2BlwBCoyMkzV1XW2eUpVsm8uyb7Z7JpLOp8tKXeft6fRLm9mDfNonF3Xza65JPtm68hcPXt6/hIfk/zuEiEAAICvo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADPPLgnXy5EnNnj1biYmJGjRokHJyctTU1NS6/8MPP9S9996rAQMGaPTo0dq6davb7fPz85WUlKT4%2BHilpKTogw8%2B6OwIAADAxvyyYGVlZSk0NFT79u3Ttm3btH//fm3atEmSdPr0af3kJz/RlClTdODAAeXk5GjZsmX66KOPJEmFhYVasmSJli9frgMHDmjy5MmaNWuWGhoavJgIAADYid8VrKNHj6qoqEhz586V0%2BnUjTfeqNmzZ2vLli2SpF27dikiIkLTpk2Tw%2BHQkCFDlJyc3Lp/69atmjhxohISEhQcHKwZM2YoMjJSBQUF3owFAABsxO8KVmlpqSIiItSrV6/WbX369NHx48d15swZlZaWKi4uzu02MTExKikpkSSVlZVdcj8AAMCVcnh7Au1VV1cnp9Pptu3C9/X19RfdHxISovr6%2Bq%2B8/T/uN6GiokKVlZVu2xyOUEVFRRk7hyQFBQW6fbULu%2BaS7JvNrrkk/8zkcHg2Z7uum11zSfbNZsdcflewQkND27xe6sL3YWFhcjqdqqmpcdvvcrkUFhYm6XwZc7lcbfZHRkYam2NeXp7WrFnjti09PV2ZmZnGzvGPwsOdlx/kh%2ByaS7JvNrvm8jeRkWHtGm/XdbNrLsm%2B2eyUy%2B8KVmxsrE6dOqUvv/xSPXr0kCQdOnRI1157ra666irFxcXpnXfecbtNWVmZYmNjW29fWlraZv/IkSONzTEtLU2jR4922%2BZwhKq6us7YOaTzTT883KkzZxrU3Nxi9NjeZNdckn2z2TWX5J%2BPqD29r7Hrutk1l2TfbB2Zq70POEzxu4IVHR2thIQELV26VE8//bSqq6u1du1apaamSpKSkpL085//XJs2bdK0adP03nvvaceOHVq7dq0kKTU1Venp6Ro/frwSEhK0ZcsWnTx5UklJScbmGBUV1eZyYGVljZqaOuaXobm5pcOO7U12zSXZN5tdc/mb9q6BXdfNrrkk%2B2azUy6/K1iStGrVKj399NMaM2aMAgMDNWXKFM2ePVuSFBkZqV/96lfKycnRqlWr1L17dy1atEiDBw%2BWJA0ZMkSLFy/WU089pfLycsXExGjDhg2KiIjwZiQAAGAjflmwevTooVWrVn3l/v79%2B%2BuVV175yv333HOP7rnnno6YGgAAgP%2B9TQMAAICvo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGOa3BevgwYOaNm2aEhMTNXz4cD3zzDM6d%2B6cJGnPnj1KTk5WfHy8xo8fr927d7vddsOGDRo5cqTi4%2BM1ffp0HT582BsR2mhoaNDSpf%2BuCRPGaNy4O7Vkyc9UX1/v7WkBAIB28suC1dLSokceeUTjxo1TUVGRtm3bpj/%2B8Y/asGGDjhw5ooyMDD366KMqLi5WRkaGsrKyVF5eLknKz8/X5s2b9dJLL6mwsFB9%2B/ZVZmamLMvycirpuef%2BQ%2BXl5Xrlldf1yiv5Ki//QuvWrfb2tAAAQDv5ZcE6ffq0Kisr1dLS0lqMAgMD5XQ6lZ%2Bfr8TERI0dO1YOh0MTJkzQwIEDlZeXJ0l69dVXdf/99ys2NlZdu3bV448/ruPHj6uwsNCbkeRyubRr15t66KFHFB5%2BtSIju2vWrEwVFLwhl8vl1bkBAID2cXh7Al9HZGSkZsyYoRUrVug//uM/1NzcrDFjxmjGjBnKyMhQXFyc2/iYmBiVlJRIksrKyvTwww%2B37gsODlZ0dLRKSko0ePBgI/OrqKhQZWWl2zaHI1RRUVFfeZsTJz5TU1OT4uLi5HAE/t%2B8%2B%2Bjs2bM6fvyY4uJuanOboKBAt692Yddckn2z2TWX5J%2BZLtyHXI5d182uuST7ZrNjLr8sWC0tLQoJCdGTTz6p1NRUHT16VD/96U%2B1atUq1dXVyel0uo0PCQlpfS3T5fabkJeXpzVr1rhtS09PV2Zm5lfeJjCwWZJ03XU9FBh4/j9YeHiIJCkoqEWRkWFfedvwcOdX7vNnds0l2TebXXP5m0vdX1yMXdfNrrkk%2B2azUy6/LFhvv/22du7cqbfeekuSFBsbq/T0dOXk5Oj2229vc0nN5XIpLOz8HY7T6bzkfhPS0tI0evRot20OR6iqq%2Bu%2B8jZNTQGSpBMnTio0NFTS%2BTIoSS0tQRe9bVBQoMLDnTpzpkHNzS2mpu91ds0l2TebXXNJ/vmI%2BlL3Nf/Irutm11ySfbN1ZK72PuAwxS8L1okTJ1r/YvACh8Oh4OBgxcXF6eDBg277ysrK1K9fP0nny1hpaalGjRolSWpsbNSRI0faXFa8ElFRUW0uB1ZW1qip6av/01x//bflcDhUWlqmvn37/d%2B8Dyk4OFjXXXfDJW/b3Nxyyf3%2Byq65JPtms2suf9PeNbDrutk1l2TfbHbK5X8PzSQNHz5clZWVevHFF9Xc3Kxjx45p3bp1Sk5O1uTJk1VUVKSCggI1NTWpoKBARUVFuueeeyRJU6dO1csvv6ySkhKdPXtWzz77rHr06KHExESvZgoJCdGYMUl68cXVqq6uVnV1tV58cbXGjh2nrl1DvDo3AADQPn75DFZMTIzWr1%2BvlStXauPGjbrqqqs0efJkpaenq0uXLnrhhReUm5ur7OxsXX/99Vq9erV69%2B4tSUpNTVVNTY3S09NVVVWl/v37a/369QoODvZyKunxxxdo9eqVeuCBH6ixsVEjRtypxx6b5%2B1pAQCAdgqwfOENoL4BKitrjB/T4QhUZGSYqqvrbPOUqmTfXJJ9s9k1l3Q%2BW1LuPm9Po13ezBrm0Ti7rptdc0n2zdaRuXr2vMro8Tzll5cIAQAAfBkFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAwzWrBSUlK0ZcsWnT592uRhAQAA/IrRgjV06FBt2LBBI0aMUFZWlvbt2yfeZgsAAHzTGC1Yc%2BbM0e7du7Vu3ToFBwcrMzNTd955p5577jn9/e9/N3kqAAAAn2X8o3ICAgI0bNgwDRs2TA0NDdq8ebPWrl2rX/7yl7r99tv1wAMP6K677jJ9WgAAAJ/RIZ9FWFFRoTfeeENvvPGG/va3v%2Bn222/X97//fZWXl2vRokU6cOCAsrOzO%2BLUAAAAXme0YG3fvl3bt29XYWGhunfvrilTpmjVqlWKjo5uHXPttdcqJyeHggUAAGzLaMHKzs7WqFGj9MILL2jkyJEKDGz7Eq/evXtr2rRpJk8LAADgU4wWrL179yoiIkKnTp1qLVcffPCB%2BvXrp%2BDgYElSQkKCEhISTJ4WAADApxj9K8Kamhrddddd2rBhQ%2Bu2Rx55RFOmTNGJEydMngoAAMBnGS1YOTk5iomJ0YMPPti67a233tINN9ygZcuWmTwVAACAzzJasN5//33Nnz9fPXr0aN3WvXt3zZkzR%2B%2B%2B%2B67JUwEAAPgsowXL4XCourq6zfaGhgaTpwEAAPBpRgvWnXfeqWeeeUZHjx5t3Xbs2DEtXbpUI0aMMHkqAAAAn2X0rwjnz5%2Bvf/mXf9Hdd9%2Bt8PBwSdKZM2fUt29fLViwwOSpAAAAfJbRgtW9e3e99tpr2r9/v/72t7/J4XAoJiZGQ4YMUUBAgMlTAQAA%2BCzjH5UTFBSk4cOHa/jw4aYPDQAA4BeMvgYLAAAAFCwAAADjKFgAAACGGS9YJSUlWrhwoX7wgx%2BovLxcW7Zs4U1GAQDAN4rRgvXxxx/rvvvu02effaaPP/5Y586d01//%2Blc9%2BOCD2r17t8lTAQAA%2BCyjBSs3N1czZ87U5s2bFRwcLEl65pln9OMf/1hr1qwxeSoAAACfZfwZrClTprTZ/sMf/lCHDx82eSoAAACfZbRgBQcHq7a2ts3248ePy%2Bl0mjwVAACAzzJasMaOHatnn33W7QOfDx06pJycHH3ve98zeSoAAACfZbRgzZ8/Xy6XS0OHDlVDQ4NSUlI0adIkORwOzZs3z%2BSpAAAAfJbRj8rp1q2bXnnlFe3fv1//%2B7//q5aWFsXFxWnEiBEKDOQttwAAwDdDh7SeIUOG6MEHH9TDDz%2BsO%2B%2B8s0PK1alTpzRv3jwNGjRIAwcO1OzZs1VRUSFJ%2BvDDD3XvvfdqwIABGj16tLZu3ep22/z8fCUlJSk%2BPl4pKSn64IMPWvc1NzdrxYoVGjp0qAYMGKBZs2a1HhcAAMATRpvP6NGjNWbMmK/8Z1JGRobq6%2Bv19ttva/fu3QoKCtKTTz6p06dP6yc/%2BYmmTJmiAwcOKCcnR8uWLdNHH30kSSosLNSSJUu0fPlyHThwQJMnT9asWbPU0NAgSVq3bp3eeecdvfbaa9q3b59CQkK0aNEio3MHAAD2ZvQS4fe//30FBAS0ft/Y2KijR49q7969ysrKMnaejz/%2BWB9%2B%2BKH%2B9Kc/qVu3bpKkJUuWqLKyUrt27VJERISmTZsm6fyzacnJydqyZYtuvfVWbd26VRMnTlRCQoIkacaMGcrLy1NBQYGmTp2qrVu3as6cOfrWt74lScrOztbw4cN17Ngx3XjjjcYyAAAA%2BzJasDIyMi66/eWXX9Z7772nH//4x0bO89FHHykmJkavvvqq/uu//ksNDQ0aMWKE5s%2Bfr9LSUsXFxbmNj4mJ0bZt2yRJZWVlmjp1apv9JSUlqqmp0RdffOF2%2Bx49eujqq6/WJ598QsECAAAeMVqwvsqoUaP0i1/8wtjxTp8%2BrU8%2B%2BUT9%2BvVTfn6%2BXC6X5s2bp/nz56tHjx5t3nMrJCRE9fX1kqS6urqv3F9XVydJCg0NbbP/wj5PVFRUqLKy0m2bwxGqqKgoj4/hiaCgQLevdmHXXJJ9s9k1l%2BSfmRwOz%2BZs13Wzay7JvtnsmKtTClZRUZG6du1q7HhdunSRdP7yXdeuXdWtWzdlZWXpvvvuU0pKilwul9t4l8ulsLAwSZLT6bzo/sjIyNbideH1WBe7vSfy8vLafDRQenq6MjMzPT5Ge4SH2/NNXO2aS7JvNrvm8jeRkZ7fX0n2XTe75pLsm81OuYwWrH%2B%2BBGhZlmpra/XJJ58Yuzwonb%2Bk19LSosbGxtbi1tLSIkn67ne/q//8z/90G19WVqbY2FhJUmxsrEpLS9vsHzlypK6%2B%2Bmr16tVLZWVlrZcJKysrderUqTaXHS8lLS1No0ePdtvmcISqutrzZ8E8ERQUqPBwp86caVBzc4vRY3uTXXNJ9s1m11ySfz6i9vS%2Bxq7rZtdckn2zdWSu9j7gMMVowbruuuvcXuQunf/4nAceeEDJycnGzjN06FDdeOONeuKJJ7Rs2TKdPXtWzz33nMaOHatJkyZp1apV2rRpk6ZNm6b33ntPO3bs0Nq1ayVJqampSk9P1/jx45WQkKAtW7bo5MmTSkpKkiSlpKRo3bp16t%2B/vyIjI7V06VLdcccd%2Bva3v%2B3x/KKiotpcDqysrFFTU8f8MjQ3t3TYsb3Jrrkk%2B2azay5/0941sOu62TWXZN9sdspltGAtX77c5OG%2BUnBwsDZv3qzly5dr3LhxOnv2rEaPHq3s7GyFh4frV7/6lXJycrRq1Sp1795dixYt0uDBgyWd/6vCxYsX66mnnlJ5ebliYmK0YcMGRURESDp/Ka%2BpqUnTpk1TXV2dBg0apJUrV3ZKLgAAYA8BlmVZpg524MABj8cOHDjQ1Gn9QmVljfFjOhyBiowMU3V1nW0av2TfXJJ9s9k1l3Q%2BW1LuPm9Po13ezBrm0Ti7rptdc0n2zdaRuXr2vMro8Txl9BmsGTNmyLKs1n8XXLhseGFbQECA/vrXv5o8NQAAgM8wWrBWr16tZcuWaf78%2BRo8eLCCg4P14Ycf6qmnntL999%2BvUaNGmTwdAACATzJasFasWKHFixdr%2BPDhrdvuuOMOPf3005o3b55%2B9KMfmTwdAMDPjV/5jren0C6eXn4FjP79cUVFRetHzPyjbt26qbq62uSpAAAAfJbRghUfH69f/OIXqq2tbd126tQp/fznP9eQIUNMngoAAMBnGb1EuGjRIj3wwAMaOXKkoqOjJUl///vf1bNnT/3mN78xeSoAAACfZbRg9enTRwUFBdqxY4cOHTokSbr//vs1ceLENp//BwAAYFfGP4swPDxc9957rz777DPdeOONks6/MSgAAMA3hdHXYFmWpdzcXA0cOFCTJk3SF198ofnz52vhwoVqbGw0eSoAAACfZbRgbd68Wdu3b9fixYvVpUsXSdLYsWP1P//zP3r%2B%2BedNngoAAMBnGS1YeXl5%2BtnPfqaUlJTWd2%2BfMGGCcnJy9Lvf/c7kqQAAAHyW0YL12Wef6bvf/W6b7TfddJO%2B/PJLk6cCAADwWUYL1vXXX6%2BPPvqozfY9e/a0vuAdAADA7oz%2BFeGDDz6of//3f1d5ebksy9L%2B/fv1yiuvaPPmzVq4cKHJUwEAAPgsowVr6tSpampq0rp16%2BRyufSzn/1M11xzjR577DH98Ic/NHkqAAAAn2W0YL3xxhu6%2B%2B67lZaWpqqqKlmWpWuuucbkKQAAAHye0ddgPfPMM60vZu/evTvlCgAAfCMZLVjR0dH65JNPTB4SAADA7xi9RBgbG6s5c%2BZo48aaNpllAAAU4klEQVSNio6OVteuXd32L1u2zOTpAAAAfJLRgvXpp58qISFBklRZWWny0AAAAH7DaMHavHmzycMBAAD4pSt%2BDdayZctUX19vYi4AAAC2cMUF6ze/%2BY0aGhrctj344IOqqKi40kMDAAD4pSsuWJZltdn2/vvv6%2BzZs1d6aAAAAL9k9G0aAAAAQMECAAAwzkjBCggIMHEYAAAAWzDyNg3PPPOM25uKNjY26uc//7nCwsLcxvFGowAA4JvgigvWwIED27yp6IABA1RdXa3q6uorPTwAAIDfueKCxZuLAgAAuONF7gAAAIZRsAAAAAyjYAEAABhGwQIAADDM7wtWc3Ozpk%2BfrgULFrRu27Nnj5KTkxUfH6/x48dr9%2B7dbrfZsGGDRo4cqfj4eE2fPl2HDx9u3VdfX6%2BFCxdq0KBBSkhI0Lx581RXV9dpeQAAgP/z%2B4K1Zs0aFRcXt35/5MgRZWRk6NFHH1VxcbEyMjKUlZWl8vJySVJ%2Bfr42b96sl156SYWFherbt68yMzNbP1NxyZIlOnHihHbu3Kldu3bpxIkTys3N9Uo2AADgn/y6YO3fv1%2B7du3SXXfd1botPz9fiYmJGjt2rBwOhyZMmKCBAwcqLy9PkvTqq6/q/vvvV2xsrLp27arHH39cx48fV2FhoRoaGrRjxw5lZmYqIiJC11xzjebMmaPXX39dDQ0N3ooJAAD8jJF3cveGkydPKjs7W2vXrtWmTZtat5eVlSkuLs5tbExMjEpKSlr3P/zww637goODFR0drZKSEkVERKixsdHt9n369JHL5dKRI0f03e9%2B16O5VVRUtHnzVYcjVFFRUe2NeUlBQYFuX%2B3Crrkk%2B2azay7JPzM5HJ7N2c7r1lE8/dl2FLuumR1z%2BWXBamlp0dy5czVz5kzdfPPNbvvq6urkdDrdtoWEhKi%2Bvv6y%2B2trayVJoaGhrfsujG3P67Dy8vK0Zs0at23p6enKzMz0%2BBjtER7uvPwgP2TXXJJ9s9k1l7%2BJjAy7/KB/wLp5rr0/245i1zWzUy6/LFjr169Xly5dNH369Db7nE6nXC6X2zaXy9X6uYiX2n%2BhWDU0NLSOv3BpsFu3bh7PLy0tTaNHj3bb5nCEqrra7Ivlg4ICFR7u1JkzDWpubjF6bG%2Byay7Jvtnsmkvyz0fUnt7X2HndOorp%2B/H2suuadWQub5VivyxY27dvV0VFhRITEyWptTD9/ve/17Rp03Tw4EG38WVlZerXr58kKTY2VqWlpRo1apSk8x9MfeTIEcXFxal3794KDg5WWVmZbrvtNknSoUOHWi8jeioqKqrN5cDKyho1NXXML0Nzc0uHHdub7JpLsm82u%2BbyN%2B1dA9bNc77yc7Lrmtkpl/89NJP01ltv6f3331dxcbGKi4s1adIkTZo0ScXFxZo8ebKKiopUUFCgpqYmFRQUqKioSPfcc48kaerUqXr55ZdVUlKis2fP6tlnn1WPHj2UmJgop9Op8ePHKzc3V1VVVaqqqlJubq4mTZqkkJAQL6cGAAD%2Bwi%2BfwbqUPn366IUXXlBubq6ys7N1/fXXa/Xq1erdu7ckKTU1VTU1NUpPT1dVVZX69%2B%2Bv9evXKzg4WJK0ePFirVixQsnJyWpsbNSYMWP05JNPejMSAADwMwHWhTeAQoeqrKwxfkyHI1CRkWGqrq6zzVOqkn1zSfbNZtdc0vlsSbn7vD2Ndnkza5hH43xh3cavfMcr5/26PP3ZdhRfWLOO0JG5eva8yujxPOWXlwgBAAB8GQULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYJhfFqySkhLNnDlTd9xxh4YNG6Z58%2BapqqrKq3Oqr6/XwoULddNNN6lfv3667bbbNHr0aG3dutWr8wIAAJ3P7wqWy%2BXSQw89pAEDBuiPf/yjfvvb3%2BrUqVN64oknvDqvJUuW6Pjx45Kkf/3Xf9Utt9yirl27atmyZfroo4%2B8OjcAANC5/K5gHT9%2BXDfffLPS09PVpUsXRUZGKi0tTQcOHPDanBoaGrRjxw4NHDhQ0dHR%2BulPf6oFCxbo%2BPHjGjdunLZs2eK1uQEAgM7ndwXrO9/5jjZu3KigoKDWbTt37lTfvn29NqejR4%2BqsbFRZ86cUVxcnCSpT58%2BcrlcuuGGG1RSUuK1uQEAgM7n8PYEroRlWVq5cqV2796tl19%2B2WvzqK2tlSTV1NTI6XSqoqJCX3zxhSSpvLxcp06dUlXVl4qKijJ63qCgQLevdmHXXJJ9s9k1l%2BSfmRwOz%2BZs53XrKJ7%2BbDuKXdfMjrn8tmDV1tZq4cKFOnjwoF5%2B%2BWXddNNNXptLaGioJCkgIEAul0t5eXlas2aNJCkvL0%2BStHPnDmVmZnbI%2BcPDnR1yXG%2Bzay7JvtnsmsvfREaGtWs86%2Ba59v5sO4pd18xOufyyYH366ad6%2BOGHdd1112nbtm3q3r27V%2BfTu3dvBQcHKzo6Wvn5%2BVq0aJFuuOEGLVq0SCNGjFBLS4vGjUtWdXWd0fMGBQUqPNypM2ca1NzcYvTY3mTXXJJ9s9k1l%2BSfj6g9va%2Bx87p1FNP34%2B1l1zXryFzeKsV%2BV7BOnz6tBx54QIMHD1ZOTo4CA71/5%2Bd0OjV%2B/Hjt27dPhw8f1htvvKE9e/YoKipKRUVFWrt2rbp376Gmpo75ZWhubumwY3uTXXNJ9s1m11z%2Bpr1rwLp5zld%2BTnZdMzvl8n47aafXX39dx48f15tvvqmEhAQNGDCg9Z83LV68WNHR0QoKCtL69etVVFSkgIAALVq0SIMHD/bq3AAAQOfyu2ewZs6cqZkzZ3p7Gm1069ZNS5Ys0ZIlS7w9FQAA4GV%2B9wwWAACAr/O7Z7AAAIBnxq98x9tT8Fhxzt3enoJRPIMFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBuoiTJ09q9uzZSkxM1KBBg5STk6OmpiZvTwsAAPgJh7cn4IuysrLUq1cv7du3T19%2B%2BaVmzZqlTZs26aGHHvL21NCJxq98x9tTaJc3s4Z5ewoAgP/DM1j/5OjRoyoqKtLcuXPldDp14403avbs2dqyZYu3pwYAAPwEBeuflJaWKiIiQr169Wrd1qdPHx0/flxnzpzx4swAAIC/4BLhP6mrq5PT6XTbduH7%2Bvp6hYeHX/YYFRUVqqysdNvmcIQqKirK3EQlBQUFKjH7LaPH7Ehvzxnh0bigoEC3r/CMw%2BG9n5ed18wfM3n6f8HO69ZRvPl7Jtl/zeyUi4L1T0JDQ9XQ0OC27cL3YWFhHh0jLy9Pa9ascdv205/%2BVBkZGWYm%2BX8qKir0wLWlSktLM17evKmiokL/7/9t9Hqu4py7jR%2BzoqJCeXl5Xs9mmq%2BsWUew6%2B%2BZ5Bvrxu9Z%2B7R3zTri59sRKioqtHr1alutmX2qoiGxsbE6deqUvvzyy9Zthw4d0rXXXqurrrrKo2OkpaXp9ddfd/uXlpZmfK6VlZVas2ZNm2fL/J1dc0n2zWbXXBLZ/JFdc0n2zWbHXDyD9U%2Bio6OVkJCgpUuX6umnn1Z1dbXWrl2r1NRUj48RFRVlmwYOAADaj2ewLmLVqlVqamrSmDFjdN9992nEiBGaPXu2t6cFAAD8BM9gXUSPHj20atUqb08DAAD4qaCnnnrqKW9PAl9fWFiY7rjjDo9fgO8v7JpLsm82u%2BaSyOaP7JpLsm82u%2BUKsCzL8vYkAAAA7ITXYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAbG3//v269957dfvtt2vYsGFasmSJXC7XRcfu2bNHycnJio%2BP1/jx47V79%2B5Oni0Au6BgAbCtqqoqPfLII/rhD3%2Bo4uJi5efnq6ioSL/85S/bjD1y5IgyMjL06KOPqri4WBkZGcrKylJ5ebkXZg7A31Gw/ERVVZWSkpJUWFj4lWP88dG3J7keeugh9e/fXwMGDGj9t3fv3k6cZfuUlJRo5syZuuOOOzRs2DDNmzdPVVVVFx3rT2vWnly%2Bsmbdu3fXn/70J6WkpCggIECnTp3S2bNn1b179zZj8/PzlZiYqLFjx8rhcGjChAkaOHCg8vLyOn3eJrXnGTxfWTdTmpubNX36dC1YsOArx%2BTn5yspKUnx8fFKSUnRBx980Ikz/GqezH38%2BPG67bbb3Nbr0KFDnTjL9ikoKNAtt9ziNt%2B5c%2BdedKyvrku7WPB5xcXF1tixY624uDjr3XffveiYv//971b//v2tt99%2B22psbLR%2B97vfWbfeeqv1xRdfdPJsPedJLsuyrEGDBlmFhYWdOLOvr6GhwRo2bJj1/PPPW2fPnrWqqqqshx9%2B2HrkkUfajPWnNWtPLsvyzTUbMWKEFRcXZ91///1WXV1dm/2zZ8%2B2li1b5rZt2bJl1qxZszprisadPHnS6t%2B/v/Xaa69Zzc3NVnl5uTVp0iTr%2Beefv%2Bh4X1y3K7Fy5Urr5ptvtubPn3/R/e%2B%2B%2B641YMAAq7i42Dp37pz161//2ho0aJBVX1/fyTNt63Jzr6mpsW666Sbrs88%2B6%2BSZfX3Lly%2B3FixYcNlxvrwu7cEzWD4uPz9fc%2BbM0WOPPXbZcf706NvTXMeOHdPp06d1yy23dNLMrszx48d18803Kz09XV26dFFkZKTS0tJ04MCBNmP9ac3ak8tX12zXrl3au3evAgMDlZmZ2WZ/XV2dnE6n27aQkBDV19d31hSNa88zeL66bl/X/v37tWvXLt11111fOWbr1q2aOHGiEhISFBwcrBkzZigyMlIFBQWdONO2PJn7xx9/rIiICF1//fWdOLMr85e//EX9%2BvW77DhfXZf2omD5uOHDh%2Bvtt9/WhAkTLjmurKxMcXFxbttiYmJUUlLSkdP72jzN9Ze//EVhYWF67LHHNHjwYE2aNEnbtm3rpFm233e%2B8x1t3LhRQUFBrdt27typvn37thnrT2vWnly%2BumYhISHq1auX5s6dq3379un06dNu%2B51OZ5tLZy6Xy%2B8/eLZbt26SpDvvvFPJycnq2bOnUlJS2ozz1XX7Ok6ePKns7Gw9%2B%2ByzbUrzP/LF30FP5/6Xv/xFTqdTP/rRjzRo0CClpKT49EsMWlpadPDgQf3hD3/QqFGjNHLkSD355JNtfg8l31yXr4OC5eN69uwph8Nx2XH%2B9ujb01znzp1TfHy8HnvsMe3bt08LFixQTk6O3nzzzU6Y5ZWxLEvPPfecdu/erezs7Db7/W3NLrhcLl9as/fff1933323zp075za/4ODgNj/7uLg4lZaWum0rKytTbGxsp8y1o13uGTxfWrcr0dLSorlz52rmzJm6%2BeabLznW134H2zP3gIAA9e/fX88884z27dunGTNmKCMjQ3/%2B8587abbtU1VVpVtuuUXjxo1TQUGBXnnlFR05cuSir8HytXX5uihYNmHXR99TpkzRxo0bdcsttyg4OFjDhw/XlClTfP5Ov7a2VpmZmdqxY4defvll3XTTTW3G%2BOOaeZLLl9bspptuksvl0rPPPqtz587p888/14oVK5SamqouXbq4jZ08ebKKiopUUFCgpqYmFRQUqKioSPfcc0%2Bnz7sjXO4ZPF9atyuxfv16denSRdOnT7/sWF/7HWzP3B966CGtWrVK0dHR6tKliyZPnqyhQ4dq586dnTDT9uvRo4e2bNmi1NRUOZ1OXXfddZo7d6727t2r2tpat7G%2Bti5fFwXLJuz66Hvbtm1t7uDPnTunrl27emlGl/fpp59q6tSpqq2t1bZt2y5aQiT/WzNPc/nSmoWFhWnjxo0qLS3VsGHDNH36dA0dOlRPPPGEJGnAgAF64403JEl9%2BvTRCy%2B8oPXr12vgwIFau3atVq9erd69e3f6vE1pzzN4vrRuV2L79u0qKipSYmKiEhMT9dvf/la//e1vlZiY2GZsbGysT/0OtmfuL730kvbv3%2B%2B2zZfXq6SkRLm5ubIsq3XbuXPnFBgY2ObBjq%2Bty9fm5RfZox0u9dd2ZWVlVv/%2B/a3f/e53rX%2BR1r9/f%2Bvw4cOdPMv2u1SuX//619aQIUOsgwcPWs3Nzdbu3butW2%2B91Tpw4EAnz9Izp06dsr73ve9ZCxYssJqbmy851p/WrD25/G3N7Ky2tta68847raVLl1pnz561PvvsMys1NdVavHhxm7F2Xbf58%2Bd/5V/i/elPf7IGDBhg7d%2B/v/Wv1QYOHGhVV1d38iwv7lJzX7JkiTVu3Djr008/tRobG62tW7dat956q3XkyJFOnqVnTpw4YcXHx1u//OUvrcbGRuvzzz%2B37rvvPuuJJ55oM9bX18VTFCw/8s9FJD4%2B3tq%2BfXvr93v37rUmT55sxcfHWxMnTrT%2B8Ic/eGOa7XapXC0tLdYLL7xgjRo1yrr11lutiRMnWm%2B%2B%2Baa3pnpZv/rVr6y4uDjrtttus%2BLj493%2BWZb/rll7cvnbmtldaWmpNXPmTCsxMdEaNWqU9Ytf/MI6e/asZVnfjHX7x5Ly%2BeefW/Hx8W6l8b//%2B7%2BtcePGWfHx8VZqaqr15z//2VtTbeNScz979qyVk5NjDR8%2B3LrtttusqVOnXvLtbnxBYWGhlZaWZg0YMMAaPHiwtWTJEsvlcvndungqwLL%2B4fk6AAAAXDFegwUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGPb/ATKW7b5cPOPQAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common4366645780344090733">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">3</td>
        <td class="number">14020</td>
        <td class="number">64.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">5677</td>
        <td class="number">26.3%</td>
        <td>
            <div class="bar" style="width:41%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">1701</td>
        <td class="number">7.9%</td>
        <td>
            <div class="bar" style="width:13%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">170</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1</td>
        <td class="number">29</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme4366645780344090733">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1</td>
        <td class="number">29</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">170</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">14020</td>
        <td class="number">64.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">5677</td>
        <td class="number">26.3%</td>
        <td>
            <div class="bar" style="width:41%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">1701</td>
        <td class="number">7.9%</td>
        <td>
            <div class="bar" style="width:13%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1</td>
        <td class="number">29</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2</td>
        <td class="number">170</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3</td>
        <td class="number">14020</td>
        <td class="number">64.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">5677</td>
        <td class="number">26.3%</td>
        <td>
            <div class="bar" style="width:41%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">1701</td>
        <td class="number">7.9%</td>
        <td>
            <div class="bar" style="width:13%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_date">date<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="alert">
            <th>Distinct count</th>
            <td>372</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>1.7%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable4885679388690043951">
    <table class="mini freq">
        <tr class="">
    <th>6/23/2014</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.7%">
            &nbsp;
        </div>
        142
    </td>
</tr><tr class="">
    <th>6/26/2014</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.6%">
            &nbsp;
        </div>
        131
    </td>
</tr><tr class="">
    <th>6/25/2014</th>
    <td>
        <div class="bar" style="width:1%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 0.6%">
            &nbsp;
        </div>
        131
    </td>
</tr><tr class="other">
    <th>Other values (369)</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 98.1%">
            21193
        </div>
        
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable4885679388690043951, #minifreqtable4885679388690043951"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable4885679388690043951">
    
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">6/23/2014</td>
        <td class="number">142</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6/26/2014</td>
        <td class="number">131</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6/25/2014</td>
        <td class="number">131</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7/8/2014</td>
        <td class="number">127</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4/27/2015</td>
        <td class="number">126</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3/25/2015</td>
        <td class="number">123</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4/28/2015</td>
        <td class="number">121</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4/14/2015</td>
        <td class="number">121</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4/22/2015</td>
        <td class="number">121</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7/9/2014</td>
        <td class="number">121</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (362)</td>
        <td class="number">20333</td>
        <td class="number">94.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_floors">floors<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>6</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>1.4941</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>1</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>3.5</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram9088482465624116579">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAARtJREFUeJzt3LENwjAUQEGCGIkh2ImanRiCnUyP0BMpICG56yO5ef6RZXkaY4wD8NZx6QXAmp2WXsCr8/U%2B%2B5vH7fKFlYAJAkkgEAQCQSAQBAJBIBAEAkEgEAQCQSAQVnfVZEtcm/l/JggEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUDY7dOjngXlEyYIBIFAEAgEgUAQCITdnmIxz15P/UwQCCbIBszd3bews//KNMYYSy8C1sovFgSBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCIQn99UXl1pfrJ4AAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives9088482465624116579,#minihistogram9088482465624116579"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives9088482465624116579">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles9088482465624116579"
                                                  aria-controls="quantiles9088482465624116579" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram9088482465624116579" aria-controls="histogram9088482465624116579"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common9088482465624116579" aria-controls="common9088482465624116579"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme9088482465624116579" aria-controls="extreme9088482465624116579"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles9088482465624116579">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>1.5</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>2</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>2</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>3.5</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>2.5</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>1</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>0.53968</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.36121</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-0.49107</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>1.4941</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>0.48835</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.6145</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>32268</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>0.29126</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram9088482465624116579">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAHtRJREFUeJzt3W9sVfd9%2BPFP8IVx7YFsRAzahEoKdqp0dFDTACLJVIinEcofUTI/QFGC1GaNWRhIpdOWTImMTDK1mVqPoqpJGFNjqQivaKKDlj5IW5YRcLsomToh2YjQamjYEP4bBwx3D/LDv7iQBsP3cjmX10vigc8599zv%2Bejr5s2109xVKBQKAQBAMiNKvQAAgHIjsAAAEhNYAACJCSwAgMQEFgBAYgILACAxgQUAkJjAAgBITGABACQmsAAAEhNYAACJCSwAgMQEFgBAYgILACAxgQUAkJjAAgBITGABACQmsAAAEhNYAACJCSwAgMQEFgBAYgILACAxgQUAkJjAAgBITGABACQmsAAAEhNYAACJCSwAgMQEFgBAYgILACAxgQUAkJjAAgBITGABACQmsAAAEhNYAACJCSwAgMQEFgBAYgILACAxgQUAkFiu1Au4U/T2nkl%2BzxEj7opx46rivffOxeXLheT3v5OZbfGYbXGZb/GYbfEUc7Z33z0m6f2ul0%2BwMmzEiLvirrvuihEj7ir1UsqO2RaP2RaX%2BRaP2RZPOc5WYAEAJCawAAASE1gAAIkJLACAxAQWAEBiAgsAIDGBBQCQmMACAEhMYAEAJCawAAASE1gAAIkJLACAxAQWAEBiuVIvgJsz85kflXoJ123XmrmlXgIA3BI%2BwQIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGKZDqzjx49Hc3NzzJw5M2bNmhWtra0xMDAweP7tt9%2BORx99NGbMmBHz5s2Lbdu2DXn99u3bo7GxMaZPnx7Lli2Lt95661Y/AgBQhjIdWGvWrInKysrYs2dPdHR0xN69e2PLli0REXHq1Kl48sknY%2BnSpdHZ2Rmtra3xwgsvxDvvvBMREfv27Yv169fHiy%2B%2BGJ2dnbF48eJ46qmn4vz58yV8IgCgHGQ2sA4fPhz79%2B%2BPdevWRT6fj0mTJkVzc3O0t7dHRMTu3bujuro6VqxYEblcLubMmROLFi0aPL9t27ZYuHBhNDQ0xMiRI%2BOJJ56Impqa2LlzZykfCwAoA7lSL%2BBGdXV1RXV1dUyYMGHw2JQpU%2BLIkSNx%2BvTp6Orqivr6%2BiGvmTp1anR0dERERHd3d3zxi1%2B86vyBAwduem09PT3R29s75FguVxm1tbU3fe8Pq6jIVh/nctlZ75XZZm3GWWC2xWW%2BxWO2xVOOs81sYJ07dy7y%2BfyQY1e%2B7uvru%2Bb50aNHR19f30e%2B/sPnb8bWrVtj48aNQ46tWrUqVq9efdP3zrKamqpSL2HYxo7Nf/xF3BCzLS7zLR6zLZ5ymm1mA6uysvKq35e68nVVVVXk8/k4c%2BbMkPP9/f1RVfXBP%2BTz%2BXz09/dfdb6mpuam19bU1BTz5s0bciyXq4wTJ87d9L0/LGuln/r5i6miYkSMHZuP06fPx6VLl0u9nLJitsVlvsVjtsVTzNmW6i/3mQ2surq6OHnyZBw7dizGjx8fEREHDx6MiRMnxpgxY6K%2Bvj7eeOONIa/p7u6Ourq6wdd3dXVddf6hhx666bXV1tZe9ePA3t4zMTBwZ39DZvH5L126nMl1Z4HZFpf5Fo/ZFk85zTZbH4F8yOTJk6OhoSE2bNgQZ8%2Bejd/85jexadOmWL58eURENDY2xrFjx2LLli1x8eLFePPNN2PHjh2Dv3e1fPny2LFjR7z55ptx8eLF2LJlSxw/fjwaGxtL%2BVgAQBnI7CdYERFtbW3R0tIS8%2BfPjxEjRsTSpUujubk5IiJqampi8%2BbN0draGm1tbTFu3Lh49tlnY/bs2RERMWfOnHjuuefi%2Beefj6NHj8bUqVPj5Zdfjurq6lI%2BEgBQBu4qFAqFUi/iTtDbe%2BbjLxqmXG5ENH5jT/L7FsuuNXNLvYTrlsuNiJqaqjhx4lzZfFx9uzDb4jLf4jHb4inmbO%2B%2Be0zS%2B12vzP6IEADgdiWwAAASE1gAAIkJLACAxAQWAEBiAgsAIDGBBQCQmMACAEhMYAEAJCawAAASE1gAAIkJLACAxAQWAEBiAgsAIDGBBQCQmMACAEhMYAEAJCawAAASE1gAAIkJLACAxAQWAEBiAgsAIDGBBQCQmMACAEhMYAEAJCawAAASE1gAAIkJLACAxAQWAEBiAgsAIDGBBQCQmMACAEhMYAEAJCawAAASE1gAAIkJLACAxAQWAEBiAgsAIDGBBQCQmMACAEhMYAEAJCawAAASE1gAAIkJLACAxAQWAEBiuVIvALjzzHzmR6VewrDsWjO31EsAMsYnWAAAiWU2sI4fPx7Nzc0xc%2BbMmDVrVrS2tsbAwMDg%2BbfffjseffTRmDFjRsybNy%2B2bds25PXbt2%2BPxsbGmD59eixbtizeeuutW/0IAECZymxgrVmzJiorK2PPnj3R0dERe/fujS1btkRExKlTp%2BLJJ5%2BMpUuXRmdnZ7S2tsYLL7wQ77zzTkRE7Nu3L9avXx8vvvhidHZ2xuLFi%2BOpp56K8%2BfPl/CJAIBykcnAOnz4cOzfvz/WrVsX%2BXw%2BJk2aFM3NzdHe3h4REbt3747q6upYsWJF5HK5mDNnTixatGjw/LZt22LhwoXR0NAQI0eOjCeeeCJqampi586dpXwsAKBMZDKwurq6orq6OiZMmDB4bMqUKXHkyJE4ffp0dHV1RX19/ZDXTJ06NQ4cOBAREd3d3b/zPADAzcjkv0V47ty5yOfzQ45d%2Bbqvr%2B%2Ba50ePHh19fX0f%2BfoPn79ZPT090dvbO%2BRYLlcZtbW1Se5/RUVFtvo4l8vOeq/MNmszzoIsztTeJcJsi6kcZ5vJwKqsrLzq96WufF1VVRX5fD7OnDkz5Hx/f39UVVVFxAcx1t/ff9X5mpqaJOvbunVrbNy4ccixVatWxerVq5PcP6tqaqpKvYRhGzs2//EXUfbsXT7MbIunnGabycCqq6uLkydPxrFjx2L8%2BPEREXHw4MGYOHFijBkzJurr6%2BONN94Y8pru7u6oq6sbfH1XV9dV5x966KEk62tqaop58%2BYNOZbLVcaJE%2BeS3P%2BKrJV%2B6ucvpoqKETF2bD5Onz4fly5dLvVyykrW9m2EvcsHzLZ4ijnbUv0FKZOBNXny5GhoaIgNGzZES0tLnDhxIjZt2hTLly%2BPiIjGxsb4%2Bte/Hlu2bIkVK1bEL3/5y9ixY0ds2rQpIiKWL18eq1atigULFkRDQ0O0t7fH8ePHo7GxMcn6amtrr/pxYG/vmRgYuLO/IbP4/JcuXc7kukkri3vA3i0esy2ecpptJgMrIqKtrS1aWlpi/vz5MWLEiFi6dGk0NzdHRERNTU1s3rw5Wltbo62tLcaNGxfPPvtszJ49OyIi5syZE88991w8//zzcfTo0Zg6dWq8/PLLUV1dXcpHAgDKRGYDa/z48dHW1vaR56dNmxbf//73P/L8kiVLYsmSJcVYGgBwh8veL0MAANzmBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAklsnAOn78eDQ3N8fMmTNj1qxZ0draGgMDA0OuOXToUDz%2B%2BOMxY8aMeOCBB%2BI73/lOiVYLANxpMhlYa9asicrKytizZ090dHTE3r17Y8uWLYPnL168GF/5yldi2rRpsW/fvvjud78b7e3tsWvXrtItGgC4Y2QusA4fPhz79%2B%2BPdevWRT6fj0mTJkVzc3O0t7cPXtPZ2Rk9PT2xevXqGDVqVNx3333x2GOPDbkGAKBYMhdYXV1dUV1dHRMmTBg8NmXKlDhy5EicPn168Jp77rknRo0aNXjN1KlT48CBA7d8vQDAnSdzgXXu3LnI5/NDjl35uq%2Bv73dec%2BU8AEAx5Uq9gOGqrKyM8%2BfPDzl25euqqqrfec2V88XW09MTvb29Q47lcpVRW1ub9H0qKrLVx7lcdtZ7ZbZZm3EWZHGm9i4RZltM5TjbzAVWXV1dnDx5Mo4dOxbjx4%2BPiIiDBw/GxIkTY8yYMYPXvPvuuzEwMBC53AeP2N3dHXV1dbdkjVu3bo2NGzcOObZq1apYvXr1LXn/21VNza0J3JTGjs1//EWUPXuXDzPb4imn2WYusCZPnhwNDQ2xYcOGaGlpiRMnTsSmTZti%2BfLlg9fMmjUrampq4qWXXoo1a9bEoUOH4nvf%2B16sXbv2lqyxqakp5s2bN%2BRYLlcZJ06cS/o%2BWSv91M9fTBUVI2Ls2HycPn0%2BLl26XOrllJWs7dsIe5cPmG3xFHO2pfoLUuYCKyKira0tWlpaYv78%2BTFixIhYunRpNDc3D57P5XKxefPmaGlpiblz50ZlZWU89thjsWzZsluyvtra2qt%2BHNjbeyYGBu7sb8gsPv%2BlS5czuW7SyuIesHeLx2yLp5xmm8nAGj9%2BfLS1tf3Oaz7xiU/Eq6%2B%2BeotWBADw/2Xvs3oAgNucwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgscwG1vHjx6O5uTlmzpwZs2bNitbW1hgYGBg8//bbb8ejjz4aM2bMiHnz5sW2bduGvH779u3R2NgY06dPj2XLlsVbb711qx8BAChTmQ2sNWvWRGVlZezZsyc6Ojpi7969sWXLloiIOHXqVDz55JOxdOnS6OzsjNbW1njhhRfinXfeiYiIffv2xfr16%2BPFF1%2BMzs7OWLx4cTz11FNx/vz5Ej4RAFAuMhlYhw8fjv3798e6desin8/HpEmTorm5Odrb2yMiYvfu3VFdXR0rVqyIXC4Xc%2BbMiUWLFg2e37ZtWyxcuDAaGhpi5MiR8cQTT0RNTU3s3LmzlI8FAJSJTAZWV1dXVFdXx4QJEwaPTZkyJY4cORKnT5%2BOrq6uqK%2BvH/KaqVOnxoEDByIioru7%2B3eeBwC4GblSL%2BBGnDt3LvL5/JBjV77u6%2Bu75vnRo0dHX1/fR77%2Bw%2BdvVk9PT/T29g45lstVRm1tbZL7X1FRka0%2BzuWys94rs83ajLMgizO1d4kw22Iqx9lmMrAqKyuv%2Bn2pK19XVVVFPp%2BPM2fODDnf398fVVVVEfFBjPX39191vqamJsn6tm7dGhs3bhxybNWqVbF69eok98%2BqmpqqUi9h2MaOzX/8RZQ9e5cPM9viKafZZjKw6urq4uTJk3Hs2LEYP358REQcPHgwJk6cGGPGjIn6%2Bvp44403hrymu7s76urqBl/f1dV11fmHHnooyfqamppi3rx5Q47lcpVx4sS5JPe/Imuln/r5i6miYkSMHZuP06fPx6VLl0u9nLKStX0bYe/yAbMtnmLOtlR/QcpkYE2ePDkaGhpiw4YN0dLSEidOnIhNmzbF8uXLIyKisbExvv71r8eWLVtixYoV8ctf/jJ27NgRmzZtioiI5cuXx6pVq2LBggXR0NAQ7e3tcfz48WhsbEyyvtra2qt%2BHNjbeyYGBu7sb8gsPv%2BlS5czuW7SyuIesHeLx2yLp5xmm8nAiohoa2uLlpaWmD9/fowYMSKWLl0azc3NERFRU1MTmzdvjtbW1mhra4tx48bFs88%2BG7Nnz46IiDlz5sRzzz0Xzz//fBw9ejSmTp0aL7/8clRXV5fykQCAMpHZwBo/fny0tbV95Plp06bF97///Y88v2TJkliyZEkxlgYA3OGy98sQAAC3OYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEMhlYBw4ciJUrV8b9998fc%2BfOja997Wvx3nvvDbnm0KFD8fjjj8eMGTPigQceiO985zs39Z4/%2B9nPYtGiRTF9%2BvRYsGBBvP766zd1PwCgfGUusPr7%2B%2BNLX/pSzJgxI/793/89fvjDH8bJkyfjb//2bwevuXjxYnzlK1%2BJadOmxb59%2B%2BK73/1utLe3x65du27oPd999914%2Bumn46/%2B6q/iF7/4RTz99NOxZs2aOHr0aKrHAgDKSOYC68iRI/GpT30qVq1aFaNGjYqamppoamqKzs7OwWs6Ozujp6cnVq9eHaNGjYr77rsvHnvssWhvb7%2Bh99y%2BfXvMnDkzHn744cjlcvHII4/E5z73udi6dWuqxwIAykjmAuuTn/xkvPLKK1FRUTF47Mc//nF8%2BtOfHvy6q6sr7rnnnhg1atTgsalTp8aBAwdu6D27u7ujvr5%2ByLGbuR8AUN5ypV7AzSgUCvHNb34zXn/99XjttdcGj587dy7y%2BfyQa/P5fPT19d3Q%2B1zrfqNHj/7I%2B/X09ERvb%2B%2BQY7lcZdTW1t7Q%2B3%2BUiops9XEul531Xplt1macBVmcqb1LhNkWUznONrOBdfbs2fibv/mb%2BNWvfhWvvfZa3HvvvYPnKisr4/z580OuP3/%2BfFRVVd3Qe%2BXz%2Bejv7x9yrL%2B//yPvt3Xr1ti4ceOQY6tWrYrVq1ff0PuXi5qaG5t/KY0dm//4iyh79i4fZrbFU06zzWRg/frXv44vf/nL8Qd/8AfR0dER48aNG3K%2Brq4u3n333RgYGIhc7oNH7O7ujrq6uht6v/r6%2BvjVr3415Fh3d3f80R/90TWvb2pqinnz5g05lstVxokT527o/T9K1ko/9fMXU0XFiBg7Nh%2BnT5%2BPS5cul3o5ZSVr%2BzbC3uUDZls8xZxtqf6ClLnAOnXqVDz%2B%2BOMxe/bsaG1tjREjrv4f61mzZkVNTU289NJLsWbNmjh06FB873vfi7Vr197Qey5evDj%2B6Z/%2BKXbu3Bl/%2Bqd/Grt37479%2B/fHM888c83ra2trr/pxYG/vmRgYuLO/IbP2/DOf%2BVGplzAsu9bMLfUSylbW9m5ExKVLlzO57iww2%2BIpp9lmLrB%2B8IMfxJEjR2LXrl3xox8N/QfgW2%2B9FRERuVwuNm/eHC0tLTF37tyorKyMxx57LJYtW3ZD7zllypT49re/Hd/4xjfimWeeiT/8wz%2BMf/zHf4x77rnnpp8HACg/mQuslStXxsqVKz/2uk984hPx6quvJnvfBx98MB588MFk9wMAylf2fhkCAOA2J7AAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBBYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAieVKvQAAyIqZz/yo1EsYll1r5pZ6CXcsn2ABACQmsAAAEhNYAACJCSwAgMQEFgBAYgILACAxgQUAkJjAAgBITGABACQmsAAAEhNYAACJ%2BW8RXsPx48fj7/7u72L//v1RUVERixcvjr/%2B67%2BOXM64gNtflv57ef5beZQrn2Bdw5o1a6KysjL27NkTHR0dsXfv3tiyZUuplwUAZITA%2Bi2HDx%2BO/fv3x7p16yKfz8ekSZOiubk52tvbS700ACAjBNZv6erqiurq6pgwYcLgsSlTpsSRI0fi9OnTJVwZAJAVfqnot5w7dy7y%2BfyQY1e%2B7uvri7Fjx37sPXp6eqK3t3fIsVyuMmpra9MtNCIqKrLVx7lcdtabtdlGZGe%2BZltcWZuv2RZXVuZ7ZbZZnPFHEVi/pbKyMs6fPz/k2JWvq6qqruseW7dujY0bNw459pd/%2BZfx9NNPp1nk/9PT0xOPT%2ByKpqam5PF2pzPb4jHb4jLf4jHb4unp6Yl//udXymq25ZOKidTV1cXJkyfj2LFjg8cOHjwYEydOjDFjxlzXPZqamuIHP/jBkD9NTU3J19rb2xsbN2686tMybp7ZFo/ZFpf5Fo/ZFk85ztYnWL9l8uTJ0dDQEBs2bIiWlpY4ceJEbNq0KZYvX37d96itrS2bAgcAhs8nWNfQ1tYWAwMDMX/%2B/PjzP//zePDBB6O5ubnUywIAMsInWNcwfvz4aGtrK/UyAICMqnj%2B%2BeefL/UiuHFVVVVx//33X/cv4HP9zLZ4zLa4zLd4zLZ4ym22dxUKhUKpFwEAUE78DhYAQGICCwAgMYEFAJCYwAIASExgAQAkJrAAABITWAAAiQksAIDEBFYGvPfee9HY2Bj79u0r9VLK0vXM90tf%2BlJMmzYtZsyYMfjn5z//%2BS1cZbYcOHAgVq5cGffff3/MnTs3vva1r8V77713zWt/9rOfxaJFi2L69OmxYMGCeP3112/xarNlOLO1b4dv79698eijj8ZnP/vZmDt3bqxfvz76%2B/uvea29OzzDmW1Z7N0Ct7Vf/OIXhYcffrhQX19fePPNN0u9nLJzvfOdNWtWYd%2B%2BfbdwZdl1/vz5wty5cwvf%2Bta3Cu%2B//37hvffeK3z5y18u/MVf/MVV1x46dKgwbdq0wk9%2B8pPCxYsXC//2b/9W%2BMxnPlP43//93xKs/PY3nNkWCvbtcB0/frwwbdq0wr/8y78ULl26VDh69GjhC1/4QuFb3/rWVdfau8MznNkWCuWxd32CdRvbvn17fPWrX421a9eWeill6Xrn%2B5vf/CZOnToV99133y1aWbYdOXIkPvWpT8WqVati1KhRUVNTE01NTdHZ2XnVtdu3b4%2BZM2fGww8/HLlcLh555JH43Oc%2BF1u3bi3Bym9/w5mtfTt848aNi//4j/%2BIZcuWxV133RUnT56M999/P8aNG3fVtfbu8AxntuWydwXWbeyBBx6In/zkJ/HII4%2BUeill6Xrn%2B1//9V9RVVUVa9eujdmzZ8cXvvCF6OjouEWrzJ5PfvKT8corr0RFRcXgsR//%2BMfx6U9/%2Bqpru7u7o76%2BfsixqVOnxoEDB4q%2Bziwazmzt2xvz%2B7//%2BxER8Sd/8iexaNGiuPvuu2PZsmVXXWfvDt/1zrZc9m6u1Avgo919992lXkJZu975XrhwIaZPnx5r166Nurq62LdvXzz99NNRVVUVCxYsKPIqs61QKMQ3v/nNeP311%2BO111676vy5c%2Bcin88POTZ69Ojo6%2Bu7VUvMrI%2BbrX17c3bv3h2nTp2Kr371q7F69ep45ZVXhpy3d2/cx822XPauT7DgYyxdujReeeWVuO%2B%2B%2B2LkyJHxwAMPxNKlS2PXrl2lXtpt7ezZs7F69erYsWNHvPbaa3HvvfdedU0%2Bn7/ql1z7%2B/ujqqrqVi0zk65ntvbtzRk9enRMmDAh1q1bF3v27IlTp04NOW/v3riPm2257F2BBR%2Bjo6Pjqm/sCxcuxO/93u%2BVaEW3v1//%2BtfxxS9%2BMc6ePRsdHR3XDICIiPr6%2Bujq6hpyrLu7O%2Brq6m7FMjPpemdr3w7ff/7nf8af/dmfxYULFwaPXbhwIUaOHHnVp1X27vAMZ7blsncFFnyMs2fPxvr16%2BO///u/4/Lly/HTn/40fvjDH0ZTU1Opl3ZbOnXqVDz%2B%2BOPx2c9%2BNl599dVr/hLrFYsXL479%2B/fHzp07Y2BgIHbu3Bn79%2B%2BPJUuW3MIVZ8dwZmvfDt%2B9994b/f398dJLL8WFCxfif/7nf%2BLv//7vY/ny5TFq1Kgh19q7wzOc2ZbN3i31v8bI9bnW/43A9OnTC//6r/9aohWVl9%2Be74dne/ny5cK3v/3twuc///nCZz7zmcLChQsLu3btKtVSb3ubN28u1NfXF/74j/%2B4MH369CF/CoWr9%2B3Pf/7zwuLFiwvTp08vLFy4sPDTn/60VEu/7Q1ntvbtjenq6iqsXLmyMHPmzMLnP//5wj/8wz8U3n///UKhYO/erOudbbns3bsKhUKh1JEHAFBO/IgQACAxgQUAkJjAAgBITGABACQmsAAAEhNYAACJCSwAgMQEFgBAYgILACAxgQUAkJjAAgBITGABACQmsAAAEhNYAACJCSwAgMT%2BDzPN8GPS8VMFAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common9088482465624116579">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">10673</td>
        <td class="number">49.4%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.0</td>
        <td class="number">8235</td>
        <td class="number">38.1%</td>
        <td>
            <div class="bar" style="width:77%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.5</td>
        <td class="number">1910</td>
        <td class="number">8.8%</td>
        <td>
            <div class="bar" style="width:18%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.0</td>
        <td class="number">611</td>
        <td class="number">2.8%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.5</td>
        <td class="number">161</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.5</td>
        <td class="number">7</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme9088482465624116579">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">10673</td>
        <td class="number">49.4%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.5</td>
        <td class="number">1910</td>
        <td class="number">8.8%</td>
        <td>
            <div class="bar" style="width:18%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.0</td>
        <td class="number">8235</td>
        <td class="number">38.1%</td>
        <td>
            <div class="bar" style="width:77%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.5</td>
        <td class="number">161</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.0</td>
        <td class="number">611</td>
        <td class="number">2.8%</td>
        <td>
            <div class="bar" style="width:6%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1.5</td>
        <td class="number">1910</td>
        <td class="number">8.8%</td>
        <td>
            <div class="bar" style="width:23%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.0</td>
        <td class="number">8235</td>
        <td class="number">38.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.5</td>
        <td class="number">161</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.0</td>
        <td class="number">611</td>
        <td class="number">2.8%</td>
        <td>
            <div class="bar" style="width:8%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.5</td>
        <td class="number">7</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_grade">grade<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>11</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.1%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>7.6579</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>3</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>13</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-8572833814874630454">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAASlJREFUeJzt3MENgkAUQEE1lmQR9uTZnizCntYGzAuQEBBn7iR7efkssJzHGOMEfHXZegGwZ9etF3Bkt8dr9jXv532FlbCUCQJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBB87r4zPpHfFxMEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIIThQcw9xSiE4jTmSAQBALBLdYMS36owG8zQSAIBIJAIAgEgk36H/L3xulMEAgCgSAQCOcxxth6EVvw0m99R9i32KSzmiM8DPjbCQJT2INAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFA%2BAAxvh/7AwsudgAAAABJRU5ErkJggg%3D%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-8572833814874630454,#minihistogram-8572833814874630454"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-8572833814874630454">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-8572833814874630454"
                                                  aria-controls="quantiles-8572833814874630454" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-8572833814874630454" aria-controls="histogram-8572833814874630454"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-8572833814874630454" aria-controls="common-8572833814874630454"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-8572833814874630454" aria-controls="extreme-8572833814874630454"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-8572833814874630454">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>3</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>6</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>7</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>7</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>8</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>10</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>13</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>10</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>1</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>1.1732</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.1532</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>1.1351</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>7.6579</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>0.9288</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.78824</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>165388</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>1.3764</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-8572833814874630454">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3X9QlWX%2B//EXcFAOIEIh1ba2msBa6iaKkkY2oehaYo0/oo11V2e1VlFzzTADR%2BdjqLS0mlHmmMZqboua1OCYYumauv6A1nSyxQEby1lNjqIGBzDg8P3DkW9nqQ3b65x7Dz4fM43DfR/Oed/XnDk9ue8b9Wtubm4WAAAAjPG3egAAAID2hsACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwjMACAAAwzGb1ADcKh6Pa6hEs5e/vp5tuClFVlVMuV7PV47Q7rK/nsLaew9p6Fut7VZcunSx5Xc5gwSv8/f3k5%2Bcnf38/q0dpl1hfz2FtPYe19SzW11oEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGEEFgAAgGE2qwcAcOMZuXy/1SNcl9LsX1o9AgAfwxksAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAwwgsAAAAw3w2sI4fP660tDTFx8crMTFRL7zwgr755htJ0p49e5SSkqK%2Bfftq5MiR2r17t9v3rl69WkOGDFHfvn01YcIEff755y37amtrNW/ePCUkJKh///7KyMiQ0%2Bn06rEBAADf5pOB5XK59NRTT2nEiBE6fPiwNm/erH379mn16tU6deqUZsyYoaefflqlpaWaMWOGZs2apXPnzkmSCgsLtX79eq1Zs0aHDh1Sr169NHPmTDU3N0uSFi1apLNnz2rHjh0qLi7W2bNnlZuba%2BXhAgAAH%2BOTgXX58mU5HA65XK6WMPL395fdbldhYaHi4%2BM1bNgw2Ww2PfTQQxowYIAKCgokSRs3btQTTzyhmJgYdezYUc8884zOnDmjQ4cOqa6uTkVFRZo5c6bCw8N18803a86cOdqyZYvq6uqsPGQAAOBDbFYP8GNERERo4sSJysnJ0YsvvqimpiYNHTpUEydO1IwZMxQbG%2Bv2%2BOjoaJWVlUmSKioqNGXKlJZ9gYGB6tatm8rKyhQeHq6Ghga37%2B/Ro4fq6%2Bt16tQp3XXXXW2ar7KyUg6Hw22bzRasqKioH3vIPi8gwN/tT5jF%2Bnoea2se71vPYn2t5ZOB5XK5FBQUpPnz52vcuHH64osvNH36dK1YsUJOp1N2u93t8UFBQaqtrZWk/7i/pqZGkhQcHNyy79pjr%2Bc%2BrIKCAuXl5bltS09P18yZM9t%2BkO1UWJj9hx%2BEH4319RzW1nNYW89ifa3hk4G1c%2BdO7dixQ9u3b5ckxcTEKD09XdnZ2erXr5/q6%2BvdHl9fX6%2BQkBBJV4Pp%2B/ZfC6u6urqWx1%2B7NBgaGtrm%2BVJTU5WUlOS2zWYL1sWLN%2B7N8gEB/goLs%2Bvrr%2BvU1OSyepx2h/X1PNbWPN63nsX6XhUREWLJ6/pkYJ09e7blNwavsdlsCgwMVGxsrI4fP%2B62r6KiQr1795Z0NcbKy8v14IMPSpIaGhp06tQpxcbGqnv37goMDFRFRYXuueceSdLJkydbLiO2VVRUVKvLgQ5HtRobb9w3%2BDVNTS7WwYNYX89hbT2HtfUs1tcaPnlhNjExUQ6HQ6%2B//rqampp0%2BvRprVy5UikpKRo9erQOHz6sbdu2qbGxUdu2bdPhw4f1yCOPSJLGjh2rt956S2VlZbpy5YpeeuklRUZGKj4%2BXna7XSNHjlRubq6qqqpUVVWl3NxcjRo1SkFBQRYfNQAA8BU%2BeQYrOjpaq1at0vLly/XGG2%2BoU6dOGj16tNLT09WhQwe9%2Buqrys3NVWZmpm6//Xa98sor6t69uyRp3Lhxqq6uVnp6uqqqqtSnTx%2BtWrVKgYGBkqQFCxYoJydHKSkpamho0NChQzV//nwrDxcAAPgYv%2BZrf88BPMrhqLZ6BEvZbP6KiAjRxYtOTlV7gK%2Bt78jl%2B60e4bqUZv/SZ9bWl/ja%2B9bXsL5XdenSyZLX9clLhAAAAP/LCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDCCwAAADDfDawLl26pIyMDCUkJGjAgAGaNm2aKisrJUlHjx7V%2BPHjFRcXp6SkJG3atMntewsLC5WcnKy%2BfftqzJgxOnLkSMu%2BpqYm5eTkaPDgwYqLi9PUqVNbnhcAAKAtfDawZsyYodraWu3cuVO7d%2B9WQECA5s%2Bfr8uXL%2BvJJ5/Uo48%2BqpKSEmVnZ2vJkiU6duyYJOnQoUNatGiRli5dqpKSEo0ePVpTp05VXV2dJGnlypXav3%2B/3nnnHe3du1dBQUHKysqy8lABAICPsVk9wI/x6aef6ujRo/r73/%2Bu0NBQSdKiRYvkcDhUXFys8PBwpaWlSZIGDRqklJQUbdiwQb/4xS%2B0adMmPfzww%2Brfv78kaeLEiSooKNC2bds0duxYbdq0SXPmzNFtt90mScrMzFRiYqJOnz6trl27WnPAACwVn7nd6hGuy/uz7rN6BOCG55OBdezYMUVHR2vjxo16%2B%2B23VVdXp/vvv19z585VeXm5YmNj3R4fHR2tzZs3S5IqKio0duzYVvvLyspUXV2tr776yu37IyMj1blzZ504caLNgVVZWSmHw%2BG2zWYLVlRU1I853HYhIMDf7U%2BYxfri22w233gf8L71LNbXWj4ZWJcvX9aJEyfUu3dvFRYWqr6%2BXhkZGZo7d64iIyNlt9vdHh8UFKTa2lpJktPp/N79TqdTkhQcHNxq/7V9bVFQUKC8vDy3benp6Zo5c2abn6O9Cguz//CD8KOxvpCkiIgQq0e4LrxvPYv1tYZPBlaHDh0kXb1817FjR4WGhmrWrFl67LHHNGbMGNXX17s9vr6%2BXiEhVz9w7Hb7d%2B6PiIhoCa9r92N91/e3RWpqqpKSkty22WzBunix7ZHW3gQE%2BCsszK6vv65TU5PL6nHaHdYX3%2BYrnzW8bz2L9b3Kqh84fDKwoqOj5XK51NDQoI4dO0qSXK6rb5677rpLf/nLX9weX1FRoZiYGElSTEyMysvLW%2B0fMmSIOnfurFtuuUUVFRUtlwkdDocuXbrU6rLjfxIVFdXqcqDDUa3Gxhv3DX5NU5OLdfAg1heSfO49wPvWs1hfa/jkhdnBgwera9euev755%2BV0OlVVVaVly5Zp2LBhGjVqlM6fP6/8/Hw1NDTo4MGDKioqarnvaty4cSoqKtLBgwfV0NCg/Px8XbhwQcnJyZKkMWPGaOXKlTp9%2BrRqamq0ePFiDRw4UHfccYeVhwwAAHyIT57BCgwM1Pr167V06VKNGDFCV65cUVJSkjIzMxUWFqa1a9cqOztbK1as0E033aSsrCzde%2B%2B9kq7%2BVuGCBQu0cOFCnTt3TtHR0Vq9erXCw8MlXb1XqrGxUWlpaXI6nUpISNDy5cutPFwAAOBj/Jqbm5utHuJG4HBUWz2CpWw2f0VEhOjiRSenqj3A19Z35PL9Vo/QrvnKX9Pga%2B9bX8P6XtWlSydLXtcnLxECAAD8LyOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADLM8sK5cuWL1CAAAAEZ5NbDq6uqUkZGhlStXtmwbPny4srKy9M0333hzFAAAAI/xamAtWbJER48e1YABA1q2ZWVlqaSkRMuWLfPmKAAAAB7j1cDatWuXcnJyFB8f37ItOTlZ2dnZ2rp1qzdHAQAA8BivBpbT6VSnTp1abY%2BIiFB1dbU3RwEAAPAYrwZWXFycVq1apaamppZtzc3N%2BvOf/6w%2Bffp4cxQAAACPsXnzxWbPnq0JEyaotLRUvXr1kp%2Bfn44fP65Lly5p7dq13hwFAADAY7x6Bqt3797aunWrRo0apYaGBrlcLo0aNUrvv/%2B%2B7rnnHm%2BOAgAA4DFePYMlSbfffrtmz57t7ZcFAADwGq8Glsvl0tatW/Xxxx%2BroaFBzc3NbvuXLFnizXEAAAA8wquBlZOTo3Xr1qlnz54KDQ315ksDAAB4jVcD67333lNWVpbS0tK8%2BbIAAABe5dWb3K9cuaL777/fmy8JAADgdV4NrPvvv1979%2B715ksCAAB4nVcvEfbp00cvvviiDhw4oB49eigwMNBt//Tp0705DgAAgEd4NbDefvtt3Xzzzfrss8/02Wefue3z8/MjsAAAQLvg1cDatWuXN18OAADAEl69B%2BuakpIS/fWvf1VNTY0qKirU0NBgxRgAAAAe4dUzWDU1Nfrd736no0ePys/PT/fdd59yc3N16tQp5efn69Zbb/XmOAAAAB7h1TNYf/rTn%2BTn56edO3cqKChIkpSRkaHg4GC9%2BOKL3hwFAADAY7waWLt371ZGRoa6du3asu3OO%2B/UggULdODAAW%2BOAgAA4DFeDayqqip16dKl1fbQ0FDV1dV5cxQAAACP8Wpg9enTR9u2bWu1fd26dbr77ru9OQoAAIDHePUm99mzZ2vSpEk6cuSIGhsbtXLlSlVUVOizzz7TmjVrvDkKAACAx3j1DFa/fv1UUFCgsLAw/exnP9Mnn3yi2267TRs2bFBCQoI3RwEAAPAYr57BkqSePXvyG4MAAKBd82pg5eXl/cf9/FM5AACgPfBqYG3ZssXt68bGRlVVVSkwMFBxcXHeHAUAAMBjLP%2B3CGtqajR37lzuwQIAAO2GJf8W4beFhobq6aef1ptvvmn1KAAAAEZYHljS/79UCAAA0B549RLhu%2B%2B%2B6/Z1c3OzqqurVVBQwD1YAACg3fBqYD333HOtB7DZ1K9fPy1YsMCbowAAAHiMVwOrrKzMmy8HAABgif%2BJe7AAAADaE6%2BewUpKSpKfn1%2BbHvvhhx96eBoAAADP8Gpg/frXv9bLL7%2BsxMREDRw4UB06dNCxY8dUVFSkcePGKTIy0pvjAAAAeIRXA%2BvQoUOaOnWqfv/737ds%2B9WvfqXevXtr165dWrhwoTfHAQAA8Aiv3oN1%2BPBhjRw5stX2xMRElZSUeHMUAAAAj/FqYEVFRWnfvn2ttu/cuVM//elPvTkKAACAx3j1EuHkyZO1cOFCHTt2TH369FFzc7M%2B/vhj7dy5U8uXL/fmKAAAAB7j1cAaP368goKCtG7dOhUXF0uS7r77br322mt64IEHvDkKAACAx3g1sCQpJSVFKSkp3n5ZAAAAr%2BEvGgUAADDM5wOrqalJEyZMcPt3Dvfs2aOUlBT17dtXI0eO1O7du92%2BZ/Xq1RoyZIj69u2rCRMm6PPPP2/ZV1tbq3nz5ikhIUH9%2B/dXRkaGnE6n144HAAD4Pp8PrLy8PJWWlrZ8ferUKc2YMUNPP/20SktLNWPGDM2aNUvnzp2TJBUWFmr9%2BvVas2aNDh06pF69emnmzJlqbm6WJC1atEhnz57Vjh07VFxcrLNnzyo3N9eSYwMAAL7JpwPrwIEDKi4u1vDhw1u2FRYWKj4%2BXsOGDZPNZtNDDz2kAQMGqKCgQJK0ceNGPfHEE4qJiVHHjh31zDPP6MyZMzp06JDq6upUVFSkmTNnKjw8XDfffLPmzJmjLVu2qK6uzqrDBAAAPsarN7nX1NQoNDTUyHNduHBBmZmZeu2115Sfn9%2ByvaKiQrGxsW6PjY6OVllZWcv%2BKVOmtOwLDAxUt27dVFZWpvDwcDU0NLh9f48ePVRfX69Tp07prrvuatNslZWVcjgcbttstmBFRUVd72G2GwEB/m5/wizWF99ms/nG%2B4D3rWexvtbyamA98sgjWrFihXr16vVfPY/L5dKzzz6rSZMmqWfPnm77nE6n7Ha727agoCDV1tb%2B4P6amhpJUnBwcMu%2Ba4%2B9nvuwCgoKlJeX57YtPT1dM2fObPNztFdhYfYffhB%2BNNYXkhQREWL1CNeF961nsb7W8GpgXblypVXc/BirVq1Shw4dNGHChFb77Ha76uvr3bbV19crJCTkB/dfC6u6urqWx1%2B7NHg9Z95SU1OVlJTkts1mC9bFizfuzfIBAf4KC7Pr66/r1NTksnqcdof1xbf5ymcN71vPYn2vsuoHDq8GVlpamqZPn660tDTdcccdCgoKcts/YMCANj3Pe%2B%2B9p8rKSsXHx0tSSzB98MEHSktL0/Hjx90eX1FRod69e0uSYmJiVF5ergcffFCS1NDQoFOnTik2Nlbdu3dXYGCgKioqdM8990iSTp482XIZsa2ioqJaXQ50OKrV2HjjvsGvaWpysQ4exPpCks%2B9B3jfehbraw2vBtbLL78s6epv6v07Pz8//fOf/2zT82zfvt3t62t/RcPSpUt18uRJvfnmm9q2bZuGDx%2Bu4uJiHT58WJmZmZKksWPH6pVXXtGQIUPUvXt3LVu2TJGRkYqPj1dgYKBGjhyp3Nzclllzc3M1atSoVjEIAADwfbwaWB9%2B%2BKHHX6NHjx569dVXlZubq8zMTN1%2B%2B%2B165ZVX1L17d0nSuHHjVF1drfT0dFVVValPnz5atWqVAgMDJUkLFixQTk6OUlJS1NDQoKFDh2r%2B/PkenxsAALQffs3X/gIoeJTDUW31CJay2fwVERGiixednKr2AF9b35HL91s9Qrv2/qz7rB6hTXztfetrWN%2BrunTpZMnr8rubAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhhFYAAAAhtmsHgAAYNbI5futHqHNSrN/afUIgEdwBgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwAgsAAMAwnw2ssrIyTZo0SQMHDtR9992njIwMVVVVSZKOHj2q8ePHKy4uTklJSdq0aZPb9xYWFio5OVl9%2B/bVmDFjdOTIkZZ9TU1NysnJ0eDBgxUXF6epU6eqsrLSq8cGAAB8m08GVn19vSZPnqy4uDjt27dPW7du1aVLl/T888/r8uXLevLJJ/Xoo4%2BqpKRE2dnZWrJkiY4dOyZJOnTokBYtWqSlS5eqpKREo0eP1tSpU1VXVydJWrlypfbv36933nlHe/fuVVBQkLKysqw8XAAA4GN8MrDOnDmjnj17Kj09XR06dFBERIRSU1NVUlKi4uJihYeHKy0tTTabTYMGDVJKSoo2bNggSdq0aZMefvhh9e/fX4GBgZo4caIiIiK0bdu2lv1TpkzRbbfdptDQUGVmZuqjjz7S6dOnrTxkAADgQ2xWD/Bj3HnnnXrjjTfctu3YsUO9evVSeXm5YmNj3fZFR0dr8%2BbNkqSKigqNHTu21f6ysjJVV1frq6%2B%2Bcvv%2ByMhIde7cWSdOnFDXrl3bNF9lZaUcDofbNpstWFFRUW0%2BxvYmIMDf7U%2BYFRDgr/jM7VaPAfwofC54Bp%2B71vLJwPq25uZmLV%2B%2BXLt379Zbb72ldevWyW63uz0mKChItbW1kiSn0/m9%2B51OpyQpODi41f5r%2B9qioKBAeXl5btvS09M1c%2BbMNj9HexUWZv/hBwG4ofC54FmsrzV8OrBqamo0b948HT9%2BXG%2B99ZZ%2B/vOfy263q7q62u1x9fX1CgkJkSTZ7XbV19e32h8REdESXtfux/qu72%2BL1NRUJSUluW2z2YJ18WLbI629CQjwV1iYXV9/XaemJpfV47Q7/IQKX8bngmfwuXtVRETb//9tks8G1pdffqkpU6boJz/5iTZv3qybbrpJkhQbG6v9%2B/e7PbaiokIxMTGSpJiYGJWXl7faP2TIEHXu3Fm33HKLKioqWi4TOhwOXbp0qdVlx/8kKiqq1eVAh6NajY037hv8mqYmF%2BsAwA2fC57F%2BlrDJ3/svXz5sn7729%2BqX79%2BWrNmTUtcSVJycrLOnz%2Bv/Px8NTQ06ODBgyoqKmq572rcuHEqKirSwYMH1dDQoPz8fF24cEHJycmSpDFjxmjlypU6ffq0ampqtHjxYg0cOFB33HGHJccKAAB8j0%2BewdqyZYvOnDmj999/X9u3u9/Ye%2BTIEa1du1bZ2dlasWKFbrrpJmVlZenee%2B%2BVJA0aNEgLFizQwoULde7cOUVHR2v16tUKDw%2BXdPVeqcbGRqWlpcnpdCohIUHLly/3%2BjECAADf5dfc3Nxs9RA3Aoej%2Bocf1I7ZbP6KiAjRxYtOTlV7gM3mr%2BTcvVaPAVy30uxf8rngIXzuXtWlSydLXtcnLxECAAD8LyOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADCOwAAAADLNZPQAA4MYVn7nd6hGuy/uz7rN6BPgIzmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYRmABAAAYZrN6AAAAfMXI5futHuG6lGb/0uoRblicwQIAADCMwAIAADCMwAIAADCMwAIAADCMwAIAADCM3yIEvoev/bYQAOB/B2ewAAAADCOwAAAADCOwvsOFCxc0bdo0xcfHKyEhQdnZ2WpsbLR6LAAA4CMIrO8wa9YsBQcHa%2B/evdq8ebMOHDig/Px8q8cCAAA%2Bgpvc/80XX3yhw4cP66OPPpLdblfXrl01bdo0/fGPf9TkyZOtHg8AgDaLz9xu9Qht9v6s%2B6wewSjOYP2b8vJyhYeH65ZbbmnZ1qNHD505c0Zff/21hZMBAABfwRmsf%2BN0OmW32922Xfu6trZWYWFhP/gclZWVcjgcbttstmBFRUWZG9THBAT4%2B9RPUgAA77LZ2tc5HwLr3wQHB6uurs5t27WvQ0JC2vQcBQUFysvLc9s2ffp0zZgxw8yQPqiyslK/vbVcqampN3RoekplZaUKCgpYXw9gbT2HtfUs1tda7SsXDYiJidGlS5d0/vz5lm0nT57Urbfeqk6dOrXpOVJTU7Vlyxa3/1JTUz01sk9wOBzKy8trdWYPZrC%2BnsPaeg5r61msr7U4g/VvunXrpv79%2B2vx4sX6v//7P128eFGvvfaaxo0b1%2BbniIqK4qcFAABuYJzB%2Bg4rVqxQY2Ojhg4dqscee0z333%2B/pk2bZvVYAADAR3AG6ztERkZqxYoVVo8BAAB8VMDChQsXWj0EbgwhISEaOHBgm39ZANeH9fUc1tZzWFvPYn2t49fc3Nxs9RAAAADtCfdgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgAQAAGEZgwWuampo0YcIEPffcc1aP0m5cunRJGRkZSkhI0IABAzRt2jRVVlZaPVa7cfz4caWlpSk%2BPl6JiYl64YUX9M0331g9VrtRVVWl5ORkHTp0qGXb0aNHNX78eMXFxSkpKUmbNm2ycELf8l3ruWPHDj3yyCPq16%2BfkpKSlJeXJ5fLZeGUNw4CC16Tl5en0tJSq8doV2bMmKHa2lrt3LlTu3fvVkBAgObPn2/1WO2Cy%2BXSU089pREjRujw4cPavHmz9u3bp9WrV1s9Wrvw8ccfKzU1VV9%2B%2BWXLtsuXL%2BvJJ5/Uo48%2BqpKSEmVnZ2vJkiU6duyYhZP6hu9az08//VQZGRmaNWuWSktLtXr1am3ZskX5%2BfnWDXoDIbDgFQcOHFBxcbGGDx9u9SjtxqeffqqjR49q6dKlCgsLU2hoqBYtWqQ5c%2BZYPVq7cPnyZTkcDrlcLl0HgIszAAADpElEQVT7F8X8/f1lt9stnsz3FRYWas6cOfrDH/7gtr24uFjh4eFKS0uTzWbToEGDlJKSog0bNlg0qW/4vvX817/%2Bpccff1wPPvig/P391aNHDyUnJ6ukpMSiSW8sBBY87sKFC8rMzNRLL73E/5wMOnbsmKKjo7Vx40YlJycrMTFROTk56tKli9WjtQsRERGaOHGicnJy1KdPHz3wwAPq1q2bJk6caPVoPi8xMVE7d%2B7UQw895La9vLxcsbGxbtuio6NVVlbmzfF8zvet54gRIzRv3ryWr%2Bvr6/W3v/1NvXr18vaINyQCCx7lcrn07LPPatKkSerZs6fV47Qrly9f1okTJ3Tq1CkVFhbq3Xff1blz5zR37lyrR2sXXC6XgoKCNH/%2BfH3yySfaunWrTp48qRUrVlg9ms/r0qWLbDZbq%2B1Op7PVD2FBQUGqra311mg%2B6fvW89tqamqUnp6uoKAgfkjwEgILHrVq1Sp16NBBEyZMsHqUdqdDhw6SpMzMTIWGhioyMlKzZs3Snj175HQ6LZ7O9%2B3cuVM7duzQE088oQ4dOigmJkbp6el6%2B%2B23rR6t3bLb7aqvr3fbVl9fr5CQEIsmah8%2B//xzPf7442psbNS6desUGhpq9Ug3hP%2BcvMB/6b333lNlZaXi4%2BMlqeXD84MPPuCG9/9SdHS0XC6XGhoa1LFjR0lq%2Be2ga/cM4cc7e/Zsq98YtNlsCgwMtGii9i82Nlb79%2B9321ZRUaGYmBiLJvJ9e/bs0ezZs/XYY4/pmWee%2BcEzXTCHM1jwqO3bt%2Bsf//iHSktLVVpaqlGjRmnUqFHElQGDBw9W165d9fzzz8vpdKqqqkrLli3TsGHD%2BAnVgMTERDkcDr3%2B%2ButqamrS6dOntXLlSqWkpFg9WruVnJys8%2BfPKz8/Xw0NDTp48KCKioo0duxYq0fzSZ988onS09M1b948zZ07l7jyMgIL8FGBgYFav369AgICNGLECI0YMUK33nqrFi9ebPVo7UJ0dLRWrVqlXbt2KSEhQb/5zW%2BUlJTU6je1YE5ERITWrl2r7du3KyEhQVlZWcrKytK9995r9Wg%2B6fXXX1djY6Oys7MVFxfX8t/kyZOtHu2G4NfMtQQAAACjOIMFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABgGIEFAABg2P8Dj0FlT3EL1NwAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-8572833814874630454">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">7</td>
        <td class="number">8974</td>
        <td class="number">41.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8</td>
        <td class="number">6065</td>
        <td class="number">28.1%</td>
        <td>
            <div class="bar" style="width:67%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9</td>
        <td class="number">2615</td>
        <td class="number">12.1%</td>
        <td>
            <div class="bar" style="width:29%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6</td>
        <td class="number">2038</td>
        <td class="number">9.4%</td>
        <td>
            <div class="bar" style="width:23%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10</td>
        <td class="number">1134</td>
        <td class="number">5.3%</td>
        <td>
            <div class="bar" style="width:13%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">11</td>
        <td class="number">399</td>
        <td class="number">1.8%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">242</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">12</td>
        <td class="number">89</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">27</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">13</td>
        <td class="number">13</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-8572833814874630454">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">3</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4</td>
        <td class="number">27</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5</td>
        <td class="number">242</td>
        <td class="number">1.1%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6</td>
        <td class="number">2038</td>
        <td class="number">9.4%</td>
        <td>
            <div class="bar" style="width:23%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7</td>
        <td class="number">8974</td>
        <td class="number">41.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">9</td>
        <td class="number">2615</td>
        <td class="number">12.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10</td>
        <td class="number">1134</td>
        <td class="number">5.3%</td>
        <td>
            <div class="bar" style="width:43%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">11</td>
        <td class="number">399</td>
        <td class="number">1.8%</td>
        <td>
            <div class="bar" style="width:16%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">12</td>
        <td class="number">89</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">13</td>
        <td class="number">13</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_id">id<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>21420</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>99.2%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>4580500000</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>1000102</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>9900000190</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-4450051481539205844">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAA3hJREFUeJzt3V8o%2B3scx/HX73CxoVzJ3LjcJdHayq3oFCVaJHPpQi5E7rhSR3Oz5MYdV0pJSi6O3M2fps6uuXAhS2orSYay%2BZy7lfzO%2B2fajv14Pu58vj58Vp59Pvb91n4455wA/NQfn70AoJJVf/YCSiEw%2B3fRc/75688yrARfzZcI5P9SbIhE%2BPvjiAUYvu0O8pFjGb4fdhDAQCCAgUAAA4EABgIBDN/2XazvjBur78cOAhgIBDAQCGAgEMBAIICBQABDxb3Ny0OEqCQVFwgq03e9d8IRCzCwg6BsvsKuww4CGAgEMBAIYCAQwMA/6RWG%2B0CVhUDKiD/23x9HLMBAIICBQAADgQAGAgEMBAIYCAQwEAhgIBDAQCCAgUAAA4EABgIBDAQCGAgEMBAIYCAQwEAggIFAAAOBAAYCAQw/nHPusxcBVCp2EMBAIICBQAADgeDDbm5u1NXVpZOTk6Lnrq2taXR09NVYPp/X4uKiOjo61NbWpvHxcaXT6VIt90MIBB%2BSTCY1NDSky8vLouY9PDwoGo0qGo2%2BubaysqKjoyNtbW3p4OBAHo9Hc3NzpVryhxAIira9va2ZmRlNTU29uXZ8fKxwOKxAIKCenh7t7Oy8ut7X16dMJqPh4eE3czc3NzU2NqampibV1dVpdnZW8XhcqVSqbK/llxxQpHQ67Z6fn51zzvn9fpdIJJxzzp2enrqWlha3t7fncrmcSyaTLhQKuXg8Xph7fX3tnHNueXnZRSKRwvjd3Z3z%2B/3u7Ozs1e8KBoNuf3%2B/3C/pP7GDoGgNDQ2qrn77wQAbGxvq7OxUd3e3qqqq1N7ersHBQa2vrxe%2Bx%2Bfz/fRnZrNZSVJNTc2rcY/HU7j2Gfj4A5TM1dWVEomEAoFAYSyfz6u5ufmXc71eryTp8fHx1fjT05Nqa2tLu9AiEAhKxufzqb%2B/X/Pz84WxdDot946HNerr69XY2Kjz83P5/X5JUiaT0e3tbeHrz8ARCyUTDoe1u7urw8NDvby86OLiQpFIRKurq%2B%2BaPzAwoJWVFaVSKd3f32thYUHBYPBdO1C5sIOgZFpbWxWLxRSLxTQ5OSmv16ve3l5NT0%2B/a/7ExIRyuZxGRkaUzWYVCoW0tLRU5lXbeFgRMHDEAgwEAhgIBDAQCGAgEMBAIICBQAADgQAGAgEMBAIYCAQwEAhg%2BBcn4cYNVZHcQQAAAABJRU5ErkJggg%3D%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-4450051481539205844,#minihistogram-4450051481539205844"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-4450051481539205844">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-4450051481539205844"
                                                  aria-controls="quantiles-4450051481539205844" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-4450051481539205844" aria-controls="histogram-4450051481539205844"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-4450051481539205844" aria-controls="common-4450051481539205844"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-4450051481539205844" aria-controls="extreme-4450051481539205844"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-4450051481539205844">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>1000102</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>512740000</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>2123000000</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>3904900000</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>7308900000</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>9297300000</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>9900000190</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>9899000088</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>5185900000</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>2876700000</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.62804</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-1.2607</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>4580500000</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>2543800000</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.24323</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>98924503192990</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>8.2756e+18</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-4450051481539205844">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3XtcVXW%2B//E3sDE2eAFT7DL20EmwY15ASVLMmUjG44UuiJGZ6ZQ6Y6Tp8dIxLR3J26TmOJ7xlGam8jiRFpWNlqeOqWOKjplZiYJl2piwFUxAkIv790c/96MdbgX77r1d%2Bno%2BHjyC7/qu5Wd9Hou93621WDvA6XQ6BQAAAGMC/V0AAADA1YaABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMs/m7gGuFw1FifJuBgQFq2jRMRUVlOnfOaXz71zr6613017vor3fRX%2B8z1ePmzRsZrKruOINlYYGBAQoICFBgYIC/S7kq0V/vor/eRX%2B9i/56n9V7TMACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMNs/i4A144%2BC7f5u4R6%2Bd8Jd/m7BACARVn2DNb27ds1cOBAde7cWQkJCcrIyFBFRYUkae/evRo4cKBiY2OVmJioNWvWuK2bnZ2tpKQkxcTEKCUlRXv27HEtq6mp0dy5c9W9e3fFxsZq1KhRKiws9Om%2BAQAAa7NkwCoqKtIf/vAHDRo0SP/85z%2BVnZ2tnTt36uWXX9YPP/ygkSNH6v7779euXbs0c%2BZMzZ49W59//rkkKScnRxkZGZozZ4527dqle%2B%2B9V6NGjVJ5ebkkacmSJdq2bZvefPNNbd26VSEhIZo6dao/dxcAAFiMJQNW06ZN9cknnyglJUUBAQE6deqUzp49q6ZNm2rjxo0KDw/X4MGDZbPZ1K1bNyUnJyszM1OStGbNGvXr109dunRRcHCwhg0bpoiICK1fv961fMSIEbrxxhvVsGFDTZkyRVu2bNHRo0f9ucsAAMBCLHsPVsOGDSVJv/nNb1RQUKC4uDilpKRo4cKFio6Odpvbpk0brV27VpKUn5%2BvAQMG1Fqem5urkpISHT9%2B3G39Zs2aqUmTJjpw4IBatmxZp9oKCwvlcDjcxmy2UEVGRtZ7Py8mKCjQ7b8wi/56F/31LvrrXfTX%2B6zeY8sGrPM2btyoH374QRMmTNCYMWPUokUL2e12tzkhISE6c%2BaMJKmsrMzj8rKyMklSaGhoreXnl9VFVlaWFi9e7DaWnp6uMWPG1Hkb9dG4sf3Sk1Bv5/tKf72L/noX/fUu%2But9Vu2x5QNWSEiIQkJCNHHiRA0cOFBDhgxRSUmJ25yKigqFhYVJkux2u%2Btm%2BJ8uj4iIcAWv8/djXWj9ukhLS1NiYqLbmM0WquLiuoe0uggKClTjxnadPl2umppzRrcN6fTpcvrrRRy/3kV/vYv%2Bep%2BpHkdE1P392yRLBqxPP/1UzzzzjN599101aNBAklRZWang4GC1adNG27a5Pw4gPz9fUVFRkqSoqCjl5eXVWt6zZ081adJELVq0UH5%2BvusyocPh0KlTp2pddryYyMjIWpcDHY4SVVd755ewpuac17Z9LTv/C22V/lrpMRgbxia4vrdKf62K/noX/fU%2Bq/bYkhc227Ztq4qKCs2fP1%2BVlZX617/%2Bpblz5yo1NVW9e/fWiRMntGLFClVVVWnHjh1at26d676r1NRUrVu3Tjt27FBVVZVWrFihkydPKikpSZKUkpKiJUuW6OjRoyotLdWsWbPUtWtX3XLLLf7cZQAAYCGWPIMVFhamZcuWadasWUpISFCjRo2UnJys9PR0NWjQQMuXL9fMmTO1aNEiNW3aVFOnTtWdd94pSerWrZumTZum6dOnq6CgQG3atNHSpUsVHh4u6cd7paqrqzV48GCVlZUpPj5eCxcu9OfuAgAAiwlwOp1OfxdxLXA4Si49qZ5stkBFRISpuLjMEqdPrXQJS/rxSe701zs2jE2w3PFrNfTXu%2Biv95nqcfPmjQxWVXeWvEQIAABwJSNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAyz5EflAL6QNG%2Brv0sAAFgUZ7AAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYddkwMrNzdXvf/97de3aVQkJCZo0aZKKiookSdOmTVP79u0VGxvr%2BsrKynKtu3TpUvXs2VMxMTEaMmSIvv76a9eyM2fOaPLkyYqPj1eXLl00adIklZWV%2BXz/AACAf11zAauiokLDhw9XbGys/vGPf%2Bi9997TqVOn9Mwzz0iS9u3bp4yMDO3Zs8f1lZaWJknKzs7WqlWr9MorrygnJ0e33367xowZI6fTKUnKyMjQ999/rw8%2B%2BEAbN27U999/r3nz5vltXwEAgH9ccwHr2LFjuu2225Senq4GDRooIiJCaWlp2rVrlyorK3Xw4EG1b9/%2Bguu%2B8cYbevjhhxUVFaXrrrtO48eP17Fjx5STk6Py8nKtW7dOY8aMUXh4uK6//npNmDBBb731lsrLy328lwAAwJ9s/i7A1379619r2bJlbmMffPCBbr/9duXm5qq6ulqLFi3S7t271ahRIw0YMEDDhw9XYGCg8vPzNWLECNd6wcHBatWqlXJzcxUeHq6qqipFR0ersLBQDodDVVVVqqio0EcffaSoqNsVGRlpdF%2BCggLd/gtYhc0WyPHrZfTXu%2Biv91m9x9dcwPopp9OphQsXatOmTVq9erVOnDihrl27asiQIVqwYIH279%2Bv9PR0BQYGavjw4SorK5PdbnfbRkhIiM6cOaPS0lJJUmhoqF599VUtXrzYNWf8%2BPFKT0/XmDFjvLIfjRvbLz0JuIJERIS5vuf49S7661301/us2uNrNmCVlpZq8uTJ%2BvLLL7V69Wq1bdtWbdu2VUJCgmtOx44dNXToUK1fv17Dhw%2BX3W5XRUWF23YqKioUFham0NBQSVJ5ebnS0tKUmJio8vJyDR48WPPnz1dU1O0qLjZ7w3tQUKAaN7br9Oly1dScM7ptwJuKi8s4fr3MKv1NmrfV3yXUy/9OuEuSdfprZaZ6/NP/ofOlazJgHTlyRCNGjNBNN92ktWvXqmnTppKkDz/8UCdOnNBDDz3kmltZWamQkBBJUlRUlPLy8nT33XdLkqqqqnT48GFFR0erdevWCg4OVn5%2Bvjp16qTIyEjt3btXwcHB6tWrl0pKqlRd7Z1fwpqac17bNuANPz1eOX69i/6a9fNe0l/vs2qPrXlh8xf44YcfNHToUHXu3FmvvPKKK1xJP14ynD17trZv3y6n06k9e/Zo5cqVrr8iHDBggFavXq3c3FydPXtW8%2BfPV7NmzRQXFye73a4%2Bffpo3rx5KioqUlFRkebNm6f%2B/fu7AhoAALg2XHNnsN566y0dO3ZMGzZs0Pvvv%2B%2B2bM%2BePZo8ebKmT5%2BugoICNWvWTKNHj9Z9990nSUpNTVVJSYnS09NVVFSkDh066KWXXlJwcLCkH5%2BhNXfuXCUnJ6uqqkr33HOPnn32WZ/vIwAA8K8A5/mHOMGrHI4S49u02QIVERGm4uIyS5w%2B7bNwm79LwBViw9gEyx2/VmOV/lrtdWHD2B/v07VKf63MVI%2BbN29ksKq6u%2BYuEQIAAHgbAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMMyyASs3N1e///3v1bVrVyUkJGjSpEkqKiqSJO3du1cDBw5UbGysEhMTtWbNGrd1s7OzlZSUpJiYGKWkpGjPnj2uZTU1NZo7d666d%2B%2Bu2NhYjRo1SoWFhT7dNwAAYG02fxdwOSoqKjR8%2BHA9%2BOCDeumll1RWVqann35azzzzjObOnauRI0dqzJgxSktL065du5Senq62bduqY8eOysnJUUZGhpYuXaqOHTsqMzNTo0aN0qZNm2S327VkyRJt27ZNb775pho1aqRnn31WU6dO1csvv%2Bzv3b6guCnv%2B7sEAADwM5Y8g3Xs2DHddtttSk9PV4MGDRQREeEKUxs3blR4eLgGDx4sm82mbt26KTk5WZmZmZKkNWvWqF%2B/furSpYuCg4M1bNgwRUREaP369a7lI0aM0I033qiGDRtqypQp2rJli44ePerPXQYAABZiyTNYv/71r7Vs2TK3sQ8%2B%2BEC333678vLyFB0d7basTZs2Wrt2rSQpPz9fAwYMqLU8NzdXJSUlOn78uNv6zZo1U5MmTXTgwAG1bNmyTvUVFhbK4XC4jdlsoYqMjKzzPtZFUJAl8zEgmy3QdfxyHHsH/fUOm829r/TXe6zeY0sGrJ9yOp1auHChNm3apNWrV2vlypWy2%2B1uc0JCQnTmzBlJUllZmcflZWVlkqTQ0NBay88vq4usrCwtXrzYbSw9PV1jxoyp8zaAq1lERJjr%2B8aN7ReZiV%2BK/pr102NXor%2B%2BYNUeWzpglZaWavLkyfryyy%2B1evVqtW3bVna7XSUlJW7zKioqFBb24y%2BF3W5XRUVFreURERGu4FVeXu5x/bpIS0tTYmKi25jNFqri4rqHtLqwaqoHiovLFBQUqMaN7Tp9ulw1Nef8XdJVh/56x/nXcfrrfaZ6/PNQ7CuWDVhHjhzRiBEjdNNNN2nt2rVq2rSpJCk6Olrbtm1zm5ufn6%2BoqChJUlRUlPLy8mot79mzp5o0aaIWLVooPz/fdZnQ4XDo1KlTtS47XkxkZGSty4EOR4mqq/klBCS5/S7U1Jzjd8OL6K9ZP%2B8l/fU%2Bq/bYkqdAfvjhBw0dOlSdO3fWK6%2B84gpXkpSUlKQTJ05oxYoVqqqq0o4dO7Ru3TrXfVepqalat26dduzYoaqqKq1YsUInT55UUlKSJCklJUVLlizR0aNHVVpaqlmzZqlr16665ZZb/LKvAADAeix5Buutt97SsWPHtGHDBr3/vvtjCvbs2aPly5dr5syZWrRokZo2baqpU6fqzjvvlCR169ZN06ZN0/Tp01VQUKA2bdpo6dKlCg8Pl/TjvVLV1dUaPHiwysrKFB8fr4ULF/p8HwEAgHUFOJ1Op7%2BLuBY4HCWXnlRPNlugkuZtNb5dwNs2jE2QzRaoiIgwFReXWfL0/5XOKv3ts3DbpSddQTaMTZBknf5amakeN2/eyGBVdefxDFZNTY2CgoJ8WQsAAFc0qwZC%2BJ7He7B69uypP//5z8rPz/dlPQAAAJbnMWA9%2BeST%2BvTTT9W/f38NHDhQr7/%2Beq3HHwAAAKA2jwFr0KBBev311/X%2B%2B%2B%2Bre/fuWrp0qXr06KHx48frk08%2B8WWNAAAAlnLJxzS0atVK48aN0/vvv6/09HR99NFHevzxx5WYmKhXX31VNTU1vqgTAADAMi75mIa9e/fq7bff1vr161VZWamkpCSlpKSooKBAf/nLX7Rv3z4tWLDAF7UCAABYgseA9be//U3vvPOOvv32W3Xo0EHjxo1T//791bBhQ9ecoKAgPffccz4pFAAAwCo8BqzVq1fr3nvvVWpqqtq0aXPBObfeeqsmTJjgteIAAACsyGPA2rJli0pLS3Xq1CnX2Pr169WtWzdFRERIktq1a6d27dp5v0oAAAAL8XiT%2B1dffaXevXsrKyvLNfbCCy8oOTlZBw8e9ElxAAAAVuQxYP35z3/W7373O40bN8419uGHH6pnz56aM2eOT4oDAACwIo8B68svv9TIkSPVoEED11hQUJBGjhypzz77zCfFAQAAWJHHe7AaNmyoI0eOqGXLlm7jx48fV0hIiNcLAwAAv4yVPjvxavvcRI9nsHr37q3p06frk08%2BUWlpqcrKyrRjxw7NmDFDSUlJvqwRAADAUjyewRo/fryOHj2qxx57TAEBAa7xpKQkTZo0ySfFAQAAWJHHgGW32/XSSy/pm2%2B%2B0YEDBxQcHKxbb71VrVq18mF5AAAA1nPJj8pp3bq1Wrdu7YtaAAAArgoeA9Y333yjGTNmaPfu3aqqqqq1fP/%2B/V4tDAAAwKo8Bqzp06fr2LFjmjBhgho1auTLmgAAACzNY8Das2ePXnvtNcXGxvqyHgAAAMvz%2BJiGiIgIhYWF%2BbIWAACAq4LHgDVkyBAtWLBAJSUlvqwHAADA8jxeIty8ebM%2B%2B%2BwzxcfH6/rrr3f7yBxJ%2Buijj7xeHAAAgBV5DFjx8fGKj4/3ZS0AAABXBY8B68knn/RlHQAAAFcNj/dgSVJubq4mT56shx56SAUFBcrMzFROTo6vagMAALAkjwHriy%2B%2B0MCBA/Xdd9/piy%2B%2BUGVlpfbv36/HHntMmzZt8mWNAAAAluIxYM2bN0%2BPPfaYVq1apeDgYEnS888/r0cffVSLFy/2WYEAAABWc9EzWPfff3%2Bt8UGDBunrr7/2alEAAABW5jFgBQcHq7S0tNb4sWPHZLfbvVoUAACAlXkMWL169dL8%2BfNVXFzsGjt06JBmzpyp3/72t76oDQAAwJI8Bqynn35aFRUV6t69u8rLy5WSkqL%2B/fvLZrNp0qRJvqwRAADAUjw%2BB6thw4Z6/fXXtX37dn311Vc6d%2B6coqOjdddddykw8KJPdwAAALimeQxY53Xr1k3dunXzRS0AAABXBY8BKzExUQEBAR5X5LMIAQAALsxjwHrggQfcAlZVVZW%2B/fZbbdmyRWPHjvVJcQAAAFbkMWCNHj36guOrV6/W7t279eijj3qtKAAAACur993qd999tzZv3uyNWgAAAK4K9Q5YO3fu1HXXXeeNWgAAAK4KHi8R/vwSoNPpVGlpqQ4cOMDlQQAAgIvweAbrpptu0s033%2Bz6%2BtWvfqUOHTpo1qxZmjhxoi9rvKiioiIlJSUpJyfHNTZt2jS1b99esbGxrq%2BsrCzX8qVLl6pnz56KiYnRkCFD3D5b8cyZM5o8ebLi4%2BPVpUsXTZo0SWVlZT7dJwAAYG0ez2DNmTPHl3Vclt27d%2Bs///M/deTIEbfxffv2KSMjQw888ECtdbKzs7Vq1Sq98soruuWWW/Tiiy9qzJgxWrdunQICApSRkaHvv/9eH3zwgWpqajR27FjNmzdP06ZN89VuAQAAi/MYsHbt2lXnjdxxxx1GiqmP7OxsLVq0SBMnTtS4ceNc45WVlTp48KDat29/wfXeeOMNPfzww4qKipIkjR8/Xm%2B88YZycnLUqVMnrVu3TitXrlR4eLgkacKECXr00Uc1adIkPuQaAADUiceANWzYMDmdTtfXeeefjXV%2BLCAgQPv37/dymbX16NFDycnJstlsbgErNzdX1dXVWrRokXbv3q1GjRppwIABGj58uAIDA5Wfn68RI0a45gcHB6tVq1bKzc1VeHi4qqqqFB0d7Vp%2B6623qqKiQocPH9a//du/1am2wsJCORwOtzGbLVSRkZG/cK/dBQXxkUWwJpst0HX8chyblzRvq79LAOrNZnN/LbD6a4THgPXXv/5Vs2fP1tNPP60777xTwcHB2rt3r6ZPn66HH35Yd999ty/rrKV58%2BYXHC8pKVHXrl01ZMgQLViwQPv371d6eroCAwM1fPhwlZWV1ToTFRISojNnzqi0tFSSFBoa6lp2fm597sPKysrS4sWL3cbS09M1ZsyYOm8DuJpFRIS5vm/cmDPDANxfF37Kqq8RHgPW3LlzNW3aNPXo0cM11rVrV82YMUOTJk3SI4884pMC6yshIUEJCQmunzt27KihQ4dq/fr1Gj58uOx2uyoqKtzWqaioUFhYmCtYlZeXKywszPW99OOHX9dVWlqaEhMT3cZstlAVF5u9Wd6qqR4oLi5TUFCgGje26/TpctXUnPN3SQD87OfvkaZeIzwFN2/zGLAKCwt144031hpv2LChiouLvVrUL/Hhhx/qxIkTeuihh1xjlZWVCgkJkSRFRUUpLy/PdQauqqpKhw8fVnR0tFq3bq3g4GDl5%2BerU6dOkqRDhw65LiPWVWRkZK3LgQ5HiaqreRMBJLn9LtTUnLvifzf6LNzm7xKAq56n1wErvEZciMdTIDExMVqwYIHrspkknTp1Si%2B88IK6devmk%2BIuh9Pp1OzZs7V9%2B3Y5nU7t2bNHK1euVFpamiRpwIABWr16tXJzc3X27FnNnz9fzZo1U1xcnOx2u/r06aN58%2BapqKhIRUVFmjdvnvr37%2B8KaAAAAJfi8QzW1KlTNXToUPXs2dN19uabb75R8%2BbNtXLlSl/VV29JSUmaPHmypk%2BfroKCAjVr1kyjR4/WfffdJ0lKTU1VSUmJ0tPTVVRUpA4dOuill15ScHCwpB%2BfoTV37lwlJyerqqpK99xzj5599ll/7hIAALCYAOdP/0TwZ06fPq1169bp0KFDkqR27dqpX79%2BPK7gMjgcJca3abMF8tdCsKQNYxNkswUqIiJMxcVlV/zpfy4RAt63YWyC28%2BmXiOaN2/0S0u7LB7PYElS48aNNXDgQH333Xdq2bKlJLnO9AAAAODCPN6D5XQ6NW/ePN1xxx3q37%2B/jh8/rqefflqTJ09WVVWVL2sEAACwFI8Ba9WqVXrnnXc0bdo0NWjQQJLUq1cv/d///Z/%2B8pe/%2BKxAAAAAq/F4iTArK0vPPfeckpKSlJGRIUnq27evGjRooJkzZ2rChAk%2BKxLA1YV7mgBc7Tyewfruu%2B8u%2BNEwbdu21YkTJ7xaFAAAgJV5DFg333yzPv/881rjmzdvdt3wDgAAgNo8XiJ8/PHH9ac//UkFBQVyOp3avn27Xn/9da1atUqTJ0/2ZY0AAACW4jFgDRgwQNXV1VqyZIkqKir03HPP6frrr9e4ceM0aNAgX9YIAABgKR4D1rvvvqt///d/V1pamoqKiuR0OnX99df7sjYAAABL8ngP1vPPP%2B%2B6mb1p06aEKwAAgDryGLBatWqlAwcO%2BLIWAACAq4LHS4RRUVGaMGGCli1bplatWum6665zWz579myvFwcAAGBFHgPWkSNH1KVLF0mSw%2BHwWUEAAABW5xawZs%2BeraeeekqhoaFatWqVv2oCAACwNLd7sFauXKny8nK3CY8//rgKCwt9WhQAAICVuQUsp9NZa8Knn36qs2fP%2BqwgAAAAq/P4V4QAAAC4PAQsAAAAw2oFrICAAH/UAQAAcNWo9ZiG559/3u2ZV1VVVXrhhRcUFhbmNo/nYAEAAFyYW8C64447aj3zKjY2VsXFxSouLvZpYQAAAFblFrB49hUAAMAvx03uAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAM81nAKioqUlJSknJyclxje/fu1cCBAxUbG6vExEStWbPGbZ3s7GwlJSUpJiZGKSkp2rNnj2tZTU2N5s6dq%2B7duys2NlajRo1SYWGhr3YHAADAI58ErN27dystLU1Hjhxxjf3www8aOXKk7r//fu3atUszZ87U7Nmz9fnnn0uScnJylJGRoTlz5mjXrl269957NWrUKJWXl0uSlixZom3btunNN9/U1q1bFRISoqlTp/pidwAAAC7K6wErOztbEyZM0Lhx49zGN27cqPDwcA0ePFg2m03dunVTcnKyMjMzJUlr1qxRv3791KVLFwUHB2vYsGGKiIjQ%2BvXrXctHjBihG2%2B8UQ0bNtSUKVO0ZcsWHT161Nu7BAAAcFE2b/8DPXr0UHJysmw2m1vIysvLU3R0tNvcNm3aaO3atZKk/Px8DRgwoNby3NxclZSU6Pjx427rN2vWTE2aNNGBAwfUsmVLL%2B7RpRUWFsrhcLiN2WyhioyMNPrvBAVxCx0A4Opgs7m/p51/j7Pqe53XA1bz5s0vOF5WVia73e42FhISojNnzlxyeVlZmSQpNDS01vLzy/wpKytLixcvdhtLT0/XmDFj/FQRAABXtoiIsAuON25sv%2BD4lc7rAcsTu92ukpISt7GKigqFhYW5lldUVNRaHhER4Qpe5%2B/HutD6/pSWlqbExES3MZstVMXFZsOfVVM9AAA/9/P3yKCgQDVubNfp0%2BWqqTl32dv1FNy8zW8BKzo6Wtu2bXMby8/PV1RUlCQpKipKeXl5tZb37NlTTZo0UYsWLZSfn%2B%2B6TOhwOHTq1Klalx39ITIystblQIejRNXVl3%2BAAABwNfP0HllTc86S759%2BOwWSlJSkEydOaMWKFaqqqtKOHTu0bt06131XqampWrdunXbs2KGqqiqtWLFCJ0%2BeVFJSkiQpJSVFS5Ys0dGjR1VaWqpZs2apa9euuuWWW/y1SwAAAJL8eAYrIiJCy5cv18yZM7Vo0SI1bdpUU6dO1Z133ilJ6tatm6ZNm6bp06fmLGCkAAAQRklEQVSroKBAbdq00dKlSxUeHi7px3uaqqurNXjwYJWVlSk%2BPl4LFy701%2B4AAAC4BDidTqe/i7gWOBwll55UTzZboJLmbTW%2BXQAAfG3D2AS3n222QEVEhKm4uOwXXSJs3rzRLy3tsnCXNAAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAsKs2YK1fv17t2rVTbGys62vixImSpM2bNys5OVkxMTHq06ePNm3aVGv97OxsJSUlKSYmRikpKdqzZ4%2BvdwEAAFiUzd8FeMu%2Bfft03333afbs2W7jhw8f1ujRo7VgwQL99re/1caNGzV27Fht3LhRLVq0kCTl5OQoIyNDS5cuVceOHZWZmalRo0Zp06ZNstvt/tgdAABgIVftGax9%2B/apffv2tcazs7MVFxenXr16yWazqW/fvrrjjjuUlZXlmrNmzRr169dPXbp0UXBwsIYNG6aIiAitX7/el7sAAAAs6qo8g3Xu3Dl9%2BeWXstvtWrZsmWpqavSb3/xGEyZMUH5%2BvqKjo93mt2nTRrm5ua6f8/PzNWDAgIvOuZjCwkI5HA63MZstVJGRkZe5RxcWFHTV5mMAwDXGZnN/Tzv/HmfV97qrMmAVFRWpXbt26t27txYtWqTi4mI9/fTTmjhxoiorK2td5gsJCdGZM2dcP5eVlV1yzsVkZWVp8eLFbmPp6ekaM2bMZe4RAABXt4iIsAuON25szVtzrsqA1axZM2VmZrp%2Bttvtmjhxoh588EHFx8eroqLCbX5FRYXCwsLc5l9oTkRERJ3%2B/bS0NCUmJrqN2WyhKi4uq%2B%2BuXJRVUz0AAD/38/fIoKBANW5s1%2BnT5aqpOXfZ2/UU3LztqgxYubm5eu%2B99zR%2B/HgFBARIkiorKxUYGKiOHTtq//79bvPz8/Pd7teKiopSXl5erTk9e/as078fGRlZ63Kgw1Gi6urLP0AAALiaeXqPrKk5Z8n3z6vyFEh4eLgyMzO1bNkyVVdX69ixY3rhhRf0wAMP6P7779fOnTu1fv16VVdXa/369dq5c6fuu%2B8%2B1/qpqalat26dduzYoaqqKq1YsUInT55UUlKSH/cKAABYRYDT6XT6uwhv2LlzpxYsWKCDBw/quuuuU79%2B/TRx4kRdd9112rp1q%2BbNm6cjR47o5ptv1sSJE/Wb3/zGbf133nlHS5YsUUFBgdq0aaOpU6eqU6dOl12Pw1HyS3epFpstUEnzthrfLgAAvrZhbILbzzZboCIiwlRcXPaLzmA1b97ol5Z2Wa7agHWlIWABAODZ1RawrspLhAAAAP5EwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwLuDkyZN64oknFBcXp/j4eM2cOVPV1dX%2BLgsAAFgEAesCxo4dq9DQUG3dulVr167V9u3btWLFCn%2BXBQAALIKA9TPffvutdu7cqYkTJ8put6tly5Z64oknlJmZ6e/SAACARRCwfiYvL0/h4eFq0aKFa%2BzWW2/VsWPHdPr0aT9WBgAArMLm7wKuNGVlZbLb7W5j538%2Bc%2BaMGjdufMltFBYWyuFwuI3ZbKGKjIw0V6ikoCDyMQDg6mCzub%2BnnX%2BPs%2Bp7HQHrZ0JDQ1VeXu42dv7nsLCwOm0jKytLixcvdht78sknNXr0aDNF/n%2BFhYUaekOe0tLSjIc3/NjfrKws%2Busl9Ne76K930V/vKyws1GuvLbNsj60ZC70oKipKp06d0okTJ1xjhw4d0g033KBGjRrVaRtpaWl666233L7S0tKM1%2BpwOLR48eJaZ8tgBv31LvrrXfTXu%2Biv91m9x5zB%2BplWrVqpS5cumjVrlmbMmKHi4mL97W9/U2pqap23ERkZacm0DQAAzOAM1gUsWrRI1dXVuueee/Tggw/qrrvu0hNPPOHvsgAAgEVwBusCmjVrpkWLFvm7DAAAYFFB06dPn%2B7vInD5wsLC1LVr1zrfgI/6ob/eRX%2B9i/56F/31Piv3OMDpdDr9XQQAAMDVhHuwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAA4HfffPONhg4dqtjYWPXo0UP//d//7e%2BSfhECFgAA8Kuqqir98Y9/VIcOHZSTk6OXX35ZmZmZ2rBhg79Lu2wErCvcyZMn9cQTTyguLk7x8fGaOXOmqqurLzh38%2BbNSk5OVkxMjPr06aNNmzb5uFrrqU9//%2Bd//ke9e/dWbGysevfurczMTB9Xaz316e95Bw8eVKdOnZSTk%2BOjKq3rYv3Nzc11nQ3o3r27Zs%2Befcnew119jt/XXntNiYmJ6ty5s5KTk/XBBx/4uFpr27VrlwoLCzVmzBg1aNBA7dq105AhQyz9OkvAusKNHTtWoaGh2rp1q9auXavt27drxYoVteYdPnxYo0eP1lNPPaV//vOfGj16tMaOHauCggLfF20hde3vhx9%2BqAULFmju3Ln69NNPNWfOHC1cuJAX0Uuoa3/PKy8v1/jx41VRUeG7Ii3MU3%2BLioo0bNgwde/eXTt37tQbb7yhjz/%2BWK%2B99pq/S7aUuh6/mzdv1ksvvaRly5bp008/1ZNPPqmxY8fqu%2B%2B%2B833RFpWXl6fWrVurQYMGrrE2bdooNzfXj1X9MgSsK9i3336rnTt3auLEibLb7WrZsqWeeOKJCyb67OxsxcXFqVevXrLZbOrbt6/uuOMOZWVl%2BaFya6hPfwsKCjRixAjFxMQoICBAsbGxio%2BP165du/xQuTXUp7/n/elPf1KvXr18WKV1Xay/b7/9tlq1aqU//OEPCg4O1q9%2B9SstX75cffr08XfZllGf4/frr7%2BW0%2Bl0fQUFBSk4OFg2m80PlVtTWVmZ7Ha725jdbteZM2f8VNEvR8C6guXl5Sk8PFwtWrRwjd166606duyYTp8%2B7TY3Pz9f0dHRbmNWT//eVp/%2BDh48WCNHjnT9fPLkSe3atUvt27f3Wb1WU5/%2BStLbb7%2Btb7/9Vk8%2B%2BaQvy7Ssi/V327Ztio6O1nPPPaeEhAT16tVL7777rm644QY/Vmwt9Tl%2B%2B/Xrp2bNmqlv3766/fbb9dRTT2nOnDn0ux5CQ0NVXl7uNlZeXm7JD3k%2Bj4B1BfOU6CXVSvUXmhsSEmLp9O9t9envTzkcDo0YMULt27dX//79vVqjldWnv4cOHdKLL76o%2BfPnKygoyGc1WtnF%2BltZWam33npLHTt21Mcff6zFixcrKytLr776qj9KtaT6HL9VVVW67bbbtGbNGn322WeaMWOGpkyZogMHDvisXquLiorS4cOH3e5xy8/PV1RUlB%2Br%2BmUIWFcwT4leUq1Ub7fba923UlFRYen072316e95n332mVJTU9W6dWstWbKESwAXUdf%2Bnj17VuPGjdMzzzyjm266yac1WtnF%2BitJHTp0UGpqqoKDg3XbbbfpkUcesfRfZPlafV4fMjIyFBUVpY4dO6pBgwYaMGCAYmJilJ2d7bN6rS4%2BPl4RERGaP3%2B%2Bzp49q9zcXK1atUqpqan%2BLu2yEbCuYFFRUTp16pROnDjhGjt06JBuuOEGNWrUyG1udHS08vLy3Masnv69rT79laS1a9dq2LBhGjp0qObPn%2B92MyZqq2t/9%2B3bp8OHD2vKlCmKi4tTXFycJOmPf/yjpk%2Bf7uuyLeNi/e3QoYMqKyvd5p87d05Op9PXZVpWfV4fjh07VqvfNptNwcHBPqnVqoYPH67nnntO0o/9Wr58uQ4ePKiEhASNHDlSQ4YMUUpKip%2Br/AWcuKINGjTIOW7cOGdJSYnzyJEjzn79%2BjkXLVpUa15%2Bfr6zQ4cOzr///e/Oqqoq59///ndnhw4dnF9//bUfqraOuvb3/fffd95%2B%2B%2B3OLVu2%2BKFK66prf38uOjrauWPHDh9UaG2e%2Bpufn%2B9s37698%2BWXX3ZWV1c7c3NznXfddZfztdde83fJllLX4/fFF190xsfHO7/44gtnTU2Nc8OGDc4OHTo4v/rqKz9UjSsFAesK53A4nKNHj3Z27drVeeeddzrnzJnjrK6udjqdTmdMTIzznXfecc3dsmWL895773XGxMQ4%2B/Xr5/z444/9VbZl1LW//fv3d952223OmJgYt69nn33Wn%2BVf8epz/P4UAatuPPW3b9%2B%2Bzr/%2B9a/Ohx9%2B2BkXF%2Bfs0aOH87/%2B67%2Bc586d83fJllLX47eqqsq5aNEi59133%2B3s3Lmz84EHHuB/xuAMcDo5ZwwAAGAS92ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAADAp4qKipSUlKScnJx6r/vqq69qyJAhbmM1NTWaO3euunfvrtjYWI0aNUqFhYWmyr0sBCwAAOAzu3fvVlpamo4cOVKv9c6cOaM5c%2BZozpw5tZYtWbJE27Zt05tvvqmtW7cqJCREU6dONVXyZSFgAQAAn8jOztaECRM0bty4Wss%2B%2BeQTpaamKi4uTv369dO7777rtvy%2B%2B%2B6Tw%2BHQoEGDaq27Zs0ajRgxQjfeeKMaNmyoKVOmaMuWLTp69KjX9uVSCFgAAMAnevToof/93/9V37593cZzc3M1atQojRw5Ujk5OcrIyNCsWbO0detW15xVq1Zp/vz5uv76693WLSkp0fHjxxUdHe0aa9asmZo0aaIDBw54d4cugoAFAAB8onnz5rLZbLXGX3/9dd1zzz363e9%2Bp6CgIHXu3FkPPvigMjMzXXNuuOGGC26zrKxMkhQaGuo2HhIS4lrmD7X3EgAAwIf%2B9a9/aceOHYqLi3ON1dTU6JZbbrnkuna7XZJUXl7uNl5RUaGwsDCzhdYDAQsAAPjVDTfcoAceeEAzZsxwjRUWFsrpdF5y3SZNmqhFixbKz893XSZ0OBw6deqU22VDX%2BMSIQAA8KvU1FS99957%2Bsc//qFz587p8OHDeuSRR7R8%2BfI6rZ%2BSkqIlS5bo6NGjKi0t1axZs9S1a9c6nQHzFs5gAQAAv%2BrUqZMWLFigBQsW6KmnnpLdblf//v31H//xH3VaPz09XdXV1Ro8eLDKysoUHx%2BvhQsXernqi/t/Tg5Yb6LL3VwAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-4450051481539205844">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">795000620</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1825069031</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2019200220</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7129304540</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1781500435</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3969300030</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2560801222</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3883800011</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2228900270</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">251300110</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (21410)</td>
        <td class="number">21576</td>
        <td class="number">99.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-4450051481539205844">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1000102</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1200019</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1200021</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2800031</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3600057</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">9842300095</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9842300485</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9842300540</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9895000040</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9900000190</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_lat">lat<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>5033</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>23.3%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>47.56</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>47.156</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>47.778</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-7618312233477638842">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAATNJREFUeJzt3MFtwkAARcE4SkkUkZ5yTk8pgp6WBtCTHMn22p65I%2B3l8cGsWMYY4wN46/PoA8DMvo4%2BAOfw%2BPlb/Zrn7/cGJ9mXBYFgQS5g7bv7Fd7Z92JBIAgEgkAgCASCQCAIBIJAIAgEgkAg%2BCX9hv5zr%2BquLAgEC8JmrnAD2IJAEAgEH7Em4wv0XCwIBIFAEAgEgUAQCASBQPCYdwX/HnI/FgSCQCAIBIJAIAgEwm2fYu1xKdDFw/OzIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBALhEpcVXQpkKxYEwnQLYg2YiQWBsIwxxtGHgFlZEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgv1hAdCUZ87zQAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-7618312233477638842,#minihistogram-7618312233477638842"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-7618312233477638842">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-7618312233477638842"
                                                  aria-controls="quantiles-7618312233477638842" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-7618312233477638842" aria-controls="histogram-7618312233477638842"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-7618312233477638842" aria-controls="common-7618312233477638842"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-7618312233477638842" aria-controls="extreme-7618312233477638842"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-7618312233477638842">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>47.156</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>47.31</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>47.471</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>47.572</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>47.678</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>47.75</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>47.778</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>0.6217</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>0.2069</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>0.13855</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.0029132</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-0.67579</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>47.56</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>0.11482</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>-0.48552</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>1027200</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>0.019197</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-7618312233477638842">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3X9UVXW%2B//EXcFAOoAIiTjnN1VGwUkvSNKXsalLXlNFQB39cMhu1MZJ0RB3S0iR/TeiYmV5TG68/1o0suWWXRlYzrprrGP7IzPyKiWbaQgUFlV9Hfni%2Bf7Q8d86ASvLh/MDnYy0W4%2Bezzz7v/Z7DPq/23mcfH7vdbhcAAACM8XV3AQAAAE0NAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGGZxdwG3i8LCEneXYJSvr4/CwoJUVFSmq1ft7i7HY9CXutGXutGX66M3daMvdbtRX9q0aeGemtzyrPB6vr4%2B8vHxka%2Bvj7tL8Sj0pW70pW705froTd3oS908sS8ELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGeX3AqqmpUWJion7/%2B987xj777DPFxcWpe/fuGjRokHbu3On0mLVr16pfv37q3r27EhMTdeLECcdceXm5UlNT1bt3b/Xo0UMzZ85UWVmZy7YHAAB4P68PWCtXrtS%2Bffsc/z558qSmTJmiF198Ufv27dOUKVM0depUnTt3TpKUmZmpTZs2af369crJyVGXLl2UnJwsu/3HL4dMS0vTmTNntGPHDmVnZ%2BvMmTNKT093y7YBAADv5NUBa/fu3crOztbjjz/uGMvMzFTPnj01cOBAWSwWPfnkk3rwwQeVkZEhSXrvvfc0ZswYRUZGqnnz5po%2Bfbry8/OVk5OjiooKbd%2B%2BXcnJyQoJCVHr1q2VkpKibdu2qaKiwl2bCQAAvIzF3QXcqgsXLmj27NlatWqVNmzY4BjPy8tTVFSU07KdOnVSbm6uY37ixImOOX9/f7Vv3165ubkKCQlRVVWV0%2BM7duwom82mkydP6p577mncjQIAeLRBy3e5u4Sf5JOpMe4u4bbllQHr6tWrmjFjhsaPH6%2B7777baa6srExWq9VpLCAgQOXl5TedLy0tlSQFBgY65q4t%2B1OuwyooKFBhYaHTmMUSqIiIiHqvw9P5%2Bfk6/caP6Evd6Evd6Mv10RszLJbbo3%2Be%2BHrxyoC1Zs0aNWvWTImJibXmrFarbDab05jNZlNQUNBN568Fq4qKCsfy104NBgcH17u%2BjIwMrVy50mksKSlJycnJ9V6Ht2jZ0nrzhW5D9KVu9KVu9OX66E3DhIYGubsEl/Kk14tXBqwPP/xQBQUF6tmzpyQ5AtOnn36qsWPH6vDhw07L5%2BXlqWvXrpKkyMhIHTt2TP3795ckVVVV6eTJk4qKilKHDh3k7%2B%2BvvLw83X///ZKk48ePO04j1ldCQoIGDBjgNGaxBKq4uOl8GtHPz1ctW1p1%2BXKFamquurscj0Ff6kZf6kZfro/emNGU3ndu5EavF3eFTK8MWH/%2B85%2Bd/n3tFg2LFy/W8ePH9ac//UlZWVl6/PHHlZ2drT179mj27NmSpOHDh%2BvNN99Uv3791KFDB/3xj39UeHi4evbsKX9/fw0aNEjp6el64403JEnp6ekaMmSIAgIC6l1fRERErdOBhYUlqq5uejuJmpqrTXK7Goq%2B1I2%2B1I2%2BXB%2B9aZjbrXee9HrxyoB1Ix07dtRbb72l9PR0zZ49W%2B3atdObb76pDh06SJJGjBihkpISJSUlqaioSN26ddOaNWvk7%2B8vSZo7d66WLFmiuLg4VVVV6bHHHtPLL7/szk0CAABexsd%2B7QZQaFSFhSXuLsEoi8VXoaFBKi4u85j/WvAE9KVu9KVu9OX6PLU3fIrQM93o9dKmTQu31OQ5l9sDAAA0EQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGNbkbtMAALc7b/qk2%2B3yKTfcfjiCBQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADPPagLV7926NHDlSDzzwgGJiYpSWliabzSZJmjt3rrp27aro6GjHT0ZGhuOxa9euVb9%2B/dS9e3clJibqxIkTjrny8nKlpqaqd%2B/e6tGjh2bOnKmysjKXbx8AAPBeXhmwioqK9Nxzz2n06NHat2%2BfMjMztWfPHr399tuSpEOHDiktLU0HDhxw/CQkJEiSMjMztWnTJq1fv145OTnq0qWLkpOTZbfbJUlpaWk6c%2BaMduzYoezsbJ05c0bp6elu21YAAOB9vDJghYWF6e9//7vi4%2BPl4%2BOjixcv6sqVKwoLC1NlZaW%2B/fZbde3atc7HvvfeexozZowiIyPVvHlzTZ8%2BXfn5%2BcrJyVFFRYW2b9%2Bu5ORkhYSEqHXr1kpJSdG2bdtUUVHh4q0EAADeyisDliQFBwdLkh599FHFxcWpTZs2io%2BPV25urqqrq7VixQr17dtXTzzxhN5%2B%2B21dvXpVkpSXl6eoqCjHevz9/dW%2BfXvl5ubq%2B%2B%2B/V1VVldN8x44dZbPZdPLkSZduHwAA8F4WdxfQUNnZ2bp06ZJSUlKUnJys8ePHq1evXkpMTNSyZct05MgRJSUlydfXVxMmTFBZWZmsVqvTOgICAlReXq7S0lJJUmBgoGPu2rI/5TqsgoICFRYWOo1ZLIGKiIi41c30OH5%2Bvk6/8SP6Ujf6Ujf6IlksdW87vTHjev1tajzx9eL1ASsgIEABAQGaMWOGRo4cqaVLl2rjxo2O%2Bfvuu0/jxo1TVlaWJkyYIKvV6rgY/hqbzaagoCBHsKqoqFBQUJDjf0v/d8SsPjIyMrRy5UqnsaSkJCUnJ9/SNnqyli2tN1/oNkRf6kZf6nY79yU0NOiG87dzb0y4WX%2BbGk96vXhlwPryyy/10ksv6aOPPlKzZs0kSZWVlfL399euXbt0%2BfJljRo1yrF8ZWWlAgICJEmRkZE6duyY%2BvfvL0mqqqrSyZMnFRUVpQ4dOsjf3195eXm6//77JUnHjx93nEasr4SEBA0YMMBpzGIJVHFx0/k0op%2Bfr1q2tOry5QrV1Fx1dzkeg77Ujb7Ujb7ouvtFemNGU3rfuZEbvV7cFTK9MmB17txZNptNS5cu1fTp01VYWKglS5ZoxIgR8vf316JFi/Qv//Iveuihh/TVV19p48aNSk1NlSQNHz5cb775pvr166cOHTroj3/8o8LDw9WzZ0/5%2B/tr0KBBSk9P1xtvvCFJSk9P15AhQxwBrT4iIiJqnQ4sLCxRdXXT20nU1FxtktvVUPSlbvSlbrdzX2623bdzb0y43XrnSa8XrwxYQUFBWrdunRYuXKiYmBi1aNFCcXFxSkpKUrNmzZSamqp58%2Bbp3LlzCg8P15QpUzR06FBJ0ogRI1RSUqKkpCQVFRWpW7duWrNmjfz9/SX9eA%2BtJUuWKC4uTlVVVXrsscf08ssvu3NzAQCAl/GxX7sBFBpVYWGJu0swymLxVWhokIqLyzzmvxY8AX2pG32pW2P1ZdDyXcbW1dg%2BmRpT57invma8qbfS9fvb1Nzo9dKmTQu31OQ5l9sDAAA0EQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADPPagLV7926NHDlSDzzwgGJiYpSWliabzSZJOnjwoEaOHKno6GgNGDBAW7dudXpsZmamYmNj1b17d8XHx%2BvAgQOOuZqaGi1ZskR9%2B/ZVdHS0Jk%2BerIKCApduGwAA8G5eGbCKior03HPPafTo0dq3b58yMzO1Z88evf3227p06ZImTZqkYcOGae/evVqwYIEWLVqkr7/%2BWpKUk5OjtLQ0LV68WHv37tWvfvUrTZ48WRUVFZKk1atXa9euXfrggw/0t7/9TQEBAZozZ447NxcAAHgZrwxYYWFh%2Bvvf/674%2BHj5%2BPjo4sWLunLlisLCwpSdna2QkBCNHTtWFotFffr0UVxcnLZs2SJJ2rp1qwYPHqwePXrI399fzzzzjEJDQ5WVleWYnzhxou644w4FBwdr9uzZ%2Bvzzz3X69Gl3bjIAAPAiXhmwJCk4OFiS9OijjyouLk5t2rRRfHy8jh07pqioKKdlO3XqpNzcXElSXl7ededLSkp09uxZp/nw8HC1atVKR48ebeQtAgAATYXF3QU0VHZ2ti5duqSUlBQlJyerbdu2slqtTssEBASovLxcklRWVnbd%2BbKyMklSYGBgrflrc/VRUFCgwsJCpzGLJVARERH1Xoen8/PzdfqNH9GXutGXutEXadDyXe4uoUmzWG6P15Yn/i15fcAKCAhQQECAZsyYoZEjRyoxMVElJSVOy9hsNgUFBUmSrFar42L4f5wPDQ11BK9r12PV9fj6yMjI0MqVK53GkpKSlJycXO91eIuWLa03X%2Bg2RF/qRl/qRl/QWEJD6//e1RR40t%2BSVwasL7/8Ui%2B99JI%2B%2BugjNWvWTJJUWVkpf39/derUSbt2Of8XUV5eniIjIyVJkZGROnbsWK35fv36qVWrVmrbtq3TacTCwkJdvHix1mnFG0lISNCAAQOcxiyWQBUX1/8omKfz8/NVy5ZWXb5coZqaq%2B4ux2PQl7rRl7rRFzS2pvS%2BcyM3%2BltyV8j0yoDVuXNn2Ww2LV26VNOnT1dhYaGWLFmiESNG6IknntDSpUu1YcMGjR07Vvv379f27du1atUqSdKIESOUlJSkQYMGqUePHtqyZYsuXLig2NhYSVJ8fLxWr16tbt26KTQ0VAsXLlSvXr30i1/8ot71RURE1DodWFhYourqprcDram52iS3q6HoS93oS93oCxrL7fa68qS/Ja8MWEFBQVq3bp0WLlyomJgYtWjRQnFxcUpKSlKzZs30zjvvaMGCBVqxYoXCwsI0Z84cPfTQQ5KkPn36aO7cuZo3b57OnTunTp06ae3atQoJCZH046m86upqjR07VmVlZerdu7eWL1/uzs0FAABexsdut9vdXcTtoLCw5OYLeRGLxVehoUEqLi7zmP9a8AT0pW70pW6N1RcuHMc1n0yNcXcJLnGjv6U2bVq4pSbPudweAACgiSBgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMO88kajAADg5rzpnmhN7Z5dHMECAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGCYxd0FAICnG7R8l7tLAOBlOIIFAABgmMsDVk1NjZH15Obmavz48erVq5diYmI0c%2BZMFRUVSZLmzp2rrl27Kjo62vGTkZHheOzatWvVr18/de/eXYmJiTpx4oRjrry8XKmpqerdu7d69OihmTNnqqyszEjNAADg9uDygNWvXz/94Q9/UF5e3i2vw2azacKECYqOjtb//u//6uOPP9bFixf10ksvSZIOHTqktLQ0HThwwPGTkJAgScrMzNSmTZu0fv165eTkqEuXLkpOTpbdbpckpaWl6cyZM9qxY4eys7N15swZpaenN3zDAQDAbcPlAeuFF17Ql19%2BqSFDhmjkyJF69913VVJS8pPWkZ%2Bfr7vvvltJSUlq1qyZQkNDlZCQoL1796qyslLffvutunbtWudj33vvPY0ZM0aRkZFq3ry5pk%2Bfrvz8fOXk5KiiokLbt29XcnKyQkJC1Lp1a6WkpGjbtm2qqKgwsfkAAOA24PKL3EePHq3Ro0fr5MmTyszM1Nq1a7Vo0SINHDhQw4cPV9%2B%2BfW%2B6jl/%2B8pdat26d09iOHTvUpUsX5ebmqrq6WitWrND%2B/fvVokULDR8%2BXBMmTJCvr6/y8vI0ceJEx%2BP8/f3Vvn175ebmKiQkRFVVVYqKinLMd%2BzYUTabTSdPntQ999xTr20sKChQYWGh05jFEqiIiIh6Pd4b%2BPn5Ov3Gj%2BhL3egLgJuxWG59/%2BCJ%2Bxi3fYqwffv2mjZtml544QX96U9/0qpVq5SVlaU77rhDiYmJevrpp%2BXn53fT9djtdi1fvlw7d%2B7U5s2bdf78efXq1UuJiYlatmyZjhw5oqSkJPn6%2BmrChAkqKyuT1Wp1WkdAQIDKy8tVWloqSQoMDHTMXVv2p1yHlZGRoZUrVzqNJSUlKTk5ud7r8BYtW1pvvtBtiL7Ujb4AuJ7Q0KAGr8OT9jFuC1gHDx7Uf//3fysrK0uVlZWKjY1VfHy8zp07pzfeeEOHDh3SsmXLbriO0tJSpaam6vDhw9q8ebM6d%2B6szp07KyYmxrHMfffdp3HjxikrK0sTJkyQ1WqVzWZzWo/NZlNQUJAjWFVUVCgoKMjxvyUpODi43tuWkJCgAQMGOI1ZLIEqLm46F8v7%2BfmqZUurLl%2BuUE3NVXeX4zHoS93oC4Cbach75I32MSaC261wecBatWqVPvzwQ33//ffq1q2bpk2bpiFDhjgFGD8/P73yyis3XM%2BpU6c0ceJE3XnnnXr//fcVFhYmSfr00091/vx5jRo1yrFsZWWlAgICJEmRkZE6duyY%2BvfvL0mqqqrSyZMnFRUVpQ4dOsjf3195eXm6//77JUnHjx93nEasr4iIiFqnAwsLS1Rd3fTeWGpqrjbJ7Woo%2BlI3%2BgLgekzsGzxpH%2BPyk5WbN29W//799fHHH2vr1q0aNWpUraNDHTt2VEpKynXXcenSJY0bN04PPPCA1q9f7whX0o%2BnDBctWqTdu3fLbrfrwIED2rhxo%2BNThMOHD9fmzZuVm5urK1euaOnSpQoPD1fPnj1ltVo1aNAgpaenq6ioSEVFRUpPT9eQIUMcAQ0AAOBmXH4E6/PPP1dpaakuXrzoGMvKylKfPn0UGhoqSbr33nt17733Xncd27ZtU35%2Bvj755BP9%2Bc9/dpo7cOCAUlNTNW/ePJ07d07h4eGaMmWKhg4dKkkaMWKESkpKlJSUpKKiInXr1k1r1qyRv7%2B/pB/vobVkyRLFxcWpqqpKjz32mF5%2B%2BWXTbQAAAE2Yj/3aDaBc5Ouvv9bEiRMVHx%2BvWbNmSZL69%2B%2BvqqoqvfPOO06f4GtKCgt/2q0oPJ3F4qvQ0CAVF5d5zOFYT0Bf6ubtfeGrcoDG98nUmJsvdB032se0adOioaXdEpefIvzDH/6gxx9/XNOmTXOMffrpp%2BrXr58WL17s6nIAAACMc3nAOnz4sCZNmqRmzZo5xvz8/DRp0iR99dVXri4HAADAOJcHrODgYJ06darW%2BNmzZ7mQHAAANAkuD1hPPPGE5s2bp7///e8qLS1VWVmZvvjiC82fP1%2BxsbGuLgcAAMA4l3%2BKcPr06Tp9%2BrSeffZZ%2Bfj4OMZjY2M1c%2BZMV5cDAABgnMsDltVq1Zo1a/Tdd9/p6NGj8vf3V8eOHX/SjTwBAAA8mdu%2BKqdDhw7q0KGDu54eAACg0bg8YH333XeaP3%2B%2B9u/fr6qqqlrzR44ccXVJAAAARrk8YM2bN0/5%2BflKSUlRixbuufkXAABAY3J5wDpw4ID%2B8z//U9HR0a5%2BagAAAJdw%2BW0aQkNDFRQU5OqnBQAAcBmXB6zExEQtW7ZMJSVN67v5AAAArnH5KcLPPvtMX331lXr37q3WrVs7fWWOJP3lL39xdUkAAABGuTxg9e7dW71793b10wIAALiMywPWCy%2B84OqnBAAAcCmXX4MlSbm5uUpNTdWoUaN07tw5bdmyRTk5Oe4oBQAAwDiXB6xvvvlGI0eO1A8//KBvvvlGlZWVOnLkiJ599lnt3LnT1eUAAAAY5/KAlZ6ermeffVabNm2Sv7%2B/JOm1117T008/rZUrV7q6HAAAAOPccgRr2LBhtcZHjx6tEydOuLocAAAA41wesPz9/VVaWlprPD8/X1ar1dXlAAAAGOfygDVw4EAtXbpUxcXFjrHjx49rwYIF%2Btd//VdXlwMAAGCcywPWrFmzZLPZ1LdvX1VUVCg%2BPl5DhgyRxWLRzJkzXV0OAACAcS6/D1ZwcLDeffdd7d69W//v//0/Xb16VVFRUXrkkUfk6%2BuWu0YAAAAY5fKAdU2fPn3Up08fdz09AABAo3F5wBowYIB8fHyuO893EQIAAG/n8oD11FNPOQWsqqoqff/99/r88881depUV5cDAABgnMsD1pQpU%2Boc37x5s/bv36%2Bnn37axRUBAACY5TFXlffv31%2BfffaZu8sAAABoMI8JWHv27FHz5s3rvXxubq7Gjx%2BvXr16KSYmRjNnzlRRUZEk6eDBgxo5cqSio6M1YMAAbd261emxmZmZio2NVffu3RUfH68DBw445mpqarRkyRL17dtX0dHRmjx5sgoKCsxsJAAAuC24/BThP58CtNvtKi0t1dGjR%2Bt9etBms2nChAn69a9/rTVr1qisrEyzZs3SSy%2B9pCVLlmjSpElKTk5WQkKC9u7dq6SkJHXu3Fn33XefcnJylJaWprVr1%2Bq%2B%2B%2B7Tli1bNHnyZO3cuVNWq1WrV6/Wrl279MEHH6hFixZ6%2BeWXNWfOHL399tuN0Q4AANAEufwI1p133ql27do5fn7%2B85%2BrW7duWrhwoWbMmFGvdeTn5%2Bvuu%2B9WUlKSmjVrptDQUEeYys7OVkhIiMaOHSuLxaI%2BffooLi5OW7ZskSRt3bpVgwcPVo8ePeTv769nnnlGoaGhysrKcsxPnDhRd9xxh4KDgzV79mx9/vnnOn36dKP1BAAANC0uP4K1ePHiBq/jl7/8pdatW%2Bc0tmPHDnXp0kXHjh1TVFSU01ynTp30/vvvS5Ly8vI0fPjwWvO5ubkqKSnR2bNnnR4fHh6uVq1a6ejRo7rrrrvqVV9BQYEKCwudxiyWQEVERNR7Gz2dn5%2Bv02/8iL7Ujb4AuBmL5db3D564j3F5wNq7d2%2B9l33wwQdvuozdbtfy5cu1c%2BdObd68WRs3bqz1pdEBAQEqLy%2BXJJWVlV13vqysTJIUGBhYa/7aXH1kZGRo5cqVTmNJSUlKTk6u9zq8RcuWfEF3XehL3egLgOsJDQ1q8Do8aR/j8oD1zDPPyG63O36uuXZvrGtjPj4%2BOnLkyA3XVVpaqtTUVB0%2BfFibN29W586dZbVaVVJS4rSczWZTUNCP/8dZrVbZbLZa86GhoY7gVVFRcd3H10dCQoIGDBjgNGaxBKq4uP4hzdP5%2BfmqZUurLl%2BuUE3NVXeX4zHoS93oC4Cbach75I32MSaC261wecB68803tWjRIs2aNUsPPfSQ/P39dfDgQc2bN09jxoxR//7967WeU6dOaeLEibrzzjv1/vvvKywsTJIUFRWlXbt2OS2bl5enyMhISVJkZKSOHTtWa75fv35q1aqV2rZtq7y8PMdpwsLCQl28eLHWaccbiYiIqHU6sLCwRNXVTe%2BNpabmapPcroaiL3WjLwCux8S%2BwZP2MS4/WblkyRLNnTtXAwcOVHBwsJo3b65evXpp/vz5euedd5wugL%2BeS5cuady4cXrggQe0fv16R7iSpNjYWJ0/f14bNmxQVVWVvvjiC23fvt1x3dWIESO0fft2ffHFF6qqqtKGDRt04cIFxcbGSpLi4%2BO1evVqnT59WqWlpVq4cKF69eqlX/ziF43bGAAA0GS4/AhWQUGB7rjjjlrjwcHBKi4urtc6tm3bpvz8fH3yySf685//7DR34MABvfPOO1qwYIFWrFihsLAwzZkzRw899JCkH79keu7cuZo3b57OnTunTp06ae3atQoJCZH047VS1dXVGjt2rMrKytS7d28tX768gVsNAABuJz72f7wQygXGjx%2BvwMBALVmyRMHBwZKkixcvavr06WrevLlWrVrlynJcprCw5OYLeRGLxVehoUEqLi7zmMOxnoC%2B1M3b%2BzJo%2Ba6bLwSgQT6ZGnPLj73RPqZNmxYNLe3WanL1E86ZM0fjxo1Tv3791L59e0nSd999pzZt2mjjxo2uLgdoMrwpBDRkRwoA3sDlAatjx47KysrS9u3bdfz4cUnSmDFjNHjw4Fq3TwAAAPBGLg9YktSyZUuNHDlSP/zwg%2BPmnf7%2B/u4oBQAAwDiXf4rQbrcrPT1dDz74oIYMGaKzZ89q1qxZSk1NVVVVlavLAQAAMM7lAWvTpk368MMPNXfuXDVr1kySNHDgQP31r3/VG2%2B84epyAAAAjHN5wMrIyNArr7yi%2BPh4x93bn3zySS1YsED/8z//4%2BpyAAAAjHN5wPrhhx90zz331Brv3Lmzzp8/7%2BpyAAAAjHN5wGrXrp2%2B/vrrWuOfffaZ44J3AAAAb%2BbyTxH%2B5je/0auvvqpz587Jbrdr9%2B7devfdd7Vp0yalpqa6uhwAAADjXB6whg8frurqaq1evVo2m02vvPKKWrdurWnTpmn06NGuLgcAAMA4lwesjz76SP/2b/%2BmhIQEFRUVyW63q3Xr1q4uAwAAoNG4/Bqs1157zXExe1hYGOEKAAA0OS4PWO3bt9fRo0dd/bQAAAAu4/JThJGRkUpJSdG6devUvn17NW/e3Gl%2B0aJFri4JAADAKJcHrFOnTqlHjx6SpMLCQlc/PQAAQKNzScAzCQj7AAAXXklEQVRatGiRXnzxRQUGBmrTpk2ueEoAAAC3cck1WBs3blRFRYXT2G9%2B8xsVFBS44ukBAABcyiUBy2631xr78ssvdeXKFVc8PQAAgEu5/FOEAAAATR0BCwAAwDCXBSwfHx9XPRUAAIBbuew2Da%2B99prTPa%2Bqqqr0%2BuuvKygoyGk57oMFAAC8nUsC1oMPPljrnlfR0dEqLi5WcXGxK0oAAABwGZcELO59BQAAbidc5A4AAGAYAQsAAMAwAhYAAIBhBCwAAADDvD5gFRUVKTY2Vjk5OY6xuXPnqmvXroqOjnb8ZGRkOObXrl2rfv36qXv37kpMTNSJEyccc%2BXl5UpNTVXv3r3Vo0cPzZw5U2VlZS7dJgAA4N28OmDt379fCQkJOnXqlNP4oUOHlJaWpgMHDjh%2BEhISJEmZmZnatGmT1q9fr5ycHHXp0kXJycmO70tMS0vTmTNntGPHDmVnZ%2BvMmTNKT093%2BbYBAADv5bUBKzMzUykpKZo2bZrTeGVlpb799lt17dq1zse99957GjNmjCIjI9W8eXNNnz5d%2Bfn5ysnJUUVFhbZv367k5GSFhISodevWSklJ0bZt21RRUeGKzQIAAE2Ay%2B7kbtrDDz%2BsuLg4WSwWp5CVm5ur6upqrVixQvv371eLFi00fPhwTZgwQb6%2BvsrLy9PEiRMdy/v7%2B6t9%2B/bKzc1VSEiIqqqqFBUV5Zjv2LGjbDabTp48qXvuuadetRUUFNS6sarFEqiIiIgGbrXn8PPzdfqNH9GX%2BrFY6A8AZw3ZL3jivtdrA1abNm3qHC8pKVGvXr2UmJioZcuW6ciRI0pKSpKvr68mTJigsrIyWa1Wp8cEBASovLxcpaWlkqTAwEDH3LVlf8p1WBkZGVq5cqXTWFJSkpKTk%2Bu9Dm/RsqX15gvdhujLjYWGBt18IQC3FRP7BU/a93ptwLqemJgYxcTEOP593333ady4ccrKytKECRNktVpls9mcHmOz2RQUFOQIVhUVFY7vSLx2ajA4OLjeNSQkJGjAgAFOYxZLoIqLm87F8n5%2BvmrZ0qrLlytUU3PV3eV4DPpSP03pbwGAGQ3ZL9xo3%2Buu/6BrcgHr008/1fnz5zVq1CjHWGVlpQICAiRJkZGROnbsmPr37y/pxy%2BdPnnypKKiotShQwf5%2B/srLy9P999/vyTp%2BPHjjtOI9RUREVHrdGBhYYmqq5veG25NzdUmuV0NRV9ujN4A%2BGcm9guetO/1nJOVhtjtdi1atEi7d%2B%2BW3W7XgQMHtHHjRsenCIcPH67NmzcrNzdXV65c0dKlSxUeHq6ePXvKarVq0KBBSk9PV1FRkYqKipSenq4hQ4Y4AhoAAMDNNLkjWLGxsUpNTdW8efN07tw5hYeHa8qUKRo6dKgkacSIESopKVFSUpKKiorUrVs3rVmzRv7%2B/pJ%2BvIfWkiVLFBcXp6qqKj322GN6%2BeWX3blJAADAy/jYr90ACo2qsLDE3SUYZbH4KjQ0SMXFZR5zONYTuLMvg5bvcunzNcQnU2NuvpAH8abeAt6qIfuFG%2B1727Rp0dDSbq0mtzwr4AV4U2089BZAU9fkrsECAABwNwIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAw7w%2BYBUVFSk2NlY5OTmOsYMHD2rkyJGKjo7WgAEDtHXrVqfHZGZmKjY2Vt27d1d8fLwOHDjgmKupqdGSJUvUt29fRUdHa/LkySooKHDZ9gAAAO/n1QFr//79SkhI0KlTpxxjly5d0qRJkzRs2DDt3btXCxYs0KJFi/T1119LknJycpSWlqbFixdr7969%2BtWvfqXJkyeroqJCkrR69Wrt2rVLH3zwgf72t78pICBAc%2BbMccv2AQAA7%2BS1ASszM1MpKSmaNm2a03h2drZCQkI0duxYWSwW9enTR3FxcdqyZYskaevWrRo8eLB69Oghf39/PfPMMwoNDVVWVpZjfuLEibrjjjsUHBys2bNn6/PPP9fp06ddvo0AAMA7WdxdwK16%2BOGHFRcXJ4vF4hSyjh07pqioKKdlO3XqpPfff1%2BSlJeXp%2BHDh9eaz83NVUlJic6ePev0%2BPDwcLVq1UpHjx7VXXfdVa/aCgoKVFhY6DRmsQQqIiLiJ22jJ/Pz83X6DQBAQ1gst/5%2B4onvSV4bsNq0aVPneFlZmaxWq9NYQECAysvLbzpfVlYmSQoMDKw1f22uPjIyMrRy5UqnsaSkJCUnJ9d7Hd6iZUvrzRcCAOAmQkODGrwOT3pP8tqAdT1Wq1UlJSVOYzabTUFBQY55m81Waz40NNQRvK5dj1XX4%2BsjISFBAwYMcBqzWAJVXFz/kObp/Px81bKlVZcvV6im5qq7ywEAeLmGvEfe6D3JRHC7FU0uYEVFRWnXrl1OY3l5eYqMjJQkRUZG6tixY7Xm%2B/Xrp1atWqlt27bKy8tznCYsLCzUxYsXa512vJGIiIhapwMLC0tUXd30gkhNzdUmuV0AANcy8V7iSe9JnnOy0pDY2FidP39eGzZsUFVVlb744gtt377dcd3ViBEjtH37dn3xxReqqqrShg0bdOHCBcXGxkqS4uPjtXr1ap0%2BfVqlpaVauHChevXqpV/84hfu3CwAAOBFmtwRrNDQUL3zzjtasGCBVqxYobCwMM2ZM0cPPfSQJKlPnz6aO3eu5s2bp3PnzqlTp05au3atQkJCJP14rVR1dbXGjh2rsrIy9e7dW8uXL3fnJgEAAC/jY7fb7e4u4nZQWFhy84W8iMXiq9DQIBUXl3nM4VjTBi3fdfOFAABGfDI15pYfe6P3pDZtWjS0tFvS5E4RAgAAuBsBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMOabMDKysrSvffeq%2BjoaMfPjBkzJEmfffaZ4uLi1L17dw0aNEg7d%2B50euzatWvVr18/de/eXYmJiTpx4oQ7NgEAAHipJhuwDh06pKFDh%2BrAgQOOn9dff10nT57UlClT9OKLL2rfvn2aMmWKpk6dqnPnzkmSMjMztWnTJq1fv145OTnq0qWLkpOTZbfb3bxFAADAWzTpgNW1a9da45mZmerZs6cGDhwoi8WiJ598Ug8%2B%2BKAyMjIkSe%2B9957GjBmjyMhINW/eXNOnT1d%2Bfr5ycnJcvQkAAMBLWdxdQGO4evWqDh8%2BLKvVqnXr1qmmpkaPPvqoUlJSlJeXp6ioKKflO3XqpNzcXElSXl6eJk6c6Jjz9/dX%2B/btlZubq4ceeqhez19QUKDCwkKnMYslUBEREQ3cMs/h5%2Bfr9BsAgIawWG79/cQT35OaZMAqKirSvffeqyeeeEIrVqxQcXGxZs2apRkzZqiyslJWq9Vp%2BYCAAJWXl0uSysrKbjhfHxkZGVq5cqXTWFJSkpKTk29xizxXy5bWmy8EAMBNhIYGNXgdnvSe1CQDVnh4uLZs2eL4t9Vq1YwZM/TrX/9avXv3ls1mc1reZrMpKCjIseyN5usjISFBAwYMcBqzWAJVXFz2UzfFY/n5%2BaplS6suX65QTc1Vd5cDAPByDXmPvNF7kongdiuaZMDKzc3Vxx9/rOnTp8vHx0eSVFlZKV9fX9133306cuSI0/J5eXmO67UiIyN17Ngx9e/fX5JUVVWlkydP1jqteCMRERG1TgcWFpaourrpBZGamqtNcrsAAK5l4r3Ek96TPOdkpUEhISHasmWL1q1bp%2BrqauXn5%2Bv111/XU089pWHDhmnPnj3KyspSdXW1srKytGfPHg0dOlSSNHz4cG3evFm5ubm6cuWKli5dqvDwcPXs2dPNWwUAALxFkzyC9bOf/Uxr1qzRsmXLtHr1ajVv3lyDBw/WjBkz1Lx5c7311ltKT0/X7Nmz1a5dO7355pvq0KGDJGnEiBEqKSlRUlKSioqK1K1bN61Zs0b%2B/v5u3ioAAOAtfOzc4MklCgtL3F2CURaLr0JDg1RcXOYxh2NNG7R8l7tLAIDbxidTY275sTd6T2rTpkVDS7slTfIUIQAAgDsRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhTfJGo/BM3FcKAHC74AgWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMP4qhwvx9fPAADgeTiCBQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgFWHCxcu6Pnnn1fPnj3Vu3dvLViwQNXV1e4uCwAAeAkCVh2mTp2qwMBA/e1vf9P777%2Bv3bt3a8OGDe4uCwAAeAkC1j/5/vvvtWfPHs2YMUNWq1V33XWXnn/%2BeW3ZssXdpQEAAC9BwPonx44dU0hIiNq2besY69ixo/Lz83X58mU3VgYAALwFX/b8T8rKymS1Wp3Grv27vLxcLVu2vOk6CgoKVFhY6DRmsQQqIiLCXKEAADQhFsutH/Px8/N1%2Bu0JCFj/JDAwUBUVFU5j1/4dFBRUr3VkZGRo5cqVTmMvvPCCpkyZYqbIf7Bvwb8ZX2d9FBQUKCMjQwkJCQTHf0Bf6kZf6kZfro/e1I2%2B1K2goED/%2BZ/rPKovnhP1PERkZKQuXryo8%2BfPO8aOHz%2Bun/3sZ2rRokW91pGQkKBt27Y5/SQkJDRWyW5RWFiolStX1jpSd7ujL3WjL3WjL9dHb%2BpGX%2BrmiX3hCNY/ad%2B%2BvXr06KGFCxdq/vz5Ki4u1qpVqzRixIh6ryMiIsJjEjQAAHA9jmDVYcWKFaqurtZjjz2mX//613rkkUf0/PPPu7ssAADgJTiCVYfw8HCtWLHC3WUAAAAv5Tdv3rx57i4C3ikoKEi9evWq98X/twv6Ujf6Ujf6cn30pm70pW6e1hcfu91ud3cRAAAATQnXYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAM46ty4KSmpkbPPPOM2rVrp8WLF2vChAnav3%2B/0zLl5eVKSEjQ/Pnzaz3%2Bhx9%2B0OLFi7Vv3z7Z7Xb16NFDqampuuuuu1y1CY2mob05cuSIFi1apMOHD8tisahfv3566aWXFBoa6qpNaBQN7cs/mjFjhs6ePatNmzY1Zsku0dC%2BHDx4UAkJCbJarY6xe%2B%2B9V1u2bGn02htTQ/ty5coVvf766/rkk09ks9nUtWtXvfLKK%2BrYsaOrNqFRNKQv%2B/bt08SJE53GqqqqVFVVpc8//1xt27Zt9PobU0NfM6dPn9b8%2BfN18OBB%2Bfn56ZFHHtGcOXPUsmXLxi3cDvyD5cuX2%2B%2B%2B%2B277rFmz6pzfunWr/dFHH7WfO3euzvlf/epX9pdeesleVlZmLy0ttaemptqHDBnSmCW7TEN6c%2BXKFXtMTIx95cqV9qqqKvulS5fs48aNs8%2BcObOxy250DX3N/ONyd999t/3f//3fG6NMl2toXzZt2tRkevGPGtqX3//%2B9/ZRo0bZz507Z79y5Yr91VdftQ8ePLgxS3YJU39HdrvdXlJSYn/yySftb731luky3aKhvRkxYoR98eLF9srKSntxcbF97Nix9tTU1MYs2W632%2B0cwYLD7t27lZ2drccff7zO%2BRMnTigtLU3r169XRERErflLly4pPDxcL774ogIDAyVJTz/9tIYOHapLly6pVatWjVp/Y2pob5o1a6bs7GwFBATI19dXly5dUkVFhcLCwhq79EbV0L5ck5eXp1WrVmnkyJH67rvvGqtclzHRl0OHDqlr166NWabLNbQvFy5c0IcffqisrCzHfEpKir777jvZ7Xb5%2BPg0av2NxdTf0TWvvfaa2rZtq%2Beff950qS5nojfHjx9Xjx49ZLfbHa%2BTfzwy3Fi4BguSftxxzZ49W0uXLr3uC%2B/VV1/VsGHD1LNnzzrnW7VqVetFvmPHDrVr186rw5WJ3khSYGCgfH19NWrUKA0cOFClpaX6zW9%2B01hlNzpTfbHZbJo2bZrmzp2rNm3aNFa5LmOqL4cOHdLhw4f1%2BOOPq2/fvpo6darOnj3bWGU3OhN9%2Beabb9SiRQt99dVXGjx4sPr06aOZM2cqNDTUa8OVqdfLNfv27VNWVpbS0tJMl%2BpypnozZcoUbd68Wd27d9dDDz2kyspKpaSkNFbZDgQs6OrVq5oxY4bGjx%2Bvu%2B%2B%2Bu85l9u3bp4MHD%2BqFF16o93r/67/%2BS%2B%2B8845ee%2B01U6W6XGP0ZsOGDdqzZ4%2BioqI0fvx41dTUmCzZJUz2Zf78%2BYqJidGjjz7aGKW6lKm%2B1NTUKCIiQg8//LA%2B%2BOADffzxx/Lx8dGkSZNu69fLpUuXVFJSouzsbG3atEnZ2dmyWq367W9/e1v35R%2B9%2BeabGj16tNq1a2eyVJcz2RsfHx9NnjxZ%2B/bt01//%2BldJ0iuvvGK85n9GwILWrFmjZs2aKTEx8brLZGRkaNCgQfU6wlBZWalXX31Vy5cv15o1a9S3b1%2BT5bqU6d5IUkBAgFq1aqU5c%2Bbo22%2B/1dGjR02V6zKm%2BvLRRx8pNzdXv/vd7xqjTJcz1Rc/Pz9t2LBBkyZNUosWLRQWFqaXX35ZR48e1fHjxxuj9EZlqi/NmjVTTU2NZs2apbCwMLVo0UKpqak6evSoV55aNr1/OXXqlPbs2XPD9XkLU7355ptv9MYbb%2Bi5555TYGCg2rVrp5kzZ2r79u0qLS1tjNIduAYL%2BvDDD1VQUOA4xGqz2SRJn376qfbt26fq6mr95S9/0VtvvXXTdRUVFWny5MmqrKzU%2B%2B%2B/7/WfHjTVmx9%2B%2BEFPP/203n33Xccp1MrKSknyytOnpvry4Ycf6rvvvnOE8CtXrqimpkY9e/bURx99pDvvvLNxN8QwU305c%2BaMNmzYoOTkZAUFBUn6v9dLQEBAI25B4zDVl06dOkn6v15Ichy5stvtjVF6ozK575V%2BvCTjgQce0M9//vNGq9lVTP4t1dTU6OrVq44xf39/%2Bfj4yM/Pr/E2QOJThKht1qxZTp/W%2BOabb%2Bz33nuv3Waz3fBxlZWV9qeeesr%2B7LPP2isqKhq7TLe41d5cvXrV/tRTT9mnTp1qLy0ttV%2B4cMH%2B3HPP2SdMmNDYJbvErfbln61YsaJJfXLuVvtSUVFhj4mJsaelpdltNpv9woUL9t/%2B9rf2cePGNXLFrtGQ18vYsWPto0aNsl%2B4cMFeWlpq/93vfmd/6qmnGrNcl2no39Fzzz1nX7ZsWWOV51a32psLFy7Ye/XqZZ87d67dZrPZz58/b3/66aftU6ZMaeyS7ZwixE2dPn1arVq1UvPmzWvN/cd//IcGDx4sSdq5c6cOHz6svXv3qk%2BfPoqOjnb85Ofnu7psl6hvb3x8fLRq1SpVV1drwIABGjp0qO644w4tW7bM1SW7RH37crupb18CAgK0bt06HT9%2BXA8//LCeeOIJBQcHa/ny5a4u2SV%2Byutl9erVioyM1LBhw/TII4%2BovLxcq1atcmW5LvNT/45%2B%2BOEHr7/nVX3VtzdhYWFav369Tp48qUceeUTDhg1T%2B/bttXDhwkav0cdu98LjqgAAAB6MI1gAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGH/H4keRQmutY%2BqAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-7618312233477638842">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">47.6624</td>
        <td class="number">17</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.5491</td>
        <td class="number">17</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.5322</td>
        <td class="number">17</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.6846</td>
        <td class="number">17</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.6711</td>
        <td class="number">16</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.6886</td>
        <td class="number">16</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.6955</td>
        <td class="number">16</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.6647</td>
        <td class="number">15</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.6904</td>
        <td class="number">15</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.68600000000001</td>
        <td class="number">15</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (5023)</td>
        <td class="number">21436</td>
        <td class="number">99.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-7618312233477638842">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">47.1559</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.1593</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.1622</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.1647</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.1764</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">47.7771</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:67%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.7772</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.7774</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:34%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.7775</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">47.7776</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_long">long<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>751</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>3.5%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>-122.21</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>-122.52</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>-121.31</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-5004649818657728902">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAASZJREFUeJzt28GJAkEQQFEVQzIIc/K8ORmEObUJyMcZdphW37sLfZhPIUUdxxjjALx02vsBMLPz3g/4D5fbffFvHn/XDV7CtzFBIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCATCVxxMreHIineYIBAEAkEgEAQCQSAQBAJBIBAEAuFnF4VrLF0uWix%2BPhMEgkAgCASCQCAIBIJAIAgEgkAgCASCTfqGnPV%2BPhMEgkAgCASCQCAIBIJAIAgEgj3IZOxO5jJdIGs%2BENjKdIGwnFv57RzHGGPvR8Cs/EmHIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKB8ATXuBwCBbx2lgAAAABJRU5ErkJggg%3D%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-5004649818657728902,#minihistogram-5004649818657728902"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-5004649818657728902">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-5004649818657728902"
                                                  aria-controls="quantiles-5004649818657728902" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-5004649818657728902" aria-controls="histogram-5004649818657728902"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-5004649818657728902" aria-controls="common-5004649818657728902"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-5004649818657728902" aria-controls="extreme-5004649818657728902"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-5004649818657728902">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>-122.52</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>-122.39</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>-122.33</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>-122.23</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>-122.12</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>-121.98</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>-121.31</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>1.204</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>0.203</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>0.14072</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>-0.0011515</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>1.0521</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>-122.21</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>0.11509</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.88489</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>-2639500</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>0.019803</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-5004649818657728902">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3X18Tnee//F3kiuROzdxE1VrRlaSdqp2ZBNSqrFCGCqmTWmmzbqZWvqIuInWaE0ogxQjLSVqrd5jR6rYjl1Fp%2BuBaS3aKltTRrRuukGuEkQk5Ob8/ugj%2Bc3VaBJ8c65cyev5eHj04fs9Oefz/biSvq9zznXiZVmWJQAAABjj7e4CAAAAGhsCFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwzOHuApoKp7PQ3SW4lbe3l1q3DtLFi0WqqLDcXU6DQ39qR49qRn9qR49q1lj7065dc7cclzNYsIW3t5e8vLzk7e3l7lIaJPpTO3pUM/pTO3pUM/pjFgELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAxzuLsANB0xGdvcXcIt%2BSD9QXeXAADwUJzBAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADPPYgHXkyBGlpKQoJiZGffr00fz583Xjxg1J0q5du5SYmKju3btr8ODB2rlzp8vXrl69WnFxcerevbtGjhypr7/%2Bumru2rVrmjFjhmJjYxUdHa3p06erqKjI1rUBAADP5pEBq6KiQk8//bQGDRqk/fv367333tOf//xnrV69WidPntSkSZM0ZcoUffrpp5o0aZLS09N1/vx5SdLmzZu1Zs0avf7669q3b5%2B6du2qyZMny7IsSdK8efN09uxZbd%2B%2BXTt27NDZs2eVlZXlzuUCAAAP45EB6/Lly3I6naqoqKgKRt7e3goICNDmzZsVExOjAQMGyOFwaMiQIerRo4dycnIkSe%2B%2B%2B66efPJJRUREqFmzZnr22WeVl5enffv2qbi4WFu2bNHkyZPVqlUrtWnTRtOmTdOmTZtUXFzsziUDAAAP4pEBKyQkRGPGjNGiRYvUrVs39e3bV507d9aYMWOUm5uryMhIl%2B3Dw8N19OhRSao27%2Bvrq86dO%2Bvo0aM6deqUSktLXea7dOmikpISnTx50pa1AQAAz%2BdwdwG3o6KiQv7%2B/po1a5aGDx%2BuU6dOaeLEiVq2bJmKiooUEBDgsr2/v7%2BuXbsmSTXOX716VZIUGBhYNVe57a3ch5Wfny%2Bn0%2Bky5nAEKjQ0tO6LbGR8fDwvyzsc9tVc2R9P7JNd6FHN6E/t6FHN6I9ZHhmwPvzwQ23fvl3btm2TJEVERCgtLU2ZmZn6x3/8R5WUlLhsX1JSoqCgIEnfB6Yfm68MVsXFxVXbV14aDA4OrnN9OTk5ys7OdhlLS0vT5MmTb2GVcLeQkCDbj9miRUDtGzVx9Khm9Kd29Khm9McMjwxYZ8%2BerfrEYCWHwyFfX19FRkbqyJEjLnO5ubm6//77JX0fxo4fP65%2B/fpJkkpLS3Xy5ElFRkYqLCxMvr6%2Bys3N1c9//nNJ0okTJ6ouI9ZVcnKy4uPjf1BfoAoKmu6nET3xHZGd/14%2BPt5q0SJAV64Uq7y8wrbjehJ6VDP6Uzt6VLPG2h93vFmWPDRg9enTRy%2B99JL%2B9V//VePGjVNeXp5WrlypxMREDRs2TG%2B%2B%2Baa2bt2qgQMHaseOHdq/f78yMjIkSY899piWL1%2BuuLg4hYWFacmSJWrbtq1iYmLk6%2BurwYMHKysrS6%2B88ookKSsrS0OHDpW/v3%2Bd6wsNDa12OdDpLFRZWeN5wTYF7vj3Ki%2Bv4HVSC3pUM/pTO3pUM/pjhkcGrPDwcK1atUpLly7Va6%2B9pubNm2vYsGFKS0uTn5%2BfVqxYoaysLGVkZKhjx45avny5wsLCJEnDhw9XYWGh0tLSdPHiRXXr1k2rVq2Sr6%2BvJGn27NlatGiREhMTVVpaqv79%2B2vWrFnuXC4AAPAwXlblcw5Qr5zOQneX4FYOh7cSsva4u4xb8kH6g7Ydy%2BHwVkhIkAoKinjn%2BCPoUc3oT%2B3oUc0aa3/atWvuluN63o0xAAAADRwBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhHhuwLl26pOnTpys2NlY9evTQhAkTlJ%2BfL0k6dOiQRowYoaioKMXHx2vDhg0uX7t582YlJCSoe/fuSkpK0sGDB6vmysvLtWjRIvXu3VtRUVFKTU2t2i8AAEBdeGzAmjRpkq5du6YPP/xQO3fulI%2BPj2bNmqXLly9r/PjxeuSRR3TgwAFlZmZqwYIFOnz4sCRp3759mjdvnhYuXKgDBw5o2LBhSk1NVXFxsSRp5cqV%2Bvjjj7Vx40bt2bNH/v7%2BmjlzpjuXCgAAPIxHBqwvv/xShw4d0sKFC9WiRQsFBwdr3rx5mjZtmnbs2KFWrVopJSVFDodDvXr1UmJiotatWydJ2rBhgx5%2B%2BGFFR0fL19dXY8aMUUhIiLZu3Vo1P27cOHXo0EHBwcHKyMjQ7t27debMGXcuGQAAeBCPDFiHDx9WeHi43n33XSUkJKhPnz5atGiR2rVrp%2BPHjysyMtJl%2B/DwcB09elSSlJub%2B6PzhYWFOnfunMt827Zt1bJlSx07dqz%2BFwYAABoFh7sLuB2XL1/WsWPHdP/992vz5s0qKSnR9OnT9dxzz6lt27YKCAhw2d7f31/Xrl2TJBUVFf3ofFFRkSQpMDCw2nzlXF3k5%2BfL6XS6jDkcgQoNDa3zPhobHx/Py/IOh301V/bHE/tkF3pUM/pTO3pUM/pjlkcGLD8/P0lSRkaGmjVrpuDgYKWnp%2Bvxxx9XUlKSSkpKXLYvKSlRUFCQJCkgIOCm8yEhIVXBq/J%2BrJt9fV3k5OQoOzvbZSwtLU2TJ0%2Bu8z7gfiEhdf83N6VFi4DaN2ri6FHN6E/t6FHN6I8ZHhmwwsPDVVFRodLSUjVr1kySVFFRIUn62c9%2Bpn//93932T43N1cRERGSpIiICB0/frzafFxcnFq2bKn27du7XEZ0Op26dOlStcuKNUlOTlZ8fLzLmMMRqIKCup8Fa2w88R2Rnf9ePj7eatEiQFeuFKu8vMK243oSelQz%2BlM7elSzxtofd7xZljw0YPXu3VudOnXSb3/7Wy1YsEDXr1/XkiVLNGDAAA0dOlTLli3TW2%2B9pZSUFH322WfasmWLXn31VUnS8OHDlZaWpsGDBys6Olrr1q3ThQsXlJCQIElKSkrSypUr1a1bN4WEhOjFF19Uz5499ZOf/KTO9YWGhla7HOh0FqqsrPG8YJsCd/x7lZdX8DqpBT2qGf2pHT2qGf0xwyMDlq%2Bvr9asWaOFCxdq0KBBun79uuLj45WRkaEWLVrojTfeUGZmppYtW6bWrVtr5syZeuCBByRJvXr10uzZszVnzhydP39e4eHhWr16tVq1aiXp%2B0t5ZWVlSklJUVFRkWJjY7V06VJ3LhcAAHgYL8uyLHcX0RQ4nYXuLsGtHA5vJWTtcXcZt%2BSD9AdtO5bD4a2QkCAVFBTxzvFH0KOa0Z/a0aOaNdb%2BtGvX3C3H9bwbYwAAABo4AhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwzCMfNArYYfDSj91dwi2x87ldAICa2X4Gq7y83O5DAgAA2Mr2gBUXF6ff//73ys3NtfvQAAAAtrA9YE2cOFGff/65hg4dqhEjRmj9%2BvUqLGzav0YGAAA0LrYHrCeeeELr16/Xtm3b1Lt3b61evVp9%2BvTRs88%2Bq08%2B%2BcTucgAAAIxz26cIO3furKlTp2rbtm1KS0vTRx99pLFjxyo%2BPl5vvvkm92oBAACP5bZPER46dEj/8R//oa1bt%2BrGjRtKSEhQUlKSzp8/r1deeUX/%2B7//q5dfftld5QEAANw22wPWq6%2B%2Bqvfff1%2BnTp1St27dNHXqVA0dOlTBwcFV2/j4%2BOiFF16wuzQAAAAjbA9Ya9eu1bBhwzR8%2BHCFh4ffdJsuXbpo2rRpNlcGAABghu0Ba/fu3bp69aouXbpUNbZ161b16tVLISEhkqT77rtP9913n92lAQAAGGH7Te5/%2BctfNGjQIOXk5FSNLV68WImJifrrX/9qdzkAAADG2R6wfv/732vgwIGaOnVq1dif/vQnxcXFaeHChXaXAwAAYJztAevIkSMaP368/Pz8qsZ8fHw0fvx4ffHFF3aXAwAAYJztASs4OFinT5%2BuNn7u3Dn5%2B/vbXQ4AAIBxtgesQYMGac6cOfrkk0909epVFRUV6X/%2B5380d%2B5cJSQk2F0OAACAcbZ/ivDZZ5/VmTNn9NRTT8nLy6tqPCEhQdOnT7e7HAAAAONsD1gBAQFatWqVvvnmGx07dky%2Bvr7q0qWLOnfubHcpAAAA9cJtvyonLCxMYWFh7jo8AABAvbE9YH3zzTeaO3euPvvsM5WWllab/%2Bqrr%2BwuCQAAwCjbA9acOXOUl5enadOmqXnz5nYfHgAAoN7ZHrAOHjyot99%2BW1FRUXYfGgAAwBa2P6YhJCREQUFBdh8WAADANrYHrJEjR%2Brll19WYWGh3YcGAACwhe2XCHft2qUvvvhCsbGxatOmjcuvzJGkjz76yO6SAAAAjLI9YMXGxio2NtbuwwIAANjG9oA1ceJEuw8JAABgK9vvwZKko0ePasaMGfrVr36l8%2BfPa926ddq3b587SgEAADDO9oD15ZdfasSIEfr222/15Zdf6saNG/rqq6/01FNPaefOnXaXAwAAYJztASsrK0tPPfWU1qxZI19fX0nS/PnzNWrUKGVnZ9tdDgAAgHFuOYP1yCOPVBt/4okn9PXXX9tdDgAAgHG2ByxfX19dvXq12nheXp4CAgLsLgcAAMA42wPWgAED9NJLL6mgoKBq7MSJE8rMzNQ//dM/2V0OAACAcbYHrOeee04lJSXq3bu3iouLlZSUpKFDh8rhcGj69Ol2lwMAAGCc7c/BCg4O1vr167V371795S9/UUVFhSIjI/XQQw/J29stT40AAAAwyvaAValXr17q1auXuw4PAABQb2wPWPHx8fLy8vrReX4XIQAA8HS2B6xHH33UJWCVlpbq1KlT2r17t9LT0%2B0uBwAAwDjbA9akSZNuOr527Vp99tlnGjVqlM0VAQAAmNVg7irv16%2Bfdu3a5e4yAAAA7liDCVj79%2B9Xs2bN3F0GAADAHbP9EuEPLwFalqWrV6/q2LFjXB4EAACNgu0B6%2B677672KUJfX1%2BNHj1aiYmJdpcDAABgnO0Ba%2BHChXYfEgAAwFa2B6wDBw7UedsePXrUYyUAAAD1w/aANWbMGFmWVfWnUuVlw8oxLy8vffXVV3aXBwAAcMdsD1jLly/XggUL9Nxzz%2BmBBx6Qr6%2BvDh06pDlz5ujJJ59Uv3797C4JAADAKNsf07Bo0SLNnj1bAwYMUHBwsJo1a6aePXtq7ty5euONN9SxY8eqPwAAAJ7I9oCVn5%2BvDh06VBsPDg5WQUGB3eUAAAAYZ3vA6t69u15%2B%2BWVdvXq1auzSpUtavHixevXqZXc5AAAAxtl%2BD9bMmTM1evRoxcXFqXPnzpKkb775Ru3atdM777xjdzkAAADG2R6wunTpoq1bt2rLli06ceKEJOnJJ5/Uww8/rICAALvLAQAAMM72gCVJLVq00IgRI/Ttt9%2BqU6dOkr5/mjsAAEBjYPs9WJZlKSsrSz169NDQoUN17tw5Pffcc5oxY4ZKS0tveX/l5eUaOXKknn/%2B%2BaqxXbt2KTExUd27d9fgwYO1c%2BdOl69ZvXq14uLi1L17d40cOVJff/111dy1a9c0Y8YMxcbGKjo6WtOnT1dRUdHtLxgAADQ5tgesNWvW6P3339fs2bPl5%2BcnSRowYID%2B%2B7//W6%2B88sot7y87O1uffvpp1d9PnjypSZMmacqUKfr00081adIkpaen6/z585KkzZs3a82aNXr99de1b98%2Bde3aVZMnT656wOm8efN09uxZbd%2B%2BXTt27NDZs2eVlZVlYOUAAKCpsD1g5eTk6IUXXlBSUlLV09uHDBmizMxM/dd//dct7Wvv3r3asWOHBg4cWDW2efNmxcTEaMCAAXI4HBoyZIh69OihnJwcSdK7776rJ598UhEREWrWrJmeffZZ5eXlad%2B%2BfSouLtaWLVs0efJktWrVSm3atNG0adO0adMmFRcXm2sCAABo1GwPWN9%2B%2B61%2B9rOfVRu/55579N1339V5PxcuXFBGRoZeeukll5vjc3NzFRkZ6bJteHi4jh49etN5X19fde7cWUePHtWpU6dUWlrqMt%2BlSxeVlJTo5MmTda4NAAA0bbbf5N6xY0cdPnxYf/d3f%2BcyvmvXrqob3mtTUVGh3/zmN/r1r3%2Bte%2B%2B912WuqKio2qcR/f39de3atVrnK5/NFRgYWDVXue2t3IeVn58vp9PpMuZwBCo0NLTO%2B2hsfHxsz/JNjsPRuHtc%2BRritXRz9Kd29Khm9Mcs2wPW2LFj9bvf/U7nz5%2BXZVnau3ev1q9frzVr1mjGjBl12seqVavk5%2BenkSNHVpsLCAhQSUmJy1hJSYmCgoJqna8MVsXFxVXbV14aDA4OrvMac3JylJ2d7TKWlpamyZMn13kfwK0KCQlydwm2aNGCx7nUhP7Ujh7VjP6YYXvAeuyxx1RWVqaVK1eqpKREL7zwgtq0aaOpU6fqiSeeqNM%2B3n//feXn5ysmJkaSqgLTn/70J6WkpOjIkSMu2%2Bfm5ur%2B%2B%2B%2BXJEVEROj48eNVv1S6tLRUJ0%2BeVGRkpMLCwuTr66vc3Fz9/Oc/lySdOHGi6jJiXSUnJys%2BPt5lzOEIVEFB0/00Iu%2BI6l9jf335%2BHirRYsAXblSrPLyCneX0%2BDQn9rRo5o11v64682n7QHrj3/8o37xi18oOTlZFy9elGVZatOmzS3tY9u2bS5/r3xEw8KFC3XixAm9%2Beab2rp1qwYOHKgdO3Zo//79ysjIkPR9wFu%2BfLni4uIUFhamJUuWqG3btoqJiZGvr68GDx6srKysqk80ZmVlaejQofL3969zfaGhodUuBzqdhSorazwvWDQ8TeX1VV5e0WTWejvoT%2B3oUc3ojxm2B6z58%2Bera9euatmypVq3bm18/126dNGKFSuUlZWljIwMdezYUcuXL1dYWJgkafjw4SosLFRaWpouXryobt26adWqVVUPOp09e7YWLVqkxMRElZaWqn///po1a5bxOgHTBi/92N0l1NkH6Q%2B6uwQAqFdeVuUDoGzy%2BOOPa8yYMRoyZIidh3U7p7PQ3SW4lcPhrYSsPe4uAw3E7QQsh8NbISFBKigo4t31TdCf2tGjmjXW/rRr19wtx7X9DFZERISmTZum1157TZ07d1azZs1c5hcsWGB3SQAAAEbZHrBOnz6t6OhoSar2KAMAAIDGwJaAtWDBAk2ZMkWBgYFas2aNHYcEAABwG1s%2BO//OO%2B9U%2B1UzY8eOVX5%2Bvh2HBwAAsJUtAetm99F//vnnun79uh2HBwAAsBVPfwQAADCMgAUAAGCYbQHLy8vLrkMBAAC4lW2PaZg/f77LM69KS0u1ePHiql%2BqXInnYAEAAE9nS8Dq0aNHtWdeRUVFqaCgQAUFBXaUAAAAYBtbAhbPvgIAAE0JN7kDAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAM89iAdfToUf36179Wz5499eCDD2r69Om6ePGiJOnQoUMaMWKEoqKiFB8frw0bNrh87ebNm5WQkKDu3bsrKSlJBw8erJorLy/XokWL1Lt3b0VFRSk1NVX5%2Bfm2rg0AAHg2jwxYJSUl%2Bpd/%2BRdFRUXpz3/%2Bs/7zP/9Tly5d0m9/%2B1tdvnxZ48eP1yOPPKIDBw4oMzNTCxYs0OHDhyVJ%2B/bt07x587Rw4UIdOHBAw4YNU2pqqoqLiyVJK1eu1Mcff6yNGzdqz5498vf318yZM925XAAA4GEc7i7gduTl5enee%2B9VWlqafHx85Ofnp%2BTkZE2fPl07duxQq1atlJKSIknq1auXEhMTtW7dOv3DP/yDNmzYoIcffljR0dGSpDFjxignJ0dbt27VY489pg0bNmjatGnq0KGDJCkjI0N9%2BvTRmTNn1KlTJ7etGWhMBi/92N0l3JIP0h90dwkAPIxHnsH6%2B7//e7322mvy8fGpGtu%2Bfbu6du2q48ePKzIy0mX78PBwHT16VJKUm5v7o/OFhYU6d%2B6cy3zbtm3VsmVLHTt2rB5XBAAAGhOPPIP1tyzL0tKlS7Vz506tXbtW77zzjgICAly28ff317Vr1yRJRUVFPzpfVFQkSQoMDKw2XzlXF/n5%2BXI6nS5jDkegQkND67yPxsbHxyOzPCBJcjga/uu38nuM77UfR49qRn/M8uiAdfXqVc2YMUNHjhzR2rVrdc899yggIECFhYUu25WUlCgoKEiSFBAQoJKSkmrzISEhVcGr8n6sm319XeTk5Cg7O9tlLC0tTZMnT67zPgA0HCEhdf/%2Bd7cWLQJq36iJo0c1oz9meGzAOn36tMaNG6e7775b7733nlq3bi1JioyM1Mcfu97fkZubq4iICElSRESEjh8/Xm0%2BLi5OLVu2VPv27V0uIzqdTl26dKnaZcWaJCcnKz4%2B3mXM4QhUQUHdz4I1NrwjgifzhO9dHx9vtWgRoCtXilVeXuHuchokelSzxtofd71B8siAdfnyZY0ePVoPPPCAMjMz5e39///nnZCQoMWLF%2Butt95SSkqKPvvsM23ZskWvvvqqJGn48OFKS0vT4MGDFR0drXXr1unChQtKSEiQJCUlJWnlypXq1q2bQkJC9OKLL6pnz576yU9%2BUuf6QkNDq10OdDoLVVbWeF6wQFPiSd%2B75eUVHlWvO9CjmtEfMzwyYG3atEl5eXn64IMPtG3bNpe5gwcP6o033lBmZqaWLVum1q1ba%2BbMmXrggQckff%2BpwtmzZ2vOnDk6f/68wsPDtXr1arVq1UrS95fyysrKlJKSoqKiIsXGxmrp0qW2rxEAAHguL8uyLHcX0RQ4nYW1b9SIORzeSsja4%2B4ygNviCY9pcDi8FRISpIKCIs4%2B/Ah6VLPG2p927Zq75bjcGAMAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABjmcHcBANDQDV76sbtLuCUfpD/o7hKAJo8zWAAAAIYRsAAAAAzjEqGH87RLFwAANAWcwQIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACG8ZgGAGhkPOnxLTx1Ho0VZ7AAAAAMI2DdxIULFzRhwgTFxMQoNjZWmZmZKisrc3dZAADAQxCwbiI9PV2BgYHas2eP3nvvPe3du1dvvfWWu8sCAAAegoD1A6dOndL%2B/fv1m9/8RgEBAerUqZMmTJigdevWubs0AADgIQhYP3D8%2BHG1atVK7du3rxrr0qWL8vLydOXKFTdWBgAAPAWfIvyBoqIiBQQEuIxV/v3atWtq0aJFrfvIz8%2BX0%2Bl0GXM4AhUaGmquUABoBBwO%2B97n%2B/h4u/y3KUjI2uPuEursw2kPubsFE3CAAAAJGElEQVQEowhYPxAYGKji4mKXscq/BwUF1WkfOTk5ys7OdhmbOHGiJk2aZKbIv/Fp5i%2BM77M%2B5OfnKycnR8nJyQTNm6A/taNHNaM/tcvPz9fbb7/WpHp0K/%2BP4DVkVtOJ8XUUERGhS5cu6bvvvqsaO3HihO666y41b968TvtITk7Wpk2bXP4kJyfXV8kewel0Kjs7u9qZPXyP/tSOHtWM/tSOHtWM/pjFGawf6Ny5s6Kjo/Xiiy9q7ty5Kigo0Kuvvqrhw4fXeR%2BhoaGkfwAAmjDOYN3EsmXLVFZWpv79%2B%2Bvxxx/XQw89pAkTJri7LAAA4CE4g3UTbdu21bJly9xdBgAA8FA%2Bc%2BbMmePuItA0BAUFqWfPnnX%2BsEBTQ39qR49qRn9qR49qRn/M8bIsy3J3EQAAAI0J92ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNg4Y4UFxcrOTlZmzZtchnft2%2BfkpOTFRMTo7i4OM2fP1/FxcVV83/4wx80aNAgRUVFadCgQVq3bl2djrdkyRLFx8cbXUN9s6NHBQUFev755/Xggw%2BqR48eGj16tL766qt6W5NJdvSnvLxcixYtUu/evRUVFaXU1FTl5%2BfX25pMu90eVdq%2Bfbv69%2B9f4zEuXryoqVOnKjY2VrGxsZowYYLy8vKMrqO%2B2NGfiooKZWdnq2/fvoqKitKIESN08OBBo%2BuoT3b06G954s9q4yzgNv31r3%2B1Hn30USsyMtLauHFj1fi5c%2Bes7t27Wzk5OVZZWZmVl5dnJSUlWXPnzrUsy7I%2B/PBDKyYmxjp48KBVUVFhff7551ZMTIy1bdu2Go/3ySefWF27drX69etXr%2Bsyya4epaamWuPHj7cuXrxoXb9%2B3Vq6dKnVu3dvq6ioyJZ13i67%2BrN8%2BXIrMTHRysvLswoLC6309HRr3LhxtqzxTt1ujyzLsm7cuGH927/9m3XffffV%2Bn0zZcoU65lnnrGKioqsoqIiKz093Ro1alS9rcsUu/qzfPlya9CgQdbXX39tlZWVWatWrbJ69uxpXb9%2Bvd7WZopdParkiT%2Br6wNnsHBb9u7dq9GjR%2BvRRx/V3Xff7TJ35swZxcfH6/HHH5ePj486dOigX/7ylzpw4IAk6fz58xo3bpy6d%2B8uLy8vRUVFKTY2tmr%2BZr777jvNnDlTI0eOrNd1mWRXjyzLkpeXl6ZMmaKQkBD5%2Bflp7Nix%2Bu6773Ty5Ek7lnpb7HwNbdiwQePGjVOHDh0UHBysjIwM7d69W2fOnKn3dd6JO%2BmRJD311FPat2%2Bfxo0bV%2BuxTpw4Icuyqv54e3srICDA%2BJpMsqs/5eXlevvttzVr1iyFhYXJx8dHY8eO1WuvvVYv6zLJzteQ5Jk/q%2BuLw90FoGEqKSnR%2BfPnbzrXrl073Xvvvdq5c6eaNWumN99802U%2BJiZGMTExVX%2BvqKjQhx9%2BqK5du0qSUlJSXLa/cOGCDhw4oBkzZtz0eBUVFZo2bZrGjRsnPz8/bd%2B%2B/U6WZkxD6ZGXl5dWrFjhMrZt2zYFBgYqLCzsttZmQkPpT2Fhoc6dO6fIyMiqsbZt26ply5Y6duyYOnXqdNtrvFP12SNJWrx4se66665ql4VuJjU1VRkZGYqOjpYk/fSnP9XatWtvZ1nGNJT%2BnDx5UleuXNGVK1eUlJSk//u//9N9992nGTNmyM/P7w5WeOcaSo8qv74h/qx2FwIWburQoUMaNWrUTedWrFihAQMG1Gk/paWlmjVrls6cOaOsrKxq806nU08//bTuv/9%2BDR069Kb7WLlypZo3b65f/epXdfomt0tD6tHf%2BuijjzR//nzNmTPHrWcgGkp/ioqKJEmBgYEu4/7%2B/lVz7lLfPbrrrrvqXEtFRYWSk5OVmpqq8vJyZWRkKD09vc73R9aHhtKfS5cuSZLWrFmj5cuXq02bNsrOztbYsWO1detWNW/evE77qQ8NpUdSw/1Z7S4ELNxUbGysjh07dkf7yM/P19SpU3X16lX94Q9/UPv27V3mv/jiC02ZMkUxMTFasGCBHI7qL8cDBw5o06ZNDfKbtaH0qJJlWVq5cqVWr16tF198UUOGDLmj2u5UQ%2BlPZcj84Y27JSUlCgoKuqP67pQdPaoLp9Op559/Xjt37lTLli0lSXPmzFFcXJyOHTume%2B65545qvF0NpT%2BVZ6kmTpyojh07SpKeeeYZrVu3Tp9//rn69u17RzXeiYbSo4b8s9pduAcL9eLw4cNKSkpShw4dtH79enXo0MFl/r333tOYMWM0evRovfTSSz96mv2Pf/yjLl68qP79%2BysmJka/%2B93vlJeXp5iYGH366ad2LKXemOqR9H14SE1N1caNG7Vu3Tq3hysTTPWnZcuWat%2B%2BvXJzc6vGnE6nLl265HLZ0BPV1qO6cjqdKi0t1Y0bN6rGKsOqr6%2BvkVrdwVR/wsLC5HA4XPrzt/ereTJTPWrMP6tvm/vur0dj0a9fP5dPppw%2BfdqKjo62li5detPtt23bZnXt2tXavXv3LR9r48aNHvnJlPru0dNPP2398pe/tAoKCozUa7f67s%2BSJUusoUOHWqdPn676FOE///M/G6ndLrfao79V2/fN9evXrf79%2B1upqalWYWGhVVhYaD3zzDPW8OHDrfLyciP117f67I9lWdb06dOtgQMHWmfOnLGuX79uLVy40HrooYes4uLiO67dLvXdozvZvjHiDBaMe/vtt1VYWKi33npLUVFRVX8efvhhSVJ2drbKy8s1efJkl/kXXnhB0vfvhKKioty5hHpnskdHjhzRzp07deLECfXr189le09952j6NZSWlqa%2BffsqJSVFffv21fXr17V06VK3rM2U2npUm7/tkZ%2Bfn15//XVJ0oABAzRw4EBZlqUVK1bI29sz/zdhsj%2BSNG/ePA0cOFCjRo1SbGysvvzyS73%2B%2Buvy9/evryXUO9M9gisvy/Lw85sAAAANjGe%2BNQEAAGjACFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwLD/BwZlZjZhMHh3AAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-5004649818657728902">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">-122.29</td>
        <td class="number">115</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.3</td>
        <td class="number">111</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.36200000000001</td>
        <td class="number">104</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.291</td>
        <td class="number">100</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.37200000000001</td>
        <td class="number">99</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.363</td>
        <td class="number">99</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.288</td>
        <td class="number">98</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.35700000000001</td>
        <td class="number">96</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.28399999999999</td>
        <td class="number">95</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.17200000000001</td>
        <td class="number">94</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (741)</td>
        <td class="number">20586</td>
        <td class="number">95.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-5004649818657728902">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">-122.51899999999999</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.515</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.514</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.512</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-122.51100000000001</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">-121.325</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-121.321</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-121.319</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-121.316</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">-121.315</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_price">price<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>3622</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>16.8%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>540300</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>78000</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>7700000</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram3496147860795532987">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAAs9JREFUeJzt2r9LanEYx/Gnq0PR3I%2BlxtoK40B/QCSBLYHkIv0LUTQEbUFDi7i6BA1BtAT9gNoiIk6Qrq5htEg0tqQ9d5PrTT7URc%2BJ7vsFDh6/R57lzfd7xD53dwPQ0a%2B4BwC%2Bs2TcA/wt2Lr48j33Ows9mARgBwEkAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgEEAgGEZNwDdEOwdfHle%2B53FnowCX4adhBAIBBAIBBA%2BBHPIP%2BC5xZ8Rp%2B7e9xDAN8VRyxAIBBAIBBAIBD0xMvLi83Pz9vd3d2n1mcyGUulUm2vyclJK5VKPZ5U%2B29/xULvlMtl29zctFqt9ul7zs/P294Xi0W7urqyfD7f7fG%2BhB0EXXV8fGwbGxu2trb24bPb21vLZrMWBIFlMhk7OTnp%2BB1hGNr%2B/r4Vi0UbHBzs9ciaA11Ur9f97e3N3d0nJiY8DEN3d69Wqz41NeWXl5feaDS8XC777OysX19ft93faDQ8nU57qVSKfPZO2EHQVUNDQ5ZMfjy5Hx4e2tzcnKXTaUskEjYzM2PLy8t2cHDQtu709NReX19tZWUlqpElnkEQiaenJwvD0IIgaF1rNps2Pj7etu7o6MhyuZz19/dHPWJHBIJIjI6O2tLSkm1vb7eu1et18z/%2ByPH8/GyVSsV2d3fjGLEjjliIRDabtbOzM7u5ubH393d7eHiwfD5ve3t7rTWVSsWGh4dtbGwsxknbsYMgEtPT01YoFKxQKNjq6qoNDAzY4uKira%2Bvt9Y8Pj7ayMhIjFN%2BxJ8VAYEjFiAQCCAQCCAQCCAQCCAQCCAQCCAQCCAQCCAQCCAQCCAQCCD8BsvSky6v/4fvAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives3496147860795532987,#minihistogram3496147860795532987"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives3496147860795532987">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles3496147860795532987"
                                                  aria-controls="quantiles3496147860795532987" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram3496147860795532987" aria-controls="histogram3496147860795532987"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common3496147860795532987" aria-controls="common3496147860795532987"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme3496147860795532987" aria-controls="extreme3496147860795532987"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles3496147860795532987">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>78000</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>210000</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>322000</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>450000</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>645000</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>1160000</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>7700000</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>7622000</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>323000</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>367370</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.67994</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>34.541</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>540300</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>234030</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>4.0234</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>11669000000</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>134960000000</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram3496147860795532987">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3XtYlWWi/vFbWCgLUCERO1y2dRAqy4JQUTFLjWksT6MUlTl20HZJMroVzVHTNE8TOea4NTPLSb12lGmlaTbNrnTM0BwrO2BgojaoIAflKKfn90c/1m6FBuqbrMP3c11ctp7nPTw3vdPcrvdd0MwYYwQAAADL%2BDT1AgAAADwNBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAItRsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAItRsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAIu5bMHKyMjQQw89pO7duysuLk6TJ09WQUGBJOmLL77Q3XffrejoaPXr109vvPGG074bN25UfHy8oqKiNGzYMO3bt88xV1NTo4ULF6pXr16Kjo7W448/rtzcXMd8fn6%2Bxo4dq65duyo2NlZz585VdXX1pQkNAAA8gksWrIqKCo0ePVrR0dH65z//qc2bN6uoqEh/%2BtOfdOrUKT366KMaOnSo9uzZo7lz52r%2B/Pn68ssvJUnp6emaM2eOFixYoD179mjw4MF6/PHHVV5eLklavny5du7cqTfffFM7duyQv7%2B/pk%2Bf7jj3%2BPHjFRAQoB07dmj9%2BvXatWuXVq9e3RTfBgAA4KZcsmDl5OTo2muvVVJSkpo3b66QkBAlJiZqz549ev/99xUcHKwRI0bIZrOpZ8%2BeGjRokNatWydJeuONN3TXXXcpJiZGfn5%2BevDBBxUSEqItW7Y45seMGaMrrrhCQUFBmjZtmrZv366jR4/q8OHD2r17t1JSUmS329W%2BfXuNHTvWcWwAAIDGsDX1As7mN7/5jV566SWnsW3btun6669XZmamIiMjneY6deqk9evXS5KysrI0fPjwevMZGRkqLi7W8ePHnfYPDQ1V69atdeDAAUlScHCw2rVr55gPDw9XTk6OTp8%2BrVatWjVq/bm5ucrLy3Maa9u2rcLCwhq1PwAAcG8uWbB%2ByhijxYsX68MPP9TatWv16quvym63O23j7%2B%2BvsrIySVJpaek550tLSyVJAQEB9ebr5n6%2Bb93rsrKyRhestLQ0LV261GksKSlJycnJjdofAAC4N5cuWCUlJZo6daq%2B/vprrV27Vtdcc43sdruKi4udtquoqFBgYKCkHwtRRUVFvfmQkBBHWap7Huvn%2Bxtj6s3Vva47fmMkJiaqX79%2BTmM2W4AKC0sbfYyf8/X1UatWdp0%2BXa6amtoLPo6r84acZPQM3pBR8o6cZPQM58oYEtL4//%2B2kssWrCNHjmjMmDG68sortX79el122WWSpMjISO3cudNp26ysLEVEREiSIiIilJmZWW%2B%2BT58%2Bat26tdq1a6esrCzHbcK8vDwVFRUpMjJStbW1Kioq0smTJxUaGipJOnjwoC6//HK1bNmy0WsPCwurdzswL69Y1dUXf1HX1NRachxX5w05yegZvCGj5B05yegZXCWjSz7kfurUKY0aNUo333yzVq1a5ShXkhQfH6%2BTJ09q9erVqqqq0qeffqpNmzY5nrtKSEjQpk2b9Omnn6qqqkqrV69Wfn6%2B4uPjJUnDhg3T8uXLdfToUZWUlGjevHnq3r27rr76anXo0EExMTGaN2%2BeSkpKdPToUS1btkwJCQlN8n0AAADuySXfwdqwYYNycnK0detWvffee05z%2B/bt08svv6y5c%2BdqyZIluuyyyzR9%2BnT16NFDktSzZ0/NnDlTs2bN0okTJ9SpUyetXLlSwcHBkn58Fqq6ulojRoxQaWmpYmNjtXjxYsfxlyxZotmzZ6t///7y8fHR0KFDNXbs2EsXHgAAuL1mxhjT1IvwBnl5xQ1v9AtsNh%2BFhASqsLDUJd76/LV4Q04yegZvyCh5R04yeoZzZWzbtvGP%2BFjJJW8RAgAAuDMKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMZf8VTlovAGLdza8kYvYOj6uqZcAAMAlwTtYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFbE29gAvxzjvvaObMmU5jVVVVkqSvvvpKkjR69Gilp6fLZvu/iM8//7z69Omjmpoapaam6u2331Z5ebl69Oihp59%2BWmFhYZKk/Px8zZgxQ7t375avr68GDx6sKVOmOB0LAADgXNzyHazBgwdr3759jq/33ntPwcHBmjt3rmObr776SqtWrXLark%2BfPpKk5cuXa%2BfOnXrzzTe1Y8cO%2Bfv7a/r06Y59x48fr4CAAO3YsUPr16/Xrl27tHr16ksdEwAAuCm3LFg/ZYxRSkqKbrvtNg0ZMkSSdPToUZ06dUqdO3c%2B6z5vvPGGxowZoyuuuEJBQUGaNm2atm/frqNHj%2Brw4cPavXu3UlJSZLfb1b59e40dO1br1q27lLEAAIAbc/t7Xm%2B//baysrK0bNkyx9j%2B/fsVGBioCRMmaP/%2B/QoNDdWDDz6ohIQEFRcX6/jx44qMjHRsHxoaqtatW%2BvAgQOSpODgYLVr184xHx4erpycHJ0%2BfVqtWrVqcE25ubnKy8tzGrPZAhy3IC%2BEr6%2BP05/uyGZreO2ekLMhZPQM3pBR8o6cZPQMrpbRrQtWbW2tli9frscee0xBQUGO8crKSkVFRWnChAmKiIhQenq6xo0bp8DAQEVHR0uSAgICnI7l7%2B%2Bv0tJSSZLdbneaq3tdVlbWqIKVlpampUuXOo0lJSUpOTn5/EP%2BTKtW9oY3clEhIYGN3tadczYWGT2DN2SUvCMnGT2Dq2R064KVnp6u3NxcJSQkOI0PHTpUQ4cOdbzu3bu3hg4dqq1bt6pXr16SpPLycqd9KioqFBgYKGNMvbm614GBjSsIiYmJ6tevn9OYzRagwsLSxgU7C19fH7VqZdfp0%2BWqqam94OM0pcbk94ScDSGjZ/CGjJJ35CSjZzhXxvP5y72V3Lpgbdu2TfHx8fXejVq/fr0CAwM1YMAAx1hlZaVatGih1q1bq127dsrKynLcJszLy1NRUZEiIyNVW1uroqIinTx5UqGhoZKkgwcP6vLLL1fLli0bta6wsLB6twPz8opVXX3xF3VNTa0lx2kK57Nud87ZWGT0DN6QUfKOnGT0DK6S0TVuVF6gvXv3qlu3bvXGS0pKNGfOHH3zzTeqra3VRx99pM2bNysxMVGSNGzYMC1fvlxHjx5VSUmJ5s2bp%2B7du%2Bvqq69Whw4dFBMTo3nz5qmkpERHjx7VsmXL6r1LBgAAcC5u/Q7WDz/8cNYHx0eNGqWysjI98cQTys/PV/v27bVw4UJ17dpV0o/PQ1VXV2vEiBEqLS1VbGysFi9e7Nh/yZIlmj17tvr37y8fHx8NHTpUY8eOvWS5AACAe2tmjDFNvQhvkJdXfFH722w%2BCgkJVGFhqdNbnwMW77zYpV0yW8fHNbjNuXJ6EjJ6Bm/IKHlHTjJ6hnNlbNu2cY/3WM2tbxECAAC4IgoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMVcvmAVFBQoPj5e6enpjrGZM2fqhhtuUHR0tOMrLS3NMb9y5Ur16dNHUVFRGjlypL7//nvHXFlZmaZOnarY2FjFxMRo8uTJKi0tdcwfOnRIo0aNUnR0tHr37q0XXnjh0gQFAAAew6UL1t69e5WYmKgjR444je/fv19z5szRvn37HF%2BJiYmSpI0bN2rNmjVatWqV0tPTdf311ys5OVnGGEnSnDlzdOzYMW3btk3vv/%2B%2Bjh07ptTUVElSVVWVHnvsMXXp0kXp6el68cUXtW7dOm3duvXSBgcAAG7NZQvWxo0bNWnSJE2YMMFpvLKyUt99951uuOGGs%2B73%2Buuv6/7771dERIRatGihiRMnKicnR%2Bnp6SovL9emTZuUnJys4OBgtWnTRpMmTdKGDRtUXl6uPXv2KDc3V8nJyWrevLk6d%2B6skSNHat26dZciMgAA8BC2pl7AufTu3VuDBg2SzWZzKlkZGRmqrq7WkiVLtHfvXrVs2VLDhw/X6NGj5ePjo6ysLI0ZM8axvZ%2Bfnzp06KCMjAwFBwerqqpKkZGRjvnw8HBVVFQoOztbmZmZ6tixo5o3b%2B6Y79Spk1588cXzWntubq7y8vKcxmy2AIWFhZ3vt8HB19fH6U93ZLM1vHZPyNkQMnoGb8goeUdOMnoGV8vosgWrbdu2Zx0vLi5W9%2B7dNXLkSC1atEjffvutkpKS5OPjo9GjR6u0tFR2u91pH39/f5WVlamkpESSFBAQ4Jir27a0tPSs%2B9rtdpWVlZ3X2tPS0rR06VKnsaSkJCUnJ5/Xcc6mVSt7wxu5qJCQwEZv6845G4uMnsEbMkrekZOMnsFVMrpswTqXuLg4xcXFOV7feOONGjVqlLZs2aLRo0fLbreroqLCaZ%2BKigoFBgY6ilV5ebkCAwMd/yxJQUFBCggIcLyu89NtGysxMVH9%2BvVzGrPZAlRYWHqOPRrm6%2BujVq3sOn26XDU1tRd8nKbUmPyekLMhZPQM3pBR8o6cZPQM58p4Pn%2B5t5LbFawPPvhAJ0%2Be1L333usYq6yslL%2B/vyQpIiJCmZmZ6tu3r6QfH1zPzs5WZGSkOnbsKD8/P2VlZemmm26SJB08eNBxGzE/P1/Z2dmqrq6WzfbjtyYrK0sRERHntcawsLB6twPz8opVXX3xF3VNTa0lx2kK57Nud87ZWGT0DN6QUfKOnGT0DK6S0TVuVJ4HY4zmz5%2BvXbt2yRijffv26dVXX3V8inD48OFau3atMjIydObMGT333HMKDQ1V165dZbfbNWDAAKWmpqqgoEAFBQVKTU3VwIED5e/vr9jYWIWEhOi5557TmTNnlJGRoTVr1ighIaGJUwMAAHfidu9gxcfHa%2BrUqZo1a5ZOnDih0NBQjRs3TkOGDJEkJSQkqLi4WElJSSooKFCXLl20YsUK%2Bfn5SfrxZ2gtXLhQgwYNUlVVlfr3768ZM2ZIkmw2m15%2B%2BWXNnj1bcXFxCggI0MiRIzVs2LAmywsAANxPM1P3A6Lwq8rLK76o/W02H4WEBKqwsNTprc8Bi3de7NIuma3j4xrc5lw5PQkZPYM3ZJS8IycZPcO5MrZt27JJ1uN2twgBAABcHQULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAItRsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIULAAAAItRsAAAACxGwQIAALAYBQsAAMBiFCwAAACLUbAAAAAsRsECAACwGAULAADAYhQsAAAAi1GwAAAALEbBAgAAsBgFCwAAwGIuX7AKCgoUHx%2Bv9PR0x9i2bds0ZMgQ3XzzzerXr5%2BWLl2q2tpax/yAAQN00003KTo62vF18OBBSVJZWZmmTp2q2NhYxcTEaPLkySotLXXse%2BjQIY0aNUrR0dHq3bu3XnjhhUsXFgAAeASXLlh79%2B5VYmKijhw54hj76quvNHnyZI0fP16fffaZVq5cqQ0bNmj16tWSpJKSEh06dEhbtmzRvn37HF/h4eGSpDlz5ujYsWPatm2b3n//fR07dkypqamSpKqqKj322GPq0qWL0tPT9eKLL2rdunXaunXrJc8OAADcl8sWrI0bN2rSpEmaMGGC0/i///1v3Xvvverbt698fHwUHh6u%2BPh47dmzR9KPBSw4OFhXXXVVvWOWl5dr06ZNSk5OVnBwsNq0aaNJkyZpw4YNKi8v1549e5Sbm6vk5GQ1b95cnTt31siRI7Vu3bpLkhkAAHgGW1Mv4Fx69%2B6tQYMGyWazOZWsO%2B64Q3fccYfjdUVFhT766CMNGjRIkrR//37Z7XY98MADyszM1FVXXaVx48apb9%2B%2BOnz4sKqqqhQZGenYPzw8XBUVFcrOzlZmZqY6duyo5s2bO%2BY7deqkF1988bzWnpubq7y8PKcxmy1AYWFh53Wcn/L19XH60x3ZbA2v3RNyNoSMnsEbMkrekZOMnsHVMrpswWrbtm2D25SUlOiPf/yj/P399eCDD0qSmjVrpi5duui//uu/dOWVV%2Bq9997TuHHjtHbtWlVXV0uSAgICHMew2%2B2SpNLSUpWWljpe/3S%2BrKzsvNaelpampUuXOo0lJSUpOTn5vI5zNq1a2RveyEWFhAQ2elt3ztlYZPQM3pBR8o6cZPQMrpLRZQtWQ77//nslJyerTZs2evXVVxUUFCRJGj16tNN2gwcP1ubNm7Vt2zbHu1zl5eUKDAx0/LMkBQUFKSAgwPG6zk%2B3bazExET169fPacxmC1BhYek59miYr6%2BPWrWy6/TpctXU1Da8gwtqTH5PyNkQMnoGb8goeUdOMnqGc2U8n7/cW8ktC9bHH3%2Bs//qv/9I999yjiRMnymb7vxirVq1S586d1bNnT8dYZWWlWrRooY4dO8rPz09ZWVm66aabJEkHDx6Un5%2BfOnTooPz8fGVnZ6u6utpxzKysLEVERJzX%2BsLCwurdDszLK1Z19cVf1DU1tZYcpymcz7rdOWdjkdEzeENGyTtyktEzuEpG17hReR4%2B//xzJSUlaerUqZoyZYpTuZKkY8eO6emnn9bRo0dVXV2t9evXa9%2B%2Bffr9738vu92uAQMGKDU1VQUFBSooKFBqaqoGDhwof39/xcbGKiQkRM8995zOnDmjjIwMrVmzRgkJCU2UFgAAuCO3ewfrhRdeUHV1tebOnau5c%2Bc6xmNiYvTSSy9p8uTJ8vHx0f3336/i4mLHQ%2Br/8R//IUmaOXOmFi5cqEGDBqmqqkr9%2B/fXjBkzJEk2m00vv/yyZs%2Berbi4OAUEBGjkyJEaNmxYk2QFAADuqZkxxjT1IrxBXl7xRe1vs/koJCRQhYWlTm99Dli882KXdslsHR/X4DbnyulJyOgZvCGj5B05yegZzpWxbduWTbIey28R1tTUWH1IAAAAt2J5werTp4/%2B/Oc/Kysry%2BpDAwAAuAXLC9YTTzyhf/3rXxo4cKDuvvtuvfbaayouvrjbYwAAAO7E8oJ133336bXXXtN7772nXr16aeXKlerdu7cmTpyoTz75xOrTAQAAuJxf7cc0dOjQQRMmTNB7772npKQk/eMf/9Ajjzyifv366ZVXXuFZLQAA4LF%2BtR/T8MUXX%2Bitt97Sli1bVFlZqfj4eA0bNkwnTpzQ888/r/3792vRokW/1ukBAACajOUFa9myZXr77bd1%2BPBhdenSRRMmTNDAgQMdv8pGknx9ffXUU09ZfWoAAACXYHnBWrt2rQYPHqyEhAR16tTprNuEh4dr0qRJVp8aAADAJVhesLZv366SkhIVFRU5xrZs2aKePXsqJCREktS5c2d17tzZ6lMDAAC4BMsfcv/mm290xx13KC0tzTH27LPPatCgQfruu%2B%2BsPh0AAIDLsbxg/fnPf9Zvf/tbTZgwwTH2wQcfqE%2BfPlqwYIHVpwMAAHA5lhesr7/%2BWo8%2B%2BqiaN2/uGPP19dWjjz6qzz//3OrTAQAAuBzLC1ZQUJCOHDlSb/z48ePy9/e3%2BnQAAAAux/KCdccdd2jWrFn65JNPVFJSotLSUn366aeaPXu24uPjrT4dAACAy7H8U4QTJ07U0aNH9fDDD6tZs2aO8fj4eE2ePNnq0wEAALgcywuW3W7XihUrdOjQIR04cEB%2Bfn4KDw9Xhw4drD4VAACAS/rVflVOx44d1bFjx1/r8AAAAC7L8oJ16NAhzZ49W3v37lVVVVW9%2BW%2B//dbqUwIAALgUywvWrFmzlJOTo0mTJqlly5ZWHx4AAMDlWV6w9u3bp7/97W%2BKjo62%2BtAAAABuwfIf0xASEqLAwECrDwsAAOA2LC9YI0eO1KJFi1RcXGz1oQEAANyC5bcIP/74Y33%2B%2BeeKjY1VmzZtnH5ljiT94x//sPqUAAAALsXyghUbG6vY2FirDwsAAOA2LC9YTzzxhNWHBAAAcCuWP4MlSRkZGZo6daruvfdenThxQuvWrVN6evqvcSoAAACXY3nB%2Buqrr3T33Xfrhx9%2B0FdffaXKykp9%2B%2B23evjhh/Xhhx9afToAAACXY3nBSk1N1cMPP6w1a9bIz89PkvTMM8/oD3/4g5YuXWr16QAAAFzOr/IO1tChQ%2BuN33ffffr%2B%2B%2B%2BtPh0AAIDLsbxg%2Bfn5qaSkpN54Tk6O7HZ7o4%2BTn5%2BvsWPHqmvXroqNjdXcuXNVXV3tmP/iiy909913Kzo6Wv369dMbb7zhtP/GjRsVHx%2BvqKgoDRs2TPv27XPM1dTUaOHCherVq5eio6P1%2BOOPKzc39wLSAgAA1Gd5wbr99tv13HPPqbCw0DF28OBBzZ07V7fddlujjzN%2B/HgFBARox44dWr9%2BvXbt2qXVq1dLkk6dOqVHH31UQ4cO1Z49ezR37lzNnz9fX375pSQpPT1dc%2BbM0YIFC7Rnzx4NHjxYjz/%2BuMrLyyVJy5cv186dO/Xmm29qx44d8vf31/Tp0y37HgAAAO9mecGaMmWKKioq1KtXL5WXl2vYsGEaOHCgbDabJk%2Be3KhjHD58WLt371ZKSorsdrvat2%2BvsWPHat26dZKk999/X8HBwRoxYoRsNpt69uypQYMGOebfeOMN3XXXXYqJiZGfn58efPBBhYSEaMuWLY75MWPG6IorrlBQUJCmTZum7du36%2BjRo1Z/OwAAgBey/OdgBQUF6bXXXtOuXbv0zTffqLa2VpGRkbrlllvk49O4PpeZmang4GC1a9fOMRYeHq6cnBydPn1amZmZioyMdNqnU6dOWr9%2BvSQpKytLw4cPrzefkZGh4uJiHT9%2B3Gn/0NBQtW7dWgcOHFD79u0vNDoAAICkX6Fg1enZs6d69ux5QfuWlpbWe16r7nVZWdlZ5/39/VVWVnbO/evmS0tLJUkBAQH15uvmLlZubq7y8vKcxmy2AIWFhV3wMX19fZz%2BdEc2W8Nr94ScDSGjZ/CGjJJ35CSjZ3C1jJYXrH79%2BqlZs2bnnG/M7yIMCAhwPC9Vp%2B51YGCg7HZ7vV8mXVFRocDAQEk/lrGKiop68yEhIY7i9fPj/3T/i5WWllbvR1IkJSUpOTn5oo/dqlXjPyjgakJCGv/9deecjUVGz%2BANGSXvyElGz%2BAqGS0vWL///e%2BdClZVVZUOHz6s7du3a/z48Y06RkREhIqKinTy5EmFhoZK%2BvFB%2Bcsvv1wtW7ZUZGSkdu7c6bRPVlaWIiIiHPtnZmbWm%2B/Tp49at26tdu3aKSsry3GbMC8vT0VFRfVuO16oxMRE9evXz2nMZgtQYeGFv0Pm6%2BujVq3sOn26XDU1tRe7xCbRmPyekLMhZPQM3pBR8o6cZPQM58p4Pn%2B5t5LlBWvcuHFnHV%2B7dq327t2rP/zhDw0eo0OHDoqJidG8efM0e/ZsFRYWatmyZUpISJAkxcfH69lnn9Xq1as1YsQI7d27V5s2bdKyZcskSQkJCUpKStKAAQMUExOjdevWKT8/X/Hx8ZKkYcOGafny5erSpYtCQkI0b948de/eXVdffbUl34OwsLB6twPz8opVXX3xF3VNTa0lx2kK57Nud87ZWGT0DN6QUfKOnGT0DK6S8Vd7Buvn%2Bvbtq0WLFjV6%2ByVLlmj27Nnq37%2B/fHx8NHToUI0dO1aSFBISopdffllz587VkiVLdNlll2n69Onq0aOHpB%2Bf/5o5c6ZmzZqlEydOqFOnTlq5cqWCg4Ml/Xi7rrq6WiNGjFBpaaliY2O1ePFi60MDAACvdMkK1u7du9WiRYtGbx8aGqolS5acc75Lly567bXXzjk/ZMgQDRky5Kxzfn5%2BmjRpkiZNmtTo9QAAADSW5QXr57cAjTEqKSnRgQMHGnV7EAAAwN1ZXrCuvPLKep8i9PPz06hRozRo0CCrTwcAAOByLC9YCxYssPqQAAAAbsXygrVnz55Gb9utWzerTw8AANDkLC9YDz74oIwxjq86dbcN68aaNWumb7/91urTAwAANDnLC9Zf//pXzZ8/X1OmTFGPHj3k5%2BenL774QrNmzdL999%2Bvvn37Wn1KAAAAl2L5L%2BxZuHChZs6cqdtvv11BQUFq0aKFunfvrtmzZ%2Bvll1/WVVdd5fgCAADwRJYXrNzcXF1xxRX1xoOCglRYWGj16QAAAFyO5QUrKipKixYtUklJiWOsqKhIzz77rHr27Gn16QAAAFyO5c9gTZ8%2BXaNGjVKfPn3UoUMHSdKhQ4fUtm1bvfrqq1afDgAAwOVYXrDCw8O1ZcsWbdq0SQcPHpQk3X///brrrrtkt9utPh0AAIDL%2BVV%2BF2GrVq10991364cfflD79u0l/fjT3AEAALyB5c9gGWOUmpqqbt26aeDAgTp%2B/LimTJmiqVOnqqqqyurTAQAAuBzLC9aaNWv09ttva%2BbMmWrevLkk6fbbb9f//u//6vnnn7f6dAAAAC7H8oKVlpamp556SsOGDXP89PY777xTc%2BfO1bvvvmv16QAAAFyO5QXrhx9%2B0HXXXVdv/JprrtHJkyetPh3xDPoEAAAe90lEQVQAAIDLsbxgXXXVVfryyy/rjX/88ceOB94BAAA8meWfInzkkUf09NNP68SJEzLGaNeuXXrttde0Zs0aTZ061erTAQAAuBzLC9bw4cNVXV2t5cuXq6KiQk899ZTatGmjCRMm6L777rP6dAAAAC7H8oL1zjvv6He/%2B50SExNVUFAgY4zatGlj9WkAAABcluXPYD3zzDOOh9kvu%2BwyyhUAAPA6lhesDh066MCBA1YfFgAAwG1YfoswIiJCkyZN0ksvvaQOHTqoRYsWTvPz58%2B3%2BpQAAAAuxfKCdeTIEcXExEiS8vLyrD48AACAy7OkYM2fP19//OMfFRAQoDVr1lhxSAAAALdlyTNYr776qsrLy53GHnnkEeXm5lpxeAAAALdiScEyxtQb%2B9e//qUzZ85YcXgAAAC3YvmnCAEAALwdBQsAAMBilhWsZs2aWXUoAAAAt2bZj2l45plnnH7mVVVVlZ599lkFBgY6bcfPwQIAAJ7OkoLVrVu3ej/zKjo6WoWFhSosLLTiFE7eeecdzZw502msqqpKkvTVV19p9OjRSk9Pl832f/Gef/559enTRzU1NUpNTdXbb7%2Bt8vJy9ejRQ08//bTCwsIkSfn5%2BZoxY4Z2794tX19fDR48WFOmTHE6FgAAwC%2BxpDVc6p99NXjwYA0ePNjx%2BsSJExo%2BfLhSUlIk/ViyVq1ape7du9fbd/ny5dq5c6fefPNNtWzZUjNmzND06dP14osvSpLGjx%2Bvdu3aaceOHTp58qQef/xxrV69WqNHj7404QAAgNtz%2B4fcjTFKSUnRbbfdpiFDhujo0aM6deqUOnfufNbt33jjDY0ZM0ZXXHGFgoKCNG3aNG3fvl1Hjx7V4cOHtXv3bqWkpMhut6t9%2B/YaO3as1q1bd4lTAQAAd%2Bb2973efvttZWVladmyZZKk/fv3KzAwUBMmTND%2B/fsVGhqqBx98UAkJCSouLtbx48cVGRnp2D80NFStW7d2/ILq4OBgtWvXzjEfHh6unJwcnT59Wq1atWrUmnJzc%2BvdMrXZAhy3IS%2BEr6%2BP05/uyGZreO2ekLMhZPQM3pBR8o6cZPQMrpbRrQtWbW2tli9frscee0xBQUGSpMrKSkVFRWnChAmKiIhQenq6xo0bp8DAQEVHR0uSAgICnI7j7%2B%2Bv0tJSSZLdbneaq3tdVlbW6IKVlpampUuXOo0lJSUpOTn5/EP%2BTKtW9oY3clEhIYENb/T/uXPOxiKjZ/CGjJJ35CSjZ3CVjG5dsNLT05Wbm6uEhATH2NChQzV06FDH6969e2vo0KHaunWrevXqJUn1fq1PRUWFAgMDZYypN1f3%2BuefhvwliYmJ6tevn9OYzRagwsLSRh/j53x9fdSqlV2nT5erpqb2go/TlBqT3xNyNoSMnsEbMkrekZOMnuFcGc/nL/dWcuuCtW3bNsXHxzu9I7V%2B/XoFBgZqwIABjrHKykq1aNFCrVu3Vrt27ZSVleW4TZiXl6eioiJFRkaqtrZWRUVFOnnypEJDQyVJBw8e1OWXX66WLVs2el1hYWH1bgfm5RWruvriL%2BqamlpLjtMUzmfd7pyzscjoGbwho%2BQdOcnoGVwlo2vcqLxAe/fuVbdu3ZzGSkpKNGfOHH3zzTeqra3VRx99pM2bNysxMVGSNGzYMC1fvlxHjx5VSUmJ5s2bp%2B7du%2Bvqq69Whw4dFBMTo3nz5qmkpERHjx7VsmXLnN4hAwAAaIhbv4P1ww8/1HunaNSoUSorK9MTTzyh/Px8tW/fXgsXLlTXrl0l/fgsVHV1tUaMGKHS0lLFxsZq8eLFjv2XLFmi2bNnq3///vLx8dHQoUM1duzYS5oLAAC4t2bGGNPUi/AGeXnFF7W/zeajkJBAFRaWOr31OWDxzotd2iWzdXxcg9ucK6cnIaNn8IaMknfkJKNnOFfGtm0b/4iPldz6FiEAAIAromABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWMxtC9aWLVvUuXNnRUdHO75SUlIkSR9//LEGDRqkqKgoDRgwQB9%2B%2BKHTvitXrlSfPn0UFRWlkSNH6vvvv3fMlZWVaerUqYqNjVVMTIwmT56s0tLSS5oNAAC4N7ctWPv379eQIUO0b98%2Bx9ezzz6r7OxsjRs3Tn/84x/12Wefady4cRo/frxOnDghSdq4caPWrFmjVatWKT09Xddff72Sk5NljJEkzZkzR8eOHdO2bdv0/vvv69ixY0pNTW3KqAAAwM24dcG64YYb6o1v3LhRXbt21e233y6bzaY777xT3bp1U1pamiTp9ddf1/3336%2BIiAi1aNFCEydOVE5OjtLT01VeXq5NmzYpOTlZwcHBatOmjSZNmqQNGzaovLz8UkcEAABuytbUC7gQtbW1%2Bvrrr2W32/XSSy%2BppqZGt956qyZNmqSsrCxFRkY6bd%2BpUydlZGRIkrKysjRmzBjHnJ%2Bfnzp06KCMjAwFBwerqqrKaf/w8HBVVFQoOztb1113XaPWl5ubq7y8PKcxmy1AYWFhFxpZvr4%2BTn%2B6I5ut4bV7Qs6GkNEzeENGyTtyktEzuFpGtyxYBQUF6ty5s%2B644w4tWbJEhYWFmjJlilJSUlRZWSm73e60vb%2B/v8rKyiRJpaWl55wvKSmRJAUEBDjm6rY9n%2Bew0tLStHTpUqexpKQkJScnNz7kObRqZW94IxcVEhLY6G3dOWdjkdEzeENGyTtyktEzuEpGtyxYoaGhWrduneO13W5XSkqK7rnnHsXGxqqiosJp%2B4qKCgUGBjq2Pdd8XbEqLy93bF93azAoKKjR60tMTFS/fv2cxmy2ABUWXvjD8r6%2BPmrVyq7Tp8tVU1N7wcdpSo3J7wk5G0JGz%2BANGSXvyElGz3CujOfzl3sruWXBysjI0ObNmzVx4kQ1a9ZMklRZWSkfHx/deOON%2Bvbbb522z8rKcjyvFRERoczMTPXt21eSVFVVpezsbEVGRqpjx47y8/NTVlaWbrrpJknSwYMHHbcRGyssLKze7cC8vGJVV1/8RV1TU2vJcZrC%2BazbnXM2Fhk9gzdklLwjJxk9g6tkdI0blecpODhY69at00svvaTq6mrl5OTo2Wef1e9//3sNHTpUu3fv1pYtW1RdXa0tW7Zo9%2B7dGjJkiCRp%2BPDhWrt2rTIyMnTmzBk999xzCg0NVdeuXWW32zVgwAClpqaqoKBABQUFSk1N1cCBA%2BXv79/EqQEAgLtwy3ewLr/8cq1YsUKLFi3S8uXL1aJFC911111KSUlRixYt9N///d9KTU3VtGnTdNVVV%2Bmvf/2rOnbsKElKSEhQcXGxkpKSVFBQoC5dumjFihXy8/OTJM2cOVMLFy7UoEGDVFVVpf79%2B2vGjBlNGRcAALiZZqbuB0DhV5WXV3xR%2B9tsPgoJCVRhYanTW58DFu%2B82KVdMlvHxzW4zblyehIyegZvyCh5R04yeoZzZWzbtmWTrMctbxECAAC4MgoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMXctmBlZGTooYceUvfu3RUXF6fJkyeroKBAkjRz5kzdcMMNio6OdnylpaU59l25cqX69OmjqKgojRw5Ut9//71jrqysTFOnTlVsbKxiYmI0efJklZaWXvJ8AADAfbllwaqoqNDo0aMVHR2tf/7zn9q8ebOKior0pz/9SZK0f/9%2BzZkzR/v27XN8JSYmSpI2btyoNWvWaNWqVUpPT9f111%2Bv5ORkGWMkSXPmzNGxY8e0bds2vf/%2B%2Bzp27JhSU1ObLCsAAHA/blmwcnJydO211yopKUnNmzdXSEiIEhMTtWfPHlVWVuq7777TDTfccNZ9X3/9dd1///2KiIhQixYtNHHiROXk5Cg9PV3l5eXatGmTkpOTFRwcrDZt2mjSpEnasGGDysvLL3FKAADgrmxNvYAL8Zvf/EYvvfSS09i2bdt0/fXXKyMjQ9XV1VqyZIn27t2rli1bavjw4Ro9erR8fHyUlZWlMWPGOPbz8/NThw4dlJGRoeDgYFVVVSkyMtIxHx4eroqKCmVnZ%2Bu6665r1Ppyc3OVl5fnNGazBSgsLOyCM/v6%2Bjj96Y5stobX7gk5G0JGz%2BANGSXvyElGz%2BBqGd2yYP2UMUaLFy/Whx9%2BqLVr1%2BrkyZPq3r27Ro4cqUWLFunbb79VUlKSfHx8NHr0aJWWlsputzsdw9/fX2VlZSopKZEkBQQEOObqtj2f57DS0tK0dOlSp7GkpCQlJydfaEyHVq3sDW/kokJCAhu9rTvnbCwyegZvyCh5R04yegZXyejWBaukpERTp07V119/rbVr1%2Bqaa67RNddco7i4OMc2N954o0aNGqUtW7Zo9OjRstvtqqiocDpORUWFAgMDHcWqvLxcgYGBjn%2BWpKCgoEavKzExUf369XMas9kCVFh44Q/L%2B/r6qFUru06fLldNTe0FH6cpNSa/J%2BRsCBk9gzdklLwjJxk9w7kyns9f7q3ktgXryJEjGjNmjK688kqtX79el112mSTpgw8%2B0MmTJ3Xvvfc6tq2srJS/v78kKSIiQpmZmerbt68kqaqqStnZ2YqMjFTHjh3l5%2BenrKws3XTTTZKkgwcPOm4jNlZYWFi924F5ecWqrr74i7qmptaS4zSF81m3O%2BdsLDJ6Bm/IKHlHTjJ6BlfJ6Bo3Ks/TqVOnNGrUKN18881atWqVo1xJP94ynD9/vnbt2iVjjPbt26dXX33V8SnC4cOHa%2B3atcrIyNCZM2f03HPPKTQ0VF27dpXdbteAAQOUmpqqgoICFRQUKDU1VQMHDnQUNAAAgIa45TtYGzZsUE5OjrZu3ar33nvPaW7fvn2aOnWqZs2apRMnTig0NFTjxo3TkCFDJEkJCQkqLi5WUlKSCgoK1KVLF61YsUJ%2Bfn6SfvwZWgsXLtSgQYNUVVWl/v37a8aMGZc8IwAAcF/NTN0PgMKvKi%2Bv%2BKL2t9l8FBISqMLCUqe3Pgcs3nmxS7tkto6Pa3Cbc%2BX0JGT0DN6QUfKOnGT0DOfK2LZty6ZZT5OcFV7Jncqg1LhCCADA2bjlM1gAAACujIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFrM19QJcUX5%2BvmbMmKHdu3fL19dXgwcP1pQpU2Sz8e3yJgMW72zqJZyXrePjmnoJAID/j3ewzmL8%2BPEKCAjQjh07tH79eu3atUurV69u6mUBAAA3QcH6mcOHD2v37t1KSUmR3W5X%2B/btNXbsWK1bt66plwYAANwEBetnMjMzFRwcrHbt2jnGwsPDlZOTo9OnTzfhygAAgLvgoaKfKS0tld1udxqre11WVqZWrVo1eIzc3Fzl5eU5jdlsAQoLC7vgdfn6%2Bjj9CfycOz0z9vdJtzT1Ei6at/xv0htyktEzuFpGCtbPBAQEqLy83Gms7nVgYGCjjpGWlqalS5c6jT3xxBMaN27cBa8rNzdXf/vbS0pMTHQqap/N/d0FH9MV5ebmKi0trV5OT0JGz3Cu/016Gm/ISUbP4GoZXaPmuZCIiAgVFRXp5MmTjrGDBw/q8ssvV8uWLRt1jMTERG3YsMHpKzEx8aLWlZeXp6VLl9Z7Z8zTeENOMnoGb8goeUdOMnoGV8vIO1g/06FDB8XExGjevHmaPXu2CgsLtWzZMiUkJDT6GGFhYS7RngEAQNPgHayzWLJkiaqrq9W/f3/dc889uuWWWzR27NimXhYAAHATvIN1FqGhoVqyZElTLwMAALgp31mzZs1q6kWgcQIDA9W9e/dGP2zvrrwhJxk9gzdklLwjJxk9gytlbGaMMU29CAAAAE/CM1gAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWG4gPz9fY8eOVdeuXRUbG6u5c%2Bequrq6ydZTUFCg%2BPh4paenO8a%2B%2BOIL3X333YqOjla/fv30xhtvOO2zceNGxcfHKyoqSsOGDdO%2BffscczU1NVq4cKF69eql6OhoPf7448rNzXXMN5T/Ys79cxkZGXrooYfUvXt3xcXFafLkySooKPCojJK0a9cu3X333br55psVFxenOXPmqKKiwuNy1tTUaOTIkXryyScdYx9//LEGDRqkqKgoDRgwQB9%2B%2BKHTPitXrlSfPn0UFRWlkSNH6vvvv3fMlZWVaerUqYqNjVVMTIwmT56s0tJSx/yhQ4c0atQoRUdHq3fv3nrhhRecjn0x5z6bLVu2qHPnzoqOjnZ8paSk/Oo5L6WioiJNnjxZsbGx6tatm8aOHeu4ptzlWu3Vq5e6devW4Hmio6N1/fXXKyoqyvHv84YbbtANN9wgyf2v3a%2B//lojRoxQ165d1bt3bz3zzDOqrKx0%2B2znvBYMXN4DDzxgJk6caMrKysyRI0fMXXfdZVauXNkka/nss8/M7bffbiIjI82nn35qjDGmqKjIdO/e3axdu9ZUVVWZTz75xERHR5svvvjCGGPMp59%2BaqKjo81nn31mKisrzSuvvGJiY2NNWVmZMcaYv/71r2bQoEEmJyfHFBcXm/Hjx5sxY8Y4zvlL%2BS/23D9VXl5u4uLizPPPP2/OnDljCgoKzJgxY8x//ud/ekxGY4zJz883Xbp0MW%2B%2B%2BaapqakxJ06cMAMHDjTPP/%2B8R%2BU0xpjFixeba6%2B91kyZMsUYY8yhQ4dMly5dzN///ndTVVVl3n33XXPjjTea48ePG2OM2bBhg7nlllvMd999ZyoqKsz8%2BfPNXXfdZWpra40xxjz55JNm1KhRprCw0Jw8edI88MADZtasWcYYYyorK81vf/tb8%2Byzz5ozZ86Yr7/%2B2vTu3dts2bLFknOfzYIFC8yTTz5Zb/zXzHmpPfDAAyYpKcmcOnXKFBcXmyeeeMI8%2BuijbnOtLl682MTFxZlu3bqZ3Nzc8zrP8ePHTVxcnHnrrbfc/tqtqakxcXFx5m9/%2B5upqakxx44dM3fccYdZunSpW2f7pWuBguXisrOzTWRkpONftjHGvPvuu%2Ba222675GvZsGGDue2228y7777rVLBef/1189vf/tZp26eeespMnjzZGGPMxIkTzfTp053mf/e735n169cbY4zp06ePeeeddxxzeXl55pprrjFHjhxpMP/FnvunDh48aB555BFTXV3tGPvggw/MzTff7DEZ6xQXFxtjjKmtrTUHDhww8fHxZs2aNR6V85NPPjF33nmnSU5OdhSsRYsWmYceeshpu0ceecQ8//zzxhhj7r33XrN8%2BXLHXGVlpYmOjja7du0yZWVl5vrrrzd79%2B51zH/%2B%2BefmxhtvNGVlZWbnzp0mKirKnDlzxjG/YsUKM2LEiIs%2B97mMGDHCrF27tt74r5nzUtq/f7/p0qWL43o1xpjCwkLz3Xffuc21WneeunM39jy1tbVm5MiRZtq0acYY9792CwoKTGRkpHnllVdMdXW1OXbsmBkwYIBZtWqVW2f7pWuBW4QuLjMzU8HBwWrXrp1jLDw8XDk5OTp9%2BvQlXUvv3r3197//XXfeeWe9NUZGRjqNderUSRkZGZKkrKysc84XFxfr%2BPHjTvOhoaFq3bq1Dhw40GD%2Bizn3z/3mN7/RSy%2B9JF9fX8fYtm3bdP3113tMxjpBQUGSpFtvvVWDBg1S27ZtNWzYMI/JmZ%2Bfr2nTpum5556T3W53jDd0jJ/P%2B/n5qUOHDsrIyNDhw4dVVVXlNB8eHq6KigplZ2crMzNTHTt2VPPmzS9o/b907rOpra3V119/rY8%2B%2Bkh9%2B/ZVnz59NGPGDJ06depXzXkpffnll%2BrUqZNef/11xcfHq3fv3lq4cKHatm3rFtfqT89TN9/Y86SlpSkrK8txe9vdr92QkBA9%2BOCDWrhwobp06aJbb71VHTp00IMPPujW2X7pWqBgubjS0lKn/4OQ5HhdVlZ2SdfStm1b2Wy2euNnW6O/v79jfb80X3efPCAgoN58aWlpg/kv5ty/xBijv/zlL/rwww81bdo0j8woSe%2B//762b98uHx8fJScne0TO2tpapaSk6KGHHtK1117rNHcx5ygpKamXr27bX8rX2PWf77/HgoICde7cWXfccYe2bNmi1157TdnZ2UpJSflVc15Kp06d0oEDB5Sdna2NGzfqrbfe0okTJzRlyhS3uFZ/ep6fzjfmPCtXrtRjjz3m%2BMuQu1%2B7tbW18vf314wZM/T5559r8%2BbNOnjwoJYsWeLW2X5pnoLl4gICAlReXu40Vvc6MDCwKZZUj91udzwgXaeiosKxvl%2Bar7swf56xbr6h/Bdz7nMpKSlRcnKyNm3apLVr1%2Bqaa67xuIx1/P391a5dO6WkpGjHjh0ekXPFihVq3ry5Ro4cWS/vxZyj7j/gP81Q989BQUHnzNfY9Z/vv8fQ0FCtW7dOCQkJstvtuvLKK5WSkqLt27fLGPOr5byU6t51mDZtmoKCghQaGqrx48fr448/vqiMl%2Bpa/el5fjrfmPMUFBQoISHBMe7u1%2B7f//53bdu2Tffff7%2BaN2%2BuiIgIJSUl6X/%2B53/cOtsvzVOwXFxERISKiop08uRJx9jBgwd1%2BeWXq2XLlk24sv8TGRmpzMxMp7GsrCxFRERI%2BjHDueZbt26tdu3aKSsryzGXl5enoqIiRUZGNpj/Ys59NkeOHNHw4cNVUlKi9evX65prrvG4jP/617/0u9/9zvHpHUmqrKyUn5%2BfOnXq5PY53377be3evVtdu3ZV165dtXnzZm3evFldu3Y973NUVVUpOztbkZGR6tixo/z8/JzyHTx40HHLICIiQtnZ2U6fNPvpsS/m3GeTkZGh1NRUGWMcY5WVlfLx8dGNN974q%2BW8lDp16qTa2lpVVVU5xmprayVJ1113nctfqz89T918Y85jt9sVHx/v9K6Mu1%2B7x44dc/pvjiTZbDb5%2Bfm5dbZf3N/A5d13331mwoQJpri42PEpkyVLljTpmn76kHtBQYHp2rWreeWVV0xlZaXZtWuX00OAdZ%2Bq2LVrl%2BPTPN26dTOFhYXGGGP%2B8pe/mIEDB5ojR444Ps3zwAMPOM71S/kv9tw/VVRUZG677Tbz5JNPmpqaGqc5T8lojDElJSXm1ltvNfPmzTNnzpwxP/zwg0lISDAzZ870qJx1pkyZ4njIPSsry3Tp0sW8%2B%2B67jk8MdenSxXz//ffGmB8fWL3lllvMt99%2B6/jEUHx8vKmsrDTGGDNp0iTzwAMPmPz8fJOfn28eeOABx7GrqqpMv379zIIFC0xFRYX59ttvTe/evc2bb75pybl/7tixYyYqKsq8%2BOKLpqqqyvz73/8299xzj/nTn/70q%2Ba8lCorK018fLwZN26cKSkpMfn5%2BeYPf/iDSUpKcptrNTU11cTFxZmYmBjzww8/NOo8PXv2NK%2B//rrT98Ldr93MzExzww03mOXLl5vq6mpz5MgRM3DgQLNgwQK3zvZL1wIFyw3k5eWZcePGme7du5sePXqYBQsWOH3SrSn8tGAZY8yXX35pEhMTTXR0tOnfv7/j4q3z1ltvmTvuuMNERUWZhIQE8/nnnzvmKisrzbPPPmtuueUWc/PNN5vHH3/cnDx50jHfUP6LOfdPvfzyyyYyMtLcdNNNJioqyunLUzLWyczMNA899JDp2rWr6du3r1m0aJHjUzaelNMY54JljDHbt283gwcPNlFRUeauu%2B4yH330kWOutrbWrFq1yvTr189ERUWZkSNHOv5Da8yPn76cPn266dWrl%2BnWrZt58sknTWlpqWM%2BOzvbPPzwwyYmJsbccsstZsWKFU5ruZhzn016errj%2B9WjRw8zZ84cU1FR8avnvJSOHz9uxo8fb%2BLi4kzXrl3N5MmTzalTp4wx7nGt3nTTTSYuLs706NGj0ee56aabnP591XH3a3fnzp3m7rvvNjExMea2225z%2Bu%2BOO2c717XQzJifvL8MAACAi8YzWAAAABajYAEAAFiMggUAAGAxChYAAIDFKFgAAAAWo2ABAABYjIIFAABgMQoWAACAxShYAAAAFqNgAQAAWIyCBQAAYDEKFgAAgMUoWAAAABajYAEAAFiMggUAAGCx/wfK09TdQs9jawAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common3496147860795532987">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">350000.0</td>
        <td class="number">172</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">450000.0</td>
        <td class="number">172</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">550000.0</td>
        <td class="number">159</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">500000.0</td>
        <td class="number">152</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">425000.0</td>
        <td class="number">150</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">325000.0</td>
        <td class="number">148</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">400000.0</td>
        <td class="number">145</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">375000.0</td>
        <td class="number">138</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">300000.0</td>
        <td class="number">133</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">525000.0</td>
        <td class="number">131</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (3612)</td>
        <td class="number">20097</td>
        <td class="number">93.1%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme3496147860795532987">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">78000.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">80000.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">81000.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">82000.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">82500.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">5350000.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5570000.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6890000.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7060000.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7700000.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_sqft_above">sqft_above<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>942</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>4.4%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>1788.6</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>370</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>9410</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram8887117319707806487">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAASBJREFUeJzt3MEJwkAQQFEVS7IIe/JsTxZhT2sD8lExZtX37oG5fIbAJNsxxtgAd%2B3WHgBmtl97gHc4nC5PP3M9HxeYhF9jg0AQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAiE6Y4VXzk8hKXYIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBCmu%2Bb9FL8r5RE2CASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUD422PFVzx74Oi48fvZIBAEAkEgEAQCwUv6gny1%2BP0EMhlRzWU7xhhrDwGz8g4CQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQC4QYQABkAZXPIpwAAAABJRU5ErkJggg%3D%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives8887117319707806487,#minihistogram8887117319707806487"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives8887117319707806487">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles8887117319707806487"
                                                  aria-controls="quantiles8887117319707806487" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram8887117319707806487" aria-controls="histogram8887117319707806487"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common8887117319707806487" aria-controls="common8887117319707806487"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme8887117319707806487" aria-controls="extreme8887117319707806487"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles8887117319707806487">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>370</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>850</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>1190</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>1560</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>2210</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>3400</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>9410</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>9040</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>1020</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>827.76</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.4628</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>3.4055</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>1788.6</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>640.19</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>1.4474</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>38628326</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>685190</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram8887117319707806487">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3Xt0jXei//EP2SE7iUvq0tPp6SxGEjWqTSrE/bSpVFWic4Jm6tJSh1mEVBcNBuVXFENV1XAcrWpxpi6lRqto5zjMUaU6LqfaqFCXLpeEJJqr3L6/P6zsc3a1c1K%2Bnkfi/VrLwve79853P89%2BeNvPk62WMcYIAAAA1tR2ewEAAAA1DYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgGYEFAABgmcftBdwusrLyfL%2BuXbuW7rgjRNnZBaqoMC6u6vbE9ncf%2B8B97AN3sf2d06RJPVe%2BLu9guaB27VqqVauWateu5fZSbktsf/exD9zHPnAX27/mI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAs87i9ANw%2Bei7Y7fYSfpaPxnR2ewkAgGqKd7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsI7AAAAAsq7aBdeTIEQ0YMEAxMTHq0qWLZsyYoZKSEknSzp07lZiYqKioKPXs2VM7duzwu%2B%2ByZcvUrVs3RUVFadCgQTpx4oRvrrCwUBMnTlRsbKzatm2rtLQ0FRQUOPrcAABA9VYtA6uiokK/%2B93v1KNHD%2B3bt0/r16/Xf/3Xf2nZsmU6efKkRo8ereeee0779%2B/X6NGjNWbMGF24cEGStHHjRq1cuVJvvvmm9u7dq9atWys1NVXGGEnS9OnTde7cOW3btk3bt2/XuXPnNG/ePDefLgAAqGaqZWBdvnxZWVlZqqio8IVR7dq15fV6tXHjRsXExKh79%2B7yeDx6/PHH1a5dO61Zs0aStHbtWvXv318RERGqW7euxo4dq7Nnz2rv3r0qKirS5s2blZqaqoYNG6pRo0YaN26cNmzYoKKiIjefMgAAqEY8bi/geoSFhWnw4MGaM2eO/vCHP6i8vFyPPPKIBg8erNGjRysyMtLv9uHh4UpPT5ckZWRkaNiwYb65wMBANWvWTOnp6WrYsKFKS0v97t%2BiRQsVFxfr5MmTatWqVZXWl5mZqaysLL8xjydYTZs2lSQFBNT2%2Bxm3Jo%2BH/XOzcAy4j33gLrZ/zVctA6uiokJBQUGaMmWK%2Bvbtq1OnTmnUqFFauHChCgoK5PV6/W4fFBSkwsJCSfq78/n5%2BZKk4OBg31zlbX/OdVhr1qzRokWL/MZSUlKUmprqN1a/vv86cGsJCwtxewk1HseA%2B9gH7mL711zVMrA%2B/vhjbdu2TVu3bpUkRUREKCUlRTNnztSDDz6o4uJiv9sXFxcrJOTqX5Zer/cn5yvDqqioyHf7ylODoaGhVV5fcnKy4uLi/MY8nmDl5FyNtICA2qpf36vvvy9SeXlFlR8XzqrcX7CPY8B97AN3sf2d49Y/lqtlYJ07d873HYOVPB6PAgMDFRkZqSNHjvjNZWRk6L777pN0NcaOHTumhx9%2BWJJUWlqqkydPKjIyUs2bN1dgYKAyMjL0wAMPSJKOHz/uO41YVU2bNvWdDqyUlZWnsjL/g6i8vOKaMdw62Dc3H8eA%2B9gH7mL711zV8uRvly5dlJWVpX/9139VeXm5zpw5oyVLligxMVG9e/fWvn37tGXLFpWVlWnLli3at2%2BfnnjiCUlSnz59tGrVKqWnp%2BvKlSt65ZVX1LhxY8XExMjr9apnz56aN2%2BesrOzlZ2drXnz5ikhIUFBQUEuP2sAAFBdVMt3sMLDw7V06VItWLBAb7zxhurVq6fevXsrJSVFderU0R//%2BEfNmzdPkyZN0t13363XX39dzZs3lyT17dtXeXl5SklJUXZ2ttq0aaOlS5cqMDBQkjR16lTNmTNHiYmJKi0t1SOPPKIpU6a4%2BXQBAEA1U8tUfs4BbqqsrDzfrz2e2goLC1FOTsFt9dZwzwW73V7Cz/LRmM5uL6HGul2PgVsJ%2B8BdbH/nNGlSz5WvWy1PEQIAANzKCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLCCwAAADLPG4vADem54Ldbi8BAAD8AO9gAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWFZtAys3N1dpaWmKjY1Vu3btNHLkSGVmZkqSDh06pH79%2Bik6OlpxcXFat26d3303btyo%2BPh4RUVFKSkpSQcOHPDNlZeXa86cOerUqZOio6M1YsQI3%2BMCAABURbUNrNGjR6uwsFAff/yxduzYoYCAAE2ZMkWXL1/W8OHD9Zvf/Eaff/65Zs6cqVmzZunw4cOSpL1792r69OmaPXu2Pv/8c/Xu3VsjRoxQUVGRJGnJkiXavXu33nvvPf31r39VUFCQJk%2Be7OZTBQAA1Uy1DKwvv/xShw4d0uzZs1W/fn2FhoZq%2BvTpGjdunLZv366GDRtqwIAB8ng86tixoxITE7V69WpJ0rp169SrVy%2B1bdtWgYGBGjx4sMLCwrRlyxbf/LBhw3TXXXcpNDRUkyZN0q5du3TmzBk3nzIAAKhGPG4v4HocPnxY4eHhWrt2rf70pz%2BpqKhIXbt21fjx43Xs2DFFRkb63T48PFzr16%2BXJGVkZKhPnz7XzKenpysvL0/nz5/3u3/jxo3VoEEDHT16VPfcc0%2BV1peZmamsrCy/MY8nWE2bNpUkBQTU9vsZtyaPh/1zs3AMuI994C62f81XLQPr8uXLOnr0qO677z5t3LhRxcXFSktL0/jx49W4cWN5vV6/2wcFBamwsFCSVFBQ8JPzBQUFkqTg4OBr5ivnqmLNmjVatGiR31hKSopSU1P9xurX918Hbi1hYSFuL6HG4xhwH/vAXWz/mqtaBladOnUkSZMmTVLdunUVGhqqMWPG6Mknn1RSUpKKi4v9bl9cXKyQkKt/WXq93h%2BdDwsL84VX5fVYP3b/qkhOTlZcXJzfmMcTrJycq5EWEFBb9et79f33RSovr6jy48JZlfsL9nEMuI994C62v3Pc%2BsdytQys8PBwVVRUqLS0VHXr1pUkVVRcfYG2atVK//7v/%2B53%2B4yMDEVEREiSIiIidOzYsWvmu3XrpgYNGujOO%2B9URkaG7zRhVlaWcnNzrznt%2BPc0bdrUdzqwUlZWnsrK/A%2Bi8vKKa8Zw62Df3HwcA%2B5jH7iL7V9zVcuTv506ddI999yj3//%2B9yooKFB2drZeffVVde/eXQkJCbp48aJWrFih0tJSffbZZ9q8ebPvuqu%2Bfftq8%2BbN%2Buyzz1RaWqoVK1bo0qVLio%2BPlyQlJSVpyZIlOnPmjPLz8/Xyyy%2Brffv2%2BuUvf%2BnmUwYAANVItXwHKzAwUCtXrtTs2bPVo0cPXblyRXFxcZo0aZLq16%2Bv5cuXa%2BbMmVq4cKHuuOMOTZ48WR06dJAkdezYUVOnTtW0adN04cIFhYeHa9myZWrYsKGkq9dKlZWVacCAASooKFBsbKwWLFjg5tMFAADVTC1jjHF7EbeDrKw83689ntoKCwtRTk7BDb813HPB7htdGn7CR2M6u72EGsvmMYDrwz5wF9vfOU2a1HPl61bLU4QAAAC3MgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAMgILAADAsmofWOXl5Ro0aJAmTJjgG9u5c6cSExMVFRWlnj17aseOHX73WbZsmbp166aoqCgNGjRIJ06c8M0VFhZq4sSJio2NVdu2bZWWlqaCggLHng8AAKj%2Bqn1gLVq0SPv37/f9/uTJkxo9erSee%2B457d%2B/X6NHj9aYMWN04cIFSdLGjRu1cuVKvfnmm9q7d69at26t1NRUGWMkSdOnT9e5c%2Be0bds2bd%2B%2BXefOndO8efNceW4AAKB6qtaBtWfPHm3fvl2PPvqob2zjxo2KiYlR9%2B7d5fF49Pjjj6tdu3Zas2aNJGnt2rXq37%2B/IiIiVLduXY0dO1Znz57V3r17VVRUpM2bNys1NVUNGzZUo0aNNG7cOG3YsEFFRUVuPU0AAFDNeNxewPW6dOmSJk2apMWLF2vFihW%2B8YyMDEVGRvrdNjw8XOnp6b75YcOG%2BeYCAwPVrFkzpaenq2HDhiotLfW7f4sWLVRcXKyTJ0%2BqVatWVVpbZmamsrKy/MY8nmA1bdpUkhQQUNvvZ9yaPB72z83CMeA%2B9oG72P41n%2BOBVV5eroCAgBt6jIqKCr3wwgsaMmSI7r33Xr%2B5goICeb1ev7GgoCAVFhb%2Bn/P5%2BfmSpODgYN9c5W1/znVYa9as0aJFi/zGUlJSlJqa6jdWv77/OnBrCQsLcXsJNR7HgPvYB%2B5i%2B9dcjgdWt27d9MQTTygpKUnh4eHX9RhLly5VnTp1NGjQoGvmvF6viouL/caKi4sVEhLyf85XhlVRUZHv9pWnBkNDQ6u8vuTkZMXFxfmNeTzBysm5GmkBAbVVv75X339fpPLyiio/LpxVub9gH8eA%2B9gH7mL7O8etfyw7HlijRo3Spk2btHz5crVp00Z9%2BvRRr169VK9evSo/xqZNm5SZmamYmBhJ8gXTJ598ogEDBujIkSN%2Bt8/IyNB9990nSYqIiNCxY8f08MMPS5JKS0t18uRJRUZGqnnz5goMDFRGRoYeeOABSdLx48d9pxGrqmnTpr7TgZWysvJUVuZ/EJWXV1wzhlsH%2B%2Bbm4xhwH/vAXWz/msvxk79PPfWU3n33XW3dulWdOnXSsmXL1KVLF40dO1affvpplR5j69at%2Btvf/qb9%2B/dr//79SkhIUEJCgvbv36/evXtr37592rJli8rKyrRlyxbt27dPTzzxhCSpT58%2BWrVqldLT03XlyhW98soraty4sWJiYuT1etWzZ0/NmzdP2dnZys7O1rx585SQkKCgoKCbuVkAAEAN4trVdc2aNdPzzz%2BvrVu3KiUlRX/5y180dOhQxcXF6a233lJ5efl1PW6LFi30xz/%2BUUuXLlW7du20ePFivf7662revLkkqW/fvho8eLBSUlLUoUMHffXVV1q6dKkCAwMlSVOnTlWzZs2UmJioxx57TP/4j/%2BoF1980drzBgAANV8tU/kBUA47dOiQ3n//fW3ZskUlJSXq3r27kpKSdOHCBb322muKjo7W/Pnz3VjaTZGVlef7tcdTW2FhIcrJKbjht4Z7Lth9o0tDDfHRmM5uL6HKbB4DuD7sA3ex/Z3TpEnVL0GyyfFrsBYvXqxNmzbp1KlTatOmjZ5//nklJCT4XUQeEBDAu0YAAKDacjywVq1apd69e6tv374/%2BV2ELVq00Lhx4xxeGQAAgB2OB9auXbuUn5%2Bv3Nxc39iWLVvUsWNHhYWFSZJ%2B/etf69e//rXTSwMAALDC8Yvcv/rqK/Xo0cP3X9dI0ty5c5WYmKhvvvnG6eUAAABY53hg/eEPf9Cjjz6q559/3jf2ySefqFu3bpo9e7bTywEAALDO8cA6cuSIhg8frjp16vjGAgICNHz4cB08eNDp5QAAAFjneGCFhobq9OnT14yfP3%2BeD/MEAAA1guOB1aNHD02bNk2ffvqp8vPzVVBQoM8%2B%2B0wvvfSS4uPjnV4OAACAdY5/F%2BHYsWN15swZPfvss6pVq5ZvPD4%2BXmlpaU4vBwAAwDrHA8vr9Wrp0qX69ttvdfToUQUGBqpFixY/6z9TBgAAuJU5HliVmjdv7vv/AQEAAGoSxwPr22%2B/1UsvvaQvvvhCpaWl18x//fXXTi8JAADAKscDa9q0aTp79qzGjRunevXc%2BQ8YAQAAbibHA%2BvAgQN6%2B%2B23FR0d7fSXBgAAcITjH9MQFhamkJAQp78sAACAYxwPrEGDBmn%2B/PnKy8tz%2BksDAAA4wvFThDt37tTBgwcVGxurRo0a%2Bf2XOZL0l7/8xeklAQAAWOV4YMXGxio2NtbpLwsAAOAYxwNr1KhRTn9JAAAARzl%2BDZYkpaena%2BLEifrtb3%2BrCxcuaPXq1dq7d68bSwEAALDO8cD68ssv1a9fP3333Xf68ssvVVJSoq%2B//lrPPvusduzY4fRyAAAArHM8sObNm6dnn31WK1euVGBgoCRpxowZevrpp7Vo0SKnlwMAAGCdK%2B9g/eY3v7lm/KmnntKJEyecXg4AAIB1jgdWYGCg8vPzrxk/e/asvF6v08sBAACwzvHA6t69u1555RXl5OT4xo4fP66ZM2fqoYcecno5AAAA1jkeWOPHj1dxcbE6deqkoqIiJSUlKSEhQR6PR2lpaU4vBwAAwDrHPwcrNDRU7777rvbs2aOvvvpKFRUVioyMVNeuXVW7tiufGgEAAGCV44FVqWPHjurYsaNbXx4AAOCmcTyw4uLiVKtWrZ%2Bc5/8iBAAA1Z3jgfXP//zPfoFVWlqqU6dOadeuXRozZozTywEAALDO8cAaPXr0j46vWrVKX3zxhZ5%2B%2BmmHVwQAAGDXLXNV%2BcMPP6ydO3e6vQwAAIAbdssE1r59%2B1S3bl23lwEAAHDDHD9F%2BMNTgMYY5efn6%2BjRo5weBAAANYLjgfWLX/zimu8iDAwM1DPPPKPExESnlwMAAGCd44E1e/Zsp78kAACAoxwPrM8//7zKt23Xrt1NXAkAAMDN4XhgDR48WMYY349KlacNK8dq1aqlr7/%2B2unlAQAA3DDHA%2Bv111/XrFmzNH78eHXo0EGBgYE6dOiQpk2bpv79%2B%2Bvhhx92ekkAAABWOf4xDXPmzNHUqVPVvXt3hYaGqm7dumrfvr1eeuklLV%2B%2BXHfffbfvBwAAQHXkeGBlZmbqrrvuumY8NDRUOTk5Ti8HAADAOscDKyoqSvPnz1d%2Bfr5vLDc3V3PnzlXHjh2dXg4AAIB1jl%2BDNXnyZD3zzDPq1q2bmjVrJkn69ttv1aRJE73zzjtOLwcAAMA6xwOrRYsW2rJlizZv3qzjx49Lkvr3769evXrJ6/U6vRwAAADrHA8sSapfv7769eun7777Tvfcc4%2Bkq5/mDgAAUBM4fg2WMUbz5s1Tu3btlJCQoPPnz2v8%2BPGaOHGiSktLnV4OAACAdY4H1sqVK7Vp0yZNnTpVderUkSR1795d//Ef/6HXXnvN6eUAAABY53hgrVmzRi%2B%2B%2BKKSkpJ8n97%2B%2BOOPa%2BbMmfrwww%2BdXg4AAIB1jgfWd999p1atWl0z3rJlS128eNHp5QAAAFjneGDdfffdOnz48DXjO3fu9F3wDgAAUJ05/l2EQ4cO1f/7f/9PFy5ckDFGe/bs0bvvvquVK1dq4sSJTi8HAADAOscDq0%2BfPiorK9OSJUtUXFysF198UY0aNdLzzz%2Bvp556yunlAAAAWOd4YP35z3/WY489puTkZGVnZ8sYo0aNGjm9DAAAgJvG8WuwZsyY4buY/Y477iCuAABAjeN4YDVr1kxHjx51%2BssCAAA4xvFThBERERo3bpzeeOMNNWvWTHXr1vWbnzVrltNLAgAAsMrxd7BOnz6ttm3bKiQkRFlZWfruu%2B/8flRVenq6hgwZovbt26tz585KS0tTdna2JOnQoUPq16%2BfoqOjFRcXp3Xr1vndd%2BPGjYqPj1dUVJSSkpJ04MAB31x5ebnmzJmjTp06KTo6WiNGjFBmZqadJw8AAG4LjryDNWvWLD333HMKDg7WypUrb/jxiouL9S//8i968skntXTpUhUUFGj8%2BPH6/e9/rzlz5mj48OFKTU1VcnKyPv/8c6WkpKhly5a6//77tXfvXk2fPl3Lli3T/fffr9WrV2vEiBHasWOHvF6vlixZot27d%2Bu9995TvXr1NGXKFE2ePFn/9m//ZmFLAACA24Ej72C98847Kioq8hsbOnTodb8zdPbsWd17771KSUlRnTp1FBYW5oup7du3q2HDhhowYIA8Ho86duyoxMRErV69WpK0bt069erVS23btlVgYKAGDx6ssLAwbdmyxTc/bNgw3XXXXQoNDdWkSZO0a9cunTlz5sY2AgAAuG048g6WMeaasb/97W%2B6cuXKdT3er371K73xxht%2BY9u2bVPr1q117NgxRUZG%2Bs2Fh4dr/fr1kqSMjAz16dPnmvn09HTl5eXp/Pnzfvdv3LixGjRooKNHj1b5k%2BYzMzOVlZXlN%2BbxBKtp06aSpICA2n4/AzZ4PNXn9cQx4D72gbvY/jWf4xe522aM0YIFC7Rjxw6tWrVK77zzjrxer99tgoKCVFhYKEkqKCj4yfmCggJJUnBw8DXzlXNVsWbNGi1atMhvLCUlRampqX5j9ev7rwO4EWFhIW4v4WfjGHAf%2B8BdbP%2Baq1oHVn5%2BviZOnKgjR45o1apVatmypbxer/Ly8vxuV1xcrJCQq3/5eL1eFRcXXzMfFhbmC68fns783/eviuTkZMXFxfmNeTzBysm5GmkBAbVVv75X339fpPLyiio/LvD3VL6%2BqgOOAfexD9zF9neOW//4dCywatWqZfXxTp8%2BrWHDhukXv/iF1q9frzvuuEOSFBkZqd27d/vdNiMjQxEREZKufkzEsWPHrpnv1q2bGjRooDvvvFMZGRm%2B04RZWVnKzc295rTj39O0aVPf6cBKWVl5KivzP4jKyyuuGQOuV3V8LXEMuI994C62f83lWGDNmDHD7zOvSktLNXfu3GveGarK52BdvnxZzzzzjDp06KCZM2eqdu3/OYcdHx%2BvuXPnasWKFRowYIC%2B%2BOILbd68WYsXL5Yk9e3bVykpKerZs6fatm2r1atX69KlS4qPj5ckJSUlacmSJWrTpo3CwsL08ssvq3379vrlL39pYzMAAIDbgCOB1a5du2su%2Bo6OjlZOTo5ycnJ%2B9uNt2LBBZ8%2Be1UcffaStW7f6zR04cEDLly/XzJkztXDhQt1xxx2aPHmyOnToIEnq2LGjpk6dqmnTpunChQsKDw/XsmXL1LBhQ0lXr5UqKyvTgAEDVFBQoNjYWC1YsOA6nzkAALgd1TI/9i1%2BsC4r63%2BuC/N4aissLEQ5OQU3/NZwzwW7/%2B8b4bbw0ZjObi%2BhymweA7g%2B7AN3sf2d06RJPVe%2BLt8fCgAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYBmBBQAAYJnH7QUAsKPngt1uL6HKPh7X1e0lAMBNxTtYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlhFYAAAAlnncXgCA20/8vL%2B6vYSf5aMxnd1eAoBqhnewAAAALCOwAAAALCOwAAAALCOwAAAALCOwAAAALCOwfsSlS5c0cuRIxcTEKDY2VjNnzlRZWZnbywIAANUEgfUjxowZo%2BDgYP31r3/V%2BvXrtWfPHq1YscLtZQEAgGqCwPqBU6dOad%2B%2BfXrhhRfk9Xp1zz33aOTIkVq9erXbSwMAANUEHzT6A8eOHVPDhg115513%2BsZatGihs2fP6vvvv1f9%2BvVdXB0AN/RcsNvtJfwsfDAq4D4C6wcKCgrk9Xr9xip/X1hYWKXAyszMVFZWlt%2BYxxOspk2bSpICAmr7/QwANlWnIPx4XFe3l%2BAK/h6o%2BQisHwgODlZRUZHfWOXvQ0JCqvQYa9as0aJFi/zGRo0apdGjR0srqWOpAAAIx0lEQVS6GmBvv/2GkpOTfdF1vfbPfOyG7n87yszM1Jo1a6xsf1wf9oH72Afusvn3AG5NpPMPREREKDc3VxcvXvSNHT9%2BXP/wD/%2BgevXqVekxkpOTtWHDBr8fycnJvvmsrCwtWrTomne54Ay2v/vYB%2B5jH7iL7V/z8Q7WDzRr1kxt27bVyy%2B/rJdeekk5OTlavHix%2BvbtW%2BXHaNq0Kf8iAQDgNsY7WD9i4cKFKisr0yOPPKInn3xSXbt21ciRI91eFgAAqCZ4B%2BtHNG7cWAsXLnR7GQAAoJoKmDZt2jS3F3E7CgkJUfv27at84TzsYvu7j33gPvaBu9j%2BNVstY4xxexEAAAA1CddgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgAQAAWEZgoUZJT0/XkCFD1L59e3Xu3FlpaWnKzs6WJB06dEj9%2BvVTdHS04uLitG7dOr/7bty4UfHx8YqKilJSUpIOHDjgmysvL9ecOXPUqVMnRUdHa8SIEcrMzLzh9ZaXl2vQoEGaMGGCb2znzp1KTExUVFSUevbsqR07dvjdZ9myZerWrZuioqI0aNAgnThxwjdXWFioiRMnKjY2Vm3btlVaWpoKCgpueJ1wXm5urtLS0hQbG6t27dpp5MiRvtfcrfhaxlVHjhzRgAEDFBMToy5dumjGjBkqKSmRxLF92zFADVFUVGQ6d%2B5sXnvtNXPlyhWTnZ1thg0bZn73u9%2BZ3Nxc0759e7Nq1SpTWlpqPv30UxMdHW0OHTpkjDHms88%2BM9HR0Wb//v2mpKTEvPXWWyY2NtYUFhYaY4x5/fXXTWJiojl79qzJy8szY8aMMcOGDbvhNS9YsMDce%2B%2B9Zvz48cYYY7799lvTpk0b8/HHH5vS0lLz4Ycfmvvvv9%2BcP3/eGGPMhg0bTNeuXc0333xjiouLzaxZs0yvXr1MRUWFMcaYCRMmmGeeecbk5OSYixcvmoEDB5pp06bd8DrhvIEDB5qUlBRz%2BfJlk5eXZ0aNGmWGDx9%2By76WYUx5ebnp3Lmzefvtt015ebk5d%2B6c6dGjh1m0aBHH9m2IwEKNcfz4cTN06FBTVlbmG/vkk0/Mgw8%2BaNauXWseffRRv9u/%2BOKLJi0tzRhjzNixY83kyZP95h977DGzfv16Y4wx3bp1M3/%2B8599c1lZWaZly5bm9OnT173eTz/91Dz%2B%2BOMmNTXVF1jz5883Q4YM8bvd0KFDzWuvvWaMMea3v/2tWbJkiW%2BupKTEREdHmz179pjCwkLTunVr88UXX/jmDx48aO6//37fX66oHv77v//btGnTxuTl5fnGcnJyzDfffHNLvpZxVXZ2tomMjDRvvfWWKSsrM%2BfOnTM9e/Y0b775Jsf2bYhThKgxfvWrX%2BmNN95QQECAb2zbtm1q3bq1jh07psjISL/bh4eHKz09XZKUkZHxk/N5eXk6f/6833zjxo3VoEEDHT169LrWeunSJU2aNEmvvPKKvF6vb/zvrePH5gMDA9WsWTOlp6fr1KlTKi0t9Ztv0aKFiouLdfLkyetaJ9xx%2BPBhhYeHa%2B3atYqPj1eXLl00Z84cNWnS5JZ7LeN/hIWFafDgwZozZ47atGmjf/qnf1KzZs00ePBgju3bEIGFGskYo1dffVU7duzQpEmTVFBQ4BcykhQUFKTCwkJJ%2Brvzldc5BAcHXzN/PddAVFRU6IUXXtCQIUN07733%2Bs3dyDrz8/OvWWflbblWo3q5fPmyjh49qpMnT2rjxo16//33deHCBY0fP/6Wei3DX0VFhYKCgjRlyhQdPHhQH3zwgY4fP66FCxdybN%2BGCCzUOPn5%2BUpNTdXmzZu1atUqtWzZUl6vV8XFxX63Ky4uVkhIiCT93fnKP8iKiop%2B8v4/x9KlS1WnTh0NGjTomrkbWWflH77/e52Vvw4NDf3Z64R76tSpI0maNGmSQkND1bhxY40ZM0Y7d%2B6UMeaWeS3D38cff6xt27apf//%2BqlOnjiIiIpSSkqI//elPHNu3IQILNcrp06fVp08f5efna/369WrZsqUkKTIyUseOHfO7bUZGhiIiIiRJERERPznfoEED3XnnncrIyPDNZWVlKTc395q3/Kti06ZN2rdvn2JiYhQTE6MPPvhAH3zwgWJiYn72OktLS3Xy5ElFRkaqefPmCgwM9Fvn8ePHfacaUH2Eh4eroqJCpaWlvrGKigpJUqtWrW6Z1zL8nTt3zvcdg5U8Ho8CAwM5tm9Hbl8EBtiSm5trHnroITNhwgRTXl7uN5ednW1iYmLMW2%2B9ZUpKSsyePXt8F5AaY3zfibVnzx7fd161a9fO5OTkGGOMefXVV01CQoI5ffq07zuvBg4caGXd48eP913knpGRYdq0aWM%2B/PBD33catWnTxpw4ccIYY8zatWtN165dzddff%2B37TqP4%2BHhTUlJijDFm3LhxZuDAgebSpUvm0qVLZuDAgb7HRvVRUlJi4uPjzejRo01%2Bfr65dOmSefrpp01KSsot/Vq%2B3R07dszcd999ZsmSJaasrMycPn3aJCQkmNmzZ3Ns34YILNQYy5cvN5GRkeaBBx4wUVFRfj%2BMMebw4cMmOTnZREdHm0ceecS89957fvd///33TY8ePUxUVJTp27evOXjwoG%2BupKTEzJ0713Tt2tU8%2BOCDZsSIEebixYtW1v2/A8sYY3bt2mV69%2B5toqKiTK9evcx//ud/%2BuYqKirMm2%2B%2BaeLi4kxUVJQZNGiQ7w9oY4zJy8szkydPNp06dTLt2rUzEyZMMAUFBVbWCWedP3/ejBkzxnTu3NnExMSYtLQ0c/nyZWPMrftahjG7d%2B82/fr1M23btjUPPfSQmT9/vrly5YoxhmP7dlPLGGPcfhcNAACgJuEaLAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMsILAAAAMv%2BP1w2ZVHu1qACAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common8887117319707806487">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1300</td>
        <td class="number">212</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1010</td>
        <td class="number">210</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1200</td>
        <td class="number">206</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1220</td>
        <td class="number">192</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1140</td>
        <td class="number">184</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1400</td>
        <td class="number">180</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1060</td>
        <td class="number">178</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1180</td>
        <td class="number">177</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1340</td>
        <td class="number">176</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1250</td>
        <td class="number">174</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (932)</td>
        <td class="number">19708</td>
        <td class="number">91.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme8887117319707806487">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">370</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">380</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">390</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">410</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">420</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">7880</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8020</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8570</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8860</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9410</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_sqft_basement">sqft_basement<br/>
            <small>Categorical</small>
        </p>
    </div><div class="col-md-3">
    <table class="stats ">
        <tr class="alert">
            <th>Distinct count</th>
            <td>304</td>
        </tr>
        <tr>
            <th>Unique (%)</th>
            <td>1.4%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (%)</th>
            <td>0.0%</td>
        </tr>
        <tr class="ignore">
            <th>Missing (n)</th>
            <td>0</td>
        </tr>
    </table>
</div>
<div class="col-md-6 collapse in" id="minifreqtable-890400873158281710">
    <table class="mini freq">
        <tr class="">
    <th>0.0</th>
    <td>
        <div class="bar" style="width:100%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 59.4%">
            12826
        </div>
        
    </td>
</tr><tr class="">
    <th>?</th>
    <td>
        <div class="bar" style="width:4%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 2.1%">
            &nbsp;
        </div>
        454
    </td>
</tr><tr class="">
    <th>600.0</th>
    <td>
        <div class="bar" style="width:2%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 1.0%">
            &nbsp;
        </div>
        217
    </td>
</tr><tr class="other">
    <th>Other values (301)</th>
    <td>
        <div class="bar" style="width:63%" data-toggle="tooltip" data-placement="right" data-html="true"
             data-delay=500 title="Percentage: 37.5%">
            8100
        </div>
        
    </td>
</tr>
    </table>
</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#freqtable-890400873158281710, #minifreqtable-890400873158281710"
       aria-expanded="true" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="col-md-12 extrapadding collapse" id="freqtable-890400873158281710">
    
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.0</td>
        <td class="number">12826</td>
        <td class="number">59.4%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">?</td>
        <td class="number">454</td>
        <td class="number">2.1%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">600.0</td>
        <td class="number">217</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">500.0</td>
        <td class="number">209</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">700.0</td>
        <td class="number">208</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">800.0</td>
        <td class="number">201</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">400.0</td>
        <td class="number">184</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1000.0</td>
        <td class="number">148</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">300.0</td>
        <td class="number">142</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">900.0</td>
        <td class="number">142</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (294)</td>
        <td class="number">6866</td>
        <td class="number">31.8%</td>
        <td>
            <div class="bar" style="width:53%">&nbsp;</div>
        </td>
</tr>
</table>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_sqft_living">sqft_living<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>1034</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>4.8%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>2080.3</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>370</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>13540</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram877366547351027798">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAARxJREFUeJzt3cEJwkAQQFEjlmQR9uTZnizCntYG5KNiTKLv3QNz%2BSxhlmQaY4wd8NB%2B6QFgzQ5LD/AJx/P15Wdul9MMk/BrnCAQBAJBIBAEAkEgEAQCQSAQBAJhdYvCd5Z%2BMBcnCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCITVfVnxW/y2jWc4QSAIBIJAIAgEgkAgCASCQCAIBIJAIPztJv0dr27fbd63TyAzcp1l%2B6Yxxlh6CFgr7yAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQ7s5SFf6TsAAiAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives877366547351027798,#minihistogram877366547351027798"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives877366547351027798">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles877366547351027798"
                                                  aria-controls="quantiles877366547351027798" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram877366547351027798" aria-controls="histogram877366547351027798"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common877366547351027798" aria-controls="common877366547351027798"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme877366547351027798" aria-controls="extreme877366547351027798"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles877366547351027798">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>370</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>940</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>1430</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>1910</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>2550</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>3760</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>13540</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>13170</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>1120</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>918.11</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.44133</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>5.2521</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>2080.3</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>698.08</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>1.4732</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>44928711</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>842920</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram877366547351027798">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3Xl4lPW9//8XYYJkMRCFQI%2BHc4IkcSnYpATCfjASkbUchKYKceNIayI0/tQAhQqlRlAQNSIWrUgLnIogaBFU1GPRi8WAC1QKmEQWvVgSskBWsn1%2Bf8w3g2OARPhkZhKej%2BviCvO573vmfb/ncw8vZu6508oYYwQAAABr/LxdAAAAQEtDwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAljm8XcDlIi%2BvuMF1/Pxa6aqrglRQUKraWuOBqnwTfXCiD0704Sx64UQfnOiDU0N96NjxSi9UxTtYPsXPr5VatWolP79W3i7Fq%2BiDE31wog9n0Qsn%2BuBEH5x8tQ8ELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwzOHtAnD5GPbsVm%2BX8KO8k9rf2yUAAJqpZvkOVn5%2BvpKTkxUbG6u4uDilp6erurratXz37t0aP368YmJiFB8frzVr1jT6vmtqavTkk0%2BqX79%2BiomJ0QMPPKDc3Nym2A0AANBCNcuAlZqaqsDAQH3yySdau3attm/fruXLl0uSTp06pcmTJ2vMmDHauXOn0tPTNW/ePO3Zs6dR9/3iiy9q69ateuONN/TJJ5%2Bobdu2mjVrVhPuDQAAaGmaXcA6fPiwMjMz9eijjyogIEBdunRRcnKyVq1aJUnavHmz2rdvrwkTJsjhcKhv374aNWqUa3lD1qxZo/vvv18/%2BclPFBwcrJkzZ%2Brjjz/Wt99%2B25S7BQAAWpBmdw5WVlaW2rdvr06dOrnGunXrpqNHj%2Br06dPKyspSVFSU2zYRERFau3Ztg/ddXFys48ePu23foUMHtWvXTgcOHFCXLl0aVWNubq7y8vLcxhyOQIWFhV1wu9at/dx%2BwrscDu8%2BD8wHJ/pwFr1wog9O9MHJV/vQ7AJWaWmpAgIC3MbqbpeVlZ1zedu2bVVWVtao%2B5akwMDAetvXLWuM1atXa/HixW5jKSkpmjp1aqO2DwkJaHglNLnQ0CBvlyCJ%2BVCHPpxFL5zogxN9cPK1PjS7gBUYGKjy8nK3sbrbQUFBCggIUHFxsdvyiooKBQU1/I9lXTD74f03dvs6iYmJio%2BPdxtzOAJVWHjhkNa6tZ9CQgJ0%2BnS5ampqG/14aBoNPV9NjfngRB/OohdO9MGJPjg11Adv/We52QWsyMhIFRUV6eTJk%2BrQoYMkKScnR507d9aVV16pqKgobd3qfjmA7OxsRUZGNnjf7dq1U6dOnZSdne36mDAvL09FRUX1Pna8kLCwsHofB%2BblFau6unEHQE1NbaPXRdPxleeA%2BeBEH86iF070wYk%2BOPlaH3zrA8tGCA8PV8%2BePfXEE0%2BopKRE3377rZYsWaJx48ZJkhISEnTy5EktX75cVVVV2rFjhzZs2KDbb7%2B9Ufc/duxYvfjii/r2229VUlKiJ554Qr1799Z//Md/NOVuAQCAFqTZvYMlSRkZGZo7d65uueUW%2Bfn5acyYMUpOTpYkhYaGatmyZUpPT1dGRoauuuoqzZo1S3369GnUfaekpKi6uloTJkxQaWmp4uLi9Oyzzzbl7gAAgBamlTHGeLuIy0FeXnGD6zgcfgoNDVJhYalPvc1pC1dy/3Fa%2BnxoLPpwFr1wog9O9MGpoT507HilF6pqhh8RAgAA%2BDoCFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAljm8XQAuzbBnt3q7BAAA8AO8gwUAAGAZAQsAAMAyAhYAAIBlBCwAAADLWnzAWrlypSorK71dBgAAuIy06ID1%2Buuva%2B/evZo9e7bi4uLUs2dPpaWlqbS01LXOwYMHdffddysmJkYDBgzQn/70J7f72LJli0aNGqXo6GgNGzZMH330kad3AwAANDMtNmAVFxdr0aJFkqRjx47pvffe0%2BbNm3Xs2DEtXLhQklRVVaXf/OY36tGjhz799FO99NJLWrVqld555x1J0qFDhzRlyhT99re/1a5duzRlyhSlpqbqxIkTXtsvAADg%2B1pswFq1apVuuOEGbdiwQVOnTlX79u119dVX65FHHtG6detUXl6unTt3Kjc3V1OnTlWbNm104403KikpSatWrZIkrV%2B/XrGxsRoyZIgcDoeGDx%2BuXr16afXq1V7eOwAA4Mta7IVG33vvPY0ZM0bbtm1TVFSUa7xbt26qqKjQoUOHlJWVpa5du6pNmzau5REREXrppZckSdnZ2W7b1i3fv3//BR87NzdXeXl5bmMOR6DCwsIuuF3r1n5uP%2BFdDod3nwfmgxN9OIteONEHJ/rg5Kt9aJEBq7KyUvv27dOMGTMkSYGBga5lAQEBkqTS0lKVlpa6bn9/eVlZmWudHy5v27ata/n5rF69WosXL3YbS0lJ0dSpUxtVf0hIQMMrocmFhgZ5uwRJzIc69OEseuFEH5zog5Ov9aFFBqyioiIZY9S2bVtJUnl5uYKCglx/l6Tg4GAFBga6btf5/roBAQGqqKhwW15RUeFafj6JiYmKj493G3M4AlVYWHqeLZxat/ZTSEiATp8uV01NbQN7iabW0PPV1JgPTvThLHrhRB%2Bc6INTQ33w1n%2BWW2TAqgtAbdq0kb%2B/v7Kzs/Wzn/1MkpSTkyN/f3%2BFh4crPz9fhw4dUnV1tRwOZyuys7MVGRkpSYqKitLevXvd7js7O1vdu3e/4OOHhYXV%2BzgwL69Y1dWNOwBqamobvS6ajq88B8wHJ/pwFr1wog9O9MHJ1/rgWx9YWhIUFKTQ0FAdPXpUw4YN08KFC1VQUKCCggItXLhQI0eOVNu2bRUXF6fQ0FA9/fTTOnPmjPbv368VK1Zo3LhxkqTRo0crMzNTmzZtUnV1tTZt2qTMzEz94he/8PIeAgAAX9YiA5YkDRw4UDt37tTs2bMVHh6uUaNG6bbbbtO///u/67HHHpMkORwOLVu2TF9//bX69%2B%2BvyZMnKykpSWPHjpXkPCH%2BhRde0NKlS9WrVy8tWbJEzz//vLp27erNXQMAAD6ulTHGeLuIpvDpp59q%2BvTp%2BvDDD%2BXn5/0cmZdX3OA6DoefQkODVFhY2ui3OYc9u/VSS8N5vJPa36uPfzHzoSWiD2fRCyf64EQfnBrqQ8eOV3qhqhb8DlZcXJzCw8P1/vvve7sUAABwmWmxAUuSHn/8cb300kv8LkIAAOBRLfJbhHWuueYavfHGG94uAwAAXGZa9DtYAAAA3kDAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFjm8YBVU1Pj6YcEAADwKI8HrEGDBumpp55Sdna2px8aAADAIzwesB588EF9/vnnGjlypMaPH6/XXntNxcXFni4DAACgyXg8YN1xxx167bXX9O6776pfv356%2BeWXNWDAAD388MPatm2bp8sBAACwzmsnuYeHh%2Buhhx7Su%2B%2B%2Bq5SUFH344YeaNGmS4uPj9eqrr3KuFgAAaLYc3nrg3bt3680339SmTZtUWVmphIQEjR07VidOnNBzzz2nf/7zn1q0aJG3ygMAALhoHg9YS5Ys0VtvvaXDhw%2BrR48eeuihhzRy5EgFBwe71mndurUee%2BwxT5cGAABghccD1sqVKzV69GiNGzdOERER51ynW7dueuSRRzxcGQAAgB0eD1gff/yxSkpKVFRU5BrbtGmT%2Bvbtq9DQUEnSjTfeqBtvvNHTpQEAAFjh8ZPc//Wvf2no0KFavXq1a2zBggUaNWqUvv76a0%2BXAwAAYJ3HA9ZTTz2lW2%2B9VQ899JBr7IMPPtCgQYM0f/58T5cDAABgnccD1t69ezV58mS1adPGNda6dWtNnjxZX375pafLAQAAsM7jASs4OFhHjhypN378%2BHG1bdvW0%2BUAAABY5/GANXToUM2ZM0fbtm1TSUmJSktLtWPHDs2dO1cJCQmeLgcAAMA6jweshx9%2BWNdee63uu%2B8%2B9erVS7Gxsbr33nsVERGhtLS0Rt9PUVGR0tLSFBcXp169eik5OVm5ubmSnBcxHT9%2BvGJiYhQfH681a9bU237Lli0aNWqUoqOjNWzYMH300Uf11snPz1dycrJiY2MVFxen9PR0VVdXX/zOAwCAy4LHL9MQEBCgpUuX6uDBgzpw4ID8/f3VrVs3hYeH/6j7mTJlitq1a6f3339ffn5%2BmjFjhn7/%2B9/rqaee0uTJkzV16lQlJiZq586dSklJ0XXXXaebbrpJknTo0CFNmTJFixYt0uDBg7V582alpqZq8%2BbN6tSpk%2BsxUlNT1alTJ33yySc6efKkHnjgAS1fvlz/8z//Y7MlAACghfHar8rp2rWrunbtelHbfvXVV9q9e7e2bdvmugL8H//4R%2BXl5Wnz5s1q3769JkyYIEnq27evRo0apVWrVrkC1vr16xUbG6shQ4ZIkoYPH65169Zp9erVmjp1qiTp8OHDyszM1Mcff6yAgAB16dJFycnJWrBgAQELAABckMcD1sGDBzV37lx99tlnqqqqqrd83759Dd7Hnj17FBERoddff11/%2B9vfVF5eroEDB2ratGnKyspSVFSU2/oRERFau3at63Z2dvY519m/f7/rdlZWltq3b%2B/2jla3bt109OhRnT59WiEhIeetLzc3V3l5eW5jDkegwsLCLrhfrVv7uf2Edzkc3n0emA9O9OEseuFEH5zog5Ov9sHjAWvOnDk6evSoHnnkEV155ZUXdR%2BnTp3SgQMH1L17d61fv14VFRVKS0vTtGnT1KFDBwUEBLit37ZtW5WVlblul5aWXtQ6dbfLysouGLBWr16txYsXu42lpKS43h1rSEhIQMMrocmFhgZ5uwRJzIc69OEseuFEH5zog5Ov9cHjAeuLL77QX/7yF8XExFz0fdRdQ2vmzJm64oorFBwcrNTUVP3yl7/U2LFjVVFR4bZ%2BRUWFgoLO/mMZEBDQ4DqBgYEqLy93W6fu9vfXO5fExETFx8e7jTkcgSosLL3gdq1b%2BykkJECnT5erpqb2guui6TX0fDU15oMTfTiLXjjRByf64NRQH7z1n2WPB6zQ0NAGA0pDIiIiVFtbq6qqKl1xxRWSpNpaZ1NvuOEG/e///q/b%2BtnZ2YqMjHTdjoqK0t69e%2But0717d9ftyMhIFRUV6eTJk%2BrQoYMkKScnR507d27wnbewsLB6Hwfm5RWrurpxB0BNTW2j10XT8ZXngPngRB/OohdO9MGJPjj5Wh88/oFlUlKSFi1apOLi4ou%2Bj379%2BqlLly763e9%2Bp9LSUhUUFOiZZ57RkCFDNHLkSJ08eVLLly9XVVWVduzYoQ0bNuj22293bT969GhlZmZq06ZNqq6u1qZNm5SZmalf/OIXrnXCw8PVs2dPPfHEEyopKdG3336rJUuWaNy4cZe0/wAAoOXz%2BDtYW7Zs0Zdffqm4uDhdffXVbr8yR5I%2B/PDDBu/D399fK1as0Pz58zV06FCdOXNG8fHxmjlzpkJCQrRs2TKlp6crIyNDV111lWbNmqU%2Bffq4tu/WrZteeOEFLVy4UDNnztQ111yj559/vt63GjMyMjR37lzdcsst8vPz05gxY5ScnGynEQAAoMXyeMCKi4tTXFzcJd9Pp06d9Mwzz5xzWY8ePfTaa69dcPuBAwdq4MCBF1ynQ4cOysjIuOgaAQDA5cnjAevBBx/09EMCAAB4lFcuGrF//37NmDFDv/rVr3TixAmtWrVKn376qTdKAQAAsM7jAeurr77S%2BPHj9d133%2Bmrr75SZWWl9u3bp/vuu%2B%2Bcvw8QAACgufF4wFq4cKHuu%2B8%2BrVixQv7%2B/pKkxx9/XHfddVe9i3MCAAA0R155B2vMmDH1xu%2B44w598803ni4HAADAOo8HLH9/f5WUlNQbP3r0aL1fTQMAANAceTxgDRkyRE8//bQKCwtdYzk5OUpPT9fgwYM9XQ4AAIB1Hg9Y06ZNU0VFhfr166fy8nKNHTtWI0eOlMPhUFpamqfLAQAAsM7j18EKDg7Wa6%2B9pu3bt%2Btf//qXamtrFRUVpYEDB8rPzytXjQAAALDK4wGrTt%2B%2BfdW3b19vPTwAAECT8XjAio%2BPV6tWrc67vDG/ixAAAMCXeTxg/fd//7dbwKqqqtLhw4f18ccfKzU11dPlAAAAWOfxgDVlypRzjq9cuVKfffaZ7rrrLg9XBAAAYJfPnFV%2B8803a8uWLd4uAwAA4JL5TMDKzMzUFVdc4e0yAAAALpnHPyL84UeAxhiVlJTowIEDfDwIAABaBI8HrH/7t3%2Br9y1Cf39/3X333Ro1apSnywEAALDO4wFr/vz5nn5IAAAAj/J4wNq5c2ej1%2B3Vq1cTVgIAANA0PB6w7rnnHhljXH/q1H1sWDfWqlUr7du3z9PlAQAAXDKPB6znn39e8%2BbN07Rp09SnTx/5%2B/tr9%2B7dmjNnju68807dfPPNni4JAADAKo9fpuHJJ5/U7NmzNWTIEAUHB%2BuKK65Q7969NXfuXC1btkzXXHON6w8AAEBz5PGAlZubq5/85Cf1xoODg1VYWOjpcgAAAKzzeMCKjo7WokWLVFJS4horKirSggUL1LdvX0%2BXAwAAYJ3Hz8GaNWuW7r77bg0aNEjh4eGSpIMHD6pjx47661//6ulyAAAArPN4wOrWrZs2bdqkDRs2KCcnR5J05513asSIEQoICPB0OQAAANZ5PGBJUkhIiMaPH6/vvvtOXbp0keS8mjsAAEBL4PFzsIwxWrhwoXr16qWRI0fq%2BPHjmjZtmmbMmKGqqipPlwMAAGCdxwPWihUr9NZbb2n27Nlq06aNJGnIkCH6v//7Pz333HOeLgcAAMA6jwes1atX67HHHtPYsWNdV28fPny40tPTtXHjRk%2BXAwAAYJ3HA9Z3332nG264od74ddddp5MnT3q6HAAAAOs8HrCuueYa7dmzp974li1bXCe8AwAANGce/xbhpEmT9Ic//EEnTpyQMUbbt2/Xa6%2B9phUrVmjGjBmeLgcAAMA6jwes22%2B/XdXV1XrxxRdVUVGhxx57TFdffbUeeugh3XHHHZ4uBwAAwDqPB6y///3vuu2225SYmKiCggIZY3T11Vd7ugwAAIAm4/FzsB5//HHXyexXXXUV4QoAALQ4Hg9Y4eHhOnDggKcfFgAAwGM8/hFhZGSkHnnkEf35z39WeHi4rrjiCrfl8%2BbN83RJAAAAVnk8YB05ckQ9e/aUJOXl5Xn64QEAAJqcRwLWvHnz9Nvf/laBgYFasWKFJx4SAADAazxyDtZf//pXlZeXu41NmjRJubm5nnh4AAAAj/JIwDLG1Bv7/PPPdebMGU88PAAAgEd5/FuEAAAALR0BCwAAwDKPBaxWrVp56qEAAAC8ymOXaXj88cfdrnlVVVWlBQsWKCgoyG09roMFAACaO48ErF69etW75lVMTIwKCwtVWFjoiRIAAAA8xiMBi2tfAQCAywknuQMAAFhGwAIAALCs2QesmpoaJSUlafr06d4uBfBZ77zzthITx2jIkAGaNClJX321x9slAUCL1uwD1uLFi7Vr1y5vlwH4rM8/36VnnlmgmTPn6N13/6Fbb71N06f/f6qoqPB2aQDQYjXrgLV9%2B3Zt3rxZt956q7dLAXzW22%2B/pVtuuVU33RQth8OhxMQJateuvT78cLO3SwOAFstj18GyLT8/XzNnztSSJUu0fPlyb5fjJjc3t95lKRyOQIWFhV1wu9at/dx%2BwrscDu8%2BD7bmw6FD32jkyF%2B47U/Xrtfqm2%2ByvL6PjcFxcRa9cKIPTvTByVf70CwDVm1trR599FHde%2B%2B9uv76671dTj2rV6/W4sWL3cZSUlI0derURm0fEhLQFGXhRwoNDWp4JQ%2B41PlQUVGuq69u57Y/ISHBqqmp8pl9bAyOi7PohRN9cKIPTr7Wh2YZsJYuXao2bdooKSnJ26WcU2JiouLj493GHI5AFRaWXnC71q39FBISoNOny1VTU9uUJaIRGnq%2Bmpqt%2BdCmzRUqKDjltj%2BnT5eoc%2BfOXt/HxuC4OIteONEHJ/rg1FAfvPUfyWYZsN566y3l5uYqNjZWklwn637wwQc%2BccJ7WFhYvY8D8/KKVV3duAOgpqa20eui6fjKc3Cp86Fr127Kyclxu4%2BDB79Rnz79fGYfG4Pj4ix64UQfnOiDk6/1oVkGrHfffdftdt0lGubPn%2B%2BNcgCfNmLEaP3ud48qPj5BN90UrXXrXldBQYEGDbrZ26UBQIvVLAMWgMaLje2thx%2BepoUL5ykvL1fh4ddq4cIMhYS083ZpANBitYiAxTtXwIUNHTpcQ4cO93YZAHDZ8K3vNAIAALQABCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWObwdgGArxr27FZvl/CjvJPa39slAAD%2BH97BAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMCyZhuwMjMzdd999%2Bn6669XXFycfvOb36igoECStHv3bo0fP14xMTGKj4/XmjVr3LZdv369EhISFB0drbFjx%2BqLL77wxi4AAIAWqlkGrIqKCiUlJSk0NFS7du3SK6%2B8oh07dmjw4ME6deqUJk%2BerDFjxmjnzp1KT0/XvHnztGfPHknSp59%2Bqj/%2B8Y%2BaP3%2B%2Bdu7cqdGjR%2BuBBx5QeXm5l/cKAAC0FM0yYB09elSSlJaWpuDgYHXv3l2zZ8%2BWv7%2B/Nm/erPbt22vChAlyOBzq27evRo0apVWrVkmS1qxZoxEjRqhnz57y9/fXPffco9DQUG3atMmbuwQAAFqQZhmwvvnmG7Vv316dOnVyjX3wwQcqKSnR559/rqioKLf1IyIitH//fklSdnb2BZcDAABcKoe3C7gYpaWlCggIkCQZY/Tss88qMzNTkuTn5%2BdaVqdt27YqKyurt%2B25ltuQm5urvLw8tzGHI1BhYWEX3K51az%2B3n8CP4XC0zHnDcXEWvXCiD070wclX%2B9AsA1ZgYKDKy8tVUlKiGTNmaO/evVq5cqVGjx6t2tpaVVRUuK1fUVGhoKAgSVJAQMA5l4eGhlqrb/Xq1Vq8eLHbWEpKiqZOndqo7UNCAhpeCfiB0NAgb5fQpDguzqIXTvTBiT44%2BVofmmXAioyMVFFRkUaPHq3//M//1Nq1a7Vjxw517txZMTExevXVV93Wz87OVmRkpGvbrKysessHDRpkrb7ExETFx8e7jTkcgSosLL3gdq1b%2BykkJECnT5erpqbWWj24PDQ0v5orjouz6IUTfXCiD04N9cFb//lslgErNDRUnTt31jXXXKOMjAwVFRVpyZIlGjdunBISErRgwQItX75cEyZM0GeffaYNGzZoyZIlkqRx48YpJSVFw4YNU8%2BePbVq1Srl5%2BcrISHBWn1hYWH1Pg7MyytWdXXjDoCamtpGrwvUaelzhuPiLHrhRB%2Bc6IOTr/WhWQasdevW6fjx4zp%2B/Lji4uJUW1srPz8/vfzyy5oyZYqWLVum9PR0ZWRk6KqrrtKsWbPUp08fSVLfvn01e/ZszZkzRydOnFBERIRefvlltW/f3st7BQAAWopWxhjj7SIuB3l5xQ2u43D4KTQ0SIWFpY1O4cOe3XqppaGFeCe1v7dLaBIXc1y0VPTCiT440QenhvrQseOVXqiqmV6mAQAAwJcRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAAAZ2oq0AAANLUlEQVTLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWObwdgEA7Bj27FZvl9Bo76T293YJANCkeAcLAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAOof8/HwlJycrNjZWcXFxSk9PV3V1tbfLAgAAzQQB6xxSU1MVGBioTz75RGvXrtX27du1fPlyb5cFAACaCQLWDxw%2BfFiZmZl69NFHFRAQoC5duig5OVmrVq3ydmkAAKCZ4EruP5CVlaX27durU6dOrrFu3brp6NGjOn36tEJCQrxYHdAyNKerzktceR7Aj0fA%2BoHS0lIFBAS4jdXdLisra1TAys3NVV5entuYwxGosLCwC27XurWf208AvqG5BcL3Hxno7RKaFK%2BVTvTByVf7QMD6gcDAQJWXl7uN1d0OCgpq1H2sXr1aixcvdht78MEHNWXKlAtul5ubq7/85c9KTExsMIzV2ZV%2BW6PWa05yc3O1evXqH9WHlog%2BONGHs%2BiF08W8VrZE9MHJV/vgW3HPB0RGRqqoqEgnT550jeXk5Khz58668sorG3UfiYmJWrdundufxMTEBrfLy8vT4sWL6737dbmhD070wYk%2BnEUvnOiDE31w8tU%2B8A7WD4SHh6tnz5564oknNHfuXBUWFmrJkiUaN25co%2B8jLCzMp1I0AADwLN7BOoeMjAxVV1frlltu0S9/%2BUsNHDhQycnJ3i4LAAA0E7yDdQ4dOnRQRkaGt8sAAADNVOs5c%2BbM8XYROCsoKEi9e/du9An1LRV9cKIPTvThLHrhRB%2Bc6IOTL/ahlTHGeLsIAACAloRzsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbB8RFlZmWbMmKG4uDj17NlTaWlpKi0t9XZZl2z//v2699571bt3b/Xv319paWkqKCiQJO3evVvjx49XTEyM4uPjtWbNGrdt169fr4SEBEVHR2vs2LH64osvXMtqamr05JNPql%2B/foqJidEDDzyg3Nxcj%2B7bxaipqVFSUpKmT5/uGtuyZYtGjRql6OhoDRs2TB999JHbNi%2B//LIGDRqk6OhoJSUl6ZtvvnEta27zpqioSGlpaYqLi1OvXr2UnJzset4ut/mwd%2B9eTZgwQbGxsRowYIAef/xxVVZWSro85kRBQYESEhL06aefusaacg7k5%2BcrOTlZsbGxiouLU3p6uqqrq5t%2BR8/jQq%2BN0uV3PEjnnhN1cnNz1a9fP61bt85t/FKOhYMHD%2Bruu%2B9WTEyMBgwYoD/96U92d8jAJ0yfPt3cfffdprCw0Jw8edJMnDjRzJkzx9tlXZLy8nLTv39/89xzz5kzZ86YgoICc//995tf//rXpqioyPTu3dusXLnSVFVVmW3btpmYmBize/duY4wxO3bsMDExMWbXrl2msrLSvPrqqyYuLs6UlZUZY4x5/vnnzahRo8zRo0dNcXGxSU1NNffff783d7dRnn32WXP99debadOmGWOMOXjwoOnRo4d5//33TVVVldm4caO56aabzPHjx40xxqxbt84MHDjQfP3116aiosLMmzfPjBgxwtTW1hpjmt%2B8mThxoklJSTGnTp0yxcXF5sEHHzSTJ0%2B%2B7OZDTU2N6d%2B/v/nLX/5iampqzLFjx8zQoUPN4sWLL4s5sWvXLjNkyBATFRVlduzYYYwxTT4HJk6caB5%2B%2BGFTVlZmjhw5YkaMGGFefvllz%2B%2B8ufBrozFN3wtfdK45UaempsYkJSWZ66%2B/3rzxxhuu8Us5FiorK82tt95qFixYYM6cOWP27t1rBgwYYDZt2mRtnwhYPqCsrMz89Kc/NZ999plr7MsvvzQ33XST64BpjnJycsykSZNMdXW1a%2ByDDz4wP//5z83rr79ubr31Vrf1H3vsMZOWlmaMMebhhx82s2bNclt%2B2223mbVr1xpjjBk0aJD5%2B9//7lqWl5dnrrvuOnPkyJGm2p1Ltm3bNjN8%2BHAzdepUV8BatGiRuffee93WmzRpknnuueeMMcb86le/Mi%2B%2B%2BKJrWWVlpYmJiTHbt29vdvPmn//8p%2BnRo4cpLi52jRUWFpqvv/76spsPBQUFJioqyrz66qumurraHDt2zAwbNsy88sorLX5OrFu3zgwePNhs3LjR7R/TppwDhw4dMlFRUa6QaowxGzduNIMHD26SfWzIhV4bjWnaXvii882JOhkZGebRRx81N998s1vAupRjYevWrSY6OtqcOXPGtXzp0qVmwoQJ1vaLjwh9wOHDh1VVVaWoqCjXWLdu3VRRUaFDhw55r7BLdO211%2BrPf/6zWrdu7Rp777339NOf/lRZWVlu%2BytJERER2r9/vyQpOzv7vMuLi4t1/Phxt%2BUdOnRQu3btdODAgSbco4uXn5%2BvmTNn6umnn1ZAQIBr/EL7ea7l/v7%2BCg8P1/79%2B5vdvNmzZ48iIiL0%2BuuvKyEhQQMGDNCTTz6pjh07XnbzITQ0VPfcc4%2BefPJJ9ejRQ//1X/%2Bl8PBw3XPPPS1%2BTgwYMEDvv/%2B%2Bhg8f7jbelHMgKytL7du3V6dOnVzLu3XrpqNHj%2Br06dO2d7FBF3ptlJq2F77ofHNCknbs2KGNGzdq9uzZ9ZZdyrGQlZWlrl27qk2bNq7l3%2B%2BxDQQsH1BSUiJJCgwMdI3V/SPsi%2BdOXAxjjJ555hl99NFHmjlzpkpLS92ChiS1bdtWZWVlknTB5XU9%2BX6/6pb7Yr9qa2v16KOP6t5779X111/vtuxS%2BtDc5s2pU6d04MABHTp0SOvXr9ebb76pEydOaNq0aZfVfJCcc6Jt27b6/e9/ry%2B//FJvv/22cnJylJGR0eLnRMeOHeVwOOqNN%2BUcONe2dbfr7t9bfvjaKDVtL3zR%2BeZEfn6%2Bfve732nhwoUKCgqqt/xSjoXzzQmb84GA5QPqJkB5eblrrO7vwcHBXqnJppKSEk2dOlUbNmzQypUrdd111ykgIEAVFRVu61VUVLgOogstrzsovt%2BvH27vS5YuXao2bdooKSmp3rJL6UNzmzd1/1OcOXOmgoOD1aFDB6WmpmrLli0yxlw280GS3n//fb333nu688471aZNG0VGRiolJUV/%2B9vfLqs58X1N%2BZoQGBhYb1ndbW/OkXO9NkpN24vmwhijtLQ0JSUlqXv37udc51KOhfPNCZs9ImD5gK5du8rf31/Z2dmusZycHNfbnc3ZkSNHdPvtt6ukpERr1651vYBERUUpKyvLbd3s7GxFRkZKkiIjI8%2B7vF27durUqZNbv/Ly8lRUVFTvbXNf8NZbbykzM1OxsbGKjY3V22%2B/rbfffluxsbE/ug9VVVU6dOiQoqKimt28iYiIUG1traqqqlxjtbW1kqQbbrjhspkPknTs2DHXNwbrOBwO%2Bfv7X1Zz4vua8jUhMjJSRUVFOnnypGt5Tk6OOnfurCuvvLIJ9%2Br8zvfaKF1er4/nc%2BzYMWVmZuqFF15wvXYePXpUf/jDH/TrX/9a0qUdC5GRkTp06JDbN0m/32MrrJ3NhUvyyCOPmIkTJ5r8/HyTn59vJk6c6DoRurkqKioygwcPNtOnTzc1NTVuywoKCkxsbKx59dVXTWVlpdm%2Bfbvr5ERjjOtbM9u3b3d9S6ZXr16msLDQGGPMM888Y0aOHGmOHDni%2BpbMxIkTPb6PF2PatGmu5zY7O9v06NHDbNy40fWNsR49ephvvvnGGOM82XXgwIFm3759rm/JJCQkmMrKSmNM85o3lZWVJiEhwUyZMsWUlJSY/Px8c9ddd5mUlJTLbj5kZWWZ7t27mxdffNFUV1ebI0eOmJEjR5r58%2BdfVnPi%2Byc0N/UcuOOOO8xDDz1kiouLXd8izMjI8PxOmwu/Nhpzeb8%2Bnusk9zo/PMn9Uo6FqqoqEx8fb%2BbPn28qKirMvn37zIABA9zu/1IRsHxEcXGxmTVrlunXr5/p1auXmT59uiktLfV2WZdk2bJlJioqyvzsZz8z0dHRbn%2BMMWbPnj0mMTHRxMTEmFtuuaXexH7zzTfN0KFDTXR0tBk3bpz58ssvXcsqKyvNggULzMCBA83Pf/5z88ADD5iTJ096dP8u1vcDljHGfPzxx2b06NEmOjrajBgxwvzjH/9wLautrTWvvPKKiY%2BPN9HR0SYpKcn1D60xzW/eHD9%2B3KSmppr%2B/fub2NhYk5aWZk6dOmWMufzmw9atW8348eNNz549zeDBg82iRYtc32i6XObED/8xbco5kJeXZ6ZMmWJ69%2B5t%2BvTpY%2BbPn%2B/2LT5Paui10ZjL73io82MC1qUeC4cOHTL33Xef6dmzpxk4cKBZunSp1X1pZYwx9t4PAwAAAOdgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACW/f9%2B19BK1I2U1gAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common877366547351027798">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1300</td>
        <td class="number">138</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1400</td>
        <td class="number">135</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1440</td>
        <td class="number">133</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1660</td>
        <td class="number">129</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1010</td>
        <td class="number">129</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1800</td>
        <td class="number">129</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1820</td>
        <td class="number">128</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1480</td>
        <td class="number">125</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1720</td>
        <td class="number">125</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1540</td>
        <td class="number">124</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (1024)</td>
        <td class="number">20302</td>
        <td class="number">94.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme877366547351027798">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">370</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">380</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">390</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">410</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">420</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">9640</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9890</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">10040</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">12050</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">13540</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_sqft_living15">sqft_living15<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>777</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>3.6%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>1986.6</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>399</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>6210</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-6035771469071839640">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAAS5JREFUeJzt3bGNwkAUQMHjREkUQU8X0xNF0NOSEqAnG8l44WZyS5s8fa%2B9sg9jjPEDPPW79wJgZse9F7CX09919TW3y3mDlTAzEwSCQCAIBIJAIAgEgkAgfMVj3lce2cISJggEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAiErzjN%2By5rTw37yMPnM0EgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgsOKG/IXq89ngkAQCASBQBAIBIFAEAgEgUDwHmQy3p3MxQSBIBAIhzHG2HsRj/yxdntuyZazB/mH7HOWm26CwEzsQSAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCDcAZ2PH/vaJ7uIAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-6035771469071839640,#minihistogram-6035771469071839640"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-6035771469071839640">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-6035771469071839640"
                                                  aria-controls="quantiles-6035771469071839640" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-6035771469071839640" aria-controls="histogram-6035771469071839640"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-6035771469071839640" aria-controls="common-6035771469071839640"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-6035771469071839640" aria-controls="extreme-6035771469071839640"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-6035771469071839640">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>399</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>1140</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>1490</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>1840</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>2360</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>3300</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>6210</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>5811</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>870</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>685.23</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.34492</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>1.5917</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>1986.6</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>536.16</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>1.1069</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>42905039</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>469540</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-6035771469071839640">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3Xt0VOW9//EPyQQySQwJQqh16YKSxCualEC4aNRIxAChNgRzFPFSCz0YibiAAAaFiuFSInIrLIoXWuAsI0qqaFS0h4IHMVxEaK2hCYpIuWQkCeY25Pb8/uiP0SFSBt3MZML7tVZWyvPs2fu7v2t2/czee3Y6GGOMAAAAYJkAXxcAAADQ3hCwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsJjN1wVcLByOal%2BX0CYEBHRQly6hqqioVUuL8XU5bRq98hy98hy98hy98lxb7lW3bpf4ZLucwYJXBQR0UIcOHRQQ0MHXpbR59Mpz9Mpz9Mpz9Mpz9Ko1AhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFiMgAUAAGAxAhYAAIDFbL4uAGirUhdt83UJ5%2BXtiYN8XQIA4P/jDBYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFiMgAUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFvPLgPXGG28oPj7e7ef666/X9ddfL0nasmWL0tLSFBcXp9TUVG3evNnt9atWrVJSUpLi4uI0ZswYff755665uro6TZ8%2BXYmJierTp49ycnJUW1vr1f0DAAD%2BzS8D1ogRI7Rnzx7XzzvvvKOIiAjl5eXp4MGDmjBhgh577DHt2rVLEyZM0MSJE3X8%2BHFJUmFhodasWaMXXnhBxcXFuu6665SdnS1jjCRp9uzZOnr0qN59911t2rRJR48eVX5%2Bvi93FwAA%2BBm/DFjfZYzRlClTdOutt%2BoXv/iFCgsLlZCQoMGDB8tms2no0KHq27evCgoKJEmvvPKK7r33XsXExKhTp06aNGmSjhw5ouLiYtXX12vjxo3Kzs5WRESELr30Uk2ePFkbNmxQfX29j/cUAAD4C78PWK%2B//rrKyso0bdo0SVJZWZliY2PdlomOjlZJScn3zgcFBalHjx4qKSnRl19%2BqcbGRrf5Xr16yel06uDBgxd%2BZwAAQLtg83UBP0ZLS4tWrFih//7v/1ZYWJgkqba2Vna73W254OBg1dXVnXO%2BpqZGkhQSEuKaO73s%2BdyHVV5eLofD4TZms4UoKirK43W0V4GBAW6/YR2b7eLtKe8rz9Erz9Erz9Gr1vw6YBUXF6u8vFwZGRmuMbvdLqfT6bac0%2BlUaGjoOedPB6v6%2BnrX8qcvDZ4OcJ4oKCjQsmXL3MaysrKUnZ3t8Trau/Bw%2B7kXwnmJjAz1dQk%2Bx/vKc/TKc/TKc/TqW34dsN59912lpKS4nXGKjY3Vp59%2B6rZcWVmZ6xuGMTExKi0t1W233SZJamxs1MGDBxUbG6uePXsqKChIZWVluvHGGyVJBw4ccF1G9FRmZqaSk5Pdxmy2EFVW8m3EwMAAhYfb9c039WpubvF1Oe3Kxfz%2B4n3lOXrlOXrlubbcK199%2BPTrgLV7927df//9bmMjRozQSy%2B9pKKiIt1xxx3atGmTduzYodzcXEnSyJEjtXTpUiUlJalnz5567rnn1LVrVyUkJCgoKEipqanKz8/X4sWLJUn5%2BfkaPny4goODPa4rKiqq1eVAh6NaTU1t603nS83NLfTDYvST99X5oFeeo1eeo1ff8uuAdfjw4VZBplevXvr973%2Bv/Px85ebm6vLLL9fSpUvVs2dPSVJGRoaqq6uVlZWliooK9e7dWytXrlRQUJAkaebMmZo/f77S0tLU2Nio22%2B/XU8%2B%2BaTX9w0AAPivDub0A6BwQTkc1b4uoU2w2QIUGRmqysraNv8pJ3XRNl%2BXcF7enjjI1yX4jD%2B9r3yNXnmOXnmuLfeqW7dLfLJdbvcHAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsJjN1wXg4pG6aJuvSwAAwCs4gwUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFjMbwNWVVWVcnJylJiYqL59%2B%2BqRRx5ReXm5JGnv3r0aNWqU4uPjlZycrPXr17u9trCwUCkpKYqLi1N6err27Nnjmmtubtb8%2BfM1cOBAxcfHa/z48a71AgAAeMJvA9aECRNUV1en9957T5s3b1ZgYKCefPJJnTx5UuPGjdNdd92lnTt3Ki8vT3PnztW%2BffskScXFxZo9e7bmzZunnTt3asSIERo/frzq6%2BslSStWrNC2bdv02muv6YMPPlBwcLBmzJjhy10FAAB%2Bxi8D1t///nft3btX8%2BbNU3h4uMLCwjR79mxNnjxZmzZtUkREhEaPHi2bzaYBAwYoLS1N69atkyStX79ew4YNU58%2BfRQUFKQHH3xQkZGRKioqcs2PHTtWl112mcLCwpSbm6utW7fqq6%2B%2B8uUuAwAAP%2BKXAWvfvn2Kjo7WK6%2B8opSUFN10002aP3%2B%2BunXrptLSUsXGxrotHx0drZKSEklSWVnZWeerq6t17Ngxt/muXbuqc%2BfO2r9//4XfMQAA0C7YfF3AD3Hy5Ent379f119/vQoLC%2BV0OpWTk6OpU6eqa9eustvtbssHBwerrq5OklRbW3vW%2BdraWklSSEhIq/nTc54oLy%2BXw%2BFwG7PZQhQVFeXxOoDzZbP55eclSwQGBrj9xtnRK8/RK8/Rq9b8MmB17NhRkpSbm6tOnTopLCxMEydO1N1336309HQ5nU635Z1Op0JDQyVJdrv9e%2BcjIyNdwev0/Vjf93pPFBQUaNmyZW5jWVlZys7O9ngdwPmKjPT8PdpehYfbz70QJNGr80GvPEevvuWXASs6OlotLS1qbGxUp06dJEktLS2SpGuuuUb/8z//47Z8WVmZYmJiJEkxMTEqLS1tNZ%2BUlKTOnTure/fubpcRHQ6HqqqqWl1W/E8yMzOVnJzsNmazhaiy0vOzYMD5upjfX4GBAQoPt%2Bubb%2BrV3Nzi63LaNHrlOXrlubbcK199%2BPTLgDVw4EBdccUVeuKJJzR37lydOnVKzz33nAYPHqzhw4dryZIlWr16tUaPHq3du3dr48aNWr58uSQpIyNDWVlZSk1NVZ8%2BfbRu3TqdOHFCKSkpkqT09HStWLFCvXv3VmRkpObMmaN%2B/frpyiuv9Li%2BqKioVpcDHY5qNTW1rTcd2hfeX1Jzcwt98BC98hy98hy9%2BpZfBqygoCCtWbNG8%2BbN05AhQ3Tq1CklJycrNzdX4eHhevHFF5WXl6clS5aoS5cumjFjhvr37y9JGjBggGbOnKlZs2bp%2BPHjio6O1qpVqxQRESHp35fympqaNHr0aNXW1ioxMVGLFi3y5e4CAAA/08EYY3xdxMXA4aj2dQk%2Bl7pom69LaNfenjjI1yX4jM0WoMjIUFVW1vLp%2BRzolefolefacq%2B6dbvEJ9vldn8AAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsJjfBqyioiJde%2B21io%2BPd/1MmTJFkrRlyxalpaUpLi5Oqamp2rx5s9trV61apaSkJMXFxWnMmDH6/PPPXXN1dXWaPn26EhMT1adPH%2BXk5Ki2ttar%2BwYAAPyb3wasv/3tb/rFL36hPXv2uH4WLFiggwcPasKECXrssce0a9cuTZgwQRMnTtTx48clSYWFhVqzZo1eeOEFFRcX67rrrlN2draMMZKk2bNn6%2BjRo3r33Xe1adMmHT16VPn5%2Bb7cVQAA4Gf8OmBdf/31rcYLCwuVkJCgwYMHy2azaejQoerbt68KCgokSa%2B88oruvfdexcTEqFOnTpo0aZKOHDmi4uJi1dfXa%2BPGjcrOzlZERIQuvfRSTZ48WRs2bFB9fb23dxEAAPgpm68L%2BCFaWlr06aefym636/nnn1dzc7NuueUWTZ48WWVlZYqNjXVbPjo6WiUlJZKksrIyjR071jUXFBSkHj16qKSkRBEREWpsbHR7fa9eveR0OnXw4EFdc801HtVXXl4uh8PhNmazhSgqKuqH7jJwTjab335e%2BtECAwPcfuPs6JXn6JXn6FVrfhmwKioqdO2112rIkCFasmSJKisrNXXqVE2ZMkUNDQ2y2%2B1uywcHB6uurk6SVFtbe9b5mpoaSVJISIhr7vSy53MfVkFBgZYtW%2BY2lpWVpezsbM93EjhPkZGhvi7B58LD7edeCJLo1fmgV56jV9/yy4DVtWtXrVu3zvVvu92uKVOm6O6771ZiYqKcTqfb8k6nU6Ghoa5lzzZ/OljV19e7lj99aTAsLMzj%2BjIzM5WcnOw2ZrOFqLKSm%2BVx4VzM76/AwACFh9v1zTf1am5u8XU5bRq98hy98lxb7pWvPnz6ZcAqKSnRm2%2B%2BqUmTJqlDhw6SpIaGBgUEBOiGG27QZ5995rZ8WVmZ636tmJgYlZaW6rbbbpMkNTY26uDBg4qNjVXPnj0VFBSksrIy3XjjjZKkAwcOuC4jeioqKqrV5UCHo1pNTW3rTYf2hfeX1NzcQh88RK88R688R6%2B%2B5ZcXSyMiIrRu3To9//zzampq0pEjR7RgwQL98pe/1F133aUdO3aoqKhITU1NKioq0o4dO/SLX/xCkjRy5EitXbtWJSUlOnXqlJ599ll17dpVCQkJstvtSk1NVX5%2BvioqKlRRUaH8/HwNHz5cwcHBPt5rAADgL/zyDNZPfvITrVy5UgsXLtSKFSvUqVMnDRs2TFOmTFGnTp30%2B9//Xvn5%2BcrNzdXll1%2BupUuXqmfPnpKkjIwMVVdXKysrSxUVFerdu7dWrlypoKAgSdLMmTM1f/58paWlqbGxUbfffruefPJJX%2B4uAADwMx3M6QdA4YJyOKp9XYLPpS7a5usS2rW3Jw7ydQk%2BY7MFKDIyVJWVtVyeOAd65Tl65bm23Ktu3S7xyXb98hIhAABAW%2Bb1gNXc3OztTQIAAHiV1wNWUlKSfve736msrMzbmwYAAPAKrwesRx99VB9//LGGDx%2BuUaNG6eWXX1Z1NfcnAQCA9sPrAeuee%2B7Ryy%2B/rHfeeUcDBw7UqlWrdNNNN2nSpEn68MMPvV0OAACA5Xx2k3uPHj30%2BOOP65133lFWVpb%2B8pe/6OGHH1ZycrJeeukl7tUCAAB%2By2fPwdq7d6/%2B/Oc/q6ioSA0NDUpJSVF6erqOHz%2BuxYsX629/%2B5sWLlzoq/IAAAB%2BMK8HrOXLl%2Bv111/Xl19%2Bqd69e%2Bvxxx/X8OHD3f7WX2BgoJ566ilvlwYAAGAJrwestWvXasSIEcrIyFB0dPT3LtOrVy9NnjzZy5UBAABYw%2BsBa%2BvWraqpqVFVVZVrrKioSAMGDFBkZKQk6dprr9W1117r7dIAAAAs4fWb3P/xj39oyJAhKigocI0tWLBAaWlp%2Buc//%2BntcgAAACzn9YD1u9/9TnfccYcef/xx19j777%2BvpKQkzZs3z9vlAAAAWM7rAevTTz/VuHHj1LFjR9dYYGCgxo0bp08%2B%2BcTb5QAAAFjO6wErLCxMhw4dajV%2B7NgxBQcHe7scAAAAy3k9YA0ZMkSzZs3Shx9%2BqJqaGtXW1uqjjz7S008/rZSUFG%2BXAwAAYDmvf4tw0qRJ%2Buqrr/SrX/1KHTp0cI2npKQoJyfH2%2BUAAABYzusBy263a%2BXKlfriiy%2B0f/9%2BBQUFqVevXurRo4e3SwEAALggfPancnr27KmePXv6avMAAAAXjNcD1hdffKGnn35au3fvVmNjY6v5zz77zNslAQAAWMrrAWvWrFk6cuSIJk%2BerEsuucTbmwcAALjgvB6w9uzZoz/%2B8Y%2BKj4/39qYBAAC8wuuPaYiMjFRoaKi3NwsAAOA1Xg9YY8aM0cKFC1VdXe3tTQMAAHiF1y8RbtmyRZ988okSExN16aWXuv3JHEn6y1/%2B4u2SAAAALOX1gJWYmKjExERvbxYAAMBrvB6wHn30UW9vEgAAwKt88qDRkpIS/fGPf9QXX3yhxYsX6/3331d0dDRntoAfIXXRNl%2BX4LG3Jw7ydQkAcEF5/Sb3v//97xo1apQOHz6sv//972poaNBnn32mX/3qV9q8ebO3ywEAALCc1wNWfn6%2BfvWrX2nNmjUKCgqSJD3zzDO6//77tWzZMm%2BXAwAAYDmfnMG66667Wo3fc889%2Bvzzz71dDgAAgOW8HrCCgoJUU1PTavzIkSOy2%2B3nvb7m5maNGTNG06ZNc41t2bJFaWlpiouLU2pqaqtLj6tWrVJSUpLi4uI0ZswYt2BXV1en6dOnKzExUX369FFOTo5qa2vPuy4AAHDx8nrAGjx4sJ599llVVla6xg4cOKC8vDzdeuut572%2BZcuWadeuXa5/Hzx4UBMmTNBjjz2mXbt2acKECZo4caKOHz8uSSosLNSaNWv0wgsvqLi4WNddd52ys7NljJEkzZ49W0ePHtW7776rTZs26ejRo8rPz/9xOw0AAC4qXg9YU6dOldPp1MCBA1VfX6/09HQNHz5cNptNOTk557Wu7du3a9OmTbrjjjtcY4WFhUpISNDgwYNls9k0dOhQ9e3bVwUFBZKkV155Rffee69iYmLUqVMnTZo0SUeOHFFxcbHq6%2Bu1ceNGZWdnKyIiQpdeeqkmT56sDRs2qL6%2B3tI%2BAACA9svrj2kICwvTyy%2B/rO3bt%2Bsf//iHWlpaFBsbq5tvvlkBAZ7nvRMnTig3N1fLly/X6tWrXeNlZWWKjY11WzY6OlolJSWu%2BbFjx7rmgoKC1KNHD5WUlCgiIkKNjY1ur%2B/Vq5ecTqcOHjyoa665xqPaysvL5XA43MZsthBFRUV5vH9Ae2azWfvZLjAwwO03zo5eeY5eeY5eteaT52BJ0oABAzRgwIAf9NqWlhZNmTJFDz30kK6%2B%2Bmq3udra2lb3cgUHB6uuru6c86fvDQsJCXHNnV72fO7DKigoaPWNyKysLGVnZ3u8DqA9i4y8MH/wPTz8/O/jvFjRK8/RK8/Rq295PWAlJyerQ4cOZ5335G8Rrly5Uh07dtSYMWNazdntdjmdTrcxp9Op0NDQc86fDlb19fWu5U9fGgwLCztnXadlZmYqOTnZbcxmC1FlJTfLA5IsPxYCAwMUHm7XN9/Uq7m5xdJ1tzf0ynP0ynNtuVcX6gPduXg9YP3yl790C1iNjY368ssvtXXrVk2cONGjdbz%2B%2BusqLy9XQkKCJLkC0/vvv6/Ro0fr008/dVu%2BrKxM119/vSQpJiZGpaWluu2221zbP3jwoGJjY9WzZ08FBQWprKxMN954o6R/34B/%2BjKip6KiolpdDnQ4qtXU1LbedICvXKhjobm5hePMQ/TKc/TKc/TqW14PWBMmTPje8bVr12r37t26//77z7mOd955x%2B3fpx/RMG/ePB04cEAvvfSSioqKdMcdd2jTpk3asWOHcnNzJUkjR47U0qVLlZSUpJ49e%2Bq5555T165dlZCQoKCgIKWmpio/P1%2BLFy%2BW9O8How4fPlzBwcE/ZrcBAMBFxGf3YJ3ptttu08KFC3/0enr16qXf//73ys/PV25uri6//HItXbpUPXv2lCRlZGSourpaWVlZqqioUO/evbVy5UrXU%2BVnzpyp%2BfPnKy0tTY2Njbr99tv15JNP/ui6AADAxaODOf0AKB8rLCzU7373O23fvt3XpVwQDke1r0vwOX/6Y8S4sKz%2BY882W4AiI0NVWVnL5YlzoFeeo1eea8u96tbtEp9s1%2BtnsM68BGiMUU1Njfbv3%2B/R5UEAAIC2zusB66c//WmrbxEGBQXpgQceUFpamrfLAQAAsJzXA9a8efO8vUkAAACv8nrA2rlzp8fL9u3b9wJWAgAAcGF4PWA9%2BOCDMsa4fk47fdnw9FiHDh302Wefebs8AACAH83rAWvp0qWaO3eupk6dqv79%2BysoKEh79%2B7VrFmzdO%2B997oeAAoAAOCvvP5XGefPn6%2BZM2dq8ODBCgsLU6dOndSvXz89/fTTevHFF3X55Ze7fgAAAPyR1wNWeXm5LrvsslbjYWFhqqys9HY5AAAAlvN6wIqLi9PChQtVU1PjGquqqtKCBQs0YMAAb5cDAABgOa/fgzVjxgw98MADSkpKcv0B5S%2B%2B%2BELdunXTn/70J2%2BXAwAAYDmvB6xevXqpqKhIGzdu1IEDByRJ9957r4YNGya73e7tcgAAACznkz/2HB4erlGjRunw4cO64oorJMn1x5YBAAD8ndfvwTLGKD8/X3379tXw4cN17NgxTZ06VdOnT1djY6O3ywEAALCc1wPWmjVr9Prrr2vmzJnq2LGjJGnw4MH63//9Xy1evNjb5QAAAFjO6wGroKBATz31lNLT011Pbx86dKjy8vL01ltvebscAAAAy3k9YB0%2BfFjXXHNNq/GrrrpKX3/9tbfLAQAAsJzXA9bll1%2Buffv2tRrfsmWL64Z3AAAAf%2Bb1bxE%2B/PDD%2Bu1vf6vjx4/LGKPt27fr5Zdf1po1azR9%2BnRvlwMAAGA5rweskSNHqqmpSStWrJDT6dRTTz2lSy%2B9VI8//rjuueceb5cDAABgOa8HrDfeeEN33nmnMjMzVVFRIWOMLr30Um%2BXAQAAcMF4/R6sZ555xnUze5cuXQhXAACg3fF6wOrRo4f279/v7c0CAAB4jdcvEcbExGjy5Ml6/vnn1aNHD3Xq1Mltfu7cud4uCQAAwFJeD1iHDh1Snz59JEkOh8PbmwcAALjgvBKw5s6dq8cee0whISFas2aNNzYJAADgM165B%2BtPf/qT6uvr3cYefvhhlZeXe2PzAAAAXuWVgGWMaTX28ccf69SpU97YPAAAgFd5/VuEAAAA7R0BCwAAwGJeC1gdOnTw1qYAAAB8ymuPaXjmmWfcnnnV2NioBQsWKDQ01G05T5%2BDtX37di1cuFAHDhyQ3W7XnXfeqSlTpig4OFh79%2B7VM888o7KyMkVGRmr8%2BPEaNWqU67WFhYVavny5HA6Hfvazn%2BnJJ59UfHy8JKm5uVn5%2Bfl6/fXXVV9fr/79%2B%2Bu3v/2toqKiLOgCAAC4GHjlDFbfvn3lcDh0%2BPBh1098fLwqKyvdxg4fPuzR%2BioqKvSb3/xG99xzj3bt2qXCwkLt2LFDf/jDH3Ty5EmNGzdOd911l3bu3Km8vDzNnTtX%2B/btkyQVFxdr9uzZmjdvnnbu3KkRI0Zo/Pjxrm85rlixQtu2bdNrr72mDz74QMHBwZoxY8YF6w0AAGh/vHIGy%2BpnX3Xp0kUffvihwsLCZIxRVVWVTp06pS5dumjTpk2KiIjQ6NGjJUkDBgxQWlqa1q1bpxtuuEHr16/XsGHDXA87ffDBB1VQUKCioiKNHDlS69ev1%2BTJk3XZZZdJknJzc3XTTTfpq6%2B%2B0hVXXGHpfgAAgPbJ609yt0pYWJgk6ZZbbtHx48eVkJCg9PR0LVq0SLGxsW7LRkdH69VXX5UklZWVaeTIka3mS0pKVF1drWPHjrm9vmvXrurcubP279/vccAqLy9v9ZR6my2Ey4zA/2ezWXvyPDAwwO03zo5eeY5eeY5etea3Aeu0TZs26eTJk5o8ebKys7PVvXt32e12t2WCg4NVV1cnSaqtrT3rfG1trSQpJCSk1fzpOU8UFBRo2bJlbmNZWVnKzs72eB1AexYZGXruhX6A8HD7uReCJHp1PuiV5%2BjVt/w%2BYAUHBys4OFhTpkzRqFGjNGbMGFVXV7st43Q6XTfT2%2B12OZ3OVvORkZGu4HXmU%2Be/%2B3pPZGZmKjk52W3MZgtRZaXnIQ1oz6w%2BFgIDAxQebtc339SrubnF0nW3N/TKc/TKc225VxfqA925%2BGXA%2Bvjjj/XEE0/ojTfeUMeOHSVJDQ0NCgoKUnR0tLZt2%2Ba2fFlZmWJiYiRJMTExKi0tbTWflJSkzp07q3v37iorK3NdJnQ4HKqqqmp12fE/iYqKanU50OGoVlNT23rTAb5yoY6F5uYWjjMP0SvP0SvP0atv%2BeXF0quuukpOp1PPPvusGhoa9K9//Uvz589XRkaGhgwZoq%2B//lqrV69WY2OjPvroI23cuNF131VGRoY2btyojz76SI2NjVq9erVOnDihlJQUSVJ6erpWrFihr776SjU1NZozZ4769eunK6%2B80pe7DAAA/IhfnsEKDQ3V888/rzlz5mjQoEG65JJLlJaWpqysLHXs2FEvvvii8vLytGTJEnXp0kUzZsxQ//79Jf37W4UzZ87UrFmzdPz4cUVHR2vVqlWKiIiQ9O97pZqamjR69GjV1tYqMTFRixYt8uXuAgAAP9PBfN9fYoblHI7qcy/UzqUu2nbuhXBReHviIEvXZ7MFKDIyVJWVtVyeOAd65Tl65bm23Ktu3S7xyXb98hIhAABAW0bAAgAAsBgBCwAAwGIELABTHFtfAAAUaklEQVQAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALCY3waskpISPfTQQ%2BrXr58GDRqknJwcVVRUSJL27t2rUaNGKT4%2BXsnJyVq/fr3bawsLC5WSkqK4uDilp6drz549rrnm5mbNnz9fAwcOVHx8vMaPH6/y8nKv7hsAAPBvfhmwnE6nfv3rXys%2BPl7/93//pzfffFNVVVV64okndPLkSY0bN0533XWXdu7cqby8PM2dO1f79u2TJBUXF2v27NmaN2%2Bedu7cqREjRmj8%2BPGqr6%2BXJK1YsULbtm3Ta6%2B9pg8%2B%2BEDBwcGaMWOGL3cXAAD4Gb8MWEeOHNHVV1%2BtrKwsdezYUZGRkcrMzNTOnTu1adMmRUREaPTo0bLZbBowYIDS0tK0bt06SdL69es1bNgw9enTR0FBQXrwwQcVGRmpoqIi1/zYsWN12WWXKSwsTLm5udq6dau%2B%2BuorX%2B4yAADwI34ZsH72s5/p%2BeefV2BgoGvs3Xff1XXXXafS0lLFxsa6LR8dHa2SkhJJUllZ2Vnnq6urdezYMbf5rl27qnPnztq/f/8F3CMAANCe2HxdwI9ljNGiRYu0efNmrV27Vn/6059kt9vdlgkODlZdXZ0kqba29qzztbW1kqSQkJBW86fnPFFeXi6Hw%2BE2ZrOFKCoqyuN1AO1Z6qJtvi7hvLw3%2BWZfl2CZwMAAt984O3rlOXrVml8HrJqaGk2fPl2ffvqp1q5dq6uuukp2u13V1dVuyzmdToWGhkqS7Ha7nE5nq/nIyEhX8Dp9P9b3vd4TBQUFWrZsmdtYVlaWsrOzPV4HgLYjMtLz499fhIfbz70QJNGr80GvvuW3AevQoUMaO3asfvrTn%2BrVV19Vly5dJEmxsbHats3903FZWZliYmIkSTExMSotLW01n5SUpM6dO6t79%2B5ulxEdDoeqqqpaXVb8TzIzM5WcnOw2ZrOFqLLS87NgANqO9nTsBgYGKDzcrm%2B%2BqVdzc4uvy2nT6JXn2nKvfPUByS8D1smTJ/XAAw%2Bof//%2BysvLU0DAt6ckU1JStGDBAq1evVqjR4/W7t27tXHjRi1fvlySlJGRoaysLKWmpqpPnz5at26dTpw4oZSUFElSenq6VqxYod69eysyMlJz5sxRv379dOWVV3pcX1RUVKvLgQ5HtZqa2tabDoBn2uOx29zc0i7360KgV56jV9/yy4C1YcMGHTlyRG%2B//bbeeecdt7k9e/boxRdfVF5enpYsWaIuXbpoxowZ6t%2B/vyRpwIABmjlzpmbNmqXjx48rOjpaq1atUkREhKR/X8pramrS6NGjVVtbq8TERC1atMjr%2BwgAAPxXB2OM8XURFwOHo/rcC7Vz/nZjM3Da2xMH%2BboEy9hsAYqMDFVlZS1nGs6BXnmuLfeqW7dLfLJdbvcHAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGJ%2BH7AqKiqUkpKi4uJi19jevXs1atQoxcfHKzk5WevXr3d7TWFhoVJSUhQXF6f09HTt2bPHNdfc3Kz58%2Bdr4MCBio%2BP1/jx41VeXu61/QEAAP7PrwPW7t27lZmZqUOHDrnGTp48qXHjxumuu%2B7Szp07lZeXp7lz52rfvn2SpOLiYs2ePVvz5s3Tzp07NWLECI0fP1719fWSpBUrVmjbtm167bXX9MEHHyg4OFgzZszwyf4BAAD/5LcBq7CwUJMnT9bjjz/uNr5p0yZFRERo9OjRstlsGjBggNLS0rRu3TpJ0vr16zVs2DD16dNHQUFBevDBBxUZGamioiLX/NixY3XZZZcpLCxMubm52rp1q7766iuv7yMAAPBPfhuwbrrpJr333nsaOnSo23hpaaliY2PdxqKjo1VSUiJJKisrO%2Bt8dXW1jh075jbftWtXde7cWfv3779AewIAANobm68L%2BKG6dev2veO1tbWy2%2B1uY8HBwaqrqzvnfG1trSQpJCSk1fzpOU%2BUl5fL4XC4jdlsIYqKivJ4HQDaDpvNbz%2BLthIYGOD2G2dHrzxHr1rz24B1Nna7XdXV1W5jTqdToaGhrnmn09lqPjIy0hW8Tt%2BP9X2v90RBQYGWLVvmNpaVlaXs7GyP1wGg7YiM9Pz49xfh4fZzLwRJ9Op80KtvtbuAFRsbq23btrmNlZWVKSYmRpIUExOj0tLSVvNJSUnq3Lmzunfv7nYZ0eFwqKqqqtVlxf8kMzNTycnJbmM2W4gqKz0/Cwag7WhPx25gYIDCw%2B365pt6NTe3%2BLqcNo1eea4t98pXH5DaXcBKSUnRggULtHr1ao0ePVq7d%2B/Wxo0btXz5cklSRkaGsrKylJqaqj59%2BmjdunU6ceKEUlJSJEnp6elasWKFevfurcjISM2ZM0f9%2BvXTlVde6XENUVFRrS4HOhzVampqW286AJ5pj8duc3NLu9yvC4FeeY5efavdBazIyEi9%2BOKLysvL05IlS9SlSxfNmDFD/fv3lyQNGDBAM2fO1KxZs3T8%2BHFFR0dr1apVioiIkPTvS3lNTU0aPXq0amtrlZiYqEWLFvlylwAAgJ/pYIwxvi7iYuBwVJ97oXYuddG2cy8EtEFvTxzk6xIsY7MFKDIyVJWVtZxpOAd65bm23Ktu3S7xyXa53R8AAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAi7W752ABgNX87REj7emxEoC/4gwWAACAxQhYAAAAFiNgAQAAWIyABQAAYDECFgAAgMUIWAAAABYjYAEAAFiMgAUAAGAxAhYAAIDFCFgAAAAWI2ABAABYjIAFAABgMQIWAACAxQhYAAAAFiNgAQAAWMzm6wLw46Qu2ubrEgAAwBk4gwUAAGAxAhYAAIDFCFgAAAAW4x4sAGhn/OnezLcnDvJ1CcAFwRksAAAAixGwAAAALEbAAgAAsBgB63ucOHFCjzzyiBISEpSYmKi8vDw1NTX5uiwAAOAnCFjfY%2BLEiQoJCdEHH3ygV199Vdu3b9fq1at9XRYAAPATfIvwDF9%2B%2BaV27NihrVu3ym6364orrtAjjzyiBQsW6Ne//rWvywOAdsWfvvEo8a1HeI4zWGcoLS1VRESEunfv7hrr1auXjhw5om%2B%2B%2BcaHlQEAAH/BGawz1NbWym63u42d/nddXZ3Cw8PPuY7y8nI5HA63MZstRFFRUdYVCgDwOpuN8xLfJzAwwO03CFithISEqL6%2B3m3s9L9DQ0M9WkdBQYGWLVvmNvboo49qwoQJ1hT5Hbvy7rR8nRdSeXm5CgoKlJmZSeA8B3rlOXrlOXrlOXrlufLycv3xj8/Tq%2B8gap4hJiZGVVVV%2Bvrrr11jBw4c0E9%2B8hNdcsklHq0jMzNTGzZscPvJzMy8UCX7FYfDoWXLlrU6w4fW6JXn6JXn6JXn6JXn6FVrnME6Q48ePdSnTx/NmTNHTz/9tCorK7V8%2BXJlZGR4vI6oqCgSPAAAFzHOYH2PJUuWqKmpSbfffrvuvvtu3XzzzXrkkUd8XRYAAPATnMH6Hl27dtWSJUt8XQYAAPBTgbNmzZrl6yJwcQkNDVW/fv08/tLAxYxeeY5eeY5eeY5eeY5euetgjDG%2BLgIAAKA94R4sAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCz8aBUVFUpJSVFxcbFrbO/evRo1apTi4%2BOVnJys9evXu72msLBQKSkpiouLU3p6uvbs2eOaa25u1vz58zVw4EDFx8dr/PjxKi8v99r%2BXAglJSV66KGH1K9fPw0aNEg5OTmqqKiQRK/OtH37do0aNUo///nPNWjQIM2ePVtOp1MSvTqb5uZmjRkzRtOmTXONbdmyRWlpaYqLi1Nqaqo2b97s9ppVq1YpKSlJcXFxGjNmjD7//HPXXF1dnaZPn67ExET16dNHOTk5qq2t9dr%2BXAhFRUW69tprFR8f7/qZMmWKJHp1pqqqKuXk5CgxMVF9%2B/bVI4884jpWOAbPgwF%2BhF27dpnBgweb2NhY89FHHxljjKmqqjL9%2BvUza9euNY2NjebDDz808fHxZu/evcYYYz766CMTHx9vdu3aZRoaGsxLL71kEhMTTV1dnTHGmKVLl5q0tDRz5MgRU11dbSZOnGjGjh3rs338serr682gQYPM4sWLzalTp0xFRYUZO3as%2Bc1vfkOvznDixAnTu3dv89prr5nm5mZz/PhxM3z4cLN48WJ69R8sWrTIXH311Wbq1KnGGGO%2B%2BOIL07t3b/Pee%2B%2BZxsZG89Zbb5kbbrjBHDt2zBhjzIYNG8zNN99s/vnPfxqn02nmzp1rhg0bZlpaWowxxkybNs088MADprKy0nz99dfmvvvuM7NmzfLZ/llh3rx5Ztq0aa3G6VVr9913n8nKyjInT5401dXV5tFHHzXjxo3jGDxPBCz8YBs2bDC33nqreeutt9wC1iuvvGLuuOMOt2Wfeuopk5OTY4wxZtKkSWbGjBlu83feead59dVXjTHGJCUlmTfeeMM153A4zFVXXWUOHTp0IXfngjlw4IB5%2BOGHTVNTk2vs/fffNz//%2Bc/p1feorq42xhjT0tJi9u/fb1JSUsyaNWvo1Vl8%2BOGHZujQoSY7O9sVsBYuXGgeeught%2BUefvhhs3jxYmOMMf/1X/9lVqxY4ZpraGgw8fHxZvv27aaurs5cd911Zvfu3a75Tz75xNxwww2u/1D6o9GjR5u1a9e2GqdX7v72t7%2BZ3r17u45DY4yprKw0//znPzkGzxOXCPGD3XTTTXrvvfc0dOhQt/HS0lLFxsa6jUVHR6ukpESSVFZWdtb56upqHTt2zG2%2Ba9eu6ty5s/bv33%2BB9uTC%2BtnPfqbnn39egYGBrrF3331X1113Hb36HmFhYZKkW265RWlpaerWrZvS09Pp1fc4ceKEcnNz9eyzz8put7vG/1Mvvm8%2BKChIPXr0UElJib788ks1Nja6zffq1UtOp1MHDx68sDt0gbS0tOjTTz/VX//6V912221KSkrSk08%2BqZMnT9KrM%2Bzbt0/R0dF65ZVXlJKSoptuuknz589Xt27dOAbPEwELP1i3bt1ks9lajdfW1rr9n70kBQcHq66u7pzzp%2B9dCAkJaTXv7/c1SJIxRs8995w2b96s3NxcevUfbNq0SVu3blVAQICys7Pp1RlaWlo0ZcoUPfTQQ7r66qvd5n5Mr2pqaiS59%2Br0sv7aq4qKCl177bUaMmSIioqK9PLLL%2BvgwYOaMmUKvTrDyZMntX//fh08eFCFhYX685//rOPHj2vq1Kkcg%2BeJgAXL2e12103JpzmdToWGhp5z/vTBWV9ff9bX%2B6uamhplZ2dr48aNWrt2ra666ip69R8EBwere/fumjJlij744AN6dYaVK1eqY8eOGjNmTKu5H9Or0/8B/G6vTv/v02cX/U3Xrl21bt06ZWRkyG6366c//ammTJmirVu3yhhDr76jY8eOkqTc3FyFhYWpa9eumjhxorZs2fKjetUej8FzIWDBcrGxsSotLXUbKysrU0xMjCQpJibmrPOdO3dW9%2B7dVVZW5ppzOByqqqpqderZnxw6dEgjR45UTU2NXn31VV111VWS6NWZPv74Y915551qaGhwjTU0NCgoKEjR0dH06jtef/117dixQwkJCUpISNCbb76pN998UwkJCef9vmpsbNTBgwcVGxurnj17KigoyK1XBw4ccF0a80clJSXKz8%2BXMcY11tDQoICAAN1www306juio6PV0tKixsZG11hLS4sk6ZprruEYPB%2B%2BvQUM7cV3b3KvqKgwCQkJ5qWXXjINDQ1m%2B/btrptCjTGub55s377d9U2Tvn37msrKSmOMMc8995wZPny4OXTokOubJvfdd5/P9u3HqqqqMrfeequZNm2aaW5udpujV%2B5qamrMLbfcYubMmWNOnTplDh8%2BbDIyMszMmTPp1TlMnTrVdZN7WVmZ6d27t3nrrbdc34zr3bu3%2Bfzzz40x//4iys0332w%2B%2B%2Bwz1zfjUlJSTENDgzHGmMmTJ5v77rvPnDhxwpw4ccLcd999rnX7o6NHj5q4uDjzhz/8wTQ2Npp//etf5u677zZPPPEEvTpDQ0ODSUlJMRMmTDA1NTXmxIkT5v777zdZWVkcg%2BeJgAVLfDdgGWPMvn37TGZmpomPjze33367ee2119yW//Of/2yGDBli4uLiTEZGhvnkk09ccw0NDWbBggXm5ptvNj//%2Bc/N%2BPHjzddff%2B21fbHaiy%2B%2BaGJjY82NN95o4uLi3H6MoVdnKi0tNQ899JBJSEgwt912m1m4cKE5deqUMYZe/SffDVjGGLN161YzYsQIExcXZ4YNG2b%2B%2Bte/uuZaWlrMCy%2B8YJKTk01cXJwZM2aMK1AY8%2B9vcs6YMcMMHDjQ9O3b10ybNs3U1tZ6dX%2BsVlxc7Hrv9O/f38yePds4nU5jDL0607Fjx8zEiRPNoEGDTEJCgsnJyTEnT540xnAMno8OxnznnCkAAAB%2BNO7BAgAAsBgBCwAAwGIELAAAAIsRsAAAACxGwAIAALAYAQsAAMBiBCwAAACLEbAAAAAsRsACAACwGAELAADAYgQsAAAAixGwAAAALEbAAgAAsBgBCwAAwGIELAAAAIv9PxEmCYVk63lHAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-6035771469071839640">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1540</td>
        <td class="number">197</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1440</td>
        <td class="number">195</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1560</td>
        <td class="number">192</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1500</td>
        <td class="number">180</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1460</td>
        <td class="number">169</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1580</td>
        <td class="number">167</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1610</td>
        <td class="number">166</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1800</td>
        <td class="number">166</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1720</td>
        <td class="number">166</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1620</td>
        <td class="number">164</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (767)</td>
        <td class="number">19835</td>
        <td class="number">91.8%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-6035771469071839640">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">399</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">460</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">620</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">670</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">690</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">5600</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:17%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5610</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:17%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5790</td>
        <td class="number">6</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6110</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:17%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6210</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:17%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_sqft_lot">sqft_lot<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>9776</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>45.3%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>15099</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>520</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>1651359</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram1988976956527359027">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAAQpJREFUeJzt1sEJAkEQRUEVQzIIc/JsTgaxObV3kQeCy4hU3Rv%2B5TFznJk5AG%2BdVg%2BAX3ZePeDV5fb4%2BGa7X3dYAl4QSAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAI59UDvuFye3x8s92vOyzh3xxnZlaPgF/liwVBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBALhCV3BDpFBjPb7AAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives1988976956527359027,#minihistogram1988976956527359027"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives1988976956527359027">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles1988976956527359027"
                                                  aria-controls="quantiles1988976956527359027" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram1988976956527359027" aria-controls="histogram1988976956527359027"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common1988976956527359027" aria-controls="common1988976956527359027"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme1988976956527359027" aria-controls="extreme1988976956527359027"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles1988976956527359027">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>520</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>1800.8</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>5040</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>7618</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>10685</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>43307</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>1651359</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>1650839</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>5645</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>41413</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>2.7427</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>285.5</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>15099</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>13825</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>13.073</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>326101931</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>1715000000</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram1988976956527359027">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3XtYVXWi//GPsFE2KEIidvk1o4NgeYUw8ZaTGDlOoo6XqMyxUpuSJB0v5WRpOmiOZOV4tDLLk/qcTNPK8jbN6eKUoeNYWScK8FaDBQomV7l9f3/swz7t0MT6ctnyfj0Pz9P%2BrrW/67s%2BD5s%2B7LU2NjPGGAEAAMAan4ZeAAAAwMWGggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALHM09AKaitzcAutz%2Bvg00yWXBCovr0hVVcb6/N6CHFzIwYUcXMjBhRxcmnIObdu2apDj8g6WF/PxaaZmzZrJx6dZQy%2BlQZGDCzm4kIMLObiQgws51D8KFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABY5mjoBeDn6fnQjoZeQq1tn9qvoZcAAEC94B0sAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFjWaAtWenq67rzzTvXq1Uv9%2BvXTrFmzlJeXJ0n6%2BOOPNWbMGEVHRysuLk4bN270eO6WLVsUHx%2BvqKgojRw5UgcOHHBvq6ys1OLFi9W3b19FR0fr3nvvVU5Ojnv7yZMnNXnyZPXs2VOxsbFKSUlRRUVF/Zw0AAC4KDTKglVaWqqJEycqOjpa//jHP/TGG2/o1KlT%2BtOf/qTvvvtOd999t0aMGKF9%2B/YpJSVFixYt0ieffCJJSktL04IFC/TYY49p3759GjZsmO69916VlJRIklauXKn3339fr7zyinbv3i1/f3/NmTPHfeypU6cqICBAu3fv1qZNm7Rnzx6tWbOmIWIAAABeqlEWrOzsbF111VVKSkpS8%2BbNFRISosTERO3bt0%2B7du1ScHCwxo4dK4fDoT59%2BighIUHr16%2BXJG3cuFE33XSTYmJi5OfnpzvuuEMhISHatm2be/ukSZN02WWXqWXLlnrooYf03nvv6auvvtLRo0e1d%2B9ezZw5U06nU1deeaUmT57snhsAAKA2GmXB%2BtWvfqXnnntOvr6%2B7rGdO3eqS5cuysjIUGRkpMf%2BHTt2VHp6uiQpMzPznNsLCgr0zTffeGwPDQ1V69at9cUXXygjI0PBwcFq166de3t4eLiys7N1%2BvTpujhVAABwEWqUBev7jDF64okn9Pbbb%2Buhhx5SUVGRnE6nxz7%2B/v4qLi6WpB/dXlRUJEkKCAiosb2oqOisz61%2BXD0/AADA%2BTgaegE/prCwULNnz9Znn32mdevWqVOnTnI6nSooKPDYr7S0VIGBgZJchai0tLTG9pCQEHdZqr4f64fPN8bU2Fb9uHr%2B2sjJyVFubq7HmMMRoLCwsFrPURu%2Bvo2%2BH3twOOpmvdU5eFsetpGDCzm4kIMLObiQQ/1rtAXr2LFjmjRpki6//HJt2rRJl1xyiSQpMjJS77//vse%2BmZmZioiIkCRFREQoIyOjxvYBAwaodevWateuncdlxNzcXJ06dUqRkZGqqqrSqVOndOLECYWGhkqSsrKydOmll6pVq1a1XvuGDRu0fPlyj7GkpCQlJydfWAgXmZCQ2pfUnyIoyHn%2BnZoAcnAhBxdycCEHF3KoP42yYH333XcaP368evfurZSUFPn4/F/jjo%2BP15IlS7RmzRqNHTtW%2B/fv19atW7VixQpJ0ujRo5WUlKQhQ4YoJiZG69ev18mTJxUfHy9JGjlypFauXKlu3bopJCRECxcuVK9evfSLX/xCkhQTE6OFCxdq/vz5ys/P14oVKzR69OgLWn9iYqLi4uI8xhyOAOXnF/2cWGrwtt9EbJ9/NV9fHwUFOXX6dIkqK6vq5BjegBxcyMGFHFzIwaUp51DXv9yfS6MsWJs3b1Z2dra2b9%2BuHTt2eGw7cOCAnn/%2BeaWkpGjZsmW65JJLNGfOHPXu3VuS1KdPH82dO1fz5s3Tt99%2Bq44dO2rVqlUKDg6W5HonqaKiQmPHjlVRUZFiY2P15JNPuudftmyZ5s%2Bfr0GDBsnHx0cjRozQ5MmTL2j9YWFhNS4H5uYWqKKiaX1T/1Bdn39lZVWTz1gih2rk4EIOLuTgQg71p5kxxjT0IpqC3NyC8%2B90gRwOH8Wn7rY%2Bb13ZPrVfnczrcPgoJCRQ%2BflFTfoHBzm4kIMLObiQg0tTzqFt29rf4mOTd11jAgAA8AIULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJY1%2BoKVl5en%2BPh4paWlSZIeeeQRRUdHe3xdffXVmjBhgiSpqqpK0dHRioqK8tinuLhYknTy5ElNnjxZPXv2VGxsrFJSUlRRUeE%2B3scff6wxY8YoOjpacXFx2rhxY/2fNAAA8GqNumDt379fiYmJOnbsmHts/vz5OnDggPvrr3/9q4KCgvTggw9KkjIzM1VeXq69e/d67BcQECBJmjp1qgICArR7925t2rRJe/bs0Zo1ayRJ3333ne6%2B%2B26NGDFC%2B/btU0pKihYtWqRPPvmk3s8dAAB4r0ZbsLZs2aIZM2Zo2rRp59wnLy9PM2bM0EMPPaSIiAhJ0sGDB9WpUyc1b968xv5Hjx7V3r17NXPmTDmdTl155ZWaPHmy1q9fL0natWuXgoODNXbsWDkcDvXp00cJCQnu7QAAALXhaOgFnEv//v2VkJAgh8NxzpKVmpqqrl27atiwYe6xgwcP6syZMxo1apT%2B/e9/Kzw8XNOnT9c111yjjIwMBQcHq127du79w8PDlZ2drdOnTysjI0ORkZEex%2BjYsaM2bdp0QWvPyclRbm6ux5jDEaCwsLALmud8fH0bbT8%2BK4ejbtZbnYO35WEbObiQgws5uJCDCznUv0ZbsNq2bfuj27/66iu9/vrrNe6R8vf3V/fu3XX//ferdevWWr9%2BvSZMmKDXX39dRUVFcjqdHvtXPy4uLj7rdn9/f/f9W7W1YcMGLV%2B%2B3GMsKSlJycnJFzTPxSYkJLBO5w8Kcp5/pyaAHFzIwYUcXMjBhRzqT6MtWOfzyiuvuG9w/77qe7GqTZgwQZs3b9a7776rdu3aqaSkxGN79ePAwEA5nU4VFBR4bC8tLVVg4IUVg8TERMXFxXmMORwBys8vuqB5zsfbfhOxff7VfH19FBTk1OnTJaqsrKqTY3gDcnAhBxdycCEHl6acQ13/cn8uXluwdu3apbvuuqvG%2BBNPPKHBgwerc%2BfO7rGysjK1aNFCEREROnXqlE6cOKHQ0FBJUlZWli699FK1atVKkZGRev/99z3my8zMdN/fVVthYWE1Lgfm5haooqJpfVP/UF2ff2VlVZPPWCKHauTgQg4u5OBCDvXHu94C%2BV/5%2BfnKysrStddeW2Pbl19%2BqZSUFOXm5qqsrEzLly9XYWGh4uPj1b59e8XExGjhwoUqLCzUV199pRUrVmj06NGSpPj4eJ04cUJr1qxReXm5PvzwQ23dulWjRo2q71MEAABezCsL1tdffy1JHjerV1u0aJF%2B8YtfaPjw4YqNjdXevXv1wgsvKDg4WJK0bNkyVVRUaNCgQbr55pt13XXXafLkyZKkkJAQPf/889qxY4diY2M1Z84czZkzR717966/kwMAAF6vmTHGNPQimoLc3ILz73SBHA4fxafutj5vXdk%2BtV%2BdzOtw%2BCgkJFD5%2BUVN%2Bq1vcnAhBxdycCEHl6acQ9u2rRrkuF75DhYAAEBjRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwrNEXrLy8PMXHxystLc09NnfuXHXt2lXR0dHurw0bNri3r1q1SgMGDFBUVJTGjRunQ4cOubcVFxdr9uzZio2NVUxMjGbNmqWioiL39sOHD2v8%2BPGKjo5W//799fTTT9fPiQIAgItGoy5Y%2B/fvV2Jioo4dO%2BYxfvDgQS1YsEAHDhxwfyUmJkqStmzZorVr12r16tVKS0tTly5dlJycLGOMJGnBggU6fvy4du7cqV27dun48eNKTU2VJJWXl%2Buee%2B5Rt27dlJaWpmeffVbr16/X9u3b6/fEAQCAV2u0BWvLli2aMWOGpk2b5jFeVlamL7/8Ul27dj3r815%2B%2BWXddtttioiIUIsWLTR9%2BnRlZ2crLS1NJSUl2rp1q5KTkxUcHKw2bdpoxowZ2rx5s0pKSrRv3z7l5OQoOTlZzZs3V%2BfOnTVu3DitX7%2B%2BPk4ZAABcJBwNvYBz6d%2B/vxISEuRwODxKVnp6uioqKrRs2TLt379frVq10qhRozRx4kT5%2BPgoMzNTkyZNcu/v5%2Ben9u3bKz09XcHBwSovL1dkZKR7e3h4uEpLS3XkyBFlZGSoQ4cOat68uXt7x44d9eyzz17Q2nNycpSbm%2Bsx5nAEKCws7EJj%2BFG%2Bvo22H5%2BVw1E3663OwdvysI0cXMjBhRxcyMGFHOpfoy1Ybdu2Pet4QUGBevXqpXHjxmnp0qX6/PPPlZSUJB8fH02cOFFFRUVyOp0ez/H391dxcbEKCwslSQEBAe5t1fsWFRWd9blOp1PFxcUXtPYNGzZo%2BfLlHmNJSUlKTk6%2BoHkuNiEhgXU6f1CQ8/w7NQHk4EIOLuTgQg4u5FB/Gm3BOpd%2B/fqpX79%2B7sfdu3fX%2BPHjtW3bNk2cOFFOp1OlpaUezyktLVVgYKC7WJWUlCgwMND935LUsmVLBQQEuB9X%2B/6%2BtZWYmKi4uDiPMYcjQPn5Red4xk/jbb%2BJ2D7/ar6%2BPgoKcur06RJVVlbVyTG8ATm4kIMLObiQg0tTzqGuf7k/F68rWG%2B99ZZOnDihW265xT1WVlYmf39/SVJERIQyMjI0cOBASa4b148cOaLIyEh16NBBfn5%2ByszMVI8ePSRJWVlZ7suIJ0%2Be1JEjR1RRUSGHwxVNZmamIiIiLmiNYWFhNS4H5uYWqKKiaX1T/1Bdn39lZVWTz1gih2rk4EIOLuTgQg71x7veApFkjNGiRYu0Z88eGWN04MABvfjii%2B5PEY4aNUrr1q1Tenq6zpw5o8cff1yhoaHq2bOnnE6nhgwZotTUVOXl5SkvL0%2BpqakaOnSo/P39FRsbq5CQED3%2B%2BOM6c%2BaM0tPTtXbtWo0ePbqBzxoAAHgTr3sHKz4%2BXrNnz9a8efP07bffKjQ0VFOmTNHw4cMlSaNHj1ZBQYGSkpKUl5enbt266ZlnnpGfn58k19/QWrx4sRISElReXq5Bgwbp4YcfliQ5HA49//zzmj9/vvr166eAgACNGzdOI0eObLDzBQAA3qeZqf4DUahTubkF1ud0OHwUn7rb%2Brx1ZfvUfuff6SdwOHwUEhKo/PyiJv3WNzm4kIMLObiQg0tTzqFt21YNclyvu0QIAADQ2FGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBl1gtWZWWl7SkBAAC8ivWCNWDAAP3lL39RZmam7akBAAC8gvWCdd999%2Blf//qXhg4dqjFjxuill15SQUGB7cMAAAA0WtYL1q233qqXXnpJO3bsUN%2B%2BfbVq1Sr1799f06dP1wcffGD7cAAAAI1Ond3k3r59e02bNk07duxQUlKS/v73v2vChAmKi4vTCy%2B8wL1aAADgouWoq4k//vhjvfrqq9q2bZvKysoUHx%2BvkSNH6ttvv9VTTz2lgwcPaunSpXV1eAAAgAZjvWCtWLFCr732mo4ePapu3bpp2rRpGjp0qFq2bOnex9fXV4888ojtQwMAADQK1gvWunXrNGzYMI0ePVodO3Y86z7h4eGaMWOG7UMDAAA0CtYL1nvvvafCwkKdOnXKPbZt2zb16dNHISEhkqTOnTurc%2BfOtg8NAADQKFi/yf1//ud/NHjwYG3YsME9tmTJEiUkJOjLL7%2B0fTgAAIBGx3rB%2Bstf/qIbb7xR06ZNc4%2B99dZbGjBggB577DHbhwMAAGh0rBeszz77THfffbeaN2/uHvP19dXdd9%2Btjz766ILny8vLU3x8vNLS0txjO3fu1PDhw3XNNdcoLi5Oy5cvV1VVlXv7kCFD1KNHD0VHR7u/srKyJEnFxcWaPXu2YmNjFRMTo1mzZqmoqMj93MOHD2v8%2BPGKjo5W//799fTTT/%2BUGAAAQBNmvWC1bNlSx44dqzH%2BzTffyN/f/4Lm2r9/vxITEz3m%2B/TTTzVr1ixNnTpV//znP7Vq1Spt3rxZa9askSQVFhbq8OHD2rZtmw4cOOD%2BCg8PlyQtWLBAx48f186dO7Vr1y4dP35cqampkqTy8nLdc8896tatm9LS0vTss89q/fr12r59%2B09MAwAANEXWC9bgwYM1b948ffDBByosLFRRUZE%2B/PBDzZ8/X/Hx8bWeZ8uWLZoxY4bHpUZJ%2Bve//61bbrlFAwcOlI%2BPj8LDwxUfH699%2B/ZJchWw4OBgXXHFFTXmLCkp0datW5WcnKzg4GC1adNGM2bM0ObNm1VSUqJ9%2B/YpJydHycnJat68uTp37qxx48Zp/fr1Py8UAADQpFj/FOH06dP11Vdf6a677lKzZs3c4/Hx8Zo1a1at5%2Bnfv78SEhLkcDg8StbgwYM1ePBg9%2BPS0lK98847SkhIkCQdPHhQTqdTt99%2BuzIyMnTFFVdoypQpGjhwoI4ePary8nJFRka6nx8eHq7S0lIdOXJEGRkZ6tChg8flzY4dO%2BrZZ5%2B9oAxycnKUm5vrMeZwBCgsLOyC5jkfX986%2B0P8dcLhqJv1VufgbXnYRg4u5OBCDi7k4EIO9c96wXI6nXrmmWd0%2BPBhffHFF/Lz81N4eLjat29/QfO0bdv2vPsUFhbq/vvvl7%2B/v%2B644w5JUrNmzdStWzf98Y9/1OWXX64dO3ZoypQpWrdunSoqKiRJAQEBHuuVpKKiIhUVFbkff397cXHxBa19w4YNWr58ucdYUlKSkpOTL2iei01ISGCdzh8U5Dz/Tk0AObiQgws5uJCDCznUnzr7p3I6dOigDh061NX0OnTokJKTk9WmTRu9%2BOKL7r8UP3HiRI/9hg0bpjfeeEM7d%2B50v8tVUlKiwMBA939LrnvHAgIC3I%2BrfX/f2kpMTFRcXJzHmMMRoPz8onM846fxtt9EbJ9/NV9fHwUFOXX6dIkqK6vO/4SLFDm4kIMLObiQg0tTzqGuf7k/F%2BsF6/Dhw5o/f77279%2Bv8vLyGts///zzn32Md999V3/84x918803a/r06XI4/u80Vq9erc6dO6tPnz7usbKyMrVo0UIdOnSQn5%2BfMjMz1aNHD0lSVlaW/Pz81L59e508eVJHjhxRRUWFe87MzExFRERc0PrCwsJqXA7MzS1QRUXT%2Bqb%2Bobo%2B/8rKqiafsUQO1cjBhRxcyMGFHOqP9YI1b948ZWdna8aMGWrVqpXt6fXRRx8pKSlJ8%2BbN0%2BjRo2tsP378uDZu3KhVq1bpsssu06uvvqoDBw7o0UcfldPp1JAhQ5SamqqnnnpKkpSamqqhQ4fK399fsbGxCgkJ0eOPP66pU6fq8OHDWrt2bY0b7QEAAH6M9YJ14MAB/ed//qeio6NtTy1Jevrpp1VRUaGUlBSlpKS4x2NiYvTcc89p1qxZ8vHx0W233aaCggL3Teq//OUvJUlz587V4sWLlZCQoPLycg0aNEgPP/ywJMnhcOj555/X/Pnz1a9fPwUEBGjcuHEaOXJknZwLAAC4ODUzxhibE/7617/WqlWrPD6pB9clQtscDh/Fp%2B62Pm9d2T61X53M63D4KCQkUPn5RU36rW9ycCEHF3JwIQeXppxD27b2r6bVhvW7pMeNG6elS5eqoMB%2BoQAAAPAG1i8Rvvvuu/roo48UGxurNm3aePxNKUn6%2B9//bvuQAAAAjYr1ghUbG6vY2Fjb0wIAAHgN6wXrvvvusz0lAACAV6mTv1SZnp6u2bNn65ZbbtG3336r9evXKy0trS4OBQAA0OhYL1iffvqpxowZo6%2B//lqffvqpysrK9Pnnn%2Buuu%2B7S22%2B/bftwAAAAjY71gpWamqq77rpLa9eulZ%2BfnyTpz3/%2Bs37/%2B9/X%2BPf5AAAALkZ18g7WiBEjaozfeuutOnTokO3DAQAANDrWC5afn58KCwtrjGdnZ8vp5F/xBgAAFz/rBeuGG27Q448/rvz8fPdYVlaWUlJSdP3119s%2BHAAAQKNjvWA98MADKi0tVd%2B%2BfVVSUqKRI0dq6NChcjgcmjVrlu3DAQAANDrW/w5Wy5Yt9dJLL2nPnj36n//5H1VVVSkyMlLXXXedfHzq5K9CAAAANCrWC1a1Pn36qE%2BfPnU1PQAAQKNlvWDFxcWpWbNm59zOv0UIAAAudtYL1u9%2B9zuPglVeXq6jR4/qvffe09SpU20fDgAAoNGxXrCmTJly1vF169Zp//79%2Bv3vf2/7kAAAAI1Kvd11PnDgQL377rv1dTgAAIAGU28Fa%2B/evWrRokV9HQ4AAKDBWL9E%2BMNLgMYYFRYW6osvvuDyIAAAaBKsF6zLL7%2B8xqcI/fz8NH78eCUkJNg%2BHAAAQKNjvWA99thjtqcEAADwKtYL1r59%2B2q977XXXmv78AAAAA3OesG64447ZIxxf1WrvmxYPdasWTN9/vnntg8PAADQ4KwXrL/%2B9a9atGiRHnjgAfXu3Vt%2Bfn76%2BOOPNW/ePN12220aOHCg7UMCAAA0Ktb/TMPixYs1d%2B5c3XDDDWrZsqVatGihXr16af78%2BXr%2B%2Bed1xRVXuL8AAAAuRtYLVk5Oji677LIa4y1btlR%2Bfv4Fz5eXl6f4%2BHilpaW5xz7%2B%2BGONGTNG0dHRiouL08aNGz2es2XLFsXHxysqKkojR47UgQMH3NsqKyu1ePFi9e3bV9HR0br33nuVk5Pj3n7y5ElNnjxZPXv2VGxsrFJSUlRRUXHB6wYAAE2X9YIVFRWlpUuXqrCw0D126tQpLVmyRH369Lmgufbv36/ExEQdO3bMPfbdd9/p7rvv1ogRI7Rv3z6lpKRo0aJF%2BuSTTyRJaWlpWrBggR577DHt27dPw4YN07333quSkhJJ0sqVK/X%2B%2B%2B/rlVde0e7du%2BXv7685c%2Ba45586daoCAgK0e/dubdq0SXv27NGaNWt%2BRiIAAKCpsV6w5syZo48//lgDBgzQyJEjNXLkSA0cOFBfffWVHnnkkVrPs2XLFs2YMUPTpk3zGN%2B1a5eCg4M1duxYORwO9enTRwkJCVq/fr0kaePGjbrpppsUExMjPz8/3XHHHQoJCdG2bdvc2ydNmqTLLrtMLVu21EMPPaT33ntPX331lY4ePaq9e/dq5syZcjqduvLKKzV58mT33AAAALVh/Sb38PBwbdu2TVu3blVWVpYk6bbbbtNNN90kp9NZ63n69%2B%2BvhIQEORwOj5KVkZGhyMhIj307duyoTZs2SZIyMzM1atSoGtvT09NVUFCgb775xuP5oaGhat26tb744gtJUnBwsNq1a%2BdxPtnZ2Tp9%2BrSCgoJqvX4AANB0WS9YkhQUFKQxY8bo66%2B/1pVXXinJ9dfcL0Tbtm3POl5UVFSjqPn7%2B6u4uPi824uKiiRJAQEBNbZXb/vhc6sfFxcX17pg5eTkKDc312PM4QhQWFhYrZ5fW76%2B9fZPSVrhcNTNeqtz8LY8bCMHF3JwIQcXcnAhh/pnvWAZY/T4449r7dq1Ki8v186dO/XEE0%2BoRYsWmj9//gUXrR9yOp0qKCjwGCstLVVgYKB7e2lpaY3tISEh7rJUfT/WD59vjKmxrfpx9fy1sWHDBi1fvtxjLCkpScnJybWe42IUElL7DH%2BKoKDav0N6MSMHF3JwIQcXcnAhh/pjvWCtXbtWr732mubOnav58%2BdLkm644QY9%2BuijatOmjWbMmPGz5o%2BMjNT777/vMZaZmamIiAhJUkREhDIyMmpsHzBggFq3bq127dopMzPTfZkwNzdXp06dUmRkpKqqqnTq1CmdOHFCoaGhkqSsrCxdeumlatWqVa3XmJiYqLi4OI8xhyNA%2BflFF3y%2BP8bbfhOxff7VfH19FBTk1OnTJaqsrKqTY3gDcnAhBxdycCEHl6acQ13/cn8u1gvWhg0b9Mgjjyg%2BPl4LFiyQJP32t79V8%2BbNlZKS8rMLVnx8vJYsWaI1a9Zo7Nix2r9/v7Zu3aoVK1ZIkkaPHq2kpCQNGTJEMTExWr9%2BvU6ePKn4%2BHhJ0siRI7Vy5Up169ZNISEhWrhwoXr16qVf/OIXkqSYmBgtXLhQ8%2BfPV35%2BvlasWKHRo0df0BrDwsJqXA7MzS1QRUXT%2Bqb%2Bobo%2B/8rKqiafsUQO1cjBhRxcyMGFHOqP9YL19ddf6%2Bqrr64x3qlTJ504ceJnzx8SEqLnn39eKSkpWrbzK5%2BmAAAevUlEQVRsmS655BLNmTNHvXv3liT16dNHc%2BfO1bx58/Ttt9%2BqY8eOWrVqlYKDgyW5LtVVVFRo7NixKioqUmxsrJ588kn3/MuWLdP8%2BfM1aNAg%2Bfj4aMSIEZo8efLPXjcAAGg6rBesK664Qp988on%2B3//7fx7j7777rvuG9wtV/Qm/at26ddNLL710zv2HDx%2Bu4cOHn3Wbn5%2BfZsyYcc530kJDQ7Vs2bKftE4AAACpDgrWhAkT9Oijj%2Brbb7%2BVMUZ79uzRSy%2B9pLVr12r27Nm2DwcAANDoWC9Yo0aNUkVFhVauXKnS0lI98sgjatOmjaZNm6Zbb73V9uEAAAAaHesF6/XXX9dvfvMbJSYmKi8vT8YYtWnTxvZhAAAAGi3rn/P/85//7L6Z/ZJLLqFcAQCAJsd6wWrfvn2Nm9IBAACaEuuXCCMiIjRjxgw999xzat%2B%2BvVq0aOGxfdGiRbYPCQAA0KhYL1jHjh1TTEyMJNX49/gAAACaAisFa9GiRbr//vsVEBCgtWvX2pgSAADAa1m5B%2BvFF1%2Bs8Y8kT5gwQTk5OTamBwAA8CpWCpYxpsbYv/71L505c8bG9AAAAF7F%2BqcIAQAAmjoKFgAAgGXWClazZs1sTQUAAODVrP2Zhj//%2Bc8ef/OqvLxcS5YsUWBgoMd%2B/B0sAABwsbNSsK699toaf/MqOjpa%2Bfn5ys/Pt3EIAAAAr2GlYPG3rwAAAP4PN7kDAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsMzaP/Zcn15//XXNnTvXY6y8vFyS9Omnn2rixIlKS0uTw/F/p/fUU09pwIABqqysVGpqql577TWVlJSod%2B/eevTRRxUWFiZJOnnypB5%2B%2BGHt3btXvr6%2BGjZsmB544AGPuQAAAH6MV76DNWzYMB04cMD9tWPHDgUHByslJUWSq2StXr3aY58BAwZIklauXKn3339fr7zyinbv3i1/f3/NmTPHPffUqVMVEBCg3bt3a9OmTdqzZ4/WrFnTEKcJAAC8lFcWrO8zxmjmzJm6/vrrNXz4cH311Vf67rvv1Llz57Puv3HjRk2aNEmXXXaZWrZsqYceekjvvfeevvrqKx09elR79%2B7VzJkz5XQ6deWVV2ry5Mlav359PZ8VAADwZl5/3eu1115TZmamVqxYIUk6ePCgAgMDNW3aNB08eFChoaG64447NHr0aBUUFOibb75RZGSk%2B/mhoaFq3bq1vvjiC0lScHCw2rVr594eHh6u7OxsnT59WkFBQfV7cgAAwCt5dcGqqqrSypUrdc8996hly5aSpLKyMkVFRWnatGmKiIhQWlqapkyZosDAQEVHR0uSAgICPObx9/dXUVGRJMnpdHpsq35cXFxc64KVk5Oj3NxcjzGHI8B9n5ctvr7e9Qakw1E3663OwdvysI0cXMjBhRxcyMGFHOqfVxestLQ05eTkaPTo0e6xESNGaMSIEe7H/fv314gRI7R9%2B3b17dtXklRSUuIxT2lpqQIDA2WMqbGt%2BnFgYGCt17VhwwYtX77cYywpKUnJycm1nuNiFBJS%2Bwx/iqAg5/l3agLIwYUcXMjBhRxcyKH%2BeHXB2rlzp%2BLj4z3ekdq0aZMCAwM1ZMgQ91hZWZlatGih1q1bq127dsrMzHRfJszNzdWpU6cUGRmpqqoqnTp1SidOnFBoaKgkKSsrS5deeqlatWpV63UlJiYqLi7OY8zhCFB%2BftHPOd0avO03EdvnX83X10dBQU6dPl2iysqqOjmGNyAHF3JwIQcXcnBpyjnU9S/35%2BLVBWv//v36/e9/7zFWWFiopUuX6pe//KWuuuoqvffee3rjjTe0evVqSdLIkSO1cuVKdevWTSEhIVq4cKF69eqlX/ziF5KkmJgYLVy4UPPnz1d%2Bfr5WrFjh8Q5ZbYSFhdW4HJibW6CKiqb1Tf1DdX3%2BlZVVTT5jiRyqkYMLObiQgws51B%2BvLlhff/11jSIzfvx4FRcX67777tPJkyd15ZVXavHixerZs6ck16W6iooKjR07VkVFRYqNjdWTTz7pfv6yZcs0f/58DRo0SD4%2BPhoxYoQmT55cr%2BcFAAC8WzNjjGnoRTQFubkF1ud0OHwUn7rb%2Brx1ZfvUfnUyr8Pho5CQQOXnFzXp38zIwYUcXMjBhRxcmnIObdvW/hYfm7zrJh4AAAAvQMECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwzGsL1rZt29S5c2dFR0e7v2bOnFnr569atUoDBgxQVFSUxo0bp0OHDrm3FRcXa/bs2YqNjVVMTIxmzZqloqKiujgNAABwEfLagnXw4EENHz5cBw4ccH8tWbKkVs/dsmWL1q5dq9WrVystLU1dunRRcnKyjDGSpAULFuj48ePauXOndu3apePHjys1NbUuTwcAAFxEvLpgde3a9Sc99%2BWXX9Ztt92miIgItWjRQtOnT1d2drbS0tJUUlKirVu3Kjk5WcHBwWrTpo1mzJihzZs3q6SkxPJZAACAi5GjoRfwU1RVVemzzz6T0%2BnUc889p8rKSv3617/WjBkz1Lp16/M%2BPzMzU5MmTXI/9vPzU/v27ZWenq7g4GCVl5crMjLSvT08PFylpaU6cuSIrr766vPOn5OTo9zcXI8xhyNAYWFhF3CW5%2Bfr61392OGom/VW5%2BBtedhGDi7k4EIOLuTgQg71zysLVl5enjp37qzBgwdr2bJlys/P1wMPPKCZM2fq2WefPe/zi4qK5HQ6Pcb8/f1VXFyswsJCSVJAQIB7W/W%2Btb0Pa8OGDVq%2BfLnHWFJSkpKTk2v1/ItVSEhgnc4fFOQ8/05NADm4kIMLObiQgws51B%2BvLFihoaFav369%2B7HT6dTMmTN18803q7CwUC1btvzR5zudTpWWlnqMlZaWKjAw0F2sSkpKFBgY6P5vSeedt1piYqLi4uI8xhyOAOXn271R3tt%2BE7F9/tV8fX0UFOTU6dMlqqysqpNjeANycCEHF3JwIQeXppxDXf9yfy5eWbDS09P1xhtvaPr06WrWrJkkqaysTD4%2BPmrevPl5nx8REaGMjAwNHDhQklReXq4jR44oMjJSHTp0kJ%2BfnzIzM9WjRw9JUlZWlvsyYm2EhYXVuByYm1ugioqm9U39Q3V9/pWVVU0%2BY4kcqpGDCzm4kIMLOdQf73oL5H8FBwdr/fr1eu6551RRUaHs7GwtWbJEv/vd72pVsEaNGqV169YpPT1dZ86c0eOPP67Q0FD17NlTTqdTQ4YMUWpqqvLy8pSXl6fU1FQNHTpU/v7%2B9XB2AADA23nlO1iXXnqpnnnmGS1dulQrV65UixYtdNNNN9X672CNHj1aBQUFSkpKUl5enrp166ZnnnlGfn5%2BkqS5c%2Bdq8eLFSkhIUHl5uQYNGqSHH364Lk8JAABcRJqZ6j/%2BhDqVm1tgfU6Hw0fxqbutz1tXtk/tVyfzOhw%2BCgkJVH5%2BUZN%2B65scXMjBhRxcyMGlKefQtm2rBjmuV14iBAAAaMwoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZ5bcFKT0/XnXfeqV69eqlfv36aNWuW8vLyJElz585V165dFR0d7f7asGGD%2B7mrVq3SgAEDFBUVpXHjxunQoUPubcXFxZo9e7ZiY2MVExOjWbNmqaioqN7PDwAAeC%2BvLFilpaWaOHGioqOj9Y9//ENvvPGGTp06pT/96U%2BSpIMHD2rBggU6cOCA%2BysxMVGStGXLFq1du1arV69WWlqaunTpouTkZBljJEkLFizQ8ePHtXPnTu3atUvHjx9Xampqg50rAADwPl5ZsLKzs3XVVVcpKSlJzZs3V0hIiBITE7Vv3z6VlZXpyy%2B/VNeuXc/63Jdfflm33XabIiIi1KJFC02fPl3Z2dlKS0tTSUmJtm7dquTkZAUHB6tNmzaaMWOGNm/erJKSkno%2BSwAA4K0cDb2An%2BJXv/qVnnvuOY%2BxnTt3qkuXLkpPT1dFRYWWLVum/fv3q1WrVho1apQmTpwoHx8fZWZmatKkSe7n%2Bfn5qX379kpPT1dwcLDKy8sVGRnp3h4eHq7S0lIdOXJEV199da3Wl5OTo9zcXI8xhyNAYWFhP%2BOsa/L19a5%2B7HDUzXqrc/C2PGwjBxdycCEHF3JwIYf655UF6/uMMXryySf19ttva926dTpx4oR69eqlcePGaenSpfr888%2BVlJQkHx8fTZw4UUVFRXI6nR5z%2BPv7q7i4WIWFhZKkgIAA97bqfS/kPqwNGzZo%2BfLlHmNJSUlKTk7%2Bqad5UQgJCazT%2BYOCnOffqQkgBxdycCEHF3JwIYf649UFq7CwULNnz9Znn32mdevWqVOnTurUqZP69evn3qd79%2B4aP368tm3bpokTJ8rpdKq0tNRjntLSUgUGBrqLVUlJiQIDA93/LUktW7as9boSExMVFxfnMeZwBCg/3%2B7N8t72m4jt86/m6%2BujoCCnTp8uUWVlVZ0cwxuQgws5uJCDCzm4NOUc6vqX%2B3Px2oJ17NgxTZo0SZdffrk2bdqkSy65RJL01ltv6cSJE7rlllvc%2B5aVlcnf31%2BSFBERoYyMDA0cOFCSVF5eriNHjigyMlIdOnSQn5%2BfMjMz1aNHD0lSVlaW%2BzJibYWFhdW4HJibW6CKiqb1Tf1DdX3%2BlZVVTT5jiRyqkYMLObiQgws51B/vegvkf3333XcaP368rrnmGq1evdpdriTXJcNFixZpz549MsbowIEDevHFF92fIhw1apTWrVun9PR0nTlzRo8//rhCQ0PVs2dPOZ1ODRkyRKmpqcrLy1NeXp5SU1M1dOhQd0EDAAA4H698B2vz5s3Kzs7W9u3btWPHDo9tBw4c0OzZszVv3jx9%2B%2B23Cg0N1ZQpUzR8%2BHBJ0ujRo1VQUKCkpCTl5eWpW7dueuaZZ%2BTn5yfJ9Te0Fi9erISEBJWXl2vQoEF6%2BOGH6/0cAQCA92pmqv8AFOpUbm6B9TkdDh/Fp%2B62Pm9d2T613/l3%2BgkcDh%2BFhAQqP7%2BoSb/1TQ4u5OBCDi7k4NKUc2jbtlWDHNcrLxECAAA0ZhQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYBkFCwAAwDIKFgAAgGUULAAAAMsoWAAAAJZRsAAAACyjYAEAAFhGwQIAALCMggUAAGAZBQsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRSsszh58qQmT56snj17KjY2VikpKaqoqGjoZQEAAC/haOgFNEZTp05Vu3bttHv3bp04cUL33nuv1qxZo4kTJzb00rzakCffb%2BglXJDtU/s19BIAAF6Kd7B%2B4OjRo9q7d69mzpwpp9OpK6%2B8UpMnT9b69esbemkAAMBLULB%2BICMjQ8HBwWrXrp17LDw8XNnZ2Tp9%2BnQDrgwAAHgLLhH%2BQFFRkZxOp8dY9ePi4mIFBQWdd46cnBzl5uZ6jDkcAQoLC7O3UEm%2BvvTjuuRtlzT/%2B4FfS%2BL7ovr8yYEcJHKoRg71j4L1AwEBASopKfEYq34cGBhYqzk2bNig5cuXe4zdd999mjJlip1F/q%2BcnByNvzRDiYmJ1subN8nJydGGDRvIISdH//mfz5EDOUgih2rk4EIO9Y8q%2BwMRERE6deqUTpw44R7LysrSpZdeqlatWtVqjsTERG3evNnjKzEx0fpac3NztXz58hrvljU15OBCDi7k4EIOLuTgQg71j3ewfqB9%2B/aKiYnRwoULNX/%2BfOXn52vFihUaPXp0recICwvjNwQAAJow3sE6i2XLlqmiokKDBg3SzTffrOuuu06TJ09u6GUBAAAvwTtYZxEaGqply5Y19DIAAICX8p03b968hl4EfrrAwED16tWr1jfgX6zIwYUcXMjBhRxcyMGFHOpXM2OMaehFAAAAXEy4BwsAAMAyChYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQXLC508eVKTJ09Wz549FRsbq5SUFFVUVDT0smolPT1dd955p3r16qV%2B/fpp1qxZysvLkyTNnTtXXbt2VXR0tPtrw4YN7ueuWrVKAwYMUFRUlMaNG6dDhw65txUXF2v27NmKjY1VTEyMZs2apaKiIvf2w4cPa/z48YqOjlb//v319NNPe6zr3XffVUJCgqKiojRkyBC9/fbbdZrDtm3b1LlzZ49znTlzZq3WcrHk8Prrr3ucf3R0tLp27aquXbtKkiZOnKhu3bp5bH/vvfckSZWVlVq8eLH69u2r6Oho3XvvvcrJyXHPfb7XyMcff6wxY8YoOjpacXFx2rhxo8fatmzZovj4eEVFRWnkyJE6cOBAnWSQl5en%2BPh4paWlWVmbN%2Bdytix27typ4cOH65prrlFcXJyWL1%2Buqqoq9/YhQ4aoR48eHt8jWVlZkur%2BtfBjr0PbOTTmn411lcNFwcDr3H777Wb69OmmuLjYHDt2zNx0001m1apVDb2s8yopKTH9%2BvUzTz31lDlz5ozJy8szkyZNMn/4wx%2BMMcb87ne/M5s3bz7rczdv3myuu%2B468%2BWXX5rS0lKzaNEic9NNN5mqqipjjDEPPvigGT9%2BvMnPzzcnTpwwt99%2Bu5k3b54xxpiysjJz4403miVLlpgzZ86Yzz77zPTv399s27bNGGPM4cOHTbdu3czf/vY3U15ebt58803TvXt3880339RZFo899ph58MEHa4yfby0XWw7f980335h%2B/fqZV1991RhjTGxsrElLSzvrvn/9619NQkKCyc7ONgUFBWbq1Klm0qRJ7u0/9ho5deqU6dWrl1m3bp0pLy83H3zwgYmOjjYff/yxMcaYDz/80ERHR5t//vOfpqyszLzwwgsmNjbWFBcXWz3ff/7zn%2BaGG24wkZGR5sMPP7SyNm/N5WxZHDx40HTv3t3893//t6msrDSZmZlm4MCBZvXq1cYYYwoKCkynTp3M119/fdY56/K1cL7Xoc0cjGm8PxvrKoeLBQXLyxw5csRERkZ6/E/vzTffNNdff30Drqp2srKyzIQJE0xFRYV77K233jLXXHONOXPmjOnSpYv58ssvz/rcW265xaxcudL9uKyszERHR5s9e/aY4uJi06VLF7N//3739o8%2B%2Bsh0797dFBcXm/fff99ERUWZM2fOuLc/88wzZuzYscYYY5YuXWruvPNOj%2BNNmDDBPPXUU1bO%2B2zGjh1r1q1bV2P8fGu52HKoVlVVZcaNG2ceeughY4wxx44dM1dddZUpKCg46/4DBgwwr7/%2Buvtxbm6u6dSpkzl27Nh5XyMvv/yyufHGGz3me%2BSRR8ysWbOMMcZMnz7dzJkzx2P7b37zG7Np06aff6L/a/Pmzeb66683b775psf/TH/u2rwxl3NlsWPHDrNw4UKPfRcuXGjuueceY4wxe/bsMbGxsWeds65fCz/2OvypzpVDY/7ZWBc5XEy4ROhlMjIyFBwcrHbt2rnHwsPDlZ2drdOnTzfgys7vV7/6lZ577jn5%2Bvq6x3bu3KkuXbooPT1dFRUVWrZsmfr27avBgwfr2WefdV8OyMzMVGRkpPt5fn5%2Bat%2B%2BvdLT03X06FGVl5d7bA8PD1dpaamOHDmijIwMdejQQc2bN3dv79ixo9LT08869w%2B321ZVVaXPPvtM77zzjgYOHKgBAwbo4Ycf1nfffXfetVxMOXzfa6%2B9pszMTD344IOSpIMHDyowMFDTpk1T7969NXToUG3atEmSVFBQoG%2B%2B%2BcZjraGhoWrdurW%2B%2BOKL875GMjIyLijjH263oX///vrb3/6m3/72tx7jP2dt3prLubIYPHiwZs%2Be7X5cWlqqd955R126dJHk%2Bh5xOp26/fbbFRsbq5EjR7ovX9X1a%2BHHXoe2c2jMPxvrIoeLiaOhF4ALU1RUJKfT6TFW/bi4uFhBQUENsawLZozRk08%2Bqbffflvr1q3TiRMn1KtXL40bN05Lly7V559/rqSkJPn4%2BGjixIlnPW9/f38VFxersLBQkhQQEODeVr1vUVHROTMrLi5273OuuetCXl6eOnfurMGDB2vZsmXKz8/XAw88oJkzZ6qsrOxH13Ix5VCtqqpKK1eu1D333KOWLVtKksrKyhQVFaVp06YpIiJCaWlpmjJligIDAxUdHS3J8zyr11p9b8mPvUbOd571kUPbtm3POv5z1lZ97t6Wy7my%2BL7CwkLdf//98vf31x133CFJatasmbp166Y//vGPuvzyy7Vjxw5NmTJF69atc99XVlevhbrI4lw5FBQUNNqfjQ31M8NbULC8TEBAgEpKSjzGqh8HBgY2xJIuWGFhoWbPnq3PPvtM69atU6dOndSpUyf169fPvU/37t01fvx4bdu2TRMnTpTT6VRpaanHPKWlpQoMDHT/8CgpKXFnUJ1Jy5Ytz5lZ9b4/NnddCA0N1fr1692PnU6nZs6cqZtvvlmxsbE/upaLKYdqaWlpysnJ0ejRo91jI0aM0IgRI9yP%2B/fvrxEjRmj79u3q27eve%2B1nW6sx5kdfI06nUwUFBWd9rnTuHEJCQn7mmZ7fz1lb9f/oLrZcDh06pOTkZLVp00Yvvviiu4RPnDjRY79hw4bpjTfe0M6dO5WQkCCp7l4L9fla6devX6P92dhQPzO8BZcIvUxERIROnTqlEydOuMeysrJ06aWXqlWrVg24sto5duyYRo0apcLCQm3atEmdOnWSJL311lt66aWXPPYtKyuTv7%2B/JNd5Z2RkuLeVl5fryJEjioyMVIcOHeTn56fMzEz39qysLPfb1RERETpy5IjHp6UyMzMVEREhSYqMjPSY%2B4fbbUtPT1dqaqqMMR7n6uPjo%2B7du//oWi6mHKrt3LlT8fHxHr9lb9q0Sdu3b/fYr6ysTC1atFDr1q3Vrl07j/PMzc3VqVOnFBkZed7XyPnO84cZ/3B7Xfo5a7sYc3n33Xc1ZswYXXfddVq9erVat27t3rZ69Wrt2bPHY//q75G6fi382OvQtsb8s7E%2Bc/BKDXsLGH6KW2%2B91UybNs0UFBS4Pwm0bNmyhl7WeZ06dcpcf/315sEHHzSVlZUe23bt2mW6d%2B9uPvjgA1NVVWX%2B9a9/mdjYWPcnyl5%2B%2BWVz3XXXmc8//9z9aZX4%2BHhTVlZmjDFmxowZ5vbbbzcnT540J0%2BeNLfffrt54IEHjDHGlJeXm7i4OPPYY4%2BZ0tJS8/nnn5v%2B/fubV155xRhjTGZmpunWrZt588033Z%2BU6datmzl06FCd5HD8%2BHETFRVlnn32WVNeXm7%2B/e9/m5tvvtn86U9/Ou9aLqYcqg0dOtS8/PLLHmMvvPCC6dOnj/nss89MZWWlefvtt0337t3Nvn37jDHGPPHEE2bo0KHm2LFj7k/L3X777e7n/9hrJC8vz/Ts2dO88MILpqyszOzZs8fjxtzqT8/t2bPH/Wm5a6%2B91uTn59fJ%2BX//huafuzZvz%2BX7WRw4cMB06dLFbNy48az7LliwwAwePNgcO3bMlJeXm40bN5ru3bubI0eOGGPq9rVwvtehzRwa88/Gus7B21GwvFBubq6ZMmWK6dWrl%2Bndu7d57LHHPD6Z11g9//zzJjIy0vTo0cNERUV5fBljzH/913%2BZG2%2B80fTo0cMMGjTI41N2VVVVZvXq1SYuLs5ERUWZcePGefyPv6CgwMyZM8f07dvXXHvttebBBx80RUVF7u1Hjhwxd911l4mJiTHXXXedeeaZZzzW9t5775lhw4aZqKgoc9NNN5l33nmnTrNIS0sziYmJJjo62vTu3dssWLDAlJaWnnctF1sOxhgTFRVV4zhVVVXmP/7jP8zAgQNN9%2B7dzU033WS2b9/u3l5WVmaWLFlirrvuOnPNNdeYe%2B%2B915w4ccK9/XyvkU8%2B%2BcSd/6BBg9z/Q6n26quvmsGDB5uoqCgzevRo89FHH9XR2ZsaH8n/OWvz9ly%2Bn8Uf/vAH06lTpxo/KyZMmGCMcX26LiUlxfTv39/06NHDjBo1yiPHunwtnO91aDMHYxrvz8a6zsHbNTPme9cpAAAA8LNxDxYAAIBlFCwAAADLKFgAAACWUbAAAAAso2ABAABYRsECAACwjIIFAABgGQULAADAMgoWAACAZRQsAAAAyyhYAAAAllGwAAAALKNgAQAAWEbBAgAAsIyCBQAAYNn/B3XT1DB8KtvRAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common1988976956527359027">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">5000</td>
        <td class="number">358</td>
        <td class="number">1.7%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6000</td>
        <td class="number">290</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4000</td>
        <td class="number">251</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7200</td>
        <td class="number">220</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7500</td>
        <td class="number">119</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4800</td>
        <td class="number">119</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4500</td>
        <td class="number">114</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8400</td>
        <td class="number">111</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">9600</td>
        <td class="number">109</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3600</td>
        <td class="number">103</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (9766)</td>
        <td class="number">19803</td>
        <td class="number">91.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme1988976956527359027">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">520</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">572</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">600</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">609</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">635</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">982998</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1024068</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1074218</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1164794</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1651359</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_sqft_lot15">sqft_lot15<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>8682</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>40.2%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>12758</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>651</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>871200</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-549848781125627893">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAAQtJREFUeJzt1rENwjAARUGCGIkh2Cl1dsoQ2cn0CD0JCXCKu97yb57sZYwxLsBb19kD4Mxuswe8uq/7x2eO7fGDJeAFgSQQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBcJs94Bvu6/6Xe47t8Zd7OI9ljDFmj4Cz8sWCIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKB8AQKrQ6RDvzkwwAAAABJRU5ErkJggg%3D%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-549848781125627893,#minihistogram-549848781125627893"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-549848781125627893">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-549848781125627893"
                                                  aria-controls="quantiles-549848781125627893" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-549848781125627893" aria-controls="histogram-549848781125627893"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-549848781125627893" aria-controls="common-549848781125627893"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-549848781125627893" aria-controls="extreme-549848781125627893"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-549848781125627893">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>651</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>2002.4</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>5100</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>7620</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>10083</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>37045</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>871200</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>870549</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>4983</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>27274</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>2.1378</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>151.4</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>12758</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>10102</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>9.5244</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>275540649</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>743900000</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-549848781125627893">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3X1cVHX%2B//%2BnMCgDXkAidvFtP7oIlqVBoGSamxjruom6ilGZa6X22SBJV8RcLU1Dc0Ur86OZWW7q7RNluqV5te2tNdc1dF0za6OAQm0xQS6US7k6vz/6MZ%2BdrMDtjTMDj/vtNreb836feZ/3e16cenLOmaGdZVmWAAAAYIyXqycAAADQ2hCwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhNldPoK0oLCwzOp6XVztdcYW/iosr1NBgGR0b/xlq4l6oh3uhHu6lLdWjW7dOLtkvZ7A8lJdXO7Vr105eXu1cPRX8/6iJe6Ee7oV6uBfq0fIIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgmM3VE8CPE5u%2B39VTaLZd0we5egoAAFwWnMECAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGFuG7CysrL0wAMPaMCAARo0aJBSU1NVXFwsSTp27JjGjx%2BviIgIxcTE6I033nB67bZt2xQbG6vw8HCNHTtWR48edfTV19dr6dKluvXWWxUREaGHH35YBQUFjv6ioiIlJiYqKipK0dHRSktLU11d3eVZNAAAaBXcMmBVV1drypQpioiI0F//%2Blft2LFDpaWl%2Bt3vfqdz587poYce0pgxY3T48GGlpaVpyZIl%2BuijjyRJmZmZWrRokZ5%2B%2BmkdPnxYo0aN0sMPP6yqqipJ0po1a3TgwAG9%2Beab2r9/v3x9fTVv3jzHvqdPny4/Pz/t379fW7Zs0cGDB7VhwwZXvA0AAMBDuWXAys/P13XXXaekpCS1b99egYGBSkhI0OHDh7V3714FBARowoQJstlsGjhwoOLi4rR582ZJ0htvvKE777xTkZGR8vHx0f3336/AwEDt3LnT0T916lRdddVV6tixo%2BbOnav3339fp06d0okTJ3To0CHNmjVLdrtd1157rRITEx1jAwAANIdbBqyf/vSneumll%2BTt7e1o27Nnj2644QZlZ2crLCzMaftevXopKytLkpSTk/O9/WVlZfr666%2Bd%2BoOCgtSlSxd99tlnys7OVkBAgLp37%2B7oDwkJUX5%2Bvs6fP98SSwUAAK2QzdUTaIplWXr22Wf13nvvadOmTXr11Vdlt9udtvH19VVlZaUkqaKi4nv7KyoqJEl%2Bfn4X9Tf2ffu1jc8rKyvVuXPnZs25oKBAhYWFTm02m5%2BCg4Ob9frm8PZ2y2z8g2w2z5vzpWisiSfWpjWiHu6FergX6tHy3DpglZeXa86cOfrkk0%2B0adMm9e7dW3a7XWVlZU7bVVdXy9/fX9I3gai6uvqi/sDAQEdYarwf69uvtyzror7G543jN0dGRoZWrVrl1JaUlKTk5ORmj9EaBQY2/z30ZJ0725veCJcN9XAv1MO9UI%2BW47YB6%2BTJk5o6daquvvpqbdmyRVdccYUkKSwsTAcOHHDaNicnR6GhoZKk0NBQZWdnX9Q/ZMgQdenSRd27d3e6jFhYWKjS0lKFhYWpoaFBpaWlOnv2rIKCgiRJubm5uvLKK9WpU6dmzz0hIUExMTFObTabn0pKKi7tTfgB3t5eHndgmFy/O2qsyfnzVaqvb3D1dNo86uFeqId7aUv1cNUv924ZsM6dO6dJkybplltuUVpamry8/u8UZmxsrJYtW6YNGzZowoQJOnLkiLZv367Vq1dLkuLj45WUlKQRI0YoMjJSmzdvVlFRkWJjYyVJY8eO1Zo1a9S3b18FBgZq8eLFGjBggH7yk59IkiIjI7V48WItXLhQJSUlWr16teLj4y9p/sHBwRddDiwsLFNdXev%2BIW5KW1l/fX1Dm1mrJ6Ae7oV6uBfq0XLcMmBt3bpV%2Bfn52rVrl3bv3u3Ud/ToUb388stKS0vTypUrdcUVV2jevHm65ZZbJEkDBw7U/PnztWDBAp05c0a9evXSunXrFBAQIOmbS3V1dXWaMGGCKioqFB0drWeffdYx/sqVK7Vw4UINGzZMXl5eGjNmjBITEy/f4gEAgMdrZ1mW5epJtAWFhWVNb3QJbDYvBQb6K2ru7qY3dhO7pg9y9RRaVGNNSkoq%2BI3QDVAP90I93Etbqke3bs2/xcckPj4AAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMLcPWMXFxYqNjVVmZqYk6YknnlBERITT4/rrr9fkyZMlSQ0NDYqIiFB4eLjTNpWVlZKkoqIiJSYmKioqStHR0UpLS1NdXZ1jf8eOHdP48eMVERGhmJgYvfHGG5d/0QAAwKO5dcA6cuSIEhISdPLkSUfbwoULdfToUcfj%2BeefV%2BfOnfXYY49JknJyclRbW6tDhw45befn5ydJmj59uvz8/LR//35t2bJFBw8e1IYNGyRJ586d00MPPaQxY8bo8OHDSktL05IlS/TRRx9d9rUDAADP5bYBa9u2bUpJSdGMGTO%2Bd5vi4mKlpKRo7ty5Cg0NlSQdP35cvXv3Vvv27S/a/sSJEzp06JBmzZolu92ua6%2B9VomJidq8ebMkae/evQoICNCECRNks9k0cOBAxcXFOfoBAACaw%2BbqCXyfwYMHKy4uTjab7XtDVnp6um688UaNGjXK0Xb8%2BHFduHBB48aN07/%2B9S%2BFhIRo5syZuvnmm5Wdna2AgAB1797dsX1ISIjy8/N1/vx5ZWdnKywszGkfvXr10pYtWy5p7gUFBSosLHRqs9n8FBwcfEnj/BBvb7fNxt/LZvO8OV%2BKxpp4Ym1aI%2BrhXqiHe6EeLc9tA1a3bt1%2BsP/UqVN6%2B%2B23L7pHytfXV/369dOjjz6qLl26aPPmzZo8ebLefvttVVRUyG63O23f%2BLyysvI7%2B319fR33bzVXRkaGVq1a5dSWlJSk5OTkSxqntQkM9Hf1FC6Lzp3tTW%2BEy4Z6uBfq4V6oR8tx24DVlDfffNNxg/u/a7wXq9HkyZO1detW7du3T927d1dVVZVTf%2BNzf39/2e12lZWVOfVXV1fL3//SgkFCQoJiYmKc2mw2P5WUVFzSOD/E29vL4w4Mk%2Bt3R401OX%2B%2BSvX1Da6eTptHPdwL9XAvbakervrl3mMD1t69e/Xggw9e1P7MM89o%2BPDh6tOnj6OtpqZGHTp0UGhoqEpLS3X27FkFBQVJknJzc3XllVeqU6dOCgsL04EDB5zGy8nJcdzf1VzBwcEXXQ4sLCxTXV3r/iFuSltZf319Q5tZqyegHu6FergX6tFyPPLia0lJiXJzc9W/f/%2BL%2Bj7//HOlpaWpsLBQNTU1WrVqlcrLyxUbG6sePXooMjJSixcvVnl5uU6dOqXVq1crPj5ekhQbG6uzZ89qw4YNqq2t1QcffKDt27dr3Lhxl3uJAADAg3lkwPrqq68kyelm9UZLlizRT37yE40ePVrR0dE6dOiQXnnlFQUEBEiSVq5cqbq6Og0bNkx33XWXbrvtNiUmJkqSAgMD9fLLL2v37t2Kjo7WvHnzNG/ePN1yyy2Xb3EAAMDjtbMsy3L1JNqCwsKypje6BDablwID/RU1d7fRcVvSrumDXD2FFtVYk5KSCk65uwHq4V6oh3tpS/Xo1q2TS/brkWewAAAA3BkBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDC3D1jFxcWKjY1VZmamo23%2B/Pm68cYbFRER4XhkZGQ4%2BtetW6chQ4YoPDxcEydO1BdffOHoq6ys1Jw5cxQdHa3IyEilpqaqoqLC0f/ll19q0qRJioiI0ODBg/XCCy9cnoUCAIBWw60D1pEjR5SQkKCTJ086tR8/flyLFi3S0aNHHY%2BEhARJ0rZt27Rx40atX79emZmZuuGGG5ScnCzLsiRJixYt0unTp7Vnzx7t3btXp0%2BfVnp6uiSptrZWv/nNb9S3b19lZmbqxRdf1ObNm7Vr167Lu3AAAODR3DZgbdu2TSkpKZoxY4ZTe01NjT7//HPdeOON3/m6119/Xffee69CQ0PVoUMHzZw5U/n5%2BcrMzFRVVZW2b9%2Bu5ORkBQQEqGvXrkpJSdHWrVtVVVWlw4cPq6CgQMnJyWrfvr369OmjiRMnavPmzZdjyQAAoJWwuXoC32fw4MGKi4uTzWZzCllZWVmqq6vTypUrdeTIEXXq1Enjxo3TlClT5OXlpZycHE2dOtWxvY%2BPj3r06KGsrCwFBASotrZWYWFhjv6QkBBVV1crLy9P2dnZ6tmzp9q3b%2B/o79Wrl1588cVLmntBQYEKCwud2mw2PwUHB1/q2/C9vL3dNht/L5vN8%2BZ8KRpr4om1aY2oh3uhHu6FerQ8tw1Y3bp1%2B872srIyDRgwQBMnTtSKFSv06aefKikpSV5eXpoyZYoqKipkt9udXuPr66vKykqVl5dLkvz8/Bx9jdtWVFR852vtdrsqKysvae4ZGRlatWqVU1tSUpKSk5MvaZzWJjDQ39VTuCw6d7Y3vREuG%2BrhXqiHe6EeLcdtA9b3GTRokAYNGuR43q9fP02aNEk7d%2B7UlClTZLfbVV1d7fSa6upq%2Bfv7O4JVVVWV/P39Hf%2BWpI4dO8rPz8/xvNG/b9tcCQkJiomJcWqz2fxUUlLxPa%2B4dN7eXh53YJhcvztqrMn581Wqr29w9XTaPOrhXqiHe2lL9XDVL/ceF7DeffddnT17VnfffbejraamRr6%2BvpKk0NBQZWdna%2BjQoZK%2BuXE9Ly9PYWFh6tmzp3x8fJSTk6ObbrpJkpSbm%2Bu4jFhUVKS8vDzV1dXJZvvmrcnJyVFoaOglzTE4OPiiy4GFhWWqq2vdP8RNaSvrr69vaDNr9QTUw71QD/dCPVqOx118tSxLS5Ys0cGDB2VZlo4ePapXX33V8SnCcePGadOmTcrKytKFCxe0fPlyBQUFKSoqSna7XSNGjFB6erqKi4tVXFys9PR0jRw5Ur6%2BvoqOjlZgYKCWL1%2BuCxcuKCsrSxs3blR8fLyLVw0AADyJx53Bio2N1Zw5c7RgwQKdOXNGQUFBmjZtmkaPHi1Jio%2BPV1lZmZKSklRcXKy%2Bfftq7dq18vHxkfTNd2gtXbpUcXFxqq2t1bBhw/T4449Lkmw2m15%2B%2BWUtXLhQgwYNkp%2BfnyZOnKixY8e6bL0AAMDztLMavyAKLaqwsMzoeDablwID/RU1d7fRcVvSrumDmt7IgzXWpKSkglPuboB6uBfq4V7aUj26devkkv163CVCAAAAd0fAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMMx6w6uvrTQ8JAADgUYwHrCFDhuj3v/%2B9cnJyTA8NAADgEYwHrEceeUT/%2BMc/NHLkSI0fP16vvfaaysrKTO8GAADAbRkPWPfcc49ee%2B017d69W7feeqvWrVunwYMHa%2BbMmfrb3/5mencAAABup8Vucu/Ro4dmzJih3bt3KykpSX/%2B8581efJkxcTE6JVXXuFeLQAA0GrZWmrgY8eO6Y9//KN27typmpoaxcbGauzYsTpz5oyee%2B45HT9%2BXCtWrGip3QMAALiM8YC1evVqvfXWWzpx4oT69u2rGTNmaOTIkerYsaNjG29vbz3xxBOmdw0AAOAWjAesTZs2adSoUYqPj1evXr2%2Bc5uQkBClpKSY3jUAAIBbMB6w3n//fZWXl6u0tNTRtnPnTg0cOFCBgYGSpD59%2BqhPnz6mdw0AAOAWjN/k/s9//lPDhw9XRkaGo23ZsmWKi4vT559/bnp3AAAAbsd4wPr973%2Bvn//855oxY4aj7d1339WQIUP09NNPX/J4xcXFio2NVWZmpqNtz549Gj16tG6%2B%2BWbFxMRo1apVamhocPSPGDFCN910kyIiIhyP3NxcSVJlZaXmzJmj6OhoRUZGKjU1VRUVFY7Xfvnll5o0aZIiIiI0ePBgvfDCC//J2wAAANow4wHrk08%2B0UMPPaT27ds72ry9vfXQQw/pww8/vKSxjhw5ooSEBJ08edLR9vHHHys1NVXTp0/X3//%2Bd61bt05bt27Vhg0bJEnl5eX68ssvtXPnTh09etTxCAkJkSQtWrRIp0%2Bf1p49e7R3716dPn1a6enpkqTa2lr95je/Ud%2B%2BfZWZmakXX3xRmzdv1q5du37kuwIAANoS4wGrY8eOToGo0ddffy1fX99mj7Nt2zalpKQ4nQmTpH/961%2B6%2B%2B67NXToUHl5eSkkJESxsbE6fPiwpG8CWEBAgK655pqLxqyqqtL27duVnJysgIAAde3aVSkpKdq6dauqqqp0%2BPBhFRQUKDk5We3bt1efPn00ceJEbd68%2BRLfBQAA0JYZv8l9%2BPDhWrBggZ588kn169dP7dq10/Hjx7Vw4ULFxsY2e5zBgwcrLi5ONpvNKWQNHz5cw4cPdzyvrq7WX/7yF8XFxUmSjh8/Lrvdrvvuu0/Z2dm65pprNG3aNA0dOlQnTpxQbW2twsLCHK8PCQlRdXW18vLylJ2drZ49ezqdfevVq5defPHFS3oPCgoKVFhY6NRms/kpODj4ksb5Id7eLfYdsS3GZvO8OV%2BKxpp4Ym1aI%2BrhXqiHe6EeLc94wJo5c6ZOnTqlBx98UO3atXO0x8bGKjU1tdnjdOvWrcltysvL9eijj8rX11f333%2B/JKldu3bq27evfvvb3%2Brqq6/W7t27NW3aNG3atEl1dXWSJD8/P8cYdrtdklRRUaGKigrH83/vr6ysbPa8JSkjI0OrVq1yaktKSlJycvIljdPaBAb6u3oKl0XnzvamN8JlQz3cC/VwL9Sj5RgPWHa7XWvXrtWXX36pzz77TD4%2BPgoJCVGPHj2M7ueLL75QcnKyunbtqldffdXxRaZTpkxx2m7UqFHasWOH9uzZ4zjLVVVVJX9/f8e/pW8ubfr5%2BTmeN/r3bZsrISFBMTExTm02m59KSiq%2B5xWXztvby%2BMODJPrd0eNNTl/vkr19Q1NvwAtinq4F%2BrhXtpSPVz1y32L/amcnj17qmfPni0y9r59%2B/Tb3/5Wd911l2bOnCmb7f%2BWsX79evXp00cDBw50tNXU1KhDhw7q2bOnfHx8lJOTo5tuukmSlJubKx8fH/Xo0UNFRUXKy8tTXV2dY8ycnByFhoZe0vyCg4MvuhxYWFimurrW/UPclLay/vr6hjazVk9APdwL9XAv1KPlGL/4%2BuWXX%2BqBBx5Qv379dP3111/0%2BLE%2B/PBDJSUlac6cOZo9e7ZTuJKk06dP68knn9SpU6dUV1enLVu26OjRo/rVr34lu92uESNGKD09XcXFxSouLlZ6erpGjhwpX19fRUdHKzAwUMuXL9eFCxeUlZWljRs3Kj4%2B/kfPGwAAtB3Gz2AtWLBA%2Bfn5SklJUadOnUwPrxdeeEF1dXVKS0tTWlqaoz0yMlIvvfSSUlNT5eXlpXvvvVdlZWWOm9T/67/%2BS5I0f/58LV26VHFxcaqtrdWwYcP0%2BOOPS5JsNptefvllLVy4UIMGDZKfn58mTpyosWPHGl8HAABovdpZlmWZHLBfv376wx/%2BoIiICJPDerzCwjKj49lsXgoM9FfU3N1Gx21Ju6YPcvUUWlRjTUpKKjjl7gaoh3uhHu6lLdWjWzfzJ3uaw/glwsDAwEu%2BKRwAAKA1MR6wJk6cqBUrVqiszOwZGwAAAE9h/B6sffv26cMPP1R0dLS6du3q9KWdkvTnP//Z9C4BAADcivGAFR0drejoaNPDAgAAeAzjAeuRRx4xPSQAAIBHaZE/QpSVlaU5c%2Bbo7rvv1pkzZ7R582ZlZma2xK4AAADcjvGA9fHHH2v8%2BPH66quv9PHHH6umpkaffvqpHnzwQb333numdwcAAOB2jAes9PR0Pfjgg9q4caN8fHwkSU899ZR%2B/etfX/QHkAEAAFqjFjmDNWbMmIva77nnHn3xxRemdwcAAOB2jAcsHx8flZeXX9Sen58vu91uencAAABux3jAuuOOO7R8%2BXKVlJQ42nJzc5WWlqbbb7/d9O4AAADcjvGANXv2bFVXV%2BvWW29VVVWVxo4dq5EjR8pmsyk1NdX07gAAANyO8e/B6tixo1577TUdPHhQ//znP9XQ0KCwsDDddttt8vJqkW%2BFAAAAcCvGA1ajgQMHauDAgS01PAAAgNsyHrBiYmLUrl277%2B3nbxECAIDWznjA%2BtWvfuUUsGpra3XixAm9//77mj59uundAQAAuB3jAWvatGnf2b5p0yYdOXJEv/71r03vEgAAwK1ctrvOhw4dqn379l2u3QEAALjMZQtYhw4dUocOHS7X7gAAAFzG%2BCXCb18CtCxL5eXl%2Buyzz7g8CAAA2gTjAevqq6%2B%2B6FOEPj4%2BmjRpkuLi4kzvDgAAwO0YD1hPP/206SEBAAA8ivGAdfjw4WZv279/f9O7BwAAcDnjAev%2B%2B%2B%2BXZVmOR6PGy4aNbe3atdOnn35qevcAAAAuZzxgPf/881qyZIlmz56tW265RT4%2BPjp27JgWLFige%2B%2B9V0OHDjW9SwAAALdi/Gsali5dqvnz5%2BuOO%2B5Qx44d1aFDBw0YMEALFy7Uyy%2B/rGuuucbxAAAAaI2MB6yCggJdddVVF7V37NhRJSUllzxecXGxYmNjlZmZ6Wg7duyYxo8fr4iICMXExOiNN95wes22bdsUGxur8PBwjR07VkePHnX01dfXa%2BnSpbr11lsVERGhhx9%2BWAUFBY7%2BoqIiJSYmKioqStHR0UpLS1NdXd0lzxsAALRdxgNWeHi4VqxYofLyckdbaWmpli1bpoEDB17SWEeOHFFCQoJOnjzpaDt37pweeughjRkzRocPH1ZaWpqWLFmijz76SJKUmZmpRYsW6emnn9bhw4c1atQoPfzww6qqqpIkrVmzRgcOHNCbb76p/fv3y9fXV/PmzXOMP336dPn5%2BWn//v3asmWLDh48qA0bNvyIdwQAALQ1xgPWvHnzdOzYMQ0ZMkRjx47V2LFjNXToUJ06dUpPPPFEs8fZtm2bUlJSNGPGDKf2vXv3KiAgQBMmTJDNZtPAgQMVFxenzZs3S5LeeOMN3XnnnYqMjJSPj4/uv/9%2BBQYGaufOnY7%2BqVOn6qqrrlLHjh01d%2B5cvf/%2B%2Bzp16pROnDihQ4cOadasWbLb7br22muVmJjoGBsAAKA5jN/kHhISop07d2r79u3Kzc2VJN1777268847Zbfbmz3O4MGDFRcXJ5vN5hSysrOzFRYW5rRtr169tGXLFklSTk6Oxo0bd1F/VlaWysrK9PXXXzu9PigoSF26dNFnn30mSQoICFD37t2d1pOfn6/z58%2Brc%2BfOzZp7QUGBCgsLndpsNj8FBwc36/XN4e192f7KkTE2m%2BfN%2BVI01sQTa9MaUQ/3Qj3cC/VoecYDliR17txZ48eP11dffaVrr71W0jff5n4punXr9p3tFRUVFwU1X19fVVZWNtlfUVEhSfLz87uov7Hv269tfF5ZWdnsgJWRkaFVq1Y5tSUlJSk5OblZr2%2BtAgP9XT2Fy6Jz5%2Bb/IoGWRz3cC/VwL9Sj5RgPWJZlafny5dq4caNqa2u1Z88ePfPMM%2BrQoYMWLlx4yUHr2%2Bx2u8rKypzaqqur5e/v7%2Bivrq6%2BqD8wMNARlhrvx/r26y3Luqiv8Xnj%2BM2RkJCgmJgYpzabzU8lJRXNHqMp3t5eHndgmFy/O2qsyfnzVaqvb3D1dNo86uFeqId7aUv1cNUv98YD1saNG/XWW29p/vz5WrhwoSTpjjvu0JNPPqmuXbsqJSXlR40fFhamAwcOOLXl5OQoNDRUkhQaGqrs7OyL%2BocMGaIuXbqoe/fuysnJcVwmLCwsVGlpqcLCwtTQ0KDS0lKdPXtWQUFBkqTc3FxdeeWV6tSpU7PnGBwcfNHlwMLCMtXVte4f4qa0lfXX1ze0mbV6AurhXqiHe6EeLcf4xdeMjAw98cQTGjt2rOPb23/5y18qLS1N77zzzo8ePzY2VmfPntWGDRtUW1urDz74QNu3b3fcdxUfH6/t27frgw8%2BUG1trTZs2KCioiLFxsZKksaOHas1a9bo1KlTKi8v1%2BLFizVgwAD95Cc/UY8ePRQZGanFixervLxcp06d0urVqxUfH/%2Bj5w0AANoO42ewvvrqK11//fUXtffu3Vtnz5790eMHBgbq5ZdfVlpamlauXKkrrrhC8%2BbN0y233CJJGjhwoObPn68FCxbozJkz6tWrl9atW6eAgABJ39wLVVdXpwkTJqiiokLR0dF69tlnHeOvXLlSCxcu1LBhw%2BTl5aUxY8YoMTHxR8/wsEfxAAAbvklEQVQbAAC0HcYD1jXXXKOPPvpI/%2B///T%2Bn9n379jlueL9UjZ/wa9S3b1%2B99tpr37v96NGjNXr06O/s8/HxUUpKyvdeqgwKCtLKlSv/o3kCAABILRCwJk%2BerCeffFJnzpyRZVk6ePCgXnvtNW3cuFFz5swxvTsAAAC3YzxgjRs3TnV1dVqzZo2qq6v1xBNPqGvXrpoxY4buuece07sDAABwO8YD1ttvv61f/OIXSkhIUHFxsSzLUteuXU3vBgAAwG0Z/xThU0895biZ/YorriBcAQCANsd4wOrRo8dFN6UDAAC0JcYvEYaGhiolJUUvvfSSevTooQ4dOjj1L1myxPQuAQAA3IrxgHXy5ElFRkZK0kV/8BgAAKAtMBKwlixZokcffVR%2Bfn7auHGjiSEBAAA8lpF7sF599dWL/kjy5MmTVVBQYGJ4AAAAj2IkYFmWdVHbP/7xD124cMHE8AAAAB7F%2BKcIAQAA2joCFgAAgGHGAla7du1MDQUAAODRjH1Nw1NPPeX0nVe1tbVatmyZ/P39nbbje7AAAEBrZyRg9e/f/6LvvIqIiFBJSYlKSkpM7AIAAMBjGAlYfPcVAADA/%2BEmdwAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGGftjz5fT22%2B/rfnz5zu11dbWSpI%2B/vhjTZkyRZmZmbLZ/m95zz33nIYMGaL6%2Bnqlp6frrbfeUlVVlW655RY9%2BeSTCg4OliQVFRXp8ccf16FDh%2BTt7a1Ro0Zp9uzZTmMBAAD8EI88gzVq1CgdPXrU8di9e7cCAgKUlpYm6ZuQtX79eqdthgwZIklas2aNDhw4oDfffFP79%2B%2BXr6%2Bv5s2b5xh7%2BvTp8vPz0/79%2B7VlyxYdPHhQGzZscMUyAQCAh/LIgPXvLMvSrFmzdPvtt2v06NE6deqUzp07pz59%2Bnzn9m%2B88YamTp2qq666Sh07dtTcuXP1/vvv69SpUzpx4oQOHTqkWbNmyW6369prr1ViYqI2b958mVcFAAA8mcdf93rrrbeUk5Oj1atXS5KOHz8uf39/zZgxQ8ePH1dQUJDuv/9%2BxcfHq6ysTF9//bXCwsIcrw8KClKXLl302WefSZICAgLUvXt3R39ISIjy8/N1/vx5de7cuVlzKigoUGFhoVObzebnuAxpgre352Vjm83z5nwpGmviibVpjaiHe6Ee7oV6tDyPDlgNDQ1as2aNfvOb36hjx46SpJqaGoWHh2vGjBkKDQ1VZmampk2bJn9/f0VEREiS/Pz8nMbx9fVVRUWFJMlutzv1NT6vrKxsdsDKyMjQqlWrnNqSkpKUnJx86YtsRQID/V09hcuic2d70xvhsqEe7oV6uBfq0XI8OmBlZmaqoKBA8fHxjrYxY8ZozJgxjueDBw/WmDFjtGvXLt16662SpKqqKqdxqqur5e/vL8uyLuprfO7v3/xwkJCQoJiYGKc2m81PJSUVzR6jKd7eXh53YJhcvztqrMn581Wqr29w9XTaPOrhXqiHe2lL9XDVL/ceHbD27Nmj2NhYpzNSW7Zskb%2B/v0aMGOFoq6mpUYcOHdSlSxd1795dOTk5jsuEhYWFKi0tVVhYmBoaGlRaWqqzZ88qKChIkpSbm6srr7xSnTp1ava8goODL7ocWFhYprq61v1D3JS2sv76%2BoY2s1ZPQD3cC/VwL9Sj5Xj0xdcjR46of//%2BTm3l5eVatGiR/vnPf6qhoUF/%2BctftGPHDiUkJEiSxo4dqzVr1ujUqVMqLy/X4sWLNWDAAP3kJz9Rjx49FBkZqcWLF6u8vFynTp3S6tWrnc6QAQAANMWjz2B99dVXF50pmjRpkiorK/XII4%2BoqKhI1157rZYuXaqoqChJ39wLVVdXpwkTJqiiokLR0dF69tlnHa9fuXKlFi5cqGHDhsnLy0tjxoxRYmLiZV0XAADwbO0sy7JcPYm2oLCwzOh4NpuXAgP9FTV3t9FxW9Ku6YNcPYUW1ViTkpIKTrm7AerhXqiHe2lL9ejWrfm3%2BJjk0ZcIAQAA3BEBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDCPDVg7d%2B5Unz59FBER4XjMmjVLkrRv3z7FxcUpPDxcI0aM0Hvvvef02nXr1mnIkCEKDw/XxIkT9cUXXzj6KisrNWfOHEVHRysyMlKpqamqqKi4rGsDAACezWMD1vHjxzV69GgdPXrU8Vi2bJny8vI0bdo0Pfroo/r73/%2BuadOmafr06Tpz5owkadu2bdq4caPWr1%2BvzMxM3XDDDUpOTpZlWZKkRYsW6fTp09qzZ4/27t2r06dPKz093ZVLBQAAHsajA9aNN954Ufu2bdsUFRWlO%2B64QzabTb/85S/Vv39/ZWRkSJJef/113XvvvQoNDVWHDh00c%2BZM5efnKzMzU1VVVdq%2BfbuSk5MVEBCgrl27KiUlRVu3blVVVdXlXiIAAPBQNldP4D/R0NCgTz75RHa7XS%2B99JLq6%2Bv1s5/9TCkpKcrJyVFYWJjT9r169VJWVpYkKScnR1OnTnX0%2Bfj4qEePHsrKylJAQIBqa2udXh8SEqLq6mrl5eXp%2Buuvb9b8CgoKVFhY6NRms/kpODj4P13yRby9PS8b22yeN%2BdL0VgTT6xNa0Q93Av1cC/Uo%2BV5ZMAqLi5Wnz59NHz4cK1cuVIlJSWaPXu2Zs2apZqaGtntdqftfX19VVlZKUmqqKj43v7y8nJJkp%2Bfn6OvcdtLuQ8rIyNDq1atcmpLSkpScnJy8xfZCgUG%2Brt6CpdF5872pjfCZUM93Av1cC/Uo%2BV4ZMAKCgrS5s2bHc/tdrtmzZqlu%2B66S9HR0aqurnbavrq6Wv7%2B/o5tv6%2B/MVhVVVU5tm%2B8NNixY8dmzy8hIUExMTFObTabn0pKzN0s7%2B3t5XEHhsn1u6PGmpw/X6X6%2BgZXT6fNox7uhXq4l7ZUD1f9cu%2BRASsrK0s7duzQzJkz1a5dO0lSTU2NvLy81K9fP3366adO2%2Bfk5Dju1woNDVV2draGDh0qSaqtrVVeXp7CwsLUs2dP%2Bfj4KCcnRzfddJMkKTc313EZsbmCg4MvuhxYWFimurrW/UPclLay/vr6hjazVk9APdwL9XAv1KPleOTF14CAAG3evFkvvfSS6urqlJ%2Bfr2XLlulXv/qVxowZo0OHDmnnzp2qq6vTzp07dejQIY0ePVqSNG7cOG3atElZWVm6cOGCli9frqCgIEVFRclut2vEiBFKT09XcXGxiouLlZ6erpEjR8rX19fFqwYAAJ7CI89gXXnllVq7dq1WrFihNWvWqEOHDrrzzjs1a9YsdejQQf/zP/%2Bj9PR0zZ07V9dcc42ef/559ezZU5IUHx%2BvsrIyJSUlqbi4WH379tXatWvl4%2BMjSZo/f76WLl2quLg41dbWatiwYXr88cdduVwAAOBh2lmNXwCFFlVYWGZ0PJvNS4GB/oqau9vouC1p1/RBrp5Ci2qsSUlJBafc3QD1cC/Uw720pXp069bJJfv1yEuEAAAA7oyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYJjHBqysrCw98MADGjBggAYNGqTU1FQVFxdLkubPn68bb7xRERERjkdGRobjtevWrdOQIUMUHh6uiRMn6osvvnD0VVZWas6cOYqOjlZkZKRSU1NVUVFx2dcHAAA8l0cGrOrqak2ZMkURERH661//qh07dqi0tFS/%2B93vJEnHjx/XokWLdPToUccjISFBkrRt2zZt3LhR69evV2Zmpm644QYlJyfLsixJ0qJFi3T69Gnt2bNHe/fu1enTp5Wenu6ytQIAAM/jkQErPz9f1113nZKSktS%2BfXsFBgYqISFBhw8fVk1NjT7//HPdeOON3/na119/Xffee69CQ0PVoUMHzZw5U/n5%2BcrMzFRVVZW2b9%2Bu5ORkBQQEqGvXrkpJSdHWrVtVVVV1mVcJAAA8lc3VE/hP/PSnP9VLL73k1LZnzx7dcMMNysrKUl1dnVauXKkjR46oU6dOGjdunKZMmSIvLy/l5ORo6tSpjtf5%2BPioR48eysrKUkBAgGpraxUWFuboDwkJUXV1tfLy8nT99dc3a34FBQUqLCx0arPZ/BQcHPwjVu3M29vzsrHN5nlzvhSNNfHE2rRG1MO9UA/3Qj1ankcGrH9nWZaeffZZvffee9q0aZPOnj2rAQMGaOLEiVqxYoU%2B/fRTJSUlycvLS1OmTFFFRYXsdrvTGL6%2BvqqsrFR5ebkkyc/Pz9HXuO2l3IeVkZGhVatWObUlJSUpOTn5P11mqxAY6O/qKVwWnTvbm94Ilw31cC/Uw71Qj5bj0QGrvLxcc%2BbM0SeffKJNmzapd%2B/e6t27twYNGuTYpl%2B/fpo0aZJ27typKVOmyG63q7q62mmc6upq%2Bfv7O4JVVVWV/P39Hf%2BWpI4dOzZ7XgkJCYqJiXFqs9n8VFJi7mZ5b28vjzswTK7fHTXW5Pz5KtXXN7h6Om0e9XAv1MO9tKV6uOqXe48NWCdPntTUqVN19dVXa8uWLbriiiskSe%2B%2B%2B67Onj2ru%2B%2B%2B27FtTU2NfH19JUmhoaHKzs7W0KFDJUm1tbXKy8tTWFiYevbsKR8fH%2BXk5Oimm26SJOXm5jouIzZXcHDwRZcDCwvLVFfXun%2BIm9JW1l9f39Bm1uoJqId7oR7uhXq0HI%2B8%2BHru3DlNmjRJN998s9avX%2B8IV9I3lwyXLFmigwcPyrIsHT16VK%2B%2B%2BqrjU4Tjxo3Tpk2blJWVpQsXLmj58uUKCgpSVFSU7Ha7RowYofT0dBUXF6u4uFjp6ekaOXKkI6ABAAA0xSPPYG3dulX5%2BfnatWuXdu/e7dR39OhRzZkzRwsWLNCZM2cUFBSkadOmafTo0ZKk%2BPh4lZWVKSkpScXFxerbt6/Wrl0rHx8fSd98h9bSpUsVFxen2tpaDRs2TI8//vhlXyMAAPBc7azGL4BCiyosLDM6ns3mpcBAf0XN3d30xm5i1/RBTW/kwRprUlJSwSl3N0A93Av1cC9tqR7dunVyyX498hIhAACAOyNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgfYeioiIlJiYqKipK0dHRSktLU11dnaunBQAAPAQB6ztMnz5dfn5%2B2r9/v7Zs2aKDBw9qw4YNrp4WAADwEDZXT8DdnDhxQocOHdL7778vu92ua6%2B9VomJiVq2bJmmTJni6ul5tBHPHnD1FC7JrumDXD0FAICHImB9S3Z2tgICAtS9e3dHW0hIiPLz83X%2B/Hl17tzZhbPD5UQgBAD8pwhY31JRUSG73e7U1vi8srKyWQGroKBAhYWFTm02m5%2BCg4ONzdPbm6u7cGaz8TPx7xqPEY4V90A93Av1aHkErG/x8/NTVVWVU1vjc39//2aNkZGRoVWrVjm1PfLII5o2bZqZSeqbEPeHP7yknY8mGA1u%2BM8VFBQoIyNDCQnUxB00HiPUwz1QD/dCPVoe0fVbQkNDVVpaqrNnzzracnNzdeWVV6pTp07NGiMhIUFbt251eiQkJBidZ2FhoVatWnXRmTK4DjVxL9TDvVAP90I9Wh5nsL6lR48eioyM1OLFi7Vw4UKVlJRo9erVio%2BPb/YYwcHB/EYAAEAbxhms77By5UrV1dVp2LBhuuuuu3TbbbcpMTHR1dMCAAAegjNY3yEoKEgrV6509TQAAICH8l6wYMECV08C/xl/f38NGDCg2Tffo%2BVRE/dCPdwL9XAv1KNltbMsy3L1JAAAAFoT7sECAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbA8kBFRUVKTExUVFSUoqOjlZaWprq6OldPy6NkZWXpgQce0IABAzRo0CClpqaquLhYknTs2DGNHz9eERERiomJ0RtvvOH02m3btik2Nlbh4eEaO3asjh496uirr6/X0qVLdeuttyoiIkIPP/ywCgoKHP1N1a6pfbd29fX1mjhxoh577DFH2759%2BxQXF6fw8HCNGDFC7733ntNr1q1bpyFDhig8PFwTJ07UF1984eirrKzUnDlzFB0drcjISKWmpqqiosLR/%2BWXX2rSpEmKiIjQ4MGD9cILLziN3dS%2BW6vS0lKlpqYqOjpa/fv3V2JiouPnmOPDNT755BNNmDBBUVFRGjx4sJ566inV1NRI4hhxWxY8zn333WfNnDnTqqystE6ePGndeeed1rp161w9LY9RVVVlDRo0yHruueesCxcuWMXFxdbUqVOt//7v/7ZKS0utAQMGWJs2bbJqa2utv/3tb1ZERIR17Ngxy7Is64MPPrAiIiKsv//971ZNTY31yiuvWNHR0VZlZaVlWZb1/PPPW3FxcVZ%2Bfr5VVlZmTZ8%2B3Zo6dapj3z9Uu6b23RY8%2B%2Byz1nXXXWfNnj3bsizL%2BvLLL62%2Bfftaf/rTn6za2lrrnXfesfr162d9/fXXlmVZ1tatW63bbrvN%2Bvzzz63q6mpryZIl1p133mk1NDRYlmVZjz32mDVp0iSrpKTEOnv2rHXfffdZCxYssCzLsmpqaqyf//zn1rJly6wLFy5Yn3zyiTV48GBr586dzdp3a3bfffdZSUlJ1rlz56yysjLrkUcesR566CGODxepr6%2B3Bg0aZP3hD3%2Bw6uvrrdOnT1vDhw%2B3Vq1axTHixghYHiYvL88KCwtz%2BgF%2B5513rNtvv92Fs/Isubm51uTJk626ujpH27vvvmvdfPPN1uuvv279/Oc/d9r%2BiSeesFJTUy3LsqyZM2da8%2BbNc%2Br/xS9%2BYW3ZssWyLMsaMmSI9fbbbzv6CgsLrd69e1snT55ssnZN7bu1%2B9vf/mb98pe/tJKTkx0Ba8WKFdYDDzzgtN3kyZOt5557zrIsy7r77rutNWvWOPpqamqsiIgI6%2BDBg1ZlZaV1ww03WEeOHHH0f/jhh1a/fv2syspK68CBA1Z4eLh14cIFR//atWutCRMmNGvfrdXx48etvn37WmVlZY62kpIS6/PPP%2Bf4cJHi4mIrLCzMeuWVV6y6ujrr9OnT1ogRI6z169dzjLgxLhF6mOzsbAUEBKh79%2B6OtpCQEOXn5%2Bv8%2BfMunJnn%2BOlPf6qXXnpJ3t7ejrY9e/bohhtuUHZ2tsLCwpy279Wrl7KysiRJOTk539tfVlamr7/%2B2qk/KChIXbp00WeffdZk7Zrad2tWVFSkuXPnavny5bLb7Y72H3q/v6vfx8dHPXr0UFZWlk6cOKHa2lqn/pCQEFVXVysvL0/Z2dnq2bOn2rdv36yxv93fWn300Ufq1auXXn/9dcXGxmrw4MFaunSpunXrxvHhIoGBgbr//vu1dOlS9e3bVz/72c/Uo0cP3X///RwjboyA5WEqKiqc/gckyfG8srLSFVPyaJZl6ZlnntF7772nuXPnfuf76%2Bvr63hvf6i/8b4FPz%2B/i/orKiqarF1T%2B26tGhoaNGvWLD3wwAO67rrrnPp%2BTD3Ky8slOdejcdsfqkdzxm7Nzp07p88%2B%2B0x5eXnatm2b/vjHP%2BrMmTOaPXs2x4eLNDQ0yNfXV48//rg%2B/PBD7dixQ7m5uVq5ciXHiBsjYHkYPz8/VVVVObU1Pvf393fFlDxWeXm5kpOTtX37dm3atEm9e/eW3W5XdXW103bV1dWO9/aH%2Bhv/Q/Pt%2BjT2N1W7pvbdWq1du1bt27fXxIkTL%2Br7MfVo/J/Gv7/njf/u2LHj99ajOWO3Zo1nK%2BbOnauOHTsqKChI06dP1759%2B2RZFseHC/zpT3/Snj17dO%2B996p9%2B/YKDQ1VUlKS/vd//5djxI0RsDxMaGioSktLdfbsWUdbbm6urrzySnXq1MmFM/MsJ0%2Be1Lhx41ReXq4tW7aod%2B/ekqSwsDBlZ2c7bZuTk6PQ0FBJ37z/39ffpUsXde/eXTk5OY6%2BwsJClZaWKiwsrMnaNbXv1uqtt97SoUOHFBUVpaioKO3YsUM7duxQVFTUJdejtrZWeXl5CgsLU8%2BePeXj4%2BNUj9zcXMclktDQUOXl5Tl9Su3fx26r9ejVq5caGhpUW1vraGtoaJAkXX/99RwfLnD69GnHJwYb2Ww2%2Bfj4cIy4M1ffBIZLd88991gzZsywysrKHJ%2B0Wblypaun5TFKS0ut22%2B/3Xrssces%2Bvp6p77i4mIrKirKeuWVV6yamhrr4MGDjhtCLctyfHLp4MGDjk9J9e/f3yopKbEsy7KeeeYZa%2BTIkdbJkycdn5K67777HOP/UO2a2ndbMXv2bMdN7jk5OVbfvn2td955x/Eppb59%2B1pffPGFZVnf3Ph82223WZ9%2B%2BqnjE1KxsbFWTU2NZVmWlZKSYt13331WUVGRVVRUZN13332OsWtra62YmBjr6aeftqqrq61PP/3UGjx4sPXmm282a9%2BtVU1NjRUbG2tNmzbNKi8vt4qKiqxf//rXVlJSEseHi2RnZ1s33nijtWbNGquurs46efKkNXLkSOvpp5/mGHFjBCwPVFhYaE2bNs0aMGCAdcstt1hPP/200yfi8MNefvllKywszLrpppus8PBwp4dlWdZHH31kJSQkWBEREdawYcMc/zFp9Mc//tEaPny4FR4ebsXHx1sffviho6%2BmpsZatmyZddttt1k333yz9fDDD1tnz5519DdVu6b23Rb8e8CyLMt6//33rVGjRlnh4eHWnXfeaf3lL39x9DU0NFjr16%2B3YmJirPDwcGvixIlO/3EvKyuz5s2bZ916661W//79rccee8yqqKhw9Ofl5VkPPvigFRkZad12223W2rVrnebyQ/tuzb7%2B%2Bmtr%2BvTp1qBBg6yoqCgrNTXVOnfunGVZHB%2BucuDAAWv8%2BPFWZGSkdfvtt1srVqxwfLqPY8Q9tbMsy3L1WTQAAIDWhHuwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADD/j%2B87pzlVqQa1QAAAABJRU5ErkJggg%3D%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-549848781125627893">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">5000</td>
        <td class="number">427</td>
        <td class="number">2.0%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4000</td>
        <td class="number">356</td>
        <td class="number">1.6%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">6000</td>
        <td class="number">288</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7200</td>
        <td class="number">210</td>
        <td class="number">1.0%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4800</td>
        <td class="number">145</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">7500</td>
        <td class="number">142</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">8400</td>
        <td class="number">116</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4500</td>
        <td class="number">111</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3600</td>
        <td class="number">111</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">5100</td>
        <td class="number">109</td>
        <td class="number">0.5%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (8672)</td>
        <td class="number">19582</td>
        <td class="number">90.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-549848781125627893">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">651</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:25%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">659</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:25%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">660</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:25%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">748</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:50%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">750</td>
        <td class="number">4</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">434728</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">438213</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">560617</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">858132</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">871200</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_view">view<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>6</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.3%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>63</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>0.23386</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>4</td>
                </tr>
                <tr class="alert">
                    <th>Zeros (%)</th>
                    <td>89.9%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-3721154334349679714">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAARZJREFUeJzt1sENgkAARUEwlGQR9OSZnijCntazxrzEg2J05r7Jh%2BSxzGOMMQFPnY4eAN9sOXrAo/Nlf/nMdVvfsATcIJAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBCWowdw73zZXz5z3dY3LGGa3CCQ/vYG%2Becv9aee/Rfe8TzGGEePgG/lFwuCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCATCDRGWF5PxsU2VAAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-3721154334349679714,#minihistogram-3721154334349679714"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-3721154334349679714">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-3721154334349679714"
                                                  aria-controls="quantiles-3721154334349679714" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-3721154334349679714" aria-controls="histogram-3721154334349679714"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-3721154334349679714" aria-controls="common-3721154334349679714"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-3721154334349679714" aria-controls="extreme-3721154334349679714"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-3721154334349679714">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>2</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>4</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>4</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>0</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>0.76569</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>3.2741</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>10.92</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>0.23386</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>0.42185</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>3.3995</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>5036</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>0.58628</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-3721154334349679714">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3X9wVOWh//EPySZksxAShYD2ixcaiAyIJiUQFKQSiBYQSBHNrcooM1o1MREqAS1yURlQaqwCAQRUmKuZKYJShILgXEFyLSZgqQIdvAQBoyBZw8/8Ir/O9w9vtu4NyAaeZHNO36%2BZHd3nPOfk%2BeSw8bPnbLCdZVmWAAAAYExIsBcAAADgNBQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGCYK9gL%2BFfh9Z4zfsyQkHa66iqPTp6sUEODZfz4weLUXJJzszk1l0Q2O3JqLsm52VoyV5cuHY0eL1BcwbKxkJB2ateunUJC2gV7KUY5NZfk3GxOzSWRzY6cmktybjYn5qJgAQAAGEbBAgAAMIyCBQAAYBgfcm8BpaWl8nq9fmMuV6RiY2ONfp3Q0BC/fzqFU3NJzs3m1FwS2ezIqbkk52ZzYq52lmU559cQ2ohFixYpLy/PbywzM1PZ2dlBWhEAAGhNFKwW0JpXsKKi3Dp7tkr19Q1Gjx1MTs0lOTebU3NJZLMjp%2BaSnJutJXPFxHiMHi9Q3CJsAbGxsU3KlNd7TnV1LfNiqK9vaLFjB5NTc0nOzebUXBLZ7MipuSTnZnNSLufc7AQAAGgjKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGH9Ng80lzfwg2EsI2OYpQ4K9BAAAWgVXsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAxzBXsBF/L%2B%2B%2B9r9uzZfmO1tbWSpH379kmSHnroIRUWFsrl%2BmeEBQsWaNiwYaqvr1dubq7Wr1%2BvqqoqDR48WM8995xiY2MlSWVlZZo1a5aKiooUGhqqcePGacaMGX7HAgAAuFxt8grWuHHjtGfPHt/jgw8%2BUHR0tObOneubs2/fPr3xxht%2B84YNGyZJWrp0qT755BO9%2B%2B67KigoUEREhJ555hnfvlOmTFFkZKQKCgq0du1a7dy5U6tWrWrtmAAAwKHaZMH6McuylJOTo9tuu03jx4%2BXJJWUlOjMmTPq27fvBfdZs2aNHn74YV1zzTXq0KGDZs6cqR07dqikpERHjx5VUVGRcnJy5Ha71b17d2VkZCg/P781YwEAAAdr8/fE1q9fr%2BLiYi1ZssQ3tnfvXnk8Hk2dOlV79%2B5V586d9eCDD2rixIk6d%2B6cvvvuO8XHx/vmd%2B7cWZ06ddKXX34pSYqOjlbXrl192%2BPi4nTs2DGdPXtWUVFRV7zm0tJSeb1evzGXK9J3i9KU0NA234/9uFyBrbcxl93yBcKp2ZyaSyKbHTk1l%2BTcbE7M1aYLVkNDg5YuXapHH31UHTp08I3X1NQoISFBU6dOVe/evVVYWKisrCx5PB4lJiZKkiIjI/2OFRERoYqKCkmS2%2B3229b4vLKy0kjBWr16tfLy8vzGMjMzlZ2dfcXHtrOYGE%2Bz5kdFuS89yaacms2puSSy2ZFTc0nOzeakXG26YBUWFqq0tFQTJ070G09LS1NaWprv%2BdChQ5WWlqbNmzfrlltukSRVVVX57VNdXS2PxyPLsppsa3zu8TSvAFxMenq6UlJS/MZcrkidOlVh5PiN7Nb0A80fGhqiqCi3zp6tUn19QwuvqnU5NZtTc0lksyOn5pKcm60lczX3zb0pbbpgbdmyRampqU2uRq1du1Yej0ejRo3yjdXU1Kh9%2B/bq1KmTunbtquLiYt9tQq/Xq9OnTys%2BPl4NDQ06ffq0vv/%2Be3Xu3FmSdOjQIXXr1k0dO3Y0su7Y2NgmtwO93nOqq3POi%2BFyNDd/fX2DY79nTs3m1FwS2ezIqbkk52ZzUq42fQnks88%2B08CBA5uMl5eXa86cOfrHP/6hhoYGbd%2B%2BXRs3blR6erokacKECVq6dKlKSkpUXl6uefPmadCgQbruuuvUo0cPDRgwQPPmzVN5eblKSkq0ZMmSJlfJAAAALlebvoL1zTffXPCD4Q888IAqKyv1%2BOOPq6ysTN27d9f8%2BfOVlJQk6YfPO9XV1em%2B%2B%2B5TRUWFkpOT9eqrr/r2X7hwoZ5//nmNGDFCISEhSktLU0ZGRqvlAgAAztbOsiwr2Iv4V%2BD1njN%2BTJcrRKm5BcaP21I2TxkS0DyXK0QxMR6dOlXhmEvFjZyazam5JLLZkVNzSc7N1pK5unQx8/Gf5mrTtwgBAADsiIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAw9pswdq0aZP69u2rxMRE3yMnJ8e3/eOPP9bYsWOVkJCgUaNGadu2bX77r1ixQsOGDVNCQoImTZqkr776yretsrJSTz/9tJKTkzVgwABNnz5dFRUVrZYNAAA4W5stWHv37tX48eO1Z88e3%2BOll16SJB05ckRZWVl64okntHv3bmVlZWnKlCk6ceKEJGndunV666239MYbb6iwsFD9%2BvVTdna2LMuSJM2ZM0fHjx/Xli1btHXrVh0/fly5ublBywoAAJylTResG2644YLb1q1bp6SkJI0cOVIul0ujR4/WwIEDtXr1aknSO%2B%2B8o3vvvVe9e/dW%2B/bt9eSTT%2BrYsWMqLCxUVVWVNmzYoOzsbEVHR%2Bvqq6/WtGnT9N5776mqqqo1IwIAAIdqkwWroaFB%2B/fv1/bt2zV8%2BHANGzZMs2bN0pkzZyRJxcXFio%2BP99unV69eOnDgwAW3h4WFqUePHjpw4ICOHj2q2tpav%2B1xcXGqrq7WkSNHWj4cAABwPFewF3AhJ0%2BeVN%2B%2BfXXHHXdo4cKFOnXqlGbMmKGcnBwtX75cFRUVcrvdfvtERESosrJSkn5ye3l5uSQpMjLSt61xrqnPYZWWlsrr9fqNuVyRio2NNXL8RqGhbbIfX5TLFdh6G3PZLV8gnJrNqbkkstmRU3NJzs3mxFxtsmB17txZ%2Bfn5vudut1s5OTm65557VF5eLrfbrerqar99qqur5fF4fPMvtr2xWFVVVfnmN94a7NChg5H1r169Wnl5eX5jmZmZys7ONnJ8u4qJ8TRrflSU%2B9KTbMqp2ZyaSyKbHTk1l%2BTcbE7K1SYL1oEDB7Rx40Y9%2BeSTateunSSppqZGISEhCg8PV3x8vPbv3%2B%2B3T3Fxse8zW71799bBgwc1fPhwSVJtba2OHDmi%2BPh49ezZU2FhYSouLtZNN90kSTp06JDvNqIJ6enpSklJ8RtzuSJ16pTZ31S0W9MPNH9oaIiiotw6e7ZK9fUNLbyq1uXUbE7NJZHNjpyaS3JutpbM1dw396a0yYIVHR2t/Px8derUSZMnT1Zpaaleeukl/frXv1Z4eLjGjRunlStXatOmTbr99tu1detWFRUVaebMmZKku%2B66S4sWLdKwYcPUs2dPvfLKK%2BrcubOSkpIUFhamUaNGKTc3VwsWLJAk5ebm6s4771RERISR9cfGxja5Hej1nlNdnXNeDJejufnr6xsc%2Bz1zajan5pLIZkdOzSU5N5uTcrXJgtWtWzctW7ZMf/zjH7V06VK1b99eY8aM8f09WHFxcVq8eLFyc3M1c%2BZM/exnP9OiRYvUs2dPSdLEiRN17tw5ZWZm6uTJk%2Brfv7%2BWLVumsLAwSdLs2bM1f/58jR07VrW1tRoxYoRmzZoVtLwAAMBZ2lmNfzkUWpTXe874MV2uEKXmFhg/bkvZPGVIQPNcrhDFxHh06lSFY97JNHJqNqfmkshmR07NJTk3W0vm6tKlo9HjBcpeH%2BIBAACwAQoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADGvzBevkyZNKTU1VYWGhb2zLli0aP368fvGLXyglJUV5eXlqaGjwbR81apRuuukmJSYm%2Bh6HDh2SJFVWVurpp59WcnKyBgwYoOnTp6uiosK37%2BHDh/XAAw8oMTFRQ4cO1WuvvdZ6YQEAgCO06YL12WefKT09XV9//bVvbN%2B%2BfZo%2BfbqmTJmi3bt3a8WKFXrvvfe0atUqSVJ5ebkOHz6sTZs2ac%2BePb5HXFycJGnOnDk6fvy4tmzZoq1bt%2Br48ePKzc2VJNXW1urRRx9V//79VVhYqOXLlys/P1%2BbN29u9ewAAMC%2B2mzBWrdunaZNm6apU6f6jX/77bf693//dw0fPlwhISGKi4tTamqqdu3aJemHAhYdHa2f/exnTY5ZVVWlDRs2KDs7W9HR0br66qs1bdo0vffee6qqqtKuXbtUWlqq7OxshYeHq2/fvpo0aZLy8/NbJTMAAHCGNluwhg4dqg8//FCjR4/2G7/jjjv09NNP%2B55XV1dr%2B/bt6tevnyRp7969crvduv/%2B%2B5WcnKwJEyZo27ZtkqSjR4%2BqtrZW8fHxvv3j4uJUXV2tI0eO6ODBg%2BrZs6fCw8N923v16qUDBw60ZFQAAOAwrmAv4GK6dOlyyTnl5eV64oknFBERoQcffFCS1K5dO/Xv31%2B/%2B93vdO211%2BqDDz5QVlaW3n77bdXV1UmSIiMjfcdwu92SpIqKClVUVPie/3h7ZWVls9ZeWloqr9frN%2BZyRSo2NrZZx7mU0NA2248vyOUKbL2NueyWLxBOzebUXBLZ7MipuSTnZnNirjZbsC7lq6%2B%2BUnZ2tq6%2B%2Bmr953/%2Bpzp06CBJeuihh/zmjRs3Ths3btSWLVs0duxYST/cKvR4PL5/l6QOHTooMjLS97zRj%2BcGavXq1crLy/Mby8zMVHZ2drOO4zQxMc37PkZFuS89yaacms2puSSy2ZFTc0nOzeakXLYsWB9//LF%2B97vf6Z577tGTTz4pl%2BufMd544w317dtXN998s2%2BspqZG7du3V8%2BePRUWFqbi4mLddNNNkqRDhw4pLCxMPXr0UFlZmY4cOaK6ujrfMYuLi9W7d%2B9mrS89PV0pKSl%2BYy5XpE6dqrjIHpfHbk0/0PyhoSGKinLr7Nkq1dc3XHoHG3FqNqfmkshmR07NJTk3W0vmau6be1NsV7D%2B/ve/KzMzU88%2B%2B6wmTpzYZPvx48e1Zs0arVixQtdcc43%2B/Oc/a8%2BePXruuefkdrs1atQo5ebmasGCBZKk3Nxc3XnnnYqIiFBycrJiYmL08ssva8qUKTp8%2BLDeeuutJh%2B0v5TY2NgmtwO93nOqq3POi%2BFyNDd/fX2DY79nTs3m1FwS2ezIqbkk52ZzUi7bFazXXntNdXV1mjt3rubOnesbHzBggF5//XVNnz5dISEhuvfee3Xu3Dn16tVLy5cv17/9279JkmbPnq358%2Bdr7Nixqq2t1YgRIzRr1ixJksvl0ptvvqnnn39eQ4YMUWRkpCZNmqQJEyYEJSsAALCndpZlWcFexL8Cr/ec8WO6XCFKzS0wftyWsnnKkIDmuVwhionx6NSpCse8k2nk1GxOzSWRzY6cmktybraWzNWlS0ejxwuUvT7EAwAAYAMULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCszReskydPKjU1VYWFhb6xzz//XHfffbcSExOVkpKiNWvW%2BO2zbt06paamKiEhQRMmTNCePXt82%2Brr6zV//nzdcsstSkxM1GOPPabS0lLf9rKyMmVkZCgpKUnJycmaO3eu6urqWj4oAABwjDZdsD777DOlp6fr66%2B/9o2dOXNGv/3tb5WWlqZdu3Zp7ty5euGFF/TFF19IkgoLCzVnzhy9%2BOKL2rVrl8aNG6fHHntMVVVVkqSlS5fqk08%2B0bvvvquCggJFRETomWee8R1/ypQpioyMVEFBgdauXaudO3dq1apVrZobAADYW5stWOvWrdO0adM0depUv/GtW7cqOjpa9913n1wul26%2B%2BWaNHTtW%2Bfn5kqQ1a9ZozJgxGjBggMLCwvTggw8qJiZGmzZt8m1/%2BOGHdc0116hDhw6aOXOmduzYoZKSEh09elRFRUXKycmR2%2B1W9%2B7dlZGR4Ts2AABAIFzBXsDFDB06VGPHjpXL5fIrWQcPHlR8fLzf3F69emnt2rWSpOLiYt11111Nth84cEDnzp3Td99957d/586d1alTJ3355ZeSpOjoaHXt2tW3PS4uTseOHdPZs2cVFRUV0NpLS0vl9Xr9xlyuSMXGxga0f6BCQ9tsP74glyuw9Tbmslu%2BQDg1m1NzSWSzI6fmkpybzYm52mzB6tKlywXHKyoq5Ha7/cYiIiJUWVl5ye0VFRWSpMjIyCbbG7f9330bn1dWVgZcsFavXq28vDy/sczMTGVnZwe0v1PFxHiaNT8qyn3pSTbl1GxOzSWRzY6cmktybjYn5WqzBeti3G63zp075zdWXV0tj8fj215dXd1ke0xMjK8sNX4e6//ub1lWk22NzxuPH4j09HSlpKT4jblckTp1qiLgYwTCbk0/0PyhoSGKinLr7Nkq1dc3tPCqWpdTszk1l0Q2O3JqLsm52VoyV3Pf3Jtiu4IVHx%2BvTz75xG%2BsuLhYvXv3liT17t1bBw8ebLJ92LBh6tSpk7p27ari4mLfbUKv16vTp08rPj5eDQ0NOn36tL7//nt17txZknTo0CF169ZNHTt2DHiNsbGxTW4Her3nVFfnnBfD5Whu/vr6Bsd%2Bz5yazam5JLLZkVNzSc7N5qRc9roEIik1NVXff/%2B9Vq1apdraWn366afasGGD73NXEydO1IYNG/Tpp5%2BqtrZWq1atUllZmVJTUyVJEyZM0NKlS1VSUqLy8nLNmzdPgwYN0nXXXacePXpowIABmjdvnsrLy1VSUqIlS5Zo4sSJwYwMAABsxnZXsGJiYvTmm29q7ty5Wrhwoa666io988wzGjx4sCTp5ptv1uzZs/Xss8/qxIkT6tWrl1asWKHo6GhJP3wWqq6uTvfdd58qKiqUnJysV1991Xf8hQsX6vnnn9eIESMUEhKitLQ0ZWRkBCUrAACwp3aWZVnBXsS/Aq/33KUnNZPLFaLU3ALjx20pm6cMCWieyxWimBiPTp2qcMyl4kZOzebUXBLZ7MipuSTnZmvJXF26BP4RH5Nsd4sQAACgraNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDBXsBfQHO%2B//75mz57tN1ZbWytJ2rdvnx566CEVFhbK5fpnrAULFmjYsGGqr69Xbm6u1q9fr6qqKg0ePFjPPfecYmNjJUllZWWaNWuWioqKFBoaqnHjxmnGjBl%2BxwIAAAiEra5gjRs3Tnv27PE9PvjgA0VHR2vu3LmSfihZb7zxht%2BcYcOGSZKWLl2qTz75RO%2B%2B%2B64KCgoUERGhZ555xnfsKVOmKDIyUgUFBVq7dq127typVatWBSMmAACwOVsVrB%2BzLEs5OTm67bbbNH78eJWUlOjMmTPq27fvBeevWbNGDz/8sK655hp16NBBM2fO1I4dO1RSUqKjR4%2BqqKhIOTk5crvd6t69uzIyMpSfn9/KqQAAgBPY9v7X%2BvXrVVxcrCVLlkiS9u7dK4/Ho6lTp2rv3r3q3LmzHnzwQU2cOFHnzp3Td999p/j4eN/%2BnTt3VqdOnfTll19KkqKjo9W1a1ff9ri4OB07dkxnz55VVFRUs9ZWWloqr9frN%2BZyRfpuR5oSGmqvfuxyBbbexlx2yxcIp2Zzai6JbHbk1FySc7M5MZctC1ZDQ4OWLl2qRx99VB06dJAk1dTUKCEhQVOnTlXv3r1VWFiorKwseTweJSYmSpIiIyP9jhMREaGKigpJktvt9tvW%2BLyysrLZBWv16tXKy8vzG8vMzFR2dnazjuM0MTGeZs2PinJfepJNOTWbU3NJZLMjp%2BaSnJvNSblsWbAKCwtVWlqqiRMn%2BsbS0tKUlpbmez506FClpaVp8%2BbNuuWWWyRJVVVVfseprq6Wx%2BORZVlNtjU%2B93iaVwokKT09XSkpKX5jLlekTp2qaPaxfordmn6g%2BUNDQxQV5dbZs1Wqr29o4VW1Lqdmc2ouiWx25NRcknOztWSu5r65N8WWBWvLli1KTU31uyK1du1aeTwejRo1yjdWU1Oj9u3bq1OnTuratauKi4t9twm9Xq9Onz6t%2BPh4NTQ06PTp0/r%2B%2B%2B/VuXNnSdKhQ4fUrVs3dezYsdnri42NbXI70Os9p7o657wYLkdz89fXNzj2e%2BbUbE7NJZHNjpyaS3JuNiflstclkP/12WefaeDAgX5j5eXlmjNnjv7xj3%2BooaFB27dv18aNG5Weni5JmjBhgpYuXaqSkhKVl5dr3rx5GjRokK677jr16NFDAwYM0Lx581ReXq6SkhItWbLE7woZAABAoGx5Beubb75pcoXogQceUGVlpR5//HGVlZWpe/fumj9/vpKSkiT98Bmouro63XfffaqoqFBycrJeffVV3/4LFy7U888/rxEjRigkJERpaWnKyMho1VwAAMAZ2lmWZQV7Ef8KvN5zxo/pcoUoNbfA%2BHFbyuYpQwKa53KFKCbGo1OnKhxzqbiRU7M5NZdENjtyai7JudlaMleXLs3/qI8JtrxFCAAA0JZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGCYbQvWpk2b1LdvXyUmJvoeOTk5vu0ff/yxxo4dq4SEBI0aNUrbtm0L4mphQlVVlebNe06jR4/QHXf8UnPm/IcqKyuDvSwAAJqwbcHau3evxo8frz179vgeL730kiTpyJEjysrK0hNPPKHdu3crKytLU6ZM0YkTJ4K8alyJV175g06cOKE//ek9/elP63TixHdaunRRsJcFAEATti5YN9xwwwW3rVu3TklJSRo5cqRcLpdGjx6tgQMHavXq1a28SphSXV2trVs366GHHlFUVCfFxFylxx7L1qZN76u6ujrYywMAwI8tC1ZDQ4P279%2Bv7du3a/jw4Ro2bJhmzZqlM2fOSJKKi4sVHx/vt0%2BvXr104MCBYCwXBpSUfK26ujrFxfXyjfXs2VPnz59XScnRIK4MAICmXMFewOU4efKk%2BvbtqzvuuEMLFy7UqVOnNGPGDOXk5Gj58uWqqKiQ2%2B322yciIqLVPq9TWloqr9frN%2BZyRSo2Ntbo1wkNtVc/drkCW29jrh/nO3%2B%2BSpLUoYNHISE/jHs8kf%2B7rTrgYwfbhbI5gVNzSWSzI6fmkpybzYm5bFmwOnfurPz8fN9zt9utnJwc3XPPPSovL5fb7W5y26i6uloej6dV1rd69Wrl5eX5jWVmZio7O7tVvn5bFRPTvO9/VNQ/S3LXrldJkiIiQnznsby8XJJ0zTWdm33sYPtxNidxai6JbHbk1FySc7M5KZctC9aBAwe0ceNGPfnkk2rXrp0kqaamRiEhIQoPD1d8fLz279/vt09xcfFFP7NlWnp6ulJSUvzGXK5InTpVYfTr2K3pB5o/NDREUVFunT1bpfr6BklSTEysXC6X/va3vbrhhv6SpH379iksLEydOnUx/r1tKRfK5gROzSWRzY6cmktybraWzBWsN%2BC2LFjR0dHKz89Xp06dNHnyZJWWluqll17Sr3/9a4WHh2vcuHFauXKlNm3apNtvv11bt25VUVGRZs6c2Srri42NbXI70Os9p7o657wYLkdz89fXN/j2cbnaa8SIVC1evFDPP/%2BiJGnx4oUaOfIOuVzhtvve/jibkzg1l0Q2O3JqLsm52ZyUy16XQP5Xt27dtGzZMv3Xf/2XBg0apLvuukv9%2B/fXf/zHf0iS4uLitHjxYi1btkwDBw7UkiVLtGjRIvXs2TNvX4TwAAAQtUlEQVTIK8eVePLJp/T//t91euCBf9e9996la665Vr/73YxgLwsAgCZseQVLkgYNGqQ//elPF91%2B66236tZbb23FFaGlRUZ6NGPGTEmtcyUSAIDLZcsrWAAAAG0ZBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDbFuwDhw4oMmTJ2vQoEEaMmSIpk%2BfrpMnT/q2L1%2B%2BXP369VNiYqLv8corr/gdo7KyUk8//bSSk5M1YMAATZ8%2BXRUVFb7thw8f1gMPPKDExEQNHTpUr732WqvlAwAA9mXLglVdXa2HHnpIiYmJ%2Bu///m9t3LhRp0%2Bf1u9//3vfnH379umxxx7Tnj17fI%2BpU6f6HWfOnDk6fvy4tmzZoq1bt%2Br48ePKzc2VJNXW1urRRx9V//79VVhYqOXLlys/P1%2BbN29u1awAAMB%2BbFmwjh07pj59%2BigzM1Ph4eGKiYlRenq6du3a5Zuzd%2B9e3XDDDRc9RlVVlTZs2KDs7GxFR0fr6quv1rRp0/Tee%2B%2BpqqpKu3btUmlpqbKzsxUeHq6%2Bfftq0qRJys/Pb42IAADAxmxZsH7%2B85/r9ddfV2hoqG9sy5Yt6tevnySprKxMx44d0zvvvKOhQ4cqJSVFf/jDH3T%2B/Hnf/KNHj6q2tlbx8fG%2Bsbi4OFVXV%2BvIkSM6ePCgevbsqfDwcN/2Xr166cCBA62QEAAA2JktC9aPWZalV155Rdu2bdPMmTMlSV6vV0lJSZowYYI%2B%2BugjrVixQgUFBXrxxRd9%2B5WXl0uSIiMjfWNut1uSVFFRoYqKCt/zH2%2BvrKxs6UgAAMDmXMFewJUoLy/X008/rf379%2Bvtt9/W9ddfL0nq06eP3628uLg4ZWRk6Nlnn9Xs2bMl/bNYVVVVyePx%2BP5dkjp06KDIyEjf80Y/nvtTSktL5fV6/cZcrkjFxsZeZtILCw21Vz92uQJbb2Muu%2BULhFOzOTWXRDY7cmouybnZnJjLtgXr66%2B/1sMPP6xrr71Wa9eu1VVXXeXbVlRUpD179uiRRx7xjdXU1CgiIsL3vGfPngoLC1NxcbFuuukmSdKhQ4cUFhamHj16qKysTEeOHFFdXZ1crh%2B%2BTcXFxerdu/cl17Z69Wrl5eX5jWVmZio7O/uKMttdTMyly%2BmPRUW5Lz3Jppyazam5JLLZkVNzSc7N5qRctixYZ86c0QMPPKDBgwdr7ty5Cgnxb7xut1uLFi3StddeqzFjxujQoUNasmSJ0tPT/eaMGjVKubm5WrBggSQpNzdXd955pyIiIpScnKyYmBi9/PLLmjJlig4fPqy33nqryW8iXkh6erpSUlL8xlyuSJ06VXGRPS6P3Zp%2BoPlDQ0MUFeXW2bNVqq9vaOFVtS6nZnNqLolsduTUXJJzs7Vkrua%2BuTelnWVZVlC%2B8hVYuXKlXnzxRbndbrVr185v2549eyRJW7duVV5enkpKStSxY0fdc889ysjI8Ctj5eXlmj9/vj766CPV1tZqxIgRmjVrlu/24dGjR/X888/r888/V2RkpO6//3799re/vaw1e73nLjPtxblcIUrNLTB%2B3JayecqQgOa5XCGKifHo1KkK1dU55weI5NxsTs0lkc2OnJpLcm62lszVpUtHo8cLlC0Llh1RsChYknOzOTWXRDY7cmouybnZnFiw7HWPCQAAwAYoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGOYK9gIAoK1LmvlBsJfQLJunDAn2EoB/eVzBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIbx1zQAAIJm1KufBHsJzcJfgYFAcQULAADAMAoWAACAYdwiBC6CWxcA7M5OP8d2z/1VsJdgFFewAAAADKNgAQAAGEbBAgAAMIzPYLWA0tJSeb1evzGXK1KxsbFGv05oqL36scsV2Hobc9ktX7AF%2Bv1tCU4%2BZ3bMxGut5QTzdSY5/5w5KVc7y7KsYC/CaRYtWqS8vDy/sccff1xZWVlGv05paalWr16t9PR04%2BUtmJyaS3JuNqfmkshmR07NJTk3mxNzOacqtiHp6el67733/B7p6enGv47X61VeXl6Tq2V259RcknOzOTWXRDY7cmouybnZnJiLW4QtIDY21jENHAAANB9XsAAAAAyjYAEAABgW%2Buyzzz4b7EXg8nk8Hg0aNEgejyfYSzHKqbkk52Zzai6JbHbk1FySc7M5LRe/RQgAAGAYtwgBAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAIDt7Ny5U3fffbd%2B8YtfaMiQIZozZ46qq6svOPfjjz/W2LFjlZCQoFGjRmnbtm2tvFr8K6JgAQBs5eTJk3rkkUf0m9/8Rrt379a6detUVFSk5cuXN5l75MgRZWVl6YknntDu3buVlZWlKVOm6MSJE0FYOf6VULDauLKyMmVkZCgpKUnJycmaO3eu6urqLjr/888/1913363ExESlpKRozZo1rbjawAWSa/ny5erXr58SExN9j1deeSVIK26%2BkydPKjU1VYWFhRedY8d31oHkeuihh9S/f3%2B/c7djx45WXGXzHDhwQJMnT9agQYM0ZMgQTZ8%2BXSdPngz2sq749dycXHY6Z1dddZX%2B%2Bte/asKECWrXrp1Onz6t8%2BfP66qrrmoyd926dUpKStLIkSPlcrk0evRoDRw4UKtXrw7CygPTnKtzdjpvjerr6zVp0iQ99dRTF52zbt06paamKiEhQRMmTNCePXtacYWGWGjT7r//fuvJJ5%2B0Kisrra%2B//toaM2aMtWLFigvOPX36tDVo0CDr7bfftmpra62//vWvVmJiovX555%2B38qovLZBcWVlZ1qJFi4K0wiuze/dua%2BTIkVZ8fLz16aefXnDO4cOHrf79%2B1sffvihVVtba/3lL3%2BxbrzxRuu7775r5dUGLpBclmVZycnJVmFhYSuu7PJVVVVZQ4YMsRYsWGCdP3/eOnnypPXwww9bjzzySFDXdaWv5%2BbmstM5%2B7Fbb73Vio%2BPt%2B69916roqKiyfaMjAzrhRde8Bt74YUXrMcee6y1ltgsZWVlVv/%2B/a13333Xqq%2Bvt06cOGHdeeed1oIFCy44347n7dVXX7X69OljzZgx44LbP/30UysxMdHavXu3VVNTY61cudJKTk62KisrW3mlV4YrWG3Y0aNHVVRUpJycHLndbnXv3l0ZGRnKz8%2B/4PytW7cqOjpa9913n1wul26%2B%2BWaNHTv2ovODJdBce/fu1Q033BCkVV6%2BdevWadq0aZo6deol59npnXWguUpKSnTmzBn17du3lVZ2ZY4dO6Y%2BffooMzNT4eHhiomJUXp6unbt2hXUdV3p67k5uex2zn5s69at2rFjh0JCQpSdnd1ke0VFhdxut99YRESEKisrW2uJzdKcq3N2PG87d%2B7U1q1bdfvtt190zpo1azRmzBgNGDBAYWFhevDBBxUTE6NNmza14kqvHAWrDTt48KCio6PVtWtX31hcXJyOHTums2fPXnB%2BfHy831ivXr104MCBFl9rcwSSq6ysTMeOHdM777yjoUOHKiUlRX/4wx90/vz5YC07YEOHDtWHH36o0aNH/%2BS84uJiW5yvRoHm2rt3rzwej6ZOnarBgwfrzjvv1Nq1a1tplc3385//XK%2B//rpCQ0N9Y1u2bFG/fv2CuKorfz03J5fdztmPRUREqGvXrsrJyVFBQYHOnDnjt93tdje5vVZdXd2m/4fCHTp0kCT98pe/1NixY9WlSxdNmDChyTy7nbeysjLNnDlTL7/8cpPS%2B2N2%2B9l4MRSsNuxC77wan1/o3Zdd3qkFksvr9SopKUkTJkzQRx99pBUrVqigoEAvvvhiq6%2B3ubp06SKXy3XJeXY5X40CzVVTU6OEhARNnTpVBQUFeuqppzR37lxt3ry5FVZ5ZSzL0iuvvKJt27Zp5syZQV2LyT8fl8plt3P2t7/9Tb/61a9UU1PjG6upqVFYWFiT71l8fLwOHjzoN1ZcXKzevXu3ylqvxKWuztnpvDU0NCgnJ0eTJ09Wnz59fnKu3X42XgwFqw2LjIxUVVWV31jj8wu9%2B7LLO7VAcvXp00f5%2BfkaOXKkwsPDFRcXp4yMDNtdIv4pdjlfzZWWlqbXX39dffv2VVhYmIYOHaq0tLQ2%2BUP/x8rLy5Wdna0NGzbo7bff1vXXXx/U9Zj68xFILruds%2Buvv17V1dV6%2BeWXVVNTo2%2B//Vbz58/XxIkTFR4e7jd33LhxKioq0qZNm1RXV6dNmzapqKhI48ePD9LqA3epq3N2Om/Lli1TeHi4Jk2adMm5TvnZSMFqw3r37q3Tp0/r%2B%2B%2B/940dOnRI3bp1U8eOHZvMt8s7tUByFRUVadmyZX771dTUKCIiolXX2pLscr6aa%2B3atU1%2BwNfU1Kh9%2B/ZBWtGlff3117rrrrtUXl6utWvXBr1cSWb%2BfASay27nzOPx6PXXX9fBgwc1ZMgQTZo0Sbfccot%2B//vfS5ISExP1/vvvS/rh4weLFy/WsmXLNHDgQC1ZskSLFi1Sz549gxnhoppzdc5O5239%2BvUqKipSUlKSkpKStHHjRm3cuFFJSUlN5vbu3dsZPxuD/Sl7/LTf/OY31tSpU61z5875fttu4cKFF5x78uRJKykpyVq5cqVVU1Nj7dy500pMTLR27tzZyqu%2BtEvl%2BuKLL6x%2B/fpZ77//vlVfX2/9z//8j3X77bdbixcvDuKqm%2B%2BnftuuuLjY6t%2B/v/WXv/zF91uE/fv3t7766qtWXmXz/VSulStXWjfffLO1f/9%2Bq76%2B3tq2bZt14403Wrt27WrlVQbm9OnT1m233WY99dRTVn19fbCX43Olr%2Bfm5LLbOXOy8vJy65e//KU1b9486/z589Y333xjTZw40Zo9e3aTuXY%2BbzNmzLjobxE2/sbszp07fb9FOHDgQOvUqVOtvMorQ8Fq47xer5WVlWUNGjTIGjx4sPXiiy9adXV1lmVZVkJCgrV%2B/Xq/%2BV988YWVnp5uJSYmWiNGjLDefffdYCz7kn4qV6MtW7ZYY8eOtRISEqxbb73VWrRoUZv6D2Ag/m8R%2Bb/nbMeOHda4ceOshIQEa8yYMdb27duDscxm%2B6lcDQ0N1uLFi63hw4dbN954ozVmzBhr8%2BbNwVrqJb355ptWfHy8ddNNN1kJCQl%2Bj9GjR1tLly4N2tqu5PX8U7ksy97nzOkOHjxoTZ482UpKSrKGDx9u/fGPf7TOnz9vWZZzztuPC9a3335rJSQk%2BBXDP//5z9Ydd9xhJSQkWBMnTrT%2B/ve/B2upl62dZVlWsK%2BiAQAAOAmfwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADPv/DIssOCd7mpEAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-3721154334349679714">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.0</td>
        <td class="number">19422</td>
        <td class="number">89.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.0</td>
        <td class="number">957</td>
        <td class="number">4.4%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.0</td>
        <td class="number">508</td>
        <td class="number">2.4%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">330</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4.0</td>
        <td class="number">317</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="missing">
        <td class="fillremaining">(Missing)</td>
        <td class="number">63</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-3721154334349679714">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.0</td>
        <td class="number">19422</td>
        <td class="number">89.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">330</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.0</td>
        <td class="number">957</td>
        <td class="number">4.4%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.0</td>
        <td class="number">508</td>
        <td class="number">2.4%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4.0</td>
        <td class="number">317</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.0</td>
        <td class="number">19422</td>
        <td class="number">89.9%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">330</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2.0</td>
        <td class="number">957</td>
        <td class="number">4.4%</td>
        <td>
            <div class="bar" style="width:5%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">3.0</td>
        <td class="number">508</td>
        <td class="number">2.4%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">4.0</td>
        <td class="number">317</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:2%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_waterfront">waterfront<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>3</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="alert">
                    <th>Missing (%)</th>
                    <td>11.0%</td>
                </tr>
                <tr class="alert">
                    <th>Missing (n)</th>
                    <td>2376</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>0.0075959</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>1</td>
                </tr>
                <tr class="alert">
                    <th>Zeros (%)</th>
                    <td>88.3%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram3516600126752775226">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAAQpJREFUeJzt1sEJAkEQRUEVQzIIc/JsTgaxObV3kQcKy4hU3Rv%2B5TFznJk5AG%2BdVg%2BAX3ZePeDV5fb4%2BGa7X3dYAl4QSAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAI59UD%2BF%2BX2%2BPjm%2B1%2B3WHJ944zM6tHwK/yxYIgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoHwBJV2DpGLuzL6AAAAAElFTkSuQmCC">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives3516600126752775226,#minihistogram3516600126752775226"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives3516600126752775226">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles3516600126752775226"
                                                  aria-controls="quantiles3516600126752775226" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram3516600126752775226" aria-controls="histogram3516600126752775226"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common3516600126752775226" aria-controls="common3516600126752775226"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme3516600126752775226" aria-controls="extreme3516600126752775226"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles3516600126752775226">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>1</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>0</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>0.086825</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>11.431</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>126.69</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>0.0075959</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>0.015076</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>11.344</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>146</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>0.0075386</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram3516600126752775226">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3X9UVXW%2B//GXcFAOKEIiWl27OAiVaUKoZDrOiHG9lqgpRT%2BuY1PqKkjSEW0cLU0Hf0zolDmaZY6jcicm0ykbf93mljqOgmNW1kQBptJgAoLJT%2BXH%2Bf7R5XznRI1Qnw3n4POx1lmr8/ns/dnv/V6derH35tDB4XA4BAAAAGO82roAAACA9oaABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwtw1YOTk5%2BulPf6rBgwdr6NChmjNnjkpLSyVJ77//vu6%2B%2B25FRUUpNjZWr776qsu%2B27dvV1xcnCIjIzVhwgQdO3bMOVdfX6/ly5frtttuU1RUlB599FEVFRU558%2BdO6ekpCQNHDhQMTExSktLU11dXeucNAAAaBfcMmDV1NRoypQpioqK0l/%2B8he9%2BeabOn/%2BvH7xi1/oyy%2B/1LRp0zR%2B/HgdOXJEaWlpWrp0qT744ANJUlZWlhYvXqxly5bpyJEjGjt2rB599FFVV1dLktauXauDBw/qtdde04EDB%2BTr66v58%2Bc7jz1jxgz5%2BfnpwIED2rp1qw4dOqSNGze2RRsAAICHcsuAVVhYqBtuuEHJycnq2LGjgoKClJiYqCNHjmjv3r0KDAzUAw88IJvNpiFDhig%2BPl4ZGRmSpFdffVV33nmnoqOj5ePjowcffFBBQUHauXOnc37q1Km6%2Buqr1blzZ82bN0/79%2B9XQUGBTp06pezsbM2ePVt2u129evVSUlKSc20AAIDmsLV1Ad/kBz/4gdavX%2B8ytmfPHt10003Kzc1VRESEy1yfPn20detWSVJeXp4mTpzYZD4nJ0fl5eX64osvXPYPDg5W165d9cknn0iSAgMD1aNHD%2Bd8WFiYCgsLdeHCBQUEBDSr/qKiIhUXF7uMde/eXSEhIc3aHwAAeDa3DFj/zOFw6Nlnn9Xbb7%2BtLVu2aNOmTbLb7S7b%2BPr6qqqqSpJUWVn5rfOVlZWSJD8/vybzjXNf37fxfVVVVbMDVmZmplavXu0ylpycrJSUlGbtDwAAPJtbB6yKigrNnTtXH330kbZs2aLrr79edrtd5eXlLtvV1NTI399f0leBqKampsl8UFCQMyw1Po/19f0dDkeTucb3jes3R2JiomJjY13GbDY/lZVVNnuN5vD29lJAgF0XLlSrvr7B6NpXOnprHXprLfprHXprHSt7GxTU/P9/m%2BS2Aev06dOaOnWqrrnmGm3dulVXXXWVJCkiIkIHDx502TYvL0/h4eGSpPDwcOXm5jaZHz58uLp27aoePXooLy/PeZuwuLhY58%2BfV0REhBoaGnT%2B/HmVlJQoODhYkpSfn6%2BePXuqS5cuza49JCSkye3A4uJy1dVZ84Gsr2%2BwbO0rHb21Dr21Fv21Dr21TnvqrVs%2B5P7ll19q8uTJuuWWW/Tyyy87w5UkxcXFqaSkRBs3blRtba0OHz6sHTt2OJ%2B7SkhI0I4dO3T48GHV1tZq48aNOnfunOLi4iRJEyZM0Nq1a1VQUKCKigotWbJEgwcP1nXXXafQ0FBFR0dryZIlqqioUEFBgdasWaOEhIQ26QMAAPBMHRwOh6Oti/i63/72t1q2bJnsdrs6dOjgMnfs2DEdP35caWlp%2BvTTT3XVVVcpKSlJEyZMcG7z%2Buuva%2B3atTp79qz69Omj%2BfPna8CAAZKk2tpaPffcc3rjjTdUWVmpmJgYLV68WN26dZMklZSUaNGiRcrKypKXl5fGjx%2Bv1NRUeXt7f69zKi4uv/xGLWSzeSkoyF9lZZXtJvG7C3prHXprLfprHXprHSt727178%2B9AmeSWAas9ImB5FnprHXprLfprHXprnfYYsNzyFiEAAIAnI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADHPbP/aM5hk4b3dbl9Bsu2YMbesSAABoFVzBAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYW4fsEpLSxUXF6esrCxJ0lNPPaWoqCiX14033qiHH35YktTQ0KCoqChFRka6bFNVVSVJOnfunJKSkjRw4EDFxMQoLS1NdXV1zuO9//77uvvuuxUVFaXY2Fi9%2BuqrrX/SAADAo7l1wDp69KgSExN1%2BvRp59iiRYt07Ngx5%2Bv5559XQECAfv7zn0uS8vLyVFtbq%2BzsbJft/Pz8JEkzZsyQn5%2BfDhw4oK1bt%2BrQoUPauHGjJOnLL7/UtGnTNH78eB05ckRpaWlaunSpPvjgg1Y/dwAA4LncNmBt375dqampmjlz5rduU1paqtTUVM2bN0/h4eGSpOPHj%2Bv6669Xx44dm2x/6tQpZWdna/bs2bLb7erVq5eSkpKUkZEhSdq7d68CAwP1wAMPyGazaciQIYqPj3fOAwAANIetrQv4NsOGDVN8fLxsNtu3hqz09HT169dPY8eOdY4dP35cFy9e1MSJE/WPf/xDYWFhmjVrlm655Rbl5uYqMDBQPXr0cG4fFhamwsJCXbhwQbm5uYqIiHA5Rp8%2BfbR169YW1V5UVKTi4mKXMZvNTyEhIS1a53K8vd02H38jm81z6m3sraf12BPQW2vRX%2BvQW%2Bu0x966bcDq3r37v5wvKCjQG2%2B80eQZKV9fX9188816/PHH1bVrV2VkZOjhhx/WG2%2B8ocrKStntdpftG99XVVV947yvr6/z%2Ba3myszM1OrVq13GkpOTlZKS0qJ12pugIP%2B2LqHFAgLsl98I3wm9tRb9tQ69tU576q3bBqzLee2115wPuP%2BzxmexGj388MPatm2b9u3bpx49eqi6utplvvG9v7%2B/7Ha7ysvLXeZramrk79%2ByYJCYmKjY2FiXMZvNT2VllS1a53I8LembPn8reXt7KSDArgsXqlVf39DW5bQr9NZa9Nc69NY6Vva2rX6499iAtXfvXj300ENNxn/9619r1KhR6tu3r3Ps0qVL6tSpk8LDw3X%2B/HmVlJQoODhYkpSfn6%2BePXuqS5cuioiI0MGDB13Wy8vLcz7f1VwhISFNbgcWF5erru7K/kB64vnX1zd4ZN2egN5ai/5ah95apz311rMugfyfsrIy5efna9CgQU3mPv30U6Wlpam4uFiXLl3S6tWrVVFRobi4OIWGhio6OlpLlixRRUWFCgoKtGbNGiUkJEiS4uLiVFJSoo0bN6q2tlaHDx/Wjh07NHHixNY%2BRQAA4ME8MmB9/vnnkuTysHqjpUuX6rrrrtO4ceMUExOj7Oxs/fa3v1VgYKAkadWqVaqrq9PIkSN1zz336Ic//KGSkpIkSUFBQdqwYYN2796tmJgYzZ8/X/Pnz9ett97aeicHAAA8XgeHw%2BFo6yKuBMXF5ZffqIVsNi/FpR8wvq5Vds0Y2tYlNJvN5qWgIH%2BVlVW2m8vV7oLeWov%2BWofeWsfK3nbv3sXoes3lkVewAAAA3BkBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAxz%2B4BVWlqquLg4ZWVlOccWLFigfv36KSoqyvnKzMx0zr/00ksaPny4IiMjNWnSJJ04ccI5V1VVpblz5yomJkbR0dGaM2eOKisrnfOfffaZJk%2BerKioKA0bNkwvvPBC65woAABoN9w6YB09elSJiYk6ffq0y/jx48e1ePFiHTt2zPlKTEyUJG3fvl2bN2/Wyy%2B/rKysLN10001KSUmRw%2BGQJC1evFhnzpzRnj17tHfvXp05c0bp6emSpNraWj3yyCPq37%2B/srKy9OKLLyojI0O7du1q3RMHAAAezW0D1vbt25WamqqZM2e6jF%2B6dEmffvqp%2BvXr9437/eEPf9D999%2Bv8PBwderUSbNmzVJhYaGysrJUXV2tHTt2KCUlRYGBgerWrZtSU1O1bds2VVdX68iRIyoqKlJKSoo6duyovn37atKkScrIyGiNUwYAAO2Era0L%2BDbDhg1TfHy8bDabS8jKyclRXV2dVq1apaNHj6pLly6aOHGipkyZIi8vL%2BXl5Wnq1KnO7X18fBQaGqqcnBwFBgaqtrZWERERzvmwsDDV1NTo5MmTys3NVe/evdWxY0fnfJ8%2BffTiiy%2B2qPaioiIVFxe7jNlsfgoJCWlpG/4lb2%2B3zcffyGbznHobe%2BtpPfYE9NZa9Nc69NY67bG3bhuwunfv/o3j5eXlGjx4sCZNmqSVK1fq448/VnJysry8vDRlyhRVVlbKbre77OPr66uqqipVVFRIkvz8/JxzjdtWVlZ%2B4752u11VVVUtqj0zM1OrV692GUtOTlZKSkqL1mlvgoL827qEFgsIsF9%2BI3wn9NZa9Nc69NY67am3bhuwvs3QoUM1dOhQ5/ubb75ZkydP1s6dOzVlyhTZ7XbV1NS47FNTUyN/f39nsKqurpa/v7/znyWpc%2BfO8vPzc75v9M/bNldiYqJiY2Ndxmw2P5WVVX7LHt%2BNpyV90%2BdvJW9vLwUE2HXhQrXq6xvaupx2hd5ai/5ah95ax8rettUP9x4XsN566y2VlJTo3nvvdY5dunRJvr6%2BkqTw8HDl5uZqxIgRkr56cP3kyZOKiIhQ79695ePjo7y8PA0YMECSlJ%2Bf77yNeO7cOZ08eVJ1dXWy2b5qTV5ensLDw1tUY0hISJPbgcXF5aqru7I/kJ54/vX1DR5Ztyegt9aiv9aht9ZpT731rEsgkhwOh5YuXapDhw7J4XDo2LFj2rRpk/O3CCdOnKgtW7YoJydHFy9e1IoVKxQcHKyBAwfKbrdr9OjRSk9PV2lpqUpLS5Wenq4xY8bI19dXMTExCgoK0ooVK3Tx4kXl5ORo8%2BbNSkhIaOOzBgAAnsTjrmDFxcVp7ty5Wrhwoc6ePavg4GBNnz5d48aNkyQlJCSovLxcycnJKi0tVf/%2B/bVu3Tr5%2BPhI%2Buo7tJYvX674%2BHjV1tZq5MiRevLJJyVJNptNGzZs0KJFizR06FD5%2Bflp0qRJmjBhQpudLwAA8DwdHI1fEAVLFReXG1/TZvNSXPoB4%2BtaZdeMoZffyE3YbF4KCvJXWVllu7lc7S7orbXor3XorXWs7G337l2MrtdcHneLEAAAwN0RsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwtw9YpaWliouLU1ZWlnNsz549GjdunG655RbFxsZq9erVamhocM6PHj1aAwYMUFRUlPOVn58vSaqqqtLcuXMVExOj6OhozZkzR5WVlc59P/vsM02ePFlRUVEaNmyYXnjhhdY7WQAA0C64dcA6evSoEhMTdfr0aefYhx9%2BqDlz5mjGjBn629/%2Bppdeeknbtm3Txo0bJUkVFRX67LPPtHPnTh07dsz5CgsLkyQtXrxYZ86c0Z49e7R3716dOXNG6enpkqTa2lo98sgj6t%2B/v7KysvTiiy8qIyNDu3btavVzBwAAnsttA9b27duVmpqqmTNnuoz/4x//0L333qsRI0bIy8tLYWFhiouL05EjRyR9FcACAwN17bXXNlmzurpaO3bsUEpKigIDA9WtWzelpqZq27Ztqq6u1pEjR1RUVKSUlBR17NhRffv21aRJk5SRkdEq5wwAANoHW1sX8G2GDRum%2BPh42Ww2l5A1atQojRo1yvm%2BpqZG77zzjuLj4yVJx48fl91u13/9138pNzdX1157raZPn64RI0bo1KlTqq2tVUREhHP/sLAw1dTU6OTJk8rNzVXv3r3VsWNH53yfPn304osvtqj2oqIiFRcXu4zZbH4KCQlp0TqX4%2B3ttvn4G9lsnlNvY289rceegN5ai/5ah95apz321m0DVvfu3S%2B7TUVFhR5//HH5%2BvrqwQcflCR16NBB/fv3189%2B9jNdc8012r17t6ZPn64tW7aorq5OkuTn5%2Bdcw263S5IqKytVWVnpfP/P81VVVS2qPTMzU6tXr3YZS05OVkpKSovWaW%2BCgvzbuoQWCwiwX34jfCf01lr01zr01jrtqbduG7Au58SJE0pJSVG3bt20adMmde7cWZI0ZcoUl%2B3Gjh2rN998U3v27HFe5aqurpa/v7/znyWpc%2BfO8vPzc75v9M/bNldiYqJiY2Ndxmw2P5WVVX7LHt%2BNpyV90%2BdvJW9vLwUE2HXhQrXq6xsuvwOajd5ai/5ah95ax8rettUP9x4ZsPbt26ef/exnuueeezRr1izZbP//NF5%2B%2BWX17dtXQ4YMcY5dunRJnTp1Uu/eveXj46O8vDwNGDBAkpSfny8fHx%2BFhobq3LlzOnnypOrq6pxr5uXlKTw8vEX1hYSENLkdWFxcrrq6K/sD6YnnX1/f4JF1ewJ6ay36ax16a5321FvPugQi6b333lNycrLmzp2rJ554wiVcSdKZM2f09NNPq6CgQHV1ddq6dauOHTumu%2B66S3a7XaNHj1Z6erpKS0tVWlqq9PR0jRkzRr6%2BvoqJiVFQUJBWrFihixcvKicnR5s3b1ZCQkIbnS0AAPBEHncF64UXXlBdXZ3S0tKUlpbmHI%2BOjtb69es1Z84ceXl56f7771d5ebnzIfV///d/lyQtWLBAy5cvV3x8vGprazVy5Eg9%2BeSTkiSbzaYNGzZo0aJFGjp0qPz8/DRp0iRNmDChTc4VAAB4pg4Oh8PR1kVcCYqLy42vabN5KS79gPF1rbJrxtC2LqHZbDYvBQX5q6ysst1crnYX9NZa9Nc69NY6Vva2e/cuRtdrLuO3COvr600vCQAA4FGMB6zhw4frV7/6lfLy8kwvDQAA4BGMB6zHHntM7777rsaMGaO7775br7zyisrLzd8eAwAAcFfGA9Z9992nV155Rbt379Ztt92ml156ScOGDdOsWbP017/%2B1fThAAAA3I5lX9MQGhqqmTNnavfu3UpOTtaf//xnPfzww4qNjdVvf/tbntUCAADtlmVf0/D%2B%2B%2B/rj3/8o3bu3KlLly4pLi5OEyZM0NmzZ/Xcc8/p%2BPHjWrlypVWHBwAAaDPGA9aaNWv0%2Buuv69SpU%2Brfv79mzpypMWPGOP%2BUjSR5e3vrqaeeMn1oAAAAt2A8YG3ZskVjx45VQkKC%2BvTp843bhIWFKTU11fShAQAA3ILxgLV//35VVFTo/PnzzrGdO3dqyJAhCgoKkiT17dtXffv2NX1oAAAAt2D8Ife///3vGjVqlDIzM51jzzzzjOLj4/Xpp5%2BaPhwAAIDbMR6wfvWrX%2Bk//uM/NHPmTOfYW2%2B9peHDh2vZsmWmDwcAAOB2jAesjz76SNOmTVPHjh2dY97e3po2bZree%2B8904cDAABwO8YDVufOnXX69Okm41988YV8fX1NHw4AAMDtGA9Yo0aN0sKFC/XXv/5VFRUVqqys1OHDh7Vo0SLFxcWZPhwAAIDbMf5bhLNmzVJBQYEeeughdejQwTkeFxenOXPmmD4cAACA2zEesOx2u9atW6fPPvtMn3zyiXx8fBQWFqbQ0FDThwIAAHBLlv2pnN69e6t3795WLQ8AAOC2jAeszz77TIsWLdLRo0dVW1vbZP7jjz82fUgAAAC3YjxgLVy4UIWFhUpNTVWXLl1MLw8AAOD2jAesY8eO6Xe/%2B52ioqJMLw0AAOARjH9NQ1BQkPz9/U0vCwAA4DGMB6xJkyZp5cqVKi8vN700AACARzB%2Bi3Dfvn167733FBMTo27durn8yRxJ%2BvOf/2z6kAAAAG7FeMCKiYlRTEyM6WUBAAA8hvGA9dhjj5leEgAAwKMYfwZLknJycjR37lzde%2B%2B9Onv2rDIyMpSVlWXFoQAAANyO8YD14Ycf6u6779bnn3%2BuDz/8UJcuXdLHH3%2Bshx56SG%2B//bbpwwEAALgd4wErPT1dDz30kDZv3iwfHx9J0i9/%2BUv95Cc/0erVq00fDgAAwO1YcgVr/PjxTcbvu%2B8%2BnThxwvThAAAA3I7xgOXj46OKioom44WFhbLb7aYPBwAA4HaMB6zbb79dK1asUFlZmXMsPz9faWlp%2BvGPf9zi9UpLSxUXF%2BfykPz777%2Bvu%2B%2B%2BW1FRUYqNjdWrr77qss/27dsVFxenyMhITZgwQceOHXPO1dfXa/ny5brtttsUFRWlRx99VEVFRc75c%2BfOKSkpSQMHDlRMTIzS0tJUV1fX4roBAMCVy3jAeuKJJ1RTU6PbbrtN1dXVmjBhgsaMGSObzaY5c%2Ba0aK2jR48qMTFRp0%2Bfdo59%2BeWXmjZtmsaPH68jR44oLS1NS5cu1QcffCBJysrK0uLFi7Vs2TIdOXJEY8eO1aOPPqrq6mpJ0tq1a3Xw4EG99tprOnDggHx9fTV//nzn%2BjNmzJCfn58OHDigrVu36tChQ9q4ceP3bwwAALhiGA9YnTt31iuvvKINGzZo9uzZmjZtml544QW99tprCgwMbPY627dvV2pqqmbOnOkyvnfvXgUGBuqBBx6QzWbTkCFDFB8fr4yMDEnSq6%2B%2BqjvvvFPR0dHy8fHRgw8%2BqKCgIO3cudM5P3XqVF199dXq3Lmz5s2bp/3796ugoECnTp1Sdna2Zs%2BeLbvdrl69eikpKcm5NgAAQHMY/6LRRkOGDNGQIUO%2B8/7Dhg1TfHy8bDabS8jKzc1VRESEy7Z9%2BvTR1q1bJUl5eXmaOHFik/mcnByVl5friy%2B%2BcNk/ODhYXbt21SeffCJJCgwMVI8ePZzzYWFhKiws1IULFxQQENCs2ouKilRcXOwyZrP5KSQkpFn7N5e3tyVfY2YZm81z6m3sraf12BPQW2vRX%2BvQW%2Bu0x94aD1ixsbHq0KHDt843928Rdu/e/RvHKysrmzws7%2Bvrq6qqqsvOV1ZWSpL8/PyazDfOfX3fxvdVVVXNDliZmZlNvpIiOTlZKSkpzdq/vQoK8m/rElosIIBfzLAKvbUW/bUOvbVOe%2Bqt8YB11113uQSs2tpanTp1Svv379eMGTO%2B9/p2u13l5eUuYzU1NfL393fO19TUNJkPCgpyhqXG57G%2Bvr/D4Wgy1/i%2Bcf3mSExMVGxsrMuYzeansrLKZq/RHJ6W9E2fv5W8vb0UEGDXhQvVqq9vaOty2hV6ay36ax16ax0re9tWP9wbD1jTp0//xvEtW7bo6NGj%2BslPfvK91o%2BIiNDBgwddxvLy8hQeHi5JCg8PV25ubpP54cOHq2vXrurRo4fy8vKctwmLi4t1/vx5RUREqKGhQefPn1dJSYmCg4MlffUbkD179lSXLl2aXWNISEiT24HFxeWqq7uyP5CeeP719Q0eWbcnoLfWor/WobfWaU%2B9bbVLICNGjNC%2Bffu%2B9zpxcXEqKSnRxo0bVVtbq8OHD2vHjh3O564SEhK0Y8cOHT58WLW1tdq4caPOnTunuLg4SdKECRO0du1aFRQUqKKiQkuWLNHgwYN13XXXKTQ0VNHR0VqyZIkqKipUUFCgNWvWKCEh4XvXDQAArhyWPeT%2BddnZ2erUqdP3XicoKEgbNmxQWlqaVq1apauuukrz58/XrbfeKumrh%2BsXLFighQsX6uzZs%2BrTp49eeukl528wJicnq66uTg888IAqKysVExOjZ5991rn%2BqlWrtGjRIo0cOVJeXl4aP368kpKSvnfdAADgytHB4XA4TC749VuADodDFRUV%2BuSTT/STn/xEP//5z00ezmMUF5dffqMWstm8FJd%2BwPi6Vtk1Y2hbl9BsNpuXgoL8VVZW2W4uV7sLemst%2BmsdemsdK3vbvXvzH/ExyfgVrGuuuabJbxH6%2BPho8uTJio%2BPN304AAAAt2M8YC1btsz0kgAAAB7FeMA6cuRIs7cdNGiQ6cMDAAC0OeMB68EHH5TD4XC%2BGjXeNmwc69Chgz7%2B%2BGPThwcAAGhzxgPW888/r6VLl%2BqJJ57QrbfeKh8fH73//vtauHCh7r//fo0YMcL0IQEAANyK8e/BWr58uRYsWKDbb79dnTt3VqdOnTR48GAtWrRIGzZs0LXXXut8AQAAtEfGA1ZRUZGuvvrqJuOdO3dWWVmZ6cMBAAC4HeMBKzIyUitXrlRFRYVz7Pz583rmmWc0ZMgQ04cDAABwO8afwZo/f74mT56s4cOHKzQ0VJL02WefqXv37tq0aZPpwwEAALgd4wErLCxMO3fu1I4dO5Sfny9Juv/%2B%2B3XnnXfKbrebPhwAAIDbseRvEQYEBOjuu%2B/W559/rl69ekn66tvcAQAArgTGn8FyOBxKT0/XoEGDNGbMGH3xxRd64oknNHfuXNXW1po%2BHAAAgNsxHrA2b96s119/XQsWLFDHjh0lSbfffrv%2B93//V88995zpwwEAALgd4wErMzNTTz31lCZMmOD89vY77rhDaWlp%2BtOf/mT3CcuzAAAZaElEQVT6cAAAAG7HeMD6/PPPdeONNzYZv/7661VSUmL6cAAAAG7HeMC69tpr9cEHHzQZ37dvn/OBdwAAgPbM%2BG8RPvzww3r66ad19uxZORwOHTp0SK%2B88oo2b96suXPnmj4cAACA2zEesCZOnKi6ujqtXbtWNTU1euqpp9StWzfNnDlT9913n%2BnDAQAAuB3jAeuNN97Qf/7nfyoxMVGlpaVyOBzq1q2b6cMAAAC4LePPYP3yl790Psx%2B1VVXEa4AAMAVx3jACg0N1SeffGJ6WQAAAI9h/BZheHi4UlNTtX79eoWGhqpTp04u80uXLjV9SAAAALdiPGCdPn1a0dHRkqTi4mLTywMAALg9IwFr6dKlevzxx%2BXn56fNmzebWBIAAMBjGXkGa9OmTaqurnYZe/jhh1VUVGRieQAAAI9iJGA5HI4mY%2B%2B%2B%2B64uXrxoYnkAAACPYvy3CAEAAK50BCwAAADDjAWsDh06mFoKAADAoxn7moZf/vKXLt95VVtbq2eeeUb%2B/v4u2/E9WAAAoL0zErAGDRrU5DuvoqKiVFZWprKyMhOHcPHGG29owYIFLmO1tbWSpA8//FBTpkxRVlaWbLb/f3rPPfechg8frvr6eqWnp%2Bv1119XdXW1br31Vj399NMKCQmRJJ07d05PPvmksrOz5e3trbFjx%2BqJJ55wWQsAAOBfMZIaWvu7r8aOHauxY8c63589e1YTJ07U7NmzJX0Vsl5%2B%2BWUNHjy4yb5r167VwYMH9dprr6lLly568sknNX/%2BfL344ouSpBkzZqhHjx46cOCASkpK9Oijj2rjxo2aMmVK65wcAADweB7/kLvD4dDs2bP14x//WOPGjVNBQYG%2B/PJL9e3b9xu3f/XVVzV16lRdffXV6ty5s%2BbNm6f9%2B/eroKBAp06dUnZ2tmbPni273a5evXopKSlJGRkZrXxWAADAk3n8fa/XX39deXl5WrNmjSTp%2BPHj8vf318yZM3X8%2BHEFBwfrwQcfVEJCgsrLy/XFF18oIiLCuX9wcLC6du3q/APVgYGB6tGjh3M%2BLCxMhYWFunDhggICAppVU1FRUZNbpjabn/M2pCne3p6Vj202z6m3sbee1mNPQG%2BtRX%2BtQ2%2Bt0x5769EBq6GhQWvXrtUjjzyizp07S5IuXbqkyMhIzZw5U%2BHh4crKytL06dPl7%2B%2BvqKgoSZKfn5/LOr6%2BvqqsrJQk2e12l7nG91VVVc0OWJmZmVq9erXLWHJyslJSUlp%2Bku1IUJD/5TdyMwEB9stvhO%2BE3lqL/lqH3lqnPfXWowNWVlaWioqKlJCQ4BwbP368xo8f73w/bNgwjR8/Xrt27dJtt90mSU3%2BrE9NTY38/f3lcDiazDW%2B//pvQ/4riYmJio2NdRmz2fxUVlbZ7DWaw9OSvunzt5K3t5cCAuy6cKFa9fUNbV1Ou0JvrUV/rUNvrWNlb9vqh3uPDlh79uxRXFycyxWprVu3yt/fX6NHj3aOXbp0SZ06dVLXrl3Vo0cP5eXlOW8TFhcX6/z584qIiFBDQ4POnz%2BvkpISBQcHS5Ly8/PVs2dPdenSpdl1hYSENLkdWFxcrrq6K/sD6YnnX1/f4JF1ewJ6ay36ax16a5321FvPugTyNUePHtWgQYNcxioqKrR48WL9/e9/V0NDg9555x29%2BeabSkxMlCRNmDBBa9euVUFBgSoqKrRkyRINHjxY1113nUJDQxUdHa0lS5aooqJCBQUFWrNmjcsVMgAAgMvx6CtYn3/%2BeZMrRZMnT1ZVVZUee%2BwxnTt3Tr169dLy5cs1cOBASV89C1VXV6cHHnhAlZWViomJ0bPPPuvcf9WqVVq0aJFGjhwpLy8vjR8/XklJSa16XgAAwLN1cDgcjrYu4kpQXFxufE2bzUtx6QeMr2uVXTOGtnUJzWazeSkoyF9lZZXt5nK1u6C31qK/1qG31rGyt927N/8RH5M8%2BhYhAACAOyJgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEeG7B27typvn37KioqyvmaPXu2JGnfvn2Kj49XZGSkRo8erbfffttl35deeknDhw9XZGSkJk2apBMnTjjnqqqqNHfuXMXExCg6Olpz5sxRZWVlq54bAADwbB4bsI4fP65x48bp2LFjztczzzyjkydPavr06Xr88cf1t7/9TdOnT9eMGTN09uxZSdL27du1efNmvfzyy8rKytJNN92klJQUORwOSdLixYt15swZ7dmzR3v37tWZM2eUnp7elqcKAAA8jEcHrH79%2BjUZ3759uwYOHKjbb79dNptNd9xxhwYNGqTMzExJ0h/%2B8Afdf//9Cg8PV6dOnTRr1iwVFhYqKytL1dXV2rFjh1JSUhQYGKhu3bopNTVV27ZtU3V1dWufIgAA8FC2ti7gu2hoaNBHH30ku92u9evXq76%2BXj/60Y%2BUmpqqvLw8RUREuGzfp08f5eTkSJLy8vI0depU55yPj49CQ0OVk5OjwMBA1dbWuuwfFhammpoanTx5UjfeeGOz6isqKlJxcbHLmM3mp5CQkO96yt/I29uz8rHN5jn1NvbW03rsCeitteivdeitddpjbz0yYJWWlqpv374aNWqUVq1apbKyMj3xxBOaPXu2Ll26JLvd7rK9r6%2BvqqqqJEmVlZXfOl9RUSFJ8vPzc841btuS57AyMzO1evVql7Hk5GSlpKQ0/yTboaAg/7YuocUCAuyX3wjfCb21Fv21Dr21TnvqrUcGrODgYGVkZDjf2%2B12zZ49W/fcc49iYmJUU1Pjsn1NTY38/f2d237bfGOwqq6udm7feGuwc%2BfOza4vMTFRsbGxLmM2m5/Kysw%2BLO9pSd/0%2BVvJ29tLAQF2XbhQrfr6hrYup12ht9aiv9aht9axsrdt9cO9RwasnJwcvfnmm5o1a5Y6dOggSbp06ZK8vLx088036%2BOPP3bZPi8vz/m8Vnh4uHJzczVixAhJUm1trU6ePKmIiAj17t1bPj4%2BysvL04ABAyRJ%2Bfn5ztuIzRUSEtLkdmBxcbnq6q7sD6Qnnn99fYNH1u0J6K216K916K112lNvPesSyP8JDAxURkaG1q9fr7q6OhUWFuqZZ57RXXfdpfHjxys7O1s7d%2B5UXV2ddu7cqezsbI0bN06SNHHiRG3ZskU5OTm6ePGiVqxYoeDgYA0cOFB2u12jR49Wenq6SktLVVpaqvT0dI0ZM0a%2Bvr5tfNYAAMBTeOQVrJ49e2rdunVauXKl1q5dq06dOunOO%2B/U7Nmz1alTJ/3mN79Renq65s2bp2uvvVbPP/%2B8evfuLUlKSEhQeXm5kpOTVVpaqv79%2B2vdunXy8fGRJC1YsEDLly9XfHy8amtrNXLkSD355JNteboAAMDDdHA0fgEULFVcXG58TZvNS3HpB4yva5VdM4a2dQnNZrN5KSjIX2Vlle3mcrW7oLfWor/WobfWsbK33bt3Mbpec3nkLUIAAAB3RsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwzw2YOXk5OinP/2pBg8erKFDh2rOnDkqLS2VJC1YsED9%2BvVTVFSU85WZmenc96WXXtLw4cMVGRmpSZMm6cSJE865qqoqzZ07VzExMYqOjtacOXNUWVnZ6ucHAAA8l0cGrJqaGk2ZMkVRUVH6y1/%2BojfffFPnz5/XL37xC0nS8ePHtXjxYh07dsz5SkxMlCRt375dmzdv1ssvv6ysrCzddNNNSklJkcPhkCQtXrxYZ86c0Z49e7R3716dOXNG6enpbXauAADA83hkwCosLNQNN9yg5ORkdezYUUFBQUpMTNSRI0d06dIlffrpp%2BrXr9837vuHP/xB999/v8LDw9WpUyfNmjVLhYWFysrKUnV1tXbs2KGUlBQFBgaqW7duSk1N1bZt21RdXd3KZwkAADyVra0L%2BC5%2B8IMfaP369S5je/bs0U033aScnBzV1dVp1apVOnr0qLp06aKJEydqypQp8vLyUl5enqZOnercz8fHR6GhocrJyVFgYKBqa2sVERHhnA8LC1NNTY1OnjypG2%2B8sVn1FRUVqbi42GXMZvNTSEjI9zjrpry9PSsf22yeU29jbz2tx56A3lqL/lqH3lqnPfbWIwPWP3M4HHr22Wf19ttva8uWLSopKdHgwYM1adIkrVy5Uh9//LGSk5Pl5eWlKVOmqLKyUna73WUNX19fVVVVqaKiQpLk5%2BfnnGvctiXPYWVmZmr16tUuY8nJyUpJSfmup9kuBAX5t3UJLRYQYL/8RvhO6K216K916K112lNvPTpgVVRUaO7cufroo4%2B0ZcsWXX/99br%2B%2Bus1dOhQ5zY333yzJk%2BerJ07d2rKlCmy2%2B2qqalxWaempkb%2B/v7OYFVdXS1/f3/nP0tS586dm11XYmKiYmNjXcZsNj%2BVlZl9WN7Tkr7p87eSt7eXAgLsunChWvX1DW1dTrtCb61Ff61Db61jZW/b6od7jw1Yp0%2Bf1tSpU3XNNddo69atuuqqqyRJb731lkpKSnTvvfc6t7106ZJ8fX0lSeHh4crNzdWIESMkSbW1tTp58qQiIiLUu3dv%2Bfj4KC8vTwMGDJAk5efnO28jNldISEiT24HFxeWqq7uyP5CeeP719Q0eWbcnoLfWor/WobfWaU%2B99axLIP/nyy%2B/1OTJk3XLLbfo5ZdfdoYr6atbhkuXLtWhQ4fkcDh07Ngxbdq0yflbhBMnTtSWLVuUk5OjixcvasWKFQoODtbAgQNlt9s1evRopaenq7S0VKWlpUpPT9eYMWOcAQ0AAOByPPIK1rZt21RYWKhdu3Zp9%2B7dLnPHjh3T3LlztXDhQp09e1bBwcGaPn26xo0bJ0lKSEhQeXm5kpOTVVpaqv79%2B2vdunXy8fGR9NV3aC1fvlzx8fGqra3VyJEj9eSTT7b6OQIAAM/VwdH4BVCwVHFxufE1bTYvxaUfML6uVXbNGHr5jdyEzealoCB/lZVVtpvL1e6C3lqL/lqH3lrHyt52797F6HrN5ZG3CAEAANwZAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbAAAAAMI2ABAAAYRsACAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAwjYAEAABhGwAIAADCMgAUAAGAYAQsAAMAwAhYAAIBhBCwAAADDCFgAAACGEbC%2Bwblz55SUlKSBAwcqJiZGaWlpqqura%2BuyAACAhyBgfYMZM2bIz89PBw4c0NatW3Xo0CFt3LixrcsCAAAegoD1NadOnVJ2drZmz54tu92uXr16KSkpSRkZGW1dGgAA8BC2ti7A3eTm5iowMFA9evRwjoWFhamwsFAXLlxQQEBAG1YHAEDzjX72YFuX0Gx/S/vPti7BKALW11RWVsput7uMNb6vqqpqVsAqKipScXGxy5jN5qeQkBBzhUry9vasC5A2m%2BfU29hbT%2BuxJ6C31qK/1qG31mtPvSVgfY2fn5%2Bqq6tdxhrf%2B/v7N2uNzMxMrV692mXsscce0/Tp080U%2BX%2BKioo0uWeuEhMTjYe3K11RUZF%2B97v19NYC9NZa9Nc6nthbT7kqVFRUpOeff96jens57ScqGhIeHq7z58%2BrpKTEOZafn6%2BePXuqS5cuzVojMTFR27Ztc3klJiYar7W4uFirV69ucrUM3x%2B9tQ69tRb9tQ69tU577C1XsL4mNDRU0dHRWrJkiRYtWqSysjKtWbNGCQkJzV4jJCSk3SRwAADQclzB%2BgarVq1SXV2dRo4cqXvuuUc//OEPlZSU1NZlAQAAD8EVrG8QHBysVatWtXUZAADAQ3kvXLhwYVsXge/O399fgwcPbvYD%2BGg%2Bemsdemst%2Bmsdemud9tbbDg6Hw9HWRQAAALQnPIMFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNguZFz584pKSlJAwcOVExMjNLS0lRXV/eN2%2B7bt0/x8fGKjIzU6NGj9fbbb7dytQAA4NsQsNzIjBkz5OfnpwMHDmjr1q06dOiQNm7c2GS7kydPavr06Xr88cf1t7/9TdOnT9eMGTN09uzZ1i8aAAA0YWnAaskVmd///vcaNWqUoqKiNGrUKGVkZFhZmts5deqUsrOzNXv2bNntdvXq1UtJSUnf2Ift27dr4MCBuv3222Wz2XTHHXdo0KBByszMbIPKgaZa8tlv9Omnn2rAgAHKyspqpSo9V0v6m52drbvvvltRUVH60Y9%2BpHXr1rVytZ6lJb393e9%2Bp9jYWN1yyy2Kj4/Xnj17Wrlaz1RaWqq4uLh/%2BVlvD3dpLA1Yzb0i89Zbb2nlypVavny53n33XS1btkzPPvvsFfUva25urgIDA9WjRw/nWFhYmAoLC3XhwgWXbfPy8hQREeEy1qdPH%2BXk5LRKrcDlNPez36i6ulqzZs1STU1N6xXpwZrb3/z8fE2bNk3333%2B/3n33Xa1bt04bNmzQ7t27W79oD9Hc3u7bt0/r1q3T%2BvXr9e677%2Bqxxx7TjBkz9Pnnn7d%2B0R7k6NGjSkxM1OnTp791m/Zyl8aygNWSKzJnz57V1KlTFRkZqQ4dOigqKkoxMTE6cuSIVeW5ncrKStntdpexxvdVVVWX3dbX17fJdkBbaMlnv9HTTz%2Bt22%2B/vRWr9Fwt6e9///d/a%2BTIkbrrrrvUoUMH3XDDDXrllVcUHR3dBpW7v5b09sSJE3I4HM6Xt7e3fHx8ZLPZ2qByz7B9%2B3alpqZq5syZl92uPdylsSxgteSKzAMPPKBp06Y53587d05HjhxRv379rCrP7fj5%2Bam6utplrPH91//wpd1ub/KTfk1NTbv5A5nwbC357EvSH//4R506dUqPPfZYa5bpsVrS3w8%2B%2BED/9m//pp/97GeKiYnR6NGjlZ2dre7du7d22R6hJb298847FRwcrDvuuEM33XSTHn/8cS1btkw9e/Zs7bI9xrBhw/Q///M/uuOOO/7ldu3lLo1lAaslV2T%2BWXFxsaZOnap%2B/fppzJgxVpXndsLDw3X%2B/HmVlJQ4x/Lz89WzZ0916dLFZduIiAjl5ua6jOXl5Sk8PLxVagX%2BlZZ89vPz8/XrX/9aK1askLe3d6vV6Mla0t8vv/xSmzZt0tixY3Xw4EEtWrRIy5cv5xbht2hJb2tra3XDDTfo1Vdf1XvvvadFixZp3rx5%2BuSTT1qtXk/TvXv3Zl3hay93aSwLWC25ItPovffeU0JCgnr37q21a9deUZdaQ0NDFR0drSVLlqiiokIFBQVas2aNEhISmmw7duxYZWdna%2BfOnaqrq9POnTuVnZ2tcePGtUHlgKvmfvYvXryomTNn6he/%2BIWuueaaVq3Rk7Xkv60dO3bUyJEj9eMf/1g2m02DBg3SuHHjtGvXrlar15O0pLeLFy9WeHi4br75ZnXs2FETJ05UZGSktm/f3mr1tlft5S6NZQGrJVdkJGnr1q168MEHNXnyZK1YsUIdO3a0qjS3tWrVKtXV1WnkyJG655579MMf/lBJSUmSpKioKL3xxhuSvrpk/Zvf/Ebr1q3ToEGDtGbNGj3//PPq3bt3W5YPSGr%2BZ//48eM6efKk5s2bp4EDB2rgwIGSpEceeUQLFy5s7bI9Rkv%2B2xoWFqZLly65jNXX18vhcLRKrZ6mJb0tLCxs0lubzSYfH59WqbU9azd3aRwWuu%2B%2B%2BxwzZ850lJeXO06fPu248847HatWrWqy3e7dux033XSTY//%2B/VaWA6CVNPez/3URERGOw4cPt0KFnq25/f3rX//q6Nu3r%2BOPf/yjo6GhwZGdne2IjIx0vPXWW21QtWdobm9//etfO2JiYhwffviho76%2B3rFr1y5H//79HX//%2B9/boGrP868%2B63l5eY7%2B/fs7/vSnPzlqa2sdf/rTnxz9%2B/d3nDhxopWr/H4sDVjFxcWO6dOnOwYPHuy49dZbHcuWLXPU1dU5HA6HIzIy0vH66687HA6HY8yYMY4bbrjBERkZ6fJ68sknrSwPgEWa%2B9n/OgJW87Skv%2B%2B8845jwoQJjqioKMfIkSMdv//979uqbI/Q3N7W1tY6Vq1a5RgxYoTjlltucdx1111cJGiBr3/Wv/7v7f79%2Bx1jx451REZGOu68807HO%2B%2B80xZlfi8dHA6uFQMAAJjEn8oBAAAwjIAFAABgGAELAADAMAIWAACAYQQsAAAAwwhYAAAAhhGwAAAADCNgAQAAGEbAAgAAMIyABQAAYBgBCwAAwDACFgAAgGEELAAAAMMIWAAAAIYRsAAAAAz7f0AIIb0d4ppzAAAAAElFTkSuQmCC"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common3516600126752775226">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.0</td>
        <td class="number">19075</td>
        <td class="number">88.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">146</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="missing">
        <td class="fillremaining">(Missing)</td>
        <td class="number">2376</td>
        <td class="number">11.0%</td>
        <td>
            <div class="bar" style="width:13%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme3516600126752775226">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.0</td>
        <td class="number">19075</td>
        <td class="number">88.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">146</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.0</td>
        <td class="number">19075</td>
        <td class="number">88.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1.0</td>
        <td class="number">146</td>
        <td class="number">0.7%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_yr_built">yr_built<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>116</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.5%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>1971</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>1900</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>2015</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-5047076293458780941">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAASpJREFUeJzt28GJwzAUQMHNkpJSxPa05/SUItKT0kB4xIZgIc/cDf/gZ0kIX8YY4wd46/foAWBm16MHYF23/8fmZ573vy9Msp8VBIJAIAgEgkAgCASCQCAIBIJAILgonMwKl2srEcgCtka1J6g94a7AFguCFeSLzvrVXYlATki4n7PFgiAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoHgj8IN/Il3PqcNxMvOJ2yxIAgEgkAgTHcGcTZgJlYQCAKBIBAIAoEgEAgCgSAQCAKBIBAIAoFwGWOMo4eAWVlBIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAILwA19saB2Pq9McAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-5047076293458780941,#minihistogram-5047076293458780941"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-5047076293458780941">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-5047076293458780941"
                                                  aria-controls="quantiles-5047076293458780941" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-5047076293458780941" aria-controls="histogram-5047076293458780941"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-5047076293458780941" aria-controls="common-5047076293458780941"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-5047076293458780941" aria-controls="extreme-5047076293458780941"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-5047076293458780941">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>1900</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>1915</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>1951</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>1975</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>1997</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>2011</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>2015</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>115</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>46</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>29.375</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.014904</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-0.65769</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>1971</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>24.566</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>-0.46945</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>42567680</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>862.9</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-5047076293458780941">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3Xt0lNW9//FPkglkEgwEIWqtFoSEHrmYQCBIJAcCEVEQ5dJ4KeVYxK4aCbBAFEFBEZGSclcWgqKI5xhAUKkIaPWgInITkSJBgiLaKBmaALmS2/790R9zOoSSDO7MMOH9WisrZj979uzn68zzfHguM0HGGCMAAABYE%2BzvCQAAADQ0BCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYJnD3xO4VLhchdbHDA4OUvPmEcrPL1Z1tbE%2BfkNDvbxDveqOWnmHenmHennn7Hq1bHmZf%2Bbhl2eFFcHBQQoKClJwcJC/pxIQqJd3qFfdUSvvUC/vUC/vXCz1ImABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUOf08AAADUj/7ztvp7CnX27tgkf0/BKo5gAQAAWEbAAgAAsCygTxEeOHBAM2fO1P79%2B%2BVwOJScnKzHHntMUVFR7j7V1dXq0qWLjDEKCgpyt2/dulXh4eGSpJKSEk2fPl0ffPCBKisr1adPH02dOlURERE%2BXycAABD4AvYIVnl5uUaNGqXExERt375d7733nlwul5599lmPfjk5OaqoqNCOHTu0Z88e98%2BZcCVJ06dP148//qhNmzZp8%2BbN%2BvHHH5WZmenrVQIAAA1EwB7BatSokTZv3qywsDAFBwfr5MmTKi0tVfPmzT367du3T%2B3atVOjRo3OOU5paanWr1%2BvFStWqFmzZpKkCRMm6He/%2B50mTpwop9NZ7%2BsCAAAaloA9giVJ4eHhCg4O1l133aW%2BffuqqKhII0eO9Oizb98%2BnT59WkOGDFH37t1177336vPPP3cv/%2B6771RRUaHY2Fh3W5s2bVRWVqYjR474alUAAEADErBHsP7Vyy%2B/rNOnT2vatGm677779OabbyokJESSFBYWpk6dOmnMmDFq2rSpXnvtNY0cOVJvv/22rrnmGhUVFUmSxynDM0etiouLL2g%2BeXl5crlcHm0OR7iio6MvaLx/JyQk2OM3zo96eYd61R218g718s6lUi%2BHw876XSz1ahABKywsTGFhYZoyZYp69OihgwcP6vrrr5ckPfroox59R44cqbVr12rLli367W9/6w5WpaWl7ovaS0tLJUlNmjS5oPlkZWVp0aJFHm3p6enKyMi4oPFqExnJaUxvUC/vUK%2B6o1beoV7eaej1ioqye2OZv%2BsVsAHrhx9%2B0O9%2B9zu9/vrr7iND5eXlkqSmTZu6%2B82dO1f9%2BvVzB64z/Ro3bixJat26tUJDQ5WTk6MbbrhBknT48GGFhoaqVatWFzS3tLQ0paSkeLQ5HOEqKLiwI2L/TkhIsCIjnTp1qlRVVdVWx26IqJd3qFfdUSvvUC/vXCr1srWPPLtetoNbXQVswLr66qvVrFkzzZw5U08//bROnz6tJ598UsnJybr66qvd/b7%2B%2Bmvt2rVL8%2BbNU9OmTfXCCy%2BoqKhIqampkv55OrB///7KzMzU/PnzJUmZmZkaMGCAwsLCLmhu0dHRNU4HulyFqqysnzdGVVV1vY3dEFEv71CvuqNW3qFe3mno9bK9bv6uV8Ce0A0KCtLzzz%2BvyspKpaSkaNCgQbrqqqs0Z84cj34zZ87Utddeq0GDBikxMVE7duzQ8uXL3XcMStLUqVPVqlUrDRw4ULfccot%2B%2Bctf6oknnvD1KgEAgAYiyBhj/D2JS4HLVWh9TIcjWFFRESooKG7Q/6qxhXp5h3rVHbXyDvXyzs%2Bp16X4XYRn16tly8usjOutgD2CBQAAcLEiYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsC8iAtW3bNg0bNkydO3dWUlKSpk%2BfrrKyMvfyvXv3atiwYYqPj1dKSopWr15dY4wtW7Zo4MCBiouLU//%2B/fXhhx/6chUAAEADFnABKz8/X3/4wx909913a9euXVq3bp127NihF154QZJ08uRJPfDAA7rjjju0c%2BdOzZgxQzNnztSXX37pHuPIkSMaPXq0xowZo127dmn06NEaO3asjh075q/VAgAADUjABazmzZvr008/1eDBgxUUFKQTJ07o9OnTat68uSRp8%2BbNatasme699145HA7deOONGjhwoF577TX3GOvWrVNCQoL69u0rh8OhW2%2B9VV27dlVWVpa/VgsAADQgDn9P4EI0adJEkvSf//mfOnbsmBISEjR48GBJ0qFDhxQbG%2BvRv23btlqzZo3775ycnHP2yc7OtjK/vLw8uVwujzaHI1zR0dFWxj8jJCTY4zfOj3p5h3rVHbXyDvXyzqVSL4fDzvpdLPUKyIB1xubNm3Xy5ElNmDBBGRkZWrZsmYqLi%2BV0Oj36hYWFqaSkxP13Xfr8HFlZWVq0aJFHW3p6ujIyMqyMf7bISGftneBGvbxDveqOWnmHenmnodcrKirC6nj%2BrldAB6ywsDCFhYXp4Ycf1rBhw3Ty5Ek5nU4VFhZ69CsrK1NExP/9j3M6nR4XxZ%2Brz8%2BRlpamlJQUjzaHI1wFBcVWxj8jJCRYkZFOnTpVqqqqaqtjN0TUyzvUq%2B6olXeol3culXrZ2keeXS/bwa2uAi5gff7553rsscf09ttvq1GjRpKk8vJyhYaGyul0KjY2Vlu3bvV4TE5OjmJiYtx/x8bGav/%2B/TX6dOjQwcoco6Oja5wOdLkKVVlZP2%2BMqqrqehu7IaJe3qFedUetvEO9vNPQ62V73fxdr4A7oduuXTuVlZXpz3/%2Bs8rLy/X3v/9ds2bN0tChQ9WoUSOlpqbq%2BPHjevnll1VRUaHPPvtM69ev15AhQ9xj3H777dqxY4c2bNigyspKbdiwQTt27NCgQYP8uGYAAKChCLiAFRERoWXLlunQoUNKSkrS8OHD1aNHDz322GOSpKioKL300kvauHGjEhMTNWXKFE2ZMkXdu3d3j9GmTRs999xzWrJkibp27arnn39eCxcuVOvWrf21WgAAoAEJuFOE0j/v%2BHvppZf%2B7fKOHTvq9ddfP%2B8YPXv2VM%2BePW1PDQAAIDADFgD4Uv95W2vvdBF5d2ySv6cAXPIC7hQhAADAxY6ABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJbxMQ0AANRRoH1kB/yHI1gAAACWEbAAAAAsI2ABAABYxjVYANDABNJ1QnytDxoqjmABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlvE5WAB8LpA%2BpwkALgRHsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYxlflAAD8hq9NQkPFESwAAADLAjZgZWdn67777lO3bt2UlJSkiRMnKj8/X5I0depUdejQQfHx8e6frKws92OXLl2q5ORkxcXFafjw4frmm2/cy0pKSjRp0iQlJiaqS5cumjhxooqLi32%2BfgAAIHAFZMAqKyvT/fffr/j4eH3yySf6y1/%2BohMnTuixxx6TJO3bt0/Tp0/Xnj173D9paWmSpHXr1unVV1/Viy%2B%2BqO3bt6t9%2B/bKyMiQMUaSNH36dP3444/atGmTNm/erB9//FGZmZl%2BW1cAABB4AjJg5ebm6te//rXS09PVqFEjRUVFKS0tTTt37lR5ebm%2B/vprdejQ4ZyPXbVqle655x7FxMSocePGGj9%2BvHJzc7V9%2B3aVlpZq/fr1ysjIULNmzXT55ZdrwoQJWrt2rUpLS328lgAAIFAFZMC67rrrtGzZMoWEhLjbNm3apPbt2ys7O1uVlZVasGCBevTooX79%2BumFF15QdXW1JCknJ0exsbHux4WGhqpVq1bKzs7Wd999p4qKCo/lbdq0UVlZmY4cOeKz9QMAAIEt4O8iNMZo3rx5%2BvDDD7Vy5UodP35c3bp10/DhwzVnzhwdOHBA6enpCg4O1v3336/i4mI5nU6PMcLCwlRSUqKioiJJUnh4uHvZmb7eXIeVl5cnl8vl0eZwhCs6OvpCV/OcQkKCPX7j/KiXd6gXAF9yOOxsay6WbVdAB6yioiJNmjRJ%2B/fv18qVK9WuXTu1a9dOSUlJ7j6dOnXSiBEjtGHDBt1///1yOp0qKyvzGKesrEwRERHuYFVaWqqIiAj3f0tSkyZN6jyvrKwsLVq0yKMtPT1dGRkZF7SetYmMdNbeCW7UyzvUC4AvREVFWB3P39uugA1YR48e1ahRo/SLX/xCa9asUfPmzSVJ77//vo4fP6677rrL3be8vFxhYWGSpJiYGB06dEi9e/eWJFVUVOjIkSOKjY1V69atFRoaqpycHN1www2SpMOHD7tPI9ZVWlqaUlJSPNocjnAVFNi9GzEkJFiRkU6dOlWqqqpqq2M3RNTLO9QLgC/Z2keeve2yHdzqKiAD1smTJzVixAh1795dM2bMUHDw/x0GNMZo5syZ%2BtWvfqXu3bvriy%2B%2B0IoVKzRp0iRJ0pAhQ7Rw4UIlJyerdevWmjt3rlq0aKGEhASFhoaqf//%2ByszM1Pz58yVJmZmZGjBggDug1UV0dHSN04EuV6EqK%2BtnJ1VVVV1vYzdE1Ms71AuAL9jezvh72xWQAWvt2rXKzc3Vu%2B%2B%2Bq40bN3os27NnjyZNmqRp06bp2LFjatGihUaPHq1BgwZJkoYOHarCwkKlp6crPz9fHTt21JIlSxQaGirpn5%2BhNWvWLA0cOFAVFRXq06ePHn/8cZ%2BvIwAACFxB5swHQKFeuVyF1sd0OIIVFRWhgoJijjDUAfXyTn3Wi69HAXC2d8cm1d6pDs7edrVseZmVcb3F7UEAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYJnPA1ZVVZWvnxIAAMCnfB6wkpOT9ac//Uk5OTm%2BfmoAAACf8HnAeuihh/T5559rwIABGjZsmF5//XUVFhb6ehoAAAD1xucB6%2B6779brr7%2BujRs3qkePHlq6dKluuukmjR8/Xp9%2B%2BqmvpwMAAGCd3y5yb9WqlcaNG6eNGzcqPT1df/3rXzVy5EilpKRo%2BfLlXKsFAAAClsNfT7x37169%2Beab2rBhg8rLy5WamqrBgwfr2LFjmj9/vvbt26c5c%2Bb4a3oAAAAXzOcB6/nnn9dbb72l7777Th07dtS4ceM0YMAANWnSxN0nJCRETzzxhK%2BnBgAAYIXPA9bKlSt1%2B%2B23a%2BjQoWrbtu05%2B7Rp00YTJkzw8cwAAADs8HnA%2Buijj1RUVKQTJ0642zZs2KAbb7xRUVFRkqTrr79e119/va%2BnBgAAYIXPL3L/6quv1K9fP2VlZbnbZs%2BerYEDB%2Brrr7/29XQAAACs83nA%2BtOf/qSbb75Z48aNc7e9//77Sk5O1rPPPuvr6QAAAFjn84C1f/9%2BPfDAA2rUqJG7LSQkRA888IC%2B%2BOILX08HAADAOp8HrCZNmujo0aM12n/66SeFhYX5ejoAAADW%2BTxg9evXT9OmTdOnn36qoqIiFRcX67PPPtNTTz2l1NRUX08HAADAOp/fRTh%2B/Hh9//33%2Bv3vf6%2BgoCB3e2pqqiZOnOjr6QAAAFjn84DldDq1ZMkSffvttzp48KBCQ0PVpk0btWrVytdTAQAAqBd%2B%2B6qc1q1bq3Xr1v56egAAgHrj84D17bff6qmnntLu3btVUVFRY/mBAwd8PSUAAACrfB6wpk2bptzcXE2YMEGXXXaZr58eAACg3vk8YO3Zs0evvPKK4uPj6%2B05srOzNWvWLO3fv1%2BhoaFKSkrSo48%2BqubNm3s9VlVVlTIzM/XWW2%2BptLRU3bt315NPPqno6GiPflu2bFFmZqa%2B//57XXXVVZo4caJ69%2B5ta5UAAEAA8fnHNERFRSkiIqLexi8rK9P999%2Bv%2BPh4ffLJJ/rLX/6iEydO6LHHHrug8RYvXqytW7fqjTfe0Mcff6ywsDBNmTLFo8%2BRI0c0evRojRkzRrt27dLo0aM1duxYHTt2zMYqAQCAAOPzgDV8%2BHDNmTNHhYWF9TJ%2Bbm6ufv3rXys9PV2NGjVSVFSU0tLStHPnzgsab/Xq1Ro1apSuuuoqNWnSRJMnT9ZHH32k77//3t1n3bp1SkhIUN%2B%2BfeVwOHTrrbeqa9euHt%2B3CAAALh0%2BP0W4ZcsWffHFF0pMTNTll1/u8ZU5kvTXv/71Z41/3XXXadmyZR5tmzZtUvv27b0eq7CwUD/99JNiY2PdbS1atFDTpk118OBBXXPNNZKknJwcjz6S1LZtW2VnZ1/AGgAAgEDn84CVmJioxMREnzyXMUbz5s3Thx9%2BqJUrV3r9%2BOLiYklSeHi4R3tYWJh72Zl%2BTqfT/XdeXp4KCwvlcrm0f/9%2BSZLDEV7juq2fKyQk2OM3zo96eYd6AfAlh8POtuZi2Xb5PGA99NBDPnmeoqIiTZo0Sfv379fKlSvVrl07r8c4E5pKS0s92svKyjyuI3M6nSorK3P/nZWVpTVr1kiSBg8eLElKT09XRkaG13Ooi8hIZ%2B2d4Ea9vEO9APhCVJTd67P9ve3yyweNZmdn65VXXtG3336r%2BfPn6/3331fbtm2tHdk6evSoRo0apV/84hdas2bNBd09KElNmzbVFVdc4XEK0OVy6cSJEx6nBGNjY91HqiQpLS1NW7duVZs2bXTPPfdI%2BucRrIKCYtkUEhKsyEinTp0qVVVVtdWxGyLq5R3qBcCXbO0jz9522Q5udeXzgPW3v/1Nd999t%2BLi4vS3v/1N5eXlOnDggJ555hktWrToZ3%2B0wcmTJzVixAh1795dM2bMUHDwzztEOHjwYC1evFgdO3ZUVFSUnnnmGXXr1k3XXnutu8/tt9%2Bu5cuXa8OGDbr55pu1a9cuffXVV5o5c6b70%2BpdrkJVVtbPTqqqqrrexm6IqJd3qBcAX7C9nfH3tsvnASszM1O///3vNW7cOPdnYT399NO67LLLrASstWvXKjc3V%2B%2B%2B%2B642btzosWzPnj1ej5eenq7Kykrde%2B%2B9Ki4uVmJioubNm%2BfRp02bNnruueeUmZmpyZMn6%2Bqrr9bChQv5KiAAAC5RQcYY48snTEhI0OrVq9W6dWvFx8fr7bff1jXXXKOjR49q0KBBFxSCAoHLZf9jKRyOYEVFRaigoJgjDHVAvbxTn/XqP2%2Br1fEABL53xyZZGefsbVfLlv751hifX2IfGhqqoqKiGu25ubked%2BIBAAAEKp8HrL59%2B%2BrPf/6zCgoK3G2HDx/WjBkz1KtXL19PBwAAwDqfB6xHHnlEZWVl6tGjh0pLSzV48GANGDBADodDEydO9PV0AAAArPP5Re5NmjTR66%2B/rm3btumrr75SdXW1YmNj1bNnz599xx8AAMDFwC%2BfgyVJN954o2688UZ/PT0AAEC98XnASklJUVBQ0L9d/nO/ixAAAMDffB6w7rzzTo%2BAVVFRoe%2B%2B%2B04fffSRxo4d6%2BvpAAAAWOfzgDV69Ohztq9cuVK7d%2B/W7373Ox/PCAAAwK6L5qry3r17a8uWLf6eBgAAwM920QSsHTt2qHHjxv6eBgAAwM/m81OEZ58CNMaoqKhIBw8e5PQgAABoEHwesH7xi1/UuIswNDRUI0aM0MCBA309HQAAAOt8HrCeffZZXz8lAACAT/k8YO3cubPOfbt27VqPMwEAAKgfPg9Y//Vf/yVjjPvnjDOnDc%2B0BQUF6cCBA76eHgAAwM/m84C1cOFCzZw5U4888oi6d%2B%2Bu0NBQ7d27V9OmTdM999yj3r17%2B3pKAAAAVvn8YxpmzZqlqVOnqm/fvmrSpIkaN26sbt266amnntJLL72kq6%2B%2B2v0DAAAQiHwesPLy8nTVVVfVaG/SpIkKCgp8PR0AAADrfB6w4uLiNGfOHBUVFbnbTpw4odmzZ%2BvGG2/09XQAAACs8/k1WFOmTNGIESOUnJysVq1aSZK%2B/fZbtWzZUitWrPD1dAAAAKzzecBq06aNNmzYoPXr1%2Bvw4cOSpHvuuUe33XabnE6nr6cDAABgnc8DliRFRkZq2LBh%2BuGHH3TNNddI%2BuenuQMAADQEPr8GyxijzMxMde3aVQMGDNBPP/2kRx55RJMmTVJFRYWvpwMAAGCdzwPWq6%2B%2BqrfeektTp05Vo0aNJEl9%2B/bVBx98oPnz5/t6OgAAANb5PGBlZWXpiSee0ODBg92f3n7rrbdqxowZeuedd3w9HQAAAOt8HrB%2B%2BOEH/cd//EeN9nbt2un48eO%2Bng4AAIB1Pg9YV199tb788ssa7Vu2bHFf8A4AABDIfH4X4ciRI/Xkk0/q2LFjMsZo27Ztev311/Xqq69q0qRJvp4OAACAdT4PWEOGDFFlZaUWL16ssrIyPfHEE7r88ss1btw43X333b6eDgAAgHU%2BD1hvv/22brnlFqWlpSk/P1/GGF1%2B%2BeW%2BngYAAEC98fk1WE8//bT7YvbmzZsTrgAAQIPj84DVqlUrHTx40NdPCwAA4DM%2BP0UYExOjCRMmaNmyZWrVqpUaN27ssXzmzJm%2BnhIAAIBVPj%2BCdfToUXXp0kURERFyuVz64YcfPH68lZ%2Bfr9TUVG3fvt3dNnXqVHXo0EHx8fHun6ysLPfypUuXKjk5WXFxcRo%2BfLi%2B%2BeYb97KSkhJNmjRJiYmJ6tKliyZOnKji4uKft9IAAOCS4pMjWDNnztSYMWMUHh6uV1991dq4u3fv1qOPPqqjR496tO/bt0/Tp0/XnXfeWeMx69at06uvvqoXX3xR1157rebOnauMjAytX79eQUFBmj59un788Udt2rRJVVVVGjt2rDIzMzV16lRr8wYAAA2bT45grVixQqWlpR5tI0eOVF5e3gWPuW7dOk2YMEHjxo3zaC8vL9fXX3%2BtDh06nPNxq1at0j333KOYmBg1btxY48ePV25urrZv367S0lKtX79eGRkZatasmS6//HJNmDBBa9eurTF/AACAf8cnAcsYU6Pt888/1%2BnTpy94zJtuuknvvfeebr31Vo/27OxsVVZWasGCBerRo4f69eunF154QdXV1ZKknJwcxcbGuvuHhoaqVatWys7O1nfffaeKigqP5W3atFFZWZmOHDlywXMFAACXFp9f5G5Ly5Ytz9leWFiobt26afjw4ZozZ44OHDig9PR0BQcH6/7771dxcbGcTqfHY8LCwlRSUqKioiJJUnh4uHvZmb7eXIeVl5cnl8vl0eZwhCs6OrrOY9RFSEiwx2%2BcH/XyDvUC4EsOh51tzcWy7QrYgPXvJCUlKSkpyf13p06dNGLECG3YsEH333%2B/nE6nysrKPB5TVlamiIgId7AqLS1VRESE%2B78lqUmTJnWeQ1ZWlhYtWuTRlp6eroyMjAtap9pERjpr7wQ36uUd6gXAF6KiIqyO5%2B9tl88CVlBQkE%2Be5/3339fx48d11113udvKy8sVFhYm6Z8fE3Ho0CH17t1bklRRUaEjR44oNjZWrVu3VmhoqHJycnTDDTdIkg4fPuw%2BjVhXaWlpSklJ8WhzOMJVUGD3bsSQkGBFRjp16lSpqqqqrY7dEFEv71AvAL5kax959rbLdnCrK58FrKefftrjM68qKio0e/Zs95GiM37u52AZYzRz5kz96le/Uvfu3fXFF19oxYoV7i%2BSHjJkiBYuXKjk5GS1bt1ac%2BfOVYsWLZSQkKDQ0FD1799fmZmZmj9/viQpMzNTAwYMcAe0uoiOjq5xOtDlKlRlZf3spKqqqutt7IaIenmHegHwBdvbGX9vu3wSsLp27VrjmqT4%2BHgVFBSooKDA6nOlpqZq0qRJmjZtmo4dO6YWLVpo9OjRGjRokCRp6NChKiwsVHp6uvLz89WxY0ctWbJEoaGhkv75GVqzZs3SwIEDVVFRoT59%2Bujxxx%2B3OkcAANCwBZlz3eIH61yuQutjOhzBioqKUEFBMUcY6oB6eac%2B69V/3lar4wEIfO%2BOTaq9Ux2cve1q2fIyK%2BN6i9uDAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsc/h7AgDs6D9vq7%2BnAAD4/ziCBQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJZxFyHwb3BXHgDgQnEECwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMCygA5Y%2Bfn5Sk1N1fbt2z3a9%2B7dq2HDhik%2BPl4pKSlavXp1ncesqqrSrFmz1KNHD8XHx%2BuPf/yj8vLybE8dAAA0YAEbsHbv3q20tDQdPXrUo/3kyZN64IEHdMcdd2jnzp2aMWOGZs6cqS%2B//LJO4y5evFhbt27VG2%2B8oY8//lhhYWGaMmVKfawCAABooAIyYK1bt07pseRdAAASkklEQVQTJkzQuHHjaizbvHmzmjVrpnvvvVcOh0M33nijBg4cqNdee61OY69evVqjRo3SVVddpSZNmmjy5Mn66KOP9P3339teDQAA0EAF5Jc933TTTRo4cKAcDkeNkHXo0CHFxsZ6tLVt21Zr1qypddzCwkL99NNPHo9v0aKFmjZtqoMHD%2Bqaa66p0/zy8vLkcrk82hyOcEVHR9fp8XUVEhLs8RvnR70A4OLlcNjZNl8s2/qADFgtW7b8t8uKi4vldDo92sLCwlRSUlLruMXFxZKk8PDwGo8/s6wusrKytGjRIo%2B29PR0ZWRk1HkMb0RGOmvvBDfqBQAXn6ioCKvj%2BXtbH5AB63ycTqcKCws92srKyhQRUfv/uDPBrLS09IIef0ZaWppSUlI82hyOcBUU1D2k1UVISLAiI506dapUVVXVVsduiKgXAFy8bO0jz97W2w5uddXgAlZsbKy2bt3q0ZaTk6OYmJhaH9u0aVNdccUVysnJcZ8mdLlcOnHiRI3TjucTHR1d43Sgy1Woysr62alXVVXX29gNEfUCgIuP7e2yv7f1De5ilNTUVB0/flwvv/yyKioq9Nlnn2n9%2BvUaMmRInR4/ePBgLV68WN9//72Kior0zDPPqFu3brr22mvreeYAAKChaHBHsKKiovTSSy9pxowZWrBggZo3b64pU6aoe/fudXp8enq6Kisrde%2B996q4uFiJiYmaN29ePc8aAAA0JEHGGOPvSVwKXK7C2jt5yeEIVlRUhAoKijnlVQfe1qv/vK219gEA2PHu2CQr45y9rW/Z8jIr43qrwZ0iBAAA8DcCFgAAgGUELAAAAMsIWAAAAJYRsAAAACxrcB/TgIsXd%2BUBAC4VHMECAACwjCNYAS5h8kZ/TwEAAJyFI1gAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLGmzA2rBhg66//nrFx8e7fx5%2B%2BGFJ0pYtWzRw4EDFxcWpf//%2B%2BvDDDz0eu3TpUiUnJysuLk7Dhw/XN998449VAAAAAarBBqx9%2B/Zp0KBB2rNnj/tn9uzZOnLkiEaPHq0xY8Zo165dGj16tMaOHatjx45JktatW6dXX31VL774orZv36727dsrIyNDxhg/rxEAAAgUDTpgdejQoUb7unXrlJCQoL59%2B8rhcOjWW29V165dlZWVJUlatWqV7rnnHsXExKhx48YaP368cnNztX37dl%2BvAgAACFANMmBVV1dr//79%2Bt///V/17t1bycnJevzxx3Xy5Enl5OQoNjbWo3/btm2VnZ0tSTWWh4aGqlWrVu7lAAAAtXH4ewL1IT8/X9dff7369eunBQsWqKCgQI888ogefvhhlZeXy%2Bl0evQPCwtTSUmJJKm4uPi8y%2BsiLy9PLpfLo83hCFd0dPQFrtG5hYQ0yHwMALgEORx29mln9o3%2B3kc2yIDVokULvfbaa%2B6/nU6nHn74Yf3mN79RYmKiysrKPPqXlZUpIiLC3fd8y%2BsiKytLixYt8mhLT09XRkaGt6sCAMAlISqq7vvZuoiMdNbeqR41yICVnZ2tv/zlLxo/fryCgoIkSeXl5QoODlanTp104MABj/45OTnu67ViYmJ06NAh9e7dW5JUUVGhI0eO1DiteD5paWlKSUnxaHM4wlVQUPxzVqsGf6dzAABssbWPDAkJVmSkU6dOlaqqqtp6cKurBhmwmjVrptdee01NmzbVfffdp7y8PM2ePVt33nmn7rjjDr3yyivasGGDbr75Zm3evFk7duzQ5MmTJUlDhgzRwoULlZycrNatW2vu3Llq0aKFEhIS6vz80dHRNU4HulyFqqystrqeAAA0FLb3kVVV1X7d7zbIgHXllVdqyZIlmjNnjhYvXqzGjRvrtttu08MPP6zGjRvrueeeU2ZmpiZPnqyrr75aCxcuVOvWrSVJQ4cOVWFhodLT05Wfn6%2BOHTtqyZIlCg0N9fNaAQCAQBFk%2BIAnn3C5Cq2P6XAEKzXzY%2BvjAgDga%2B%2BOTbIyjsMRrKioCBUUFKuyslotW15mZVxvcREPAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDIC1jn84x//0IMPPqiEhAQlJiZqxowZqqys9Pe0AABAgCBgncPYsWMVHh6ujz/%2BWGvWrNG2bdv08ssv%2B3taAAAgQBCwzvLdd99px44devjhh%2BV0OnXNNdfowQcf1GuvvebvqQEAgABBwDrLoUOH1KxZM11xxRXutjZt2ig3N1enTp3y48wAAECgcPh7Aheb4uJiOZ1Oj7Yzf5eUlCgyMrLWMfLy8uRyuTzaHI5wRUdH25uopJAQ8jEAoGFwOOzs087sG/29jyRgnSU8PFylpaUebWf%2BjoiIqNMYWVlZWrRokUfbQw89pNGjR9uZ5P%2BXl5enEVceUlpamvXw1hDl5eUpKyuLetUR9ao7auUd6uUd6uWdvLw8vfLKMr/Xi0MgZ4mJidGJEyd0/Phxd9vhw4d15ZVX6rLLLqvTGGlpaVq7dq3HT1pamvW5ulwuLVq0qMbRMpwb9fIO9ao7auUd6uUd6uWdi6VeHME6S6tWrdSlSxc988wzeuqpp1RQUKDnn39eQ4cOrfMY0dHR/CsDAIBLGEewzmHBggWqrKxUnz599Jvf/EY9e/bUgw8%2B6O9pAQCAAMERrHNo0aKFFixY4O9pAACAABUybdq0af6eBC5cRESEunXrVucL8C911Ms71KvuqJV3qJd3qJd3LoZ6BRljjN%2BeHQAAoAHiGiwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQSsi8A//vEPPfjgg0pISFBiYqJmzJihyspK9/K9e/dq2LBhio%2BPV0pKilavXu3H2V5c8vPzlZqaqu3bt7vbtmzZojvuuEPx8fG6/fbb9d5777mXGWP03HPPKSUlRZ07d9bAgQO1ceNG9/KqqirNmjVLPXr0UHx8vP74xz8qLy/Pp%2Btk2%2BHDhzVy5EglJCSoV69eWrx4saqrq/09rYtWbe9HeKrt9fXOO%2B%2Bof//%2B6ty5s/r166f/%2BZ//8Xj80qVLlZycrLi4OA0fPlzffPONr1fhonK%2B7f26deuUmpqquLg4DR48WHv27HEva4jbLknKzs7Wfffdp27duikpKUkTJ05Ufn6%2Be3lt%2B8fz1azeGfjdb3/7WzN%2B/HhTUlJijh49am677TazdOlSY4wxJ06cMN26dTMrV640FRUV5tNPPzXx8fFm7969fp61/%2B3atcv07dvXxMbGms8%2B%2B8wYY8zf/vY30759e7Nq1SpTUVFhdu7caeLj493Lly9fblJSUkxOTo6prq42f/3rX03Hjh3d9Vy4cKEZOHCgyc3NNYWFhWbs2LFm1KhRflvHn6uoqMj06tXLTJ482RQXF5sffvjBDBgwwCxcuNDfU7tone/9CE%2B1vb4OHjxobrjhBrNnzx5jjDG7d%2B827du3Nzt37jTGGLN27VrTs2dP8/XXX5uysjIzc%2BZMc9ttt5nq6mq/rZM/nW97/9lnn5n4%2BHiza9cuU15ebpYvX24SExNNSUmJMabhbbuMMaa0tNQkJSWZ%2BfPnm9OnT5v8/HwzatQo84c//MEYU/v%2Bsbaa1TcClp8dOXLExMbGmp9%2B%2Bsnd9s4775hevXoZY4xZtWqVufnmmz0e88QTT5iJEyf6dJ4Xm7Vr15pevXqZd955xyNgzZ492wwfPtyj7xNPPGHGjBljjDFm/vz55o033vBYfscdd5jly5cbY4xJTk42b7/9tnuZy%2BUy7dq1M0ePHq3Htak/W7ZsMR07djSnT592t73zzjumR48el%2BxO7Hxqez/CU22vr02bNpn27dub3bt3m%2BrqavP555%2BbTp06uXeAd911l1m8eLH7seXl5SY%2BPt5s27bN5%2BtyMTjf9n78%2BPFmypQpHstuueUWs2bNGmNMw9t2GWPM4cOHzciRI01lZaW77f333zedO3c2xtS%2Bf6ytZvWNU4R%2BdujQITVr1kxXXHGFu61NmzbKzc3VqVOndOjQIcXGxno8pm3btsrOzvb1VC8qN910k9577z3deuutHu1VVVUKDw/3aAsODnafdsjIyNDgwYPdyw4fPqxDhw6pffv2Kiws1E8//eRR7xYtWqhp06Y6ePBgPa5N/amurlZoaKhCQ0PdbUFBQTp%2B/LhOnTrlx5ldnGp7P8JTba%2Bvm266SXFxcbr77rvVvn173XXXXRozZow6deokScrJyfF4v4WGhqpVq1aX7PbtfNv7s2v1r8sa4rZLkq677jotW7ZMISEh7rZNmzapffv2ks5fL6nm6%2Bvs5fWNgOVnxcXFcjqdHm1n/i4pKTnn8rCwMJWUlPhsjhejli1byuFw1GhPTU3VJ598ok2bNqmyslK7d%2B/Whg0bdPr06Rp9v/32W40aNUq33367unbtquLiYkmqEdDCwsLcywJN586dFRYWpj//%2Bc8qLS3V3//%2Bd7344ouSpLKyMj/P7uJT2/sRnmp7fZWXl%2BuXv/ylli9frr1792rJkiVauHChPvnkE0nnrvelvH07Xz1qWyY1rG3X2Ywxmjt3rj788ENNnjxZUu2vH3%2B/vghYfhYeHq7S0lKPtjN/R0REyOl01tgRlpWVKSIiwmdzDCSdO3fWn/70Jy1atEhJSUl68cUXNXjwYEVGRnr0%2B%2BCDD5SWlqabb75ZM2bMkPR/O9Kz/38Ecr0jIyO1dOlS7d27V7169dLYsWN1xx13uJfBU23vR3iq7fW1cOFCNWrUSD169FBoaKh69eql2267TVlZWZLE9u0s56tHbcukhrXt%2BldFRUXKyMjQ%2BvXrtXLlSrVr105S7a8ff7%2B%2BCFh%2BFhMToxMnTuj48ePutsOHD%2BvKK6/UZZddptjYWB06dMjjMTk5OYqJifH1VAPCiRMnFBMTo/Xr12v79u16/vnn9eOPP6pDhw7uPs8995zGjx%2Bvxx9/XI8%2B%2BqiCgoIkSU2bNtUVV1yhnJwcd1%2BXy6UTJ07UOMwcKMrLy1VZWakVK1Zo%2B/btWr16tYKDg9W2bdsa/7JD7e9HeKrt9ZWbm6uKigqPxzgcDvcpxZiYGI/tW0VFhY4cORKw77ef63zb%2B7Nr9a/LGuK264yjR49qyJAhKioq0po1a9zhSjp/vaSar6%2Bzl9c7n1zphfO6%2B%2B67zbhx40xhYaH7rqUFCxYYY4zJz883CQkJZvny5aa8vNxs27btkr4I9Fz%2B9SL3L774wsTFxZkDBw6YiooK884775hOnTqZr7/%2B2hhjzEsvvWS6dOli9u/ff86x5s6dawYMGGCOHj3qvhPnt7/9rc/WxbbTp0%2BbhIQEs2rVKlNdXW327dtnevbsabKysvw9tYvW%2Bd6P8FTb62vVqlWmU6dO5qOPPjLV1dVm%2B/btJj4%2B3nzwwQfu5T179jQHDhxw30WYmppqysvL/blafnO%2B7f2ZO%2BS2bdvmviOua9eupqCgwBjT8LZdxvzzLsFevXqZRx991FRVVdVYXtv%2Bsbaa1TcC1kXA5XKZ0aNHm27dupnu3bubZ5991uOuiS%2B//NKkpaWZ%2BPh406dPnxp3wV3q/jVgGWPMf//3f5vevXubuLg4M3jwYPPpp58aY4yprq42Xbp0Mddff72Ji4vz%2BDlzJ1N5ebmZPXu26dmzp%2BncubP54x//aI4fP%2B6X9bJlx44d5s477zRxcXGmT58%2BZsWKFf6e0kWttvcjPNX2%2BlqxYoW5%2BeabTXx8vLntttvMW2%2B95V5WXV1tXnzxRZOSkmLi4uLM8OHDzTfffOPrVbionG97/%2Babb5p%2B/fqZuLg4M3ToUPPFF1%2B4lzXEbddLL71kYmNjzQ033FBjm31GbfvH89WsvgUZY4xvjpUBAABcGrgGCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACw7P8BLOGeSBZ4hTIAAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-5047076293458780941">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">2014</td>
        <td class="number">559</td>
        <td class="number">2.6%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2006</td>
        <td class="number">453</td>
        <td class="number">2.1%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2005</td>
        <td class="number">450</td>
        <td class="number">2.1%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2004</td>
        <td class="number">433</td>
        <td class="number">2.0%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2003</td>
        <td class="number">420</td>
        <td class="number">1.9%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2007</td>
        <td class="number">417</td>
        <td class="number">1.9%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1977</td>
        <td class="number">417</td>
        <td class="number">1.9%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1978</td>
        <td class="number">387</td>
        <td class="number">1.8%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1968</td>
        <td class="number">381</td>
        <td class="number">1.8%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2008</td>
        <td class="number">367</td>
        <td class="number">1.7%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (106)</td>
        <td class="number">17313</td>
        <td class="number">80.2%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-5047076293458780941">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">1900</td>
        <td class="number">87</td>
        <td class="number">0.4%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1901</td>
        <td class="number">29</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:34%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1902</td>
        <td class="number">27</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:31%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1903</td>
        <td class="number">46</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:53%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1904</td>
        <td class="number">45</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:52%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">2011</td>
        <td class="number">130</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:24%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2012</td>
        <td class="number">170</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:31%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2013</td>
        <td class="number">201</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:36%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2014</td>
        <td class="number">559</td>
        <td class="number">2.6%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2015</td>
        <td class="number">38</td>
        <td class="number">0.2%</td>
        <td>
            <div class="bar" style="width:7%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_yr_renovated">yr_renovated<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>71</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.3%</td>
                </tr>
                <tr class="alert">
                    <th>Missing (%)</th>
                    <td>17.8%</td>
                </tr>
                <tr class="alert">
                    <th>Missing (n)</th>
                    <td>3842</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>83.637</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>0</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>2015</td>
                </tr>
                <tr class="alert">
                    <th>Zeros (%)</th>
                    <td>78.8%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-6247364184281289719">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAAQlJREFUeJzt1cEJAlEQBUEVQzIIc/JsTgaxOY13kQaF5YtU3QfepZnjzMwBeOu0egD8svPqAa8ut8fHN9v9usMS8EEgCQSCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCASCQCAIBIJAIAgEgkAgCATCefUA/tfl9vj4Zrtfd1jyPR8EwnFmZvUI%2BFU%2BCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIBIFAEAgEgUAQCIQn/z8Okdc9YtkAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-6247364184281289719,#minihistogram-6247364184281289719"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-6247364184281289719">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-6247364184281289719"
                                                  aria-controls="quantiles-6247364184281289719" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-6247364184281289719" aria-controls="histogram-6247364184281289719"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-6247364184281289719" aria-controls="common-6247364184281289719"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-6247364184281289719" aria-controls="extreme-6247364184281289719"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-6247364184281289719">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>0</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>2015</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>2015</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>0</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>399.95</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>4.7819</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>18.92</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>83.637</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>160.26</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>4.5734</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>1485000</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>159960</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-6247364184281289719">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3XtYVWXC/vFb2CgbEKEUO0xdOgJ2UIMwUTTfxMgxxRykmHKcLLNSkvT1UKapo%2BFhopOZjlnmjPq%2BeaRy0rTmMnVK8VBZOWFgeZjxAAooR%2BWwfn/0sn/tMN02j8Lafj/X5R/7edZe%2B7nBBTdrrQ2NLMuyBAAAAGN86nsBAAAA3oaCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMc9T3Ai4X%2BfnFxvfp49NIV1wRqIKCUtXUWMb3X1/IZS/ksh9vzUYue7lUuVq0aHrR9n0unMGyMR%2BfRmrUqJF8fBrV91KMIpe9kMt%2BvDUbuezFW3PVomABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGGO%2Bl4A/jMdJ3xQ30vw2LqRXet7CQAAXBKcwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwxp8wSooKFBCQoKysrJcY9nZ2XrwwQcVHR2tuLg4zZgxQ1VVVa75zMxMJSQkKCoqSklJSfr8889dc9XV1Zo1a5bi4uIUHR2tYcOGKS8vzzV/4sQJDR8%2BXB07dlRsbKzS09Pd9g0AAHA%2BDbpg7dq1SykpKTp48KBrrKCgQIMHD1ZcXJy2b9%2Bu5cuX6%2BOPP9Zf/vIXSVJWVpamTZummTNnaseOHerXr5%2BGDRum8vJySdK8efP0ySefaNWqVdqyZYv8/f01ceJE1/5HjhypgIAAbdmyRStXrtTWrVu1aNGiS5obAADYW4MtWJmZmRozZoxGjRrlNv7OO%2B%2BoVatWeuyxx%2BTn56df/epXWrhwoXr37i1JWrFihfr06aOYmBj5%2Bflp8ODBCg0N1dq1a13zQ4cO1dVXX62goCBNmDBBmzdv1qFDh3TgwAFt375dY8eOldPp1HXXXafhw4dr6dKllzw/AACwL0d9L%2BDndOvWTYmJiXI4HG4l68svv1RkZKQmTZqkv//973I6nRowYIAee%2BwxSVJubq4GDBjgtq/w8HBlZ2eruLhYR48eVWRkpGuuefPmatasmfbu3StJCgkJUcuWLV3zbdq00eHDh3Xq1CkFBwd7tPa8vDzl5%2Be7jTkcAQoLC7uwD8J5%2BPo22H58Vg6HZ%2ButzWW3fOdDLnvx1lyS92Yjl714a65aDbZgtWjR4qzjJ0%2Be1EcffaQpU6bo2Wef1b59%2B/T444%2BrcePGGjJkiEpLS%2BV0Ot2e4%2B/vr7KyMpWWlkqSAgIC6szXzv30ubWPy8rKPC5Yy5Yt05w5c9zGUlNTlZaW5tHzvVVoaOAFbR8c7Dz/RjZELnvx1lyS92Yjl714a64GW7B%2BTuPGjdW%2BfXslJydLkm644Qb9/ve/17p16zRkyBA5nU5VVFS4PaeiokKhoaGuslR7P9aP5wMDA2VZVp252seBgZ6Xg5SUFMXHx7uNORwBKiws9XgfnrBb6/c0v6%2Bvj4KDnTp1qlzV1TUXeVWXDrnsxVtzSd6bjVz2cqlyXegP96bYrmC1adPG7R2FklRTUyPLsiRJERERysnJcZvPzc1V9%2B7d1axZM7Vs2VK5ubmuy4T5%2BfkqKipSZGSkampqVFRUpOPHj6t58%2BaSpH379umqq65S06ZNPV5jWFhYncuB%2BfnFqqryngPjl7jQ/NXVNV75MSOXvXhrLsl7s5HLXrw1l71OgUgaMGCAvv32Wy1YsEDV1dXau3evlixZonvuuUeSlJycrDVr1mjbtm2qrKzUokWLdOLECSUkJEiSkpKSNG/ePB06dEglJSWaPn26OnXqpOuvv16tWrVSTEyMpk%2BfrpKSEh06dEhz5851nS0DAADwhC3PYC1ZskR/%2BtOf9Prrr8vf31/333%2B/Bg0aJEnq0qWLJk%2BerClTpujYsWMKDw/XggULFBISIumHe6Gqqqo0cOBAlZaWKjY2Vi%2B//LJr/7Nnz9bUqVPVs2dP%2Bfj4qH///ho%2BfHi9ZAUAAPbUyKq9toaLKj%2B/2Pg%2BHQ4fJWRsMb7fi2XdyK4ebedw%2BCg0NFCFhaVeddqYXPbirbkk781GLnu5VLlatPD8Fh%2BTbHeJEAAAoKGjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABjW4AtWQUGBEhISlJWVVWcuLy9PcXFxWr16tdv4ggUL1L17d0VFRWnQoEH67rvvXHNlZWUaP368YmNjFRMTo3Hjxqm0tNQ1//333%2BvBBx9UdHS0unXrpj//%2Bc8XLxwAAPBKDbpg7dq1SykpKTp48GCduZqaGo0ZM0aFhYVu45mZmVq8eLHefPNNZWVl6eabb1ZaWposy5IkTZs2TUeOHNH69eu1YcMGHTlyRBkZGZKkyspKPf7442rfvr2ysrL0%2Buuva%2BnSpVq3bt3FDwsAALxGgy1YmZmZGjNmjEaNGnXW%2Bddee01XXXWVrr76arfx5cuX64EHHlBERISaNGmi0aNH6/Dhw8rKylJ5ebnWrFmjtLQ0hYSE6Morr9SYMWO0evVqlZeXa8eOHcrLy1NaWpoaN26sm266SYMGDdLSpUsvRWQAAOAlHPW9gJ/TrVs3JSYmyuFw1ClZ27Zt0/vvv69Vq1YpMTHRbS43N1dDhw51Pfbz81OrVq2UnZ2tkJAQVVZWKjIy0jXfpk0bVVRUaP/%2B/crJyVHr1q3VuHFj13x4eLhef/31C1p7Xl6e8vPz3cYcjgCFhYVd0H7Ox9e3wfbjs3I4PFtvbS675TsfctmLt%2BaSvDcbuezFW3PVarAFq0WLFmcdP3HihJ555hnNnj1bgYGBdeZLS0vldDrdxvz9/VVWVqaSkhJJUkBAgGuudtvS0tKzPtfpdKqsrOyC1r5s2TLNmTPHbSw1NVVpaWkXtB9vExpa9/N1LsHBzvNvZEPkshdvzSV5bzZy2Yu35mqwBetsLMvSuHHjNGjQILVr1%2B6s2zidTlVUVLiNVVRUKDAw0FWsysvLXeWsvLxckhQUFKSAgADX41o/3tZTKSkpio%2BPdxtzOAJUWFj6M8/4ZezW%2Bj3N7%2Bvro%2BBgp06dKld1dc1FXtWlQy578dZckvdmI5e9XKpcF/rDvSm2KlhHjhzR9u3btXv3br322muSpJKSEv3xj3/U%2BvXrNX/%2BfEVERCgnJ0c9evSQ9MON6/v371dkZKRat24tPz8/5ebm6pZbbpEk7du3z3UZ8cSJE9q/f7%2BqqqrkcPzwocnNzVVERMQFrTMsLKzO5cD8/GJVVXnPgfFLXGj%2B6uoar/yYkctevDWX5L3ZyGUv3prLVqdArrnmGn311VfauXOn698111yjyZMna/78%2BZKkAQMGaMmSJcrOztbp06f1wgsvqHnz5urYsaOcTqd69%2B6tjIwMFRQUqKCgQBkZGerbt6/8/f0VGxur0NBQvfDCCzp9%2BrSys7O1ePFiJScn13NyAABgJ7Y6g%2BWJ5ORkFRcXKzU1VQUFBWrfvr3mz58vPz8/SdLkyZM1a9YsJSYmqrKyUj179tSzzz4rSXI4HFq4cKGmTp2qrl27KiAgQIMGDVJSUlJ9RgIAADbTyKr9BVG4qPLzi43v0%2BHwUULGFuP7vVjWjezq0XYOh49CQwNVWFjqVaeNyWUv3ppL8t5s5LKXS5WrRYumF23f52KrS4QAAAB2QMECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwrMEXrIKCAiUkJCgrK8s1tn79et1zzz269dZbFR8frzlz5qimpsY1n5mZqYSEBEVFRSkpKUmff/65a666ulqzZs1SXFycoqOjNWzYMOXl5bnmT5w4oeHDh6tjx46KjY1Venq6qqqqLk1YAADgFRp0wdq1a5dSUlJ08OBB19jXX3%2BtcePGaeTIkdq5c6cWLFig1atXa9GiRZKkrKwsTZs2TTNnztSOHTvUr18/DRs2TOXl5ZKkefPm6ZNPPtGqVau0ZcsW%2Bfv7a%2BLEia79jxw5UgEBAdqyZYtWrlyprVu3uvYNAADgiQZbsDIzMzVmzBiNGjXKbfzf//63fve736lHjx7y8fFRmzZtlJCQoB07dkiSVqxYoT59%2BigmJkZ%2Bfn4aPHiwQkNDtXbtWtf80KFDdfXVVysoKEgTJkzQ5s2bdejQIR04cEDbt2/X2LFj5XQ6dd1112n48OFaunTpJc8PAADsy1HfC/g53bp1U2JiohwOh1vJ6tWrl3r16uV6XFFRoY8//liJiYmSpNzcXA0YMMBtX%2BHh4crOzlZxcbGOHj2qyMhI11zz5s3VrFkz7d27V5IUEhKili1buubbtGmjw4cP69SpUwoODvZo7Xl5ecrPz3cbczgCFBYW5mF6z/j6Nth%2BfFYOh2frrc1lt3znQy578dZckvdmI5e9eGuuWg22YLVo0eK825SUlOjJJ5%2BUv7%2B/Bg8eLEkqLS2V0%2Bl0287f319lZWUqLS2VJAUEBNSZr5376XNrH5eVlXlcsJYtW6Y5c%2Ba4jaWmpiotLc2j53ur0NDAC9o%2BONh5/o1siFz24q25JO/NRi578dZcDbZgnc93332ntLQ0XXnllfrrX/%2BqoKAgST8UooqKCrdtKyoqFBoa6ipLtfdj/Xg%2BMDBQlmXVmat9HBjoeTlISUlRfHy825jDEaDCwlKP9%2BEJu7V%2BT/P7%2BvooONipU6fKVV1dc/4n2AS57MVbc0nem41c9nKpcl3oD/em2LJgbdq0Sf/93/%2Bt%2B%2B67T6NHj5bD8f9jREREKCcnx2373Nxcde/eXc2aNVPLli2Vm5vrukyYn5%2BvoqIiRUZGqqamRkVFRTp%2B/LiaN28uSdq3b5%2BuuuoqNW3a1OP1hYWF1bkcmJ9frKoq7zkwfokLzV9dXeOVHzNy2Yu35pK8Nxu57MVbc9nrFIikL774QqmpqRo/fryeeuopt3IlScnJyVqzZo22bdumyspKLVq0SCdOnFBCQoIkKSkpSfPmzdOhQ4dUUlKi6dOnq1OnTrr%2B%2BuvVqlUrxcTEaPr06SopKdGhQ4c0d%2B5cJScn10dUAABgU7Y7g/XnP/9ZVVVVSk9PV3p6ums8JiZGb7zxhrp06aLJkydrypQpOnbsmMLDw7VgwQKFhIRI%2BuFeqKqqKg0cOFClpaWKjY3Vyy%2B/7NrP7NmzNXXqVPXs2VM%2BPj7q37%2B/hg8ffslzAgAA%2B2pkWZZV34u4HOTnFxvfp8Pho4SMLcb3e7GsG9nVo%2B0cDh%2BFhgaqsLDUq04bk8tevDWX5L3ZyGUvlypXixae3%2BJjku0uEQIAADR0FCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYJjxglVdXW16lwAAALZivGB1795df/rTn5Sbm2t61wAAALZgvGA98cQT%2Buyzz9S3b1/de%2B%2B9evvtt1VcXGz6ZQAAABos4wXr/vvv19tvv60PPvhAcXFxWrBggbp166bRo0fr008/Nf1yAAAADc5Fu8m9VatWGjVqlD744AOlpqbq73//u4YMGaL4%2BHi99dZb3KsFAAC8luNi7Xj37t165513tHbtWp05c0YJCQlKSkrSsWPH9Morr%2Birr77Siy%2B%2BeLFeHgAAoN4YL1hz587Vu%2B%2B%2BqwMHDqh9%2B/YaNWqU%2Bvbtq6CgINc2vr6%2BmjRpkumXBgAAaBCMF6wlS5aoX79%2BSk5OVnh4%2BFm3adOmjcaMGWP6pQEAABoE4wVr8%2BbNKikpUVFRkWts7dq16tKli0JDQyVJN910k2666SbTLw0AANAgGL/J/Z///Kd69eqlZcuWucaef/55JSYm6ttvvzX9cgAAAA2O8YL1pz/9SXfddZdGjRrlGvvoo4/UvXt3zZw584L3V1BQoISEBGVlZbnGdu/erXvvvVfR0dGKj4/XihUr3J6TmZmphIQERUVFKSkpSZ9//rlrrrq6WrNmzVJcXJyio6M1bNgw5eXlueZPnDih4cOHq2PHjoqNjVV6erqqqqoueN0AAODyZbxg7dmzR48%2B%2BqgaN27sGvP19dWjjz6qL7744oL2tWvXLqWkpOjgwYOusZMnT%2BrRRx9V//79tWPHDqWnp2vGjBn68ssvJUlZWVmaNm2aZs6cqR07dqhfv34aNmyYysvLJUnz5s3TJ598olWrVmnLli3y9/fXxIkTXfsfOXKkAgICtGXLFq1cuVJbt27VokWL/oOPCAAAuNwYL1hBQUFuhajW0aNH5e/v7/F%2BMjMzNWbMGLczYZK0YcMGhYSEaODAgXI4HOrSpYsSExO1dOlSSdKKFSvUp08fxcTEyM/PT4MHD1ZoaKjWrl3rmh86dKiuvvpqBQUFacKECdq8ebMOHTqkAwcOaPv27Ro7dqycTqeuu%2B46DR8%2B3LVvAAAATxgvWL169dKUKVP06aefqqSkRKWlpdq2bZumTp2qhIQEj/fTrVs3ffjhh7r77rvdxnNychQZGek2Fh4eruzsbElSbm7uz84XFxfr6NGjbvPNmzdXs2bNtHfvXuXk5CgkJEQtW7Z0zbdp00aHDx/WqVOnPF47AAC4vBl/F%2BHo0aN16NAhPfzww2rUqJFrPCEhQePGjfN4Py1atDjreGlpqZxOp9uYv7%2B/ysrKzjtfWloqSQoICKgzXzv30%2BfWPi4rK1NwcLBHa8/Ly1N%2Bfr7bmMMRoLCwMI%2Be7ylf34v2i/gvCofDs/XW5rJbvvMhl714ay7Je7ORy168NVct4wXL6XRq/vz5%2Bv7777V37175%2BfmpTZs2atWqlbH9//SPR1dUVCgwMNA1X1FRUWc%2BNDTUVZZq78f66fMty6ozV/u4dv%2BeWLZsmebMmeM2lpqaqrS0NI/34Y1CQz3/GEpScLDz/BvZELnsxVtzSd6bjVz24q25LtqfymndurVat25tfL%2BRkZH65JNP3MZyc3MVEREhSYqIiFBOTk6d%2Be7du6tZs2Zq2bKl22XE/Px8FRUVKTIyUjU1NSoqKtLx48fVvHlzSdK%2Bfft01VVXqWnTph6vMSUlRfHx8W5jDkeACgtLLzjvudit9Xua39fXR8HBTp06Va7q6pqLvKpLh1z24q25JO/NRi57uVS5LvSHe1OMF6zvv/9eU6dO1a5du1RZWVln/ptvvvmP9p%2BQkKDnn39eixYt0sCBA7Vr1y6tWbNGc%2BfOlSQlJycrNTVVvXv3VkxMjJYuXaoTJ0647v9KSkrSvHnz1L59e4WGhmr69Onq1KmTrr/%2BeklSTEyMpk%2BfrqlTp6qwsFBz585VcnLyBa0xLCyszuXA/PxiVVV5z4HxS1xo/urqGq/8mJHLXrw1l%2BS92chlL96ay3jBmjJlig4fPqwxY8Zc0FkfT4WGhmrhwoVKT0/X7NmzdcUVV2jixInq3LmzJKlLly6aPHmypkyZomPHjik8PFwLFixQSEiIpB8u1VVVVWngwIEqLS1VbGysXn75Zdf%2BZ8%2BeralTp6pnz57y8fFR//79NXz4cOM5AACA92pkWZZlcocdOnTQX/7yF0VHR5vcre3l5xeff6ML5HD4KCFji/H9XizrRnb1aDuHw0ehoYEqLCz1qp9qyGUv3ppL8t5s5LKXS5WrRQvzJ3s8YfwmntDQ0Au6IRwAAMDbGC9YgwYN0osvvljnnX4AAACXC%2BP3YG3atElffPGFYmNjdeWVV7r9yRxJ%2Bvvf/276JQEAABoU4wUrNjZWsbGxpncLAABgG8YL1hNPPGF6lwAAALZyUX5TZXZ2tsaPH6/f/e53OnbsmJYuXaqsrKyL8VIAAAANjvGC9fXXX%2Bvee%2B/Vv/71L3399dc6c%2BaMvvnmGz388MPauHGj6ZcDAABocIwXrIyMDD388MNavHix/Pz8JEnPPfec/vCHP9T5%2B3wAAADe6KKcwerfv3%2Bd8fvvv1/fffed6ZcDAABocIwXLD8/P5WUlNQZP3z4sJxO7/yL2QAAAD9mvGDdeeedeuGFF1RYWOga27dvn9LT03XHHXeYfjkAAIAGx3jBeuqpp1RRUaG4uDiVl5crKSlJffv2lcPh0Lhx40y/HAAAQINj/PdgBQUF6e2339bWrVv1z3/%2BUzU1NYqMjNTtt98uH5%2BL8lshAAAAGhTjBatWly5d1KVLl4u1ewAAgAbLeMGKj49Xo0aNfnaev0UIAAC8nfGC9dvf/tatYFVWVurAgQPavHmzRo4cafrlAAAAGhzjBWvEiBFnHV%2ByZIl27dqlP/zhD6ZfEgAAoEG5ZHed9%2BjRQ5s2bbpULwcAAFBvLlnB2r59u5o0aXKpXg4AAKDeGL9E%2BNNLgJZlqaSkRHv37uXyIAAAuCwYL1jXXHNNnXcR%2Bvn56cEHH1RiYqLplwMAAGhwjBesmTNnmt4lAACArRgvWDt27PB429tuu830ywMAANQ74wVr8ODBsizL9a9W7WXD2rFGjRrpm2%2B%2BMf3yAAAA9c54wXr11Vc1Y8YMPfXUU%2BrcubP8/Py0e/duTZkyRQ888IB69Ohh%2BiUBAAAaFOO/pmHWrFmaPHmy7rzzTgUFBalJkybq1KmTpk6dqoULF%2Braa691/QMAAPBGxgtWXl6err766jrjQUFBKiwsNPY6e/bs0cCBA9WxY0d169ZNzz33nM6cOSNJ2rRpkxITExUVFaXevXtr48aNbs9dsGCBunfvrqioKA0aNEjfffeda66srEzjx49XbGysYmJiNG7cOJWWlhpbNwAA8H7GC1ZUVJRefPFFlZSUuMaKior0/PPPq0uXLkZeo6amRo899ph69eql7du3a%2BXKlfrHP/6hBQsWaP/%2B/RoxYoSefPJJ7dy5UyNGjNDIkSN17NgxSVJmZqYWL16sN998U1lZWbr55puVlpbmujds2rRpOnLkiNavX68NGzboyJEjysjIMLJuAABweTBesCZOnKjdu3ere/fuSkpKUlJSknr06KFDhw5p0qRJRl7j5MmTys/PV01NjasY%2Bfj4yOl0KjMzUx07dtSdd94ph8Ohu%2B%2B%2BW7fddpuWLVsmSVq%2BfLkeeOABRUREqEmTJho9erQOHz6srKwslZeXa82aNUpLS1NISIiuvPJKjRkzRqtXr1Z5ebmRtQMAAO9nvGC1adNGa9eu1ejRoxUVFaWoqChNmDBB7777rq666iojrxEaGqrBgwdr1qxZat%2B%2Bvf7rv/5LrVq10uDBg5Wbm6vIyEi37cPDw5WdnS1Jdeb9/PzUqlUrZWdn68CBA6qsrHSbb9OmjSoqKrR//34jawcAAN7P%2BLsIJSk4OFj33nuv/vWvf%2Bm6666T9EORMaWmpkb%2B/v569tlnlZycrAMHDuiJJ57Q7NmzVVpaKqfT6ba9v7%2B/ysrKJOmc87WXNQMCAlxztdteyH1YeXl5ys/PdxtzOAIUFhbmeUgP%2BPpesj8laYTD4dl6a3PZLd/5kMtevDWX5L3ZyGUv3pqrlvGCZVmWXnjhBS1evFiVlZVav369XnrpJTVp0kRTp041UrQ%2B/PBDrV%2B/Xh988IEkKSIiQqmpqUpPT9ett96qiooKt%2B0rKioUGBgo6YfC9HPztcWqvLzctX3tpcGgoCCP17ds2TLNmTPHbSw1NVVpaWkXkNL7hIYGXtD2wcHO829kQ%2BSyF2/NJXlvNnLZi7fmMl6wFi9erHfffVeTJ0/W1KlTJUl33nmn/vjHP7ruafpPHTlyxPWOwVoOh0N%2Bfn6KjIzUnj173OZyc3PVrl07ST%2BUsZycHNfv46qsrNT%2B/fsVGRmp1q1by8/PT7m5ubrlllskSfv27XNdRvRUSkqK4uPjf7K%2BABUWmn03ot1av6f5fX19FBzs1KlT5aqurrnIq7p0yGUv3ppL8t5s5LKXS5XrQn%2B4N8V4wVq2bJkmTZqkhIQETZs2TZJ09913q3HjxkpPTzdSsLp166YXXnhBf/7znzV06FAdPnxY8%2BbNU2Jiovr166e33npLa9eu1V133aUNGzZo%2B/btmjBhgiRpwIABevXVV9W9e3e1bt1aL730kpo3b66OHTvKz89PvXv3VkZGhl555RVJUkZGhvr27St/f3%2BP1xcWFlbncmB%2BfrGqqrznwPglLjR/dXWNV37MyGUv3ppL8t5s5LIXb81lvGD961//0o033lhnvG3btjp%2B/LiR1wgPD9f8%2BfP18ssv64033lDTpk3Vr18/paamqnHjxnrttdeUkZGhCRMm6Nprr9Wrr76q1q1bS5KSk5NVXFys1NRUFRQUqH379po/f77r0uXkyZM1a9YsJSYmqrKyUj179tSzzz5rZN0AAODyYLxgXXvttfryyy/1q1/9ym1806ZNrhveTYiLi1NUPMBWAAAdtElEQVRcXNxZ526//XbdfvvtZ51r1KiRHn74YT388MNnnQ8KCtK0adNcZ98AAAAulPGCNWTIEP3xj3/UsWPHZFmWtm7dqrfffluLFy/W%2BPHjTb8cAABAg2O8YA0YMEBVVVWaN2%2BeKioqNGnSJF155ZUaNWqU7r//ftMvBwAA0OAYL1jvvfeefvOb3yglJUUFBQWyLEtXXnml6ZcBAABosIy/z/%2B5555z3cx%2BxRVXUK4AAMBlx3jBatWqlfbu3Wt6twAAALZh/BJhRESExowZozfeeEOtWrVSkyZN3OZnzJhh%2BiUBAAAaFOMF6%2BDBg4qJiZGkOn%2BPDwAA4HJgpGDNmDFDTz75pAICArR48WITuwQAALAtI/dg/fWvf3X9UeRaQ4YMUV5enondAwAA2IqRgmVZVp2xzz77TKdPnzaxewAAAFsx/i5CAACAyx0FCwAAwDBjBatRo0amdgUAAGBrxn5Nw3PPPef2O68qKyv1/PPPKzAw0G07fg8WAADwdkYK1m233Vbnd15FR0ersLBQhYWFJl4CAADANowULH73FQAAwP/HTe4AAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADLNtwSoqKtK4ceMUGxur2267TcOHD1deXp4kaffu3br33nsVHR2t%2BPh4rVixwu25mZmZSkhIUFRUlJKSkvT555%2B75qqrqzVr1izFxcUpOjpaw4YNc%2B0XAADAE7YtWCNGjFBZWZk%2B/PBDbdy4Ub6%2Bvnr22Wd18uRJPfroo%2Brfv7927Nih9PR0zZgxQ19%2B%2BaUkKSsrS9OmTdPMmTO1Y8cO9evXT8OGDVN5ebkkad68efrkk0%2B0atUqbdmyRf7%2B/po4cWJ9RgUAADZjy4L19ddfa/fu3Zo5c6aCg4MVFBSkadOmacyYMdqwYYNCQkI0cOBAORwOdenSRYmJiVq6dKkkacWKFerTp49iYmLk5%2BenwYMHKzQ0VGvXrnXNDx06VFdffbWCgoI0YcIEbd68WYcOHarPyAAAwEZsWbC%2B/PJLhYeHa/ny5UpISFC3bt00a9YstWjRQjk5OYqMjHTbPjw8XNnZ2ZKk3Nzcn50vLi7W0aNH3eabN2%2BuZs2aae/evRc/GAAA8AqO%2Bl7AL3Hy5Ent3btX7dq1U2ZmpioqKjRu3Dg99dRTat68uZxOp9v2/v7%2BKisrkySVlpb%2B7HxpaakkKSAgoM587Zwn8vLylJ%2Bf7zbmcAQoLCzM4314wtfXXv3Y4fBsvbW57JbvfMhlL96aS/LebOSyF2/NVcuWBatx48aSpAkTJqhJkyYKCgrSyJEjdd999ykpKUkVFRVu21dUVCgwMFCS5HQ6zzofGhrqKl6192Od7fmeWLZsmebMmeM2lpqaqrS0NI/34Y1CQz3/GEpScLDz/BvZELnsxVtzSd6bjVz24q25bFmwwsPDVVNTo8rKSjVp0kSSVFNTI0m68cYb9T//8z9u2%2Bfm5ioiIkKSFBERoZycnDrz3bt3V7NmzdSyZUu3y4j5%2BfkqKiqqc1nxXFJSUhQfH%2B825nAEqLDQ87NgnrBb6/c0v6%2Bvj4KDnTp1qlzV1TUXeVWXDrnsxVtzSd6bjVz2cqlyXegP96bYsmDFxcXpuuuu0zPPPKMZM2bo9OnTeumll3TnnXeqb9%2B%2Bmj17thYtWqSBAwdq165dWrNmjebOnStJSk5OVmpqqnr37q2YmBgtXbpUJ06cUEJCgiQpKSlJ8%2BbNU/v27RUaGqrp06erU6dOuv766z1eX1hYWJ3Lgfn5xaqq8p4D45e40PzV1TVe%2BTEjl714ay7Je7ORy168NZctC5afn58WL16smTNnqlevXjp9%2BrTi4%2BM1YcIEBQcHa%2BHChUpPT9fs2bN1xRVXaOLEiercubMkqUuXLpo8ebKmTJmiY8eOKTw8XAsWLFBISIikHy7lVVVVaeDAgSotLVVsbKxefvnl%2BowLAABsppFlWVZ9L%2BJykJ9fbHyfDoePEjK2GN/vxbJuZFePtnM4fBQaGqjCwlKv%2BqmGXPbirbkk781GLnu5VLlatGh60fZ9Lva6iQcAAMAGKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACG2b5gVVdXa9CgQXr66addY5s2bVJiYqKioqLUu3dvbdy40e05CxYsUPfu3RUVFaVBgwbpu%2B%2B%2Bc82VlZVp/Pjxio2NVUxMjMaNG6fS0tJLlgcAANif7QvWnDlztHPnTtfj/fv3a8SIEXryySe1c%2BdOjRgxQiNHjtSxY8ckSZmZmVq8eLHefPNNZWVl6eabb1ZaWposy5IkTZs2TUeOHNH69eu1YcMGHTlyRBkZGfWSDQAA2JOtC9bWrVu1YcMG3XXXXa6xzMxMdezYUXfeeaccDofuvvtu3XbbbVq2bJkkafny5XrggQcUERGhJk2aaPTo0Tp8%2BLCysrJUXl6uNWvWKC0tTSEhIbryyis1ZswYrV69WuXl5fUVEwAA2IyjvhfwS504cUITJkzQ3LlztWjRItd4bm6uIiMj3bYNDw9Xdna2a37o0KGuOT8/P7Vq1UrZ2dkKCQlRZWWl2/PbtGmjiooK7d%2B/XzfeeKNHa8vLy1N%2Bfr7bmMMRoLCwsAuNeU6%2Bvvbqxw6HZ%2ButzWW3fOdDLnvx1lyS92Yjl714a65atixYNTU1Gjt2rB566CHdcMMNbnOlpaVyOp1uY/7%2B/iorKzvvfElJiSQpICDANVe77YXch7Vs2TLNmTPHbSw1NVVpaWke78MbhYYGXtD2wcHO829kQ%2BSyF2/NJXlvNnLZi7fmsmXBmj9/vho3bqxBgwbVmXM6naqoqHAbq6ioUGBg4Hnna4tVeXm5a/vaS4NBQUEery8lJUXx8fFuYw5HgAoLzd4sb7fW72l%2BX18fBQc7depUuaqray7yqi4dctmLt%2BaSvDcbuezlUuW60B/uTbFlwXr33XeVl5enjh07SpKrMH300UcaOHCg9uzZ47Z9bm6u2rVrJ0mKiIhQTk6OevToIUmqrKzU/v37FRkZqdatW8vPz0%2B5ubm65ZZbJEn79u1zXUb0VFhYWJ3Lgfn5xaqq8p4D45e40PzV1TVe%2BTEjl714ay7Je7ORy168NZe9ToH8nw8%2B%2BECfffaZdu7cqZ07d6pv377q27evdu7cqX79%2Bmn79u1au3atqqqqtHbtWm3fvl333HOPJGnAgAFasmSJsrOzdfr0ab3wwgtq3ry5OnbsKKfTqd69eysjI0MFBQUqKChQRkaG%2BvbtK39//3pODQAA7MKWZ7DOpU2bNnrttdeUkZGhCRMm6Nprr9Wrr76q1q1bS5KSk5NVXFys1NRUFRQUqH379po/f778/PwkSZMnT9asWbOUmJioyspK9ezZU88%2B%2B2x9RgIAADbTyKr9BVC4qPLzi43v0%2BHwUULGFuP7vVjWjezq0XYOh49CQwNVWFjqVaeNyWUv3ppL8t5s5LKXS5WrRYumF23f52LLS4QAAAANGQULAADAMAoWAACAYRQsAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMNsWrOzsbD300EPq1KmTunbtqnHjxqmgoECStHv3bt17772Kjo5WfHy8VqxY4fbczMxMJSQkKCoqSklJSfr8889dc9XV1Zo1a5bi4uIUHR2tYcOGKS8v75JmAwAA9mbLglVRUaFHHnlE0dHR%2Bsc//qG//e1vKioq0jPPPKOTJ0/q0UcfVf/%2B/bVjxw6lp6drxowZ%2BvLLLyVJWVlZmjZtmmbOnKkdO3aoX79%2BGjZsmMrLyyVJ8%2BbN0yeffKJVq1Zpy5Yt8vf318SJE%2BszLgAAsBlbFqzDhw/rhhtuUGpqqho3bqzQ0FClpKRox44d2rBhg0JCQjRw4EA5HA516dJFiYmJWrp0qSRpxYoV6tOnj2JiYuTn56fBgwcrNDRUa9eudc0PHTpUV199tYKCgjRhwgRt3rxZhw4dqs/IAADARhz1vYBf4te//rXeeOMNt7H169fr5ptvVk5OjiIjI93mwsPDtXLlSklSbm6uBgwYUGc%2BOztbxcXFOnr0qNvzmzdvrmbNmmnv3r267rrrPFpfXl6e8vPz3cYcjgCFhYV5nNETvr726scOh2frrc1lt3znQy578dZckvdmI5e9eGuuWrYsWD9mWZZefvllbdy4UUuWLNFf//pXOZ1Ot238/f1VVlYmSSotLf3Z%2BdLSUklSQEBAnfnaOU8sW7ZMc%2BbMcRtLTU1VWlqax/vwRqGhgRe0fXCw8/wb2RC57MVbc0nem41c9uKtuWxdsEpKSjR%2B/Hjt2bNHS5YsUdu2beV0OlVcXOy2XUVFhQIDf/jm7nQ6VVFRUWc%2BNDTUVbxq78c62/M9kZKSovj4eLcxhyNAhYWelzRP2K31e5rf19dHwcFOnTpVrurqmou8qkuHXPbirbkk781GLnu5VLku9Id7U2xbsA4ePKihQ4fqmmuu0cqVK3XFFVdIkiIjI/XJJ5%2B4bZubm6uIiAhJUkREhHJycurMd%2B/eXc2aNVPLli2Vm5vrukyYn5%2BvoqKiOpcdzyUsLKzO5cD8/GJVVXnPgfFLXGj%2B6uoar/yYkctevDWX5L3ZyGUv3prLXqdA/s/Jkyf14IMP6tZbb9Wbb77pKleSlJCQoOPHj2vRokWqrKzUtm3btGbNGtd9V8nJyVqzZo22bdumyspKLVq0SCdOnFBCQoIkKSkpSfPmzdOhQ4dUUlKi6dOnq1OnTrr%2B%2BuvrJSsAALAfW57BWr16tQ4fPqx169bpgw8%2BcJv7/PPPtXDhQqWnp2v27Nm64oorNHHiRHXu3FmS1KVLF02ePFlTpkzRsWPHFB4ergULFigkJETSD/dKVVVVaeDAgSotLVVsbKxefvnlS54RAADYVyPLsqz6XsTlID%2B/%2BPwbXSCHw0cJGVuM7/diWTeyq0fbORw%2BCg0NVGFhqVedNiaXvXhrLsl7s5HLXi5VrhYtml60fZ%2BLLS8RAgAANGQULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAYRsECAAAwjIIFAABgGAULAADAMAoWAACAYY76XgAAALg4er/8SX0vwWPrRnat7yUYxRksAAAAwyhYAAAAhlGwAAAADKNgAQAAGEbBAgAAMIyCBQAAYBgFCwAAwDAK1lmcOHFCw4cPV8eOHRUbG6v09HRVVVXV97IAAIBNULDOYuTIkQoICNCWLVu0cuVKbd26VYsWLarvZQEAAJugYP3EgQMHtH37do0dO1ZOp1PXXXedhg8frqVLl9b30gAAgE1QsH4iJydHISEhatmypWusTZs2Onz4sE6dOlWPKwMAAHbB3yL8idLSUjmdTrex2sdlZWUKDg4%2B7z7y8vKUn5/vNuZwBCgsLMzcQiX5%2BtqrHzscnq23Npfd8p0PuezFW3NJ3puNXPbm6fcIu6Bg/URAQIDKy8vdxmofBwYGerSPZcuWac6cOW5jTzzxhEaMGGFmkf8nLy9PD16Vo5SUFOPlrT7l5eXpL395g1w2QS778dZs5KprZ/pvLtKq/nN5eXlatmyZ132%2BanlXXTQgIiJCRUVFOn78uGts3759uuqqq9S0aVOP9pGSkqLVq1e7/UtJSTG%2B1vz8fM2ZM6fO2TK7I5e9kMt%2BvDUbuezFW3PV4gzWT7Rq1UoxMTGaPn26pk6dqsLCQs2dO1fJycke7yMsLMwr2zgAAPAMZ7DOYvbs2aqqqlLPnj1133336fbbb9fw4cPre1kAAMAmOIN1Fs2bN9fs2bPrexkAAMCmfKdMmTKlvheBXy4wMFCdOnXy%2BAZ8uyCXvZDLfrw1G7nsxVtzSVIjy7Ks%2Bl4EAACAN%2BEeLAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAcEllZ2dr0KBBatu2rTp16qR27dqpoKCgvpcFGEXBAgBcMiUlJbrnnnvUoUMHffXVV1q4cKEcDofuuOOO%2Bl4aYBQFy6ZOnDih4cOHq2PHjoqNjVV6erqqqqrqe1nnlZ2drYceekidOnVS165dNW7cONdPrpMnT1a7du0UHR3t%2Brds2TLXcxcsWKDu3bsrKipKgwYN0nfffVdfMepYu3atbrrpJre1jx07VpK0adMmJSYmKioqSr1799bGjRvdnttQc7333ntueaKjo9WuXTu1a9dOkvTII4%2Boffv2bvObN2%2BWJFVXV2vWrFmKi4tTdHS0hg0bpry8vPqMI0kqKChQQkKCsrKyXGO7d%2B/Wvffeq%2BjoaMXHx2vFihVuz8nMzFRCQoKioqKUlJSkzz//3DXXUHM2ZH/729/UpEkTPfnkk2rcuLHatWunRx55RH5%2Bfq5tznU8SfY9pn7sp/8XJ02aVOd4u/HGGzVkyBBJUk1NjaKjoxUVFeW2TVlZmaT6/55wrq/tl%2B0xZsGWfv/731ujR4%2B2ysrKrIMHD1p9%2BvSxFixYUN/LOqfy8nKra9eu1iuvvGKdPn3aKigosIYOHWo99thjlmVZ1m9/%2B1tr9erVZ33u6tWrrdtvv9369ttvrYqKCmvGjBlWnz59rJqamksZ4WfNnDnTevrpp%2BuMf//991b79u2tDz/80KqsrLTef/99q0OHDtbRo0cty2r4uX7s6NGjVteuXa133nnHsizLio2NtbKyss667auvvmolJiZahw8ftoqLi62RI0daQ4cOvZTLrWPnzp3WnXfeaUVGRlrbtm2zLMuyioqKrE6dOllLliyxKisrrU8//dSKjo62du/ebVmWZW3bts2Kjo62du7caZ05c8Z66623rNjYWKusrMyyrIaZs6FbtGiR9dvf/tZtrGPHjlbbtm1dj3/ueLIs7zimzvZ/8ae2bNliderUyfr2228ty7KsvXv3WjfffLN1%2BvTps25fn98TzvW1/XI%2BxihYNrR//34rMjLS9QXFsizr/ffft%2B644456XNX57du3zxoyZIhVVVXlGvvoo4%2BsW2%2B91Tp9%2BrR18803u76Y/NTvfvc7a968ea7HZ86csaKjo62tW7de9HV7YuDAgdaSJUvqjL/44ovWQw895DY2ZMgQ65VXXrEsq%2BHnqlVTU2MNGjTImjBhgmVZlnXw4EHrhhtusIqLi8%2B6fffu3a333nvP9Tg/P99q27atdfDgwUuy3p9avXq1dccdd1jvv/%2B%2B2ze15cuXW3fddZfbtpMmTbLGjRtnWZZljR492po4caLb/G9%2B8xtr5cqVlmU1vJx28Nprr1kPPPCAZVk//L968cUXrY4dO1o33HCDa5ufO54sy/7H1M/9X/yxEydOWLGxsda7777rGlu5cqWVlJR01n3W9/eEc31tv5yPMS4R2lBOTo5CQkLUsmVL11ibNm10%2BPBhnTp1qh5Xdm6//vWv9cYbb8jX19c1tn79et18883Kzs5WVVWVZs%2Berbi4OPXq1Uuvv/66ampqJEm5ubmKjIx0Pc/Pz0%2BtWrVSdnb2Jc/xUzU1NdqzZ48%2B/vhj9ejRQ927d9ezzz6rkydP1lm3JIWHh7vW3ZBz/di7776r3NxcPf3005Kkr776SoGBgRo1apQ6d%2B6svn37auXKlZKk4uJiHT161C1X8%2BbN1axZM%2B3du7de1t%2BtWzd9%2BOGHuvvuu93Gc3JyLujz8%2BP5hpjTDgICAlReXq6SkhKlpaVpzZo1SktLU1BQkKRzH0/SuT8nZ5tvaMfUz/1f/LGMjAy1a9dO/fr1c4199dVXOn36tAYMGKDOnTtr4MCB%2BuyzzyTV//eEc31tv5yPMQqWDZWWlsrpdLqN1T6uvR7f0FmWpZdeekkbN27UhAkTVFxcrE6dOmnQoEHatGmTnn/%2BeS1evFgLFy6UdPbM/v7%2BDSJvQUGBbrrpJvXq1Utr167V22%2B/rf3792vs2LHnXXdDzlWrpqZG8%2BbN0%2BOPP%2B76JnjmzBlFRUVp1KhR2rJli55%2B%2Bmmlp6dr3bp1Ki0tlfTDN9If8/f3d81dai1atJDD4agz/p98fhpiTjuIiIjQ999/r/79%2B6ukpEQrV65UeXm5IiIiJJ37eJL%2Bs89ZQ/Bz/xdrHTp0SO%2B9955Gjx7tNu7v768OHTpo7ty5%2BvjjjxUfH68hQ4bo0KFDDep7wk%2B/tl/Ox9jPf5bRYNX%2BBPhjtY8DAwPrY0kXpKSkROPHj9eePXu0ZMkStW3bVm3btlXXrl1d23To0EEPPvig1q5dq0ceeUROp1MVFRVu%2B6moqGgQeZs3b66lS5e6HjudTo0dO1b33XefYmNjz7nuhpyrVlZWlvLy8pScnOwa69%2B/v/r37%2B963K1bN/Xv31/r1q1TXFycJNX5P9rQckk/fPyLi4vdxjz5/ISGhrq%2BKdghZ0Ny4403qqysTL/5zW80adIkHThwQIsXL9aoUaMknft4KikpOe8xY4dj6lxWrVrlusH9x2rPHtcaMmSIVq9erU2bNqlly5YN4nvC2b62X87HGGewbCgiIkJFRUU6fvy4a2zfvn266qqr1LRp03pc2fkdPHhQAwYMcP3k2rZtW0nSRx99pLfffttt2zNnzsjf31/SD5lzcnJcc5WVldq/f3%2BdU8v1ITs7WxkZGbIsyzV25swZ%2Bfj4qEOHDm7rln44JV7703pDzlVr/fr1SkhIcPspcuXKlVq3bp3bdmfOnFGTJk3UrFkztWzZUrm5ua65/Px8FRUVNahckhQZGXlBn58fz9spZ0Py7rvvSpLeeecdxcXF6f7779fJkyc1bdo0Sec%2Bnho3bnzBn7OGeEydy4YNG3TPPffUGX/ppZf0z3/%2B022s9phrCN8Tfu5r%2B2V9jNXvLWD4pe6//35r1KhRVnFxsesdI7Nnz67vZZ1TUVGRdccdd1hPP/20VV1d7Ta3YcMGq0OHDtann35q1dTUWJ999pkVGxvresfa8uXLrdtvv9365ptvXO8MSkhIsM6cOVMfUdwcOXLEioqKsl5//XWrsrLS%2Bve//23dd9991jPPPGPl5uZa7du3t95//33XO57at29vfffdd5ZlNexctfr27WstX77cbeytt96yunTpYu3Zs8eqrq62Nm7caHXo0MHasWOHZVmW9dJLL1l9%2B/a1Dh486Hrnz%2B9///v6WH4dP76xuKCgwOrYsaP11ltvWWfOnLG2bt3qdkN07Tuetm7d6nqH02233WYVFhZaltWwc9rVuY4ny7K84piq9dOb3AsKCqzIyEhr//79dbZ9/PHHrQceeMDKy8uzTp8%2Bbb366qtW586dXf8X6/N7wrm%2Btl/OxxgFy6by8/OtESNGWJ06dbI6d%2B5szZw50%2B0dHA3RwoULrcjISOuWW26xoqKi3P5ZlmX97//%2Br3XXXXdZt9xyi9WzZ0%2B3dxHV1NRYb775phUfH29FRUVZgwYNcn1BbQiysrKslJQUKzo62urcubM1bdo0q6KiwrIsy9q8ebPVr18/KyoqyurTp4/18ccfu57X0HNZlmVFRUW5rdmyflj3a6%2B9ZvXo0cPq0KGD1adPH2vdunWu%2BTNnzljPP/%2B8dfvtt1u33nqrNWzYMOv48eOXeuln9dNval9%2B%2BaXrc9ezZ09r1apVbtu/8847Vq9evayoqCgrOTnZ%2BuKLL1xzDTmnnZ3reLIs%2Bx9Ttc72fzEyMtIqLy%2Bvs21hYaH19NNPW126dHHl%2Buabb1zz9fk94Xxf2y/XY6yRZf3oPCwAAAD%2BY9yDBQAAYBgFCwAAwDAKFgAAgGEULAAAAMMoWAAAAIZRsAAAAAyjYAEAABhGwQIAADCMggUAAGAYBQsAAMAwChYAAIBhFCwAAADDKFgAAACGUbAAAAAMo2ABAAAY9v8AX%2B4TM3IxHb0AAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-6247364184281289719">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.0</td>
        <td class="number">17011</td>
        <td class="number">78.8%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2014.0</td>
        <td class="number">73</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2003.0</td>
        <td class="number">31</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2013.0</td>
        <td class="number">31</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2007.0</td>
        <td class="number">30</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2005.0</td>
        <td class="number">29</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2000.0</td>
        <td class="number">29</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1990.0</td>
        <td class="number">22</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2004.0</td>
        <td class="number">22</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2009.0</td>
        <td class="number">21</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (60)</td>
        <td class="number">456</td>
        <td class="number">2.1%</td>
        <td>
            <div class="bar" style="width:3%">&nbsp;</div>
        </td>
</tr><tr class="missing">
        <td class="fillremaining">(Missing)</td>
        <td class="number">3842</td>
        <td class="number">17.8%</td>
        <td>
            <div class="bar" style="width:23%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-6247364184281289719">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">0.0</td>
        <td class="number">17011</td>
        <td class="number">78.8%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1934.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1940.0</td>
        <td class="number">2</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1944.0</td>
        <td class="number">1</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">1945.0</td>
        <td class="number">3</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:1%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">2011.0</td>
        <td class="number">9</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:13%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2012.0</td>
        <td class="number">8</td>
        <td class="number">0.0%</td>
        <td>
            <div class="bar" style="width:11%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2013.0</td>
        <td class="number">31</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:43%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2014.0</td>
        <td class="number">73</td>
        <td class="number">0.3%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">2015.0</td>
        <td class="number">14</td>
        <td class="number">0.1%</td>
        <td>
            <div class="bar" style="width:19%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div><div class="row variablerow">
    <div class="col-md-3 namecol">
        <p class="h4 pp-anchor" id="pp_var_zipcode">zipcode<br/>
            <small>Numeric</small>
        </p>
    </div><div class="col-md-6">
    <div class="row">
        <div class="col-sm-6">
            <table class="stats ">
                <tr>
                    <th>Distinct count</th>
                    <td>70</td>
                </tr>
                <tr>
                    <th>Unique (%)</th>
                    <td>0.3%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Missing (n)</th>
                    <td>0</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (%)</th>
                    <td>0.0%</td>
                </tr>
                <tr class="ignore">
                    <th>Infinite (n)</th>
                    <td>0</td>
                </tr>
            </table>

        </div>
        <div class="col-sm-6">
            <table class="stats ">

                <tr>
                    <th>Mean</th>
                    <td>98078</td>
                </tr>
                <tr>
                    <th>Minimum</th>
                    <td>98001</td>
                </tr>
                <tr>
                    <th>Maximum</th>
                    <td>98199</td>
                </tr>
                <tr class="ignore">
                    <th>Zeros (%)</th>
                    <td>0.0%</td>
                </tr>
            </table>
        </div>
    </div>
</div>
<div class="col-md-3 collapse in" id="minihistogram-4118645445111044905">
    <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAABLCAYAAAA1fMjoAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAATZJREFUeJzt3MttwzAUAMEocEkuwj35nJ5ShHtiGggWUhBBtDRzF8CDF4%2B0PssYY3wAv/o8egEws9vRC/gP9%2Bf35mteX48dVsLZmCAQTjFBzsQ0nIsJAkEgEAQCQSAQBAJBIBAEAkEgEC57o9ANOdYwQSAIBIJAIAgEwnSH9L8cnmEvJggEgUAQCASBQBAIBIFAEAgEgUAQCASBQBAIhOmexWK7rc%2BvefFrPYFs4Id4PbZYEAQCQSAQBAJBIBD8i3VBvgm2nkBY5apR2WJBEAgEWyx2c4ZtmQkCQSAQBALBGWRHPqP6/kwQCAKBIBAIyxhjHL0ImJUJAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAkEgEAQCQSAQBAJBIBAEAuEHE1Ejb5/sx7sAAAAASUVORK5CYII%3D">

</div>
<div class="col-md-12 text-right">
    <a role="button" data-toggle="collapse" data-target="#descriptives-4118645445111044905,#minihistogram-4118645445111044905"
       aria-expanded="false" aria-controls="collapseExample">
        Toggle details
    </a>
</div>
<div class="row collapse col-md-12" id="descriptives-4118645445111044905">
    <ul class="nav nav-tabs" role="tablist">
        <li role="presentation" class="active"><a href="#quantiles-4118645445111044905"
                                                  aria-controls="quantiles-4118645445111044905" role="tab"
                                                  data-toggle="tab">Statistics</a></li>
        <li role="presentation"><a href="#histogram-4118645445111044905" aria-controls="histogram-4118645445111044905"
                                   role="tab" data-toggle="tab">Histogram</a></li>
        <li role="presentation"><a href="#common-4118645445111044905" aria-controls="common-4118645445111044905"
                                   role="tab" data-toggle="tab">Common Values</a></li>
        <li role="presentation"><a href="#extreme-4118645445111044905" aria-controls="extreme-4118645445111044905"
                                   role="tab" data-toggle="tab">Extreme Values</a></li>

    </ul>

    <div class="tab-content">
        <div role="tabpanel" class="tab-pane active row" id="quantiles-4118645445111044905">
            <div class="col-md-4 col-md-offset-1">
                <p class="h4">Quantile statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Minimum</th>
                        <td>98001</td>
                    </tr>
                    <tr>
                        <th>5-th percentile</th>
                        <td>98004</td>
                    </tr>
                    <tr>
                        <th>Q1</th>
                        <td>98033</td>
                    </tr>
                    <tr>
                        <th>Median</th>
                        <td>98065</td>
                    </tr>
                    <tr>
                        <th>Q3</th>
                        <td>98118</td>
                    </tr>
                    <tr>
                        <th>95-th percentile</th>
                        <td>98177</td>
                    </tr>
                    <tr>
                        <th>Maximum</th>
                        <td>98199</td>
                    </tr>
                    <tr>
                        <th>Range</th>
                        <td>198</td>
                    </tr>
                    <tr>
                        <th>Interquartile range</th>
                        <td>85</td>
                    </tr>
                </table>
            </div>
            <div class="col-md-4 col-md-offset-2">
                <p class="h4">Descriptive statistics</p>
                <table class="stats indent">
                    <tr>
                        <th>Standard deviation</th>
                        <td>53.513</td>
                    </tr>
                    <tr>
                        <th>Coef of variation</th>
                        <td>0.00054562</td>
                    </tr>
                    <tr>
                        <th>Kurtosis</th>
                        <td>-0.854</td>
                    </tr>
                    <tr>
                        <th>Mean</th>
                        <td>98078</td>
                    </tr>
                    <tr>
                        <th>MAD</th>
                        <td>46.73</td>
                    </tr>
                    <tr class="">
                        <th>Skewness</th>
                        <td>0.40532</td>
                    </tr>
                    <tr>
                        <th>Sum</th>
                        <td>2118189526</td>
                    </tr>
                    <tr>
                        <th>Variance</th>
                        <td>2863.6</td>
                    </tr>
                    <tr>
                        <th>Memory size</th>
                        <td>168.8 KiB</td>
                    </tr>
                </table>
            </div>
        </div>
        <div role="tabpanel" class="tab-pane col-md-8 col-md-offset-2" id="histogram-4118645445111044905">
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAlgAAAGQCAYAAAByNR6YAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD%2BnaQAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvOIA7rQAAIABJREFUeJzt3Xl0FGW%2B//FPkk7IwpIgyzjoXLgkkSsuiURC2AaDAZHlYoDJRYy4jMwAinhFEWGEESJwRUVAuciIDMjRsKqMqBxHjzDILqLjsAVFQZSEEDBpErI9vz/80ZeeOJLA09Xd4f06Jwetp7r6%2B03V0/mkqroTYowxAgAAgDWh/i4AAACgviFgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLXP4u4FJRUFBsbVuhoSFq2jRGJ064VV1trG03ENBbcKK34ERvwYne6qZ580ZWtlNXnMEKQqGhIQoJCVFoaIi/S7GO3oITvQUnegtO9BYcCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYJnL3wXg0tFn9iZ/l1An74zt4u8SAABBijNYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMv4oFEAOI%2BUie/6u4Q64UNyAf/jDBYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALAv6gFVVVaXs7Gw99thjnmUfffSR%2Bvfvr6SkJPXp00cffvih12MWLlyo7t27KykpSdnZ2fryyy89Y6dPn9aECROUmpqqDh066NFHH5Xb7XasHwAAEPyCPmDNmzdPO3bs8Pz/oUOH9MADD%2BjBBx/Ujh079MADD2js2LE6duyYJGnNmjVaunSpXn75ZW3dulXt27fXmDFjZIyRJE2dOlXfffed3nvvPa1fv17fffedZs2a5ZfeAABAcArqgLV582atX79evXr18ixbs2aNUlJSdPPNN8vlcunWW2/VjTfeqNzcXEnS8uXLdfvttyshIUENGjTQww8/rKNHj2rr1q0qLS3V2rVrNWbMGMXGxuqyyy7TuHHjtHr1apWWlvqrTQAAEGSCNmAVFhZq4sSJeuaZZxQVFeVZnpeXp8TERK914%2BPjtXfv3p8cDw8PV%2BvWrbV37159/fXXqqio8Bpv27atysrKdOjQId82BAAA6g2Xvwu4ENXV1XrkkUd09913q127dl5jbrfbK3BJUmRkpE6fPn3e8ZKSEklSdHS0Z%2BzsunW5Dys/P18FBQVey1yuaLVo0aLW2/g5YWGhXv/CN1wuu9/f%2BrzfLoXegkltj91LYb/RW3CpT70FZcBasGCBIiIilJ2dXWMsKipKZWVlXsvKysoUExNz3vGzwaq0tNSz/tlLgw0bNqx1fbm5uZo3b57XstGjR2vMmDG13kZtNG4cdf6VcMHi4mJ8st36vN/qc2/BpK7Hbn3eb/QWnOpDb0EZsN58803l5%2BcrJSVFkjyB6f3339ewYcP0xRdfeK2fl5ena665RpKUkJCgAwcO6KabbpIkVVRU6NChQ0pMTFSbNm0UHh6uvLw8XX/99ZKkgwcPei4j1lZWVpbS09O9lrlc0SoqsvNuxLCwUDVuHKUffihVVVW1lW2iJlv766z6vN/qe2/BprbHbn3fb/QWfHzRm69%2BWT6foAxY7777rtf/n/2IhhkzZujgwYN65ZVXtG7dOvXq1Uvr16/Xtm3bNHHiREnSoEGDNHfuXHXv3l1t2rTRc889p2bNmiklJUXh4eHq06ePZs2apeeff16SNGvWLPXr10%2BRkZG1rq9FixY1LgcWFBSrstLuRKiqqra%2BTfwfX31v6/N%2Bq8%2B9BZO67oP6vN/oLTjVh96CMmD9nLZt2%2BqFF17QrFmzNHHiRLVq1Upz585VmzZtJEmDBw9WcXGxRo8erRMnTujaa6/VggULFB4eLkmaPHmyZs6cqf79%2B6uiokI9e/bUH/7wB3%2B2BAAAgkyIOfsBUPCpgoJia9tyuUIVFxejoiJ3UCX8PrM3%2BbuEOnlnbBer2wvW/VYb9b23jFkb/V1GndT22K3v%2B43ego8vemvevJGV7dRV8N1cAAAAEOAIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAy1z%2BLgAIVH1mb/J3CXXyztgu/i4BAPD/cQYLAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlQRuwNm/erCFDhuiGG25Qly5dNHXqVJWVlUmSdu/erSFDhig5OVnp6elasWKF12PXrFmjjIwMJSUlKTMzU7t27fKMVVVVaebMmercubOSk5M1cuRI5efnO9obAAAIbkEZsE6cOKHf/e53Gjp0qHbs2KE1a9Zo27Zteumll3Tq1CmNGDFCAwcO1Pbt25WTk6Pp06frs88%2BkyRt3bpVU6dO1YwZM7R9%2B3YNGDBAI0eOVGlpqSRp/vz52rRpk1atWqWNGzcqMjJSkyZN8me7AAAgyARlwGratKk%2B/vhjZWZmKiQkRCdPntSZM2fUtGlTrV%2B/XrGxsRo2bJhcLpfS0tLUv39/LVu2TJK0YsUK9e3bVx06dFB4eLjuuusuxcXFad26dZ7x%2B%2B67T5dffrkaNmyoiRMnasOGDTp8%2BLA/WwYAAEHE5e8CLlTDhg0lSb/%2B9a917NgxpaSkKDMzU7Nnz1ZiYqLXuvHx8Vq5cqUkKS8vT4MGDaoxvnfvXhUXF%2Bv777/3enyzZs3UpEkT7du3T1deeWWtasvPz1dBQYHXMpcrWi1atKhznz8lLCzU619Aklwu/x0P9fmYDMaeanssXAr7jd6CS33qLWgD1lnr16/XqVOnNG7cOI0ZM0YtW7ZUVFSU1zqRkZE6ffq0JMntdv/LcbfbLUmKjo6uMX52rDZyc3M1b948r2WjR4/WmDFjar2N2mjcOOr8K%2BGSERcX4%2B8SOCYDRF2Phfq83%2BgtONWH3oI%2BYEVGRioyMlKPPPKIhgwZouzsbBUXF3utU1ZWppiYH19woqKiPDfDnzseFxfnCV5n78f6qcfXRlZWltLT072WuVzRKiqqfUj7OWFhoWrcOEo//FCqqqpqK9tE8LN1fF2I%2BnxMBuNv0rU9Fur7fqO34OOL3vz1y2dQBqxPPvlEjz/%2BuN566y1FRERIksrLyxUeHq74%2BHht2rTJa/28vDwlJCRIkhISEnTgwIEa4927d1eTJk3UsmVL5eXleS4TFhQU6OTJkzUuO/6cFi1a1LgcWFBQrMpKuxOhqqra%2BjYRvALhWOCYDAx13Qf1eb/RW3CqD70F369mkq666iqVlZXpmWeeUXl5ub799lvNnDlTgwcPVu/evXX8%2BHEtXrxYFRUV2rJli9auXeu572rw4MFau3attmzZooqKCi1evFiFhYXKyMiQJGVmZmr%2B/Pk6fPiwSkpK9NRTT6ljx4761a9%2B5c%2BWAQBAEHH8DFZVVZXCwsIuahsxMTH605/%2BpKeeekpdunRRo0aN1L9/f40ePVoRERFatGiRcnJyNGfOHDVt2lSTJk1Sp06dJElpaWmaPHmypkyZomPHjik%2BPl4LFy5UbGyspB/vlaqsrNSwYcPkdruVmpqq2bNnX3TfAADg0hFijDFOPmGXLl30n//5n8rMzFR8fLyTT%2B1XBQXF51%2BpllyuUMXFxaioyB1Up1D7zN50/pVwwd4Z28Vvzx2sx2RtuFyhypi10d9l1Eltj4X6vt/oLfj4orfmzRtZ2U5dOX6J8P7779cnn3yifv36aciQIXr99ddr3JQOAAAQzBwPWEOHDtXrr7%2Bud999V507d9bChQvVtWtXPfzww/r444%2BdLgcAAMA6v93k3rp1az300EN69913NXr0aP31r3/Vvffeq/T0dL3yyiuqqqryV2kAAAAXxW8f07B792698cYbWrduncrLy5WRkaHMzEwdO3ZMzz//vD7//HM9%2B%2Byz/ioPAADggjkesF588UW9%2Beab%2Bvrrr3XttdfqoYceUr9%2B/Tx/%2BkaSwsLC9MQTTzhdWlDixnEAAAKP4wHr1Vdf1YABAzR48OB/%2BS7Ctm3baty4cQ5XBgAAYIfjAWvDhg0qKSnRyZMnPcvWrVuntLQ0xcXFSZKuvvpqXX311U6XBgAAYIXjN7n/4x//UO/evZWbm%2BtZ9vTTT6t///7av3%2B/0%2BUAAABY53jA%2Bp//%2BR/16tVLDz30kGfZ%2B%2B%2B/r%2B7du2vGjBlOlwMAAGCd4wHriy%2B%2B0IgRIzx/pFn68ab2ESNG6NNPP3W6HAAAAOscD1gNGzbUN998U2P5999/r8jISKfLAQAAsM7xgNW7d29NmTJFH3/8sUpKSuR2u7VlyxY9%2BeSTysjIcLocAAAA6xx/F%2BHDDz%2Bsw4cP65577lFISIhneUZGhh599FGnywEAALDO8YAVFRWlBQsW6KuvvtK%2BffsUHh6utm3bqnXr1k6XAgAA4BN%2B%2B1M5bdq0UZs2bfz19AAAAD7jeMD66quv9OSTT2rnzp2qqKioMb5nzx6nSwIAALDK8YA1ZcoUHT16VOPGjVOjRo2cfnoAAACfczxg7dq1S3/%2B85%2BVnJzs9FMDAAA4wvGPaYiLi1NMTIzTTwsAAOAYxwNWdna2nn32WRUXFzv91AAAAI5w/BLhRx99pE8//VSpqam67LLLvP5kjiT99a9/dbokAAAAqxwPWKmpqUpNTXX6aQEAABzjeMC6//77nX5KAAAARzl%2BD5Yk7d27VxMmTNB//dd/6dixY1q2bJm2bt3qj1IAAACsczxg/f3vf9eQIUN05MgR/f3vf1d5ebn27Nmje%2B65Rx9%2B%2BKHT5QAAAFjneMCaNWuW7rnnHi1dulTh4eGSpGnTpunOO%2B/UvHnznC4HAADAOr%2BcwRo4cGCN5UOHDtWXX37pdDkAAADWOR6wwsPDVVJSUmP50aNHFRUV5XQ5AAAA1jkesG6%2B%2BWY988wzKioq8iw7ePCgcnJy1KNHD6fLAQAAsM7xgDV%2B/HiVlZWpc%2BfOKi0tVWZmpvr16yeXy6VHH33U6XIAAACsc/xzsBo2bKjXX39dmzdv1j/%2B8Q9VV1crMTFR3bp1U2ioXz41AgAAwCrHA9ZZaWlpSktL89fTAwAA%2BIzjASs9PV0hISH/cpy/RQgAAIKd4wHrtttu8wpYFRUV%2Bvrrr7VhwwaNHTvW6XIAAACsczxgPfDAAz%2B5/NVXX9XOnTt15513OlwRAACAXQFzV/lNN92kjz76yN9lAAAAXLSACVjbtm1TgwYN/F0GAADARXP8EuE/XwI0xqikpET79u3j8iAAAKgXHA9Yv/zlL2u8izA8PFzDhw9X//79nS4HAADAOscD1owZM5x%2BSgAAAEc5HrC2b99e63VvvPFGH1YCAADgG44HrLvuukvGGM/XWWcvG55dFhISoj179jhdHgAAwEVzPGDNnTtX06dP1/jx49WpUyeFh4dr9%2B7dmjJlim6//XbddNNNTpcEAABgleMf0zBz5kxNnjxZN998sxo2bKgGDRqoY8eOevLJJ7Vo0SK1atXK8wUAABCMHA9Y%2Bfn5uvzyy2ssb9iwoYqKipwuBwAAwDrHA1ZSUpKeffZZlZSUeJadPHlSTz/9tNLS0pwuBwAAwDrH78GaNGmShg8fru7du6t169aSpK%2B%2B%2BkrNmzfXkiVLnC4HAADAOscDVtu2bbVu3TqtXbtWBw8elCTdfvvt6tu3r6KiopwuBwAAwDrHA5YkNW7cWEOGDNGRI0d05ZVXSvrx09wBAADqA8fvwTLGaNasWbrxxhvVr18/ff/99xo/frwmTJigiooKp8sBAACwzvGAtXTpUr355puaPHmyIiIiJEk333yzPvjgAz3//PNOlwMAAGCd4wErNzdXTzzxhDIzMz2f3n7rrbcqJydHb7/9ttPlAAAAWOd4wDpy5Ij%2B4z/%2Bo8byq666SsePH3e6HAAAAOscD1itWrXSZ599VmP5Rx995LnhHQAAIJg5/i7Ce%2B%2B9V3/84x917NgxGWO0efNmvf7661q6dKkmTJjgdDkAAADWOR6wBg0apMrKSs2fP19lZWV64okndNlll%2Bmhhx7S0KFDnS4HAADAOscD1ltvvaVbbrlFWVlZOnHihIwxuuyyy5wuAwAAwGccvwdr2rRpnpvZmzZtSrgCAAD1juMBq3Xr1tq3b99Fb2fv3r26%2B%2B671bFjR3Xp0kWPPvqoTpw4IUnavXu3hgwZouTkZKWnp2vFihVej12zZo0yMjKUlJSkzMxM7dq1yzNWVVWlmTNnqnPnzkpOTtbIkSOVn59/0fUCAIBLh%2BMBKyEhQePGjVNmZqb%2B%2B7//WxMmTPD6qo2ysjL99re/VXJysv72t7/pL3/5i06ePKnHH39cp06d0ogRIzRw4EBt375dOTk5mj59uuedi1u3btXUqVM1Y8YMbd%2B%2BXQMGDNDIkSNVWloqSZo/f742bdqkVatWaePGjYqMjNSkSZN89v0AAAD1j%2BMB65tvvlGHDh0UExOjgoICHTlyxOurNo4ePap27dpp9OjRioiIUFxcnLKysrR9%2B3atX79esbGxGjZsmFwul9LS0tS/f38tW7ZMkrRixQr17dtXHTp0UHh4uO666y7FxcVp3bp1nvH77rtPl19%2BuRo2bKiJEydqw4YNOnz4sM%2B%2BJwAAoH5x5Cb36dOn68EHH1R0dLSWLl160dv793//d/3pT3/yWvbee%2B%2Bpffv2OnDggBITE73G4uPjtXLlSklSXl6eBg0aVGN87969Ki4u1vfff%2B/1%2BGbNmqlJkybat28fn9MFAABqxZEzWEuWLPFcgjvr3nvvtXJvkzFGzz33nD788ENNnDhRbrdbUVFRXutERkbq9OnTkvSz4263W5IUHR1dY/zsGAAAwPk4cgbLGFNj2SeffKIzZ85c1HZLSko0YcIEffHFF3r11Vd11VVXKSoqSsXFxV7rlZWVKSYmRpIUFRWlsrKyGuNxcXGe4PXPYfDcx9dGfn6%2BCgoKvJa5XNFq0aJFrbfxc8LCQr3%2BBSTJ5fLf8VCfj8lg7Km2x8KlsN/oLbjUp94c/xwsW7755hvdd999%2BuUvf6mVK1eqadOmkqTExERt2rTJa928vDwlJCRI%2BvEm%2BwMHDtQY7969u5o0aaKWLVsqLy/Pc5mwoKBAJ0%2BerHHZ8efk5uZq3rx5XstGjx6tMWPG1LnPn9O4cdT5V8IlIy6u9r8E%2BArHZGCo67FQn/cbvQWn%2BtBbUAasU6dOafjw4erUqZNycnIUGvp/STcjI0NPP/20Fi9erGHDhmnnzp1au3atXnzxRUnS4MGDNXr0aPXp00cdOnTQsmXLVFhYqIyMDElSZmam5s%2Bfr2uvvVZxcXF66qmn1LFjR/3qV7%2BqdX1ZWVlKT0/3WuZyRauoyM5lxrCwUDVuHKUffig9/8q4ZNg6vi7EucdkVVW13%2BrwhWD8Tbq2x0J932/0Fnx80Zu/fvl0LGCFhIRY29bq1at19OhRvfPOO3r33Xe9xnbt2qVFixYpJydHc%2BbMUdOmTTVp0iR16tRJkpSWlqbJkydrypQpOnbsmOLj47Vw4ULFxsZK%2BvFMU2VlpYYNGya3263U1FTNnj27TvW1aNGixuXAgoJiVVbanQj1bWLh4tg%2Bvi5EVVV1QNRxqavrPqjP%2B43eglN96C3E/NQNUpa1a9dOt956qxo0aOBZtnbtWqWnp9e4t2n69Om%2BLscvCgqKz79SLblcoYqLi1FRkVsZszZa2y6C2ztju/jtuc89JoP9RfGfuVyhQTfPanss1Pf9Rm/Bxxe9NW/eyMp26sqRM1g33nhjjZu%2Bk5OTVVRUpKKiIidKAAAAcIwjAcvGZ18BAAAEi%2BC7exMAACDAEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZUH5twgBAP9an9mbzr9SgPDnXyAAfIkzWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlfA4WUE/w2UcAEDg4gwUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALHP5uwAAwKWrz%2BxN/i6hTt4Z28XfJSBIcAYLAADAMgIWAACAZQQsAAAAyy7pgHXixAllZGRo69atnmW7d%2B/WkCFDlJycrPT0dK1YscLrMWvWrFFGRoaSkpKUmZmpXbt2ecaqqqo0c%2BZMde7cWcnJyRo5cqTy8/Md6wcAAASGSzZg7dy5U1lZWfrmm288y06dOqURI0Zo4MCB2r59u3JycjR9%2BnR99tlnkqStW7dq6tSpmjFjhrZv364BAwZo5MiRKi0tlSTNnz9fmzZt0qpVq7Rx40ZFRkZq0qRJfukPAAD4zyUZsNasWaNx48bpoYce8lq%2Bfv16xcbGatiwYXK5XEpLS1P//v21bNkySdKKFSvUt29fdejQQeHh4brrrrsUFxendevWecbvu%2B8%2BXX755WrYsKEmTpyoDRs26PDhw473CAAA/OeS/JiGrl27qn///nK5XF4h68CBA0pMTPRaNz4%2BXitXrpQk5eXladCgQTXG9%2B7dq%2BLiYn3//fdej2/WrJmaNGmiffv26frrU33YEQDACXysBGrrkgxYzZs3/8nlbrdbUVFRXssiIyN1%2BvTp84673W5JUnR0tPLz81VQUCBJCgsL0/79%2B3XllW3VokULK/WHhYV6/QsEG5creI5d5hmCWTDNNal%2B/Xy7JAPWvxIVFaXi4mKvZWVlZYqJifGMl5WV1RiPi4vzBK/S0lK98cYbmjdvnmed559/XpWVlRozZozVehs3jjr/SkAAiouL8XcJwCUhWOdaffj5RsA6R2JiojZt8j79m5eXp4SEBElSQkKCDhw4UGO8e/fuatKkiVq2bKm8vDxlZWUpPT1dRUVFuvfee/Xiiy/qiiv%2BXUVFbit1hoWFqnHjKP3wQ6mV7QFOszUXnFAffpPGpSuY5prk/fOtqqrayjb9FTJ55ThHRkaGjh8/rsWLF6uiokJbtmzR2rVrPfddDR48WGvXrtWWLVtUUVGhxYsXq7CwUBkZGZKkzMxMzZ8/X2fOnNG//du/adWqVerYsaN69uyppk2bqbKy2srX2YPO1sEHOM3WXHDii3mGYObv%2BXMxP99sbdNfOIN1jri4OC1atEg5OTmaM2eOmjZtqkmTJqlTp06SpLS0NE2ePFlTpkzRsWPHFB8fr4ULFyo2NlaSNHr0aFVWVmrYsGFyu91KTU3V7Nmz/dkSAADwgxBjjPF3EZeCgoLi869USy5XqOLiYlRU5FbGrI3Wtgs4JZje2eRyhTLPELSCaa5J3j/fbJ19at68kZXt1BWXCAEAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZS5/FwAAAHyjz%2BxN/i6h1t4Z28XfJVjFGSwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWufxdAIBLT5/Zm/xdAgD4FGewAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwDICFgAAgGUELAAAAMsIWAAAAJYRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsC6AIWFhRo1apRSUlKUmpqqnJwcVVZW%2BrssAAAQIAhYF2Ds2LGKjo7Wxo0btXLlSm3evFmLFy/2d1kAACBAELDq6Ouvv9a2bdv0yCOPKCoqSldeeaVGjRqlZcuW%2Bbs0AAAQIAhYdXTgwAHFxsaqZcuWnmVt27bV0aNH9cMPP/ixMgAAEChc/i4g2LjdbkVFRXktO/v/p0%2BfVuPGjZWfn6%2BCggKvdVyuaLVo0cJKDWFhoV7/AgAQ7Fyu0Hr1842AVUfR0dEqLS31Wnb2/2M3vsaoAAAMa0lEQVRiYiRJubm5mjdvntc6999/vx544AErNeTn5%2BvPf/6TsrKytCPnFivbDBT5%2BfnKzc1VVlaWtUAaKOgtONFbcKK34HTuz7dg7y34I6LDEhISdPLkSR0/ftyz7ODBg/rFL36hRo0aSZKysrK0evVqr6%2BsrCxrNRQUFGjevHk1zpLVB/QWnOgtONFbcKK34MAZrDpq3bq1OnTooKeeekpPPvmkioqK9OKLL2rw4MGedVq0aBH0yRsAAFw4zmBdgDlz5qiyslI9e/bUb37zG3Xr1k2jRo3yd1kAACBAcAbrAjRr1kxz5szxdxkAACBAhU2ZMmWKv4tA3cXExKhjx46eG%2BvrE3oLTvQWnOgtONFb4Asxxhh/FwEAAFCfcA8WAACAZQQsAAAAywhYAAAAlhGwAAAALCNgAQAAWEbAAgAAsIyABQAAYBkBCwAAwLKAD1gHDx7Uvffeq5SUFPXo0UPz589XdXW1JOntt99Wnz59dMMNN6h379567bXXvB67cOFCde/eXUlJScrOztaXX37pGTt9%2BrQmTJig1NRUdejQQY8%2B%2Bqjcbrdn/KuvvtLw4cOVnJysrl276n//93%2BdaTiI%2BWpfHTlyRPfff786deqk1NRUjRo1SocPH/aMv/TSS2rfvr2Sk5M9X88991xQ9LZ79261a9fOq/Zhw4Z5jQ8ZMkTJyclKT0/XihUrrPblVK87duzw6jE5OVnXXHONrrrqKh07dkySNHnyZF1zzTVe6%2BTm5vqtl7OmTZumxx57zGtZILx%2B%2BKq3QJhvvuwvEOacL3rz1xyz1duZM2eUk5Oj7t27q0OHDhoyZIi2bNniGQ%2BEOVdnJoCVlJSYHj16mIkTJxq3222OHDli%2BvXrZ%2BbOnWv27dtnrr/%2BerNr1y5jjDE7d%2B407du3N9u3bzfGGLN69WrTrVs3s3//flNWVmamT59u%2Bvbta6qrq40xxjz22GNm%2BPDhpqioyBw/ftzccccdZsqUKcYYY8rLy02vXr3M008/bc6cOWO%2B%2BOIL07VrV7Nu3Tr/fCOCgC/31YABA8zjjz9u3G63KSkpMRMmTDD9%2BvXzPPcDDzxg5s6dG5S9LV261Nxxxx0/%2BbwnT540HTt2NK%2B%2B%2BqqpqKgwH3/8sUlOTja7d%2B8Oyl7PVVxcbG699VbzwgsveJbddtttZvXq1QHRizHGnDhxwjz88MMmMTHRjB8/3mvb/n798GVv/p5vvu7P33POl72dy4k5ZrO3adOmmczMTHP06FFTWVlpcnNzzfXXX2%2B%2B/fZbY4z/59yFCOgzWDt37lRhYaGeeOIJRUdHq1WrVho5cqRee%2B01ffXVV6qsrFR1dbWMMQoJCVFYWJgiIiIkScuXL9ftt9%2BuhIQENWjQQA8//LCOHj2qrVu3qrS0VGvXrtWYMWMUGxuryy67TOPGjdPq1atVWlqq7du3Kz8/X2PGjFFERISuvvpqZWdna9myZX7%2BjgQuX%2B2rU6dOqVmzZnrwwQcVHR2tmJgY3Xnnndq/f79OnTolSfr88891zTXXBF1v56t9/fr1io2N1bBhw%2BRyuZSWlqb%2B/fv79Dj0Za/nmjZtmlq2bKlRo0ZJksrLy7V//36r%2B/FienG73brlllvUuHFj9e7d22u7gfD64aveAmG%2B%2BbK/89XvxJzzZW/ncmKO2eztzJkzGjNmjC6//HKFhYXpN7/5jSIiIvTFF18ExJy7EC6/Pvt5VFdXKzw8XOHh4Z5lISEhOn78uJKSkpSUlKShQ4cqLCxMVVVVGj9%2BvK677jpJUl5enu677z7P48LDw9W6dWvt3btXsbGxqqioUGJiome8bdu2Kisr06FDh3TgwAG1adPGs%2BMlKT4%2BXi%2B99JIDXQcnX%2B2rTp066eWXX/Z6rvfee0%2BtWrVSkyZNVFhYqKNHj2r58uWaNGmSIiIidMstt%2BjBBx9UgwYNAr63zz//XM2aNVOvXr1UUlKijh076rHHHtMvfvELHThwwOsYlX48DleuXGmlL6d7PWvHjh1at26d3nnnHc%2ByvXv3qrKyUnPmzNHOnTvVqFEjDRo0SL/97W8VGnphvwdeTC8NGjTQ22%2B/rWbNmtW4xPT111/7/fXDV701adLE7/PNl/1J8vuc82VvZzk1x2z29uSTT3pta/PmzSouLla7du0CYs5diIA%2Bg3XDDTcoMjJSzzzzjEpLS/Xtt996Tf4rrrhCr7zyinbv3q0FCxZo7ty5%2Btvf/ibpx6QfFRXltb3IyEidPn1aJSUlkqTo6GjP2Nl13W73Tz42KipKp0%2Bf9kmf9YGv9tU/e%2B2117Ro0SJNmzZNklRQUKCUlBRlZmbqgw8%2B0MKFC7Vx40bNmDEj4HurqqpSixYt1LVrV61atUp/%2BctfFBISohEjRqiqqqpO35dA7/Vcc%2BfO1dChQ9WqVSvPsuLiYnXs2FHZ2dn66KOP9PTTT2vp0qVatGiRX3pxuVxq1qzZT243EF4/fNXbP/PHfPNlf4Ew55zYd07NMZu9nevTTz/V2LFjdf/99%2BvKK68MiDl3IQI6YDVu3FgLFy7U7t271aNHD40dO1YDBw6U9OMBFBERoc6dOys8PFw9evRQ3759PTfsRUVFqayszGt7ZWVliomJ8eyk0tJSz9jZ/27YsKGio6O9xs6Ox8TE%2BKzXYOerfXVWeXm5/vjHP2r27NlasGCBOnfuLElq166dli1bpptvvlkRERFq27atRo0apXXr1gV8b2FhYVq8eLFGjBihRo0aqWnTpvrDH/6gffv26eDBg7X6vtjm6/34zTffaNu2bcrOzvZar0uXLlqyZIk6duyo8PBwXXfddRo%2BfPhF7ceL6eXnBMLrh696O8uf882X/QXCnPP1vnNyjvmitxUrVujuu%2B/W73//e40ePVpSYMy5CxHQAau8vFyVlZVasmSJtm7dqhUrVig0NFTx8fEqLCxURUWF1/oul8tzajIhIUEHDhzwjFVUVOjQoUNKTExUmzZtFB4erry8PM/4wYMHPZc0EhISdOjQIVVWVnrG8/LylJCQ4OOOg5ev9pUknThxQtnZ2fr000%2B1cuVKr8tN27Zt04IFC2rUEhkZGfC9fffdd5o%2BfbrXO2HKy8sl/fhbc2JiotdjJd8fh77cj9KPl5tuuOEGXXHFFV7bef/99/X666/XqOVi9uPF9PJzAuH1w1e9Sf6fb2e36Yv%2BAmHO%2BXLfSc7OsX92Mb1VVVXpiSee0DPPPKMXXnhBd999t2e9QJhzF8Rvt9fXwpkzZ0xKSopZvny5qa6uNp9//rnp1q2byc3NNcuXLzfXXXed2bBhg6murjZbt241ycnJ5oMPPjDGGLN8%2BXLTrVs3s2fPHs87mjIyMkx5ebkxxphx48aZO%2B64wxQWFprCwkJzxx13eN6RUVFRYdLT082MGTNMWVmZ2bNnj%2BnatatZtWqV374Xgc5X%2B6q8vNzcdttt5p577jGlpaU1nvezzz4z7du3N2%2B99Zapqqoy%2B/fvN7169fJ650yg9lZaWmq6dOlipk6dasrKykxhYaH5/e9/b4YPH26M%2BfHdQikpKeaVV14x5eXlZvPmzSY5Odls3rzZWm9O9XrW7373O/Pss8/WeN7169eb6667znz88cemurrafPLJJyY1NdW88cYbfunlXOPHj6/xbi1/v374qrdAmG%2B%2B7C8Q5pwvj0tjnJ1jNnubOnWq%2BfWvf22OHDnyk9v295y7EAEdsIwxZtu2bea2224zSUlJpmfPnmbJkiWesSVLlphevXqZ5ORk07dvX/Pmm296xqqrq83LL79s0tPTTVJSksnOzjZffvmlZ7y4uNhMmjTJdO7c2dx4443mscceM2632zN%2B6NAhc88995gOHTqYbt26mQULFjjTcBDzxb567733TGJiorn22mtNUlKS19fZt%2B%2B%2B9957ZsCAASYpKcl069bNzJ0711RVVQV8b8YYs2fPHnPXXXeZlJQUk5KSYsaNG2eKioo845999pnJysoyycnJpmfPno68YPiqV2OM6du3r1m2bNlPPu9rr71mevXqZa6//nrTs2dP8%2Bqrr/qtl3P91A%2ByQHj98EVvgTLffNWfMYEx53zVmzHOz7F/diG9FRYWmnbt2pn27dvXOO7OrhMIc66uQowxxr/n0AAAAOqXgL4HCwAAIBgRsAAAACwjYAEAAFhGwAIAALCMgAUAAGAZAQsAAMAyAhYAAIBlBCwAAADLCFgAAACWEbAAAAAsI2ABAABYRsACAACwjIAFAABgGQELAADAMgIWAACAZf8P4OR7MkgKhD0AAAAASUVORK5CYII%3D"/>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12" id="common-4118645445111044905">
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">98103</td>
        <td class="number">602</td>
        <td class="number">2.8%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98038</td>
        <td class="number">589</td>
        <td class="number">2.7%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98115</td>
        <td class="number">583</td>
        <td class="number">2.7%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98052</td>
        <td class="number">574</td>
        <td class="number">2.7%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98117</td>
        <td class="number">553</td>
        <td class="number">2.6%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98042</td>
        <td class="number">547</td>
        <td class="number">2.5%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98034</td>
        <td class="number">545</td>
        <td class="number">2.5%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98118</td>
        <td class="number">507</td>
        <td class="number">2.3%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98023</td>
        <td class="number">499</td>
        <td class="number">2.3%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98006</td>
        <td class="number">498</td>
        <td class="number">2.3%</td>
        <td>
            <div class="bar" style="width:4%">&nbsp;</div>
        </td>
</tr><tr class="other">
        <td class="fillremaining">Other values (60)</td>
        <td class="number">16100</td>
        <td class="number">74.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
        <div role="tabpanel" class="tab-pane col-md-12"  id="extreme-4118645445111044905">
            <p class="h4">Minimum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">98001</td>
        <td class="number">361</td>
        <td class="number">1.7%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98002</td>
        <td class="number">199</td>
        <td class="number">0.9%</td>
        <td>
            <div class="bar" style="width:55%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98003</td>
        <td class="number">280</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:77%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98004</td>
        <td class="number">317</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:87%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98005</td>
        <td class="number">168</td>
        <td class="number">0.8%</td>
        <td>
            <div class="bar" style="width:47%">&nbsp;</div>
        </td>
</tr>
</table>
            <p class="h4">Maximum 5 values</p>
            
<table class="freq table table-hover">
    <thead>
    <tr>
        <td class="fillremaining">Value</td>
        <td class="number">Count</td>
        <td class="number">Frequency (%)</td>
        <td style="min-width:200px">&nbsp;</td>
    </tr>
    </thead>
    <tr class="">
        <td class="fillremaining">98177</td>
        <td class="number">255</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:80%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98178</td>
        <td class="number">262</td>
        <td class="number">1.2%</td>
        <td>
            <div class="bar" style="width:82%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98188</td>
        <td class="number">136</td>
        <td class="number">0.6%</td>
        <td>
            <div class="bar" style="width:43%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98198</td>
        <td class="number">280</td>
        <td class="number">1.3%</td>
        <td>
            <div class="bar" style="width:88%">&nbsp;</div>
        </td>
</tr><tr class="">
        <td class="fillremaining">98199</td>
        <td class="number">317</td>
        <td class="number">1.5%</td>
        <td>
            <div class="bar" style="width:100%">&nbsp;</div>
        </td>
</tr>
</table>
        </div>
    </div>
</div>
</div>
    <div class="row headerrow highlight">
        <h1>Correlations</h1>
    </div>
    
    <div class="row headerrow highlight">
        <h1>Sample</h1>
    </div>
    <div class="row variablerow">
    <div class="col-md-12" style="overflow:scroll; width: 100%%; overflow-y: hidden;">
        <table border="1" class="dataframe sample">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>10/13/2014</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>1180</td>
      <td>0.0</td>
      <td>1955</td>
      <td>0.0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>12/9/2014</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>7</td>
      <td>2170</td>
      <td>400.0</td>
      <td>1951</td>
      <td>1991.0</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2/25/2015</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>6</td>
      <td>770</td>
      <td>0.0</td>
      <td>1933</td>
      <td>NaN</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>12/9/2014</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>7</td>
      <td>1050</td>
      <td>910.0</td>
      <td>1965</td>
      <td>0.0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2/18/2015</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>8</td>
      <td>1680</td>
      <td>0.0</td>
      <td>1987</td>
      <td>0.0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
    </tr>
  </tbody>
</table>
    </div>
</div>
</div>



*After familiarizing myself with the data through initial observations, I've decided to remove the 'id' and 'yr_renovated' columns due to the ambiguos nature of these data, i.e., 'renovations' is a subjective term and 'id' is not needed for any statistical analysis using only one datasource - no joining of data necessary during this project.*


```python
df.drop(columns = ['id', 'yr_renovated'], inplace = True, axis = 1)
```

*Sought the variables with missing values while calculating the percentage of null values against the 
total dataset count*


```python
(df.isna().sum() / len(df))*100
```




    date              0.000000
    price             0.000000
    bedrooms          0.000000
    bathrooms         0.000000
    sqft_living       0.000000
    sqft_lot          0.000000
    floors            0.000000
    waterfront       11.001528
    view              0.291707
    condition         0.000000
    grade             0.000000
    sqft_above        0.000000
    sqft_basement     0.000000
    yr_built          0.000000
    zipcode           0.000000
    lat               0.000000
    long              0.000000
    sqft_living15     0.000000
    sqft_lot15        0.000000
    dtype: float64



*Decided to remove rows with null values in the 'view' column after determining there would be only .3% amount of data loss*


```python
df.dropna(how = 'any', axis = 0, subset = ['view'], inplace=True)
```

*Viewing the 'waterfront' histogram to better understand the distribution of values for future data filling*


```python
df['waterfront'].plot(kind='hist', figsize = (8,5), title = 'Waterfront Histogram')
plt.show()
```


![png](Mod_One_Project_files/Mod_One_Project_26_0.png)


*Calculating the median 'waterfront' value to ensure it matches what the histogram above indicates*


```python
df.waterfront.median()
```




    0.0



*Replaced the 'waterfront' null values with the median value for the feature, which is 0*



```python
df.waterfront = df.waterfront.fillna(value = df.waterfront.median())
```

*Confirming all null values have been removed from the dataset*


```python
df.isna().sum()
```




    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    zipcode          0
    lat              0
    long             0
    sqft_living15    0
    sqft_lot15       0
    dtype: int64



*Identifying data in the 'sqft_basement' feature that are not integers as the datatype for the column is currently recognized as a string. Based on the profiling report above, the '?' needs to be addressed as it's likely a placeholder input.*

*Below, I'm summing the count of string characters in the 'sqft_basement' variable*


```python
print(df['sqft_basement'].head())

print((df['sqft_basement'] == '?').sum())

print(df['sqft_basement'].describe())
```

    0      0.0
    1    400.0
    2      0.0
    3    910.0
    4      0.0
    Name: sqft_basement, dtype: object
    452
    count     21534
    unique      302
    top         0.0
    freq      12798
    Name: sqft_basement, dtype: object


*Developed a for loop to cycle through the 'sqft_basement' and for each float value, appended to the 'basement_vals'
list. From there, we calculated the mean value for all the values extracted from the 'sqft_basement' feature that now exists in the newly created 'basement_vals' list*


```python
basement_vals = []

for footage in df.sqft_basement:
    try:
        basement_vals.append(float(footage))
    except:
        continue

sqft_mean = np.mean(basement_vals).round(2)
sqft_mean
```




    291.36



*With the mean calculated above, I'm replacing all of the '?' placeholder values with the mean value for 'sqft_basement'*

*Following that, I confirm all '?' characters were removed usint the print sum statement*


```python
df.sqft_basement.replace(to_replace = '?', value = sqft_mean, inplace = True)

print((df['sqft_basement'] == '?').sum())
```

    0


*Casting the 'sqft_basement' variable to a float from string*


```python
df.sqft_basement.dtype
```




    dtype('O')




```python
df['sqft_basement'] = df['sqft_basement'].astype(float).astype(int)

df.sqft_basement.dtype
```




    dtype('int64')



*Once more, I'm confirming all feature variables have been assigned to the appropriate datatype*


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 21534 entries, 0 to 21596
    Data columns (total 19 columns):
    date             21534 non-null object
    price            21534 non-null float64
    bedrooms         21534 non-null int64
    bathrooms        21534 non-null float64
    sqft_living      21534 non-null int64
    sqft_lot         21534 non-null int64
    floors           21534 non-null float64
    waterfront       21534 non-null float64
    view             21534 non-null float64
    condition        21534 non-null int64
    grade            21534 non-null int64
    sqft_above       21534 non-null int64
    sqft_basement    21534 non-null int64
    yr_built         21534 non-null int64
    zipcode          21534 non-null int64
    lat              21534 non-null float64
    long             21534 non-null float64
    sqft_living15    21534 non-null int64
    sqft_lot15       21534 non-null int64
    dtypes: float64(7), int64(11), object(1)
    memory usage: 3.3+ MB


### Exploring The Data

I attempt to answer the following questions through visualizations from the dataset per LREA's request:
- How are housing prices (home values) distributed throughout King County? And what explains the concentration of expensive housing in the northeast corner of the county?
- Does the age of the house have a noticeable impact on value?
- Does King County exhibit any seasonality in terms of housing inventory turnover?

--------------------------------------------------------------------------------------------------------------

##### Question One: 

How are housing prices (home values) distributed throughout King County? And what explains the concentration of expensive housing in the northeast corner of the county?

*The graphic below functions as a heatmap using the positional data, 'long' and 'lat', to create a layout depicting the price of house with a color scale throughout King County*


```python
df.plot.hexbin(x = 'long', y = 'lat', C = 'price', gridsize = 60, cmap = 'coolwarm', figsize = (10, 7))

plt.title('Location By Sale Price', fontsize = 15)
plt.ylabel('Latitude', fontsize = 12)
plt.xlabel('Longitude', fontsize = 12)
plt.show()
```


![png](Mod_One_Project_files/Mod_One_Project_47_0.png)


*Below is an import of a map image of King County to reference against home values distributed throughout King County from the graphic above*


```python
from IPython.display import Image
Image(filename = 'King County Metro Area.png', width=400, height=400)
```




![png](Mod_One_Project_files/Mod_One_Project_49_0.png)



##### Question One - Answer:

The distribution of home values throughout King County highlight a stark concentration of higher valued homes in and around the Kirkland area. This is likely explained due to it's proximity to highly concentrated white collar work environments, close proximity to shopping, upscale eateries, et cetera. 

Further, housing prices are likely exacerbated by proximity to the popular Lake Washington. As per convention, lakeside housing prices are often correlated with higher prices based on quality water views, access to beaches, and recreation.

It's likely these are the two largest contributing factors to the concentration of higher home values. 

--------------------------------------------------------------------------------------------------------------

##### Question Two: 

Does the age of the house have a noticeable impact on value?

*Using the seaborn jointplot method to plot house age against price during the time period for which sales data was 
captured*


```python
sns.jointplot(x = 'yr_built', y = 'price' , data = df, kind = 'reg', height = 7, xlim = (1892, 2023), 
              color = 'g', ratio = 3)

plt.title('Sale Price by Year Built', fontsize = 15)
plt.ylabel('Sell Price', fontsize = 14)
plt.xlabel('Year Built', fontsize = 14)
plt.show()
```

    /anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](Mod_One_Project_files/Mod_One_Project_53_1.png)


##### Question Two - Answer:

Based on the scatter plot above, which plots the age of the house against the price, it seems there is no clear impact from age on sale price. The jointplot includes a regression line as seen on either side of the plotted points. This regression line indicates little to no correlation between sale price and the age of the house.

Further, histogram plots are included opposite to the variable axis to visualize the distribtion of housing age and price. Of note, there was a greater number of sales for newer properties while the sale price is normally distributed around the $500,000 price point with multiple higher valued properties serving as outliers.  

--------------------------------------------------------------------------------------------------------------

#### *Question Three:* 

Does King County exhibit any seasonality in terms of housing inventory turnover?

*To answer this question, I began by casting the 'date' variable to a datetime series from an object data type for graphing a distribution plot of total housing sales by month*


```python
df.date.dtype
```




    dtype('O')




```python
df['date'] = pd.to_datetime(df['date'], errors = 'coerce')

df.date.dtype
```




    dtype('<M8[ns]')



*Below I'm adding a new column to the dateframe that captures the numeric number of the month as an 'int' datatype*


```python
df["month"] = df.date.dt.month
df.describe()
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
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.153400e+04</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>2.153400e+04</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
      <td>21534.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.400577e+05</td>
      <td>3.373038</td>
      <td>2.115712</td>
      <td>2079.827854</td>
      <td>1.509060e+04</td>
      <td>1.494126</td>
      <td>0.006734</td>
      <td>0.233863</td>
      <td>3.409724</td>
      <td>7.657425</td>
      <td>1788.557537</td>
      <td>291.352419</td>
      <td>1971.002275</td>
      <td>98077.939352</td>
      <td>47.560180</td>
      <td>-122.213948</td>
      <td>1986.299944</td>
      <td>12751.079502</td>
      <td>6.575555</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.660596e+05</td>
      <td>0.926410</td>
      <td>0.768602</td>
      <td>917.446520</td>
      <td>4.138021e+04</td>
      <td>0.539806</td>
      <td>0.081783</td>
      <td>0.765686</td>
      <td>0.650654</td>
      <td>1.172643</td>
      <td>827.745641</td>
      <td>437.344155</td>
      <td>29.376044</td>
      <td>53.506639</td>
      <td>0.138528</td>
      <td>0.140735</td>
      <td>685.121001</td>
      <td>27255.483308</td>
      <td>3.113740</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.800000e+04</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>370.000000</td>
      <td>5.200000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>370.000000</td>
      <td>0.000000</td>
      <td>1900.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>651.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.220000e+05</td>
      <td>3.000000</td>
      <td>1.750000</td>
      <td>1430.000000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1190.000000</td>
      <td>0.000000</td>
      <td>1951.000000</td>
      <td>98033.000000</td>
      <td>47.471200</td>
      <td>-122.328000</td>
      <td>1490.000000</td>
      <td>5100.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1910.000000</td>
      <td>7.617000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>0.000000</td>
      <td>1975.000000</td>
      <td>98065.000000</td>
      <td>47.571900</td>
      <td>-122.230000</td>
      <td>1840.000000</td>
      <td>7620.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.068775e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2210.000000</td>
      <td>550.000000</td>
      <td>1997.000000</td>
      <td>98118.000000</td>
      <td>47.678100</td>
      <td>-122.125000</td>
      <td>2360.000000</td>
      <td>10083.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>4820.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>871200.000000</td>
      <td>12.000000</td>
    </tr>
  </tbody>
</table>
</div>



*In order to properly label the months with string values graphing, I created a list of the months in a year to later combine with the 'month' column*


```python
months = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
```

*Here I'm creating a 'month_map' dictionary with the kays as a range 1-12 and the values as the 'months' list above.*


```python
month_map = dict(zip(range(1, 13), months))
```

*Finally, I'm creating a new column, 'month_name', by mapping the 'month' column in the dataframe using the 'month_map' dictionary*


```python
df["month_name"] = df["month"].map(month_map)
df.head()
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
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>...</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>month</th>
      <th>month_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-10-13</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>...</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>10</td>
      <td>Oct</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-12-09</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>...</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>12</td>
      <td>Dec</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-02-25</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>...</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>2</td>
      <td>Feb</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-12-09</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>...</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>12</td>
      <td>Dec</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-02-18</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>...</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>2</td>
      <td>Feb</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



*Plotting the visualization which aggregates home sales by month*


```python
plt.figure(figsize = (10,6))
ax = df["month"].value_counts().sort_index().plot(kind='bar')

plt.title("Home Sales By Month", fontsize = 20)
plt.ylabel('Total Home Sales', fontsize = 15)
plt.xlabel('Month', fontsize = 15)
plt.xticks(ticks=range(12), labels=months)

plt.show()
```


![png](Mod_One_Project_files/Mod_One_Project_68_0.png)


#### *Question Three - Answer:*

From the histogram above, it's clear inventory turnover is lowest during the winter months with January having less than 1,000 closings throughout King County. The month of May had the highest turnover with roughly 2,400 closings. That's more than double housing sales from January. More broadly, the spring and summer (March - August) months saw much higher turnover than the fall and winter months (September - February). 

The obvious implaction for LREA is to adjust the necessary resources to account for lower sales volumes during the winter months, i.e., prepare for cashflow variations throughout the fiscal year, increase marketing spend during the winter months to boost sales (commisions), and offer incentives for sellers during December and January. 

Conversly, adjusting the commision rates higher during the summer months to further increase cashflow would be an astute strategy. This would increase resources on hand prior to the fall and winter months.  


### Exploring The Data - Summary

In summary, seasonality affects sales volumes which should be accounted for during LREA's resource planning process. Age of a property has no affect on the price. And generally speaking, the further north and west a property is within the county, the more expensive the house will be. Particularly along the Lake Washington lakeshore near Kirkland. 

--------------------------------------------------------------------------------------------------------------

### Modeling the Data

- Conditioning the data
- Running the OLS models
- Final model
- Model summary
- Recommendations
- Conclusion
- Areas for future work

------------------------------------------------------------------------------------------------------------------

#### *Conditioning the Data - Correlation, Collinearity*

*In preparation for modeling the data, I began by running a few correlation matrices using a seaborn heatmap. This step is necessary for identifying features that correlate with each other (colinearity) in addition to correlation with the target variable.*

*I begin by running a correlation matrix using all remaining features from the dataframe.* 


```python
corr = df.corr()
plt.figure(figsize=(17, 12))
sns.heatmap(corr, fmt='.3g', cmap = 'coolwarm', annot = True, linewidth = 2, robust = True, )
plt.show()
```


![png](Mod_One_Project_files/Mod_One_Project_73_0.png)


*After reviewing the correlation matrix, I've decided to initally remove the 'date', 'month', 'sqft_lot15', 'zipcode', 'yr_built', 'condition', 'waterfront','sqft_lot', 'month_name' features based on their poor corelation with the predicted variable, 'price'.*


```python
df.drop(columns = ['date', 'month', 'sqft_lot15', 'zipcode', 'yr_built', 
                   'condition', 'waterfront','sqft_lot', 'month_name'], axis=1, inplace=True)
```

*Confirming the desired remaining features were not removed by calling the columns by calling the df.head() method*


```python
df.head()
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
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>floors</th>
      <th>view</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221900.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1180</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1180</td>
      <td>0</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
    </tr>
    <tr>
      <th>1</th>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>2170</td>
      <td>400</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
    </tr>
    <tr>
      <th>2</th>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>770</td>
      <td>0</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
    </tr>
    <tr>
      <th>3</th>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>1050</td>
      <td>910</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
    </tr>
    <tr>
      <th>4</th>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>1680</td>
      <td>0</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
    </tr>
  </tbody>
</table>
</div>



*Once more, I'm checking for correlation and multicolinearity among the remaining variables.*


```python
corr = df.corr()
plt.figure(figsize=(17, 12))
sns.heatmap(corr, fmt='.3g', cmap = 'coolwarm', annot = True, linewidth = 4, robust = True, )
plt.show()
```


![png](Mod_One_Project_files/Mod_One_Project_79_0.png)


*Due to extreme collinearity between 'sqft_living' and 'sqft_above' + 'sqft_basement', I further dropped the 'sqft_living' column along with the 'grade' feature due to collinearity with other features*


```python
df.drop(columns = ['sqft_living', 'grade'], axis = 1, inplace = True)
```

*To address correlation between 'bathrooms' and 'bedrooms', I'm creating a new column, 'bathrooms_plus_bedrooms', by summing the 'bathrooms' and 'bedrooms' columns.*


```python
df['bathrooms_plus_bedrooms'] = df['bathrooms'] + df['bedrooms']
```

*Here, I'm setting a new features datafram by setting it equal to the orignal dataframe while dropping the target variable as well as the now unneeded 'bathrooms' and 'bedrooms' features.*


```python
df_features = df.drop(columns = ['bathrooms', 'bedrooms', 'price'])
```

*Inspecting the newly created features dataframe to confirm the desired features remain.*


```python
df_features.head()
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
      <th>floors</th>
      <th>view</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>bathrooms_plus_bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1180</td>
      <td>0</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>2170</td>
      <td>400</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>770</td>
      <td>0</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1050</td>
      <td>910</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1680</td>
      <td>0</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>5.00</td>
    </tr>
  </tbody>
</table>
</div>



*I then build a correlation matrix using the features from the 'df_features' datafram to eyeball any collinearity.*


```python
df_features_corr = df_features.corr()
df_features_corr
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
      <th>floors</th>
      <th>view</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>bathrooms_plus_bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>floors</th>
      <td>1.000000</td>
      <td>0.028436</td>
      <td>0.523367</td>
      <td>-0.242841</td>
      <td>0.049419</td>
      <td>0.125329</td>
      <td>0.279176</td>
      <td>0.372647</td>
    </tr>
    <tr>
      <th>view</th>
      <td>0.028436</td>
      <td>1.000000</td>
      <td>0.166299</td>
      <td>0.273381</td>
      <td>0.006141</td>
      <td>-0.077894</td>
      <td>0.279561</td>
      <td>0.146279</td>
    </tr>
    <tr>
      <th>sqft_above</th>
      <td>0.523367</td>
      <td>0.166299</td>
      <td>1.000000</td>
      <td>-0.052879</td>
      <td>-0.000889</td>
      <td>0.345051</td>
      <td>0.731543</td>
      <td>0.657611</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>-0.242841</td>
      <td>0.273381</td>
      <td>-0.052879</td>
      <td>1.000000</td>
      <td>0.109392</td>
      <td>-0.143052</td>
      <td>0.198715</td>
      <td>0.332148</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>0.049419</td>
      <td>0.006141</td>
      <td>-0.000889</td>
      <td>0.109392</td>
      <td>1.000000</td>
      <td>-0.135439</td>
      <td>0.048569</td>
      <td>0.006210</td>
    </tr>
    <tr>
      <th>long</th>
      <td>0.125329</td>
      <td>-0.077894</td>
      <td>0.345051</td>
      <td>-0.143052</td>
      <td>-0.135439</td>
      <td>1.000000</td>
      <td>0.336019</td>
      <td>0.200094</td>
    </tr>
    <tr>
      <th>sqft_living15</th>
      <td>0.279176</td>
      <td>0.279561</td>
      <td>0.731543</td>
      <td>0.198715</td>
      <td>0.048569</td>
      <td>0.336019</td>
      <td>1.000000</td>
      <td>0.542594</td>
    </tr>
    <tr>
      <th>bathrooms_plus_bedrooms</th>
      <td>0.372647</td>
      <td>0.146279</td>
      <td>0.657611</td>
      <td>0.332148</td>
      <td>0.006210</td>
      <td>0.200094</td>
      <td>0.542594</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



*Below I re-plot the correlation matrix using the seaborn heatmap.*


```python
plt.figure(figsize=(10, 6))
sns.heatmap(df_features_corr, fmt='.3g', cmap = 'coolwarm', annot = True, linewidth = 4, robust = True, )
plt.show()
```


![png](Mod_One_Project_files/Mod_One_Project_91_0.png)


*Lastly, I've removed the 'sqft_living' and 'sqft_living15' due to their colinearity with multiple other features. The intent here is to keep features of properties within the model that clients can influence that would increase the value of the property by a predicted amount, i.e., adding a bathroom would increase house value by x amount, et cetera*


```python
df_features.drop(columns = ['sqft_living15', 'sqft_above'], axis=1, inplace=True)
```


```python
df_features_corr = df_features.corr()
df_features_corr
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
      <th>floors</th>
      <th>view</th>
      <th>sqft_basement</th>
      <th>lat</th>
      <th>long</th>
      <th>bathrooms_plus_bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>floors</th>
      <td>1.000000</td>
      <td>0.028436</td>
      <td>-0.242841</td>
      <td>0.049419</td>
      <td>0.125329</td>
      <td>0.372647</td>
    </tr>
    <tr>
      <th>view</th>
      <td>0.028436</td>
      <td>1.000000</td>
      <td>0.273381</td>
      <td>0.006141</td>
      <td>-0.077894</td>
      <td>0.146279</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>-0.242841</td>
      <td>0.273381</td>
      <td>1.000000</td>
      <td>0.109392</td>
      <td>-0.143052</td>
      <td>0.332148</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>0.049419</td>
      <td>0.006141</td>
      <td>0.109392</td>
      <td>1.000000</td>
      <td>-0.135439</td>
      <td>0.006210</td>
    </tr>
    <tr>
      <th>long</th>
      <td>0.125329</td>
      <td>-0.077894</td>
      <td>-0.143052</td>
      <td>-0.135439</td>
      <td>1.000000</td>
      <td>0.200094</td>
    </tr>
    <tr>
      <th>bathrooms_plus_bedrooms</th>
      <td>0.372647</td>
      <td>0.146279</td>
      <td>0.332148</td>
      <td>0.006210</td>
      <td>0.200094</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



*Running the correlation heatmap one last time to ensure any instances of extreme collinearity has been solved for.*


```python
plt.figure(figsize=(10, 6))
sns.heatmap(df_features_corr, fmt='.3g', cmap = 'coolwarm', annot = True, linewidth = 4, robust = True, )
plt.show()
```


![png](Mod_One_Project_files/Mod_One_Project_96_0.png)


*Another way to assess collinearity is to run a condition against the features correlation matrix to confirm no collinearity above a defined threshold.*


```python
abs(df_features_corr) > 0.50
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
      <th>floors</th>
      <th>view</th>
      <th>sqft_basement</th>
      <th>lat</th>
      <th>long</th>
      <th>bathrooms_plus_bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>floors</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>view</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>long</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>bathrooms_plus_bedrooms</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



*Below, I run a scatter matrix of the features to asses the distribution and correlation of final list of features. I did this as a last step to get one last overview of the data before running the models.*


```python
pd.plotting.scatter_matrix(df_features, figsize = (12,12), diagonal='hist', alpha = .8, )
plt.show()
```


![png](Mod_One_Project_files/Mod_One_Project_100_0.png)


#### *Running The OLS Models*


*To begin building the regression model, I first separated out the target variable from the remaining feature variables*


```python
y = df['price']

x = df_features
```


```python
y.head()
```




    0    221900.0
    1    538000.0
    2    180000.0
    3    604000.0
    4    510000.0
    Name: price, dtype: float64




```python
x.head()
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
      <th>floors</th>
      <th>view</th>
      <th>sqft_basement</th>
      <th>lat</th>
      <th>long</th>
      <th>bathrooms_plus_bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>400</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>910</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>7.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>5.00</td>
    </tr>
  </tbody>
</table>
</div>



*Below, I'm creating a constant variable to run two models with and without the constant to compare and contrast it's effect on the R-squared value*


```python
X = sm.add_constant(x)
```

*Running the OLS model first WITH the constant.*


```python
linreg = sm.OLS(y, X).fit()
print(linreg.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.439
    Model:                            OLS   Adj. R-squared:                  0.439
    Method:                 Least Squares   F-statistic:                     2807.
    Date:                Sun, 19 May 2019   Prob (F-statistic):               0.00
    Time:                        16:56:51   Log-Likelihood:            -3.0019e+05
    No. Observations:               21534   AIC:                         6.004e+05
    Df Residuals:                   21527   BIC:                         6.005e+05
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ===========================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    const                    -2.83e+07   1.77e+06    -16.021      0.000   -3.18e+07   -2.48e+07
    floors                    1.01e+05   4149.573     24.335      0.000    9.28e+04    1.09e+05
    view                      1.47e+05   2556.575     57.481      0.000    1.42e+05    1.52e+05
    sqft_basement             117.7922      5.310     22.184      0.000     107.384     128.200
    lat                      7.518e+05   1.38e+04     54.640      0.000    7.25e+05    7.79e+05
    long                     6.187e+04    1.4e+04      4.406      0.000    3.43e+04    8.94e+04
    bathrooms_plus_bedrooms  7.751e+04   1598.582     48.486      0.000    7.44e+04    8.06e+04
    ==============================================================================
    Omnibus:                    18944.663   Durbin-Watson:                   1.972
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):          1579615.796
    Skew:                           3.852   Prob(JB):                         0.00
    Kurtosis:                      44.245   Cond. No.                     5.02e+05
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 5.02e+05. This might indicate that there are
    strong multicollinearity or other numerical problems.


*Inspecting the coefficients*


```python
linreg.params
```




    const                     -2.829937e+07
    floors                     1.009780e+05
    view                       1.469553e+05
    sqft_basement              1.177922e+02
    lat                        7.518067e+05
    long                       6.187197e+04
    bathrooms_plus_bedrooms    7.750851e+04
    dtype: float64



*Observations:* 

 - Low P-values
 - Mediocre R-squared value
 - Residuals distribution has high skew and kurtosis
 - Jarque-Bera score appears to be high

*Running the initial model WITHOUT the constant*


```python
linreg = sm.OLS(y, x).fit()
print(linreg.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.821
    Model:                            OLS   Adj. R-squared:                  0.821
    Method:                 Least Squares   F-statistic:                 1.649e+04
    Date:                Sun, 19 May 2019   Prob (F-statistic):               0.00
    Time:                        16:56:51   Log-Likelihood:            -3.0032e+05
    No. Observations:               21534   AIC:                         6.007e+05
    Df Residuals:                   21528   BIC:                         6.007e+05
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ===========================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    floors                   1.053e+05   4165.503     25.268      0.000    9.71e+04    1.13e+05
    view                     1.488e+05   2569.215     57.901      0.000    1.44e+05    1.54e+05
    sqft_basement             137.2429      5.200     26.394      0.000     127.051     147.435
    lat                      6.949e+05   1.34e+04     51.969      0.000    6.69e+05    7.21e+05
    long                     2.711e+05   5197.479     52.159      0.000    2.61e+05    2.81e+05
    bathrooms_plus_bedrooms  7.084e+04   1552.595     45.627      0.000    6.78e+04    7.39e+04
    ==============================================================================
    Omnibus:                    18949.439   Durbin-Watson:                   1.972
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):          1538311.343
    Skew:                           3.864   Prob(JB):                         0.00
    Kurtosis:                      43.679   Cond. No.                     4.05e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 4.05e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.


*Inspecting the coefficients*


```python
linreg.params
```




    floors                     105253.011409
    view                       148759.785820
    sqft_basement                 137.242927
    lat                        694932.300271
    long                       271097.367748
    bathrooms_plus_bedrooms     70840.775935
    dtype: float64



*Observations:* 

- Low P-values
- Comfortably high R-squared value
- Residuals distribution has high skew and kurtosis
- Jarque-Bera dropped slightly


#### *Final Model*

*The final model below is run without the constant, and the feature variables: floors, view, sqft_basement, lat, long, and bathrooms_plus_bedrooms.*


```python
linreg = sm.OLS(y, x).fit()
print(linreg.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  price   R-squared:                       0.821
    Model:                            OLS   Adj. R-squared:                  0.821
    Method:                 Least Squares   F-statistic:                 1.649e+04
    Date:                Sun, 19 May 2019   Prob (F-statistic):               0.00
    Time:                        16:56:52   Log-Likelihood:            -3.0032e+05
    No. Observations:               21534   AIC:                         6.007e+05
    Df Residuals:                   21528   BIC:                         6.007e+05
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ===========================================================================================
                                  coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------
    floors                   1.053e+05   4165.503     25.268      0.000    9.71e+04    1.13e+05
    view                     1.488e+05   2569.215     57.901      0.000    1.44e+05    1.54e+05
    sqft_basement             137.2429      5.200     26.394      0.000     127.051     147.435
    lat                      6.949e+05   1.34e+04     51.969      0.000    6.69e+05    7.21e+05
    long                     2.711e+05   5197.479     52.159      0.000    2.61e+05    2.81e+05
    bathrooms_plus_bedrooms  7.084e+04   1552.595     45.627      0.000    6.78e+04    7.39e+04
    ==============================================================================
    Omnibus:                    18949.439   Durbin-Watson:                   1.972
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):          1538311.343
    Skew:                           3.864   Prob(JB):                         0.00
    Kurtosis:                      43.679   Cond. No.                     4.05e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 4.05e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
linreg = sm.OLS(y, x).fit()
linreg.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.821</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.821</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.649e+04</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 19 May 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>16:58:00</td>     <th>  Log-Likelihood:    </th> <td>-3.0032e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21534</td>      <th>  AIC:               </th>  <td>6.007e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21528</td>      <th>  BIC:               </th>  <td>6.007e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>floors</th>                  <td> 1.053e+05</td> <td> 4165.503</td> <td>   25.268</td> <td> 0.000</td> <td> 9.71e+04</td> <td> 1.13e+05</td>
</tr>
<tr>
  <th>view</th>                    <td> 1.488e+05</td> <td> 2569.215</td> <td>   57.901</td> <td> 0.000</td> <td> 1.44e+05</td> <td> 1.54e+05</td>
</tr>
<tr>
  <th>sqft_basement</th>           <td>  137.2429</td> <td>    5.200</td> <td>   26.394</td> <td> 0.000</td> <td>  127.051</td> <td>  147.435</td>
</tr>
<tr>
  <th>lat</th>                     <td> 6.949e+05</td> <td> 1.34e+04</td> <td>   51.969</td> <td> 0.000</td> <td> 6.69e+05</td> <td> 7.21e+05</td>
</tr>
<tr>
  <th>long</th>                    <td> 2.711e+05</td> <td> 5197.479</td> <td>   52.159</td> <td> 0.000</td> <td> 2.61e+05</td> <td> 2.81e+05</td>
</tr>
<tr>
  <th>bathrooms_plus_bedrooms</th> <td> 7.084e+04</td> <td> 1552.595</td> <td>   45.627</td> <td> 0.000</td> <td> 6.78e+04</td> <td> 7.39e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>18949.439</td> <th>  Durbin-Watson:     </th>  <td>   1.972</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1538311.343</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.864</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>43.679</td>   <th>  Cond. No.          </th>  <td>4.05e+03</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 4.05e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
linreg.params
```




    floors                     105253.011409
    view                       148759.785820
    sqft_basement                 137.242927
    lat                        694932.300271
    long                       271097.367748
    bathrooms_plus_bedrooms     70840.775935
    dtype: float64



#### *Final Price Equation:* 

$$ \hat y = 0 + 105,253 \text{ (floors)} + 148,759 \text{ (view)} + 137 \text{ (sqft basement)} + 694,932 \text{ (latitude)} + 271,097 \text{ (longitude)} + 70,840 \text{ (bathrooms plus bedrooms)} $$


#### *Model Summary:*

- Model:

    - R-squared = .821 --> The feature variables predict 82.1% of the variance for the target
    - F-statistic = 1.649e+04

- Targets:

    - p-value = 0 --> The features predicting the target are not random
        - We have high confidence in our model!
    - coef = 
        - floors --> has a positive correlation with price
        - view --> has a positive correlation with price
        - sqft_basement --> has a positive correlation with price
        - lat --> has a positive correlation with price
        - long --> has a positive correlation with price
        - bathrooms_plus_bedrooms --> has a positive correlation with price
        
- Residuals:

    - Kurtosis = 43.679 -->  most of the residuals fall outside of 3 standard deviations of the mean
    - Skewness = 3.864 --> residuals are not symmetric
    - Jarque-Bera = 1,538,311.34 --> measure of normality appears very high
    
Write Up:

From the model above, the Prob(F-Test) indicates the features used in the model have coefficients that are not equal to 0 with high confidence. We can that features, given their positive coefficients, have a positive correlation with our target. We know this isn't random because our p-value is 0. Therefore, our features predict 82.1% of the varience of the target. Our residuals do not seem normally distributed, meaning our linear regression model is favoring 1 side of the data. 
        
    
------------------------------------------------------------------------------------------------------------------

#### *Model Validation:*

#### Train Test Split

*In order to validate the model and prove it's accuracy in predicting prices, we'll perform a train-test split using sklean's built in train_test_split feature. Below we split the data with 80% for training and 20% for testing*


```python
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
print(len(X_train), len(X_test), len(y_train), len(y_test))
```

    17227 4307 17227 4307


*Below I run the regression model with the training set*


```python
linreg = LinearRegression()
linreg.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)



*Then I calculate the prediction on the training set and on the testing set*


```python
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)
```

*Next, I calculate the residuals on the train and test sets*


```python
train_residuals = y_hat_train - y_train
test_residuals = y_hat_test - y_test
```

*Lastly, I print the MSE for the predicted values on the train and test sets.*


```python
mse_train = mean_squared_error(y_train, y_hat_train)
mse_test = mean_squared_error(y_test, y_hat_test)       

print('Train Mean Squarred Error:', mse_train)
print('Test Mean Squarred Error:', mse_test)
```

    Train Mean Squarred Error: 75447749058.17166
    Test Mean Squarred Error: 74080625496.24939


#### Cross Validation


```python
cv_5_results = cross_val_score(linreg, X, y, cv=5, scoring="neg_mean_squared_error")
cv_5_results
```




    array([-8.40320459e+10, -8.03451138e+10, -6.62322812e+10, -6.83551531e+10,
           -7.91834421e+10])



**Interpreting The Results**

The training data returned a higher MSE meaning the test data returned less accurate predictions than the training set. The delta between the test and train data seems relatively minimal thus there is no cause for concern regarding our model. 

#### *Recommendations for LREA Going Forward:*

With the model above, LREA now has the tools necessary to consult clients on what factors might increase the final sale price of their property. Going forward, LREA should prepare the necessary infrastructure to allow the model to drive an application that accepts inputs on a given property and calculates the change in predicted value. 

LREA should begin vetting and contracting with mobile application developers to build out an interface by which LREA consultants can engage in the field when consulting prospective sellers, current sellers, and those looking to increase the value of the home. The application should accept inputs, run those inputs through the final model equation, and produce an output by which clients can visualize the impact on their properties value.

Additionally, LREA should establish a robust pipeline from the producers of these data to a locally owned server or cloud environment whereby the data can be access and run through the model for continious improvement in it's accruacy and predictive power over time. The more training data, the better. 

Finally, LREA should test the applications effectivness in the field and evaluate areas of further improvement for the application. Improvements should be surfaced to contracted developers to tweak the final application. 

#### *Conclusions:*

In conclusion, property values tend to vary with respect to geographic location throughout King County. In general, housing values increase the further West and North the property is within the county.

Additionally, and not surprisingly, property values increase as the number of bedrooms, bathrooms, square footage of the basements, and the number of floors increase. These factors contribute to the overall living size of the propterty which correlates highly with price. 

Last, the number of views a property has during the sales process tends to increase the property value. The implications here translate into more viewers means higher final sales price. Thus, it would behoove sellers & LREA to do what's necessary to garner more attention to the property during the listing price of the property. 


#### *Further Work / Areas for Improvement:*


- Data:
    - Collect additional housing sales data to confirm seasonality in home sales
    - Source additional feature data to improve the models paramaters thus increasing it's predictive accuracy. Below are some suggested features:
        - Median Household Incomes
        - Crime Rates
        - School Districts
        - Zoned Central Business Districts
        - Geographic Locations of Recreational Parks
    
- Modeling:
    - Further data conditioning is needed through min/max scaling or log transformations to ensure normal distributions among the feature variables
    - Skewness and Kurtosis figures indicate a non-normal distribution of the residuals, thus, needed to correct these scores 
    - Test the model on real world examples using homes that have yet to sell - attempt to predict sale price prior to closings. Adjust model as needed
    
