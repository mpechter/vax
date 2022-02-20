from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
from scipy.stats.stats import pearsonr
import math

plt.style.use('ggplot')

def get_partisan_lean():

    lean_df = pd.read_csv('fivethirtyeight_partisan_lean_STATES.CSV')

    lean_df.rename(columns={"2021": "partisan_lean"}, inplace = True)

    raw_dict = lean_df.to_dict('index')

    state_array = []
    lean_array = []

    for key in raw_dict:
        lean = raw_dict[key]['partisan_lean']
        state = raw_dict[key]['state']
        lean = round(lean, 1)

        state_array.append(state)
        lean_array.append(lean)

    final_dict = {'state':state_array,'partisan_lean':lean_array}

    df_lean = pd.DataFrame.from_dict(final_dict)

    return df_lean

def get_median_age():

    median_df = pd.read_csv('median age.CSV')

    raw_dict = median_df.to_dict('index')

    state_array = []
    median_array = []

    for key in raw_dict:
        median = raw_dict[key]['median_age']
        state = raw_dict[key]['state']

        state_array.append(state)
        median_array.append(median)

    final_dict = {'state':state_array,'median_age':median_array}

    df_median = pd.DataFrame.from_dict(final_dict)

    return df_median

def get_vax_percent():

    percent_df = pd.read_csv('COVID19_CDC_Vaccination_CSV_Download.csv')

    raw_dict = percent_df.to_dict('index')

    current_percent = {}

    for key in raw_dict:
        state = raw_dict[key]['GEOGRAPHY_NAME']
        if state == 'New York State':
            state = 'New York'
        if state == 'United States':
            continue
        if state == 'District of Columbia':
            continue
        vax_percent = raw_dict[key]['FULLY_VACCINATED_PERCENT']

        if state not in current_percent:
            current_percent[state] = round((vax_percent)*100,0)
        elif state in current_percent and vax_percent > current_percent[state]:
            current_percent[state] = round((vax_percent)*100,0)

    state_array = []
    percent_array = []

    for key in current_percent:
        state_array.append(key)
        percent_array.append(current_percent[key])

    final_dict = {'state':state_array, 'vax_percent':percent_array}

    df_vax = pd.DataFrame.from_dict(final_dict)

    return df_vax

def get_population():

    df = pd.read_csv('COVID19_CDC_Vaccination_CSV_Download.csv')

    raw_dict = df.to_dict('index')

    pop_by_state = {}

    for key in raw_dict:
        state = raw_dict[key]['GEOGRAPHY_NAME']
        if state == 'New York State':
            state = 'New York'
        population = raw_dict[key]['POPULATION']

        if math.isnan(population):
            continue

        if state not in pop_by_state:
            pop_by_state[state] = round((population)*100,0)
        elif state in pop_by_state and population > pop_by_state[state]:
            pop_by_state[state] = round((population)*100,0)

    state_array = []
    pop_array = []

    for key in pop_by_state:
        state_array.append(key)
        pop_array.append(pop_by_state[key])

    final_dict = {'state':state_array, 'population':pop_array} 

    df_pop = pd.DataFrame.from_dict(final_dict)

    return df_pop

def merge_vax():

    df_lean = get_partisan_lean()
    df_vax = get_vax_percent()
    df_final = pd.merge(df_lean, df_vax, on='state')

    return df_final

def merge_pop():

    df_vax = get_vax_percent()
    df_pop = get_population()
    df_final = pd.merge(df_pop, df_vax, on='state')

    return df_final

def merge_median():

    df_vax = get_vax_percent()
    df_median = get_median_age()
    df_final = pd.merge(df_median, df_vax, on='state')

    return df_final

def plot_lean_scatter(df):

    ax = df.plot.scatter(x='partisan_lean', y='vax_percent', title='Partisan Lean and Vaccination Rates')

    plt.xlabel('Partisan Lean')
    plt.ylabel('Percent Fully Vaxxed')

    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)

    x = 'partisan_lean'

    regression(df, x)

    plt.show()

def plot_median_scatter(df):

    ax = df.plot.scatter(x='median_age', y='vax_percent', title='Median Age and Vaccination Rates')

    plt.xlabel('Median Age')
    plt.ylabel('Percent Fully Vaxxed')

    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)

    x = 'median_age'

    #regression(df, x)

    plt.show()

def plot_pop_scatter(df):

    ax = df.plot.scatter(x='population', y='vax_percent', title='Population and Vaccination Rates')

    plt.xlabel('Population')
    plt.ylabel('Percent Fully Vaxxed')

    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)

    regression(df)

    plt.show()

def regression(df, x):

    X = df.iloc[:, 1].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df.iloc[:, 2].values.reshape(-1, 1)  # -1 means you aren't sure the number of rows, but should have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    plt.plot(X, Y_pred, color='red')

    plt.show()

    partisan_list = df[x].values.tolist()
    vax_list = df['vax_percent'].values.tolist()

    print(pearsonr(partisan_list,vax_list))

df_vax = merge_vax()

plot_lean_scatter(df_vax)

df_pop = merge_pop()

plot_pop_scatter(df_pop)

df_median = merge_median()

plot_median_scatter(df_median)








