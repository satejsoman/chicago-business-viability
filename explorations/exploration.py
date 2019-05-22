#==============================================================================#
# USEFUL FUNCTIONS FOR EXPLORING DATA
# CAPP 30254 - MACHINE LEARNING FOR PUBLIC POLICY
#
# Cecile Murray
#==============================================================================#

# dependencies
import numpy as np
import pandas as pd 
import seaborn as sns
import plotnine as p9
import matplotlib.pyplot as plt


#==============================================================================#
# 2. EXPLORE DATA
#==============================================================================#

def get_desc_stats(df, *cols):
    ''' compute basic descriptive stats for any number of specified columns (as string)
        if none provided, computes only for numeric type columns'''

    if cols:
        return df[df[list(cols)]].describe()
    else:
        return df.select_dtypes(include = np.number).describe()


def find_outliers(df, lb, ub, var):
    ''' Checks whether all values of variable(s) fall within reasonable bounds '''

    too_small = df[var].loc[df[var] < lb]
    too_big = df[var].loc[df[var] > ub]

    print('# of values smaller than lower bound: ', len(too_small.index))
    print(too_small.head().sort_values())
    print('# of values larger than upper bound:', len(too_big.index))
    print(too_big.head().sort_values(ascending = False))
    print('\n')

    return 


def plot_correlations(df, cols):
    '''Takes a data frame, a list of columns, and a pair of names for plot axes
        Returns a plot of pairwise correlations between all variables
    '''

    axisnames = ["Variable 1", "Variable 2", "Correlation"]

    corr_df = pd.DataFrame(df[cols].corr().stack()).reset_index()
    dict(zip(list(corr_df.columns), axisnames))
    corr_df.rename(columns = dict(zip(list(corr_df.columns), axisnames)), inplace = True)

    return (p9.ggplot(corr_df, p9.aes(axisnames[0], axisnames[1], fill=axisnames[2]))
        + p9.geom_tile(p9.aes(width=.95, height=.95)) 
        + p9.theme(
        axis_text_x = p9.element_text(rotation = 90))
        )


def plot_distr(df, *cols):
    ''' Create histograms of numeric variables in dataframe; 
        optionally specify which variables to use '''

    if not cols:
        cols = df.columns
    
    if len(cols) == 1:
        cols = [cols[0]]

    # this part is still plotting on top of everything
    for c in cols:
        if df[c].dtype == np.number:
            sns.distplot(df[c].loc[df[c].notnull()], kde = False)    
    
    return


def plot_cond_dist(df, y, *x):
    ''' Plot conditional distributiofn of x on categorical or binary y '''

    for v in x:
        sns.FacetGrid(df, col=y).map(plt.hist, v)



def tab(df, y, *x):
    ''' Compute summary statistics about y conditioned on categorical variable(s) x '''

    if len(x) == 0:
        return False
    
    else:
        return df.groupby(list(x))[y].describe()
