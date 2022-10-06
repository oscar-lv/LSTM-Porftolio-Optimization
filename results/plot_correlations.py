#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 17:52:30 2022

@author: oscar
"""
import itertools as it
import pandas as pd
from dataset import make_dataset
import matplotlib.pyplot as plt
import seaborn as sns

data = make_dataset.generate_data()
alternative_data = make_dataset.generate_data(flag='alternative')

def correlation_analysis(df, lookback_window):
    etfs_pairs = list(it.combinations(df.columns, 2))
    correlation = pd.DataFrame()
    for pair in etfs_pairs:
        correlation[str(pair[0])+' <--> '+str(pair[1])] = df[list(pair)].rolling(lookback_window).corr().iloc[0::2,-1].droplevel(1, axis=0)
    return correlation


#%%
lookback = 50
rolling_corrs = correlation_analysis(data.pct_change().fillna(0), lookback)[49:] #first 49 values are NaN due to lookback = 50

rolling_corrs.index = rolling_corrs.index.strftime('%Y-%m-%d')
fig, ax = plt.subplots(1,1, figsize=(28,5))
plt.title('Rolling Correlations')
sns.heatmap(rolling_corrs.transpose())
plt.xticks = None
#%%
fig.savefig('results/correlations.png', dpi=300)


#%%
rolling_corrs = correlation_analysis(alternative_data.pct_change().fillna(0), lookback)[49:] #first 49 values are NaN due to lookback = 50

rolling_corrs.index = rolling_corrs.index.strftime('%Y-%m-%d')
fig, ax = plt.subplots(1,1, figsize=(28,5))
plt.title('Rolling Correlations')
sns.heatmap(rolling_corrs.transpose())
plt.xticks = None
fig.savefig('results/correlations_alternative.png', dpi=300)
