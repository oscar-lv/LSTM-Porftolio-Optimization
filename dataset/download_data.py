import yfinance as yf
import pandas as pd
#%%
data = yf.download(['VTI','AGG','DBC','^VIX'], start_date='2006-01-01')
data = data['Adj Close']
data = data[['VTI','AGG','DBC','^VIX']]
data = data.loc['2006-01-01':].dropna()

data.to_csv('data/data.csv')

#%%
import yfinance as yf
import pandas as pd

#%%
data = yf.download(['^GSPC', '^RUT', '^DJI', '^IXIC', 'AGG','DBC','^VIX'], start_date='2006-01-01')
data = data['Adj Close']
data = data[['^GSPC', '^RUT', '^DJI', '^IXIC','AGG','DBC','^VIX']]
data = data.loc['2006-01-01':].dropna()

data.to_csv('data/alternative_data.csv')
