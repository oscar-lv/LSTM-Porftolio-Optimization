import pandas as pd
import numpy as np


# %%

def generate_data(flag=None):
	if flag=='alternative':
		data = pd.read_csv('data/alternative_data.csv').set_index('Date')
	else:
		data = pd.read_csv('data/data.csv').set_index('Date')
	data.index = pd.to_datetime(data.index)
	return data


def build_sequences(data, predict_data, window):
	X, y, labels = [], [], []
	for i in range(data.shape[0] - window):
		X.append(data[i:(i + window)])
		y.append(predict_data.iloc[i + window])
		labels.append(data.index[i + window])
	return np.array(X), np.array(y), np.array(labels)


def generate_dataset(start_date='2011-01-01', end_date='2020-04-01', flag=None):
	data = generate_data(flag)
	X_train = data.loc[:start_date]  # ~80%
	X_test = data.loc[start_date:end_date]  # ~10%
	return X_train, X_test, data


def generate_rolling_dataset(window=50, start_date='2011-01-01', end_date='2020-04-01', flag=None):
	data = generate_data(flag)
	train = data.loc[:start_date]  # ~80%
	test = data.loc[start_date:end_date]  # ~10%

	X_train, y_train, labels_train = build_sequences(train, train, window)
	X_test, y_test, labels_test = build_sequences(test, test, window)
	return X_train, y_train, labels_train, X_test, y_test, labels_test
