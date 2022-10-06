import numpy as np

# setting the seed allows for reproducible results
np.random.seed(123)

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


class Model:
	def __init__(self):
		self.data = None
		self.model = None
		self.history = None
		self.predictions = []

	def __build_model(self, input_shape, outputs):
		'''
		Builds and returns the Deep Neural Network that will compute the allocation ratios
		that optimize the Sharpe Ratio of the portfolio

		inputs: input_shape - tuple of the input shape, outputs - the number of assets
		returns: a Deep Neural Network model
		'''

		model = Sequential([
			LSTM(64, input_shape=input_shape),
			# LSTM(64),
			Dropout(0.1),
			Flatten(),
			Dense(outputs, activation='softmax')
		])

		def sharpe_loss(_, y_pred):
			self.predictions.append(y_pred)
			# make all time-series start at 1
			data = tf.divide(self.data, self.data[0])

			# value of the portfolio after allocations applied
			portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1)

			portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[
																				 :-1]  # % change formula

			sharpe = (K.mean(portfolio_returns) * 255) / (K.std(portfolio_returns) * np.sqrt(255))

			# since we want to maximize Sharpe, while gradient descent minimizes the loss,
			#   we can negate Sharpe (the min of a negated function is its max)
			return -sharpe

		model.compile(loss=sharpe_loss, optimizer='adam')
		return model

	def train(self, data, validation_data=None):
		'''
		Computes and returns the allocation ratios that optimize the Sharpe over the given data

		input: data - DataFrame of historical closing prices of various assets

		return: the allocations ratios for each of the given assets
		'''

		# data with returns
		data_w_ret = np.concatenate([data.values[1:], data.pct_change().values[1:]], axis=1)

		data = data.iloc[1:]
		self.data = tf.cast(tf.constant(data), float)

		if self.model is None:
			self.model = self.__build_model(data_w_ret.shape, len(data.columns))

		fit_predict_data = data_w_ret[np.newaxis, :]
		self.history = self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=20, shuffle=False, batch_size=64)
		return self.model.predict(fit_predict_data)[0]

	def predict(self, data):
		# data with returns
		data_w_ret = np.concatenate([data.values[1:], data.pct_change().values[1:]], axis=1)

		data = data.iloc[1:]
		self.data = tf.cast(tf.constant(data), float)

		if self.model is None:
			self.model = self.__build_model(data_w_ret.shape, len(data.columns))

		fit_predict_data = data_w_ret[np.newaxis, :]
		return self.model.predict(fit_predict_data, verbose=0)[0]
