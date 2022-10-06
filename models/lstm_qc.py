import numpy as np

# setting the seed allows for reproducible results

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


class Model:
	def __init__(self, tc=0, variance_weight=1):
		np.random.seed(123)
		self.data = None
		self.model = None
		self.history = None
		self.tc = tc
		self.variance_weight = variance_weight
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
			# Dropout(0.1),
			Flatten(),
			Dense(outputs, activation='softmax')
		])

		def sharpe_loss(_, y_pred):
			self.predictions.append(y_pred)

			# Normalizing
			data = tf.divide(self.data, self.data[0])

			# Dot product
			portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1)

			rebalancing = tf.reduce_sum(tf.abs(tf.subtract(self.predictions[-1], y_pred)))
			portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[
																				 :-1]  # % change formula

			sharpe = (K.mean(portfolio_returns) * 255) / (K.std(portfolio_returns) * np.sqrt(255)*self.variance_weight)

			# since we want to maximize Sharpe, while gradient descent minimizes the loss,
			#   we can negate Sharpe (the min of a negated function is its max)
			return -sharpe + (self.tc * rebalancing)

		model.compile(loss=sharpe_loss, optimizer='adam', run_eagerly=False)
		return model

	def train(self, data, validation_data=None):
		'''
		Train the model using a dataframe of stock prices

		Args
			data - DataFrame of historical closing prices of various assets

		returns
			: the allocations ratios for each of the given assets for the next day
		'''

		# data with returns
		data_w_ret = np.concatenate([data.values[1:], data.pct_change().values[1:]], axis=1)

		data = data.iloc[1:]
		self.data = tf.cast(tf.constant(data), float)

		if self.model is None:
			self.model = self.__build_model(data_w_ret.shape, len(data.columns))

		fit_predict_data = data_w_ret[np.newaxis, :]
		if not self.history:
			self.history = self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=100, shuffle=False,
										  batch_size=64,

										  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)])
		else:
			self.model.fit(fit_predict_data, np.zeros((1, len(data.columns))), epochs=50, shuffle=False,
						   batch_size=64,

						   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)])
		return self.model.predict(fit_predict_data)[0]

	def predict(self, data):
		"""

		:param data: dataframe of prices
		:return:
			allocations for the next day
		"""
		# data with returns
		data_w_ret = np.concatenate([data.values[1:], data.pct_change().values[1:]], axis=1)

		data = data.iloc[1:]
		self.data = tf.cast(tf.constant(data), float)

		if self.model is None:
			self.model = self.__build_model(data_w_ret.shape, len(data.columns))

		fit_predict_data = data_w_ret[np.newaxis, :]
		return self.model.predict(fit_predict_data, verbose=0)[0]

	def plot_history(self):
		"""

		:return:
		"""
		loss = self.history.history['loss']
		epochs = len(loss)
		epochs_range = range(epochs)

		plt.figure(figsize=(20, 8))

		plt.subplot(1, 2, 2)
		plt.plot(epochs_range, loss, label='Training Loss (-ve Sharpe)')
		plt.legend(loc='upper right')
		plt.title('Training Loss')
		plt.show()
