import numpy as np

# setting the seed allows for reproducible results
np.random.seed(123)

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K


class Model:
	def __init__(self):
		self.data = None
		self.model = None

	def __build_model(self, input_shape, outputs):
		'''
		Builds and returns the Deep Neural Network that will compute the allocation ratios
		that optimize the Sharpe Ratio of the portfolio

		inputs: input_shape - tuple of the input shape, outputs - the number of assets
		returns: a Deep Neural Network model
		'''
		model = Sequential([
			LSTM(1, input_shape=input_shape),
			Flatten(),
			Dense(outputs, activation='softmax')
		])

		def sharpe_loss(_, y_pred):
			# make all time-series start at 1
			data = tf.divide(self.data, self.data[0])

			# value of the portfolio after allocations applied
			portfolio_values = tf.reduce_sum(tf.multiply(data, y_pred), axis=1)

			portfolio_returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[
																				 :-1]  # % change formula

			sharpe = K.mean(portfolio_returns) / K.std(portfolio_returns)

			# since we want to maximize Sharpe, while gradient descent minimizes the loss,
			#   we can negate Sharpe (the min of a negated function is its max)
			return -sharpe

		model.compile(loss=sharpe_loss, optimizer='adam')
		return model

	def get_allocations(self, X, y):
		'''
		Computes and returns the allocation ratios that optimize the Sharpe over the given data

		input: data - DataFrame of historical closing prices of various assets

		return: the allocations ratios for each of the given assets
		'''

		# data with returns
		pct_change = np.diff(X, axis=1) / X[:, :-1]
		data_w_ret = np.concatenate([X[:, :-1], pct_change], axis=2)

		data = X[:, :-1]

		self.data = tf.cast(tf.constant(data), float)

		if self.model is None:
			self.model = self.__build_model((data_w_ret.shape[1],data_w_ret.shape[2]) , data.shape[2])

		fit_predict_data = data_w_ret[np.newaxis, :]
		
		self.model.fit(data_w_ret, y, epochs=1, shuffle=False, batch_size=1)
		return self.model.predict(fit_predict_data)[0]
