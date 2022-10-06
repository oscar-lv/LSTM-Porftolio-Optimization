import numpy as np
from dataset import make_dataset
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
import cvxpy as cp
from tqdm import tqdm

plt.style.use('seaborn')

dico = {
	"Equal-Weighted": np.array([0.25, 0.25, 0.25, 0.25]),
	"Stock Heavy": np.array([0.5, 0.1, 0.2, 0.2]),
	"Bond Heavy": np.array([0.1, 0.5, 0.2, 0.2]),
	"Bonds and Stocks": np.array([0.4, 0.4, 0.1, 0.1])
}

dico_alternative = {
	"Equal-Weighted": np.array([1/7 for x in range(7)]),
	"Stock Heavy": np.array([0.125, 0.125, 0.125, 0.125, 0.1, 0.2, 0.2]),
	"Bond Heavy": np.array([0.025, 0.025, 0.025, 0.025, 0.5, 0.2, 0.2]),
	"Bonds and Stocks": np.array([0.1,0.1,0.1, 0.1,  0.4, 0.1, 0.1])
}

def get_weighted_book_series(prices, weights, tc=None):
	if len(weights.shape) == 1:
		weights_matrix = np.tile(weights.T, (prices.shape[0], 1))
	else:
		weights_matrix = weights.copy()

	if tc:
		performance = (weights_matrix * prices.pct_change().add(1-(tc*0.001)).cumprod()).sum(axis=1)
		performance.iloc[0] = 1
	else:
		performance = (weights_matrix * prices).sum(axis=1)
	return performance


def mean_variance_optimization(windows, mod=2):
	def optimize(window):
		returns = pd.DataFrame(window).pct_change().dropna().values
		w = cp.Variable(returns.shape[1])
		ret = np.mean(returns, axis=0).T @ w
		gamma = 10**(-9)
		cov = np.cov(returns.T)
		risk = cp.quad_form(w, cov)
		prob = cp.Problem(cp.Maximize((ret*gamma)-risk), [cp.sum(w) == 1, w >= 0])

		prob.solve()
		return w.value

	weight_array = []
	i = 0
	for window in tqdm(windows):
		if i % mod == 0:
			optimal = optimize(window)
		weight_array.append(optimal)
		i += 1
	return np.array(weight_array)


# %%
def compute_metrics(book_series):
	metrics = {}

	returns = book_series.pct_change()

	metrics['E[r]'], metrics['Std[r]'] = returns.mean() * 252, returns.std() * np.sqrt(252)
	metrics['Sharpe'] = (metrics['E[r]']) / (metrics['Std[r]'])
	metrics['DD Vol'] = returns[returns < 0].std() * np.sqrt(252)
	metrics['Sortino'] = (metrics['E[r]'] / metrics['DD Vol'])
	metrics['% of + Ret'] = len(returns[returns > 0]) / len(returns)
	metrics['P/L ratio'] = returns[returns > 0].mean() / abs(returns[returns < 0].mean())
	return metrics


# %%

def evaluate_all(prices, model_allocations=None, rolling_data=None, plot=True, tc=None, alternative=False, mod=2):
	result = []
	use = dico if not alternative else dico_alternative
	for name, weights in use.items():
		book = get_weighted_book_series(prices, weights, tc)
		if not tc:
			book = book/book.iloc[0]
		metrics = compute_metrics(book)
		keys = list(metrics.keys())
		metrics['Strategy'] = name
		result.append(metrics)
		if plot:
			plt.plot(book, label=name)

	lstm_book = get_weighted_book_series(prices, model_allocations)
	lstm_book = lstm_book/lstm_book.iloc[0]
	metrics = compute_metrics(lstm_book)
	metrics['Strategy'] = 'LSTM'
	result.append(metrics)

	weights_mv = mean_variance_optimization(rolling_data, mod=mod)
	mv_book = get_weighted_book_series(prices, weights_mv, tc)
	if not tc:
		mv_book = mv_book / mv_book.iloc[0]
	metrics = compute_metrics(mv_book)
	metrics['Strategy'] = 'Mean-Variance'
	result.append(metrics)
	if plot:
		plt.plot(lstm_book, label='LSTM')
		plt.plot(mv_book, label='MV')
		plt.title('Comparison of strategy returns')
		plt.legend()
		plt.show()

	return pd.DataFrame(result)[['Strategy'] + keys].set_index('Strategy').round(4)


def plot_allocations(data, labels=None):
	for dat, lab in zip(data.T, labels):
		plt.plot(dat, label=lab)
	plt.title("Evolution of allocations through time")
	plt.legend()
	plt.show()