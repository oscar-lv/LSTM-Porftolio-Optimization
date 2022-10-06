from benchmarking import benchmark_book
from dataset import make_dataset
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

X_train, X_test, data = make_dataset.generate_dataset()
X_test = X_test.iloc[50:,:]
_, _, _, X_test_rolling, y_test , labels_test= make_dataset.generate_rolling_dataset()
allocations_model = pd.read_excel('results/allocations_30_09.xlsx', sheet_name='Allocations').values

res = benchmark_book.evaluate_all(X_test, allocations_model, X_test_rolling)

benchmark_book.plot_allocations(allocations_model, X_test.columns)
