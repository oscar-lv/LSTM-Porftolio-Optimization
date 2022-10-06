#%%
from models.lstm_qc import Model
from dataset import make_dataset
from benchmarking import benchmark_book
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

model = Model(tc=0.05)
X_train, X_test, data = make_dataset.generate_dataset(end_date='2022-03-01')
X_test = X_test.iloc[50:,:]
_, _,_, X_test_rolling, y_test , labels_test= make_dataset.generate_rolling_dataset(end_date='2022-03-01')
allocation = model.train(X_train)
#%%
allocations = [] 

i = 0 
for window, label in tqdm(zip(X_test_rolling, enumerate(labels_test))):
	i, label = label
	if i % 900 == 0 and i > 0:
		print('Retraining up to', label)
		model.train(data.loc[:label])
	allocations.append(model.predict(pd.DataFrame(window)))
# 	
allocations = np.array(allocations)

res = benchmark_book.evaluate_all(X_test, allocations, X_test_rolling, tc=0.1)

benchmark_book.plot_allocations(allocations[-500:], X_test.columns)

#%% Alternative data
from models.lstm_qc import Model
from dataset import make_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from benchmarking import benchmark_book

model = Model()
X_train, X_test, data = make_dataset.generate_dataset(flag='alternative')
X_test = X_test.iloc[50:,:]
_, _,_, X_test_rolling, y_test , labels_test= make_dataset.generate_rolling_dataset(flag='alternative')
allocation = model.train(X_train)
#%%
allocations = [] 

i = 0 
for window, label in tqdm(zip(X_test_rolling, enumerate(labels_test))):
	i, label = label
	if i % 900 == 0 and i > 0:
		print('Retraining up to', label)
		model.train(data.loc[:label])
	allocations.append(model.predict(pd.DataFrame(window)))
# 	
allocations = np.array(allocations)

allocations = np.array(allocations)

#%%
res = benchmark_book.evaluate_all(X_test, allocations, X_test_rolling, alternative=True, mod=2400)

benchmark_book.plot_allocations(allocations, X_test.columns)
