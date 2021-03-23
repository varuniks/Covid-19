import pandas as pd
import numpy as np
import multiprocessing as mp
import sys
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ML_model import fit_model, make_forecasts, eval_forecasts
from utils import get_weekly_covid_data, convert_to_supervised, apply_scaling
from process import process_region
np.set_printoptions(threshold=sys.maxsize)

in_win = 3 
out_win = 4 
batch_size = 1
n_epochs = 250 
n_counties = 1
state = 'Virginia'

pool = mp.Pool(processes=4)
# get data
X_d = get_weekly_covid_data(state)
X_d_raw = X_d.values
X_d_raw = np.transpose(X_d_raw) # 53-weeks, 3200+-regions  

ind_rmse = np.zeros(shape=(X_d_raw.shape[1], out_win))
ind_mape = np.zeros(shape=(X_d_raw.shape[1], out_win))
#X_d_raw = X_d_raw[:,:,np.newaxis]
X_d_raw = X_d_raw[:,:,np.newaxis]
print(X_d_raw.shape)

results = [pool.apply(process_region, args=(X_d_raw[:,x,:], in_win, out_win, n_epochs, batch_size)) for x in range(X_d_raw.shape[1])]
results = np.array(results)
print(results)
print(results.shape)
np.save('ind_counties_'+state, results)
"""
for i in range(X_d_raw.shape[1]):
    print(f"******************     Processing region : {i}  ********************************")
    #X_raw = X_d_raw[:,i,:]
    #metrics = process_region(X_raw, in_win, out_win, n_epochs, batch_size)
    results = [pool.apply(process_region, args=(X_d_raw[:,x,:], in_win, out_win, n_epochs, batch_size)) for x in range(X_d_raw.shape[1])]
    #ind_rmse[i,:], ind_mape[i,:]
"""
print('done')
