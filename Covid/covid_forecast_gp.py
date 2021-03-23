import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ML_model import fit_model, make_forecasts, eval_forecasts
from ML_GP import *
from utils import get_weekly_covid_data, convert_to_supervised, apply_scaling, perform_pod
np.set_printoptions(threshold=sys.maxsize)

in_win = 3 
out_win = 4
batch_size = 1
n_epochs = 1500
n_counties = 5 
state = 'Virginia'  
# get data
X_d = get_weekly_covid_data(state)
X_raw = X_d.values
#X_raw = X_raw[:n_counties,:]
#X_raw = np.transpose(X_raw) # 53-weeks, 3200+-regions  
#np.save('covid_matrix',X_raw)

#phi, X_coeffs = perform_pod(X_raw)
#print(X_coeffs.shape)
#print(var)

X_coeffs = np.transpose(X_raw)

scaled_values, scaler = apply_scaling(X_coeffs)
X_tr, Y_tr, X_t, Y_t = convert_to_supervised(scaled_values, in_win, out_win, 0.4) 

model = fit_model(X_tr, Y_tr, in_win, out_win, n_epochs, batch_size, 'ED')
P_tr = make_forecasts(model, batch_size, X_tr, in_win, out_win)
x_e = Y_tr - P_tr
print(Y_tr.shape)

P_t = make_forecasts(model, batch_size, X_t, in_win, out_win)
train_e, test_e = fit_error_model(X_tr, X_t, x_e, out_win)

P_tr_e = P_tr + train_e
P_t_e = P_t + test_e

# rescale for train 
for i in range(len(P_tr)):
    P_tr[i,:,:] = np.rint(scaler.inverse_transform(P_tr[i,:,:]))
    P_tr_e[i,:,:] = np.rint(scaler.inverse_transform(P_tr_e[i,:,:]))
    Y_tr[i,:,:] = scaler.inverse_transform(Y_tr[i,:,:])

rmse, mape = eval_forecasts(Y_tr, P_tr, in_win, out_win)
print("train error before Gp ")
print(rmse,mape)

rmse, mape = eval_forecasts(Y_tr, P_tr_e, in_win, out_win)
print("train error after Gp ")
print(rmse,mape)

# rescale for test
for i in range(len(P_t)):
    P_t[i,:,:] = np.rint(scaler.inverse_transform(P_t[i,:,:]))
    P_t_e[i,:,:] = np.rint(scaler.inverse_transform(P_t_e[i,:,:]))
    Y_t[i,:,:] = scaler.inverse_transform(Y_t[i,:,:])

rmse, mape = eval_forecasts(Y_t, P_t, in_win, out_win)
print("test error before Gp ")
print(rmse,mape)

rmse, mape = eval_forecasts(Y_t, P_t_e, in_win, out_win)
print("test error after Gp ")
print(rmse,mape)

