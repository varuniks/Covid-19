import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ML_model import fit_model, make_forecasts, eval_forecasts
from utils import get_weekly_covid_data, convert_to_supervised, apply_scaling, perform_pod
np.set_printoptions(threshold=sys.maxsize)

in_win = 3 
out_win = 4
batch_size = 1
n_epochs = 1
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
prediction = make_forecasts(model, batch_size, X_t, in_win, out_win)
print(f"pred: {prediction.shape}")
print(f"pred: {len(prediction)}")
# rescale
for i in range(len(prediction)):
    prediction[i,:,:] = np.rint(scaler.inverse_transform(prediction[i,:,:]))
    Y_t[i,:,:] = scaler.inverse_transform(Y_t[i,:,:])

rmse, mape = eval_forecasts(Y_t, prediction, in_win, out_win)
print(rmse,mape)
