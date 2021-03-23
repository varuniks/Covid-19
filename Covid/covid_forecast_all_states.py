import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ML_model import fit_model, make_forecasts, eval_forecasts
from utils import get_weekly_covid_data, get_weekly_covid_data_for_state, convert_to_supervised, apply_scaling, perform_pod
np.set_printoptions(threshold=sys.maxsize)

in_win = 3 
out_win = 4
batch_size = 1
n_epochs = 1000 
n_counties = 5 
state = 'Virginia' 
mets = {} 
# get data
X = pd.read_csv ('time_series_covid19_confirmed_US.csv')
for state in X['Province_State'].unique():
    X_d = get_weekly_covid_data(state)
    X_raw = X_d.values
    X_raw = np.transpose(X_raw) # 53-weeks, 3200+-regions  

    scaled_values, scaler = apply_scaling(X_raw)
    X_tr, Y_tr, X_t, Y_t = convert_to_supervised(scaled_values, in_win, out_win, 0.4) 

    model = fit_model(X_tr, Y_tr, in_win, out_win, n_epochs, batch_size, 'ED')
    prediction = make_forecasts(model, batch_size, X_t, in_win, out_win)

    # rescale
    for i in range(len(prediction)):
        prediction[i,:,:] = np.rint(scaler.inverse_transform(prediction[i,:,:]))
        Y_t[i,:,:] = scaler.inverse_transform(Y_t[i,:,:])

    rmse, mape = eval_forecasts(Y_t, prediction, in_win, out_win)
    print(rmse,mape)
    mets[state] = [rmse, mape]

g_file = open('mets_state_wise', 'wb') 
pickle.dump(mets, g_file) 
g_file.close() 

