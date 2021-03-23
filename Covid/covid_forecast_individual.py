import pandas as pd
import numpy as np
import sys
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from ML_model import fit_model, make_forecasts, eval_forecasts
np.set_printoptions(threshold=sys.maxsize)


# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def get_weekly_covid_data():
   
    X = pd.read_csv ('time_series_covid19_confirmed_US.csv')
    X = X.loc[X['Province_State'] == 'Virginia'] # get only virginia state counties
    # drop all unwanted columns
    X.drop(['iso2', 'iso3', 'code3', 'FIPS','Admin2', 'Province_State', 'Country_Region', 'Combined_Key','Lat', 'Long_'], axis=1, inplace=True)
    X.drop(list(X.columns)[1:40], axis=1, inplace=True) # start from march 1st 2020 
    X.drop(['UID'], axis=1, inplace=True)
    
    # get the total cases at the end of week
    num_weeks = len(list(X.columns)) // 7
    ndays_to_remove = len(list(X.columns)) % 7
    #print(f"ndays_to_remove: {ndays_to_remove}")
    days_r = list(X.columns)[-1*(ndays_to_remove):]
    #print(days_r)
    X.drop(days_r, axis=1, inplace=True)
    Y = X.copy()
    for i in range(num_weeks):
        days_l = list(Y.columns)[i*7:((i*7)+7)]
        last_day = days_l.pop(-1)
        X[last_day] = X[last_day] + X.loc[:,days_l].sum(axis=1) 
        X.drop(days_l, axis=1, inplace=True) # remove the other 6 days of data.

    # remove any region with no data or zero cases throughout 
    #z_ = (X != 0).any(axis=1) 
    #X = X.loc[z_]
    return X

def convert_to_supervised(data, in_win, out_win, split=0.4):
    print(data.shape)
    #print(data)
    num_regions = data.shape[1]
    print(f"num_regions, {num_regions}")
    t_size = data.shape[0] - in_win - out_win + 1
    print(f"t_size, {t_size}")
    in_seq = np.zeros(shape=(t_size,in_win,num_regions))
    out_seq = np.zeros(shape=(t_size,out_win,num_regions))
    for t in range(t_size):
        in_seq[t,:,:] = data[None,t:t+in_win,:]
        out_seq[t,:,:] = data[None,t+in_win:t+in_win+out_win,:]
    
    # if we are going to shuffle
    #idx = np.arange(total_size)
    #np.random.shuffle(idx)

    #in_seq = in_seq[idx,:,:]
    #out_seq = out_seq[idx,:,:]

    test_split = int(np.floor(t_size * split))
    #print(test_split)
    #print(in_seq.shape)
    #print(out_seq.shape)
    X_train =  in_seq[0:-test_split,:,:]
    Y_train =  out_seq[0:-test_split,:,:]

    X_test = in_seq[-test_split:,:,:]
    Y_test = out_seq[-test_split:,:,:]

    #print(X_train.shape)
    #print(Y_train.shape)
    #print(X_test.shape)
    #print(Y_test.shape)

    return X_train, Y_train, X_test, Y_test
        
in_win = 3 
out_win = 4
batch_size = 1
n_epochs = 200 
n_counties = 1

# get data
X_d = get_weekly_covid_data()
X_d_raw = X_d.values
    
#X_raw = X_raw[:n_counties,:]
X_d_raw = np.transpose(X_d_raw) # 53-weeks, 3200+-regions  
print(f"X_raw shape bf scaling : {X_d_raw.shape}")
# rescale values to -1, 1
#scaler = MinMaxScaler(feature_range=(-1, 1))
#scaled_values = scaler.fit_transform(X_d_raw)
#print(f"X_raw shape after scaling : {X_d_raw.shape}")

ind_rmse = np.zeros(shape=(X_d_raw.shape[1], out_win))
ind_mape = np.zeros(shape=(X_d_raw.shape[1], out_win))
for i in range(X_d_raw.shape[1]):
    print(f"******************     Processing region : {i}  ********************************")
    X_raw = X_d_raw[:,i]
    X_raw = X_raw[:,np.newaxis]

    print(f"X_raw shape bf scaling : {X_raw.shape}")
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(X_raw)
    #print(f"X_raw shape after scaling : {scaled_values.shape}")
    X_tr, Y_tr, X_t, Y_t = convert_to_supervised(scaled_values, in_win, out_win, 0.4) 

    model = fit_model(X_tr, Y_tr, in_win, out_win, n_epochs, batch_size, 'ED')
    prediction = make_forecasts(model, batch_size, X_t, in_win, out_win)

    # rescale
    for i in range(len(prediction)):
        prediction[i,:,:] = np.rint(scaler.inverse_transform(prediction[i,:,:]))
        Y_t[i,:,:] = np.rint(scaler.inverse_transform(Y_t[i,:,:]))

    ind_rmse[i,:], ind_mape[i,:] = eval_forecasts(Y_t, prediction, in_win, out_win)


np.save('rmse_each_region', ind_rmse)
np.save('mape_each_region', ind_mape)
