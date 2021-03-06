import pandas as pd
import numpy as np
from POD import generate_pod_bases, generate_pod_bases_from_svd, load_coefficients_pod
from sklearn.preprocessing import MinMaxScaler


def perform_pod(X_raw):
    # apply pod for above data to reduce the dimensions
    # Eigensolve
    #generate_pod_bases(X_raw,5)
    generate_pod_bases_from_svd(X_raw,5)
    phi, cf = load_coefficients_pod(5)
    print(phi.shape)
    print(cf)

    return phi, np.transpose(cf)
    

def apply_scaling(X_raw):
    print(f"X_raw shape bf scaling : {X_raw.shape}")
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(X_raw)
    return scaled_values, scaler

def get_weekly_covid_data(state):

    # get the covid data and get only the required columns and rows
    X = pd.read_csv ('time_series_covid19_confirmed_US.csv')
    if state != '':
        X = X.loc[X['Province_State'] == state] # get only given state counties
    # drop all unwanted columns
    X.drop(['iso2', 'iso3', 'code3', 'FIPS','Admin2', 'Province_State', 'Country_Region', 'Combined_Key','Lat', 'Long_'], axis=1, inplace=True)
    X.drop(list(X.columns)[1:40], axis=1, inplace=True) # start from march 1st 2020 
    X.drop(['UID'], axis=1, inplace=True)

    # aggregate cases at the end of week
    num_weeks = len(list(X.columns)) // 7
    ndays_to_remove = len(list(X.columns)) % 7
    days_r = list(X.columns)[-1*(ndays_to_remove):]
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

def get_weekly_covid_data_for_state(X, state):

    # get the covid data and get only the required columns and rows
    if state != '':
        X = X.loc[X['Province_State'] == state] # get only given state counties
    # drop all unwanted columns
    X.drop(['iso2', 'iso3', 'code3', 'FIPS','Admin2', 'Province_State', 'Country_Region', 'Combined_Key','Lat', 'Long_'], axis=1, inplace=True)
    X.drop(list(X.columns)[1:40], axis=1, inplace=True) # start from march 1st 2020 
    X.drop(['UID'], axis=1, inplace=True)

    # aggregate cases at the end of week
    num_weeks = len(list(X.columns)) // 7
    ndays_to_remove = len(list(X.columns)) % 7
    days_r = list(X.columns)[-1*(ndays_to_remove):]
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
    #print(f"num_regions, {num_regions}")
    t_size = data.shape[0] - in_win - out_win + 1
    #print(f"t_size, {t_size}")
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

