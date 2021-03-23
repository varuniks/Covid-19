from models import get_model
from tensorflow.keras import optimizers, models, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
def fit_model(input_seq, output_seq, in_win, out_win, num_epochs, batchsize, model_name):
    n_regions =  input_seq.shape[-1]
    #print(f"n_regions: {n_regions}")
    model = get_model(model_name, in_win, out_win, n_regions, dropout=0, training_t=True)

    # design network
    my_adam = optimizers.Adam(lr=0.0001, decay=0.0)
    earlystopping = EarlyStopping(monitor='loss', patience=50, verbose=1)
    callbacks_list = [earlystopping]
    #print(model.summary())
    # fit network
    model.compile(optimizer=my_adam,loss='mean_squared_error', run_eagerly=True)
    model.fit(input_seq, output_seq, epochs=num_epochs, batch_size=batchsize, verbose=1, callbacks=callbacks_list) 

    return model

def forecast_model(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = np.expand_dims(X[:,:],0)
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    #print(forecast.shape)
    #print([x for x in forecast[0, :, :]])
    #return [x for x in forecast[0, :, :]]
    return forecast

def forecast_with_fb_model(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = np.expand_dims(X[:,:],0)
    # make forecast
    forecast = model.predict(X)
    # convert to array
    #print(forecast.shape)
    #print([x for x in forecast[0, :, :]])
    #return [x for x in forecast[0, :, :]]
    return forecast
def make_forecasts(model, n_batch, data, in_win, out_win):
    #print(data.shape)
    #forecasts = list()
    forecasts = np.zeros(shape=(len(data), out_win, data.shape[2]))
    for i in range(len(data)):
        X = data[i, 0:in_win, :]
        # make forecast
        forecast = forecast_model(model, X, n_batch)
        #print(forecast)
        forecasts[i,:,:] = forecast
        # store the forecast
        #forecasts.append(forecast)
    return forecasts

    
def eval_forecasts(true, pred, in_win, out_win):
    #print(f"true data: {true}")
    #print(f"pred data: {pred}")
    rmse_out = np.zeros(shape=(out_win))
    mape_out = np.zeros(shape=(out_win))
    for i in range(out_win):
        actual = true[:,i,:] 
        forecast = pred[:,i,:] 
        RMSE = np.sqrt(mean_squared_error(actual, forecast))
        MAPE = np.mean(abs((actual - forecast) / (actual+1)))
        print('t+%d RMSE: %f' % ((i+1), RMSE))    
        print('t+%d MAPE: %f' % ((i+1), MAPE))    
        mape_out[i] = MAPE
        rmse_out[i] = RMSE
    return rmse_out, mape_out

