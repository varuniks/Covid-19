from utils import *
from ML_model import fit_model, make_forecasts, eval_forecasts
import numpy as np

def process_region(X_raw, in_win, out_win, n_epochs, batch_size):
    print("process_region called")
    scaled_values, scaler = apply_scaling(X_raw)
    X_tr, Y_tr, X_t, Y_t = convert_to_supervised(scaled_values, in_win, out_win, 0.4)

    model = fit_model(X_tr, Y_tr, in_win, out_win, n_epochs, batch_size, 'ED')
    prediction = make_forecasts(model, batch_size, X_t, in_win, out_win)

    # rescale
    for i in range(len(prediction)):
        prediction[i,:,:] = np.rint(scaler.inverse_transform(prediction[i,:,:]))
        Y_t[i,:,:] = np.rint(scaler.inverse_transform(Y_t[i,:,:]))

    rmse, mape = eval_forecasts(Y_t, prediction, in_win, out_win)
    print(f"rmse:{rmse}, mape:{mape}")
    return [rmse, mape] 

