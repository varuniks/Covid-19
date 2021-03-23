import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize
from sklearn.utils.optimize import _check_optimize_result

class MyGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=2e05, gtol=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter, 'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min

def fit_GP(X_train, Y_train, nl=0.3**2, nlb=(0.1**2, 0.5**2)):
    kernel_0 = WhiteKernel(noise_level=nl, noise_level_bounds=nlb)
    kernel_1 = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(length_scale=1.44, periodicity=1)
    ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
    ker_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)
    kernel_l = [kernel_0+kernel_1, kernel_0+kernel_1+ker_rbf, kernel_1]
    param_grid = {"kernel": kernel_l,  "alpha": [1e1,0], "optimizer": ["fmin_l_bfgs_b"], "n_restarts_optimizer": [5, 10, 15],
              "normalize_y": [True, False]}
    #gpm = GaussianProcessRegressor(kernel=kernel_0+kernel_1, n_restarts_optimizer=10, normalize_y=True, alpha=0.0)
    gpm = MyGPR(kernel=kernel_0+kernel_1, n_restarts_optimizer=10, normalize_y=True, alpha=0.0)
    #gp = GaussianProcessRegressor()
    #gpm = GridSearchCV(gp, param_grid=param_grid)
    gpm.fit(X_train, Y_train)
    return gpm

# Generate predictions.
def evaluate_error_predictions(model_e, data_in):
    y_pred, y_std = model_e.predict(data_in, return_std=True)
    return y_pred, y_std

def fit_error_model(cf, cf_t, cf_e, out_win):
    
    # X values for gp fit
    x_cf = np.transpose(cf,(0,2,1))
    x_cf = x_cf.reshape(x_cf.shape[0]*x_cf.shape[1], x_cf.shape[2]) # 3915, 3
    x_t_cf = np.transpose(cf_t,(0,2,1))
    x_t_cf = x_t_cf.reshape(x_t_cf.shape[0]*x_t_cf.shape[1], x_t_cf.shape[2]) # 19*135, 3
    # y values for gp fit
    xe_cf = np.transpose(cf_e, (0,2,1))
    xe_cf = xe_cf.reshape(xe_cf.shape[0]*xe_cf.shape[1], xe_cf.shape[2]) # 3915, 4
    print("error from model")
    print(cf_e)
    print("error from model")
    print(xe_cf)
    cf_e_pred = np.zeros(shape=(xe_cf.shape[0], xe_cf.shape[1]))
    cf_te_pred = np.zeros(shape=(x_t_cf.shape[0], xe_cf.shape[1]))
    print(cf_e_pred.shape, cf_te_pred.shape)

    for tstep in range(xe_cf.shape[1]):
        # fit GP

        X_train = x_cf
        Y_train = xe_cf[:,tstep]
        X_test = x_t_cf
        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.reshape(-1,1).shape)
        print(X_train)  
        print(Y_train)  
        print("GP Fit")  
        #gp_fit = fit_GP(X_train.reshape(-1,1), Y_train.reshape(-1,1), 0.2**2, (0.1**2, 0.3**2))
        gp_fit = fit_GP(X_train, Y_train, 0.2**2, (0.1**2, 0.3**2))

        #evaluate the prediction on the fitted model
        # training data
        Y_e_pred, _ = evaluate_error_predictions(gp_fit, X_train)
        Y_te_pred, _ = evaluate_error_predictions(gp_fit, X_test)
        print("GP eval #################################################")
        print(cf_e_pred)
        print(Y_e_pred.shape, Y_te_pred.shape)
       
        cf_e_pred[: ,tstep] = Y_e_pred
        cf_te_pred[:,tstep] = Y_te_pred

    print(cf_e_pred)
    print(cf_te_pred)

    cf_e_pred = cf_e_pred.reshape(cf.shape[0], cf_e.shape[2], cf_e.shape[1])
    cf_te_pred = cf_te_pred.reshape(cf_t.shape[0], cf_e.shape[2], cf_e.shape[1])

    cf_e_pred = np.transpose(cf_e_pred, (0,2,1))
    cf_te_pred = np.transpose(cf_te_pred, (0,2,1))
    
    print(cf_e_pred)
    print(cf_te_pred)
    return cf_e_pred, cf_te_pred     
