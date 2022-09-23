#%%

import pandas as pd
import time
import numpy as np
import scipy.interpolate as sci
from scipy.optimize import curve_fit, least_squares, minimize
import sys
import os

pd.set_option('display.max_rows', 600)
pd.set_option('display.expand_frame_repr', False)
#mpl.use("Agg")
#os.chdir("/mnt/c/Users/hb4959/OneDrive - DuPont/virtual_testing")

if sys.platform == 'win32':
    home = 'D:\\'
else:
    home=os.path.expanduser('~')

#import matplotlib as mpl
#import matplotlib.pyplot as plt
#plt.style.use(os.path.join(home, "mplstyles", "mypaper.mplstyle"))


def altair_model_red(strain, E, a, b, sigmab=100, epsb=1):
    """"""
    A = ( (E*epsb/sigmab)**(b/a) -1) / (E*epsb/sigmab)**b
    stressout=E*strain*(1+A*(E*strain/sigmab)**b)**(-a/b)
    return stressout

def altair_model_red_constraint_1(strain, E, a, b, sigmab=100, epsb=1):
    """
    Gradient d sigma / d epsilon >= 0 \forall \epsilon \belongsto (0, \epsilon_b)
    """
    A = ( (E*epsb/sigmab)**(b/a) -1) / (E*epsb/sigmab)**b
    # there should be another term before this: 
    sigma = altair_model_red(strain, E, a, b, sigmab=sigmab, epsb=epsb)
    #return sigma / (strain * (1 +  A*(E*strain/sigmab)**b)) * (1 + A*(1-a)*(E*strain/sigmab)**b)
    #slightly modifies constraint. 
    return 1 / (strain * (1 +  A*(E*strain/sigmab)**b)) * (1 + A*(1-a)*(E*strain/sigmab)**b)


class ConstrainedOptimizer:
    """
    Performs constrained optimization and allows user to specify the objective function and the optimizer. 
    Wrapper around scipy.optimize's minimize.  
    """

    def predicted_parameter(self, X_data, params_data):
        """
        """
        #Perhaps, it might be a good idea to normalize the data here. We could use scikit-learn for this. 
        return params_data[0] + X_data @ params_data[1:]
    
    def predict_required_parameters(self, X_data, params_data_full):
        """
        Predicts all required model parameters from the parameters related to the data. 
        """
        a_hat = self.predicted_parameter(X_data, params_data_full[:X_data.shape[1]+1]).values

        model_parameters = [a_hat]
        return model_parameters
        
    def loss_function(self, params_data, params_exp, X_data, pred_info_array, lambda_=0):
        """
        Loss function which can be customized externally to include whatever. 
        Pred info array supplies the values of the previously predicted parameters from ML. 
        """
        #a_hat = self.predicted_parameter(X_data, params_data[:X_data.shape[1]+1]).values
        #b_hat = self.predicted_parameter(X_data, params_data[X_data.shape[1]+1:]).values

        [a_hat] = self.predict_required_parameters(X_data, params_data)

        b_hat = pred_info_array[:, 2]
        
        a, b = params_exp 
        E = pred_info_array[:, 0]
        epsb = pred_info_array[:, 4] #match with how the y_B_pred_array is constructed. 
        sigmab = pred_info_array[:, 3]

        #SSE = 1 / (2 * X_data.shape[0]) * ((a_hat - a) @ (a_hat - a) + (b_hat - b) @ (b_hat - b))
        #SSE = 1 / (2 * X_data.shape[0]) * ((a_hat - a) @ (a_hat - a))

        def relu(x):
            """
            ReLu function implemented.
            """
            return max(0, x)
        
        #Next, we will add a penalty term summed over all structures in the dataset. 
        #The form of the penalty term should be as defined previously. 
        
        SSE_term = 0
        penalty_term = 0

        for index_num, index in enumerate(X_data.index):
            #create and sum the penalty term for everything and then make the loss function. 
            #try:
            #print(f"index_num: {index_num}")
            E_loc = E[index_num]
            epsb_loc = epsb[index_num]
            sigmab_loc = sigmab[index_num]
            eps = np.linspace(0.001, epsb_loc, 50)

            sigma = altair_model_red(eps, E_loc, a[index_num], b[index_num], sigmab=sigmab_loc, epsb=epsb_loc)
            sigma_hat = altair_model_red(eps, E_loc, a_hat[index_num], b[index_num], sigmab=sigmab_loc, epsb=epsb_loc)
            SSE_term +=  1 / eps.shape[0] * ((sigma - sigma_hat) @ (sigma - sigma_hat))

            #penalty_name = altair_model_red_constraint_1(eps, E_loc, a_hat[index_num], b_hat[index_num], 
            #                sigmab=sigmab_loc, epsb=epsb_loc) / E_loc
            penalty_name = altair_model_red_constraint_1(eps, E_loc, a_hat[index_num], b_hat[index_num], 
                            sigmab=sigmab_loc, epsb=epsb_loc) #/ E_loc
            penalty_term += relu(-np.min(penalty_name)) 

            #print(f"Index num: {index_num}; Index: {index}; SSE_term: {SSE_term}; Penalty_term: {penalty_term}")
            #if not np.isfinite(SSE_term):
            #   print(f"Index num: {index_num}; Index: {index}; SSE_term: {SSE_term}; Penalty_term: {penalty_term}")
            #   pass

            """
            except:
                print(f"index_num: {index_num}")
                print(f"E_loc: {E_loc}")
                print(f"epsb_loc: {epsb_loc}")
                print(f"sigmab_loc: {sigmab_loc}")
                print(f"a_hat: {a_hat[index_num]}")
                print(f"b_hat: {b_hat[index_num]}")
                print(f"penalty_name: {penalty_name}")

                continue
            """

        SSE_term /= X_data.shape[0]
        penalty_term /= X_data.shape[0]

        #loss = SSE + lambda_ * penalty_term
        loss = SSE_term + lambda_ * penalty_term

        #print(f"Penalty term: {penalty_term}, Loss: {loss}")
        return loss

    def fit(self, *args, **kwargs):
        """
        Wrapper around the scipy.optimize.minimize method. 
        """
        res = minimize(*args, **kwargs)
        self.res = res
        
    def predict(self, X_data):
        """
        Predicts a from the fitted parameter. 
        In future iterations, there will be a function that will relate 
        """
        try:
            popt = self.res.x
        except NameError:
            raise "The fit object has been called without the model having been fit."
    
        y_pred = self.predict_required_parameters(X_data, popt)
        if len(y_pred) == 1:
            #print("Yes")    
            return y_pred[0]
        else:
            return y_pred


    


# %%
