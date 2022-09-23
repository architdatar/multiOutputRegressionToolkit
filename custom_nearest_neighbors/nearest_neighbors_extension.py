"""Extension of the KNeighborsRegressor class to enable linear interpolation. 
"""
#%%
#When writing it as a package, use relative imports.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import check_is_fitted
from sklearn.utils.fixes import delayed

from joblib import Parallel
from scipy.interpolate import Rbf

import numpy as np
import pandas as pd

def custom_distance(x, y, VI, grade_num_col=-1, grade_dist_weight=1):
    """
    Adds Mahalanobis distance and a custom distance to identify those with the same grade.
    The idea is to ensure that the "nearest neighbors" of a particular curve are those with
    the same grade. 
    
    Parameters
    -----------
    x, y: 
    V: Variance matrix
    grade_num_col: Index of column which contains the grade. 
    grade_dist_weight: Weight of the grade wise distance. 
    
    Returns
    -----------
    Distance of the two points.
    """
        
    x_grade_num = x[grade_num_col]; y_grade_num = y[grade_num_col]
    x_exc_grade = np.delete(x, grade_num_col)
    y_exc_grade = np.delete(y, grade_num_col)
        
    #mahalanobis_distance = maha_dist.pairwise([x_exc_grade, y_exc_grade], V=V)[-1, 0]
    #mahalanobis_distance = np.sqrt((x_exc_grade-y_exc_grade).T @ np.linalg.inv(V) @ (x_exc_grade-y_exc_grade))
    #mahalanobis_distance = np.sqrt((x_exc_grade-y_exc_grade).T @ VI @ (x_exc_grade-y_exc_grade))
    diff_array = (x_exc_grade-y_exc_grade).reshape(-1, 1)
    mahalanobis_distance = np.sqrt(diff_array.T @ VI @ diff_array)
    
    if x_grade_num == y_grade_num:
        grade_distance = 0 #for the same grade 
    else:
        grade_distance = 1 #for a different grade
    combined_distance = mahalanobis_distance + grade_dist_weight * grade_distance
    
    #return [mahalanobis_distance, grade_distance, combined_distance]
    return combined_distance


def get_column_index(X_train, column_name="temperature"):
    """
    Returns index of the temperature column when the 
    X input is dataframe.
    """

    try:
        temp_index = X_train.columns.to_list().index(column_name)
    except:
        raise KeyError(f"{column_name} not in dataframe index and\
            therefore not found.")

    return temp_index


class KNeighborsRegressor_LI(KNeighborsRegressor):

    def __init__(self,     
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        interp_function="linear",
        temp_index=0,
    ):
        """
        We cannot have *args and **kwargs which would have been the
        standard way of extending classes. This causes a problem when
        we perform cross-validation. Thus, we must have named arguments
        as shown here. 
        """
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            )
        
        self.interp_function = interp_function
        self.temp_index = temp_index


    def predict(self, X):
        """
        Nearest neighbors prediction with radial basis function interpolation.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.RBFInterpolator
        """

        # From values of training X and y, get NN for the test points.
        # Then, use those to perform interpolation. 
        
        check_is_fitted(self)
        
        # if not hasattr(self, "temp_index"):
        #     raise AttributeError("KNeighbors_LI class does not\
        #         have the attribute temp_index. Please set it externally.")

        _X = self._fit_X
        _y = self._y

        if type(X) == pd.core.frame.DataFrame or type(X) == list:
            X = np.asarray(X)            

        neigh_dist, neigh_ind = self.kneighbors(X)

        y_pred = np.full((X.shape[0],), np.nan)

        def interpolate_value(row_ind):
        #for row_ind in range(X.shape[0]):
            row = X[[row_ind], :] # 1D matrix

            nearest_X = _X[neigh_ind[row_ind, :]]
            nearest_y = _y[neigh_ind[row_ind, :]]

            # neigh_dist, neigh_ind = self.kneighbors(row)

            # nearest_X = _X[neigh_ind[0, :]]
            # nearest_y = _y[neigh_ind[0, :]]

            if self.n_neighbors > 2:
                # Calculate everything row wise.
                #neigh_dist, neigh_ind = self.kneighbors(row)
                #nearest_X = _X[neigh_ind][0]
                #nearest_y = _y[neigh_ind][0]

                nearest_x_args = (nearest_X[:, col_ind] for col_ind in range(nearest_X.shape[1]))

                # Currently, we are writing for loops. Eventually, this for loop can be parallelized using multithreading. Look at the
                # code used in other fitting values. Use delayed. 
                # https://stackoverflow.com/questions/42220458/what-does-the-delayed-function-do-when-used-with-joblib-in-python
                rbfi = Rbf(*nearest_x_args, nearest_y, function=self.interp_function)
                interp_val = float(rbfi(*row[0, :]))

            else:
                temp_index = self.temp_index
                #temp_index = 3

                #if nearest_X[1, temp_index] != nearest_X[0, temp_index]:
                #Linear interpolation:
                interp_val = nearest_y[0] +\
                    (nearest_y[1] - nearest_y[0]) / (nearest_X[1, temp_index] - nearest_X[0, temp_index]) * (row[0, temp_index] - nearest_X[0, temp_index])

                # If the value is not finite, it means that the 
                # nearest neighbor is such that it has same temperatures. 
                # In this case, we want to simply predict the value at the nearest neighbor.
                # Similarly, if some of the parameters are negative, we would still want to
                # predict the values of the nearest. 
                if not np.isfinite(interp_val) or interp_val < 0: 
                    #Predict the value for the nearest neighbor.
                    nearest_index = np.argmin(neigh_dist[row_ind,:])
                    interp_val = nearest_y[nearest_index]

                    #interp_val = np.mean(nearest_y)

            if not np.isfinite(interp_val):
                print(f"row_ind: {row_ind}, row: {row}, interp_val:{interp_val}")
                print(f"nearest_X: {nearest_X}, nearest_y: {nearest_y}")

                #Nearest value
                #nearest_index = np.argmin(neigh_dist[0,:])
                #interp_val = nearest_y[nearest_index]
                #interp_val = 0
            
            #y_pred[row_ind] = interp_val
            return interp_val
        
        if self.n_jobs is not None:
            if self.n_jobs > 1 and X.shape[0]>1:
                y_pred = Parallel(n_jobs=self.n_jobs)(
                    delayed(interpolate_value)(row_ind) for row_ind in range(X.shape[0])
                )
            else:
                y_pred = [interpolate_value(row_ind) for row_ind in range(X.shape[0])]
        else:
            y_pred = [interpolate_value(row_ind) for row_ind in range(X.shape[0])]

        # Vectorize this for loop. It seems that the 
        # np.apply_along_axis function is slower than the for loop.
        # so, for now, we are using the for loop. However, later, if we are able to figure out
        # a way to vectorize the process, we will certainly do that.
        """
        def interpolate_from_NN(row, interp_function="linear"):
            #row is a slice. 
            neigh_dist, neigh_ind = self.kneighbors(row.reshape(1, -1))
            nearest_X = _X[neigh_ind][0]
            nearest_y = _y[neigh_ind][0]

            nearest_x_args = (nearest_X[:, col_ind] for col_ind in range(nearest_X.shape[1]))

            rbfi = Rbf(*nearest_x_args, nearest_y, function=interp_function)
            interp_val = float(rbfi(*row))

            return interp_val

        y_pred = np.apply_along_axis(interpolate_from_NN, 1, X, interp_function=interp_function)
        """
        return y_pred


if __name__ == "__main__":
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    kNN_LI = KNeighborsRegressor_LI(n_neighbors=2, n_jobs=1, 
        temp_index=0)

    X_train = np.array([[0, 0.3], [1, 0.6], [2, 0.7], [3, .9]])
    #X_train = [[0], [1], [2], [3]]

    #y_train = [0, 1, 2, 3]
    y_train = [0, 2, 4, 6]
    #y_train = [0, 1.5, 2.5, 3]

    kNN_LI.fit(X_train, y_train)

    X_test = np.array([[0, 0], [0.5, 0]])
    #X_test = [[.5], [1.5]]
    #X_test = X_train

    kNN_LI.predict([[.5, 0.5]])

    """
    t1 = time.time()
    for _ in range(1000):
        kNN_LI.predict_LI(X_test)
    t2 = time.time()
    print(f"Time for program completion: {t2-t1:.2f}")
    """

    #X_test = np.linspace(-10, 10, 40).reshape(-1, 1)
    #y_pred = kNN_LI.predict_LI(X_test.reshape(-1,1))
    #plt.plot(X_test.reshape(-1,), y_pred); plt.scatter(np.array(X_train).reshape(-1,), y_train)

# %%
