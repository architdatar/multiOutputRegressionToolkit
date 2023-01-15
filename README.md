# multiOutputRegressionTooklit

## Constrained multi-output problem (in development)
Multi-output regression is when a mdoel is trained to predict more than one outcomes at once. 

Such problems are quite common in science and business. 
For instance, i want to create a material that has as high an elastic modulus as possible while having a melting point higher than 100$\degree C$. What formulation might give me this result? 
I have a dataset for existing materials containing these properties and i need to predict the formulation for a new material with these properties. 
The challenge then is to train a model that can predict a material with these characteristics. 

There are two common ways to approach this problem:
1. *Standard multioutput regressor:* One can fit these regressors independently of each other; i.e., train one model to predict elastic modulus, and another one to predict melting point. There are easy implementations of this in common ML libraries like [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html).
2. *Regressor Chain:* Predict one output at a time and then use it to predict the next one. This can also be implemented using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html). 

### Challenge

In these two specific scenarios, this is fine. But what if the outputs are constrained together? For example, and this is purely hypothetical: maximize elastic modulus stress such that the sum of the elastic modulus and melting point is less than 100. 

Now, this becomes a constrained multioutput regression problem.

### Method
We develop tools which can handle these kind of problems where the underlying relationships can be visualized with various machine learning models. 
The constraints are added to the loss function as a penalty with a hyperparameter ($\lambda$). This function is then minimized with various optimizers found in the Python ecosystem. 


As a starting point, we illustrate it with multivariate linear relationships. 


## Custom nearest neighbors (in development)

In the classic nearest neighbors regression / classification, there are many pre-set methods of quantifying "similarity" between data points. 

The most famous is the Euclidian distance ($||x-y||$). But what if one needs a different metric to quantify similarity for specific problems? 

We have created scikit-learn-like classes for one such custom metric: the modified Mahalanobis distance. 

### Contribution required:
Currently, this implementation is extremely slow as it runs in Python and cannot uses the Brute force algorithm to calculate distances between each data point and scales as $\mathcal{O}(N^2)$. 

If someone can develop a routine in Cython where the KDTree algorithm can be implemented for custom distance metrics, it could be scale as $\mathcal{O}(n\log(n))$.
