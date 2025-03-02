import numpy as np 
np.random.seed(42)
number_of_instances=100
X=2*np.random.rand(number_of_instances,1) # colummn vector
y=4+3*X+np.random.rand(number_of_instances,1) # columnn vector

from sklearn.preprocessing import add_dummy_feature

X_b=add_dummy_feature(X)
theta_best = np.linalg.inv(X_b.T@X_b)@X_b.T@y
print(theta_best)

# make predictions

X_new = np.array([[0],[2]])
X_new_b=add_dummy_feature(X_new) # add x0 = 1 to each instance
y_predict = X_new_b@theta_best
print(y_predict)

# plot model predictions
# import matplotlib.pyplot as plt

# plt.plot(X_new,y_predict,"r-",label="Predictions")
# plt.plot(X,y,"b.")
# [...]
# plt.show()

# Linear regression using scikit-learn
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)
print(lin_reg.intercept_)
print(lin_reg.coef_)

# least squares

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b,y,rcond=1e-6)
print(theta_best_svd)

print(np.linalg.pinv(X_b)@y)


# Batch gradient descent algorithm

learning_rate=0.1
n_epochs=1000
instances_count=len(X_b)

np.random.seed(42)
theta=np.random.rand(2,1)

for epoch in range(n_epochs):
    gradients=2/instances_count*X_b.T@(X_b@theta-y )
    theta=theta-learning_rate*gradients

print(theta)

# Stochastic Gradient descent
n_epochs=50
t0,t1 = 5,50 # learning schedule hyperparameters

def learning_schedule(t):
    return t0/(t+t1)

np.random.seed(42)
theta=np.random.randn(2,1) # random initialization

for epoch in range(n_epochs):
    for iteration in range(number_of_instances):
        random_index=np.random.randint(number_of_instances)
        xi=X_b[random_index:random_index+1]
        yi=y[random_index:random_index+1]
        gradients=2*xi.T @(xi @theta-yi ) # for SGD, do not divide by m
        eta = learning_schedule(epoch*number_of_instances+iteration) 
        theta = theta-eta*gradients

print(theta)

# Stochastic Gradiant descent using sklearn
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000,tol=1e-5,penalty=None,eta0=0.01,n_iter_no_change=100,random_state=42)
sgd_reg.fit(X,y.ravel()) # y.ravel() because fit expects 1D targets

print(sgd_reg.intercept_)
print(sgd_reg.coef_)