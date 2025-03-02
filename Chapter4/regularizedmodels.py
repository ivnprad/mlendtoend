import numpy as np

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(6, 4))
# plt.plot(X, y, ".")
# plt.xlabel("$x_1$")
# plt.ylabel("$y$  ", rotation=0)
# plt.axis([0, 3, 0, 3.5])
# plt.grid()
# plt.show()

from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
print(ridge_reg.predict([[1.5]]))

from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(penalty="l2", alpha=0.1/m,tol=None,max_iter=1000,eta0=0.01,random_state=42)
sgd_reg.fit(X,y.ravel()) # y.ravel() because fit() expect 1D targets
print(sgd_reg.predict([[1.5]]))

# LASSO regression
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X,y)
print(lasso_reg.predict([[1.5]]))

sgd_reg_lasso = SGDRegressor(penalty="l1", alpha=0.1/m,tol=None,max_iter=1000,eta0=0.01,random_state=42)
sgd_reg_lasso.fit(X,y.ravel()) 
print(sgd_reg_lasso.predict([[1.5]]))

# early stopping