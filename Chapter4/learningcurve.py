# Polynomial regression
import numpy as np

np.random.seed(42)
number_of_instances=100
X=6*np.random.rand(number_of_instances,1)-3
y=0.5*X**2+X+2+np.random.randn(number_of_instances,1)

# plot model predictions
# import matplotlib.pyplot as plt

# extra code – this cell generates and saves Figure 4–12
# plt.figure(figsize=(6, 4))
# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$")
# plt.ylabel("$y$", rotation=0)
# plt.axis([-3, 3, 0, 10])
# plt.grid()
# plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2,include_bias=False)
X_poly=poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])

# Linear regression using scikit-learn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
print(lin_reg.intercept_)
print(lin_reg.coef_)

# extra code – this cell generates and saves Figure 4–13

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(6, 4))
# plt.plot(X, y, "b.")
# plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
# plt.xlabel("$x_1$")
# plt.ylabel("$y$", rotation=0)
# plt.legend(loc="upper left")
# plt.axis([-3, 3, 0, 10])
# plt.grid()
# plt.show()

from sklearn.model_selection import learning_curve

train_sizes,train_scores,valid_scores = learning_curve(
    LinearRegression(),X,y,train_sizes=np.linspace(0.01,1.0,40),cv=5,
    scoring="neg_root_mean_squared_error")
train_errors=-train_scores.mean(axis=1)
valid_errors=-valid_scores.mean(axis=1)

# import matplotlib.pyplot as plt

# plt.plot(train_sizes,train_errors,"r-+",linewidth=2,label="train")
# plt.plot(train_sizes,valid_errors,"b-",linewidth=3,label="valid")
# plt.xlabel("Training set size")
# plt.ylabel("RMSE")
# plt.grid()
# plt.legend(loc="upper right")
# plt.axis([0, 80, 0, 2.5])
# plt.show()

from sklearn.pipeline import make_pipeline

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10,include_bias=False),
    LinearRegression()
)

train_sizes,train_scores,valid_scores = learning_curve(
    polynomial_regression,X,y,train_sizes=np.linspace(0.01,1.0,40),cv=5,
    scoring="neg_root_mean_squared_error")
train_errors=-train_scores.mean(axis=1)
valid_errors=-valid_scores.mean(axis=1)

import matplotlib.pyplot as plt

plt.plot(train_sizes,train_errors,"r-+",linewidth=2,label="train")
plt.plot(train_sizes,valid_errors,"b-",linewidth=3,label="valid")
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.grid()
plt.legend(loc="upper right")
plt.axis([0, 80, 0, 2.5])
plt.show()





