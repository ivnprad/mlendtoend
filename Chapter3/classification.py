from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784',as_frame=False)

X,y = mnist.data, mnist.target

print(X.shape)
print(y.shape)

import matplotlib.pyplot as plt

def plot_digit(image_data):
    image= image_data.reshape(28,28)
    plt.imshow(image,cmap="binary")
    plt.axis("off")

some_digit=X[0]
# plot_digit(some_digit)
# plt.show()

X_train, X_test, y_train, y_test = X[:60000],X[60000:],y[:60000],y[60000:]

y_train_5 = (y_train =='5')
y_test_5 = (y_test=='5')

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)
print(sgd_clf.predict([some_digit]))

# PERFORMANCE MEASURES

from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf,X_train,y_train_5,cv=3,scoring="accuracy"))

from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier()
dummy_clf.fit(X_train,y_train_5)
print( any (dummy_clf.predict(X_train)))

print(cross_val_score(dummy_clf,X_train,y_train_5,cv=3,scoring="accuracy"))

# CONFUSION MATRICES
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_5,y_train_pred)
print(cm)

# PRECISION AND RECALL 
from sklearn.metrics import precision_score, recall_score
print( precision_score(y_train_5,y_train_pred) )
print( recall_score(y_train_5,y_train_pred))

# threshold|{}|][]

y_scores = sgd_clf.decision_function([some_digit])
threshold = 0
y_some_digit_pred = (y_scores>threshold)
print(y_some_digit_pred)

threshold = 3000
y_some_digit_period = (y_scores>threshold)
print(y_some_digit_pred)


# HOW to decide threshold
y_scores = cross_val_predict(sgd_clf,X_train,y_train_5,cv=3,method="decision_function")
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)

plt.plot(thresholds, precisions[:-1], "b--", label="Precision",linewidth=2)
plt.plot(thresholds,recalls[:-1],"g--",label="Recall",linewidth=2)
plt.vlines(threshold,0,1.0,"k","dotted",label="threshold")
[...]
plt.show()


