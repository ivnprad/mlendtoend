from housing import load_housing_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

housing = load_housing_data()
print(housing.head())

# prepare categories to stratidifed sampling since median income is the attribute more correlated with the target label
housing["income_cat"] = pd.cut( housing["median_income"], 
                               bins=[0.,1.5, 3.0,4.5,6., np.inf],
                               labels=[1,2,3,4,5])

strat_train_set, strat_test_set = train_test_split(housing,test_size=0.2,stratify=housing["income_cat"],random_state=42)

#drop income_cat after stratified sampling

for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1, inplace=True)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Total bedrooms attribute has some missing values 

# housing.dropna(subset=["total_bedrooms"],inplace=True) # option 1 get rif of the corresponding districts

# housing.drop("total_bedrooms",axis=1) # option 2 get rid of the whole attribute

# median = housing["total_bedrooms"].median() # option 3 set the missing values to some value (zero,the mean, the median,etc)
# housing["total_bedrooms"].fillna(median,inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

# only for numerical values
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)

# there are also more powerful imputers available in the sklean KNNImputer and Iterativeimputer
# KNN Imputer calculate mean of the neighbors values for the that feature

# IterateImputer trains a regression model per feature to predict the missing value based on all 
# the other avaialble feature. It then trains the model again on the updated data, and repeats
# the process several times, improving the models and the replacement values at each iteration

X = imputer.transform(housing_num)

# Scikit-learn transfomers output Numpy arrays even when they are fed Pandas Dataframes as input
housing_tr = pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
print (housing_tr.head())


# Handling Textand Categorical Attributes
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))

# CATEGORIAL ATTRIBUTES Since ML algorithms like to work with number 
from sklearn.preprocessing import OrdinalEncoder

ordinalEncoder = OrdinalEncoder()
housing_cat_encoded = ordinalEncoder.fit_transform(housing_cat)
print(housing_cat_encoded[:8])
print(ordinalEncoder.categories_)

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#print(housing_cat_1hot) # store nonzero value and position
print(cat_encoder.categories_)
print(cat_encoder.feature_names_in_)


# FEATURE SCALING AND TRANSFORMATION

# scaling

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

import  matplotlib.pyplot as plt

fig, axs = plt.subplots(1,2, figsize=(8,3),sharey=True)
housing["population"].hist(ax=axs[0],bins=50)
housing["population"].apply(np.log).hist(ax=axs[1],bins=50)
plt.show()
# housing_num["population"].hist(bins=50,figsize=(12,8))
# plt.show()

# from sklearn.metrics.pairwise import rbf_kernel

# age_simil_35 = rbf_kernel(housing_num[["housing_median_age"]],[[35]],gamma=0.1)
# # how to plot age_simil_35

# from sklearn.linear_model import LinearRegression

# target_scaler = StandardScaler()
# scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

# model = LinearRegression()
# model.fit(housing[["median_income"]],scaled_labels)
# some_new_data = housing[["median_income"]].iloc[:5] # pretend this is is new data

# scaled_predictions = model.predict(some_new_data)
# predictions = target_scaler.inverse_transform(scaled_predictions)
# print(predictions)
# print(some_new_data)