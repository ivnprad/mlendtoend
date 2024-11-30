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
housing_num = housing.select_dtypes(include=[np.number])
housing_labels = strat_train_set["median_house_value"].copy()

import  matplotlib.pyplot as plt

housing["population"].hist(bins=50,figsize=(12,8))
#plt.show()

# apply logarithm if the feature is positive and tail is on the right
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel

# log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
# log_pop = log_transformer.transform(housing[["population"]])
# log_pop.hist(bins=50,figsize=(12,8))
# plt.show()

rbf_transformer = FunctionTransformer(rbf_kernel,kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35=rbf_transformer.transform(housing[["housing_median_age"]])
 

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_array,check_is_fitted

class StandardScalerClone(BaseEstimator,TransformerMixin):
    def __init___(self,with_mean=True): # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self,X,y=None): # y is required even though we don't use it
        X = check_array(X) # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1] # every estimator stores this in fit()
        return self # always return self 
    
    def transform(self,X):
        check_is_fitted(self) # looks for learned attributes ( with trailing__)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X/self.scale_
    


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# num_pipeline = Pipeline([
#     ("impute",SimpleImputer(strategy="median")),
#     ("standarize",StandardScaler()),
# ])

num_pipeline = make_pipeline(SimpleImputer(strategy="median"),StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num)
print(housing_num_prepared[:2].round(2))

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared,columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index
)
print(df_housing_num_prepared.head())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder#

num_attribs = ["longitude","latitude","housing_median_age","total_rooms",
               "total_bedrooms","population","households","median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num",num_pipeline,num_attribs),
    ("cat",cat_pipeline,cat_attribs)
])

from sklearn.compose import make_column_selector,make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline,make_column_selector(dtype_include=np.number)),
    (cat_pipeline,make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)
df_housing_num_prepared = pd.DataFrame(
    housing_prepared,columns=preprocessing.get_feature_names_out(),
    index=housing.index
)
print(df_housing_num_prepared.head())