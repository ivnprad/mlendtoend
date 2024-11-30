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
    
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator,TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
 
    def fit (self, X,y=None,sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters,n_init=10,
                              random_state = self.random_state)
        self.kmeans_.fit(X,sample_weight=sample_weight)
        return self
    
    def transform(self,X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_,gamma=self.gamma)
    
    def get_features_name_out(self,names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]],
                                           sample_weight=housing_labels)

print(similarities[:3].round(2))

housing_renamed = housing.rename(columns={
    "latitude": "Latitude", "longitude": "Longitude",
    "population": "Population",
    "median_house_value": "Median house value (ᴜsᴅ)"})
housing_renamed["Max cluster similarity"] = similarities.max(axis=1)

housing_renamed.plot(kind="scatter", x="Longitude", y="Latitude", grid=True,
                     s=housing_renamed["Population"] / 100, label="Population",
                     c="Max cluster similarity",
                     cmap="jet", colorbar=True,
                     legend=True, sharex=False, figsize=(10, 7))
plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1],
         cluster_simil.kmeans_.cluster_centers_[:, 0],
         linestyle="", color="black", marker="X", markersize=20,
         label="Cluster centers")
plt.legend(loc="upper right")
#save_fig("district_cluster_plot")
plt.show()