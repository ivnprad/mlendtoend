# def monkey_patch_get_signature_names_out():
#     """Monkey patch some classes which did not handle get_feature_names_out()
#        correctly in Scikit-Learn 1.0.*."""
#     from inspect import Signature, signature, Parameter
#     import pandas as pd
#     from sklearn.impute import SimpleImputer
#     from sklearn.pipeline import make_pipeline, Pipeline
#     from sklearn.preprocessing import FunctionTransformer, StandardScaler

#     default_get_feature_names_out = StandardScaler.get_feature_names_out

#     if not hasattr(SimpleImputer, "get_feature_names_out"):
#       print("Monkey-patching SimpleImputer.get_feature_names_out()")
#       SimpleImputer.get_feature_names_out = default_get_feature_names_out

#     if not hasattr(FunctionTransformer, "get_feature_names_out"):
#         print("Monkey-patching FunctionTransformer.get_feature_names_out()")
#         orig_init = FunctionTransformer.__init__
#         orig_sig = signature(orig_init)

#         def __init__(*args, feature_names_out=None, **kwargs):
#             orig_sig.bind(*args, **kwargs)
#             orig_init(*args, **kwargs)
#             args[0].feature_names_out = feature_names_out

#         __init__.__signature__ = Signature(
#             list(signature(orig_init).parameters.values()) + [
#                 Parameter("feature_names_out", Parameter.KEYWORD_ONLY)])

#         def get_feature_names_out(self, names=None):
#             if callable(self.feature_names_out):
#                 return self.feature_names_out(self, names)
#             assert self.feature_names_out == "one-to-one"
#             return default_get_feature_names_out(self, names)

#         FunctionTransformer.__init__ = __init__
#         FunctionTransformer.get_feature_names_out = get_feature_names_out

# monkey_patch_get_signature_names_out()

from housing import load_housing_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

housing = load_housing_data()
# print(housing.head())

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

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    
def column_ratio(X):
    return X[:,[0]] / X[:,[1]]

def ratio_name(function_transformer,feature_names_in):
    return ["ratio"] # features names out

def ratio_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"),
                         FunctionTransformer(column_ratio,feature_names_out=ratio_name),
                         StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)
cluster_simil=ClusterSimilarity(n_clusters=10,gamma=1,random_state=42)
default_num_pipeline=make_pipeline(SimpleImputer(strategy="median"),
                                   StandardScaler())


from sklearn.compose import ColumnTransformer, make_column_selector

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer(
    [
        ("bedrooms",ratio_pipeline(),["total_bedrooms","total_rooms"]),
        ("rooms_per_house",ratio_pipeline(),["total_rooms","households"]),
        ("people_per_house",ratio_pipeline(),["population","households"]),
        ("log",log_pipeline,["total_bedrooms","total_rooms","population","households","median_income"]),
        ("geo",cluster_simil,["latitude","longitude"]),
        ("cat",cat_pipeline,make_column_selector(dtype_include=object)),
],
    remainder=default_num_pipeline
)

# housing_prepared = preprocessing.fit_transform(housing)
# print(housing_prepared.shape)
# print(preprocessing.get_feature_names_out())

from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing,LinearRegression())
lin_reg.fit(housing,housing_labels)

housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))
print(housing_labels.iloc[:5].values)

from sklearn.metrics import mean_squared_error
lin_rmse = mean_squared_error(housing_labels, housing_predictions,
                              squared=False)
print(lin_rmse)  