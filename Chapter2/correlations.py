from housing import load_housing_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

housing = load_housing_data()
print(housing.head())
housing["income_cat"] = pd.cut( housing["median_income"], 
                               bins=[0.,1.5, 3.0,4.5,6., np.inf],
                               labels=[1,2,3,4,5])

strat_train_set, strat_test_set = train_test_split(housing,test_size=0.2,stratify=housing["income_cat"],random_state=42)

#drop income_cat after stratified sampling

for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1, inplace=True)

housing = strat_train_set.copy()

#housing.plot(kind="scatter",x="longitude",y="latitude", grid=True, alpha =0.2)
#plt.show()

print(housing.head())

# housing.plot(kind="scatter",x="longitude",y="latitude",grid=True,
#              s=housing["population"]/100, label="population",
#              c="median_house_value",cmap="jet",colorbar=True,
#              legend=True,sharex=False,figsize=(10,7))
# plt.show()


# look for correlations
correlation = housing.copy()
correlation.drop("ocean_proximity",axis=1,inplace=True)
corr_matix = correlation.corr()
# how much each attribue correlates with the median house
print(corr_matix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
#scatter_matrix(correlation[attributes], figsize=(12,8))
#plt.show()

# find data quirks ...horizontal lines where there is no correlation
#correlation.plot(kind="scatter",x="median_income",y="median_house_value",alpha=.1, grid=True)
#plt.show()

# add attributes 
correlation["rooms_per_house"] = correlation["total_rooms"]/correlation["households"]
correlation["bedrooms_ratio"] = correlation["total_bedrooms"]/correlation["total_rooms"]
correlation["people_per_house"] = correlation["population"]/correlation["households"]

corr_matix = correlation.corr()
print( corr_matix["median_house_value"].sort_values(ascending=False))
