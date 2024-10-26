from housing import load_housing_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

housing = load_housing_data()
housing["income_cat"] = pd.cut( housing["median_income"], 
                               bins=[0.,1.5, 3.0,4.5,6., np.inf],
                               labels=[1,2,3,4,5])

strat_train_set, strat_test_set = train_test_split(housing,test_size=0.2,stratify=housing["income_cat"],random_state=42)

#drop income_cat after stratified sampling

for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat",axis=1, inplace=True)

housing = strat_train_set.copy()
