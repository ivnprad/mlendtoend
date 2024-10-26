import numpy as np 
from housing import load_housing_data
from zlib import crc32
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

def shuffle_and_split_data(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices= shuffled_indices[:test_set_size]
    train_indices= shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def is_id_in_test_set(identifier,test_ratio):
    max_uint32_t = 2**32
    int_identifier = np.int64(identifier)
    bytes_identifier = int_identifier.tobytes()
    return crc32(np.int64(bytes_identifier))< test_ratio*max_uint32_t

def split_data_with_id_hash(data,test_ratio,identifier_column):
    ids=data[identifier_column]
    in_test_set= ids.apply(lambda id_:is_id_in_test_set(id_,test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]

def add_index_and_split_data_with_hash(housing):
    housing_with_id=housing.reset_index() # adds an an 'index' column equal to row index
    test_ratio = 0.2
    return split_data_with_id_hash(housing_with_id,test_ratio,"index")
    housing_with_id["id"] = housing["longitude"]*1000+housing["latitude"]
    train_set, test_set = split_data_with_id_hash(housing_with_id,0.2,"id")

def plot_income_category(housing):
    housing["income_cat"].value_counts().sort_index().plot.bar(rot=0,grid=True)
    plt.xlabel("Income category")
    plt.ylabel("Number of districts")
    plt.show()

def stratified_shuffle_split(housing):
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2,random_state=42)
    strat_splits=[]
    for train_index, test_index in splitter.split(housing,housing["income_cat"]):
        strat_train_set_n = housing.iloc[train_index]
        strat_test_set_n = housing.iloc[test_index]
        strat_splits.append([strat_train_set_n,strat_test_set_n])

    return strat_splits

housing = load_housing_data()

# train_set, test_set = shuffle_and_split_data(housing,0.2)
# print(f"train set length {len(train_set)} and test set length {len(test_set)}")

# train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
# print(f"train set length {len(train_set)} and test set length {len(test_set)}")  

# if an attribute is important for a data set it is crucial to get the right number of samples
# for each stratum 

housing["income_cat"] = pd.cut( housing["median_income"], 
                               bins=[0.,1.5, 3.0,4.5,6., np.inf],
                               labels=[1,2,3,4,5])
#print(housing.head())
#plot_income_category(housing)

strat_splits = stratified_shuffle_split(housing)
strat_train_set,strat_test_set = strat_splits[0]
print(f"strat train set length {len(strat_train_set)} and strat test set length {len(strat_test_set)}")  

strat_train_set, strat_test_set = train_test_split(housing,test_size=0.2,stratify=housing["income_cat"],random_state=42)
print(f"strat train set length {len(strat_train_set)} and strat test set length {len(strat_test_set)}")  

