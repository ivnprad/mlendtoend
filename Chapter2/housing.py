import os
import tarfile
from six.moves import urllib
import ssl 
import requests
import pandas as pd
import matplotlib.pyplot as plt

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/blob/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH): 
    if not os.path.isdir(housing_path):
         os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")

    urllib.request.urlretrieve(housing_url, tgz_path)
    file_size = os.path.getsize(tgz_path)
    if file_size > 409400:  # File size greater than 400 KB
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
        print("Download successful, and file size is greater than 400 KB.")
    else:
        print("Downloaded file is smaller than 400 KB.")

def load_housing_data(housing_path=HOUSING_PATH): 
    currentDirectory = os.getcwd()
    csv_path = os.path.join(currentDirectory+"/Chapter2/"+housing_path, "housing.csv")
     
    if not os.path.exists(csv_path):
        print(f"Error: the file {csv_path} does not exist")
        return None

    return pd.read_csv(csv_path)

#housing = load_housing_data()

# if housing is not None:
#     print(housing.head())
#     housing.info()
#     print(housing["ocean_proximity"].value_counts())
#     print(housing.describe())
    #housing.hist(bins=50,figsize=(12,8))
    #plt.show()
    
