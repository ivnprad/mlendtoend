import os
import tarfile
from six.moves import urllib
import ssl 
import requests
import pandas as pd

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
    csv_path = os.path.join(housing_path, "housing.csv") 
    return pd.read_csv(csv_path)



housing = load_housing_data()
print(housing.head())
print("hello")