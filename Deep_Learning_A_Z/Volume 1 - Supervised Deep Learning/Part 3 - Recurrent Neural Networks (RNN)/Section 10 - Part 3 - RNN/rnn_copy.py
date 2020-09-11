
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set

# we extract the "Open" column 
# we want to create numpy array, so we do the extraction by [:, 1:2] and not [:, 1] 
# ranges in python has their upper bound excluded, so the 2 in [:, 1:2] is excluded.
# we use .values to create numpy array.  
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values


# Feature Scaling
# standardization = (x-mean(x))\(std deviation)
# normalization = (x-mean(x))\(max(x) - min(x))

