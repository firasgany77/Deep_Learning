
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

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
Y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# the frist line of observation in X_train corresponds with t=60. 
# all the values to the right of index 1 (at first row) are the previous
# values of the last 60 finanicial days.

#Anytime we want to add a dimention in numpy array, we always
# need to use the reshape function.
# At this point X_train has two dimentions (np array with 1198 rows and 60 columns )

# inputs: A 3D tensor, with shape [batch, timesteps, feature].
# batch: the total number of observations that we have. (1198)
# feature: the number of indicators we have , which are in this case 1 indicator.
# we are adding a new finanical indicator that could help predict Google 
# stock price trends.
# example: since Apple & Samsung depend on each other, their stock prices 
# might be highly correlated.

#(X_train.shape[0]) : number of lines in X_train
#(X_train.shape[1]): number of columns in X_train = num of timestamps.
# the last arguments is the number of indicators (num of predictors) which is 1 (the 
# Open google stock price)


X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
# this is a structure with three dimentions: stock prices, timestamps, number of indicators.

from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout


#initializing the RNN
regressor = Sequential( )
# we named this as regressor because we are predicting a continuous value

# adding the first LSTM layer and some Dropout regularization (to avoid overfitting)
# units: number of units/LSTM cells (we want high number of dimentionality)
# return_sequence = True (becasue we want to add another LSTM layer after this one)
# we are making a stacked LSTM network: more than one LSTM network one after another.
# the input shape is that last two dimentions: timestamps, indicators. 

regressor.add(LSTM(units = 50, return_sequence = True, input_shape = (X_train.shape[1], 1)))
regressor