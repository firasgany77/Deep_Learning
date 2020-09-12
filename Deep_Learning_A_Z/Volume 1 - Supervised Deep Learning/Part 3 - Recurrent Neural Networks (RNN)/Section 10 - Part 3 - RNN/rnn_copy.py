
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

# reshaping:
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
# this is a structure with three dimentions: stock prices, timestamps, number of indicators.

from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout


#initializing the RNN
regressor = Sequential( )
# we named this as regressor because we are predicting a continuous value

# units: number of units/LSTM cells (we want high number of dimentionality)
# return_sequence = True (becasue we want to add another LSTM layer after this one)
# we are making a stacked LSTM network: more than one LSTM network one after another.
# the input shape is that last two dimentions: timestamps, indicators. 

# adding the first LSTM layer and some Dropout regularization (to avoid overfitting)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) 

# adding the second LSTM layer and some Dropout regularization (to avoid overfitting)
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

# adding the third LSTM layer and some Dropout regularization (to avoid overfitting)
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) 

# adding the fourth LSTM layer and some Dropout regularization (to avoid overfitting)
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2)) 

# adding the output layer
#units: the # of neurons that should be in the output layer.
#units = 1 , the stock price at time t+1.
# the output includes the ground truth, which is the stock price at time t+1.
# we're training the RNN on the Truth (true stock prices that are happening at
# time t+1 after the 60 stock prices during the 60 finanicial days, and that's
# why we also need to include the ground truth, and therefore y train)
regressor.add(Dense(units = 1)) 


# Compiling the RNN
regressor.compile(optimizer ='adam', loss='mean_squared_error')

# Fitting the RNN to Training Set
# we insert 4 args: the inputs of the training set, 
# will be forward propagated to the output which will be the prediction,
# which will be compared to the ground truth that is in Y_train
# epochs: how many times we want the data to be forward probagated inside the network.

regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)

#if we get a loss too small at the end we might get overfitting.

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# getting the predicted stock price of january 2017
# reminder: we trained or model to be able to predict the stock price
# at time t+1 based on the sixty previous stock prices.
# therefore, in order to predict each stock price of each finanical day
# of jaunary 2017 we will need the 60 previous stock prices before the the
# actual day.

# in order to get at each day of Jan 2017 the sixty previous stock prices of 
# each day, we will need both the training and test sets, becasue some of the 60
# days will be from the training set (from DEC 2016), and some from the test set
# which will come from JAN 2017.

# now we want to concatenate the training set and test set, and that to be able to get
# the 60  previous inputs for each day of JAN 2017.

# if we concatenate : training_set = dataset_train.iloc[:, 1:2].values 
# with real_stock_price = dataset_test.iloc[:, 1:2].values
# that will lead us to a problem, because what we have to do is to scale this 
# concacenating of the train set and the test set.
# and that means that we will change the test values, but we don't want to do this
# we want to keep the test values as they are.

# to handle this problem we will concatenate the original data frames: 
# dataset_train = pd.read_csv('Google_Stock_Price_Train.csv') 
# with dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

# and from this concatenating we will get the input of each prediction, that is
# the 60 previous stock prices at each time t, and this is the data that we will scale 
# in order to get the prediction. that way we will be only scaling the inputs without 
# changing the actual test values.

# why do we need to scale the inputs?
# Answer: because our RNN was trained on the scaled values of the training set.

 dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']) , axis=0)
 # the vertical axis is labeled by zero
 # we're going to need the stock prices from the first finanicial day of JAN 2017 , 
 # minus 60, up to the last stock price of our whole data set.
 
 # first we get the lower bound of range of inputs we need:
 # the index: (len(dataset_total) - len(dataset_test)) gives the first finanicial
 # day of JAN 2017.
 
 
 inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:]
 # : at the end means continue to the end of the vector dataset_total
 inputs = inputs.values.reshape(-1,1)
 
 # the scaling we are applying to our input is the same as the one we did
 # on the training set, so we directly use fit.transform:
 inputs = sc.transform(inputs)

# we don't use the fit.transform because our sc object is already fitted
# to the training set.
X_test = [] # the name is consistent with the training input X_train
            # we don't need Y_test because we're not doing training, 
            # but directly we will do predictions.

for i in range(60, 80): #range for 20 finanicial days
    X_test.append(inputs[i-60:i, 0]) #moving window of 60 pins
X_test = np.array(X_test) 
# reshaping:
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
#inverse the scaling
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
  

# Visualization of results
# real google 
plt.plot(real_stock_price, color = 'green', label='Real Google Stock Price')
plt.plot(real_stock_price, color = 'blue', label='Predicted Google Stock Price')


