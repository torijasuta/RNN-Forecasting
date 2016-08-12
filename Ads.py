import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import csv
from csv import reader
import os, datetime
import time
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from model import generate_data, lstm_model

# hyperparameters: these are adjustable and lead to different results
LOG_DIR = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
TIMESTEPS = 8
RNN_LAYERS = [150]
DENSE_LAYERS = None
TRAINING_STEPS = 5000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 10
LEARNING_RATE = 0.05

# TensorFlowEstimator does all the training work
regressor = learn.TensorFlowEstimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, 
                                                          DENSE_LAYERS), 
                                      n_classes=0,
                                      verbose=1,  
                                      steps=TRAINING_STEPS, 
                                      optimizer='SGD',
                                      learning_rate=LEARNING_RATE, 
                                      batch_size=BATCH_SIZE,
                                      continue_training=True)

#read the data 
print("Reading CSV file...")
with open('pub.csv') as f:
    data = list(reader(f.read().splitlines()))

    # get output
    # for 'data.csv', standardized impressions are in column 5
    adOps = [float(i[5]) for i in data[1::]]
    tf.to_float(adOps, name='ToFloat') 

X, y = generate_data(adOps, TIMESTEPS, seperate=False)
regressor.fit(X['train'], y['train'])
    
# based off training, get the predictions
# these initial predictions use measured values in the input, 
# not the predicted values. will implement recursive technique later
predicted = regressor.predict(X['test'])

# store the initial predictions
predicted1 = predicted
    
# recursive prediction set up:
# get first prediction, insert into testing data, rerun predictions
# prediction is cycled in TIMESTEPS number of times
# ie: first prediction becomes final value of X1, second to last value of X2, etc. 
#      X['test][i+1][-1]=predicted[i]
#      X['test][i+2][-2]=predicted[i]
#      X['test][i+3][-3]=predicted[i], etc.
# "else" needed when there are less than TIMESTEPS inputs remaining

for i in range(len(X['test'])):
    if i < (len(X['test'])-TIMESTEPS):
        for x in range(TIMESTEPS):
            X['test'][i+x+1][(x+1)*(-1)]=predicted[i]
    else: 
        for x in range(len(X['test'])-i-1):
            X['test'][i+x+1][(x+1)*(-1)]=predicted[i]
    predicted = regressor.predict(X['test'])

# calculate score to measure success
rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)

# plot stuff
plot_predicted, = plt.plot(predicted, label='prediction')
plot_test, = plt.plot(y['test'], label='test')
plt.title("Data Forecast" + "MSE: " + str(score))
plt.xlabel("Date")
plt.ylabel("Normalized Impressions")
plt.legend(handles=[plot_predicted, plot_test])

# location to save plots
# user must create folder to save to
my_dir = os.sep.join([os.path.expanduser('~'), 'Desktop', 'test'])

# name the plot, save the plot
name = "8, 150, 5k, 0.05" + '.png' #named after the hyperparameters
plt.savefig(os.path.join(my_dir, name))
plt.clf()
