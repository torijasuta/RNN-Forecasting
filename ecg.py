import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import csv
from csv import reader
import os, datetime

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

from model import generate_data, lstm_model
from ECGdata import ecg_template_noisy

my_dir = os.sep.join([os.path.expanduser('~'), 'Desktop', 'Heartbeat'])

LOG_DIR = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
TIMESTEPS = 50
RNN_LAYERS = [150]
DENSE_LAYERS = None
TRAINING_STEPS = 5000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 10

regressor = learn.TensorFlowEstimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, 
                                                          DENSE_LAYERS), 
                                      n_classes=0,
                                      verbose=2,  
                                      steps=TRAINING_STEPS, 
                                      optimizer='SGD',
                                      learning_rate=0.03, 
                                      batch_size=BATCH_SIZE,
                                      class_weight = [1])


X, y = generate_data(ecg_template_noisy, TIMESTEPS, seperate=False)

# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=100000)

regressor.fit(X['train'], y['train'], monitors=[validation_monitor], logdir=LOG_DIR)

# based off training, get the predictions
predicted = regressor.predict(X['test'])

predicted1 = predicted
print("INITIAL PRED: ", predicted)

'''recursive prediction:
cycle predictions into the testing input one at a time
ie: get first prediction, insert into testing data, rerun predictions
prediction is cycled in TIMESTEPS number of times
ie: first prediction becomes final value of X1, second to last value of X2, etc. 
  X['test][i+1][-1]=predicted[i]
  X['test][i+2][-2]=predicted[i]
  X['test][i+3][-3]=predicted[i], etc.
'else' needed when there are less than TIMESTEPS inputs remaining '''

for i in range(len(X['test'])):
    if i < (len(X['test'])-TIMESTEPS):
        for x in range(TIMESTEPS):
            X['test'][i+x+1][(x+1)*(-1)]=predicted[i]
    else: 
        for x in range(len(X['test'])-i-1):
            X['test'][i+x+1][(x+1)*(-1)]=predicted[i]
    predicted = regressor.predict(X['test'])

# calculate score
rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)

# plot and save
plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.title("Electrocardiogram Forecast. MSE: " + str(score))
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.legend(handles=[plot_predicted, plot_test])
plt.savefig(os.path.join(my_dir, "ECG"))
plt.clf()
