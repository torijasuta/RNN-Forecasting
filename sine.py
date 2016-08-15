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

LOG_DIR = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
TIMESTEPS = 80
RNN_LAYERS = [80]
DENSE_LAYERS = None
TRAINING_STEPS = 30000
BATCH_SIZE = 100
PRINT_STEPS = TRAINING_STEPS / 100

my_dir = os.sep.join([os.path.expanduser('~'), 'Desktop', 'sine'])

regressor = learn.TensorFlowEstimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, 
                                                          DENSE_LAYERS), 
                                      n_classes=0,
                                      verbose=2,  
                                      steps=TRAINING_STEPS, 
                                      optimizer='SGD',
                                      learning_rate=0.001, 
                                      batch_size=BATCH_SIZE,
                                      class_weight = [1])

#generate SINE WAVE data
X, y = generate_data(np.sin, np.linspace(0, 100, 5000), TIMESTEPS, seperate=False)
# create a lstm instance and validation monitor
validation_monitor = learn.monitors.ValidationMonitor(X['val'], y['val'],
                                                      every_n_steps=PRINT_STEPS,
                                                      early_stopping_rounds=100000)

regressor.fit(X['train'], y['train'], monitors=[validation_monitor], logdir=LOG_DIR)

# based off training, get the predictions
predicted = regressor.predict(X['test'])

predicted1 = predicted
print("INITIAL PRED: ", predicted)

# cycle predictions into the testing input one at a time
# ie: get first prediction, insert into testing data, rerun predictions
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
    print("---------",i,"---------")
print(predicted)


rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
score = mean_squared_error(predicted, y['test'])
print ("MSE: %f" % score)

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(y['test'], label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Sine Wave Forecast   MSE: " + str(score))

name = ‘NAME_HERE.png'
plt.savefig(os.path.join(my_dir, name))
