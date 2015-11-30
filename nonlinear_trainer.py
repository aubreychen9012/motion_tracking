__author__ = 'aubrey9012'

import numpy as np
import pandas as pd
import scipy
import math
import scipy.io

from keras.models import Sequential
from keras.layers.core import Dense, Activation,Dropout
from keras.layers.recurrent import LSTM

import sys
sys.path.append('/Users/aubrey9012/PycharmProjects/untitled5/')
import filters
import estimate


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def diffdist(array,dist):
    step = int(dist)
    array1 = np.asarray(array[:-step])
    array2 = np.asarray(array[step:])
    diff5 = array2-array1
    blank_start = np.asarray([0]*step)
    diff5 = np.append(blank_start,diff5)
    return diff5

def phase(array):
    fft_array = np.fft.fft(array)
    phase_ = np.array([i.imag for i in fft_array])[1:]
    phase_ = np.append(np.array([0]),phase_)
    return phase_

def amp(array):
    fft_array = np.fft.fft(array)
    amp_ = np.array([i.real for i in fft_array])[1:]
    amp_ = np.append(np.array([0]),amp_)
    amp_ = amp_[len(amp_)/2:]
    amp_ = upsampling(amp_,len(amp_)*2)
    return amp_


def gen_features(array,window_length,dist):
    feat_orig = rolling_window(array,window_length)
    feat_mean = [np.nan_to_num(pd.rolling_mean(i,6)) for i in feat_orig]
    feat_diff = np.diff(array)
    feat_diff = np.append([0],feat_diff)
    feat_diff = rolling_window(feat_diff,window_length)
    #feat_std = np.nan_to_num(pd.rolling_std(array,6))
    #feat_std = rolling_window(feat_std, window_length)
    feat_std = [np.nan_to_num(pd.rolling_std(i,6)) for i in feat_orig]
    feat_amp = np.array([amp(i) for i in feat_orig])
    feat_grad = [np.gradient(i,1) for i in feat_orig]
    feat_2grad = [np.gradient(i,2) for i in feat_orig]
    feat_diffdist = np.array([diffdist(i,dist) for i in feat_orig])
    feat = np.hstack((feat_orig,feat_mean,feat_diff,feat_diffdist))
    feat = feat.reshape(-1,4*window_length)
    feat = feat.astype('float32')
    return feat

def gen_targets(array,window_length,dist):
    #dist = frame_to_dist(FRAME_RATE,LATENCY)
    dist += window_length
    targets = array[dist:]
    return targets

def upsampling(data,upscale):
    x = range(len(data))
    newx = [i/(float(upscale)-1)*x[-1] for i in range(upscale)]
    f = scipy.interpolate.interp1d(x,data)
    new_data = f(newx)
    new_data= new_data[1:-1]
    st = [data[0]]
    st.extend(new_data)
    st.append(data[-1])
    return st

def load_data(array, window_length, dist):
    X = gen_features(array, window_length,dist)
    y = gen_targets(array,window_length,dist)
    return X,y

def split_data(array, split_rate):
    length = array.shape[0]
    idx = math.ceil(length*split_rate)
    array_t = array[:idx]
    array_v = array[idx:]
    return array_t, array_v


file_names = ['Paff01','Paff02','Paff03','Paff04','Paff05',\
             'Paff07','Paff08','Paff09','Paff10',\
             'Paff11','Paff12','Paff13','Paff14','Paff15',\
             'Paff16','Paff17','Paff18','Paff19','Paff20',\
             'Paff21','Paff22','Paff23','Paff24','Paff25']
## 'paff06' excluded

LATENCY = 600
WINDOW_LENGTH= 100
TRAIN_VALID_SPLIT = 0.8
TEST_RATE=0.2

file_name = 'Paff01'
pf = '/Users/aubrey9012/Documents/projects/masterarbeit/AffineRes/'+\
     str(file_name)+'.mat'
paff = scipy.io.loadmat(pf)
paff = paff['Paff']

## regress on two dimensions separately

paff_x = paff[0]
paff_y = paff[1]

# load data
paff_x_train = paff_x[:len(paff_x)*(1-TEST_RATE)]
paff_x_test = paff_x[len(paff_x)*(1-TEST_RATE)-WINDOW_LENGTH:]

## filter
filter = filters.MovingAverageFilter(paff,8)
filtered_paff_x_train = filter.filt(paff_x_train)


X,y = load_data(filtered_paff_x_train,WINDOW_LENGTH, dist=15)
X = X[:len(y)]
X_t,X_v = split_data(X, TRAIN_VALID_SPLIT)
y_t,y_v = split_data(y, TRAIN_VALID_SPLIT)


in_neurons = 2
out_neurons = 2
hidden_neurons1 = 150
hidden_neurons2 = 100

## build RNN LSTM
model = Sequential()
model.add(LSTM(input_dim=in_neurons, output_dim=hidden_neurons1, return_sequences=False))
#model.add(LSTM((6,100), 300, return_sequences=True)) ## input layer
#model.add(LSTM(300, 200, return_sequences=True))     ## LSTM layer
#model.add(Dropout(0.2))
#model.add(LSTM(500, 200, return_sequences=False))
#model.add(Dropout(0.2))
model.add(Dense(hidden_neurons1,hidden_neurons2))
model.add(Dense(hidden_neurons2, out_neurons))                             ## output layer
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")


