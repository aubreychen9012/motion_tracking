__author__ = 'aubrey9012'

import numpy as np
import pandas as pd
import math
import sklearn
import sys
import scipy.io
sys.path.append('/Users/aubrey9012/PycharmProjects/untitled5/')
import data_info

class ma_filter():
    def __init__(self,data,window_length):
        self.data = data
        self.window_length = window_length
    def filt(self, d):
        filtered_data = pd.rolling_mean(np.asarray(d),self.window_length)
        filtered_data = filtered_data[5:]
        return filtered_data
    def pos_filt(self):
        if self.data.shape[0]!=2:
            return "input shape: 2-d array"
        fil_d = [self.filt(d).tolist() for d in self.data]
        fil_d = np.asarray(fil_d).T
        return fil_d

def RMSE_2d(array_x,array_y,true_x,true_y):
    sum_x = sum(((np.asarray(array_x)-np.asarray(true_x)))**2)
    sum_y = sum(((np.asarray(array_y)-np.asarray(true_y)))**2)
    sum_val = sum_x+sum_y
    error = sum_val/len(array_x)
    error = math.sqrt(error)
    return error

class linear_trainer():
    def __init__(self):
        self.model = sklearn.linear_model.LinearRegression()
        self.score = 0
        self.ransac = sklearn.linear_model.RANSACRegressor(sklearn.linear_model.LinearRegression())
    def frame_to_dist(self,frame_rate,latency):
        time = 1/float(frame_rate)*1000
        dist = math.ceil(latency/time)
        return dist
    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    def gen_features(self,data, history_length):
        l = len(data)
        roll_idx = self.rolling_window(np.array(range(l)),history_length)
        features = data[roll_idx]
        features = features.reshape(-1,features.shape[1]*features.shape[2])
        return features
    def gen_targets(self,array,history_length,dist):
        dist += history_length
        target = array[dist:]
        return target
    def train(self, data, frame_rate, latency,history_length=17,split=0.8,RANSAC=False):
        dist = self.frame_to_dist(frame_rate,latency)
        features = self.gen_features(data,history_length)
        targets = self.gen_targets(data,history_length,dist)
        split_point = math.ceil(len(features)*split)
        features_t = features[:split_point]
        features_v = features[split_point:len(targets)]
        targets_t = targets[:split_point]
        targets_v = targets[split_point:]
        if RANSAC == False:
            self.model = self.model.fit(features_t,targets_t)
        elif RANSAC ==True:
            self.model = self.ransac.fit(features_t,targets_t)
        pred = self.model.predict(features_v)
        self.score = RMSE_2d(pred.T[0],pred.T[1],targets_v.T[0],targets_v.T[1])
        return
    def predict(self,data,frame_rate,latency,history_length=17,split=0.8):
        dist = self.frame_to_dist(frame_rate,latency)
        features = self.gen_features(data,history_length)
        targets = self.model.predict(features)
        return targets


file_names = ['Paff01','Paff02','Paff03','Paff04','Paff05',\
             'Paff07','Paff08','Paff09','Paff10',\
             'Paff11','Paff12','Paff13','Paff14','Paff15',\
             'Paff16','Paff17','Paff18','Paff19','Paff20',\
             'Paff21','Paff22','Paff23','Paff24','Paff25']

LATENCY = 150
WINDOW_LENGTH= 17
TRAIN_SPLIT=0.8

overall = []
scores = []
for name in file_names:
    pf = '/Users/aubrey9012/Documents/projects/masterarbeit/AffineRes/'+\
         str(name)+'.mat'
    FRAME_RATE= data_info.frame_dict[name][0]
    CONVERSION = data_info.frame_dict[name][1]
    paff = scipy.io.loadmat(pf)
    paff = paff['Paff']*CONVERSION
    paff_train = paff[:paff.shape[0]*0.8]
    paff_test = paff[paff.shape[0]*0.8-WINDOW_LENGTH:]
    fil = ma_filter(paff_train,5)
    fil_paff = fil.pos_filt()
    trainer = linear_trainer()
    trainer.train(fil_paff,FRAME_RATE,LATENCY,split=0.5)
    overall.append(trainer.score*data_info.frame_dict[name][-1])
    scores.append(trainer.score)
    predicted = trainer.predict(paff_test)
    dist = linear_trainer.frame_to_dist(FRAME_RATE,LATENCY)
    test_targets = linear_trainer.gen_targets(paff_test,WINDOW_LENGTH,dist)
    RMSE_2d(predicted.T[0],predicted.T[1],test_targets.T[0],test_targets.T[1])

overall = sum(overall)/159819
print overall                            ## overall score of validation