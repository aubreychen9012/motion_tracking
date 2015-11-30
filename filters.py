__author__ = 'aubrey9012'

import numpy as np
import pandas as pd
import scipy


class MovingAverageFilter():
    def __init__(self,data,window_length):
        self.data = data
        self.window_length = window_length
    def filt(self, d):
        filtered_data = pd.rolling_mean(np.asarray(d),self.window_length)
        filtered_data = filtered_data[self.window_length:]
        return filtered_data
    def filt2(self):
        if self.data.shape[0]!=2:
            return "input shape: 2-d array"
        fil_d = [self.filt(d).tolist() for d in self.data]
        fil_d = np.asarray(fil_d)
        return fil_d


class LowPassFilter():
    def __init__(self,cutoff,fs,order):
        self.fs = fs
        self.cutoff = cutoff
        self.order = order

    def butter_lowpass(self):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = scipy.signal.butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a
    def butter_lowpass_filter(self,data):
        b, a = self.butter_lowpass()
        butter_signal = scipy.signal.lfilter(b, a, data)
        return butter_signal
    ## forward and backward filter
    def filt_filt_filtering(self,data):
        b, a = self.butter_lowpass()
        sig_ff = scipy.signal.filtfilt(b, a, data)
        return sig_ff

