__author__ = 'aubrey9012'

import numpy as np
import math

## basic RMSE
def RMSE(array,array_real):
    s = np.mean(((np.asarray(array)-np.asarray(array_real)))**2)
    rmse = math.sqrt(s)
    return rmse

## compute RMSE with two predicted arrays and their true values
def RMSE_2d(array_x,array_y,true_x,true_y):
    sum_x = sum(((np.asarray(array_x)-np.asarray(true_x))*10)**2)
    sum_y = sum(((np.asarray(array_y)-np.asarray(true_y))*10)**2)
    sum_val = sum_x+sum_y
    error = sum_val/len(array_x)
    error = math.sqrt(error)
    return error

## compute RMSE with two predicted arrays's RMSE
def error2d(x,y):
    x = np.asarray(x)
    y = np.asarray(y)
    err2d = x**2+y**2
    err2d_ = [math.sqrt(i) for i in err2d]
    err2d_ = np.mean(err2d_)
    return err2d_

## compute baseline method's error, one dim.
def base_error(array,dist):
    array_shifted = array[dist:]
    array = array[:len(array_shifted)]
    err = RMSE(array,array_shifted)
    return err

## compute baseline method's error, two dim.
def base_error_2d(array_x,array_y,dist):
    array_shifted_x = array_x[dist:]
    array_x = array_x[:len(array_shifted_x)]
    array_shifted_y = array_y[dist:]
    array_y = array_y[:len(array_shifted_y)]
    err = RMSE_2d(array_shifted_x,array_shifted_y,array_x,array_y)
    return err

