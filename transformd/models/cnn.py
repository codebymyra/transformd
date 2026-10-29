'''
Hyperparameter tuning and training for by-feature CNNs.
1-D convolutional neural network

Contains all the things needed for a CNN but actual invocation exists in an ABC
'''

import torch
import torch.nn as nn

#get_datasets (train, test, val)
#data now: individual pickle files with dict format: {seq: label}
#pytorch 1d cnn intput shape: (N, Cin, Lin) or (Cin, Lin) where N = batch size, C = number of channels (embedding dim for a given sequence?), L = length of sequence
#invoke factory that invokes appropriate sequence featurizer.
#save some kind of log with model parameters, which features were run, time, etc.



#run experiment: set certain hyperparameters?
#create model
#train