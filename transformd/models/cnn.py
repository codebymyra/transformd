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
load_and_transform_data #this is what will be called in the md_feature_predictor part
   split_train_test
   featurize_sequence(sequence)
   transform_shape #(batch size, embedding dim, seq length)


def split_train_test():
   load_data()
   shuffle_data
   split_data
   save_splits #somehow attach these data to a dictionary with the model parameters?


def featurize_sequence(sequence):
   #just featurize one sequence, or a batch?


build_model
train_model
record_model_parameters
inference
'''
Methods for embedding/featurizing protein sequences.
The input is a string (the protein sequence).



#run experiment: set certain hyperparameters?
#create model
#train