'''
Abstract base class for the shape of models used to predict MD features.

e.g. CNN would be an implementation of this ABC, and would draw upon cnn.py for its functions.
'''



load_and_transform_data
   featurize_sequence(sequence)
   transform_shape #(batch size, embedding dim, seq length)


build_model
train_model
record_model_parameters
inference
