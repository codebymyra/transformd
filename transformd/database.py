"""Handle data saving, loading, and checking which features to calculate for a given trajectory."""

from pathlib import Path
import pickle
import numpy as np


#need to be able to add a sequence entry to the database
#load the database
#save the database
#check database for sequence -> it can already do this since it's a dictionary
#append an array to a feature of a given sequence
class SequenceDatabase:
    '''
    The database object stores many sequence entries. It handles all sequences and calculated MD features for those sequences.

    Structure:
    {
    'Seq1': {feature1: [array1, array2, ...], feature2: [array1, array2, ...]},
    'Seq2': {feature1: [array1, array2, ...], feature2: [array1, array2, ...]},
     }

    Where arrays are numpy arrays of calculated MD features.
    If only trajectory held that protein sequence, there will only be one data entry.

    Functions:
    - load database
    - save database
    - check database for a sequence
    - convert raw dictionary into SequenceEntry object

    '''
    def __init__(self, filepath: str):
        self.filepath = Path(filepath) #to database file
        self.data = {}

    def load_pickle(self):
        '''
        Loads database from disk or creates one if it doesn't exist yet
        '''
        if self.filepath.exists():
            with open(self.filepath, 'rb') as f:
                self.data = pickle.load(f)

    def save_pickle(self):
        temp_pkl = self.filepath.with_suffix('.tmp')
        with open(temp_pkl, 'wb') as f:
            pickle.dump(self.data, f)

        temp_pkl.replace(self.filepath)

    def add_data(self, sequence: str, feature_name: str, data: np.ndarray):
        '''
        Append a new calculated MD feature array to a feature for a given sequence.
        '''
        #create new sequence entry and feature_name entry if they don't already exist
        sequence = sequence.strip().upper()
        if sequence not in self.data:
            self.data[sequence] = {}
        if feature_name not in self.data[sequence]:
            self.data[sequence][feature_name] = []

        #append new data
        self.data[sequence][feature_name].append(np.asarray(data, dtype=np.float32))

