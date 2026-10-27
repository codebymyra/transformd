'''
Add any functions related to converting aggregated data from SequenceDatabase into a different format for downstream tasks.
Inputs are:
protein sequence: str
feature name: str
data: numpy array of dimensions (1,) or (n_residues,)
'''

from pathlib import Path
import pickle
import numpy as np
from datetime import datetime
from database import SequenceDatabase

class ArrayDatabase:
    '''
    Converts a Sequence Database object into a numpy array.

    Example
    seq1: {feature1: [arr1], feature2: [arr2]} is converted to
    np.array([
    [s e q u e n c e],
    [feature 1 array],
    [feature 2 array]
    ])

    This only works on by-residue features.
    '''
    def __init__(self, filedir: str):
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filedir = Path(filedir)
        self.datafile = self.filedir / f'{time}_features.npy'

    @staticmethod
    def by_residue(sequence: str, data: np.ndarray):
        '''
        Returns boolean based on whether the data received are by-residue or by-protein.
        '''
        assert data.shape[0] > 0, f'Received an empty array for {sequence}'
        assert data.ndim == 1, f'Expected a 1D array but got {data.ndim}D'
        n_vals = data.shape[0]
        if n_vals == len(sequence):
            return True
        return False

    def load_npy(self):
        '''
        Load numpy array from file.
        If the file does not yet exist, return an empty list to be converted later.
        '''
        if self.datafile.is_file():
            data_arr = np.load(self.datafile)
        else:
            data_arr = []
        return data_arr

    def save_npy(self, data: np.ndarray):
        '''
        Save numpy array to file.
        '''
        np.save(self.datafile, data, allow_pickle=False)

    def find_array(self, feature_array: np.ndarray):
        '''
        Searches all the trajectory arrays for a specific sequence.
        '''
        pass
        # todo

    #todo: turn aggregated data into an array format
    def seq_db_to_feature_db(self, sequence: str, seq_dict: SequenceDatabase, feature_name: str):
        '''
        Add aggregated data for a sequence to a numpy array.
        Converts dictionary into a numpy array.

        Parameters
        ---
        seq_dict: dict
            Dictionary entry from Sequence Database. Format:
            'sequence': {feature1: [[nparray]], feature2: [[nparray]]}, 'sequence2': {feature1: [[nparray], [nparray]], feature2: [[nparray], [nparray]]}

        Returns
        ---
        data_array: numpy array
            (n_features, n_residues) array containing features for a given trajectory.

        '''
        pass




        def append_array(self, data_array: np.ndarray):
            '''
            Loads the feature array.
            Append the array from dict_to_array to a loaded larger array.
            Save the updated array.
            '''
            pass
            #todo





#spaghetti coding what I want to do to turn raw data into numpy array for transformer
#map features to specific rows of a list t obe turned into an array.
#how to add features later without causing trouble, or








#I'm not making this a class because I don't want to initialize an object; I just want to perform functions on loaded data.
class ByFeatureDatabase:
    '''
    Functions responsible for converting data from a SequenceDatabase to data split into sequence - label pairs.
    Resulting files are pickled dictionaries with 'sequence': [single label]
    '''
    def __init__(self, filedir: str):
        self.filedir = Path(filedir)

    def load_dict(self, feature_name: str):
        '''
        Opens pickled dict or creates new dict if file does not exist.
        Used to contain datasets usable for downstream ML tasks.
        '''
        feature_file = self.filedir / f'{feature_name}.pkl'

        if Path(feature_file).is_file():
            with open(feature_file, 'rb') as f:
                return pickle.load(f)

        return {}

    def save_dict(self, feature_name: str, data_dict: dict):
        '''
        Save updated dictionary to pickle.
        Should save a file like 'rmsf-ca-res1.pkl' or 'rg.pkl'
        '''

        filepath = self.filedir / f'{feature_name}.pkl'
        temp_pkl = filepath.with_suffix('.tmp')
        with open(temp_pkl, 'wb') as f:
            pickle.dump(data_dict, f)

        temp_pkl.replace(filepath)

    def add_individual_feature_data(self, data_dict: dict, feature_name: str, sequence: str, data: np.ndarray):
        '''
        Get a single sequence-label pair and add it to a dataset.
        If the input array has dimensions (n_res,), the values will be split out to by-residue files.
        This automatically saves the updated dictionary using the save_dict function in this class.

        Parameters
        ---
        data_dict: dict
            Loaded dictionary object that can have new sequence-label pairs added or updated.
            Keys: sequences, values: labels.
        sequence: str
            Protein sequence from a trajectory
        feature_name: str
            Name of the MD feature that is calculated
        data: numpy array
            The actual data for that MD feature. Can have dimensions (1,) or (n_residues,)

        Returns
        ---
        Saves out CSV files for each individual feature.

        Examples
        ---
        >>add_individual_feature_data('ASDF', 'rmsf_ca', [[1,2,3]])
        Adds entry to three CSV files:

        '''
        #get number of data values. Should be either 1 or n_res
        assert data.shape[0] > 0, f'{feature_name} is empty'
        n_vals = data.shape[0]

        if n_vals == 1: #for by-protein features
            data_dict[sequence] = data
            self.save_dict(feature_name, data_dict)
        elif len(sequence) == n_vals: #by-residue features
            for res in range(n_vals):
                res_feature_name = f'{feature_name}_res_{res}'
                data_dict[sequence] = data[res]
                self.save_dict(res_feature_name, data_dict)






#CNN:
#get a single sequence - label value
#apend these seq-label values to their respective files
#split out data for each individual feature
#save out data
#load data in the future to adjust

#transformer:
#if the data is by-residue (more than dim 1),
#split out sequence into individual string/letters in a numpy array
#below that, append each of the aggregated by-residue feature arrays
#we need to keep the order of features constant. Maybe by implementing a dictionary?
#turn a sequence entry into a feature format
#save out data
#load data in the future to adjust