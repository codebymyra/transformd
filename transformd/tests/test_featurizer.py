'''
Unit tests for featurizers.
These classes and functions convert a database of data into a format intermediate to ML.
'''

from transformd import featurizer
import numpy as np
from pathlib import Path


test_path_1d = 'transformd/tests/test_data/1d_data'
test_file = featurizer.ByFeatureDatabase(test_path_1d)
def test_add_individual_feature_data():
    test_data = np.array([0.24038284, 0.90880023, 0.46224285, 0.4249765 , 0.10644279])
    test_dict = {'hi': np.array([1,2,3,4,5])}
    feature_name = 'test'
    test_seq = 'ASDF'
    test_file.add_individual_feature_data(test_dict, feature_name, test_seq, test_data)
    #which files should exist?
    for i in range(test_data.shape[0]):
        res_feature_name = f'test_res_{i}'
        filepath = Path(test_path_1d) / f'{res_feature_name}.pkl'
        assert Path(filepath).is_file()
        test_dict = test_file.load_dict(res_feature_name)
        assert isinstance(test_dict, dict)
        assert len(test_dict) == 2
        #todo: add a different sequence, add the same sequence again, then check that the number of entries is correct
        #todo: check that the values/labels within the dictionary are single values
