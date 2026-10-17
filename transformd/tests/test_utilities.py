import os
import tempfile
import transformd
from pathlib import Path, PurePath

def test_get_working_dir():
    #create a temporary directory and test to see if it exists
    #And assert that it returns a Path object
    with tempfile.TemporaryDirectory() as working_dir:
        temp_path = transformd.get_working_dir(working_dir)
        assert os.path.isdir(temp_path)
        assert isinstance(temp_path, PurePath)

def test_get_trajs_in_working_dir():
    #assert list format
    #assert that each item in the list has two files
    test_working_dir = 'transformd/tests/test_data/test_working_directory'
    pair_list = transformd.get_trajs_in_working_dir(test_working_dir)
    assert isinstance(pair_list, List), f'Did not return a list.'
    assert len(pair_list) > 0, f'List of trajectory file pairs is empty.'
    for item in pair_list:
        assert len(item) == 2, f'{item} should have exactly 2 files'
        assert isinstance(item[0], str), f'{item[0]} should be a string'
        assert isinstsance(item[1], str), f'{item[1]} should be a string'
