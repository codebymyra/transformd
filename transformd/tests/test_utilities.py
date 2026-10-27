import os
import tempfile
import transformd
from pathlib import PurePath

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
    test_working_dir = '/jumbo/burginlab/myra/transformd/transformd/tests/test_data/test_working_directory'
    traj_dict = transformd.get_trajs_in_working_dir(test_working_dir)
    print(traj_dict)
    assert isinstance(traj_dict, dict), f'Did not return a dictionary.'
    assert len(traj_dict.keys()) > 0 and len(traj_dict.values()) > 0, f'Dict of trajectory file pairs is empty.'
    for top, trajs in traj_dict.items():
        assert isinstance(top, str), 'Topology is not a string.'
        assert isinstance(trajs, list), 'Trajectories is not a list.'





