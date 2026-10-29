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
    traj_list = transformd.get_trajs_in_working_dir(test_working_dir)
    assert isinstance(traj_list, list), f'Did not return a dictionary.'
    assert len(traj_list) > 0, f'List of trajectory file pairs is empty.'
    for top, traj in traj_list:
        assert isinstance(top, str), 'Topology is not a string.'
        assert isinstance(traj, str), 'Trajectories is not a string.'


def test_pipeline():
    #todo
    #run test.slurm
    #check that raw_data.pkl is created, trajectory_log.pkl, and feautre_dump
    #check that a few of the feature_dump features have 4 keys with int labels
    #check that raw_data and trajectory_log both have 4 keys. Trajectory_log should have 14 labels for each key
    #finally, delete all the files
    pass





