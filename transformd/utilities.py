'''
utilities.py

Misc. utility functions
'''
from pathlib import Path
import pickle
import isee_framework
from transformd import features

def get_working_dir(working_dir: str):
    '''
    Interfaces with user settings to get an isEE working directory.

    Parameters
    ___
    working_dir: str
        Absolute file path to be converted to Path object.

    Returns
    ___
    wd_path: Path
        Absolute file path of the working directory.
    '''

    wd_path = Path(working_dir)

    return wd_path

def get_trajs_in_working_dir(working_dir: str):
    '''
    Takes in the wd_path from get_working_dir().
    Return a list of trajectory:topology file pairs based on the restart.pkl file in the working directory.

    Parameters
    ___
    working_dir: str
        String of working directory path. To be converted to Path object.
    Returns
    ___
    pair_list: list
        List of topology:trajectory pairs from the restart.pkl file located in the working directory.

    Example
    ___
    >>get_working_dir('working')
    [(traj_tleap.prmtop_0, traj_0.nc), (traj_tleap.prmtop_1, traj_1.nc)]
    '''

    # specific to isEE
    tops = []
    trajs = []
    wd_path = get_working_dir(working_dir)
    restart_file = wd_path / 'restart.pkl'

    with open(restart_file, 'rb') as f:
        restart = pickle.load(f)
    #print(f'Opened {restart_file}', flush=True)

    for i, thread in enumerate(restart):
        top_files = thread.history.tops
        traj_files = thread.history.trajs
        #print(f'Found {len(top_files)} top files and {len(traj_files)} traj files in thread {i}.', flush=True)
        for top in top_files:
            tops.extend(top)
        for traj in traj_files:
            trajs.extend(traj)

    # assert that the number of trajs and tops should be the same
    assert len(tops) == len(trajs), (
        f'Different number of trajectories ({len(trajs)}) and topology files ({len(tops)}).')

    pair_list = list(zip(tops, trajs))

    #print(f'{len(pair_list)} pairs found.\n {pair_list}', flush=True)  # for debugging
    return pair_list

def load_or_create_log_for_traj(log_dir: str):
    '''
    Loads a log file containing information about which trajectories have been processed,
    and which features have been calculated for that trajectory.

    It either opens a log file or creates a new one if one does not exist.

    Parameters
    ---
    log_dir:
        Directory where the log file should be stored.

    Returns
    ---
    log: dict
        From a pickled dictionary. If the log is new, the dictionary will be empty.
        Otherwise, it contains {trajectory filename: [feature list]}
    '''
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / 'trajectory_log.pkl'
    path = Path(log_file)
    #if log doesn't exist yet, create it
    if not path.exists():
        starter_log = {}
        with open(path, 'wb') as f:
            pickle.dump(starter_log, f)
        return starter_log

    #else, return existing log
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_features_for_traj(traj_file: str, log_dict: dict):
    '''
    Opens a log file to check whether a trajectory has been processed and if so, which features remain to be calculated.
    Currently, the log file is a pickled dictionary containing {trajectory file: [list of features]}
    Parameters
    ---
    traj_file: str
       File path to the trajectory to be processed.
    log_dict: dict
        A returned log dictionary from load_or_create_log_for_traj().

    Returns
    ---
    feature_list: list
       Returns a list of features to be calculated. If there is no entry, the full FEATURES list is returned.
       If all features have been calculated, an empty list is returned.
    '''
    #get updated list of features
    feature_list = list(features.FEATURES.keys())

    #search for traj_file
    if traj_file not in log_dict.keys():
        return feature_list
    else:
        finished_features = list(log_dict[traj_file])
        remaining_features = list(set(feature_list) - set(finished_features))
        return remaining_features

def update_log_for_traj(traj_file: str, log_file: str, feature_list: list):
    '''
    Once a feature or multiple features are calculated, update the status for that trajectory.

    Parameters
    ---
    traj_file: str
    Trajectory that is being calculated on/updated.
    log_file: str
    Log file path to be updated.
    feature_list: list
    List of features to be updated.

    Returns
    ---
    Saves updated pickled log file (dict).
    '''
    #load log file
    log_dict = load_or_create_log_for_traj(log_file)

    #create empty entry and list if trajectory has not been processed before
    if traj_file not in log_dict.keys():
        log_dict[traj_file] = []

    log_dict[traj_file].extend(feature_list)

    with open(log_file, 'wb') as f:
        pickle.dump(log_dict, f)















