'''
utilities.py

Misc. utility functions
'''
from pathlib import Path
import pickle
from isee_framework.main import Thread
import features
#from transformd import features

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

#todo: right now assumes corresponding file order between topology and trajectory files in restart.pkl. This will break if this isn't the case
def get_trajs_in_working_dir(working_dir: str):
    '''
    Takes in the wd_path from get_working_dir().
    Return a list of trajectory:topology file pairs based on the restart.pkl file in the working directory.
    Note: this assumes the order of topology and trajectory files in restart.pkl correspond with each other.
    This might not be the case and in that case the code will break.

    Parameters
    ___
    working_dir: str
        String of working directory path. To be converted to Path object.
    Returns
    ___
    file_dict: dict
        Dictionary object with keys: topology files, values: corresponding trajectories.

    Example
    ___
    >>get_working_dir('working')
    {'nucb_5omt_state1.rst7_WT_tleap.prmtop': ['nucb_5omt_state1.rst7_WT_0.nc']}
    '''


    tops = []
    trajs = []
    wd_path = get_working_dir(working_dir)
    restart_file = wd_path / 'restart.pkl' #specific to isEE

    with open(restart_file, 'rb') as f:
        restart = pickle.load(f)
    #print(f'Opened {restart_file}', flush=True)

    pairs = []
    for thread in restart:
        traj_files = thread.history.trajs #nested lists of trajectories corresponding to base topology
        top_files = thread.history.tops
        assert len(traj_files) == len(top_files), f'Number of topology files ({len(top_files)}) does not match number of trajectory files ({len(traj_files)})'

        #todo: probably should add a try/except block for cases when trajectories are missing
        for top_idx, top in enumerate(top_files):
            associated_trajs = traj_files[top_idx] #e.g. [protein_0.nc, protein_1.nc]
            for traj_idx, traj in enumerate(associated_trajs):
                top_name = f'{top}_{traj_idx}'
                pairs.append((top_name, traj))

    return pairs


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

def update_log_for_traj(traj_file: str, log_dir: str, feature_list: list):
    '''
    Once a feature or multiple features are calculated, update the status for that trajectory.

    Parameters
    ---
    traj_file: str
    Trajectory that is being calculated on/updated.
    log_dir: str
    Directory where the log file is located.
    feature_list: list
    List of features to be updated.

    Returns
    ---
    Saves updated pickled log file (dict).
    '''
    #load log file
    log_dict = load_or_create_log_for_traj(log_dir)

    #create empty entry and list if trajectory has not been processed before
    if traj_file not in log_dict.keys():
        log_dict[traj_file] = []

    log_dict[traj_file].extend(feature_list)

    log_file = Path(log_dir) / 'trajectory_log.pkl' #todo: make this less repeated and breakable
    with open(log_file, 'wb') as f:
        pickle.dump(log_dict, f)


#todo: move all pickle loading adn saving to utilities, instead of in Database and Featurizer?











