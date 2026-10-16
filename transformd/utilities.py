'''
utilities.py

Misc. utility functions
'''
from pathlib import Path
import pickle
from isee_framework.main import Thread

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

def get_trajs_in_working_dir(wd_path: Path):
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






def title_case(sentence):
    '''
    Convert a string to title case.
    Title case means that the first letter of each word is capitalized, and all other letters in the word are lowercase. 

    Parameters
    ____
    sentence: str
        String to be converted to title case
    Returns
    _____
    ret: str
        String converted to title case
    Example
    _____
    >>title_case('ThIS is A sTRiNg to BE ConvERTed.')
    'This Is A String To Be Converted.'
    '''

    ret = sentence[0].upper()

    # Loop through the rest of the characters
    for i in range(1, len(sentence)):
        if sentence[i - 1] == ' ':
            ret = ret + sentence[i].upper()
        else:
            ret = ret + sentence[i].lower()

    return ret
