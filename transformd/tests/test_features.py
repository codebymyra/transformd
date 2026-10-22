'''
Unit testing for features.py
'''
import numpy as np
from transformd import features
import mdtraj as md
import pytest
from transformd import mdengine

#todo: load a trajectory for testing

def test_get_traj_info():
    topfile = 'transformd/tests/test_data/nucb_5omt_state1.rst7_WT_tleap.prmtop_0'
    trajfile = 'transformd/tests/test_data/nucb_5omt_state1_WT_short.nc'

    top = md.load_prmtop(topfile)
    traj = md.load(trajfile, top=top)

    n_frames, n_residues, seq = features.get_traj_info(traj) #should return n_frames, n_residues, seq
    assert isinstance(n_frames, int)
    assert n_frames > 0
    assert isinstance(n_residues, int)
    assert n_residues > 0
    assert isinstance(seq, str)
    assert len(seq) > 0

def test_feature():
    '''
    Test that feature returns a feature list based on feature decorator.
    '''
    feature_dict = features.FEATURES
    assert isinstance(feature_dict, dict)
    feature_list = list(feature_dict.keys())
    assert len(feature_list) > 0
    assert 'rmsf_ca' in feature_list

def test_rmsf_ca():
    '''
    Test that at least one feature calculator returns the expected array.
    '''
    #todo: link this to the loading trajectory function I defined in mdengine
    #todo: make sure that when I save values for ML, I only take the first 5 decimals or so because weird errors happen after that
    topfile = 'transformd/tests/test_data/nucb_5omt_state1.rst7_WT_tleap.prmtop_0'
    trajfile = 'transformd/tests/test_data/nucb_5omt_state1_WT_short.nc'

    top = md.load_prmtop(topfile)
    traj = md.load(trajfile, top=top)

    arr = features.FEATURES['rmsf_ca'](traj)
    assert arr.dtype == np.float32
    assert arr.shape == (109,)
    assert arr[0] == pytest.approx(0.12114416, rel=1e-5)
