'''
Unit testing for mdengine.py
'''
from transformd.mdengine import MDEngine, AdaptAmber

#Todo
import mdtraj as md

def test_load_trajectory():
    '''
    Here, I am copying a modified version of the Amber adapter for testing purposes.
    This tests whether an Amber-formatted netcdf trajectory can be loaded.
    '''
    test_top = 'transformd/tests/test_data/nucb_5omt_state1.rst7_WT_tleap.prmtop_0'
    test_traj = 'transformd/tests/test_data/nucb_5omt_state1_WT_short.nc'

    top = md.load_prmtop(test_top)
    assert isinstance(top, md.Topology), f'Topology {test_top} could not be loaded as Amber topology object'
    traj = md.load(test_traj, top=top, stride=1)
    assert isinstance(traj, md.Trajectory), f'Trajectory {test_traj} could not be loaded as an Amber trajectory object'
    assert traj.n_frames > 0, f'{test_traj} has fewer than 1 frame'
    assert traj.n_residues > 0, f'{test_traj} has fewer than 1 residue'
    traj_cdf = md.load_netcdf(test_traj, top=top, stride=1)
    assert isinstance(traj_cdf, md.Trajectory), f'Trajectory {traj_cdf} could not be loaded as a netCDF Amber trajectory object'
    assert traj.n_frames > 0, f'NetCDF loading: {test_traj} has fewer than 1 frame'
    assert traj.n_residues > 0, f'NetCDF loading: {test_traj} has fewer than 1 residue'
