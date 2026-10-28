'''
Area for anything that is MD-engine specific, such as loading trajectories.

New MDEngines can be implemented by constructing a new class that inherits from MDEngine
and implements its abstract methods.
'''
# Todo: add to isEE's MD engine class?

from abc import ABC, abstractmethod
from pathlib import Path

import mdtraj as md

class MDEngine(ABC):
    '''
    Abstract class for molecular dynamics engines.

    Implements methods for all MD-engine specific tasks.
    '''

    @abstractmethod
    def load_trajectory(self, traj_file, top_file):
        '''
        Return a loaded trajectory based on the input and output files.

        Parameters
        ---
        traj_file: str
            Name of trajectory filepath.
        top_file: str
            Name of topology filepath.

        Returns
        ---
        traj: mdtra.Trajectory
            Loaded trajectory object.
        '''
        pass

    def select_frames(self, first_frame: int, last_frame: int):
        '''
        Cuts loaded trajectory into a chunk of certain frames.

        Parameters
        ___
        traj: Trajectory
            Loaded trajectory object

        Returns
        ---
        traj_chunk: Trajectory
            Part of a loaded trajectory
        '''
        # Todo: implement this in a way that takes in user settings
        #traj = load_trajectory(self)
        #Then, cut trajectory based on frist and last frame
        #Finally, return truncated trajectory object
        pass

class AdaptAmber(MDEngine):
    '''
    Adapter class for Amber MDEngine.
    '''

    def load_trajectory(self, working_dir: str, traj_file: str, top_file: str, stride=1):
        '''
        Returns loaded trajectory.

        Parameters
        ---
        working_dir: str
            The working directory containing the trajectory files
        traj_file: str
            Name of trajectory filepath.
        top_file: str
            Name of topology filepath.
        stride: int
            Load every nth frame depending on user input settings.

        Returns
        ---
        traj: mdtraj.Trajectory
        Loaded trajectory object. This trajectory will already be superposed to the 0th frame.
        '''
        top_path = Path(working_dir) / top_file
        traj_path = Path(working_dir) / traj_file
        if not top_path.exists() and not traj_path.exists():
            print(f'Neither topology nor trajectory files exist ({top_path}, {traj_path}')
        elif not top_path.exists() and traj_path.exists():
            print(f"Trajectory file {traj_path} exists, but topology {top_path} does not.")
        elif not traj_path.exists() and top_path.exists():
            print(f'Topology file {top_path} exists but trajectory file does not {traj_path}')
        else:
            try:
                top = md.load_prmtop(top_path)
            except Exception as e:
                print(f'Amber topology {top_path} could not be loaded:{e}')
                return None#todo: o I need a return statement?
            try:
                traj = md.load(traj_path, top=top, stride=stride)
            except Exception as e:
                print(f'Amber trajectory {traj_path} could not be loaded: {e}. Trying NetCDF-specific loader.')

                try:
                    traj = md.load_netcdf(traj_path, top=top, stride=stride)
                except Exception as e:
                    print(f'Amber trajectory {traj_path} could not be loaded: {e}. Skipping.')
                    return None

        #select only protein atoms
        atom_sele = traj.topology.select('protein')
        sele_traj = traj.atom_slice(atom_sele)

        #superpose trajectory
        traj_superpose = md.Trajectory.superpose(sele_traj, sele_traj, frame=0)

        return traj_superpose
