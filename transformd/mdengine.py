'''
Area for anything that is MD-engine specific, such as loading trajectories.

New MDEngines can be implemented by constructing a new class that inherits from MDEngine
and implements its abstract methods.
'''
# Todo: add to isEE's MD engine class?

from abc import ABC, abstractmethod
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

    def load_trajectory(self, traj_file: str, top_file: str, stride=1):
        '''
        Returns loaded trajectory.

        Parameters
        ---
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
        try:
            top = md.load_prmtop(top_file)
        except Exception as e:
            print(f'Amber topology {top_file} could not be loaded:{e}')
            return #do I need a return statement? 
        try:
            traj = md.load(traj_file, top=top, stride=stride)
        except Exception as e:
            print(f'Amber trajectory {traj_file} could not be loaded: {e}. Trying NetCDF-specific loader.')

            try:
                traj = md.load_netcdf(traj_file, top=top, stride=stride)
            except Exception as e:
                print(f'Amber trajectory {traj_file} could not be loaded: {e}. Skipping.')
                return None

        #select only protein atoms
        atom_sele = traj.topology.select('protein')
        sele_traj = traj.atom_slice(atom_sele)

        #superpose trajectory
        traj_superpose = md.Trajectory.superpose(sele_traj, sele_traj, frame=0)

        return traj_superpose
