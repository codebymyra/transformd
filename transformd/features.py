"""
    Functions for calculating raw MD features.
    Here, I return time-averaged data.

    Parameters
    ---
    traj: mdtraj.Trajectory
        Loaded trajectory object.

    Returns
    ---
    time_average_arr: np.ndarray
        Feature data averaged over frames.

    Example
    ---
    (n_res, n_frames) -> (n_res,)
    (n_frames,) -> (1,)
    (n_res,) -> (n_res,)
"""
#Todo: note. previously I performed time-averaging after returning raw data.
#Todo: This was so that I could save raw data if desired.
#However, implementation was tricky, so I have switched to using time-average data for now.

import mdtraj as md
import numpy as np

#Pre-processing
def get_traj_info(traj):
    '''
    Returns trajectory information including:
    - number of frames
    - number of residues
    - FASTA sequence

    Parameters
    ---
    traj: Trajectory
        Loaded trajectory object

    Returns
    ---
    n_frames: int, n_residues: int, seq: str
    '''

    n_frames = traj.n_frames
    n_residues = traj.n_residues
    seq = traj.topology.to_fasta()

    return n_frames, n_residues, seq

#Registered methods

#Todo: not sure where to put this but need a way to keep track of MD features
FEATURES = {} #dict for storing current feature names
def feature(func):
    #record names of methods with "feature" decorator.
    FEATURES[func.__name__] = func
    return func

@feature
def rmsf_ca(traj):
    '''
    Returns ndarray (n_residues,) with each RMSF of the alpha carbon of each residue.
    '''
    ca_indices = traj.topology.select('protein and name CA')
    traj_align = md.Trajectory.superpose(traj, traj, frame=0, atom_indices=ca_indices)
    ca_rmsf = md.rmsf(traj_align, traj_align, frame=0, atom_indices=ca_indices)
    return ca_rmsf #(n_res,)

@feature
def sasa_per_residue(traj):
    '''
    Returns ndarray (n_frames, n_residues)
    with the solvent accessible area (shrake-rupley) for each residue of each frame.
    '''
    sasa = md.shrake_rupley(traj, probe_radius=0.14,mode='residue')
    return np.mean(sasa, axis=0)

@feature
def hbond_wn(traj):
    '''
    Returns ndarray (n_frames, n_residues)
    with a count of the number of hydrogen bonds based on Wernet Nilsson cutoffs
    '''
    n_frames = traj.n_frames
    n_res = traj.n_residues
    hbond_count = np.zeros((n_frames, n_res))

    wn_list = md.wernet_nilsson(traj) #returns list of len n_frames
    #each item in list is an array with dimensions (n_hbonds, 3)
    #this takes a while to calculate

    for frame_idx, bonds in enumerate(wn_list):
        if len(bonds) == 0:
            continue
        donor_atoms = bonds[:, 0]
        acceptor_atoms = bonds[:, 2]

        donors = [traj.topology.atom(d).residue.index for d in donor_atoms]
        acceptors = [traj.topology.atom(a).residue.index for a in acceptor_atoms]

        np.add.at(hbond_count[frame_idx], donors, 1)
        np.add.at(hbond_count[frame_idx], acceptors, 1)

    return np.mean(hbond_count, axis=0) #(n_frames, n_res)

@feature
def hbond_ks(traj, energy_threshold=-0.5):
    '''Compute number of Kabsch-Sander hbonds each residue participates in at each frame.
    Returns: numpy ndarray of shape (n_frames, n_res)
    '''
    n_frames = traj.n_frames
    n_res = traj.n_residues
    hbond_count = np.zeros((n_frames, n_res))

    ks_matrices = md.kabsch_sander(traj) #returns n_frames list of sparse matrices with dimensions (n_res, n_res)

    for frame_idx, matrix in enumerate(ks_matrices):
        #convert to coo format to access data
        coo = matrix.tocoo()

        #check whether hbond energy is below threshold (it should be)
        active_bonds = coo.data < energy_threshold

        donors = coo.row[active_bonds]
        acceptors = coo.col[active_bonds]

        np.add.at(hbond_count[frame_idx], donors, 1)
        np.add.at(hbond_count[frame_idx], acceptors, 1)

    return np.mean(hbond_count, axis=0) #(n_frames, n_res)


def hbond_bh(traj):
    '''
    Compute number of Baker-Hubbard hydrogen bonds each residue participates in over the entire trajectory
    Returns numpy ndarray of shape (n_res,)
    '''
    n_res = traj.n_residues
    hbond_count = np.zeros((n_res, 2)) #for storing counts
    #set residue numbers from zero in left column
    for i in range(n_res):
        hbond_count[i,0] = i

    hbonds_bh = md.baker_hubbard(traj)

    for hbond in hbonds_bh:
        res_donor = traj.topology.atom(hbond[0]).residue.index
        res_acceptor = traj.topology.atom(hbond[2]).residue.index
        #add residues to count
        #Todo: Perhaps take an average rather than summation?
        hbond_count[res_donor, 1] += 1
        hbond_count[res_acceptor, 1] += 1

    by_res_hbond_bh = hbond_count[:,1]  #take only count values

    return by_res_hbond_bh # (n_res,)



def residue_contacts(traj, cutoff=0.6, scheme='ca'):
    '''
    Compute number of heavy atom contacts between pairs at each frame
    Note: this one takes a long time to compute.
    Returns:
    nparray of shape (n_frames, n_res)
    '''
    n_frames = traj.n_frames
    n_res = traj.n_residues
    contact_count = np.zeros((n_frames, n_res))

    #create array of pairs separated by 3 residues (arbitrary)
    pairs = []
    for i in range(n_res):
        for j in range(i+3, n_res):
            pairs.append([i,j])
    pairs = np.array(pairs)

    if len(pairs) == 0: #no residues
        return contact_count

    contact_dist = md.compute_contacts(traj, contacts=pairs, scheme=scheme)[0]

    contacts = contact_dist < cutoff

    #map contacts to individual residues
    for idx, (res1, res2) in enumerate(pairs):
        active_frames = contacts[:, idx]
        contact_count[active_frames, res1] += 1
        contact_count[active_frames, res2] += 1

    return np.mean(contact_count, axis=0) # n_frames, n_residues

def res_rmsd(traj):
    '''
    Compute RMSD of each residue in each frame compared to the first frame
    Returns numpy ndarray of shape (n_frames, n_res)
    '''
    superpose_traj = traj.superpose(traj, 0)

    n_frames = superpose_traj.n_frames
    n_res = superpose_traj.n_residues
    res_rmsds = np.zeros((n_frames, n_res))

    for i in range(n_res):
        atom_indices = superpose_traj.topology.select(f'resid {i} and not element H')
        if len(atom_indices) == 0:
            continue

        res_rmsds[:, i] = md.rmsd(superpose_traj, superpose_traj, frame=0, atom_indices=atom_indices)

    return np.mean(res_rmsds, axis=0) #(n_frames, n_res)

def res_cos_psi(traj):
    '''
    Compute cosine of psi angle.
     Both sine and cosine are calculated to avoid large jumps in angle degrees.
     Returns numpy array of shape (n_frames, n_res)
    '''
    n_frames = traj.n_frames
    n_res = traj.n_residues
    indices, angles = md.compute_psi(traj) #indices: (n_res, 4) angles: (n_frames, n_res)

    #take cosine and sine of angles to create continuous and circular labels
    cos_angles = np.cos(angles) #(n_res,)
    cos_psi = np.zeros((n_frames, n_res))

    for idx, atom_quad in enumerate(indices):
        #for psi angle, Ni - CAi - Ci - Ni+1
        res_idx = traj.topology.atom(atom_quad[0]).residue.index
        #slice across residues and add values for each frame
        cos_psi[:, res_idx] = cos_angles[:, idx]

    return np.mean(cos_psi, axis=0)

def res_sin_psi(traj):
    '''
    Compute sine of psi angle.
     Both sine and cosine are calculated to avoid large jumps in angle degrees.
     Returns numpy array of shape (n_frames, n_res)
    '''
    n_frames = traj.n_frames
    n_res = traj.n_residues
    indices, angles = md.compute_psi(traj) #indices: (n_res, 4) angles: (n_frames, n_res)

    #take cosine and sine of angles to create continuous and circular labels
    sin_angles = np.sin(angles)
    sin_psi = np.zeros((n_frames, n_res))

    for idx, atom_quad in enumerate(indices):
        #for psi angle, Ni - CAi - Ci - Ni+1
        res_idx = traj.topology.atom(atom_quad[0]).residue.index
        #slice across residues and add values for each frame
        sin_psi[:, res_idx] = sin_angles[:, idx]

    return np.mean(sin_psi, axis=0)

def res_cos_phi(traj):
    '''
    Comput cosine of phi angle
    Both sine and cosine are calculated to avoid large jumps in angle degrees.
    Returns: ndarray of shape (n_frames, n_res)
    '''
    n_frames = traj.n_frames
    n_res = traj.n_residues
    indices, angles = md.compute_phi(traj)
    cos_angles = np.cos(angles)
    cos_phi = np.zeros((n_frames, n_res))

    for idx, atom_quad in enumerate(indices):
        #for phi angle, Ci-1 - Ni - CAi - Ci
        res_idx = traj.topology.atom(atom_quad[1]).residue.index
        cos_phi[:, res_idx] = cos_angles[:, idx]

    return np.mean(cos_phi, axis=0)

def res_sin_phi(traj):
    '''
    Comput cosine of phi angle
    Returns: ndarray of shape (n_frames, n_res)
    '''
    #Todo: address the fact that the first or last residue for angles is going to be 0.
    n_frames = traj.n_frames
    n_res = traj.n_residues
    indices, angles = md.compute_phi(traj)
    sin_angles = np.sin(angles)
    sin_phi = np.zeros((n_frames, n_res))

    for idx, atom_quad in enumerate(indices):
        #for phi angle, Ci-1 - Ni - CAi - Ci
        res_idx = traj.topology.atom(atom_quad[1]).residue.index
        sin_phi[:, res_idx] = sin_angles[:, idx]

    return np.mean(sin_phi, axis=0)

def res_cos_chi1(traj):
    '''
    Compute cosine of Chi1 side chain angle
    Both sine and cosine are calculated to avoid large jumps in angle degrees (0 degrees and 360 are the same)
    Returns: ndarray of shape (n_frames, n_res)
    '''
    indices, angles = md.compute_chi1(traj)
    cos_angles = np.cos(angles)

    #create empty arrays for data
    cos_chi1 = np.zeros((traj.n_frames, traj.n_residues))

    for idx, atom_quad in enumerate(indices):
        #for chi1, N - Ca - Cb - Cg
        res_idx = traj.topology.atom(atom_quad[1]).residue.index
        cos_chi1[:, res_idx] = cos_angles[:, idx]

    return np.mean(cos_chi1, axis=0)

def res_sin_chi1(traj):
    '''
    Compute sine of Chi1 side chain angle
    Returns: two ndarrays of shape (n_frames, n_res)
    '''
    indices, angles = md.compute_chi1(traj)
    sin_angles = np.sin(angles)

    #create empty arrays for data
    sin_chi1 = np.zeros((traj.n_frames, traj.n_residues))

    for idx, atom_quad in enumerate(indices):
        #for chi1, N - Ca - Cb - Cg
        res_idx = traj.topology.atom(atom_quad[1]).residue.index
        sin_chi1[:, res_idx] = sin_angles[:, idx]

    return np.mean(sin_chi1, axis=0)

def rg(traj):
    '''
    Returns ndarray (n_frames,) with radius of gyration values for each from.
    '''
    rg = md.compute_rg(traj)
    return np.array([np.mean(rg)])
