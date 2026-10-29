'''
Methods for embedding/featurizing protein sequences.
The input is a string (the protein sequence).
The output is a featurized/encoded version of the sequence to be used as input to an ML model.
'''

import numpy as np

def aaindex1_pca(seq_array: np.ndarray):
    '''
    Use the aaindex1_pca matrix to encode the protein sequence.
    https://www.pnas.org/doi/10.1073/pnas.2104878118

    Parameters
    ---
    seq_array: np.array
        Array of protein sequences of dim (n_sequences,)

    Returns
    ---






        pytorch 1d cnn intput shape: (N, Cin, Lin) or (Cin, Lin)
    '''
    pass