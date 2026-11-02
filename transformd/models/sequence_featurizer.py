'''
Methods for embedding/featurizing protein sequences.
The input is a string (the protein sequence).
The output is a featurized/encoded version of the sequence to be used as input to an ML model.
Dimensions of the featurized sequence are typically some order of (batch size, embedding/feature size, length of input)
'''

import numpy as np
from abc import ABC, abstractmethod
import torch

wt_seq = ''
seq_len = len(wt_seq)

class SequenceFeaturizer(ABC):
    '''
    This class defines how a sequence will be featurized.
    Input: sequence (str)
    Returns: numpy array with dim (seq length, feature_dim)
    This output array can be modified for the appropriate model architecture as needed.
    '''

    @property
    @abstractmethod
    def feature_dim(self):
        pass

    @abstractmethod
    def get_tensor(self, seq: str):
        '''
        Parameters
        ---
        seq: str
            Protein sequence
        Returns
        ---
        featurized_seq: tensor
            Tensor of shape (seq length, feature_dim)
        '''
        pass

class OneHot(SequenceFeaturizer):
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.vocab = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYV')}

    @property
    def feature_dim(self):
        return len(self.vocab)

    def get_tensor(self, seq: str):
        '''
        Returns featurized sequence tensor of dim (seq length, 21) for 20 canonical amino acids + 1 non-canonical
        '''
        tensor = torch.zeros(self.seq_len, self.feature_dim)
        for i, aa in enumerate(seq.upper()):
            if aa in self.vocab:
                tensor[i, self.vocab[aa]] = 1.0
        return tensor

class Physicochemical(SequenceFeaturizer):
    '''
    Uses aaindex1_pca embeddings to imbue physicochemical information to each amino acids.
    Functions as an enrichment to a one-hot type embedding.
    Currently, these embeddings are of dimension (39, 20) to represent 19 PCA components and a 1-hot 20 AA encoding.
    '''
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.vocab = {aa: i for i, aa in enumerate('ARNDCQEGHILKMFPSTWYV')} #must be in this order for aaindex1_pca
        self.embeddings = np.load('data/aaindex1_pca.npy') #dimensions (39, 20)

    @property
    def feature_dim(self):
        return self.embeddings.shape[0]

    def get_tensor(self, seq: str):
        '''
        Returns featurized sequence tensor of dim (seq length, 39)
        '''
        tensor = torch.zeros(self.seq_len, self.feature_dim)
        for i, aa in enumerate(seq.upper()):
            if aa in self.vocab:
                tensor[i] = self.embeddings[:, self.vocab[aa]]

        return tensor



