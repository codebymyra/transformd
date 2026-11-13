'''
Test sequence featurizer methods.
'''

from models import sequence_featurizer
import torch

def test_one_hot():
    seq = 'ARNDCQEGHILKMFPSTWYV'
    transform = sequence_featurizer.OneHot()
    one_hot_tensor = transform(seq)
    assert one_hot_tensor.shape == torch.Size([20, 20])
    assert one_hot_tensor[0][0] == 1
    #assert one_hot_tensor == torch.eye(20)

def test_physicochemical():
    seq = 'ASDF'
    transform = sequence_featurizer.Physicochemical()
    pchem = transform(seq)
    assert pchem.shape == torch.Size([4, 39])
    assert pchem[0][19] == torch.tensor(1.0)
    #assert pchem[0][0] == torch.tensor(0.2291) #not sure why this doesn't work
   