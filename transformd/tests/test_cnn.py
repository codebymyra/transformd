'''
Unit testing for CNN module 
'''

import torch
from models import cnn
from models import sequence_featurizer


def test_ConfigurableCNN():
    model1 = cnn.ConfigurableCNN(in_channels=39, seq_len=20, num_layers=1, num_filters=32, kernel_size=3, dense_dim=64, dropout=0.2, filter_scaling="double", pooling_type="max_pool")
    model2 = cnn.ConfigurableCNN(in_channels=39, seq_len=20, num_layers=1, num_filters=32, kernel_size=3, dense_dim=64, dropout=0.2, filter_scaling="constant", pooling_type="flatten")
    test_input = torch.randn(4, 4, 39)
    prediction1 = model1(test_input)
    assert prediction1.shape == torch.Size([4])
    prediction2 = model2(test_input)
    assert prediction2.shape == torch.Size([4])