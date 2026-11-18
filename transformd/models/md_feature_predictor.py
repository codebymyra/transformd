'''
Abstract base class for the shape of models used to predict MD features.

e.g. CNN would be an implementation of this ABC, and would draw upon cnn.py for its functions.
'''

'''
load_data
featurize_data()
convert_to_dataset_object()
get_data_splits

#training
train_model 
save_best_and_worst_model
record_model_parameters

#inference
load_best_model
generate_predictions
plot_predictions 
'''

from transformd.models import sequence_featurizer, cnn
import torch
import torch.nn as nn 
from transformd import factory 
from ray import tune
import matplotlib.pyplot as plt
import os
import ray
from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler

#TODO: how to get seq_len?
#Loading data and getting splits 
#dummy_dict = {'A': 1, 'N': 2, 'T': 3, 'K': 4, 'S': 1, 'P': 6}
dummy_dict = cnn.load_file('/jumbo/burginlab/myra/transformd/transformd/data/hbond_bh_59.pkl')
#print(dummy_dict[0])
print('First loading', type(dummy_dict))
featurizer = factory.sequence_featurizer_factory('physicochemical')
feature_dim = featurizer.feature_dim  
print('Feature dimensions:', feature_dim)
#dataset = cnn.CNNDataset(dummy_dict, featurizer)
train, val, test = cnn.shuffle_and_split(dummy_dict, featurizer, val_split=0.1, test_split=0.1, seed=5)
print(len(train), len(val), len(test))

#todo: where to actually get the sequence length? 

#configure and train model
#model = cnn.ConfigurableCNN(in_channels=feature_dim, seq_len=20, num_layers=1, num_filters=32, kernel_size=3, dense_dim=64, dropout=0.2, filter_scaling="double", pooling_type="flatten")
loss = nn.MSELoss()
#batch_size = 6
#cnn.train(train, val, batch_size, model, loss, lr=0.1, epochs=10, save_path='./best_model_trial.pt')

#HP tune 
search_space = {
    'lr': tune.choice([1e-4, 1e-3, 1e-2]), # 0.0001, 0.001, 0.01
    'batch_size': tune.choice([8, 16, 32, 64]),
    'num_layers': tune.choice([1,2,3,4]),
    'num_filters': tune.choice([16, 32, 64, 128]),
    'kernel_size': tune.choice([3, 5, 7]),
    'dense_dim': tune.choice([16, 32, 64, 128]),
    'dropout': tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'filter_scaling': tune.choice(['double', 'constant']),
    'pooling_type': tune.choice(['flatten', 'max_pool']),
    'epochs': 100 
}

tuner = tune.Tuner(cnn.trainable(train, val, loss, feature_dim, 109), 
                   param_space=search_space,
                   tune_config=tune.TuneConfig(
                       num_samples=20,
                       scheduler=ASHAScheduler(metric='val_loss', mode='min')
                   ),
                   run_config=tune.RunConfig(
                       storage_path=os.path.abspath('/jumbo/burginlab/myra/09_transformer/flip2_isEE/nucb/5omt/6.ml-testing')
                   ),
                   )
results = tuner.fit()

#plot tuner results
fig, ax = plt.subplots()
for result in results:
    df = result.metrics_dataframe
    if df is not None:
        ax.plot(df['training_iteration'], df['val_loss'], alpha=0.6)
ax.set_xlabel('epoch')
ax.set_ylabel('validation loss')
plt.show()




#load model best model
#inference and plotting 

