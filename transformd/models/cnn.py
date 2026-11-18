'''
Hyperparameter tuning and training for by-feature CNNs.
1-D convolutional neural network

Contains all the things needed for a CNN but actual invocation exists in an ABC
'''

from functools import partial 
import pickle
import os 
import tempfile
import numpy as np
import torch
from torch.utils.data import random_split, Dataset, DataLoader 
import torch.nn as nn
import torch.optim as optim 
from pathlib import Path
from transformd.models import sequence_featurizer
from transformd import factory
#from . import sequence_featurizer #need to do this for test package. How to do relative file paths? 

# New: imports for Ray Tune
import ray
from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler


#get_datasets (train, test, val)
#data now: individual pickle files with dict format: {seq: label}
#pytorch 1d cnn intput shape: (N, Cin, Lin) or (Cin, Lin) where N = batch size, C = number of channels (embedding dim for a given sequence?), L = length of sequence
#invoke factory that invokes appropriate sequence featurizer.
#save some kind of log with model parameters, which features were run, time, etc.



class CNNDataset(Dataset):
   '''
   Create a Dataset class. PyTorch loaders expect a Dataset object as input with __getitem__ and __len__ methods.
   This specific class takes in data in the form of a dictionary, with examples: labels.

   Parameters
   ---
   data: dict
      example - label data where the keys are sequences and values are labels
   featurize: Sequence Featurizer of choice. This should be a Sequence Featurizer object. 
   '''
   def __init__(self, data: dict, featurize):
      self.examples = list(data.keys())
      self.labels = list(data.values())
      self.featurize = featurize #pass an instance of a class 
      print(self.featurize, type(self.featurize))

   def __len__(self):
      return len(self.examples)

   def __getitem__(self, index):
      '''
      Returns a single example and corresponding label from the dataset.
      PyTorch expects these to be tensors.
      '''
      example = self.examples[index]
      example = self.featurize(example)
      label = torch.tensor(self.labels[index], dtype=torch.float32)

      return example, label


#featurizer = sequence_featurizer.Physicochemical()
#test_dict = {'A': 1, 'N': 2, 'T': 3, 'K': 4}
#test_cnn = CNNDataset(test_dict, featurizer)
#print(test_cnn.__getitem__(0))


def load_file(filepath: str):
   '''
   Load data file.
   Supports pickle and numpy objects saved with .pkl or .npy file extensions.
   Example of a data file name (by-residue): 'rmsf_ca_100.pkl'

   Parameters
   ---
   filepath (str):
      Location of the data file

   Returns
   ---
   data: dict or np array
      Returns data in the form of a dictionary, or a numpy array, depending on the input
   file.stem: str
      Here, this is the stem of the file, which is the equivalent of the feature name e.g. rmsf_ca_100
   '''
   file = Path(filepath)
   if file.exists():
      if filepath.endswith('.pkl'):
         with open(file, 'rb') as f:
            data = pickle.load(f)
      elif filepath.endswith('.npy'):
         data = np.load(file)
      else:
         print(f'File extension not recognized as .pkl or .npy: {file}', flush=True)
         return None
      return data
   else:
      print(f'No file found at {filepath}')
      return None


def shuffle_and_split(data: dict, featurize: str, val_split: int, test_split: int, seed: int):
   '''
   Given a dataset, randomly shuffle based on the split percentage and save out train, validation, and test sets.
   Datasets are saved as torch tensors.
   This is only done once per model training.

   Parameters
   ---
   data: dict
      Dictionary of sequence-label pairs.
      Will be converted to a Torch Dataset object.
   featurize: str
      Type of featurization from sequence_featurizer. 
   val_split: int
      Percent that should be reserved for validation data.
   test_split: int
      Percent that should be reserved for test data.
   seed: int
      Fixed seed for generating train/test splits

   Returns
   ---
   train_ds: torch tensor
   val_ds: torch tensor
   test_ds: torch tensor
      Saves train/val/test splits from input data. Sequence information is featurized here.
   '''
   #create Torch Dataset object using incoming data
   torch_data = CNNDataset(data, featurize)

   #split into train, validation, and test sets 
   generator = torch.Generator().manual_seed(seed) #todo: change to random eventually
   train_ds, val_ds, test_ds = random_split(torch_data, [1 - val_split - test_split, val_split, test_split], generator=generator)

   #train_data = DataLoader(train_ds, batch_size, shuffle=True)
   #val_data = DataLoader(val_ds, batch_size, shuffle=True)
   #test_data = DataLoader(test_ds, batch_size, shuffle=True)

   #I also probably need to get the embedding dimension here 
   #return as (batch, seq_len, features)
   return train_ds, val_ds, test_ds

class SimpleCNN(nn.Module):
   '''
   Set a simple CNN for testing
   '''
   def __init__(self, in_channels):
      super().__init__()
      self.features = nn.Sequential(
         nn.Conv1d(in_channels, out_channels=64, kernel_size=2, stride=1), nn.LeakyReLU(),
         nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.LeakyReLU(),
      )

      self.head = nn.Sequential(
         nn.Flatten(),
         nn.LazyLinear(50),
         nn.LeakyReLU(),
         nn.Dropout(0.2),
         nn.Linear(50, 1),
      )

   def forward(self, x):
      x = x.permute(0,2,1)
      x = self.features(x)
      out = self.head(x)
      return out.squeeze(-1)

#test without changing model weights
#model = SimpleCNN(in_channels=39)
#test_input = torch.randn(4, 20, 39) 
#predictions = model(test_input)
#print("Output shape:", predictions.shape)
#print("Predictions:", predictions)

#pytorch 1d cnn intput shape: (N, Cin, Lin) or (Cin, Lin) where N = batch size, C = number of channels (embedding dim for a given sequence?), L = length of sequence
class ConfigurableCNN(nn.Module):
   '''
   1D CNN for processing sequences. 
   
   Parameters
   ---
   Pre-featurized sequence tensors of shape (batch, length, feature_dim). This class will convert to the expected (batch, feature_dim, length).
   '''
   def __init__(self,
                in_channels: int, #from featurizer.feature_dim
                seq_len: int, #seq length
                num_layers: int, #1, 2, 3, 4
                num_filters: int, #Num features. 16, 32, 64, 128
                kernel_size: int, #Scanning window. 3, 5, 7
                dense_dim: int, #16, 32, 64, 128
                dropout: float, #0 to 0.5 increments of 0.1
                filter_scaling: str="double", #double filter or constant e.g. 32>64>128 or 32>32>32
                pooling_type: str = "flatten", #max pooling vs. flatten. Will require different math for input to dense layer
   ):
      super().__init__()
      self.pooling_type = pooling_type 

      #convoltuion layers
      #use ModuleList to create flexible number of conv1d layers
      self.conv_layers = nn.ModuleList()
      #self.bn_layers = nn.ModuleList()
      current_in_channels = in_channels #start with embedding feature_dim, then change to num_filters

      for i in range(num_layers):
         if filter_scaling == "double":
            print('doudbles')
            current_out = num_filters * (2 ** i)
         else:
            current_out = num_filters

         self.conv_layers.append(
            nn.Sequential(
               nn.Conv1d(
                  in_channels=current_in_channels,
                  out_channels=current_out,
                  kernel_size=kernel_size,
                  #padding?
               ),
               #batch norm? 
               #nn.BatchNorm1d(current_out)
               nn.LeakyReLU()
            ) 
         )
         #change next layer input to previous layer output
         current_in_channels = current_out

      #get final number of channels 
      final_channels = current_in_channels 

      #Pooling vs. flattening 
      if self.pooling_type == "max_pool":
         self.pool = nn.AdaptiveMaxPool1d(1) #figures out kernel size and stride on its own, collapses to 1
         linear_features = final_channels #num features for final dense layer
         #print('Max pool:', linear_features)
      elif self.pooling_type == "flatten":
         self.pool = nn.Identity()
         linear_features = final_channels * seq_len
         #print('Flatten:', linear_features)
      

      #Dense layer 
      self.head = nn.Sequential(
         nn.Flatten(),
         nn.LazyLinear(dense_dim), #automatically calculates the correct input dim 
         nn.LeakyReLU(), #hard coded for now, could change
         nn.Dropout(dropout),
         nn.Linear(dense_dim, 1) #linear output 1 value 
      )

   def forward(self, x: torch.Tensor): #actually passes data through model
      #permute data from (batch, seq_len, features) to (batch, features, seq_len)
      x = x.permute(0, 2, 1) #(batch, features, seq_len)
      for layer in self.conv_layers:
         x = layer(x) #(batch, filter, seq_len)

      x = self.pool(x) #(batch, filter) or (batch, filter * seq_len)
      out = self.head(x) #(batch, 1)
      return out.squeeze(-1) #(batch,)

#model = ConfigurableCNN(in_channels=39, seq_len=20, num_layers=1, num_filters=32, kernel_size=3, dense_dim=64, dropout=0.2, filter_scaling="double", pooling_type="flatten")
#toy_train= torch.randn(4, 20, 39) 
#toy_test = torch.randn(2, 20, 39)
#predictions = model(test_input)
#print("Output shape:", predictions.shape)
#print("Predictions:", predictions)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #want to test multiple learning rates 
#loss = nn.MSELoss()

def train_one_epoch(dataloader, model, optimizer, loss_fn, DEVICE):
   '''
   Runs one single training epoch. 

   Parameters
   ---
   dataloader: DataLoader
      Tensor DataLoader object containing training data. 
   model: Model class object 
      Example: ConfigurableCNN object 
   optimizer: torch Optimizer object 
      E.g. Adam optimizer 
   loss_fn: torch.nn loss object
      e.g. nn.MSELoss()
   DEVICE: torch device object
      e.g. CUDA if available 

   Returns
   ---
   loss / n : float
      Average loss over the batch where n is batch size 
   '''

   model.train()
   total_loss, n = 0.0, 0
   for x, y in dataloader:
      x, y = x.to(DEVICE), y.to(DEVICE) # x = (batch, seq_len, feature_dim) 
      pred = model(x)
      loss = loss_fn(pred, y)
      optimizer.zero_grad() #clear previous gradients 
      loss.backward() # evaluate how weights contributed to loss
      optimizer.step() #update weights to reduce error 
      total_loss += loss.item() * len(x)  # len(x) is batch size, normalize by batch size 
      n += len(x) # increase total batch size sum 

   return total_loss / n # average loss per sample in a single epoch

@torch.no_grad #disable gradient/weight updating 
def eval_model(dataloader, model, loss_fn, DEVICE):
   '''
   Evaluation method. No weights are updated here. 
   Similar to train, except that the gradient is not reset to zero and weights are not computed. 
   '''
   model.eval()
   loss, n = 0.0, 0 
   for x, y in dataloader:
      x,y = x.to(DEVICE), y.to(DEVICE)
      pred = model(x)
      loss += loss_fn(pred, y).item() * len(x) # return total loss since loss_fn returns mean 
      n += len(x)

   return loss / n 



def train(train_ds, val_ds, batch_size, model, loss_fn, lr, epochs, save_path: str, DEVICE=None):
   '''
   Load model and train.
   Calculate loss and readjust weights based on loss. 
   Uses the train_one_epoch and eval_model methods above. 

   Parameters
   ---
   train_ds: Dataset
      Training split as Dataset object
   val_ds: Dataset
      Validation split as Dataset object 
   batch_size: int
      Batch size of sequences for a given training round  
   model: model class object
      E.g. ConfigurableCNN model 
   loss_fn: loss object
      E.g. torch.MSELoss
   lr: float
      Learning rate 
   epochs: int
      Max number of epochs 
   save_path: str
      Location to save the best model 
   DEVICE: str
      Cuda or cpu 

   Returns
   ---
   history: model.pt 
      Loadable pytorch model weights. These are weights from the lowest loss in training. 
   '''
   DEVICE = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(DEVICE)

   train_loader = DataLoader(train_ds, batch_size, shuffle=True)
   val_loader = DataLoader(val_ds, batch_size, shuffle=True)

   #initialize weights for LazyLinear before creating optimizer 
   #model.eval()
   #with torch.no_grad():
      #initial_weights, _ = next(iter(val_loader))
      #model(initial_weights.to(device))

   optimizer = torch.optim.Adam(model.parameters(), lr=lr) #hard code adam optimizer 

   history = [] #keep track of training history 
   best_val = float("inf") #infinity to start with loss 
   #todo: how to implement early stopping? 
   for epoch in range(epochs):
      print('Train loader:', type(train_loader))
      train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
      val_loss = eval_model(val_loader, model, loss_fn, DEVICE)
      history.append((train_loss, val_loss))

      print(f'Epoch: {epoch + 1}')
      print(f'Training loss: {train_loss:.4f}, val loss: {val_loss:.4f}')

      if val_loss < best_val:
         best_val = val_loss
         torch.save(model.state_dict(), save_path)

   return history 

#----Ray Tune Implementation----#
def train_tune(config, train_ds, val_ds, in_channels, seq_len, loss_fn):
   '''
   Hyperparameter optimization using Ray Tune.
   This method passes in the hyperparameters to be tuned from the config dict. 

   Parameters
   ---
   config: dict
      Variables to configure within the model
   train_ds: Dataset
      Training data represented as a Dataset object post-featurization
   val_ds: Dataset
      Validation data represented as a Dataset object 
   in_channels: int
      Number of features used to represent the input (calculated automatically during featurization as feature_dim)
   seq_len: int
      Size of the input sequence 
   loss_fn: nn.loss_function
      Loss function

   Returns
   ---
   Checkpoint files for the best model found. 
   '''
   device = "cuda" if torch.cuda.is_available() else "cpu" #maybe pass in explicit DEVICE variable? 

   train_loader = DataLoader(train_ds, config['batch_size'], shuffle=True)
   val_loader = DataLoader(val_ds, config['batch_size'], shuffle=True)

   #instantiate model
   model = ConfigurableCNN(
      in_channels=in_channels,
      seq_len=seq_len,
      num_layers=config['num_layers'],
      num_filters=config['num_filters'],
      kernel_size=config['kernel_size'],
      dense_dim=config['dense_dim'],
      dropout=config['dropout'],
      filter_scaling=config['filter_scaling'],
      pooling_type=config['pooling_type']
   ).to(device)

   optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
   loss_fn = loss_fn

   #load a checkpoint if it exists, start from the recorded epoch
   start_epoch = 0
   checkpoint = tune.get_checkpoint()
   if checkpoint:
      with checkpoint.as_directory() as d:
         state = torch.load(os.path.join(d, 'state.pt'), map_location=device)
      model.load_state_dict(state['model'])
      optimizer.load_state_dict(state['optimizer']) 
      start_epoch = state['epoch'] + 1
   
   #run training and validation
   for epoch in range(start_epoch, config['epochs']):
      train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
      val_loss = eval_model(val_loader, model, loss_fn, device)

      # save a temporary file 
      with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
         torch.save(
            {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}, 
             os.path.join(temp_checkpoint_dir, 'state.pt'),
         )
         tune.report(
            {'train_loss': train_loss, 'val_loss': val_loss},
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
         )

def trainable(train_ds, val_ds, loss_fn, in_channels, seq_len, cpus=1, gpus_per_trial=0):
   '''
   Executes a single hyperparameter configuration using Ray Tune. 
   Passes all of the parameters that are constant between trials.
   Wrapping the trainable with_parameters allows passing of training data and other parameters that are constant between trials. 

   Parameters
   ---
   config: dict
      Dictionary with parameters for Ray Tune to test. Batch size is used here.
   train_ds: Dataset 
      Train split as Dataset object. This is turned into a DataLoader in the training function. 
   val_ds: Dataset
      Validation split as Dataset object. This is turned into a DataLoader in the training function.
   loss_fn: loss function
      nn loss function
   in_channels: int
      Feature dimension of sequence
   seq_len: int
      Length of sequence
   cpus: int
      Number of CPUs allocated 
   gpus_per_trial: float
      Allocated GPUs

   Returns
   ---
   a trainable object that outilnes what resources and data are passed into a training method.
   '''

   return tune.with_resources(
      tune.with_parameters(
         train_tune,
         train_ds = train_ds, val_ds = val_ds,
         loss_fn = loss_fn, 
         in_channels = in_channels, seq_len = seq_len,
      ),
      {"cpu": cpus, "gpu": gpus_per_trial},
   )

   #save loss data for reporting 
   #save best model
   #record and save model parameters, including featurizer, data related to that, seq length?, data splits


   



'''
build_model
train_model
optimize_model
record_model_parameters
save_model
load_model
inference
'''

