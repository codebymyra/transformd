'''
Add any functions related to converting aggregated data from SequenceDatabase into a different format for downstream tasks.
Inputs are:
protein sequence: str
feature name: str
data: numpy array of dimensions (1,) or (n_residues,)
'''

class Featurizer(object):



#CNN:
#get a single sequence - label value
#apend these seq-label values to their respective files
#split out data for each individual feature
#save out data
#load data in the future to adjust

#transformer:
#if the data is by-residue (more than dim 1),
#split out sequence into individual string/letters in a numpy array
#below that, append each of the aggregated by-residue feature arrays
#we need to keep the order of features constant. Maybe by implementing a dictionary?
#turn a sequence entry into a feature format
#save out data
#load data in the future to adjust