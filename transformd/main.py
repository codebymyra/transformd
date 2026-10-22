'''
main.py
Framework for featurizing MD trajectories into downstream transformer task
Will plug into isEE.

This script handles featurization of MD trajectories for preparation for ML.
'''
from transformd import check_log_for_traj



def main()

 #method for retrieving user-specified isEE working directory
 get_working_dir()
 #method that uses isEE restart file to get list of completed trajectories
 list_of_trajs = get_trajs_in_working_dir()

 #mdengine contains anything that is engine specific, such as loading trajectories
 mdengine = factory.get_mdengine()
 #checks a database for whether a trajectory has been run already. Loads it if so, otherwise creates a new one
 #need to create a wrapper so that the actual type of object (dataframe, whatever) doesn't matter; the calls to it wouldn't change
 #but that the actual data type can change later.
 create_or_open_log(logfile)

#registry: get current list of features that could be calculated
 #loads trajectories one-by-one based on restart file
 for traj in list_of_trajs:

  feature_list = check_log_for_traj(traj) #checks whether a trajectory has been run yet. Returns a list of features to calculate.
  #if the feature list is empty, skip. Otherwise, go on to load the trajectory.

 #seq: {feature: [data]}, {feature2: [data]}
  #or, for a given feature, seq: [data]
  loaded = mdengine.load_traj(traj)
  seq, n_Frames, n_Res = loaded.get_traj_info() #returns seq, n_frames, n_res

    #a feature object is an array containing rows representing sequence, and values per residue for that sequence
  #this is what is used for training the transformer
  #might have a CNN data object which is by-feature and contains seq: label for each feature
  #and a transformer data object that is [[seq],[feat1],[feat2],...]
  #and both of these get updated as features are calculated and saved separately
  #or have one file, and during train time iterate the i,j object or something?
    create_feature_object = Feature()

  new_features = []
load_sequence_database(filepath) #middle step for tracking indiivudal trajectory data
load_feature_database(feature_filepath)
  for ft in feature_list:
    feature = features.calculate_feature # feature, numpy array
    database.add_data_to_sequence_database(seq, feature)
    database.save_sequence_database(filepath)  # nested dictionary
    database.get_aggregated_data #numpy array to be added to downstream ML files



    #if a new array or sequence has to be added, that means the featurized version also needs to change, as do downstream files
    save_by_protein_features
    update_byres_feature_database #e.g. if new feature or new trajectory is added, if the data is by-residue


    #once all that is finished, append the feature to the new feature list so that the log can be updated
    new_features.append(ft)

  #so now I have a  database hanging out

  update_log_for_traj(traj, new_features)
  save_feature_database






  #maybe just record sequence

  #want to operate on a feature object that contains seq, n_frames, n_res, and later, a feature.
  #In this format, I think it makes sense for the object to look like:
  #I want to make something like the restart.pkl file, except that it is a feature object containing:
  #sequence, frames residues, maybe topology file name, trajectory file name, each feature value.
  #This gets saved in a namespace object.
  #I can refer to this any time I am processing trajectories. I can also use it to restart? I can use the log file as a shortcut to that, too.
  #From the object, I need to extract the sequence, for each of the features, return seq-label to some output file.
  #If there are more than one label, then I take a weighted average or sth.

  #modify feature object so it contains all the new features?
  for ft in feature_list:
   #call on functions from features.py
   features.calculate feature
   #add feature calculation to the feature object for that sequence
   save_to_feature_object
   #{'ASDF': 'trajectory': [str], 'topology': [str], 'frames': [int,], 'residues': [int], 'rmsf-ca': [np.ndarray], 'rg': [np.ndarray]}
   #and if we end up with the same sequence again, we can check if the trajectory has been processed before. and if it has, skip!
   #process and turn into something I want to save for downstream ML, e.g. taking a weighted average
   featurizer.convert_to_features()


   #Want to operate on a feature object
   database.calc_all_new_features(loaded)  # database stores features
   #average these features using time_average

  #after calculating raw feature data, do some operation (modular). For now, take time average
  #raw feature data will always be saved for now. Will need to change to bool if data gets too big
  featurizer = Featurizer()

  #add to a second database once that downstream alculation hasd been done
  feature_database = FeatureDatabase()
  feature_database = featurizer.convert_to_features(database, feature_database)
  #save out the feature for model training. needs to be by-feature format.
  write_feature_model_training_datafiles(feature_database)

 # Train CNNs

 # Train transformer

 # Downstream task

if __name__ == "__main__":
 main()
```