'''
main.py
Framework for featurizing MD trajectories into downstream transformer task
Will plug into isEE.

This script handles featurization of MD trajectories for preparation for ML.
'''

```
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
 if settings.load_existing_database:
  database = load_an_existing_database(settings)
 else:
  database = Database()

 #loads trajectories one-by-one based on restart file
 for traj in list_of_trajs:
  loaded = mdengine.load_traj(traj)
  traj_info = loaded.get_traj_info()
  #this somehow directs to specific features gbased on database info
  database.calc_all_new_features(loaded)  # database stores features
  #average these features using time_average

 #isee's configure.py. Keep type checking and format but get rid of settings I don't want
 if settings.just_save_raw_feature_data:
  pass  # implement writing out raw_feature_data

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