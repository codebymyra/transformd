'''
main.py
Framework for featurizing MD trajectories into downstream transformer task
Will plug into isEE.

This script handles featurization of MD trajectories for preparation for ML.
'''
import utilities
import factory
import database
import featurizer
import features
#from transformd import utilities
#from transformd import factory
#from transformd import database
#from transformd import featurizer
#from transformd import features


#user-defined variables
#todo: how to get these to interface with isEE?

working_dir = '/jumbo/burginlab/myra/transformd/transformd/tests/test_data/test_working_directory'
engine = 'amber'
traj_log_dir = 'transformd/tests/test_data/file_dump'
md_data_dir = 'transformd/tests/test_data/raw_data_dump'
feature_data_dir = 'transformd/tests/test_data/feature_dump'

def main():
    #get list of trajectories from working directory
    isee_working_dir = utilities.get_working_dir(working_dir)
    traj_dict = utilities.get_trajs_in_working_dir(isee_working_dir)

    #create an MD engine object for loading trajectories
    mdengine = factory.mdengine_factory(engine) #todo: how to create an engine-specific object such as AdaptAmber? Oh. See factory.py in Atesa

    #create or open log file. should be dict
    traj_log = utilities.load_or_create_log_for_traj(traj_log_dir)

    #loop through list of trajectories.
    #calculate features as needed
    for top, trajectories in traj_dict.items():
        #loop through each traj associated with a topology file
        for traj in trajectories:
            features_to_run = utilities.get_features_for_traj(traj, traj_log) #list
            loaded = mdengine.load_trajectory(traj, top)
            seq, n_frames, n_res = loaded.get_traj_info()

            #initialize features to add to log
            added_features = []
            #initialize Sequence Database object to hold feature arrays
            seq_db = database.SequenceDatabase(md_data_dir)
            seq_db.load_pickle() #assigns dict to seq_db?
            #load feature dictionary to update
            feature_db = featurizer.ByFeatureDatabase(feature_data_dir)
            #todo: implement array database

            #calculate missing features
            #instantiate md feature calculator object
            md_calculator = features.MDFeatureCalculator()
            for feat_name in features_to_run:
                feature_arr = getattr(md_calculator, feat_name)(traj)
                #update sequence database, feature dictionary, and added_feature list
                seq_db.add_data(seq, feat_name, feature_arr)
                feature_agg_arr = seq_db.aggregate_data(seq, feat_name) #returns single numpy array for a feature
                feature_dict = feature_db.load_dict(feat_name)
                feature_db.add_individual_feature_data(feature_dict, feat_name, seq, feature_agg_arr)
                added_features.append(feat_name)

 # Convert data to training format

 # Save training data somewhere
 # Featurize sequences

 # Train CNNs

 # Train transformer

 # Downstream task

if __name__ == "__main__":
    main()
