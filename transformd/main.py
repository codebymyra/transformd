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
import configure
import sys
from isee_framework.main import Thread


#user-defined variables
#todo: how to get these to interface with isEE?

def main(settings):
    #get list of trajectories from working directory
    isee_working_dir = utilities.get_working_dir(settings.working_directory)
    traj_list= utilities.get_trajs_in_working_dir(isee_working_dir) #return list of tuples
    print(f'Found {len(traj_list)} trajectories in {isee_working_dir}.', flush=True)
    #create an MD engine object for loading trajectories
    mdengine = factory.mdengine_factory(settings.md_engine) #todo: how to create an engine-specific object such as AdaptAmber? Oh. See factory.py in Atesa

    #create or open log file. should be dict
    #todo: if this process is run by parallel workers, loading of the traj_log should happen newly for each trajectory rather than just once, probably.
    #todo: the way i have it is that a trajectory with the same file name will only ever be processed once. Perhaps change based on sequences and not trajectory name?
    traj_log = utilities.load_or_create_log_for_traj(settings.log_directory)

    #loop through list of trajectories.
    #calculate features as needed
    for top, traj in traj_list:
        #print(f'Loading {top} and {traj}')
        features_to_run = utilities.get_features_for_traj(traj, traj_log) #list
        print(f'Running the following features for {traj}: {features_to_run}', flush=True)
        loaded = mdengine.load_trajectory(isee_working_dir, traj, top)
        if loaded is None:
            print(f'Skipping {traj} and {top}')
            continue
        #print(loaded)
        seq, n_frames, n_res = features.get_traj_info(loaded)
        #print(f'Seq: {seq}, n_frames: {n_frames}, n_res: {n_res}')

        #initialize features to add to log
        added_features = []
        #initialize Sequence Database object to hold feature arrays
        seq_db = database.SequenceDatabase(settings.md_data_directory)
        seq_db.load_pickle() #assigns dict to seq_db?
        #load feature dictionary to update
        feature_db = featurizer.ByFeatureDatabase(settings.feature_data_directory)
        #todo: implement array database

        #calculate missing features
        #instantiate md feature calculator object
        md_calculator = features.MDFeatureCalculator()
        for feat_name in features_to_run:
            feature_arr = getattr(md_calculator, feat_name)(loaded)
            #update sequence database, feature dictionary, and added_feature list
            seq_db.add_data(seq, feat_name, feature_arr)
            seq_db.save_pickle()
            feature_agg_arr = seq_db.aggregate_data(seq, feat_name) #returns single numpy array for a feature
            feature_db.add_individual_feature_data(feat_name, seq, feature_agg_arr)
            #added_features.append(feat_name) #todo: delete this later as it is redundant
            utilities.update_log_for_traj(traj, settings.log_directory, [feat_name])

    return f'Feature extraction completed.' #Exit Message. change when adding additional steps

 # Convert data to training format

 # Save training data somewhere
 # Featurize sequences

'''
 # Train CNNs
split_train_test_data
shuffle_data
featurize_sequence
train_models_per_residue

# Train transformer
generate_training_data(from cnn)
shuffle_training_data(from nucb dataset)
split_train_test_data
train_model
'''

# Downstream task

def run_main():
    #get settings namespace from config file and move into main
    #based on isEE
    try:
        working_directory = sys.argv[2]
    except IndexError:
        working_directory = ''
    settings = configure.configure(sys.argv[1], working_directory)
    exit_message = main(settings)
    print(exit_message)



if __name__ == "__main__":
    run_main()
