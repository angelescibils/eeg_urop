################## IMPORTS ################## 
from xml.sax.handler import feature_namespace_prefixes
import numpy as np 
import mne
from mne.io import RawArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
import glob 

# From phdutils 
import predict as pr
from staging import SleepStaging
import utils as uts 
from PCA_functions import * 


##################################################################
######################## EXTRACT FEATURES ######################## 
##################################################################

################## INTERNAL DATA ################### 

## Comment for User: Change Directory 
data_folder = '/Volumes/Extreme'
animals = ['MLA152', 'MLA153']
all_animal_folders = [os.path.join(data_folder, a) for a in animals]

## Get features and save for all 
for animal_folder in all_animal_folders:
    get_animal_features(animal_folder, sampling_frequency=100, epoch_sec=2.5) 



##################################################################
########################### APPPLY PCA ###########################
##################################################################

for animal_folder in all_animal_folders:

    animal_id = os.path.basename(animal_folder)

    ## Assuming Ref_Electrode to be EEG8 of the first session 
    eeg_feature_dict = dict()
    first_session_id = os.listdir(animal_folder)[0]
    path_to_features = os.path.join(animal_folder, first_session_id, 'features')
    feature_files = os.listdir(path_to_features)
    
    for file in feature_files:
        if 'EEG8' in file:
            eeg_feature_dict['EEG8'] = pd.read_csv(os.path.join(path_to_features, file))


    var, ref_pca = get_PCA(eeg_feature_dict, ref_electrode='EEG8', n_components=3)
    
    ## Apply PCA to all other electrodes of the animal 
    ref_title = f'{animal_id} {first_session_id} EEG8'
    apply_PCA_all(animal_folder, ref_title, ref_pca, n_components=3, xlim=[-20, 30], ylim=[-30, 50])

### Extract features of one animal with multiple sessions 
### This can be turned into a function, there is a lot of place for improvement and error detection here! 
# all_session_ids = os.listdir(animal_folder)

# for session_id in all_session_ids:

#     ## Add other measures to ensure the session
#     if session_id == 'rec' or session_id == 'start': continue 

#    # Going into Features Folder
#     features_folder = os.path.join(animal_folder, session_id, 'features')
#     features_files = os.listdir(features_folder)

#     sleep_folder = os.path.join(animal_folder, session_id, 'sleep')
#     sleep_file = os.listdir(sleep_folder)[0] 
#     sleep_file_path = os.path.join(sleep_folder, sleep_file)
       
#     ## Initializing Variables 
#     i = 0 
#     eeg_data = dict()

#     ## Going through Each Channel Feature (Assuming they're organized by EEG1-EEG9)
#     for file in features_files:
#             i +=1
#             eeg_data[f'EEG{i}'] = pd.read_csv(os.path.join(features_folder, file))
#             if i ==9: break 

#     variance, ref_pca = get_PCA(eeg_data, ref_electrode='EEG8', n_components=3)
#     dict_pca_df = apply_PCA(eeg_data, ref_pca,  n_components=3)
#     # print(f'{session} has variance of {variance}')

#     sleep_df = pd.read_csv(sleep_file_path)
#     plot2D(dict_pca_df, sleep_df, f'MLA152 {session_id} with MLA154 EEG8')



##################################################################
################## LOAD FEATURE & SLEEP DF DATA ################## 
##################################################################


################## INTERNAL DATA ################### 

# ## EEG Data
# ## Hypno Prediction Data


################## ACCUSLEEP DATA ################## 
## Path to AccuSleep Data Directory 
# data_folder = '/Volumes/Extreme SSD/AccuSleep' 
# root_folder = os.path.join(data_folder, '24-hour_recordings')
# p = Path(root_folder)
# all_paths = list(Path(root_folder).glob('*/*'))
# all_dirs = sorted([folder for folder in all_paths if folder.is_dir() and "cycle" in str(folder)])

## Extracting Accu Features 
# accu_feat_dict = dict()
# accu_sleep_dict = dict()
# for dir in all_dirs:
#     mouse_name = (os.path.basename(os.path.dirname(dir)))
#     session = os.path.basename(dir)
#     feature_path = os.path.join(dir, 'features.csv')
#     accu_feat_dict[f'{mouse_name}_{session}'] = pd.read_csv(feature_path)
#     eeg, emg, labels = get_files(dir)
#     accu_sleep_dict[f'{mouse_name}_{session}'] = labels 