## IMPORTS
import predict as pr
from staging import SleepStaging
import utils as uts 
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

import PCA as PCA_script  

### Getting Features of the Animals I have files for 

## DATA Folder 
data_folder = '/Volumes/Extreme SSD'

sf = 100    ## sampling frequency 
es = 2.5    ## epoch sec


####### MANUALLY ANNOTATED ANIMAL
aligned = 'aligned/sub-MLA127_ses-20230807T090647_alignedeeg.csv'
aligned_animal = os.path.join(data_folder, aligned)
aligned_eeg_data = pd.read_csv(aligned_animal)

df_dict = PCA_script.get_eeg_features(aligned_eeg_data, sampling_frequency=sf, epoch_sec=es)

for channel, df in df_dict.items():
    feature_file_name = f'{os.path.basename(aligned_animal)[:-4]}_{channel}_features.csv'   
    path = os.path.join(data_folder,'aligned', feature_file_name)
    df.to_csv(path)  

####### "MESSY" ANIMALS
# Go through each animal's folder (animal/date/eeg/data and animal/date/sleep/data)
# target_animals = ['MLA152', 'MLA153']
# animal_folders = [os.path.join(data_folder, animal) for animal in target_animals]
# for folder in animal_folders:

#         ## GETTING EEG FILES
#         for eeg_file_path in glob.glob(f"{folder}/*/eeg/*.csv", recursive=True):

#             animal_date_folder_path = os.path.dirname(os.path.dirname(eeg_file_path))
#             feature_folder = os.path.join(animal_date_folder_path, 'features')

#             name = f'{os.path.basename(eeg_file_path)[:-4]}_features.csv'            
#             # CREATING FEATURE FOLDER
#             if not os.path.exists(feature_folder):
#                 # Create the new folder if it doesn't exist
#                 os.makedirs(feature_folder)

#             ## EXTRACTING THE FEATURES FOR THIS ANIMAL
#             eeg_data = pd.read_csv(eeg_file_path)
#             df_dict = PCA_script.get_eeg_features(eeg_data, sampling_frequency=sf, epoch_sec=es)

#             ## SAVE THE FEATURES 
#             # When saving the files, there will be 97 columns (epoch is added)
#             for channel, df in df_dict.items():
#                 feature_file_name = f'{os.path.basename(eeg_file_path)[:-4]}_{channel}_features.csv'   
#                 path = os.path.join(feature_folder, feature_file_name)
#                 df.to_csv(path)  