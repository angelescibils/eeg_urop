################## IMPORTS ################## 
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


##################################################################
################## LOAD FEATURE & SLEEP DF DATA ################## 
##################################################################

# ## EEG Data
# file_path = "/Users/angelescibils/Dropbox/Angie/School & Education/Massachusetts Institute of Technology (MIT)/23-24 Junior/24 Spring/EEG UROP/Testing data/eeg/sub-MLA154_ses-20231226T004732_desc-down10_eeg.csv"
# animal = 'MLA154'
# eeg_data = pd.read_csv(file_path)

# ## Hypno Prediction Data
# sleep_path = '/Users/angelescibils/Dropbox/Angie/School & Education/Massachusetts Institute of Technology (MIT)/23-24 Junior/24 Spring/EEG UROP/Testing data/sleep/sub-MLA154_ses-20231226T004732_hypno_predictions_df.csv'
# sleep_df = pd.read_csv(sleep_path)


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



################## INTERNAL DATA ################### 

### Extract feature of one mouse with multiple sessions 
# for session in sessions_folders:
#         if session == 'rec' or session=='start': continue 

#         ## Going into Features Folder
#         features_folder = os.path.join(folder, session, 'features')
#         features_files = os.listdir(features_folder)
#         sleep_folder = os.path.join(folder, session, 'sleep')
#         sleep_file = os.listdir(sleep_folder)[0] 
#         sleep_file_path = os.path.join(sleep_folder, sleep_file)
       
#         ## Initializing Variables 
#         i = 0 
#         eeg_data = dict()

#         ## Going through Each Channel Feature
#         for file in features_files:
#                 i +=1
#                 eeg_data[f'EEG{i}'] = pd.read_csv(os.path.join(features_folder, file))
#                 if i ==9: break 

#         # variance, pca = my_PCA.get_PCA(eeg_data, ref_electrode='EEG8', n_components=5)
#         dict_pca_df = my_PCA.apply_PCA(eeg_data, ref_pca,  n_components=5)
#         # print(f'{session} has variance of {variance}')

#         sleep_df = pd.read_csv(sleep_file_path)
#         my_PCA.plot2D(dict_pca_df, sleep_df, f'MLA152 {session} with MLA154 EEG8')

