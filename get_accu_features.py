
########################################################################################################
############################################## MY VERSION ##############################################
########################################################################################################


import yasa
import matplotlib.pyplot as plt
import numpy as np
import os 
from scipy.io import loadmat
import pandas as pd 
from pathlib import Path 
import mne 
from mne.io import RawArray 
from staging import SleepStaging 


#### 
 # yasa values for hypnogram
  #* -2  = Unscored
  #* -1  = Artefact / Movement
  #* 0   = Wake
  #* 1   = N1 sleep
  #* 2   = N2 sleep
  #* 3   = N3 sleep
  #* 4   = REM sleep


### Variables
sf = 512 #Hz
epoch_len = 2.5 # sec


### Helper Functions 
def get_files(target_folder):

    ## Path File Names 
    eeg_file = os.path.join(target_folder, 'EEG.mat')
    emg_file = os.path.join(target_folder, 'EMG.mat')
    label_file = os.path.join(target_folder, 'labels.mat')

    ## Getting the File Paths 
    file_list = ["EEG.mat", "EMG.mat", "labels.mat"]
    full_file_list = list(
                map(lambda x: os.path.join(target_folder, x), 
                file_list)
                )

    paths_are_files = list(
                map(lambda x: os.path.isfile(x),
                full_file_list)
                ) 

    if not all(paths_are_files):
        raise FileNotFoundError
    
    ## Load Files 
    eeg = loadmat(eeg_file)
    emg = loadmat(emg_file)
    label = loadmat(label_file)

    ## Yasa Values for Hypnogram 
    accusleep_dict = {
                        1:4, #R
                        2:0, #W
                        3:2, #N
                        }
    
    ## Reshape 
    label = np.squeeze(label['labels'])
    eeg_array = np.squeeze(eeg['EEG'])
    emg_array = np.squeeze(emg['EMG'])

    ## Re-Map The Label Values
    label_df = pd.DataFrame({"label": label})
    label_df['label'] = label_df['label'].map(accusleep_dict)
    label_array = label_df['label'].values 
    
    ## Return 
    return eeg_array, emg_array, label_array 


# we need to inlcude these because there will be key error
custom_bands =  [
          (0.4, 1, 'sdelta'), 
          (1, 4, 'fdelta'), 
          (4, 8, 'theta'),
          (8, 12, 'alpha'), 
          (12, 16, 'sigma'), 
          (16, 30, 'beta')
      ]

# try to read
def extract_accu_features(directory):
  try:
    eeg, emg, labels = get_files(directory)
    # try to read
    print(f'Data read from {directory}')
    # Create array
    info =  mne.create_info(["eeg","emg"], 
                            sf, 
                            ch_types='misc', 
                            verbose=False)
    raw_array = RawArray(np.vstack((eeg, 
                                    emg)),
                                    info, verbose=False)
    #print("Creating Staging Class")
    sls = SleepStaging(raw_array,
                       eeg_name="eeg", 
                       emg_name="emg")
    # this will use the new fit function
    sls.fit(epoch_sec=2.5)
    print("Fit default bands finished")
    features = sls.get_features()
    # deal with label issues
    if features.shape[0] - len(labels) == 1:
      print("shape mismatch, appending 0")
      labels = np.append(labels,0) 
		  # print("shape mismatch, appending 0")
      # labels = np.append(labels, 0)
		# we add the labels to the features 
    features["stage"] = labels
    # save the stuff
    # features.to_csv(os.path.join(directory,"features.csv"),
    #                index=None)
    # np.save(os.path.join(directory, "labels.npy"), labels)
    return features 
	  # return features
  
  except:
    print("Error happened")
    return None 
  # except:
  #   print("Error happened")
  #   return None


########## OWN CODE 

### Path to AccuSleep Data Directory 
data_folder = '/Volumes/Extreme SSD/AccuSleep' 
# 24hr Recordings
root_folder = os.path.join(data_folder, '24-hour_recordings')
# Build the paths
p = Path(root_folder)
all_paths = list(Path(root_folder).glob('*/*'))
all_dirs = sorted([folder for folder in all_paths if folder.is_dir() and "cycle" in str(folder)])

for dir in all_dirs:
    # print(dir)
    feat = extract_accu_features(dir)
    feat.to_csv(f'{dir}/features.csv')
    # break