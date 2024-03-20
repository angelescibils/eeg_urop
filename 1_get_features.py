################## IMPORTS ################## 
import os 
import glob
import  pandas as pd 
from py_console import console 
import numpy as np 
import mne
from mne.io import RawArray         ## create MNE array 

# From phdutils 
from staging import SleepStaging    ## to find MNE features
import utils as uts                 ## naming 


################## VARIABLES ################## 
sf = 100      ## Hz original sampling is 1000, data is downsampled by 10
eps = 2.5     ## sec


################## DESCRIPTION ################## 
# INPUT: animal folder path (string or os.path to animal folder)
# OUTPUT: features folder and feature files  
    # animal_id/session_id/features/sub_{animal_id}_{session_id}_chann-EEGi_features.csv.gz

################## EXTRACT FEATURES ################## 

def get_eeg_features(eeg_data, sampling_frequency=sf, epoch_sec=eps):
    '''
    Gets the features of all EEG channels


    Parameters
    ----------
    eeg_data : :py:class:`pandas.DataFrame`
            dataframe with EEG and EMG as columns, time as rows, 
    sampling_frequency: int
    epoch_sec: int
    
    Returns
    -------
    eeg_features_df_dict : dict of "EEG X": py:class:`pandas.DataFrame`
            A dictionary mapping electrode string names (i.e. 'EEG8') to their corresponding features dataframe 
    '''

    ## Initialize a Dictionary of features 
    df_dict = dict() 

    ## For each EEG channel, get the 96 features 
    for column in eeg_data.columns:
        if 'EEG' in column:
            eeg = eeg_data[column].to_numpy()
            emg_diff = eeg_data["EMG1"]  

            info = mne.create_info(["eeg","emg"],
                                    sampling_frequency, 
                                    ch_types='misc',
                                    verbose=False)
            raw_array = RawArray(np.vstack((eeg, emg_diff)), info, verbose=False)  
            sls = SleepStaging(raw_array, eeg_name = 'eeg', emg_name = 'emg')

            ## Store in Dictionary if 
            df_dict[column] = sls.get_features(epoch_sec = epoch_sec)  

    return df_dict 

### CAREFUL WITH THIS FUNCITON CALLING UTS!!!! 
def save_features(data_dict, saving_folder, animal_id, session_id):
    '''
    Saves the features of all EEG channels of a specific animal and session


    Parameters
    ----------
    data_dict: dict of "EEG X": py:class:`pandas.DataFrame`
            A dictionary mapping electrode string names (i.e. 'EEG8') to their corresponding features dataframe 
    saving_folder: str
            A string (or os.path) indicating the location of where to save files (animal/session/features)
    animal_id: str 
    session_id: str 
    '''

    for key, df in data_dict.items():
        # Generate a filename for each DataFrame based on its key
        filename = uts.bids_naming(saving_folder, animal_id, session_id, f'chann-{key}_features.csv.gz')
        df.to_csv(filename, index=False)
        console.success(f"Wrote {key} feature data to {filename}")


def get_animal_features(animal_folder, sampling_frequency=sf, epoch_sec=eps): 
    '''
    Assuming BIDS convention, creates new folder called "features" where features are saved 
    Gets the features of all eeg channels of all sessions of one animal 

    Parameters
    ----------
    animal_folder : str or os.path
            string indicating path to animal folder
    sampling_frequency: int
    epoch_sec: int
    
    '''

    ## Getting Animal ID
    animal_id = os.path.basename(animal_folder)

    #### When running this on my computer, I had to change the file extensions to just .csv (pandas couldn't read the .csv.gz for the eeg files, but it could for the sleep files)
    #### Assuming we're creating a new "features" folder within each session
    for eeg_file_path in glob.glob(f'{animal_folder}/*/eeg/*.csv.gz', recursive=True):

        ## Getting Session Path & Info
        session_folder_path = os.path.dirname(os.path.dirname(eeg_file_path))
        session_id = uts.parse_bids_session(eeg_file_path)

        ## Creating Features folder 
        feature_folder_path = os.path.join(session_folder_path, 'features')
        if not os.path.exists(feature_folder_path):
            os.makedirs(feature_folder_path)

        # Get Features for this session for this animal
        eeg_data = pd.read_csv(eeg_file_path)
        df_dict = get_eeg_features(eeg_data, sf, eps)

        # Save Features 
        save_features(df_dict, feature_folder_path, animal_id, session_id)

