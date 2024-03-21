
################## IMPORTS ################## 
from re import T
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from py_console import console 
import os
import glob 
from rlist_files import list_files
import re

# From phdutils 
import utils as uts 

################## VARIABLES ################## 
pca_n = 3 

################## HELPER FUNCTIONS ##################
def parse_bids_electrode(string):
    return os.path.basename(string).split("_")[2].replace("chann-", "")

def read_participants(path):
    return pd.read_csv(path, sep='\t')

def extract_baseline_dates(df):
    df = df.assign(baseline_date=pd.to_datetime(df['baseline_rec_start']).dt.date)
    return df[['id', 'baseline_date']]

def filter_subject(df, subject_id):
    return df.query(f"id == '{subject_id}'")

def find_baseline_session(participants_path, subject_id):
    participants = read_participants(participants_path)
    return extract_baseline_dates(filter_subject(participants, subject_id))

def find_session_folders(animal_folder_path):
    """
    Lists session folders in the given path and checks if they end with a date in the format 'YYYY-mm-dd'.

    Parameters:
    animal_folder_path (str): The path of the animal folder.

    Returns:
    list: A list of session folders that end with a date in the format 'YYYY-mm-dd'.
    """
    # Use glob to get all folders in the animal_folder_path
    all_folders = glob.glob(os.path.join(animal_folder_path, '*'))
    # Regular expression to match the date format 'YYYY-mm-dd' at the end of a string
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}$')
    # Filter the folders to only include those that end with a date in the format 'YYYY-mm-dd'
    session_folders = [folder for folder in all_folders if date_pattern.search(os.path.basename(folder))]
    return session_folders

def extract_session_from_path(folder_path):
    """
    Extracts the session date from the given folder path. The date is expected to be in the format 'YYYY-mm-dd'.

    Parameters:
    folder_path (str): The path of the folder.

    Returns:
    str: The session date in the format 'YYYY-mm-dd', or None if no date is found.
    """
    # Regular expression to match the date format 'YYYY-mm-dd'
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
    # Search for the date in the folder name
    match = date_pattern.search(folder_path)
    # If a date is found, return it. Otherwise, return None.
    return match.group() if match else None


def read_sleep(session_folder):
    sleep_folder = os.path.join(session_folder, "sleep")
    sleep_dict = dict()
    for file in list_files(sleep_folder, pattern = "hypno_predictions", full_names=True):
        session_id = extract_session_from_path(file)
        sleep_dict[session_id] = pd.read_csv(file)
    return sleep_dict

################## PCA OF REFERENCE EEG ################## 
def extract_PCA(ref_electrode_df, n_components):
    '''
    Extract PCA of reference electrode

    Parameters
    ----------
    ref_electrode_df : py:class:`pandas.DataFrame`
            The feature.csv.gz of the reference electrode 
    n_components: int
            Number of PCA components extracted

    Returns
    -------
    explained_variance_ratio : lst
            List of variance corresponding to each PCA element
    pca_df: py:class:`pandas.DataFrame`
            Dataframe with PC as columns 
    '''

    # print('Extracting PCA...')

    ## SANITY CHECK: 
        # There is no index or "epoch" column
        # IF THERE IS A SLEEP COLUMN DELETE BEFORE RUNNING PCA 

    ## Standardize Data
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(ref_electrode_df)

    ## Extract PCA
    pca = PCA(n_components = n_components)
    principal_components = pca.fit_transform(df_standardized)

    ## Return Results
    explained_variance_ratio = pca.explained_variance_ratio_
    # pca_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(n_components)])

    return explained_variance_ratio, pca 


################## APPLY PCA TO OTHER EEG ################## 
# Transform the other EEG feature dataframes with reference channel's PCA
def apply_PCA(session_feature_dict, ref_pca, n_components):
    '''
    Applies reference electrode's PCA to the other electrode's features matrices

    Parameters
    ----------
    eeg_feature_dict : dict of "EEG X": py:class:`pandas.DataFrame`
            A dictionary mapping electrode string names (i.e. 'EEG8') to their corresponding features dataframe 
    ref_pca: sklearn.decomposition._pca.PCA

            
    n_components: int
            Number of PCA components extracted

    Returns
    -------
    dict_pca_df: dict of "EEG X": py:class:`pandas.DataFrame`
            A dictionary mapping electrode string names (i.e. 'EEG8') to their corresponding PCA transformed features dataframe 
    '''

    # print('Applying PCA...')

    ## Initialize Dicitonary of Dataframes
    dict_pca_df = dict()

    ## Apply PCA on every channel
    for target_session in session_feature_dict.keys():

        ## SANITY CHECK: there is no index or "epoch" column

        ## Standardize Data
        scaler = StandardScaler()
        df_standardized_target = scaler.fit_transform(session_feature_dict[target_session])
        
        ## Apply PCA
        principal_components_target = ref_pca.transform(df_standardized_target)

        ## Create Dataframe 
        # explained_variance_ratio_target = ref_pca.explained_variance_ratio_
        pc_df_target = pd.DataFrame(data=principal_components_target, columns=[f"PC{i+1}" for i in range(n_components)])
        dict_pca_df[target_session] = pc_df_target 

    return dict_pca_df


################## SAVE PCA OF ALL EEG ################## 

def save_pca(data_dict, saving_folder, animal_id, session_id, ref_electrode, ref_session):
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

    for session, df in data_dict.items():
        # Generate a filename for each DataFrame based on its key
        filename = uts.bids_naming(saving_folder, animal_id, session_id, f'chann-{ref_electrode}_between-pca_ref-{ref_session}.csv.gz')
        df.to_csv(filename, index=False)
        console.success(f"Wrote {session} after transformed PCA data to {filename}")


################## ALL TOGETHER ################## 

def between_session_pca(animal_folder_path, ref_session, ref_electrode='EEG8', n_components=3):
    '''
    Given an animal's folder:
        Extracts PCA of ref electrode on ref session 
        Applies ref PCA to other sessions 
        Saves PCAs
    '''
    assert len(ref_session) == 1, "More than one animal in ref_session"
    ## list animal features
    animal_id = str(ref_session.id.values[0])
    ref_session_id = str(ref_session.baseline_date.values[0])
    all_sessions_path = find_session_folders(animal_folder)

    ## SANITY CHECK: What to do when session recordings aren't of the same length 
    ## For this, we don't really care they are not the same length, what matters is that they have the same parameters (96) for fit_transform()

    
    #### SANITY CHECK: Variance ratio
    features_dict = dict()
    for session_path in all_sessions_path:
        console.info(f"Reading Features from {session_path}")
        features_path = list_files(os.path.join(session_path, 'features'), pattern = ref_electrode, full_names=True)
        assert len(features_path) == 1, "Too many feature files found"
        features_dict[os.path.basename(session_path)] = pd.read_csv(features_path[0])
    ## Extract PCA of ref electrode 
    ratio, ref_pca = extract_PCA(features_dict[ref_session_id], n_components)

    # Apply ref_pca transform to all electrodes (including ref_electrode)
    pca_dict = apply_PCA(features_dict, ref_pca, n_components)

    # Create saving_folder 
    pca_folder_path = os.path.join(animal_folder_path, 'between-sessions', 'pca')
    if not os.path.exists(pca_folder_path):
        os.makedirs(pca_folder_path)

    # Save Files 
    # Design decision: for within_session_pca no need to store sleep df alongside
    save_pca(pca_dict, pca_folder_path, animal_id, session_id, ref_electrode) 
     


########

#### Modify Here For Now 
## animal id
animal_id = 'MLA152'
base_path = 'data'
### ------------------------ ####

# keep working here
# potential issues, if to csv.gz in one session
# e.g., recording was stopped and needed to be restarted, keys will be repeated
# think about maybe concat the data ?
read_sleep("data/MLA152/2023-12-11/")
# TODO: bind this with the pca dict

animal_folder = os.path.join(base_path, animal_id)
# find the reference session
ref_session = find_baseline_session(os.path.join(base_path, 'participants.tsv'), animal_id)
between_session_pca(animal_folder, ref_session, ref_electrode='EEG8', n_components=3)