
################## IMPORTS ################## 
from re import T
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from py_console import console 
import os
import glob 

# From phdutils 
import utils as uts 

################## VARIABLES ################## 
pca_n = 3 

################## HELPER FUNCTIONS ##################
def parse_bids_electrode(string):
    return os.path.basename(string).split("_")[2].replace("chann-", "")

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

    ## SANITY CHECK: There is no index or "epoch" column

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
def apply_PCA(eeg_feature_dict, ref_pca, n_components):
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
    for target_electrode in eeg_feature_dict.keys():

        ## SANITY CHECK: there is no index or "epoch" column

        ## Standardize Data
        scaler = StandardScaler()
        df_standardized_target = scaler.fit_transform(eeg_feature_dict[target_electrode])
        
        ## Apply PCA
        principal_components_target = ref_pca.transform(df_standardized_target)

        ## Create Dataframe 
        # explained_variance_ratio_target = ref_pca.explained_variance_ratio_
        pc_df_target = pd.DataFrame(data=principal_components_target, columns=[f"PC{i+1}" for i in range(n_components)])
        dict_pca_df[target_electrode] = pc_df_target 

    return dict_pca_df


################## SAVE PCA OF ALL EEG ################## 

def save_pca(data_dict, saving_folder, animal_id, session_id, ref_electrode):
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
        filename = uts.bids_naming(saving_folder, animal_id, session_id, f'chann-{key}_within-pca_ref-{ref_electrode}.csv.gz')
        df.to_csv(filename, index=False)
        console.success(f"Wrote {key} feature data to {filename}")


################## ALL TOGETHER ################## 

def within_session_pca(features_folder_path, ref_electrode, n_components):
    '''
    Given a features folder:
        Extracts PCA of ref electrode 
        Applies ref PCA to other electrodes 
        Saves PCA
    '''

    ## Read through Features + Save in Dictionary 
    features_dict = dict()

    first_file = True
    for feature_file_path in glob.glob(f'{features_folder_path}/*.csv.gz', recursive=True):

        ## Get Animal and Session ID
        if first_file:
            animal_id = uts.parse_bids_subject(feature_file_path) 
            session_id = uts.parse_bids_session(feature_file_path)
            first_file = False 

        channel = parse_bids_electrode(feature_file_path)
        features_dict[channel] = pd.read_csv(feature_file_path)

    ## SANITY CHECK: There should be 9 channels with df's of the same length

    ## Extract PCA of ref electrode 
    ratio, ref_pca = extract_PCA(features_dict[ref_electrode], n_components)
    
    ##### SANITY CHECK: Variance ratio

    ## Apply PCA to other electrodes 
    pca_dict = apply_PCA(features_dict, ref_pca, n_components)

    ## Create saving_folder 
    session_folder_path = os.path.dirname(features_folder_path)
    pca_folder_path = os.path.join(session_folder_path, 'within-session', 'pca')
    if not os.path.exists(pca_folder_path):
        os.makedirs(pca_folder_path)

    ## Save Files 
    ## Design decision: for within-session/pca no need to store sleep df alongside
    save_pca(pca_dict, pca_folder_path, animal_id, session_id, ref_electrode) 
     


########

## Is this last function necessary? (do within for all session in animal)
def multiple_within_session_pca(animal_folder, ref_electrode='EEG8', n_components=pca_n):
    '''
    Run within_session_pca on all sessions within one animal
    '''
    for feature_folder_path in glob.glob(f'{animal_folder}/*/features', recursive=True): 
        print(f'Working with {feature_folder_path}')
        within_session_pca(feature_folder_path, ref_electrode, n_components) 



#### RUN ACTUAL FUNCTIONS 
animal_folder ='data/MLA153'
multiple_within_session_pca(animal_folder)

# for feature_file_path in glob.glob(f'{animal_folder}/*/within_session_pca/*.csv.gz', recursive=True): 
#     df = pd.read_csv(feature_file_path)
#     print(len(df), len(df.columns))