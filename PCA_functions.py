################## IMPORTS ################## 
import numpy as np 
import mne
from mne.io import RawArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from py_console import console 
import os
import glob 
import re 

# From phdutils 
import predict as pr
from staging import SleepStaging
import utils as uts 


################## VARIABLES ################## 
sf = 100      ## Hz original sampling is 1000, data is downsampled by 10
eps = 2.5     ## sec
pca_n = 3 


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

    print('Getting features...')

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

def save_features(data_dict, saving_folder, animal_id, session_id):
    for key, df in data_dict.items():
        # Generate a filename for each DataFrame based on its key
        filename = uts.bids_naming(saving_folder, animal_id, session_id, f'chann-{key}_features.csv.gz')
        df.to_csv(filename, index=False)
        console.success(f"Wrote {key} feature data to {filename}")


def get_animal_features(animal_folder, sampling_frequency=sf, epoch_sec=eps): 
    '''
    Assuming BIDS convention
    Create new folder called "features" where features are saved 
    '''
    ## Getting Animal ID
    ## consider using ut.sparse_bids_subject(string)
    animal_id = os.path.basename(animal_folder)

    #### When running this on my computer, I had to change the file extensions to just .csv (pandas couldn't read the .csv.gz for the eeg files, but it could for the sleep files)
    #### Assuming we're creating a new "features" folder within each session
    for eeg_file_path in glob.glob(f'{animal_folder}/*/eeg/*.csv.gz', recursive=True):

        ## Getting Session Path & Info
        session_folder_path = os.path.dirname(os.path.dirname(eeg_file_path))
        ## consider using uts.parse_bids_session(string: str)
        session_id = os.path.basename(session_folder_path)

        ## Creating Features folder
        feature_folder_path = os.path.join(session_folder_path, 'features')
        if not os.path.exists(feature_folder_path):
            os.makedirs(feature_folder_path)

        ## Get Features for this session for this animal
        eeg_data = pd.read_csv(eeg_file_path)
        df_dict = get_eeg_features(eeg_data, sf, eps)

        ## Save Features 
        save_features(df_dict, feature_folder_path, animal_id, session_id)

        
################## PCA OF REFERENCE EEG ################## 
## To Improve: don't pass entire dictionary, just ref_electrode dataframe (should improve time + space) 
def get_PCA(eeg_feature_dict, ref_electrode='EEG8', n_components=pca_n):
    '''
    Extract PCA of reference electrode

    Parameters
    ----------
    eeg_feature_dict : dict of "EEG X": py:class:`pandas.DataFrame`
            A dictionary mapping electrode string names (i.e. 'EEG8') to their corresponding features dataframe 
    ref_electrode: str
            Name of reference electrode upon which to base PCA
    n_components: int
            Number of PCA components extracted

    Returns
    -------
    explained_variance_ratio : lst
            List of variance corresponding to each PCA element
    pca_df: py:class:`pandas.DataFrame`
            Dataframe with PC as columns 
    '''

    print('Getting PCA...')
    ## Standardize Data
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(eeg_feature_dict[ref_electrode])

    ## Extract PCA
    pca = PCA(n_components = n_components)
    principal_components = pca.fit_transform(df_standardized)

    ## Return Results
    explained_variance_ratio = pca.explained_variance_ratio_
    # pca_df = pd.DataFrame(data=principal_components, columns=[f"PC{i+1}" for i in range(n_components)])

#     print(f'{type(pca)=}')
    return explained_variance_ratio, pca 


################## APPLY PCA TO OTHER EEG & VISUALIZE ################## 
# Transform the other EEG feature dataframes with reference channel's PCA
def apply_PCA(eeg_feature_dict, ref_pca, n_components=pca_n):
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

    print('Applying PCA...')

    ## Initialize Dicitonary of Dataframes
    dict_pca_df = dict()

    ## Apply PCA on every channel
    for target_electrode in eeg_feature_dict.keys():

        ## Standardize Data
        scaler = StandardScaler()
        df_standardized_target = scaler.fit_transform(eeg_feature_dict[target_electrode])
        
        ## Apply PCA
        principal_components_target = ref_pca.transform(df_standardized_target)

        ## Create Dataframe 
        explained_variance_ratio_target = ref_pca.explained_variance_ratio_
        pc_df_target = pd.DataFrame(data=principal_components_target, columns=[f"PC{i+1}" for i in range(n_components)])
        dict_pca_df[target_electrode] = pc_df_target 

    return dict_pca_df


################## VISUALIZE PCA ################## 
# Visualize the PCA of the EEG features 
## To improve
    ## Add second figure with bar
    ## Add title options (mention ref electrode/animal and against who)

def plot2D(animal_folder, dict_pca_df, sleep_df, title, xlim=[-20, 30], ylim=[-30, 50]):
    '''
    Plots PC1 vs. PC2 of each electrode 

    Parameters
    ----------
    dict_pca_df: dict of "EEG X": py:class:`pandas.DataFrame`
            A dictionary mapping electrode string names (i.e. 'EEG8') to their corresponding PCA transformed features dataframe 
    sleep_df: py:class:`pandas.DataFrame`
            Hypno predictions of each electrode channel         

    Returns
    -------
    A 3x3 plot 
    
    '''

    print('Plotting...')
 
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))


    for i in range(3):
        for j in range(3):
            electrode = f'EEG{i * 3 + j + 1}'
            merged_df = pd.concat([dict_pca_df[f'chann-{electrode}'], sleep_df[electrode]], axis=1)
            
            # scatter_plot = plt.scatter(merged_df['PC1'], merged_df['PC2'], c=merged_df[electrode], cmap='viridis')
            scatter = axes[i,j].scatter(merged_df['PC1'], merged_df['PC2'], c=merged_df[electrode], alpha=0.1)
            axes[i, j].set_title(f'EEG{i * 3 + j + 1}')
            axes[i, j].set_xlabel('Principal Component 1')
            axes[i, j].set_ylabel('Principal Component 2')

            axes[i, j].set_xlim(xlim)  # Adjust the limits as needed
            axes[i, j].set_ylim(ylim)  # Adjust the limits as needed

            
    legend_labels = {'$\\mathdefault{0}$': 'Wake','$\\mathdefault{2}$': 'NREM', '$\\mathdefault{4}$': 'REM'}
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    fig.legend(handles, [legend_labels[label] for label in labels], title='Hypno Prediction', loc='upper right')

    plt.suptitle(f'{title} PCA', fontsize=16, fontweight='bold')
#     subtitle = f'Scatter Plot of PC1 vs PC2 with Sleep Stage Classification'
    plt.tight_layout()
    plt.savefig(f'{animal_folder}/{title}_PCA.png',  dpi=300)
    plt.show()

def get_percentages(sleep_df):
        '''
        For each session in animal folder 
        Assumptions:
                *  sleep_df has EEGi for columns and predictions for rows
                *  0: Wake, 2:REM, 4: NREM 
        '''

        # Create New DF with percentages 
        percentage_df = pd.DataFrame(columns=['electrode', '0-Wake', '2-REM', '4-NREM'])

        # Find percentages of each prediction for each EEG channel 
        ## there is definitely an easiert / more efficient way to do this 
        for eeg_channel in sleep_df.columns:
                
                interim = sleep_df[eeg_channel].value_counts(normalize=True).reset_index()
                interim.columns = ['Prediction', 'Percentage']
                interim['Percentage'] *= 100 # convert to percentage
                interim = interim.sort_values(by='Prediction')

                ## ADD ERROR DETECTION: what if there is a prediction missing? (i.e. no NREM predicted)

                list_form = interim['Percentage'].to_list()
                list_form.insert(0, eeg_channel)

                percentage_df.loc[len(percentage_df)] = list_form 
        
        return percentage_df 

def plot_stacked_sleep(percentage_df, title):

        ## Plot Data
        fig, ax = plt.subplots(figsize=[10,6])

        percentage_df.plot(
                        x='electrode', 
                        kind = 'bar',
                        stacked=True, 
                        title=title, 
                        colormap='viridis',
                        ax=ax, 
                        )

        ## Add Percentage Annotations to the plot 
        for i, (index, row) in enumerate(percentage_df.iterrows()):
                x_position, total_height = i, 0 
                for j, value in enumerate(row[1:]):
                        total_height += value 
                        text_color = 'black' if j==2 else 'white' #set NREM percentages to black 

                        ax.text(
                                x_position, 
                                total_height -value/2,
                                f'{value:.2f}%', 
                                ha = 'center', va='center', 
                                color = text_color, 
                                fontsize=8, 
                                fontweight='bold'
                        )
        ax.set_title(title, fontsize = 16, fontweight='bold')
        ax.set_xlabel('Electrode Channel', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')

        return plt 

def plot_all_stacked_sessions(animal_folder):
        '''
        For each session in animal folder, 
        Plot a stacked bar graph of percentages of predicted sleep stage
        Assuming BIDS convention
        '''
        ## Getting Animal ID
        ## consider using ut.sparse_bids_subject(string)
        animal_id = os.path.basename(animal_folder)

        #### When running this on my computer, I had to change the file extensions to just .csv (pandas couldn't read the .csv.gz for the eeg files, but it could for the sleep files)
        #### Assuming we're creating a new "features" folder within each session
        for sleep_file_path in glob.glob(f'{animal_folder}/*/sleep/*.csv.gz', recursive=True):

                ## Getting Session Path & Info
                session_folder_path = os.path.dirname(os.path.dirname(sleep_file_path))
                ## consider using uts.parse_bids_session(string: str)
                session_id = os.path.basename(session_folder_path)

        
                ## Get Features for this session for this animal
                sleep_df = pd.read_csv(sleep_file_path)

                ## Create Bar Graph 
                title = f'{animal_id} {session_id} Sleep Predictions'
                percentage_df = get_percentages(sleep_df)
                plt = plot_stacked_sleep(percentage_df, title)

                ## Saving Bar Graph 
                plot_name = f'{animal_id}_{session_id}_hypno-predictions.png'
                saving_dir = os.path.join(session_folder_path, 'sleep', plot_name)
                plt.savefig(saving_dir, dpi=300, bbox_inches='tight')
                plt.show() 

      
################## APPLY + VISUALIZE PCA ################## 

def apply_PCA_all(animal_folder, ref_title, ref_pca, n_components=pca_n, xlim=[-20, 30], ylim=[-30, 50]):
        '''
        '''
        ## Getting Animal ID
        ## consider using ut.sparse_bids_subject(string)
        animal_id = os.path.basename(animal_folder)
        all_sessions_ids = os.listdir(animal_folder)

        ## Looping through each session:
        for session_id in all_sessions_ids:

                # Skip over sessions that aren't sessions 
                if session_id == 'rec' or session_id =="start": continue

                # Features Folder 
                features_folder = os.path.join(animal_folder, session_id, 'features')
                features_files = os.listdir(features_folder)

                # Hypno Predictions Folder
                sleep_folder = os.path.join(animal_folder, session_id, 'sleep')
                sleep_file = os.listdir(sleep_folder)[0]                        ##should be a better way to do this 
                sleep_file_path = os.path.join(sleep_folder, sleep_file)
                sleep_df = pd.read_csv(sleep_file_path)

                # Reading Features.csv
                features_data = dict() 
                pattern = re.compile(r'chann-EEG\d+') ## careful with this line if name of files is changed
                for file in features_files:
                        match = re.search(pattern, file)
                        if match: extracted_part = match.group(0)
                        electrode = extracted_part 
                        features_data[electrode] = pd.read_csv(os.path.join(features_folder, file))

                dict_pca_df = apply_PCA(features_data, ref_pca, n_components)

                ## Visualizing 
                ## PCA 
                plot_title = f'{animal_id} {session_id} with ref {ref_title}'
                plot2D(animal_folder, dict_pca_df, sleep_df, plot_title, xlim, ylim)

                ## Sleep Prediction PDF
                ## Saving Bar Graph 
                plot_name = f'{animal_id}_{session_id}_hypno-predictions.png'
                saving_dir = os.path.join(sleep_folder, plot_name)
                percentages_df = get_percentages(sleep_df)
                plt = plot_stacked_sleep(percentages_df, plot_name)
                plt.savefig(saving_dir, dpi=300, bbox_inches='tight')
                plt.show() 


### THIS IS CODE FROM PREDICT.PY
### Idea: do something similar so that PCA is just run in same file? 

if __name__ == '__main__':
        ### TEST 1
        sleep_test = '/Volumes/Extreme SSD/MLA152/2023-12-16/sleep/sub-MLA152_ses-20231216T001047_hypno_predictions_df.csv.gz'
        test_sleep_df = pd.read_csv(sleep_test)
        for eeg_channel in test_sleep_df.columns:
                # print(test_sleep_df[eeg_channel].unique())
                print(eeg_channel, test_sleep_df[eeg_channel].value_counts()[4])
        # plt = plot_stacked_sleep(test_sleep_df, 'Testing')
        # plt.show()

#   parser = argparse.ArgumentParser()
#   parser.add_argument("--animal_id", required=True, help="Animal ID for constructing the base path. /path_to_storage/animal_id")
#   parser.add_argument("--date", required=True, 
#     type=datetime.date.fromisoformat,
#     help="Date that wants to be analized yyyy-mm-dd, used to construct folder path (/path_to_storage/animal_id/date/eeg/)")
#   parser.add_argument('--config_folder', help='Path to the config folder')
#   parser.add_argument("--epoch_sec", type=float, required=True, help="Epoch for sleep predictions in seconds. Ideally, it matches the classifier epoch_sec")
#   args = parser.parse_args()
#   config = read_config(args.config_folder)
#   sf = config['down_freq_hz']
#   console.log(f'Running `predict.py` with sf={sf} and epoch_sec={args.epoch_sec}')
#   run_and_save_predictions(animal_id = args.animal_id, date = args.date, epoch_sec = args.epoch_sec, config = config)  
