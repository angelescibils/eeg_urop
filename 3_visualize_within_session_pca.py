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

#### testing 
## t

### NEW VERSION 

def plot_within_session(session_folder, xlim=[-20, 30], ylim=[-30, 50]):

    ## Get PCA dataframes
    dict_pca_df = dict()
    for pca_file in glob.glob(f"{session_folder}/within-session/pca/*.csv.gz"):
        ### test 
        pass 

    ## Get Sleep DF
    sleep_df = pd.DataFrame() 


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
    # plt.savefig(f'{animal_folder}/{title}_PCA.png',  dpi=300)
    # plt.show()


#### OLD VERSION
# def plot2D(animal_folder, dict_pca_df, sleep_df, title, xlim=[-20, 30], ylim=[-30, 50]):
#     '''
#     Plots PC1 vs. PC2 of each electrode 

#     Parameters
#     ----------
#     dict_pca_df: dict of "EEG X": py:class:`pandas.DataFrame`
#             A dictionary mapping electrode string names (i.e. 'EEG8') to their corresponding PCA transformed features dataframe 
#     sleep_df: py:class:`pandas.DataFrame`
#             Hypno predictions of each electrode channel         

#     Returns
#     -------
#     A 3x3 plot 
    
#     '''

#     print('Plotting...')
 
#     fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 8))


#     for i in range(3):
#         for j in range(3):
#             electrode = f'EEG{i * 3 + j + 1}'
#             merged_df = pd.concat([dict_pca_df[f'chann-{electrode}'], sleep_df[electrode]], axis=1)
            
#             # scatter_plot = plt.scatter(merged_df['PC1'], merged_df['PC2'], c=merged_df[electrode], cmap='viridis')
#             scatter = axes[i,j].scatter(merged_df['PC1'], merged_df['PC2'], c=merged_df[electrode], alpha=0.1)
#             axes[i, j].set_title(f'EEG{i * 3 + j + 1}')
#             axes[i, j].set_xlabel('Principal Component 1')
#             axes[i, j].set_ylabel('Principal Component 2')

#             axes[i, j].set_xlim(xlim)  # Adjust the limits as needed
#             axes[i, j].set_ylim(ylim)  # Adjust the limits as needed

            
#     legend_labels = {'$\\mathdefault{0}$': 'Wake','$\\mathdefault{2}$': 'NREM', '$\\mathdefault{4}$': 'REM'}
#     handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
#     fig.legend(handles, [legend_labels[label] for label in labels], title='Hypno Prediction', loc='upper right')

#     plt.suptitle(f'{title} PCA', fontsize=16, fontweight='bold')
# #     subtitle = f'Scatter Plot of PC1 vs PC2 with Sleep Stage Classification'
#     plt.tight_layout()
#     plt.savefig(f'{animal_folder}/{title}_PCA.png',  dpi=300)
#     plt.show()
