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



################## LOAD DATA ################## 

# ## EEG Data
file_path = "/Users/angelescibils/Dropbox/Angie/School & Education/Massachusetts Institute of Technology (MIT)/23-24 Junior/24 Spring/EEG UROP/Testing data/eeg/sub-MLA154_ses-20231226T004732_desc-down10_eeg.csv"
animal = 'MLA154'
eeg_data = pd.read_csv(file_path)

## Hypno Prediction Data
sleep_path = '/Users/angelescibils/Dropbox/Angie/School & Education/Massachusetts Institute of Technology (MIT)/23-24 Junior/24 Spring/EEG UROP/Testing data/sleep/sub-MLA154_ses-20231226T004732_hypno_predictions_df.csv'
sleep_df = pd.read_csv(sleep_path)