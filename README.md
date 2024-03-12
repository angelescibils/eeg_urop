# eeg_urop
Repository for EEG data analysis 


## File structure

We are following BIDS folder structure
The current file structure with `subjects/sessions/filetypes`. For example: `animal_id/session_id/features/sub_{animal_id}_{session_id}_chann-EEGi_features.csv.gz`.
Here's an example of a The current folder structure for one animal with one session.
The code organization is likely to change. The `data/` folder is not present in this repository due to size.

```
.
├── data
│   └── MLA152
│       └── 2023-12-11
│           └── features
│               ├── sub-MLA152_ses-20231211_chann-EEG1_features.csv.gz
│               ├── sub-MLA152_ses-20231211_chann-EEG2_features.csv.gz
│               ├── sub-MLA152_ses-20231211_chann-EEG3_features.csv.gz
│               ├── sub-MLA152_ses-20231211_chann-EEG4_features.csv.gz
│               ├── sub-MLA152_ses-20231211_chann-EEG5_features.csv.gz
│               ├── sub-MLA152_ses-20231211_chann-EEG6_features.csv.gz
│               ├── sub-MLA152_ses-20231211_chann-EEG7_features.csv.gz
│               ├── sub-MLA152_ses-20231211_chann-EEG8_features.csv.gz
│               └── sub-MLA152_ses-20231211_chann-EEG9_features.csv.gz
├── get_accu_features.py
├── PCA_functions.py
├── README.md
└── run_PCA.py
```

Important functions are in the PCA_functions.py file


Reading the files and running the functions are in the run_PCA.py file 

## Current issues

* Paths are hard-coded, needs to be handled a bit better
* There are some hard-coded variables (sampling frequency and epoch_sec = 2.5). 
    * Sampling frequency for Accusleep data is 512 Hz. 
    * Sampling frequency for our data is 100 Hz.
