# eeg_urop
Repository for EEG data analysis 

Important functions are in the PCA_functions.py file


Reading the files and running the functions are in the run_PCA.py file 

Comments for the User:
* Make sure to change the directory and file paths
* We're assuming BIDS convention: animal_id / session _id / features / sub_{animal_id}_{session_id}_chann-EEGi_features.csv.gz
* You'll need to import libraries from phdutils repository
* There are important adjustable variables to take into account: sampling frequency and epoch seconds.
