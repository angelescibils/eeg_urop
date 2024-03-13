import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cosine
import os
import re

def extract_electrode_name(file_name):
    # Regular expression to match the electrode name pattern
    match = re.search(r'chann-(EEG\d+)', file_name)
    if match:
        return match.group(1)  # Returns the matched group (EEG1, EEG2, etc.)
    return None  #
    


# Define the path to the directory containing the files
directory_path = 'data/MLA152/2023-12-11/features/'

# List all the CSV files in the directory
file_names = [f for f in os.listdir(directory_path) if f.endswith('.csv.gz')]

# Load the datasets into a list of pandas DataFrames
dataframes = [pd.read_csv(os.path.join(directory_path, file_name)) for file_name in file_names]

# Ensure all DataFrames are aligned by index (this step may be redundant if they are guaranteed to be aligned)
for df in dataframes:
    df.reset_index(drop=True, inplace=True)

# Step 1: Identify all unique electrode names
unique_electrodes = set(extract_electrode_name(fn) for fn in file_names)

# Step 2: Initialize distances_data with all unique electrode names
distances_data = {ele: {'Euclidean': [], 'Cosine': [], 'Comparison': []} for ele in unique_electrodes if ele is not None}


# Calculate distances for each electrode against all others
for i in range(len(dataframes)):
    electrode_i_name = extract_electrode_name(file_names[i])
    for j in range(len(dataframes)):
        if i != j:
            electrode_j_name = extract_electrode_name(file_names[j])
            for k in range(len(dataframes[0])):
                row_i = dataframes[i].iloc[k]
                row_j = dataframes[j].iloc[k]
                distances_data[electrode_i_name]['Euclidean'].append(euclidean(row_i, row_j))
                distances_data[electrode_i_name]['Cosine'].append(cosine(row_i, row_j))
                distances_data[electrode_i_name]['Comparison'].append(electrode_j_name)

# Plotting
n_electrodes = len(file_names)
fig, axes = plt.subplots(n_electrodes, 2, figsize=(15, n_electrodes * 5), sharey='row') # 2 for Euclidean and Cosine

for i, (electrode, distances) in enumerate(distances_data.items()):
    for j, distance_type in enumerate(['Euclidean', 'Cosine']):
        sns.boxplot(x='Comparison', y=distance_type, 
                    data=pd.DataFrame({
                        distance_type: distances[distance_type],
                        'Comparison': distances['Comparison']
                    }), ax=axes[i, j])
        axes[i, j].set_title(f'{electrode} {distance_type} Distances')
        axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(), rotation=45, ha='right')
        axes[i, j].set_xlabel('Compared to Electrode')
        if j == 0:
            axes[i, j].set_ylabel('Distance')
        else:
            axes[i, j].set_ylabel('')

plt.tight_layout()
plt.show()
