import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean, cosine
import os
import re
from rlist_files import list_files
import seaborn as sns
import numpy as np

def extract_electrode_name(file_name):
    """
    Extracts the electrode name from the given file name.
    Might fail with paths that contain more than one electrode name or the electrode name several times

    Parameters:
    file_name (str): The name of the file.

    Returns:
    str: The extracted electrode name if it matches the pattern 'chann-(EEG\d+)', otherwise None.
    """
    # Regular expression to match the electrode name pattern
    match = re.search(r'chann-(EEG\d+)', file_name)
    if match:
        return match.group(1)  # Returns the matched group (EEG1, EEG2, etc.)
    return None  #

def calculate_distances(df_list, file_names):
    """
    Calculates distances between dataframes.

    Parameters:
    df_list (list): A list of pandas DataFrames.
    file_names (list): A list of file names corresponding to the DataFrames.

    Raises:
    AssertionError: If not all dataframes have the same length.

    Returns:
    DataFrame: The calculated distances for all comparisons 
    """
    # Check that all dataframes have the same number of rows
    lengths = [len(df) for df in df_list]
    assert all(length == lengths[0] for length in lengths), "Not all dataframes have the same length."

    # Ensure all DataFrames are aligned by index (this step may be redundant if they are guaranteed to be aligned)
    for df in df_list:
        df.reset_index(drop=True, inplace=True)

    # Pre-allocate a list to store the distance data
    distance_records = []
    # Only one loop through the dataframes, leveraging enumerate for index and value
    for i, (df_i, file_i) in enumerate(zip(df_list, file_names)):
        electrode_i_name = extract_electrode_name(file_i)
        for j, (df_j, file_j) in enumerate(zip(df_list, file_names)):
            if i >= j:  # Skip redundant comparisons and self-comparison
                continue
            electrode_j_name = extract_electrode_name(file_j)
            print(f"Calculating distances for Electrode {electrode_i_name} vs {electrode_j_name}")
            for k in range(len(df_i)):  # Assuming all dataframes have the same length
                row_i = df_i.iloc[k].to_numpy()
                row_j = df_j.iloc[k].to_numpy()
                # Record each distance calculation
                distance_records.append({
                    'Electrode': electrode_i_name,
                    'Comparison': electrode_j_name,
                    'Euclidean Distance': euclidean(row_i, row_j),
                    'Cosine Distance': cosine(row_i, row_j),
                })
    # Convert the list of records to a DataFrame
    df_distances = pd.DataFrame(distance_records)
    return df_distances

def plot_distances(df_distances):
    # TODO: add plot from file so we don't compute
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # Euclidean Distance Boxplot
    sns.boxplot(x='Comparison', y='Euclidean Distance', data=df_distances, showfliers=False, ax=axes[0])
    axes[0].set_yscale('log')
    axes[0].set_title('Euclidean Distance Comparisons on Log Scale (Outliers Excluded)')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

    # Cosine Distance Boxplot
    sns.boxplot(x='Comparison', y='Cosine Distance', data=df_distances, showfliers=False, ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title('Cosine Distance Comparisons on Log Scale (Outliers Excluded)')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show(block = False)

    # Compute average Euclidean Distance for each electrode comparison
    avg_distances = df_distances.groupby(['Electrode', 'Comparison'])['Euclidean Distance'].mean().unstack()
    # Calculate relevant percentiles to set color scale limits, excluding extreme outliers
    # vmin, vmax = df_distances['Euclidean Distance'].quantile([0.10, 0.75])
    plt.figure(figsize=(12, 10))
    # Adjust heatmap creation to use vmin and vmax
    # add vmin=vmin, vmax=vmax
    sns.heatmap(avg_distances.apply(lambda x: np.log10(x) if np.issubdtype(x.dtype, np.number) else x), 
                annot=True, 
                fmt=".2f", 
                cmap="viridis",  
                cbar_kws={'label': 'Average Euclidean Distance (log10)'})
    plt.title('Adjusted Average Euclidean Distance Between Electrodes')
    plt.tight_layout()
    plt.show(block = False)

def save_distances(df_distances, output_path):
    if not os.path.isdir(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    df_distances.to_csv(output_path, index=False)
    

# Define the path to the directory containing the files
directory_path = 'data/MLA152/2023-12-11/features/'
# List all the CSV files in the directory
file_names = list_files(directory_path, pattern = "features.csv.gz", full_names = True)
# Load the datasets into a list of pandas DataFrames
dataframes = [pd.read_csv(file_name) for file_name in file_names]
df_distances = calculate_distances(df_list=dataframes, file_names = file_names)
output_dir = 'data/MLA127/2023-08-07/within-session/distances'
output_fn = "sub-MLA152_ses-2023-08-07_distances.csv.gz"
# save
save_distances(df_distances=df_distances, output_path = os.path.join(output_dir, output_fn))
plot_distances(df_distances)


if __name__ == "__main__":
    pass