import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gzip
from rlist_files import list_files

# Define the path
path = "data/MLA152/between-sessions/pca"

# List all files in the directory
files = list_files(path, pattern = 'between-pca', full_names=True)

# Define the palette and state dictionary as per user's instruction
palette1 = ['#3F6F76FF', '#69B7CEFF',  '#F4CE4BFF']
state_dict = {
    0: "Wake",
    2: "NREM",
    4: "REM"
}

# Define a function to read and process each file
def process_file(filepath):
    df = pd.read_csv(filepath)
    # Transform the 'sleep' column to string based on the state_dict
    df['sleep'] = df['sleep'].map(state_dict)
    return df

# Process all files
dfs = dict()
for file in files:
    session_date = file.split('_')[1].split('-')[1]
    dfs[session_date] = process_file(file)

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), sharex=True, sharey=True)

for ax, session_date in zip(axes.flatten(), dfs):
    sns.scatterplot(data=dfs[session_date], x='PC1', y='PC2', hue='sleep', palette=palette1, ax=ax, legend=False, alpha=0.2)
    ax.set_title(f'Session: {session_date}')
    # Compute the proportions of each sleep category
    value_counts = dfs[session_date]['sleep'].value_counts(normalize=True) * 100
    text_str = '\n'.join([f'{state}: {value_counts[state]:.1f}%' for state in state_dict.values() if state in value_counts])
    
    # Add text annotation to the subplot
    ax.text(1.05, 0.95, text_str, transform=ax.transAxes, verticalalignment='top')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# Adjust layout
plt.tight_layout()

plt.show(block = False)

