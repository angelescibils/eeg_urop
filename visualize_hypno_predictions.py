import pandas as pd
import matplotlib.pyplot as plt
import os 
import glob 

#### 

test_df = pd.DataFrame ({
        'EEG1': [0,2,4], 
        'EEG2': [0,2,4], 
        'EEG3': [0,2,4], 
        'EEG4': [0,2,4], 
         })

test_df.plot(
                        x='electrode', 
                        kind = 'bar',
                        stacked=True, 
                        title='TEST', 
                        colormap='viridis',
                        )
plt.show() 

def get_percentage_df(sleep_df):
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
                percentage_df = get_percentage_df(sleep_df)
                plt = plot_stacked_sleep(percentage_df, title)

                ## Saving Bar Graph 
                plot_name = f'{animal_id}_{session_id}_hypno-predictions.png'
                saving_dir = os.path.join(session_folder_path, 'sleep', plot_name)
                plt.savefig(saving_dir, dpi=300, bbox_inches='tight')
                plt.show() 



