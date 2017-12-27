"""
This script reads raw logs from the experiment and processes them so they can be used by MNE-python library.
     - clean_log() is called first inside the MNE_prepare_event. It selects a subset of information (event times) from the log and labels it with conditions info.
     - then MNE_prepare_events reorganizes this cleaned log in the format specified by MNE-python.
"""


import pandas as pd
import numpy as np  
import glob


def MNE_prepare_events(path, timestamp, include_keys = ['time'], exclude_keys = ['psychopy', 'start'], factor_keys = ['accuracy', 'trial_type'] ):
    """
    Prepare the events for MNE format.
    Convert event times to sample number relative to the first eeg sample.
    Process the events times and info untill it is in the format: np.ndarray with 3 columns.

    :param path (str): path to the top-most folder with results of a singe subject. ex: 'C:/Users/rcetnarski/Desktop/Nencki/EEG_data/Statki Wyniki/Vira/'
    :param timestamp(1-d array[datetime64]): datetime vector with times of eeg samples.  
    :param include_keys (list(str)): list of partial strings to identify usefull columns by
    :param exclude_keys (list(str)): strings that are useless columns that also have the usefull string. For example, we want to have ISI_time (identifeid by "time"), but we don't want ISI_time_psychopy (which also includes "time")
    :param factor_keys (list(str)): names of columns to create unique event_names, a.k.a. conditions for analysis.

    :return event_sample_indexes(Dict): key is the event name, value is an array of sample indexes
    """
    
    cleaned_log, event_id = clean_log(path, include_keys, exclude_keys, factor_keys)
    
    
    # This dictionary will contain arrays of sample indexes as values, and event labels as keys
    event_sample_indexes = {}
    
    #Replace the times with sample number relative to eeg signal.
    for col in cleaned_log:
        # Subtract the time of the first EEG sample from all event times
        time_diff = (cleaned_log[col].dropna() - timestamp[0]).as_matrix()
        # Find first indexes in EEG timestamp where event time is greater than EEG sample time.
        samples = np.digitize(time_diff.astype('float'), (timestamp - timestamp[0]).astype('float'), right = True)
        # Store the results in the dict
        event_sample_indexes[col] = samples
            
    
    # This DataFrame will have the events information in MNE format. 3 columns, first column is sample index, second is just 0's, third is event label (int)
    events = pd.DataFrame(columns = ['sample_nr',  'code'])
    # Stack vertically all sample numbers for different events
    for event_label, sample_numbers in event_sample_indexes.items(): # Iterate over dict
        # Create temporary data frame with event indexes from one type of event
        tmp = pd.DataFrame(sample_numbers, columns = ['sample_nr'])
        tmp['code'] = event_id[event_label]
        # stack all different event types in one main DataFrame
        events = events.append(tmp)
   
    # Sort events chronologically
    events = events.sort_values(by = 'sample_nr')
    # Change to numpy array of ints
    events = events.as_matrix().astype('int')
    # MNE needs an extra column of zeros in the middle, it won't be used but has to be there
    events = np.insert(events, 1, 0, axis=1)
        
    return events, event_id



    
def clean_log(path, include_keys, exclude_keys, factor_keys):
    """Select only the columns with event times and important experimental information (raw logs have a lot of redundant information)
       Do preprocessing: 
           -convert strings to datetime, 
           -create combined_condition column that will be used for analysis
       
        :param path (str): path to the top-most folder with results of a singe subject. ex: 'C:/Users/rcetnarski/Desktop/Nencki/EEG_data/Statki Wyniki/Vira/'
        :param include_keys (list(str)): list of partial strings to identify usefull columns by
        :param exclude_keys (list(str)): strings that are useless columns that also have the usefull string. For example, we want to have ISI_time (identifeid by "time"), but we don't want ISI_time_psychopy (which also includes "time")
        :param factor_keys (list(str)): names of columns to create unique event_names, a.k.a. conditions for analysis.
        
    :return factored_events_log (DatFrame): Columns are event times separated for each unique event name. ex. column is 'ISI_time_correct_control_horizontal'
    :return event_id(Dict): re-coded event names from strings to ints.

    """
    # Read the experiment log from a csv file
    assert len(glob.glob(path + "/*.csv")) == 1, 'problem with number of .csv files'
    log = pd.read_csv(glob.glob(path + "/*.csv")[0],parse_dates = True, index_col = 0, skiprows = 1, skipfooter = 1, engine='python')
    
    #Make a list of column names with times, by filtering the list of all original column names
    event_time_columns = [col_name for col_name in log.columns if any(substring in col_name for substring in include_keys)
                                                           and not any(substring in col_name for substring in exclude_keys)]
    
    # Apply calls function on all columns. Here we parse all columns from string to datetime
    log[event_time_columns] = log[event_time_columns].apply(pd.to_datetime, errors='raise')
    # Here we select only time columns from the original log
    events_log = log[event_time_columns]
    
    if factor_keys: # Check if there are any factors to split the event times by
    #Here we make a new column which will be used to group the times in conditions. Conditions can be made out of combinations of different factors, 
    #for example control, horizontal, and don't know would be one condition based on 3 factors.
        log[factor_keys[:-1]] = log[factor_keys[:-1]] +'_' # We add a '_' to factor values, so when whe combine them they are readable,ex. factor1_factor2
        events_log['combined_condition'] = log[factor_keys].sum(axis =1)# We create a new column by summing the factor columns. They will create unique labels for each condition. for example match_correct
        
        # Use the factors to split the timestamps for separate conditions into separate columns. This will help to convert them into MNE format.
        factored_events_log =  events_log.pivot(columns = 'combined_condition')
        factored_events_log.columns = ['_'.join(col) for col in factored_events_log.columns]

    else:
        factored_events_log = events_log# If there are no factors, just use the original time columns
    
    # Event types have to be encoded with ints starting from 1 for MNE format
    event_id = { event_name : idx + 1 for idx, event_name in enumerate(factored_events_log.columns)}
    
    return factored_events_log, event_id


