# Script is for reading *.edf files exported from EEG DigiTrack Elmiko's amplifier with addition of information
# about appropriate sampling rate drawn from *.1 file, as well as information about the time of starting
# the signal recording took from file *.evx.

import glob
import struct
import xml.etree.ElementTree as etree
import pandas as pd
import numpy as np
import mne


def events_for_MNE(event_sample_indexes, event_id):
    """
    Process the events info untill it is in the format specified by MNE, i.e. ndarray with 3 columns.
    :param event_sample_indexes(Dict): key is the event name, value is an array of sample indexes.
    :param event_id(Dict): key is the event name, value is its' integer code.
    :return events(ndarray): 2-dimensional array of events as required by MNE (ex. http://martinos.org/mne/dev/auto_tutorials/plot_creating_data_structures.html).
    """
    events = pd.DataFrame(columns = ['sample_nr',  'code'])
    # Stack vertically all sample numbers for different events
    for event_label, sample_numbers in event_sample_indexes.items():
        tmp = pd.DataFrame(sample_numbers, columns = ['sample_nr'])
        tmp['code'] = event_id[event_label]
        # stack 
        events =  events.append(tmp)
   
    # Sort events chronologically
    events = events.sort_values(by = 'sample_nr')
    # Change to numpy array of ints
    events = events.as_matrix().astype('int')
    # MNE needs an extra column of zeros in the middle, it won't be used but has to be there
    events = np.insert(events, 1, 0, axis=1)
    
    return events


def parse_events(event_times, timestamp):
    """
    Prepare the events for MNE format. Convert event times to sample number relative to the first eeg sample.
    :param event_time_columns (DataFrame): each column represents the times of events of the same type. 
    :param timestamp(1-d array[datetime64]): datetime vector with times of eeg samples.  
    :return event_sample_indexes(Dict): key is the event name, value is an array of sample indexes
    """
    event_sample_indexes = {}
    
    #Replace the times with sample number relative to eeg signal.
    for col in event_times:
        time_col = pd.to_datetime(event_times[col].dropna())
        time_col = (time_col - timestamp[0]).as_matrix()
        samples = np.digitize(time_col.astype('float'), (timestamp - timestamp[0]).astype('float'), right = True)
        event_sample_indexes[col] = samples
    
    return event_sample_indexes


def MNE_Read_EDF(path):
    """
    Read .edf exported from digitrack using MNE library. Use digitrack files to recover the timestamp for eeg signal.
    :param path(str): a folder containing the following files: <filename>.edf, digi_log.xml, digi_binary.1
    :return: (mne.Raw) EEG signal in mne format. http://martinos.org/mne/dev/generated/mne.io.RawArray.html#mne.io.RawArray
    :return: timestamp vector.
    """
    edf_path = glob.glob(path +'*.edf')
    assert len(edf_path) == 1, edf_path # There can only be one edf in the directory.
    edf_path = edf_path[0] 
    
    raw_mne =  mne.io.read_raw_edf(edf_path,stim_channel = None, preload = True)
    #Reorder channels and drop unused ones.
    
    # Fix the sampling rate info saved in the original sygnal.edf. Our software does not save it with high precision in the .edf file, so we will replace it manually.
    exact_sr = get_exact_sampling_rate(path) # Get the high precision sampling rate
    raw_mne.info.update({'sfreq' : exact_sr }) # Update the mne info with the high precision sampling rate
    
    # Create a timestamp vector, so that event times can be expressed as a sample number.
    timestamp = exact_timestamp(path, raw_mne.n_times, exact_sr)
    
    #Correct the order of electrodes so the standard montage can be used.
    raw_mne = fix_montage(raw_mne, timestamp) 

    
    return raw_mne, timestamp


def get_exact_sampling_rate(path_to_file):
    """
    Read exact sampling rate of the recorded signal from *.1 file.

    Parameters
    ----------
    path_to_file : String
        Path to *.1 file.

    Returns
    -------
    sampling_rate : float
        Exact sampling rate extracted from *.1 file.
    """

    # Open *.1 file.
    binary_file = open(path_to_file, "rb")

    # Get exact sampling rate from *.1 file.
    power_of_two = 32
    binary_file.seek(490 + (89 * power_of_two))
    couple_bytes = binary_file.read(8)
    sampling_rate = struct.unpack("d", couple_bytes)[0]
    return sampling_rate


def read_xml(path_to_dir):
    """
    Read time of first EEG sample from *.evx file.
    :param path_to_dir: path to directory with only one *.evx file.
    :return: (DataFrame) time of rist EEG sample.
    """

    # Test wheter there is only one *.evx file in the directory.
    assert len(glob.glob(path_to_dir + "*.evx")) == 1

    # Open *.evx file and get time of the first EEG sample.
    with open(glob.glob(path_to_dir + "*.evx")[0], mode='r', encoding="utf-8") as xml_file:
        xml_tree = etree.parse(xml_file)
        root = xml_tree.getroot()

    # Store this information in a data-frame in a datetime/timestamp format.
    df = pd.DataFrame()
    for child_of_root in root:

        if child_of_root.attrib["strId"] == "Technical_ExamStart":

            time_event = child_of_root.find("event")

            # Timestamp in unix time.
            unix_time = time_event.attrib["time"]

            # Timestamp in DateTime.
            dt_time = time_event.find("info").attrib["time"]

            # TODO: Make sure the timestamps will be possible to compare between tz (utc) naive and tz aware formats.
            timezone_info = dt_time.find('+')
            df["UNIXTIME"] = pd.to_datetime([unix_time], unit="us").tz_localize("UTC") + \
                             pd.Timedelta(hours=int(dt_time[timezone_info + 1: dt_time.find('+') + 3]))

            df["DateTime"] = pd.to_datetime([dt_time], infer_datetime_format=True).tz_localize('UTC') + \
                             pd.Timedelta(hours=int(dt_time[timezone_info + 1: dt_time.find('+') + 3]))

    return df


def exact_timestamp(path_to_dir, n_samples, sampling_rate):
    """
    Create exact timestamp vector based on exact sampling rate and exact number of samples in EEG signal.
    :param path_to_dir: path to directory with only one *.evx file.
    :param n_samples: number of samples in EEG signal.
    :param sampling_rate: exact sampling rate of EEG signal.
    :return: timestampt vector.
    """
    
    # Exact sample duration in nanoseconds.
    exact_sample_ns = int(1000.0 / sampling_rate * 10**6)

    # Create time vector for nanosecond-precision date-times.
    timestamp = np.empty(n_samples, dtype="datetime64[ns]")

    # Set the first value using the first sample time saved by DigiTrack in .evx file.
    df = read_xml(path_to_dir)
    print("INFO", df)
    timestamp[0] = read_xml(path_to_dir)["DateTime"].iloc[0]

    # Create the time vector by adding sample duration to each next vector index.
    for i in range(n_samples - 1):
        timestamp[i+1] = timestamp[i] + np.timedelta64(exact_sample_ns, "ns")

    return timestamp


def fix_montage(raw, timestamp):
    """
    MNE can only use a standard montage if the electrodes come in specific order.
    :return (mne.raw): Raw object with channels in mne-specific order.
    """
    # These channels are not recorded during an EEG experiment.
    exclude = ['SaO2 SpO2', 'HR HR','Pulse Plet']
    if 'SaO2 SpO2' in raw.ch_names: raw.drop_channels(exclude)
    
    raw.info['ch_names'] = [name.split(' ')[-1] for name in raw.info['ch_names']]

    orig_names = raw.ch_names
    montage = mne.channels.read_montage(kind = 'standard_1020', ch_names=raw.info['ch_names'])
    
    data = raw.get_data()
    
    channels_dict = {}
        
    for channel_name, channel_data in zip(orig_names, data):
        channels_dict[channel_name] =  channel_data
    
    reordered_data = np.zeros(shape = data.shape)  
   
    for idx, channel_name in enumerate(montage.ch_names):
        reordered_data[idx, :] = channels_dict[channel_name]
        
    new_info = mne.create_info(
            ch_names= list(montage.ch_names),
            sfreq = raw.info['sfreq'],
            ch_types = ['eeg'] * len(list(montage.ch_names)),
            #meas_date = [timestamp[0], 0] # Time of the first sample and something else. Not well documented.
            )
    
    new_raw = mne.io.RawArray(reordered_data, new_info)
    
    new_raw.set_montage(montage)
    
    return new_raw
