# Created by pdzwiniel at 2017-09-17.
# Contact: piotr.dzwiniel@gmail.com.
# This file is a part of EEG-IO project.

from EEGDigiTrack import UtilityReadEDF as ur_edf
import mne
import numpy as np
import scipy.io as sio

# Specify paths to files.
path_to_1_file = "ExampleData/data.1"
path_to_evx_file = "ExampleData/data.evx"
path_to_edf_unfiltered_file = "ExampleData/data-unfiltered.edf"
path_to_edf_filtered_file = "ExampleData/data-filtered-3-70.edf"

# Get exact sampling rate of the recorded EEG.
# (in default *.edf file exported from EEGDigiTrack software (Elmiko) contains integer sampling rate value,
# which is inaccurate; the accurate float value of the sampling rate is stored in *.1 file)
exact_samp_rate = ur_edf.get_exact_sampling_rate(path_to_1_file)

# Load *.edf files.
raw_mne_unfiltered = mne.io.read_raw_edf(path_to_edf_unfiltered_file, stim_channel=None, preload=True)
raw_mne_filtered = mne.io.read_raw_edf(path_to_edf_filtered_file, stim_channel=None, preload=True)

# Replace old sampling rate with the new one.
raw_mne_unfiltered.info.update({"sfreq": exact_samp_rate})
raw_mne_filtered.info.update({"sfreq": exact_samp_rate})

# Save data as *.fif files (uncomment lines below if needed).
raw_mne_unfiltered.save("ExampleData/edf-raw-unfiltered.fif")
raw_mne_filtered.save("ExampleData/edf-raw-filtered.fif")

# Save data as *.npy files (uncomment lines below if needed).
data_unfiltered, _ = raw_mne_unfiltered[:]  # Get data from MNE-Raw structure as ndarray.
data_filtered, _ = raw_mne_filtered[:]

np.save("ExampleData/edf-raw-unfiltered.npy", data_unfiltered)  # Save ndarray into *.npy file.
np.save("ExampleData/edf-raw-filtered.npy", data_filtered)

# Save data as *.mat files (uncomment lines below if needed).
sio.savemat("ExampleData/edf-raw-unfiltered.mat", mdict={"data": data_unfiltered})  # Save ndarray into *.mat file.
sio.savemat("ExampleData/edf-raw-filtered.mat", mdict={"data": data_filtered})
