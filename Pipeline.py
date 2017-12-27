"""

"""


import glob
import mne
import numpy as np

import Events as ev
import ICA_preprocessing as ica
import UtilityReadEDF as ur


def parse_events():
    path = 'C:/Users/rcetnarski/Desktop/Nencki/EEG_data/Statki Wyniki/Vira/'
    raw, timestamp = ur.MNE_Read_EDF(path) # Read the EEG data
    mne_events, events_dict = ev.MNE_prepare_events(path, timestamp, factor_keys = []) # Read the experimental info 
    
    # Error in find eog epochs, it does not find any, byuuut whyyy, bekos rejecting?
    yy = ica.ica_artifact(raw, 'test', 'Fp1')
    
    
    return raw, timestamp, mne_events, events_dict


def check_sync_by_erps():
    raw, timestamp, mne_events, events_dict = parse_events()
    
    tmin =-1.0
    tmax = 1.0
    
    baseline=(None, 0)

    picks = mne.pick_types(raw.info, meg = False, eeg = True)
    
    key_epochs = mne.Epochs(raw, mne_events, events_dict['key_time'], tmin, tmax, picks = picks,
                     baseline = baseline,  proj = False,  preload=True)
    
    target_epochs = mne.Epochs(raw, mne_events, events_dict['target_time'], tmin, tmax, picks = picks,
                 baseline = baseline,  proj = False,  preload=True)

    probe_epochs = mne.Epochs(raw, mne_events, events_dict['probe_time'], tmin, tmax, picks = picks,
             baseline = baseline,  proj = False,  preload=True)

    
   # key_epochs.plot_image([0])
    target_epochs.plot_image(np.arange(1,19))
    #probe_epochs.plot_image(np.arange(1,19))
    
        




