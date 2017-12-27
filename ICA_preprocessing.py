# -*- coding: utf-8 -*-
"""
Automated ICA cleaning of EOG artifact. Components are labeled as artifacts based on their correlation with EOG channel. 
Therefore no manual selection of artifact components is necessary.
"""

import numpy as np
import mne
from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs
import os
#num_cpu = '8' # Set as a string
#os.environ['OMP_NUM_THREADS'] = num_cpu

        
def ica_artifact(raw,subject_name, eog_channel = 'ExG1'):
    """Important - we can manually remove components by adding their index to eog_inds, or any list passed to exclude at ica.apply()"""
    raw.filter(1, 40,)# n_jobs=8)  # 1Hz high pass is often helpful for fitting ICA
    
    method = 'extended-infomax' #'fastica'  # for comparison with EEGLAB try "extended-infomax" here
    
    ica = ICA(method=method) # Note that ICA is a non-deterministic algorithm, if exactly the same results are required set seed using random_state = some integer
    
    result = ica.fit(raw) # Even though it says result is unused, ica.fit function has to be called.
    #result = ica.fit(raw, reject = {'eeg': 0.0004}) # Even though it says result is unused, ica.fit function has to be called.
    
    eog_average = create_eog_epochs(raw, ch_name = eog_channel).average()
    #eog_average = create_eog_epochs(raw, ch_name = eog_channel, reject = {'eeg': 0.0005}).average()

    eog_epochs = create_eog_epochs(raw,ch_name = eog_channel )  # get single EOG trials, i.e. individual blinks
    #eog_epochs = create_eog_epochs(raw,ch_name = eog_channel, reject = {'eeg': 0.0005} )  # get single EOG trials, i.e. individual blinks
    
    # here we had a problem that treshold was too high (z-score - 99-95-66 rule) for finding an eog component
    eog_inds, scores = ica.find_bads_eog(eog_epochs,threshold=2.0, ch_name =eog_channel)  # find via correlation
    
    corr_plot = ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components
    
    ica_sources = ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course
        
    # here we had a problem that there was an extra eog channel and dimensions didnt match
    sig_comps_plot = ica.plot_properties(eog_epochs, picks = np.arange(0, len(raw.ch_names[:-1])), psd_args={'fmax': 35.},
                    image_args={'sigma': 1.}) # Parameter to show only sig eog comps: , picks=eog_inds
    fig = ica.plot_components(inst=raw)
    ica.plot_sources(raw) 
    #return fig
    #fig[0].savefig('eog_comp_properties/' + subject_name + '_ica.png')
  
    #corr_plot.savefig('eog_comp_properties/'+subject_name + '_eog_corr.png')
    ica_sources.savefig('eog_sources/'+subject_name)
    
    #for ind, plot in enumerate(sig_comps_plot):
    #    plot.savefig('eog_comp_properties/'+subject_name + '/'+ subject_name + '_C%i'%ind)
        
    # here we clean the data from bad components
    ica.apply(inst = raw, exclude = eog_inds)
    
    
    return raw        

