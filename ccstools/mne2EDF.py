
def octaveEDF(filename, mnedata, mneevents = [], mneeventdict = []):
    """
    Purpose:        To write mne EEG data to EDF using octave
    
    Inputs:         filename    = name of out put .edf file
                    mnedata     = the data object of mne
                    mneevents   = the events array extracted using mne
                                    3 columns [latency, duration, code]
                    mneeventdict = dictionary for annotation codes
    
    Dependency:     Should have abrl_saveEDF.m code added 
                    to octave path
    
    Created on Sat Apr  4 11:58:30 2020
    
    @author: Arun Sasidharan
    """
    import oct2py
    octave = oct2py.Oct2Py()
    
    # Get values from mnedata
    EEGdata     = mnedata._data * 10e5
    srate       = mnedata.info.get('sfreq')
    chanlist    = mnedata.info.get('ch_names')
    
    # Get events
    if len(mneevents) > 0:
        if len(mneeventdict) >0:
            key_list = list(mneeventdict.keys()) 
            val_list = list(mneeventdict.values()) 
            event_type      = [str(key_list[val_list.index(i)]) for i in mneevents[:,2]]
            event_latency   = mneevents[:,0]/srate
            event_duration  = mneevents[:,1]/srate 
        else:
            event_type      = [str(i) for i in mneevents[:,2]]
            event_latency   = mneevents[:,0]/srate
            event_duration  = mneevents[:,1]/srate 
    else:
        event_type      = ['']
        event_latency   = 0
        event_duration  = 0
    
    
    header  = octave.abrl_createEDFheader(
                len(chanlist),srate,chanlist,
                event_type,event_latency,event_duration)
    data    = octave.abrl_createEDFdata(EEGdata) 
    
    octave.abrl_SaveEDF(filename, data, header)
