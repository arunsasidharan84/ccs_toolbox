# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 14:06:08 2020

@author: Arun Sasidharan
"""
import numpy as np
from scipy.stats import trim_mean,iqr
from scipy.signal.windows import gaussian,hamming
from scipy.signal import convolve,find_peaks
from scipy.interpolate import UnivariateSpline
import pandas as pd
import sys

def bootstrapERP(epochdata,ntrials,niterations=500):
    ''' 
    non-parametric bootstrap procedure to derive ERP:
        - arguments:    
            epochdata   = number of trials (nTR) x number of channels (nCH) x number of time steps (nTIME)
            ntrials     = number of trials to average at each bootstrap
            niterations = number of boot strap iterations        
    '''
    nTR, nCH, nTIME = epochdata.shape 
    #epochdata 		= epochdata.transpose((1,2,0))    
    bootstrap_erps  = np.zeros((niterations,nCH, nTIME))     
    for iter_no in range(niterations):
        randomtrials    = np.random.randint(0,nTR,ntrials)
        epoch_cumsum    = trim_mean(epochdata[randomtrials,:,:],0.1,axis=0)              
        bootstrap_erps[iter_no,:,:] = epoch_cumsum
    return bootstrap_erps


def pca(data):
    ''' 
    Principle Component Analysis on an epoch:
        - arguments:    
            data        = number of channels (nCH) x number of time steps (nTIME)
        - returns:
            pc          = PCA component time series [ncomp x nsamples]
            pc_wt       = PCA component weights [ncomp x nchan]
            pc_resvar   = explained residual variance by each PCA component [ncomp x 1]
    '''
    nCH, nTIME  = data.shape
    
    # Substract mean (Make mean of each channel around zero)
    data        = data - data.mean(axis=1, keepdims=True)
    
    # Compute covariance matrix
    covar       = np.dot(data,data.transpose())/(nTIME-1)
    
    # Principle components analysis via eigenvalue decomposition
    pc_resvar,pc_wt = np.linalg.eig(covar)
    
    # Convert residual variance (eigen values) to percent change
    pc_resvar   = 100*pc_resvar/sum(pc_resvar) 
    
    # Sort PC in decending order of residual variance
    sort_indx   = np.argsort(pc_resvar)
    pc_resvar   = pc_resvar[sort_indx[::-1]]
    pc_wt       = pc_wt[:,sort_indx[::-1]]
    pc_wt       = pc_wt.transpose()
    
    # Compute the Principle components
    pc         = np.dot(pc_wt,data)
    
    return pc,pc_resvar,pc_wt


def generate_sinewave(frequency=10,amplitude=1,samplingrate=500,duration=1,phase=0):
    """    
    Generate Sine Wave
    
    Parameters
    ----------
    frequency :     [FLOAT] in Hz
    amplitude :     [FLOAT] in uV
    samplingrate :  [FLOAT] in Hz
    duration :      [FLOAT] in Secs
    phase :         [FLOAT] in degrees

    Returns
    -------
    sinewave :      [ARRAY] in uV
    timepoints :    [ARRAY] in Secs

    """
    timepoints  = np.linspace(0,duration,num=int(samplingrate*duration))
    sinewave    = amplitude*np.sin(2*np.pi*frequency*timepoints + (2*np.pi/360)*phase)
    return sinewave,timepoints


def generate_moreletwavelet(wavelet_freq,wavelet_ncycles,wavelet_srate):
    """
    Generate a single Morelet Wavelet

    Parameters
    ----------
    wavelet_freq :      [FLOAT] in Hz
    wavelet_ncycles :   [FLOAT] number of wavelet cycle
    wavelet_srate :     [FLOAT] in Hz

    Returns
    -------
    wavelet:            [ARRAY] in uV
    wavelet_timepoints: [ARRAY] in Secs

    """
    wavelet_phasestart  = -90
    wavelet_duration    = (1/wavelet_freq)*(wavelet_ncycles-(
        wavelet_ncycles - np.floor(wavelet_ncycles)))
    wavelet, wavelet_timepoints = generate_sinewave(
        wavelet_freq, 1, wavelet_srate,wavelet_duration,wavelet_phasestart)
    
    wavelet_window      = gaussian(len(wavelet),len(wavelet)/6)
    #wavelet_window = hamming(len(wavelet))
    wavelet             = wavelet * wavelet_window
    
    # Scale the Wavelet in Frequency domain
    waveletFFT          = np.fft.fft(wavelet)/max(np.fft.fft(wavelet))/2
    wavelet             = np.real(np.fft.ifft(waveletFFT))
    return wavelet,wavelet_timepoints

def smooth(y, box_pts):
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth

def wavelet_amplitudephase(data,samplingrate,frequency,ncycles):
    """
    Extract instantaneous amplitude and phase using Morelet Wavelet

    Parameters
    ----------
    data :              [ARRAY] single channel EEG segment
    samplingrate :      [INT] in Hz
    frequency :         [FLOAT] in Hz
    ncycles :           [FLOAT] number of wavelet cycle

    Returns
    -------
    envelope_pos:       [ARRAY] instantaneous positive amplitude (same size as data)
    envelope_neg:       [ARRAY] instantaneous negative amplitude (same size as data)
    
    wavelet_convn:      [ARRAY] wavelet filtered data (same size as data)

    """
    wavelet,wavelet_timepoints = generate_moreletwavelet(
        frequency,ncycles,samplingrate)
    wavelet_convn = convolve(data,wavelet,mode='same')     
    
    # Extract instantaneous measures from peak and trough detection  
    envelope_pos,envelope_neg,phase_values,peakindx_pos,peakindx_neg = peaktroughmeasures(wavelet_convn) 
    
    return envelope_pos,envelope_neg,phase_values,wavelet_convn
    
    
def peaktroughmeasures(wave):
    #%% Get the peaks
    peakindx = np.diff(wave)
    peakindx = np.sign(peakindx)
    peakindx_pos = np.diff(peakindx)<0 # positive peaks (they are negative in double differential)
    peakindx_neg = np.diff(peakindx)>0 # negative peaks (they are positive in double differential)
    
    #%% Get positive amplitude envelope
    envelope_pos = np.concatenate([[True],peakindx_pos,[True]]) #add boolian at start and end to make up the length
    envelope_pos = np.flatnonzero(envelope_pos)
    envelope_pos = UnivariateSpline(
        envelope_pos, wave[envelope_pos], s=0, k=3)
    envelope_pos = envelope_pos(np.arange(len(wave)))
    
    #%% Get positive amplitude envelope
    envelope_neg = np.concatenate([[True],peakindx_neg,[True]]) #add boolian at start and end to make up the length
    envelope_neg = np.flatnonzero(envelope_neg)
    envelope_neg = UnivariateSpline(
        envelope_neg, wave[envelope_neg], s=0, k=3)
    envelope_neg = envelope_neg(np.arange(len(wave)))
    
    #%% Get instantaneous phase
    phase_values = np.zeros(wave.shape) # Initialize
    peakindx_pos = np.flatnonzero(peakindx_pos)+1 # Convert to indices
    peakindx_neg = np.flatnonzero(peakindx_neg)+1 # Convert to indices
    
    # If peak starts early, fill phase for incomplete first wave cycle from next peak to trough
    if peakindx_neg[0] > peakindx_pos[0]:
        peakstartearly          = True
        phase_indx1             = np.arange(0,peakindx_pos[0]+1)
        phase_indx2             = np.arange(peakindx_pos[0],peakindx_neg[0])
        phaseindx_all           = np.unique(np.concatenate([phase_indx1,phase_indx2]))
        missingvalues           = np.linspace(-np.pi,np.pi,2*len(phase_indx2))
        if len(missingvalues)>=len(phaseindx_all):
            phase_values[phaseindx_all] = missingvalues[-len(phaseindx_all):]
    else:
        peakstartearly          = False

    
    # If peak ends later, fill phase for incomplete last cycle from previous trough to peak
    if peakindx_neg[-1] < peakindx_pos[-1]:
        peakendearly            = False
        phase_indx1             = np.arange(peakindx_neg[-1],peakindx_pos[-1]+1)
        phase_indx2             = np.arange(peakindx_pos[-1],len(phase_values))
        phaseindx_all           = np.unique(np.concatenate([phase_indx1,phase_indx2]))
        missingvalues           = np.linspace(-np.pi,np.pi,2*len(phase_indx1))
        if len(missingvalues)>=len(phaseindx_all):
            phase_values[phaseindx_all] = missingvalues[:len(phaseindx_all)]
    else:
        peakendearly            = True
    
    # Complete the full cycles between two consequtive troughs
    for trough_no in range(len(peakindx_neg)-1): # Each cycle is between two troughs
        if peakstartearly:
            nextpeak = peakindx_pos[trough_no+1]
        else:
            nextpeak = peakindx_pos[trough_no]

        phase_indx1 = np.arange(peakindx_neg[trough_no],nextpeak+1)       # trough to peak
        phase_indx2 = np.arange(nextpeak,peakindx_neg[trough_no+1])   # peak to next trough
        phase_values[phase_indx1] = np.linspace(-np.pi,0,len(phase_indx1))
        phase_values[phase_indx2] = np.linspace(0,np.pi,len(phase_indx2))
        phaseindx_all = np.unique(np.concatenate([phase_indx1,phase_indx2]))
        
        # If peak starts later, compute phase for incomplete first wave cycle from first trough to peak
        if trough_no == 0 and not peakstartearly:
            n_missingvalues = peakindx_neg[0]
            if len(phaseindx_all)>n_missingvalues:
                phase_values[:peakindx_neg[0]] = phase_values[phaseindx_all+1][::-1][-n_missingvalues:]
   
        
        # If peak ends early, compute phase for incomplete last wave cycle from last trough to peak
        if trough_no == len(peakindx_neg)-2 and peakendearly:
            n_missingvalues = len(phase_values)-peakindx_neg[-1]
            if len(phaseindx_all)>n_missingvalues:
                phase_values[peakindx_neg[-1]:] = phase_values[phaseindx_all][:n_missingvalues]

    return envelope_pos,envelope_neg,phase_values,peakindx_pos,peakindx_neg
    
    

def detectdeltawave(data_filt,srate,
                    dur_neg=(0.3,1.5),dur_pos=(0.1,1.0),
                    amp_neg=(40, 300), amp_pos=(10, 150), amp_ptp=(75,700)):
    """
    Delta wave down state detection based on: YASA algorithm:
        https://github.com/raphaelvallat/yasa/blob/master/notebooks/05_sw_detection.ipynb
    ----------
    filtereddata : band pass filtered data in delta range

    Returns
    -------
    None.

    """
    times = np.linspace(0,len(data_filt)/srate,num=len(data_filt))
    
    # Negative peaks with value comprised between -40 to -300 uV
    idx_neg_peaks, _ = find_peaks(-1 * data_filt, height=amp_neg)
    
    # Positive peaks with values comprised between 10 to 150 uV
    idx_pos_peaks, _ = find_peaks(data_filt, height=amp_pos)
        
    try:
        # For each negative peak, we find the closest following positive peak
        pk_sorted         = np.searchsorted(idx_pos_peaks, idx_neg_peaks)
        closest_pos_peaks = idx_pos_peaks[pk_sorted] - idx_neg_peaks
        closest_pos_peaks = closest_pos_peaks[np.nonzero(closest_pos_peaks)]
        
        idx_pos_peaks   = idx_neg_peaks + closest_pos_peaks
        
        # Now we check that the total PTP amplitude is within our bounds (75 to 400 uV)
        sw_ptp          = np.abs(data_filt[idx_neg_peaks]) + data_filt[idx_pos_peaks]
        good_ptp        = np.logical_and(sw_ptp > amp_ptp[0], sw_ptp < amp_ptp[1])
        
        # Remove the slow-waves with peak-to-peak ampitude outside the bounds
        sw_ptp = sw_ptp[good_ptp]
        idx_neg_peaks = idx_neg_peaks[good_ptp]
        idx_pos_peaks = idx_pos_peaks[good_ptp]
        
        # Then we check the negative and positive phase duration. To do so,
        # we first need to compute the zero crossings of the filtered signal:
        pos             = data_filt > 0
        npos            = ~pos
        zero_crossings  = ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
        
        # Safety check: Make sure that there is a zero-crossing after the last detected peak
        if zero_crossings[-1] < max(idx_pos_peaks[-1], idx_neg_peaks[-1]):
            # If not, append the index of the last peak
            zero_crossings = np.append(zero_crossings,
                                       max(idx_pos_peaks[-1], idx_neg_peaks[-1]))
            
        # For each negative peak, we find the previous and following zero-crossings
        neg_sorted          = np.searchsorted(zero_crossings, idx_neg_peaks)
        previous_neg_zc     = zero_crossings[neg_sorted - 1] - idx_neg_peaks
        following_neg_zc    = zero_crossings[neg_sorted] - idx_neg_peaks
        
        # And from that we calculate the duration of the negative phase
        neg_phase_dur       = (np.abs(previous_neg_zc) + following_neg_zc) / srate
            
        # For each positive peak, we find the previous and following zero-crossings
        pos_sorted          = np.searchsorted(zero_crossings, idx_pos_peaks)
        previous_pos_zc     = zero_crossings[pos_sorted - 1] - idx_pos_peaks
        following_pos_zc    = zero_crossings[pos_sorted] - idx_pos_peaks
        
        # And from that we calculate the duration of the positive phase
        pos_phase_dur       = (np.abs(previous_pos_zc) + following_pos_zc) / srate
        
        # Now we can start computing the properties of each detected slow-waves
        sw_start            = times[idx_neg_peaks + previous_neg_zc]
        sw_end              = times[idx_pos_peaks + following_pos_zc]
        sw_dur              = sw_end - sw_start  # Same as pos_phase_dur + neg_phase_dur
        sw_midcrossing      = times[idx_neg_peaks + following_neg_zc]
        sw_idx_neg,sw_idx_pos = times[idx_neg_peaks], times[idx_pos_peaks]
        sw_slope            = sw_ptp / (sw_midcrossing - sw_idx_neg)  # Slope between peak trough and midcrossing
            
        # Finally we apply a set of logical thresholds to exclude "bad" slow waves
        good_sw = np.logical_and.reduce((
                                        # Data edges
                                        previous_neg_zc != 0,
                                        following_neg_zc != 0,
                                        previous_pos_zc != 0,
                                        following_pos_zc != 0,
                                        # Duration criteria
                                        neg_phase_dur > dur_neg[0],
                                        neg_phase_dur < dur_neg[1],
                                        pos_phase_dur > dur_pos[0],
                                        pos_phase_dur < dur_pos[1],
                                        # Sanity checks
                                        sw_midcrossing > sw_start,
                                        sw_midcrossing < sw_end,
                                        sw_slope > 0,
                                        ))
        
        # Create the dataframe
        events = pd.DataFrame({'Start': sw_start,
                               'NegPeak': sw_idx_neg,
                               'MidCrossing': sw_midcrossing,
                               'PosPeak': sw_idx_pos,  
                               'End': sw_end, 
                               'Duration': sw_dur,
                               'ValNegPeak': data_filt[idx_neg_peaks], 
                               'ValPosPeak': data_filt[idx_pos_peaks], 
                               'PTP': sw_ptp, 
                               'Slope': sw_slope, 
                               'Frequency': 1 / sw_dur,
                                })[good_sw]
        
        # Remove all duplicates and reset index
        events.drop_duplicates(subset=['Start'], inplace=True, keep=False)
        events.drop_duplicates(subset=['End'], inplace=True, keep=False)
        events.reset_index(drop=True, inplace=True)
        return events
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
        pass

def detecterppeak(erpdata,times,type_peak="neg",lat_peak=(0.3,1.0),
                  dur_peak=(0.0,1.0),amp_peak=(-100, 100)):
    """
    ERP peak detection based on: YASA algorithm:
        https://github.com/raphaelvallat/yasa/blob/master/notebooks/05_sw_detection.ipynb
    ----------
    erpdata : single channel ERP data

    Returns
    -------
    None.

    """
    # breakpoint()
    srate = round(1/np.diff(times)[0])
    
    if type_peak=="neg":
        scale = -1
    else:
        scale = 1
        
    # Target peaks
    idx_tar_peaks, _ = find_peaks(scale * erpdata)
    
    # Other peaks
    idx_oth_peaks, _ = find_peaks(-scale * erpdata)
    
    # Compute the zero crossings
    pos             = erpdata > 0
    npos            = ~pos
    zero_crossings  = ((pos[:-1] & npos[1:]) | (npos[:-1] & pos[1:])).nonzero()[0]
    
    # Safety check: Make sure that there is a zero-crossing after the last detected peak
    if zero_crossings[-1] < max(idx_oth_peaks[-1], idx_tar_peaks[-1]):
        # If not, append the index of the last peak
        zero_crossings = np.append(
            zero_crossings,max(idx_oth_peaks[-1], idx_tar_peaks[-1]))
    if idx_oth_peaks[-1] < idx_tar_peaks[-1]:
        # If not, append the index of the last peak
        idx_oth_peaks = np.append(
            idx_oth_peaks,max(zero_crossings[-1], idx_tar_peaks[-1]))
    
    try:
        
        # Now check and remove peaks are within our latency bounds
        good_peaks      = np.logical_and(
            times[idx_tar_peaks] > lat_peak[0], 
            times[idx_tar_peaks] < lat_peak[1])
        idx_tar_peaks = idx_tar_peaks[good_peaks]
        
        # For each target peak, find the previous and following zero-crossings
        zc_sorted     = np.searchsorted(zero_crossings,idx_tar_peaks,side='left')        
        previous_zc   = zero_crossings[zc_sorted - 1] - idx_tar_peaks
        following_zc  = zero_crossings[zc_sorted] - idx_tar_peaks
        
        # For each target peak, find the previous and following peaks
        oth_sorted          = np.searchsorted(idx_oth_peaks, idx_tar_peaks,side='left')         
        previous_oth_peak   = idx_oth_peaks[oth_sorted - 1] - idx_tar_peaks
        following_oth_peak  = idx_oth_peaks[oth_sorted] - idx_tar_peaks
        
        # Calculate the peak parameters 
        # peak_start  = times[np.amax([[idx_tar_peaks + previous_zc],
        #                              [idx_tar_peaks + previous_oth_peak]],axis=0)[0]]
        # peak_end    = times[np.amin([[idx_tar_peaks + following_zc],
        #                              [idx_tar_peaks + following_oth_peak]],axis=0)[0]]
        # peak_dur    = peak_end - peak_start                
        
        
        
        peak_start  = []
        peak_end    = []
        peak_auc    = []
        for i in range(len(idx_tar_peaks)):
            if previous_zc[i] < previous_oth_peak[i]:
                if np.sign(erpdata[idx_tar_peaks[i]]) == scale:
                    previous_zc[i] = previous_oth_peak[i]
                else:
                    previous_zc[i] = 0
            if following_zc[i] > following_oth_peak[i]:
                if np.sign(erpdata[idx_tar_peaks[i]]) == scale:
                    following_zc[i] = following_oth_peak[i]
                else:
                    following_zc[i] = 0
            peak_start.append(times[idx_tar_peaks[i] + previous_zc[i]])
            peak_end.append(times[idx_tar_peaks[i] + following_zc[i] + 1])
            
            peak_indx = np.where(np.logical_and(
                times>peak_start[i],times<=peak_end[i]))[0]
            
            # Compute area under curve
            # peak_auc.append(np.trapz(erpdata[peak_indx]))
            peak_auc.append(np.sum(erpdata[peak_indx]))
        
        peak_start  = np.array(peak_start)
        peak_end    = np.array(peak_end)
        peak_auc    = np.array(peak_auc)
        peak_dur    = peak_end - peak_start 
        peak_lat    = times[idx_tar_peaks]
        peak_amp    = erpdata[idx_tar_peaks]

        # Apply a set of logical thresholds to exclude "bad" slow waves
        good_peaks = np.logical_and.reduce((
                                        # Duration criteria
                                        peak_dur >= dur_peak[0],
                                        peak_dur <= dur_peak[1],
                                        # Amplitude criteria
                                        scale*peak_amp >= amp_peak[0],
                                        scale*peak_amp <= amp_peak[1],
                                        ))
        
        # Create the dataframe
        events = pd.DataFrame({'Start': peak_start,
                               'PeakAmplitude': peak_amp,
                               'PeakLatency': peak_lat,
                               'AreaUnderCurve': peak_auc,  
                               'End': peak_end, 
                               'Duration': peak_dur,
                                })[good_peaks]
        
        # Remove all duplicates and reset index
        events.drop_duplicates(subset=['Start'], inplace=True, keep=False)
        events.drop_duplicates(subset=['End'], inplace=True, keep=False)
        events.reset_index(drop=True, inplace=True)
        return events
    except:
        print("Oops!", sys.exc_info()[0], "occurred.")
        raise
        pass


    return events
