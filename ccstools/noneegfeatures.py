#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:31:11 2024

@author: arun
"""
#%% Import libraries
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


#%% ECG parameters
def extract_ecgfeatures(
        data,srate,show=False,
        ):
    ''' 
    Extract ECG parameters
    
    Parameters
    ----------
    data            : [ARRAY] ECG data [ntime]
    srate           : [FLOAT] in Hz    

    Returns
    -------
    ecg_rate        : [ARRAY] instantaneous heart rate [ntime]
    edr_data        : [ARRAY] ECG derived Resp data [ntime]
    rpeaks          : [DATAFRAME] samples with r peaks 
    info            : [DICT] r peaks info 
    '''   
    if abs(np.percentile(data,85)) < abs(np.percentile(data,15)):
        scalingfactor = 1
    else:
        scalingfactor = 1
    rpeaks, info = nk.ecg_peaks(data*scalingfactor, sampling_rate=srate)
    ecg_rate = nk.signal_rate(rpeaks, sampling_rate=srate, desired_length=len(rpeaks))
    edr_data = nk.ecg_rsp(ecg_rate, sampling_rate=srate)
    
    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20,8))
        plt.subplot(3,1,1)
        plt.plot(data)
        plt.plot(np.where(rpeaks)[0],data[np.where(rpeaks)[0]],'o')
        plt.subplot(3,1,2)        
        plt.plot(ecg_rate)
        plt.legend(['Heart rate'])
        plt.subplot(3,1,3)
        plt.plot(edr_data)
        plt.legend(['ECG-derived Resp'])
        
    return ecg_rate,edr_data,rpeaks,info

#%% HRV parameters
def extract_hrvfeatures(
        data,srate,show=False,
        featurelist = ['time','frequency','nonlinear','rsa','rqa'],
        ):
    ''' 
    Extract HRV parameters
    
    Parameters
    ----------
    data            : [ARRAY] ECG data [ntime]
    srate           : [FLOAT] in Hz    

    Returns
    -------
    hrvfeatures_df  : 
    signals :     
    '''   
    if abs(np.percentile(data,85)) < abs(np.percentile(data,15)):
        scalingfactor = -1
    else:
        scalingfactor = 1
    
    signals, __     = nk.bio_process(data*scalingfactor, sampling_rate=srate)
    
    df_list = []
    if 'time' in featurelist:
        df_list.append(nk.hrv_time(signals, sampling_rate=srate))
    
    if 'frequency' in featurelist:
        df_list.append(nk.hrv_frequency(signals, sampling_rate=srate))
        
    if 'nonlinear' in featurelist:
        df_list.append(nk.hrv_nonlinear(signals, sampling_rate=srate))
        
    if 'rsa' in featurelist:
        df_list.append(nk.hrv_rsa(signals, sampling_rate=srate))
        
    if 'rqa' in featurelist:
        df_list.append(nk.hrv_rqa(signals, sampling_rate=srate))
    
    # Compile all features
    hrvfeatures_df         = pd.concat(df_list, axis=1).reset_index()
    hrvfeatures_df         = hrvfeatures_df.drop(['index'],axis=1)
    
    return hrvfeatures_df,signals
    

#%% Resp parameters
def compute_respfeatures(
        data,srate,smoothingfactor=None,
        ):
    ''' 
    Compute Resp features
    
    Parameters
    ----------
    data            :       [ARRAY] RESP data [ntime]
    srate           :       [FLOAT] in Hz    
    smoothingfactor :       [INT] Default is 2*srate

    Returns
    -------
    respfeatures_df  : 
        Resp_Rate_Mean,Resp_Rate_SD,Resp_DurRatio,Resp_AmpRatio,Resp_AreaRatio,
        Resp_cycleDur,Resp_InDur,Resp_ExDur,Resp_InAmp,Resp_ExAmp,
        Resp_InArea,Resp_ExArea,Resp_ncycles
    data : smoothened resp data
    '''   
    
    # Smooth the data
    if smoothingfactor is None:
        smoothingfactor = int(srate*2)
    data = savgol_filter(data, smoothingfactor, 3)
    
    # Get peaks and troughs
    respsignal, info_resp = nk.rsp_process(data, sampling_rate = srate)
    resp_troughs     = info_resp['RSP_Troughs']
    resp_peaks       = info_resp['RSP_Peaks']
    resp_cleaned     = respsignal['RSP_Clean']
    
    # Get durations of all Inspirations and Expirations 
    resp_InDur = resp_peaks - resp_troughs
    resp_ExDur = resp_troughs[1:] - resp_peaks[:-1]
    resp_DurRatio = resp_InDur[:-1] / resp_ExDur
    
    resp_InDur = resp_InDur/srate
    resp_Exdur = resp_ExDur/srate
    resp_cycleDur = resp_InDur[:-1] + resp_Exdur
    
    # Get amplitudes of all Inspirations and Expirations 
    resp_InAmp = []
    resp_ExAmp = []
    for i in range(len(resp_peaks)):
        if i<len(resp_troughs):
            resp_InAmp.append(resp_cleaned[resp_peaks[i]] - resp_cleaned[resp_troughs[i]])
        if i<len(resp_troughs)-1:
            resp_ExAmp.append(resp_cleaned[resp_peaks[i]] - resp_cleaned[resp_troughs[i+1]])
    resp_InAmp= np.array(resp_InAmp)
    resp_ExAmp= np.array(resp_ExAmp)
    resp_AmpRatio = resp_InAmp[:-1]/resp_ExAmp
    
    # Get areas of all Inspirations and Expirations 
    resp_InArea = []
    resp_ExArea = []
    for i in range(len(resp_troughs)-1):
        trough1 = resp_troughs[i]
        trough2 = resp_troughs[i+1]
        peak1 = resp_peaks[i]
        n1 = int(peak1 - trough1) + 1
        n2 = int(trough2 - peak1) + 1
        trough_peak_x = np.linspace(trough1,peak1,n1)
        peak_trough_x = np.linspace(peak1,trough2,n2)
        if resp_cleaned[trough1]<0:
            trough_to_peak_y = np.squeeze(resp_cleaned[trough1:peak1+1]) + abs(resp_cleaned[trough1])
        else:
            trough_to_peak_y = np.squeeze(resp_cleaned[trough1:peak1+1]) - abs(resp_cleaned[trough1])
        
        area_inhale = np.trapz(trough_to_peak_y, trough_peak_x, dx = 1)
        resp_InArea.append(area_inhale)
        
        if resp_cleaned[trough2]<0:
            peak_to_trough_y = np.squeeze(resp_cleaned[peak1:trough2+1]) + abs(resp_cleaned[trough2])
        else:
            peak_to_trough_y = np.squeeze(resp_cleaned[peak1:trough2+1]) - abs(resp_cleaned[trough2])
        
        area_exhale = np.trapz(peak_to_trough_y,peak_trough_x, dx = 1)
        resp_ExArea.append(area_exhale)
    resp_InArea = np.array(resp_InArea)
    resp_ExArea = np.array(resp_ExArea)
    resp_AreaRatio = resp_InArea/resp_ExArea
    
    respfeatures_df  = pd.DataFrame({
        'Resp_Rate_Mean':   np.mean(60/np.percentile(resp_cycleDur,range(30,70))),
        'Resp_Rate_SD':     np.std(60/np.percentile(resp_cycleDur,range(30,70))),
        'Resp_DurRatio':    np.mean(np.percentile(resp_DurRatio,range(30,70))),
        'Resp_AmpRatio':    np.mean(np.percentile(resp_AmpRatio,range(30,70))),
        'Resp_AreaRatio':   np.mean(np.percentile(resp_AreaRatio,range(30,70))),
        'Resp_cycleDur':    np.mean(np.percentile(resp_cycleDur,range(30,70))),
        'Resp_InDur':       np.mean(np.percentile(resp_InDur,range(30,70))),
        'Resp_ExDur':       np.mean(np.percentile(resp_ExDur,range(30,70))),
        'Resp_InAmp':       np.mean(np.percentile(resp_InAmp,range(30,70))),
        'Resp_ExAmp':       np.mean(np.percentile(resp_ExAmp,range(30,70))),
        'Resp_InArea':      np.mean(np.percentile(resp_InArea,range(30,70))),
        'Resp_ExArea':      np.mean(np.percentile(resp_ExArea,range(30,70))),
        'Resp_ncycles':     np.max([len(resp_peaks),len(resp_troughs)])
    },index=[0])
    
    return respfeatures_df,data



#%% HRV from RR quadrant parameters
def compute_quadranthrv(
        data,srate,plot=False
        ):
    ''' 
    Compute Resp features
    
    Parameters
    ----------
    data            :       [ARRAY] ECG data [ntime]
    srate           :       [FLOAT] in Hz    

    Returns
    -------
    quadranthrv_df  : 
    '''   
    
    if abs(np.percentile(data,85)) < abs(np.percentile(data,15)):
        scalingfactor = -1
    else:
        scalingfactor = 1
    
    # Pre process the ECG data
    __ , info1      = nk.ecg_process(data*scalingfactor, sampling_rate = srate)


    #%% Quadrant HRV
    r_peaks = info1["ECG_R_Peaks"]
    ibi     = np.diff(r_peaks)/100
    ibi_x   = ibi[:-2] - ibi[1:-1]  # IBI(i) - IBI(i+1)
    ibi_y   = ibi[1:-1] - ibi[2:]   # IBI(i+1) - IBI(i+2)

    Q1, Q2, Q3,Q4 = [], [], [], []

    for i in range(len(ibi_x)):
        x, y = ibi_x[i], ibi_y[i]
        if x >= 0 and y >= 0:
            Q1.append([x, y])
        elif x < 0 and y > 0:
            Q2.append([x, y])
        elif x <= 0 and y <= 0:
            Q3.append([x, y])
        elif x > 0 and y < 0:
            Q4.append([x, y])
            
    Q1, Q2, Q3, Q4 = map(lambda q: np.array(q) if q else np.array([[0, 0]]), [Q1, Q2, Q3, Q4])
    def mean_vector(quadrant_points):
        return np.mean(quadrant_points, axis=0) if len(quadrant_points) > 0 else np.array([0, 0])
    
    def calculate_var(quadrant_points):
        if quadrant_points.size == 0:
            return 0  # Handle empty quadrants
        
        distances = np.sqrt(quadrant_points[:, 0]**2 + quadrant_points[:, 1]**2)
        return 2 * np.std(distances)
    
    def compute_entropy(Q1, Q2, Q3, Q4):
        E0 = 0
        Q = np.array([len(Q1), len(Q2), len(Q3), len(Q4)])
        proportions = Q / np.sum(Q)
        
        # Compute entropy
        for p in proportions:
            if p > 0:
                E0 -= p * np.log2(p)
        
        return E0
        
    P_Q1, P_Q2, P_Q3, P_Q4          = map(mean_vector, [Q1, Q2, Q3, Q4])
    
    Q1_var, Q2_var, Q3_var, Q4_var  = map(calculate_var, [Q1, Q2, Q3, Q4])
    
    Arr = Q2_var-Q4_var                     # arrhythmic behaviour
    Acc = Q3_var-Q1_var                     # acceleration–deceleration balance
    SPE = compute_entropy(Q1, Q2, Q3, Q4)   # second-order Poincaré plot entropy
    
    
    quadranthrv_df  = pd.DataFrame({
        'P_Q1': P_Q1,
        'P_Q2': P_Q2,
        'P_Q3': P_Q3,
        'P_Q4': P_Q4,
        'Q1_var': Q1_var,
        'Q2_var': Q2_var,
        'Q3_var': Q3_var,
        'Q4_var': Q4_var,
        'Arr': Arr,
        'Acc': Acc,
        'SPE': SPE
        },index=[0])
    
    return quadranthrv_df

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,8))
        plt.axhline(0, color='black', linestyle='-', linewidth=1)
        plt.axvline(0, color='black', linestyle='-', linewidth=1)
    
        plt.scatter(ibi_x, ibi_y, color='gray', alpha=0.7, label="IBI Points")
    
        arrow_colors    = ["purple", "blue", "pink", "orange"]
        arrows          = [P_Q1, P_Q2, P_Q3, P_Q4]
        labels          = ["P(Q1)", "P(Q2)", "P(Q3)", "P(Q4)"]
    
        for (dx, dy), color, label in zip(arrows, arrow_colors, labels):
            plt.quiver(0, 0, dx, dy, angles='xy', scale_units='xy', scale=1, color=color, label=label)
            
        plt.text(0.5, 0.5, "Q1:\nD+D", fontsize=12, ha='center', color='black', alpha=0.7)
        plt.text(-0.5, 0.5, "Q2:\nD+A", fontsize=12, ha='center', color='black', alpha=0.7)
        plt.text(-0.5, -0.5, "Q3:\nA+A", fontsize=12, ha='center', color='black', alpha=0.7)
        plt.text(0.5, -0.5, "Q4:\nA+D", fontsize=12, ha='center', color='black', alpha=0.7)
    
        plt.xlabel("IBI(i+1) - IBI(i) (s)")
        plt.ylabel("IBI(i+2) - IBI(i+1) (s)")
        plt.title("Second-Order Poincaré Plot (ECG)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()





