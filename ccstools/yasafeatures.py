# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 06:05:21 2022

@author: Sruthi & Arun Sasidharan
"""

#%% Import libraries
from numpy import log10,median,logspace,arange,mean,percentile
import yasa
import pandas as pd
from scipy.signal import welch
import numpy as np
from scipy import signal
import logging 
import antropy as ant
from numpy import apply_along_axis as apply
from fooof import FOOOF
from lspopt import spectrogram_lspopt

def compute_psd_features(
        data,sf,winsize=1,psdtype='welch',
        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
               (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
               (30, 40, 'Gamma1')]):
    ''' 
    Compute PSD Features (12 Features: Regular Relative Band Power, Power Ratios)
    
    Parameters
    ----------
    data :          EEG data [nchan x ntime]
    sf :            [FLOAT] in Hz
    winsize :       [FLOAT] in Secs
    psdtype :       'welch' 
                 or 'multitaper' 
                 or 'welch-multitaper'- welch for low freq and multitaper for high freq

    Returns
    -------
    psd_features :   dataframe with PSD features
    '''    
    if psdtype=='welch':

        freqs, psd = welch(data, sf, nperseg=int(winsize*sf), 
                           average='median',scaling='density')
        psd_features = yasa.bandpower_from_psd(psd, freqs, bands=bands,relative=True)
        psd_features["TotalBandPower"] = log10(psd_features["TotalAbsPow"]) # Log transform the total band power
        
    elif psdtype=='multitaper':
        
        freqs,_,psd = spectrogram_lspopt(
            data,sf,c_parameter=int(winsize*4),scaling='density',
            nperseg=int(winsize*sf),noverlap=int(winsize*sf/2))        
        psd = median(psd,axis=-1)
        psd_features = yasa.bandpower_from_psd(psd, freqs, bands=bands,relative=True)
        psd_features["TotalBandPower"] = log10(psd_features["TotalAbsPow"]) # Log transform the total band power
        
    elif psdtype=='welch-multitaper':
        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha')]
        freqs, psd = welch(data, sf, nperseg=int(winsize*sf), 
                           average='median',scaling='density')
        rbp1 = yasa.bandpower_from_psd(psd, freqs, bands=bands,relative=True)
        
        bands=[(8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'),
               (30, 40, 'Gamma1')]
        freqs,_,psd = spectrogram_lspopt(
            data,sf,c_parameter=int(winsize*4),scaling='density',
            nperseg=int(winsize*sf),noverlap=int(winsize*sf/2))        
        psd = median(psd,axis=-1)
        rbp2 = yasa.bandpower_from_psd(psd, freqs, bands=bands,relative=True)

        psd_features = pd.concat([rbp1,rbp2],axis=1)
        psd_features["TotalBandPower"] = log10(psd_features["TotalAbsPow"]) # Log transform the total band power
        
    psd_features = psd_features.drop(columns=['Chan','FreqRes','Relative','TotalAbsPow'])

    pd.eval("ATratio = psd_features.Alpha / psd_features.Theta", target=psd_features,inplace=True)
    pd.eval("DTratio = psd_features.Delta / psd_features.Theta", target=psd_features,inplace=True)
    pd.eval("DAratio = psd_features.Delta / psd_features.Alpha", target=psd_features,inplace=True)
    pd.eval("ABratio = psd_features.Alpha / psd_features.Beta1", target=psd_features,inplace=True)

    return psd_features
    
def compute_irasa_features(
        data,sf,winsize=1,psdtype='welch',
        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
               (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
               (30, 40, 'Gamma1')]):
    ''' 
    Compute IRASA Features (14 Features: Relative Band power in aperiodic, oscillations, Fit the fractal component (1/f))
    
    Parameters
    ----------
    data :          EEG data [nchan x ntime]
    sf :            [FLOAT] in Hz
    winsize :       [FLOAT] in Secs
    psdtype :       'welch' 
                 or 'multitaper' 
                 or 'welch-multitaper'- welch for low freq and multitaper for high freq

    Returns
    -------
    irasa_features :   dataframe with PSD features
    '''
        
    freqs, psd_aperiodic, psd_osc, fit_params = irasa(
        data,sf,win_sec=winsize,band=(bands[0][0], bands[-1][1]),
        kwargs_welch=dict(nfft=int(winsize*sf*1)),psdtype='welch')
    psd_osc[psd_osc<0] = 0 # removing the negative values of psd 
    psd_aperiodic[psd_aperiodic<0] = 0 # removing the negative values of psd 
    
    
    irasa_features_o = yasa.bandpower_from_psd(psd_osc, freqs, bands=bands,relative=True)
    irasa_features_o["TotalBandPower"] = log10(irasa_features_o["TotalAbsPow"]) # Log transform the total band power    
    pd.eval("ATratio = irasa_features_o.Alpha / irasa_features_o.Theta", target=irasa_features_o,inplace=True)
    pd.eval("DTratio = irasa_features_o.Delta / irasa_features_o.Theta", target=irasa_features_o,inplace=True)
    pd.eval("DAratio = irasa_features_o.Delta / irasa_features_o.Alpha", target=irasa_features_o,inplace=True)
    pd.eval("ABratio = irasa_features_o.Alpha / irasa_features_o.Beta1", target=irasa_features_o,inplace=True)           
    irasa_features_o = irasa_features_o.drop(columns=['Chan','FreqRes','Relative','TotalAbsPow'])
    irasa_features_o = irasa_features_o.add_prefix('o_')
    
    irasa_features_a = yasa.bandpower_from_psd(psd_aperiodic, freqs, bands=bands,relative=True)
    irasa_features_a["TotalBandPower"] = log10(irasa_features_a["TotalAbsPow"]) # Log transform the total band power    
    pd.eval("ATratio = irasa_features_a.Alpha / irasa_features_a.Theta", target=irasa_features_a,inplace=True)
    pd.eval("DTratio = irasa_features_a.Delta / irasa_features_a.Theta", target=irasa_features_a,inplace=True)
    pd.eval("DAratio = irasa_features_a.Delta / irasa_features_a.Alpha", target=irasa_features_a,inplace=True)
    pd.eval("ABratio = irasa_features_a.Alpha / irasa_features_a.Beta1", target=irasa_features_a,inplace=True)           
    irasa_features_a = irasa_features_a.drop(columns=['Chan','FreqRes','Relative','TotalAbsPow'])
    irasa_features_a = irasa_features_a.add_prefix('a_')
    
    irasa_features = pd.concat([irasa_features_o,irasa_features_a,fit_params],axis=1)
    irasa_features = irasa_features.drop(columns=['Chan'])
    irasa_features['Slope'] = abs(irasa_features['Slope'])    
    irasa_features = irasa_features.add_prefix('i_')
    
    return irasa_features 

def compute_fooof_features(
        data,sf,winsize=1,psdtype='welch',
        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'),
               (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'),
               (30, 40, 'Gamma1')]):
    '''
    Parameters
    ----------
    data :          EEG data [nchan x ntime]
    sf :            [FLOAT] in Hz
    winsize :       [FLOAT] in Secs
    psdtype :       'welch' 
                 or 'multitaper' 

    Returns
    -------
    fooof_features : dataframe with foof_params, osc_psd
    '''    
    
    if psdtype=='welch':
        freqs, psd = welch(data, sf, nperseg=int(winsize*sf), 
                           average='median',scaling='density')
    elif psdtype=='multitaper':
        freqs,_,psd = spectrogram_lspopt(
            data,sf,c_parameter=int(winsize*4),scaling='density',
            nperseg=int(winsize*sf),noverlap=int(winsize*sf/2))        
        psd = median(psd,axis=-1)

    # Define frequency range across which to model the spectrum
    freq_range = [bands[0][0], bands[-1][-2]]
    
    for i in range(len(data)):
        # Model the power spectrum with FOOOF    
        fm = FOOOF(verbose=False) # Initialize FOOOF object   
        fm.fit(freqs, psd[i],freq_range)
        
        psd_osc = 10**(fm.fooofed_spectrum_)
        psd_osc[psd_osc<0] = 0 # removing the negative values of psd 
        psd_all = 10**(fm.power_spectrum)
        psd_all[psd_all<0] = 0 # removing the negative values of psd 
        psd_aperiodic = psd_all - psd_osc
        psd_aperiodic[psd_aperiodic<0] = 0 # removing the negative values of psd
        selfreqs   = fm.freqs
        
        fooof_temp_o = yasa.bandpower_from_psd(psd_osc, selfreqs, bands=bands,relative=True)
        fooof_temp_o["TotalBandPower"] = log10(fooof_temp_o["TotalAbsPow"]) # Log transform the total band power    
        pd.eval("ATratio = fooof_temp_o.Alpha / fooof_temp_o.Theta", target=fooof_temp_o,inplace=True)
        pd.eval("DTratio = fooof_temp_o.Delta / fooof_temp_o.Theta", target=fooof_temp_o,inplace=True)
        pd.eval("DAratio = fooof_temp_o.Delta / fooof_temp_o.Alpha", target=fooof_temp_o,inplace=True)
        pd.eval("ABratio = fooof_temp_o.Alpha / fooof_temp_o.Beta1", target=fooof_temp_o,inplace=True)           
        fooof_temp_o = fooof_temp_o.drop(columns=['Chan','FreqRes','Relative','TotalAbsPow'])                       
        fooof_temp_o = fooof_temp_o.add_prefix('o_')
        
        fooof_temp_a = yasa.bandpower_from_psd(psd_aperiodic, selfreqs, bands=bands,relative=True)
        fooof_temp_a["TotalBandPower"] = log10(fooof_temp_a["TotalAbsPow"]) # Log transform the total band power    
        pd.eval("ATratio = fooof_temp_a.Alpha / fooof_temp_a.Theta", target=fooof_temp_a,inplace=True)
        pd.eval("DTratio = fooof_temp_a.Delta / fooof_temp_a.Theta", target=fooof_temp_a,inplace=True)
        pd.eval("DAratio = fooof_temp_a.Delta / fooof_temp_a.Alpha", target=fooof_temp_a,inplace=True)
        pd.eval("ABratio = fooof_temp_a.Alpha / fooof_temp_a.Beta1", target=fooof_temp_a,inplace=True)           
        fooof_temp_a = fooof_temp_a.drop(columns=['Chan','FreqRes','Relative','TotalAbsPow'])              
        fooof_temp_a = fooof_temp_a.add_prefix('a_')
        
        fooof_temp = pd.concat([fooof_temp_o,fooof_temp_a],axis=1)
        fooof_temp["Intercept"] = fm.aperiodic_params_[0]
        fooof_temp["Slope"] = fm.aperiodic_params_[1]
        fooof_temp["R^2"] = fm.get_params('r_squared') 
        
        if i == 0:
            fooof_features = fooof_temp
        else:
            fooof_features = pd.concat([fooof_features,fooof_temp],axis=0)
    
    fooof_features = fooof_features.add_prefix('f_')
    fooof_features = fooof_features.reset_index(drop=True) # because of loop indices are zero
    
    return fooof_features 

def compute_nonlinear_features(data): 
    '''
    Parameters
    ----------
    data :          EEG data [nchan x ntime]

    Returns
    -------
    nonlinear_features : dataframe with Entropy & Fractal dimension measures
    '''

    df_nonlinear = {
        
        # Entropy
        'perm_entropy': apply(ant.perm_entropy, 1, data, normalize=True),
        'svd_entropy': apply(ant.svd_entropy, 1, data, normalize=True),
        'sample_entropy': apply(ant.sample_entropy, 1, data),
        
        # Fractal dimension
        'dfa': apply(ant.detrended_fluctuation, 1, data),
        'petrosian': apply(ant.petrosian_fd, 1, data),
        'katz': apply(ant.katz_fd, 1, data),
        'higuchi': apply(ant.higuchi_fd, 1, data),
        'lziv' : apply(ant.lziv_complexity, 1, data > data.mean(), normalize=True) # Binarize the EEG signal before calculating Lempel-Ziv complexity
    }
    nonlinear_features = pd.DataFrame(df_nonlinear)
    return nonlinear_features


def compute_pac_features(
        data,sf,
        freq_pha = arange(2,8,0.5),
        freq_amp = logspace(log10(12),log10(30),10),
        idpac=(5, 3, 1),
        dcomplex='hilbert'): 
    '''
    Phase Amplitude Coupling
    Parameters
    ----------
    data :          EEG data [nchan x ntime]
    sf :            [FLOAT] in Hz

    Returns
    -------
    pac_features : dataframe with Phase Amplitude Coupling measures
    '''
    from tensorpac import Pac
    
    # Define a PAC object
    p = Pac(idpac=idpac, 
            f_pha=freq_pha, 
            f_amp=freq_amp,
            dcomplex=dcomplex)
    
    # Filter the data and extract PAC
    xpac = p.filterfit(sf, data,mcp='maxstat',n_jobs=1,p=0.05)
    
    # Get prominent PAC value
    pac_features = pd.DataFrame({
        'mean_pac': [mean(percentile(xpac,range(95,99)))]
        })
    return pac_features
    

#%%############################################################################

def generate_single_epoch_features(
        data,sf,winsize,psdtype='welch',
        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
               (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
               (30, 40, 'Gamma1')]):
    psd_features        = compute_psd_features(data,sf,winsize=winsize,psdtype=psdtype,bands=bands)
    try:
        irasa_features      = compute_irasa_features(data,sf,winsize=int(winsize*0.5),bands=bands) # lower freq resolution removes warnings with minimal effect on results
    except:
        irasa_features      = compute_irasa_features(data,sf,winsize=int(winsize*1),bands=bands)
    fooof_features      = compute_fooof_features(data,sf,winsize=winsize,psdtype=psdtype,bands=bands)
    nonlinear_features  = compute_nonlinear_features(data)
    
    all_eegfeatures = pd.concat([psd_features,irasa_features,fooof_features,nonlinear_features],axis=1)
    return all_eegfeatures

#%%############################################################################

def generate_single_epoch_features_mini(
        data,sf,winsize,psdtype='welch',
        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
               (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
               (30, 40, 'Gamma1')]):
    try:
        irasa_features      = compute_irasa_features(data,sf,winsize=int(winsize*0.5),bands=bands,psdtype=psdtype) # lower freq resolution removes warnings with minimal effect on results
    except:
        irasa_features      = compute_irasa_features(data,sf,winsize=int(winsize*1),bands=bands,psdtype=psdtype)
    # fooof_features      = compute_fooof_features(data,sf,winsize=winsize,psdtype='welch',bands=bands)
    nonlinear_features  = compute_nonlinear_features(data)
    pac_features        = compute_pac_features(data,sf)
    
    all_eegfeatures = pd.concat([irasa_features,nonlinear_features,pac_features],axis=1)
    # all_eegfeatures["f_Intercept"]  = fooof_features["f_Intercept"]
    # all_eegfeatures["f_Slope"]      = fooof_features["f_Slope"]
    # all_eegfeatures["f_R^2"]        = fooof_features["f_R^2"]
    return all_eegfeatures






def irasa(
    data,
    sf=None,
    ch_names=None,
    band=(1, 30),
    psdtype='welch',
    hset=[
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.4,
        1.45,
        1.5,
        1.55,
        1.6,
        1.65,
        1.7,
        1.75,
        1.8,
        1.85,
        1.9,
    ],
    return_fit=True,
    win_sec=4,
    kwargs_welch=dict(average="median", window="hamming"),
    verbose=True,
):
    r"""
    Separate the aperiodic (= fractal, or 1/f) and oscillatory component
    of the power spectra of EEG data using the IRASA method.

    .. versionadded:: 0.1.7

    Parameters
    ----------
    data : :py:class:`numpy.ndarray` or :py:class:`mne.io.BaseRaw`
        1D or 2D EEG data. Can also be a :py:class:`mne.io.BaseRaw`, in which
        case ``data``, ``sf``, and ``ch_names`` will be automatically
        extracted, and ``data`` will also be converted from Volts (MNE default)
        to micro-Volts (YASA).
    sf : float
        The sampling frequency of data AND the hypnogram.
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    ch_names : list
        List of channel names, e.g. ['Cz', 'F3', 'F4', ...]. If None,
        channels will be labelled ['CHAN000', 'CHAN001', ...].
        Can be omitted if ``data`` is a :py:class:`mne.io.BaseRaw`.
    band : tuple or None
        Broad band frequency range.
        Default is 1 to 30 Hz.
    hset : list or :py:class:`numpy.ndarray`
        Resampling factors used in IRASA calculation. Default is to use a range
        of values from 1.1 to 1.9 with an increment of 0.05.
    return_fit : boolean
        If True (default), fit an exponential function to the aperiodic PSD
        and return the fit parameters (intercept, slope) and :math:`R^2` of
        the fit.

        The aperiodic signal, :math:`L`, is modeled using an exponential
        function in semilog-power space (linear frequencies and log PSD) as:

        .. math:: L = a + \text{log}(F^b)

        where :math:`a` is the intercept, :math:`b` is the slope, and
        :math:`F` the vector of input frequencies.
    win_sec : int or float
        The length of the sliding window, in seconds, used for the Welch PSD
        calculation. Ideally, this should be at least two times the inverse of
        the lower frequency of interest (e.g. for a lower frequency of interest
        of 0.5 Hz, the window length should be at least 2 * 1 / 0.5 =
        4 seconds).
    kwargs_welch : dict
        Optional keywords arguments that are passed to the
        :py:func:`scipy.signal.welch` function.
    verbose : bool or str
        Verbose level. Default (False) will only print warning and error
        messages. The logging levels are 'debug', 'info', 'warning', 'error',
        and 'critical'. For most users the choice is between 'info'
        (or ``verbose=True``) and warning (``verbose=False``).

    Returns
    -------
    freqs : :py:class:`numpy.ndarray`
        Frequency vector.
    psd_aperiodic : :py:class:`numpy.ndarray`
        The fractal (= aperiodic) component of the PSD.
    psd_oscillatory : :py:class:`numpy.ndarray`
        The oscillatory (= periodic) component of the PSD.
    fit_params : :py:class:`pandas.DataFrame` (optional)
        Dataframe of fit parameters. Only if ``return_fit=True``.

    Notes
    -----
    The Irregular-Resampling Auto-Spectral Analysis (IRASA) method is
    described in Wen & Liu (2016). In a nutshell, the goal is to separate the
    fractal and oscillatory components in the power spectrum of EEG signals.

    The steps are:

    1. Compute the original power spectral density (PSD) using Welch's method.
    2. Resample the EEG data by multiple non-integer factors and their
       reciprocals (:math:`h` and :math:`1/h`).
    3. For every pair of resampled signals, calculate the PSD and take the
       geometric mean of both. In the resulting PSD, the power associated with
       the oscillatory component is redistributed away from its original
       (fundamental and harmonic) frequencies by a frequency offset that varies
       with the resampling factor, whereas the power solely attributed to the
       fractal component remains the same power-law statistical distribution
       independent of the resampling factor.
    4. It follows that taking the median of the PSD of the variously
       resampled signals can extract the power spectrum of the fractal
       component, and the difference between the original power spectrum and
       the extracted fractal spectrum offers an approximate estimate of the
       power spectrum of the oscillatory component.

    Note that an estimate of the original PSD can be calculated by simply
    adding ``psd = psd_aperiodic + psd_oscillatory``.

    For an example of how to use this function, please refer to
    https://github.com/raphaelvallat/yasa/blob/master/notebooks/09_IRASA.ipynb

    For an article discussing the challenges of using IRASA (or fooof) see [5].

    References
    ----------
    [1] Wen, H., & Liu, Z. (2016). Separating Fractal and Oscillatory
        Components in the Power Spectrum of Neurophysiological Signal.
        Brain Topography, 29(1), 13–26.
        https://doi.org/10.1007/s10548-015-0448-0

    [2] https://github.com/fieldtrip/fieldtrip/blob/master/specest/

    [3] https://github.com/fooof-tools/fooof

    [4] https://www.biorxiv.org/content/10.1101/299859v1

    [5] https://doi.org/10.1101/2021.10.15.464483
    """
    import fractions

    # Safety checks
    assert isinstance(data, np.ndarray), "Data must be a numpy array."
    data = np.atleast_2d(data)
    assert data.ndim == 2, "Data must be of shape (nchan, n_samples)."
    nchan, npts = data.shape
    assert nchan < npts, "Data must be of shape (nchan, n_samples)."
    assert sf is not None, "sf must be specified if passing a numpy array."
    assert isinstance(sf, (int, float))
    if ch_names is None:
        ch_names = ["CHAN" + str(i).zfill(3) for i in range(nchan)]
    else:
        ch_names = np.atleast_1d(np.asarray(ch_names, dtype=str))
        assert ch_names.ndim == 1, "ch_names must be 1D."
        assert len(ch_names) == nchan, "ch_names must match data.shape[0]."
    hp = 0  # Highpass filter unknown -> set to 0 Hz
    lp = sf / 2  # Lowpass filter unknown -> set to Nyquist

    # Check the other arguments
    hset = np.asarray(hset)
    assert hset.ndim == 1, "hset must be 1D."
    assert hset.size > 1, "2 or more resampling fators are required."
    hset = np.round(hset, 4)  # avoid float precision error with np.arange.
    band = sorted(band)
    assert band[0] > 0, "first element of band must be > 0."
    assert band[1] < (sf / 2), "second element of band must be < (sf / 2)."
    win = int(win_sec * sf)  # nperseg

    # Inform about maximum resampled fitting range
    h_max = np.max(hset)
    band_evaluated = (band[0] / h_max, band[1] * h_max)
    freq_Nyq = sf / 2  # Nyquist frequency
    freq_Nyq_res = freq_Nyq / h_max  # minimum resampled Nyquist frequency
    logging.info(f"Fitting range: {band[0]:.2f}Hz-{band[1]:.2f}Hz")
    logging.info(f"Evaluated frequency range: {band_evaluated[0]:.2f}Hz-{band_evaluated[1]:.2f}Hz")
    if band_evaluated[0] < hp:
        logging.warning(
            "The evaluated frequency range starts below the "
            f"highpass filter ({hp:.2f}Hz). Increase the lower band"
            f" ({band[0]:.2f}Hz) or decrease the maximum value of "
            f"the hset ({h_max:.2f})."
        )
    if band_evaluated[1] > lp and lp < freq_Nyq_res:
        logging.warning(
            "The evaluated frequency range ends after the "
            f"lowpass filter ({lp:.2f}Hz). Decrease the upper band"
            f" ({band[1]:.2f}Hz) or decrease the maximum value of "
            f"the hset ({h_max:.2f})."
        )
    if band_evaluated[1] > freq_Nyq_res:
        logging.warning(
            "The evaluated frequency range ends after the "
            "resampled Nyquist frequency "
            f"({freq_Nyq_res:.2f}Hz). Decrease the upper band "
            f"({band[1]:.2f}Hz) or decrease the maximum value "
            f"of the hset ({h_max:.2f})."
        )
       
    # Calculate the original PSD over the whole data   
    if psdtype=='welch':
        freqs, psd = welch(data, sf, nperseg=int(win*sf), **kwargs_welch)
    elif psdtype=='multitaper':
        freqs,_,psd = spectrogram_lspopt(
            data,sf,c_parameter=win,scaling='density',
            nperseg=int(win*sf),noverlap=int(win/2))        
        psd = median(psd,axis=-1)

    # Start the IRASA procedure
    psds = np.zeros((len(hset), *psd.shape))

    for i, h in enumerate(hset):
        # Get the upsampling/downsampling (h, 1/h) factors as integer
        rat = fractions.Fraction(str(h))
        up, down = rat.numerator, rat.denominator
        # Much faster than FFT-based resampling
        data_up = signal.resample_poly(data, up, down, axis=-1)
        data_down = signal.resample_poly(data, down, up, axis=-1)
        # Calculate the PSD using same params as original
        freqs_up, psd_up = signal.welch(data_up, h * sf, nperseg=int(win*sf), **kwargs_welch)
        freqs_dw, psd_dw = signal.welch(data_down, sf / h, nperseg=int(win*sf), **kwargs_welch)
        # Geometric mean of h and 1/h
        psds[i, :] = np.sqrt(psd_up * psd_dw)

    # Now we take the median PSD of all the resampling factors, which gives
    # a good estimate of the aperiodic component of the PSD.
    psd_aperiodic = np.median(psds, axis=0)

    # We can now calculate the oscillations (= periodic) component.
    psd_osc = psd - psd_aperiodic

    # Let's crop to the frequencies defined in band
    mask_freqs = np.ma.masked_outside(freqs, *band).mask
    freqs = freqs[~mask_freqs]
    psd_aperiodic = np.compress(~mask_freqs, psd_aperiodic, axis=-1)
    psd_osc = np.compress(~mask_freqs, psd_osc, axis=-1)

    if return_fit:
        # Aperiodic fit in semilog space for each channel
        from scipy.optimize import curve_fit

        intercepts, slopes, r_squared = [], [], []

        def func(t, a, b):
            # See https://github.com/fooof-tools/fooof
            return a + np.log(t**b)

        for y in np.atleast_2d(psd_aperiodic):
            y_log = np.log(y)
            # Note that here we define bounds for the slope but not for the
            # intercept.
            popt, pcov = curve_fit(
                func, freqs, y_log, p0=(2, -1), bounds=((-np.inf, -10), (np.inf, 2))
            )
            intercepts.append(popt[0])
            slopes.append(popt[1])
            # Calculate R^2: https://stackoverflow.com/q/19189362/10581531
            residuals = y_log - func(freqs, *popt)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
            r_squared.append(1 - (ss_res / ss_tot))

        # Create fit parameters dataframe
        fit_params = {
            "Chan": ch_names,
            "Intercept": intercepts,
            "Slope": slopes,
            "R^2": r_squared,
            "std(osc)": np.std(psd_osc, axis=-1, ddof=1),
        }
        return freqs, psd_aperiodic, psd_osc, pd.DataFrame(fit_params)
    else:
        return freqs, psd_aperiodic, psd_osc