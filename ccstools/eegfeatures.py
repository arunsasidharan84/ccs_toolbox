# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:54:12 2024

@author: Arun Sasidharan
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import fractions
import networkx as nx

# Scientific computing
from scipy.signal import welch, resample_poly
from scipy.optimize import curve_fit
from scipy import integrate

# EEG-specific libraries
from lspopt import spectrogram_lspopt
from fooof import FOOOFGroup
from fooof.analysis.error import compute_pointwise_error_fm
import antropy as ant
from statsmodels.tsa.stattools import acf
from pycatch22 import catch22_all
from yasa import bandpower_from_psd_ndarray

# Note: bct and other specialized libraries are imported within functions 
# if they are optional or used only in specific contexts.


# %% PSD
def compute_psd(
        data, srate, psdtype='welch',
        kwargs_psd=dict(scaling='density', average='median', window="hamming",
                        nperseg=None)):
    ''' 
    Compute Power Spectral Density (PSD)

    Parameters
    ----------
    data    :       EEG data [nchan x ntime] or [n_epoch x nchan x ntime]
    srate   :       [FLOAT] in Hz    
    psdtype :       'welch' or 'multitaper'
    kwargs_psd : dict
        Optional keywords arguments that are passed to PSD function
        scaling :   'density' or 'spectrum'
        nperseg :   [INT] in Samples. Ideally, this should correspond to 
                    at least two times the inverse of the lower frequency of interest

    Returns
    -------
    psdvals  :      PSD values
    psdfreqs :      PSD frequencies
    '''

    if kwargs_psd['nperseg'] is None:
        kwargs_psd['nperseg'] = srate
    kwargs_psd['nperseg'] = int(kwargs_psd['nperseg'])

    if psdtype == 'welch':
        psdfreqs, psdvals = welch(data, srate, axis=-1, **kwargs_psd)
    elif psdtype == 'multitaper':
        if 'window' in kwargs_psd.keys():
            del kwargs_psd['window']
        if 'average' in kwargs_psd.keys():
            del kwargs_psd['average']
        psdfreqs, _, psdvals = spectrogram_lspopt(
            data, srate, **kwargs_psd)
        psdvals = np.mean(psdvals, axis=-1)

    return psdvals, psdfreqs


# %% FOOOF
def compute_fooof(psdvals, psdfreqs, freq_range=[1, 30], npeaks=2, psdout=False):
    '''
    Separate the aperiodic and oscillatory component
    of the power spectra of EEG data using the FOOOF or SpecParam method.
    
    Parameters
    ----------
    psdvals     : PSD values   [nchan x nfreq] or [n_epoch x nchan x nfreq]
    psdfreqs    : PSD frequencies
    freq_range  : Frequency Range within which FOOOF is estimated
    npeaks      : number of oscillatory peaks to extract
    psdout      : whether to extract psd values from FOOOF     

    Returns
    -------
    fooofpsd            : psd values of FOOOF ('aperiodic','oscillatory','errorfreq')
    fooofparamvals      : values of FOOOF parameters
    foooffreqs          : Freqencies for which FOOOF is computed
    fooofparamlabels    : names of FOOOF parameters
    fooofpsdlabels      : names of FOOOF psd values ('aperiodic','oscillatory','errorfreq')
    '''              
    
    
    if len(psdvals.shape) == 1:
        psdvals = np.expand_dims(np.expand_dims(psdvals, axis=0), axis=0)
    elif len(psdvals.shape) == 2:
        psdvals = np.expand_dims(psdvals, axis=0)
    n_epoch, n_chan, _ = psdvals.shape

    fooofpsd = []
    fooofparamvals = []
    fm = FOOOFGroup(verbose=False)  # Initialize FOOOF object   
    for i in tqdm(range(n_epoch), desc="Computing FOOOF"):
        fm.fit(psdfreqs, psdvals[i], freq_range)
        fooofparamvals.append(fm.to_df(npeaks).to_numpy())
        if psdout:
            for j in range(n_chan):
                full = fm.get_fooof(j).get_data(component='full', space='linear')
                aperiodic = fm.get_fooof(j).get_data(component='aperiodic', space='linear')
                peaks = fm.get_fooof(j).get_data(component='peak', space='linear')
                errors = abs(aperiodic-(full-peaks))

                fooofpsd.append(aperiodic)
                fooofpsd.append(peaks)
                fooofpsd.append(errors)

    fooofparamvals = np.array(fooofparamvals)
    foooffreqs = fm.freqs
    fooofparamlabels = list(fm.to_df(npeaks).columns)
    if psdout:
        fooofpsd = np.array(fooofpsd).reshape([n_epoch, n_chan, 3, len(foooffreqs)])
        fooofpsdlabels = ['aperiodic', 'oscillatory', 'errorfreq']

        auc_oscaperdiff = np.expand_dims(integrate.simpson(fooofpsd[:, :, 1]-fooofpsd[:, :, 0], dx=1, axis=-1), axis=-1)

        foofoscvals_temp = fooofpsd[:, :, 1]
        foofoscvals_temp[foofoscvals_temp < 0] = 0
        oscspectraledge = np.expand_dims(fractional_latency(foofoscvals_temp, axis=-1), axis=-1)

        fooofparamvals = np.concatenate([fooofparamvals, auc_oscaperdiff, oscspectraledge], axis=-1)
        fooofparamlabels = fooofparamlabels + ['auc', 'oscspectraledge']

    else:
        fooofpsd = np.array(fooofpsd)
        fooofpsdlabels = []

    return fooofpsd, fooofparamvals, foooffreqs, fooofparamlabels, fooofpsdlabels

#%% Entropy & Fractal dimension
def compute_nonlinear(data): 
    '''
    Parameters
    ----------
    data :          EEG data  [nchan x ntime] or [n_epoch x nchan x ntime]

    Returns
    -------
    nonlinearvals : Entropy & Fractal dimension measures
    '''
    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data, axis=0), axis=0)
    elif len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    n_epoch, n_chan, _ = data.shape

    perm_entropy = np.zeros([n_epoch, n_chan])
    svd_entropy = np.zeros([n_epoch, n_chan])
    sample_entropy = np.zeros([n_epoch, n_chan])
    dfa = np.zeros([n_epoch, n_chan])
    petrosian = np.zeros([n_epoch, n_chan])
    katz = np.zeros([n_epoch, n_chan])
    higuchi = np.zeros([n_epoch, n_chan])
    lziv = np.zeros([n_epoch, n_chan])
    for i in tqdm(range(n_epoch), desc="Computing non-linear measures"):
        for chan_no in range(n_chan):
            perm_entropy[i][chan_no] = ant.perm_entropy(data[i][chan_no], normalize=True)
            svd_entropy[i][chan_no] = ant.svd_entropy(data[i][chan_no], normalize=True)
            sample_entropy[i][chan_no] = ant.sample_entropy(data[i][chan_no])
            dfa[i][chan_no] = ant.detrended_fluctuation(data[i][chan_no])
            petrosian[i][chan_no] = ant.petrosian_fd(data[i][chan_no])
            katz[i][chan_no] = ant.katz_fd(data[i][chan_no])
            higuchi[i][chan_no] = ant.higuchi_fd(data[i][chan_no])
            lziv[i][chan_no] = ant.lziv_complexity(data[i][chan_no] > data[i][chan_no].mean(), normalize=True)

    nonlinearvals = np.concatenate([
        perm_entropy, svd_entropy, sample_entropy,
        dfa, petrosian, katz, higuchi, lziv
    ], axis=-1).reshape([n_epoch, 8, n_chan]).transpose([0, 2, 1])
    nonlinearlabels = [
        'perm_entropy', 'svd_entropy', 'sample_entropy',
        'dfa', 'petrosian', 'katz', 'higuchi', 'lziv']

    return nonlinearvals, nonlinearlabels


#%% IRASA
def compute_irasa(
        data,srate,freq_range=[1,30],
        psdtype='welch',
        kwargs_psd=dict(scaling='density',average='median',window="hamming",
                        nperseg = None),
        hset=[1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9],
        return_fit=True):
    '''
    Separate the aperiodic and oscillatory component
    of the power spectra of EEG data using the IRASA method.

    Parameters
    ----------
    data        :   EEG data  [nchan x ntime] or [n_epoch x nchan x ntime]
    srate       :   [FLOAT] in Hz
    psdtype     :   'welch' 
                    or 'multitaper'
    kwargs_psd : dict
        Optional keywords arguments that are passed to PSD function
        scaling :   'density' or 'spectrum'
        nperseg :   [INT] in Samples. Ideally, this should correspond to 
                    at least two times the inverse of the lower frequency of interest
    freq_range  :   Frequency Range within which IRASA is estimated
    hset        :   Resampling factors used in IRASA calculation. Default is to use a range
                    of values from 1.1 to 1.9 with an increment of 0.05.
    return_fit  :   [boolean]. If True (default), fit an exponential function to the aperiodic PSD
                    and return the fit parameters (intercept, slope) and :math:`R^2` of
                    the fit.

                    The aperiodic signal, :math:`L`, is modeled using an exponential
                    function in semilog-power space (linear frequencies and log PSD) as:
            
                    .. math:: L = a + \text{log}(F^b)
            
                    where :math:`a` is the intercept, :math:`b` is the slope, and
                    :math:`F` the vector of input frequencies.

    Returns
    -------
    irasaoscvals    : oscillatory psd values using IRASA
    irasaapervals   : aperiodic psd values using IRASA
    irasafreqs      : Freqencies for which IRASA is computed
    irasaaperlabels : names of aperiodic values computed using IRASA

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
    '''
    import fractions
    import logging
    from scipy.signal import resample_poly    
    
    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data,axis=0),axis=0)        
    elif len(data.shape) == 2:
        data = np.expand_dims(data,axis=0)
    n_epoch,n_chan,_ = data.shape
    
    if kwargs_psd['nperseg'] is None:
        kwargs_psd['nperseg'] = srate

    hp = 0          # Highpass filter unknown -> set to 0 Hz
    lp = srate / 2  # Lowpass filter unknown -> set to Nyquist

    # Check arguments
    hset = np.asarray(hset)
    assert hset.ndim == 1, "hset must be 1D."
    assert hset.size > 1, "2 or more resampling fators are required."
    hset = np.round(hset, 4)  # avoid float precision error with np.arange.
    freq_range = sorted(freq_range)
    assert freq_range[0] > 0, "first element of band must be > 0."
    assert freq_range[1] < (srate / 2), "second element of band must be < (srate / 2)."


    # Inform about maximum resampled fitting range
    h_max           = np.max(hset)
    band_evaluated  = (freq_range[0] / h_max, freq_range[1] * h_max)
    freq_Nyq        = srate / 2  # Nyquist frequency
    freq_Nyq_res    = freq_Nyq / h_max  # minimum resampled Nyquist frequency
    logging.info(f"Fitting range: {freq_range[0]:.2f}Hz-{freq_range[1]:.2f}Hz")
    logging.info(f"Evaluated frequency range: {band_evaluated[0]:.2f}Hz-{band_evaluated[1]:.2f}Hz")
    if band_evaluated[0] < hp:
        logging.warning(
            "The evaluated frequency range starts below the "
            f"highpass filter ({hp:.2f}Hz). Increase the lower band"
            f" ({freq_range[0]:.2f}Hz) or decrease the maximum value of "
            f"the hset ({h_max:.2f})."
        )
    if band_evaluated[1] > lp and lp < freq_Nyq_res:
        logging.warning(
            "The evaluated frequency range ends after the "
            f"lowpass filter ({lp:.2f}Hz). Decrease the upper band"
            f" ({freq_range[1]:.2f}Hz) or decrease the maximum value of "
            f"the hset ({h_max:.2f})."
        )
    if band_evaluated[1] > freq_Nyq_res:
        logging.warning(
            "The evaluated frequency range ends after the "
            "resampled Nyquist frequency "
            f"({freq_Nyq_res:.2f}Hz). Decrease the upper band "
            f"({freq_range[1]:.2f}Hz) or decrease the maximum value "
            f"of the hset ({h_max:.2f})."
        )
       
    # Calculate the original PSD over the whole data   
    psd,freqs = compute_psd(data,srate,psdtype=psdtype,kwargs_psd=kwargs_psd)

    # Start the IRASA procedure
    psds = np.zeros((len(hset), *psd.shape))

    for i in tqdm(range(len(hset)),desc="Computing IRASA | Resampled PSDs"):
        h = hset[i]
        # Get the upsampling/downsampling (h, 1/h) factors as integer
        rat         = fractions.Fraction(str(h))
        up, down    = rat.numerator, rat.denominator
        # Much faster than FFT-based resampling
        data_up     = resample_poly(data, up, down, axis=-1)
        data_down   = resample_poly(data, down, up, axis=-1)
        # Calculate the PSD using same params as original
        psd_up,freqs_up = compute_psd(data_up,srate*h,psdtype=psdtype,kwargs_psd=kwargs_psd)
        psd_dw,freqs_dw = compute_psd(data_down,srate/h,psdtype=psdtype,kwargs_psd=kwargs_psd)
        # Geometric mean of h and 1/h
        psds[i, :] = np.sqrt(psd_up * psd_dw)    

    # Now we take the median PSD of all the resampling factors, which gives
    # a good estimate of the aperiodic component of the PSD.
    irasaaperiodic = np.median(psds, axis=0)

    # We can now calculate the oscillations (= periodic) component.
    irasaoscvals = psd - irasaaperiodic

    # Let's crop to the frequencies defined in band
    mask_freqs      = np.ma.masked_outside(freqs, *freq_range).mask
    irasafreqs      = freqs[~mask_freqs]
    irasaaperiodic  = np.compress(~mask_freqs, irasaaperiodic, axis=-1)
    irasaoscvals    = np.compress(~mask_freqs, irasaoscvals, axis=-1)

    if return_fit:
        # Aperiodic fit in semilog space for each channel
        from scipy.optimize import curve_fit
        from scipy import integrate
        
        def func(t, a, b):
            # See https://github.com/fooof-tools/fooof
            return a + np.log(t**b)               

        intercepts, slopes, r_squared = [], [], []
        for i in tqdm(range(n_epoch),desc="Computing IRASA | Aperiodic Fitting"):
            for y in np.atleast_2d(irasaaperiodic[i]):
                y_log = np.log(y)
                # Note that here we define bounds for the slope but not for the
                # intercept.
                popt, pcov = curve_fit(
                    func, irasafreqs, y_log, p0=(2, -1), bounds=((-np.inf, -10), (np.inf, 2))
                )
                intercepts.append(popt[0])
                slopes.append(popt[1])
                # Calculate R^2: https://stackoverflow.com/q/19189362/10581531
                residuals = y_log - func(irasafreqs, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
                r_squared.append(1 - (ss_res / ss_tot))
        intercepts  = np.array(intercepts).reshape([n_epoch,n_chan,1])
        slopes      = np.array(slopes).reshape([n_epoch,n_chan,1])
        r_squared   = np.array(r_squared).reshape([n_epoch,n_chan,1])
        auc_oscaperdiff   = np.expand_dims(integrate.simpson(irasaoscvals-irasaaperiodic, dx=1, axis=-1),axis=-1)
        
        irasaoscvals_temp = irasaoscvals
        irasaoscvals_temp[irasaoscvals_temp < 0] = 0  # removing the negative values of psd
        oscspectraledge = np.expand_dims(fractional_latency(irasaoscvals_temp, axis=-1), axis=-1)

        irasaapervals = np.concatenate([intercepts, slopes, r_squared, auc_oscaperdiff, oscspectraledge], axis=-1)
        irasaaperlabels = ['intercept', 'slope', 'rsquared', 'auc', 'oscspectraledge']

        return irasaoscvals, irasaapervals, irasafreqs, irasaaperlabels
    else:
        return irasaoscvals, irasaaperiodic, irasafreqs


# %% ACW (Autocorrelation Window)
def compute_acw(data, srate, drop=50):
    '''
    Compute Autocorrelation Window (ACW)

    Parameters
    ----------
    data        :   EEG data  [nchan x ntime] or [n_epoch x nchan x ntime]
    srate       :   [FLOAT] in Hz
    drop        :   Percentage drop in autocorrelation to find window (default 50)

    Returns
    -------
    acwvals     :   ACW measures
    '''

    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data, axis=0), axis=0)
    elif len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    n_epoch, n_chan, n_pnts = data.shape

    acwvals = []
    for i in tqdm(range(n_epoch), desc="Computing ACW"):
        for chan_no in range(n_chan):
            tempvals = acf(data[i][chan_no], nlags=n_pnts)
            acwvals.append(((np.where(tempvals <= (drop/100))[0][0]))/srate)
    acwvals = np.array(acwvals).reshape([n_epoch, n_chan, 1])

    return acwvals


# %% Catch22
def compute_catch22(data):
    '''
    Parameters
    ----------
    data        :   EEG data  [nchan x ntime] or [n_epoch x nchan x ntime]

    Returns
    -------
    catch22vals     : Catch22 measures
    catch22labels   : Names of Catch22 measures
    '''

    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data, axis=0), axis=0)
    elif len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    n_epoch, n_chan, _ = data.shape

    catch22vals = []
    for i in tqdm(range(n_epoch), desc="Computing Catch22", position=0):
        for chan_no in range(n_chan):
            catch22vals.append(catch22_all(list(data[i][chan_no]))['values'])
    catch22vals = np.array(catch22vals).reshape([n_epoch, n_chan, 22])
    catch22labels = catch22_all(list(data[i][chan_no]))['names']

    return catch22vals, catch22labels

    


def fractional_latency(array, axis=0,fraction=0.5, dx=1.0):
    """
    Calculate the 50% fractional latency (half-maximum latency) along a specified axis 
    for each curve in a multidimensional array.

    Parameters:
        array (np.array): The input multidimensional array.
        axis (int): The axis along which to calculate the fractional latency.

    Returns:
        np.array: An array of indices representing the 50% fractional latency along the specified axis.
    """
    # Move the specified axis to the last position for easier handling
    array = np.moveaxis(array, axis, -1)
    original_shape = array.shape
    
    # Reshape to a 2D array for easier processing
    reshaped_array = array.reshape(-1, original_shape[-1])
    
    # Initialize an array to store fractional latency indices
    latency_indices = np.full(reshaped_array.shape[0], np.nan)
    
    for i, curve in enumerate(reshaped_array):
        # Calculate the cumulative area
        cumulative_area = np.cumsum(curve) * dx
        
        # Calculate the total area under the curve
        total_area = cumulative_area[-1]
        
        # Find the half-area value
        half_area = fraction * total_area
        
        # Find the first index where the cumulative area reaches or exceeds half of the total area
        above_half_area = np.where(cumulative_area >= half_area)[0]
        if above_half_area.size > 0:
            latency_indices[i] = above_half_area[0]
    
    # Reshape the result back to the original array shape, excluding the integration axis
    latency_indices = latency_indices.reshape(original_shape[:-1])
    
    return latency_indices

#%%############################################################################
def permutation_lziv_complexity(signal, m=3, tau=1):
    """
    Calculates the Permutation Lempel-Ziv Complexity (PLZC) of a signal.

    Args:
        signal (np.ndarray): The input 1D signal.
        m (int): Embedding dimension (number of consecutive samples to form a pattern).
        tau (int): Time delay between consecutive samples in a pattern.

    Returns:
        float: The Permutation Lempel-Ziv Complexity (PLZC) value.
    """
    n = len(signal)
    if n < m:
        return 0.0  # Not enough data for the embedding

    patterns = []
    for i in range(n - (m - 1) * tau):
        pattern = tuple(np.argsort(signal[i : i + m * tau : tau]))
        patterns.append(pattern)

    unique_patterns = []
    for pattern in patterns:
        if pattern not in unique_patterns:
            unique_patterns.append(pattern)

    complexity = len(unique_patterns)

    # Normalize the complexity (optional, but often done)
    if complexity > 1:
        normalized_complexity = complexity * np.log(len(unique_patterns)) / n
        return normalized_complexity
    else:
        return 0.0

def multiscale_permutation_lziv_complexity(signal, scales, m=3, tau=1):
    """
    Calculates the Multiscale Permutation Lempel-Ziv Complexity (MPLZC) of a signal.

    Args:
        signal (np.ndarray): The input 1D signal (EEG channel data).
        scales (list or np.ndarray): A list or array of integer scaling factors.
        m (int): Embedding dimension for PLZC.
        tau (int): Time delay for PLZC.

    Returns:
        np.ndarray: An array of MPLZC values, one for each scale.
    """
    mplzc_values = []
    for scale in scales:
        if scale == 1:
            coarse_grained_signal = signal
        else:
            n = len(signal)
            remainder = n % scale
            if remainder != 0:
                signal_trimmed = signal[:-remainder]
            else:
                signal_trimmed = signal

            num_segments = len(signal_trimmed) // scale
            coarse_grained_signal = np.mean(signal_trimmed.reshape((num_segments, scale)), axis=1)

        if len(coarse_grained_signal) >= m:
            plzc = permutation_lziv_complexity(coarse_grained_signal, m, tau)
            mplzc_values.append(plzc)
        else:
            mplzc_values.append(0.0)  # Not enough data after coarse-graining

    return np.array(mplzc_values)


#%%############################################################################

def generate_multieegfeatures(
        data,srate,chanlist,
        featurelist = ['psd','fooof','irasa','nonlinear','acw'],
        psdtype='welch',
        kwargs_psd=dict(scaling='density',average='median',window="hamming",
                        nperseg = None),
        freq_range=[1,40],
        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
               (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
               (30, 40, 'Gamma1')]):
    
    from yasa import bandpower_from_psd_ndarray

    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data,axis=0),axis=0)        
    elif len(data.shape) == 2:
        if len(chanlist) == data.shape[0]: 
            data = np.expand_dims(data,axis=0)
        else:
            data = np.expand_dims(data,axis=1)
    n_epoch,n_chan,_ = data.shape
    
    df_list = []
    # import pdb; pdb.set_trace()
    # PSD
    if ('psd' in featurelist) or ('fooof' in featurelist):
        psdvals,psdfreqs = compute_psd(
            data,srate,
            psdtype=psdtype,kwargs_psd=kwargs_psd)
        psdvals[psdvals<0] = 0 # removing the negative values of psd
        psdbandvals = bandpower_from_psd_ndarray(
            psdvals.transpose([1,0,2]), psdfreqs, bands=bands)
        # df_list.append(pd.DataFrame(
        #     psdbandvals.reshape([len(bands),n_epoch*n_chan]).T,
        #     columns=[x[-1]+'_PSD' for x in bands]))
        df_list.append(pd.DataFrame(
            psdbandvals.transpose([2,1,0]).reshape([n_epoch*n_chan,len(bands)]),
            columns=[x[-1]+'_PSD' for x in bands]))
    
    # FOOOF
    if 'fooof' in featurelist:
        fooofpsd,fooofparamvals,foooffreqs,fooofparamlabels,fooofpsdlabels = compute_fooof(
            psdvals,psdfreqs,freq_range=freq_range,psdout=True)
        fooofpsd[fooofpsd<0] = 0 # removing the negative values of psd
        fooofbandvals = bandpower_from_psd_ndarray(
            fooofpsd[:,:,1,:].transpose([1,0,2]), foooffreqs, bands=bands)
        # df_list.append(pd.DataFrame(
        #     np.concatenate(
        #         [fooofbandvals.reshape([len(bands),n_epoch*n_chan]).T,
        #          fooofparamvals.transpose([2,1,0]).reshape([len(fooofparamlabels),n_epoch*n_chan]).T],
        #         axis=-1),
        #     columns=[x[-1]+'_FOOOF' for x in bands]+[x+'_FOOOF' for x in fooofparamlabels]))
        df_list.append(pd.DataFrame(
            np.concatenate(
                [fooofbandvals.transpose([2,1,0]).reshape([n_epoch*n_chan,len(bands)]),
                 fooofparamvals.reshape([n_epoch*n_chan,len(fooofparamlabels)])],
                axis=-1),
            columns=[x[-1]+'_FOOOF' for x in bands]+[x+'_FOOOF' for x in fooofparamlabels]))
    
    # IRASA
    if 'irasa' in featurelist:
        irasaoscvals,irasaapervals,irasafreqs,irasaaperlabels = compute_irasa(
            data,srate,freq_range=freq_range,
            psdtype=psdtype,kwargs_psd=kwargs_psd)
        irasaoscvals[irasaoscvals<0] = 0 # removing the negative values of psd
        irasabandvals = bandpower_from_psd_ndarray(
            irasaoscvals.transpose([1,0,2]), irasafreqs, bands=bands)
        # df_list.append(pd.DataFrame(
        #     np.concatenate(
        #         [irasabandvals.reshape([len(bands),n_epoch*n_chan]).T,
        #          irasaapervals.transpose([2,1,0]).reshape([len(irasaaperlabels),n_epoch*n_chan]).T],
        #         axis=-1),
        #     columns=[x[-1]+'_Irasa' for x in bands]+[x+'_Irasa' for x in irasaaperlabels]))
        df_list.append(pd.DataFrame(
            np.concatenate(
                [irasabandvals.transpose([2,1,0]).reshape([n_epoch*n_chan,len(bands)]),
                 irasaapervals.reshape([n_epoch*n_chan,len(irasaaperlabels)])],
                axis=-1),
            columns=[x[-1]+'_Irasa' for x in bands]+[x+'_Irasa' for x in irasaaperlabels]))
    
    # NON-LINEAR
    if 'nonlinear' in featurelist:
        nonlinearvals,nonlinearlabels = compute_nonlinear(
            data)
        # df_list.append(pd.DataFrame(
        #     nonlinearvals.transpose([2,1,0]).reshape([len(nonlinearlabels),n_epoch*n_chan]).T,
        #     columns=[x+'_nonlinear' for x in nonlinearlabels]))
        df_list.append(pd.DataFrame(
            nonlinearvals.reshape([n_epoch*n_chan,len(nonlinearlabels)]),
            columns=[x+'_nonlinear' for x in nonlinearlabels]))
        
    # ACW
    if 'acw' in featurelist:
        acwvals = compute_acw(
            data,srate)
        df_list.append(pd.DataFrame(
            acwvals.reshape([n_epoch*n_chan,1]),
            columns=['ACW']))
    
    # Catch22
    if 'catch22' in featurelist:
        catch22vals,catch22labels = compute_catch22(
            data)
        # df_list.append(pd.DataFrame(
        #     catch22vals.transpose([2,1,0]).reshape([len(catch22labels),n_epoch*n_chan]).T,
        #     columns=[x+'_catch22' for x in catch22labels]))
        df_list.append(pd.DataFrame(
            catch22vals.reshape([n_epoch*n_chan,len(catch22labels)]),
            columns=[x+'_catch22' for x in catch22labels]))
    
    # Compile all features
    multieegfeatures_df         = pd.concat(df_list, axis=1).reset_index()
    multieegfeatures_df         = multieegfeatures_df.drop(['index'],axis=1)
    multieegfeatures_df['Chan'] = chanlist*n_epoch
    epoch_nos                   = []                                 
    for i in range(n_epoch):
        for j in range(n_chan):
            epoch_nos.append(i+1)
    epoch_nos                   = np.array(epoch_nos)
    multieegfeatures_df['Epoch'] = epoch_nos
    
    return multieegfeatures_df




#%%############################################################################

# Graph theory
def compute_connectivitygraphtheory(
        con_vals,chanlist,
        threshold_percentage    = 90,
        bands                   = [
            (1,4,'Delta'),(4,8,'Theta'),(6,10,'ThetaAlpha'),(8,12,'Alpha'), 
            (12,18,'Beta1'),(18,30,'Beta2'),(30,40,'Gamma1')
            ]
        ):
    import networkx as nx
    import bct
    
    # Prepare
    if len(con_vals.shape) == 1:
        con_vals = np.expand_dims(np.expand_dims(con_vals,axis=-1),axis=-1)        
    elif len(con_vals.shape) == 2:
        if len(bands) == con_vals.shape[-1]: 
            con_vals = np.expand_dims(con_vals,axis=-2)
        else:
            con_vals = np.expand_dims(con_vals,axis=-1)
    n_convals,n_epoch,n_bands = con_vals.shape
    n_chan = len(chanlist)

    df = []
    for epoch_no in range(n_epoch):      
        temp_df3 = []
        for band_no in range(n_bands):
            
            # Convert to square matrix            
            if n_convals < n_chan*n_chan:
                conn_matrix     = np.zeros([n_chan,n_chan])
                con_indices              = np.tril_indices(n_chan, k=-1)
                conn_matrix[con_indices] = con_vals[:,epoch_no,band_no]
            else:
                conn_matrix = con_vals.reshape([n_chan,n_chan])
                        
            # Binarize the matrix based on threshold
            threshold = np.percentile(conn_matrix,threshold_percentage)
            bin_matrix = (conn_matrix > threshold).astype(int)
            
            # Create a graph using NetworkX
            G = nx.from_numpy_array(bin_matrix)
            
            # Degree centrality
            degree_centrality = nx.degree_centrality(G)
            
            # Betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(G)
            
            # Clustering coefficient
            clustering_coeff = nx.clustering(G)
            
            # Global efficiency
            global_eff = bct.efficiency_bin(bin_matrix)
            
            # Small-worldness
            C_real = np.mean(list(clustering_coeff.values()))
            L_real = nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan
        
            rand_graph = nx.erdos_renyi_graph(G.number_of_nodes(), np.mean(bin_matrix))
            C_rand = np.mean(list(nx.clustering(rand_graph).values()))
            L_rand = nx.average_shortest_path_length(rand_graph) if nx.is_connected(rand_graph) else np.nan
            
            small_worldness = (C_real / C_rand) / (L_real / L_rand) if C_rand > 0 and L_rand > 0 else np.nan
            
            # Compute connectivity intensity
            con_chintensity = (np.nansum(conn_matrix,axis=0) + 
                               np.nansum(conn_matrix,axis=1))/n_chan
            
            # Compile the measures
            measures = {
                "DegreeCentrality":         degree_centrality,
                "BetweennessCentrality":    betweenness_centrality,
                "ClusteringCoefficient":    clustering_coeff,
                "GlobalEfficiency":         global_eff,
                "SmallWorldness":           small_worldness,
                "ChanIntensity":            con_chintensity
            }    
            temp_df2 = []
            for measure in measures:
                temp_df1 = pd.DataFrame([measures[measure]])
                if temp_df1.shape[-1] == n_chan:
                    temp_df1.columns = [f'{bands[band_no][-1]}_{measure}_{x}' for x in chanlist]  
                else:
                    temp_df1.columns = [f'{bands[band_no][-1]}_{measure}_allchan']
                    
                if len(temp_df2) == 0:                            
                    temp_df2 = temp_df1              
                else:
                    temp_df2  = pd.concat([temp_df2,temp_df1],axis=1)
            
            
            if len(temp_df3) == 0:                            
                temp_df3 = temp_df2              
            else:
                temp_df3  = pd.concat([temp_df3,temp_df2],axis=1)
        
        temp_df3['Epoch_no']  = epoch_no+1
        if len(df) == 0:                            
            df = temp_df3              
        else:
            df  = pd.concat([df,temp_df3],axis=0)
            
    return df



# Bivariate EEG phase connectivity
def compute_mnebivariateconnectivity(
        data,srate,chanlist,
        con_types       = ['coh','plv','ciplv','pli','wpli'],
        freqs           = [], 
        freqcycles      = [],
        decim           = 3,
        chanlevel       = True, 
        bands           = [
            (1,4,'Delta'),(4,8,'Theta'),(6,10,'ThetaAlpha'),(8,12,'Alpha'), 
            (12,18,'Beta1'),(18,30,'Beta2'),(30,40,'Gamma1')
            ]
        ):
    
    import mne_connectivity as mnecon    
    
    # Prepare
    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data,axis=0),axis=0)        
    elif len(data.shape) == 2:
        if len(chanlist) == data.shape[0]: 
            data = np.expand_dims(data,axis=0)
        else:
            data = np.expand_dims(data,axis=1)
    n_epoch,n_chan,ntimes = data.shape
    
    fmin    = min([x[0] for x in bands])
    fmax    = max([x[1] for x in bands])
    
    if len(freqs) == 0:
        freqs   = np.logspace(np.log10(fmin), np.log10(fmax), 15)
        freqs   = freqs[np.logical_and((freqs >= fmin),(freqs <= fmax))]
    
    if len(freqcycles) == 0:
        freqcycles = np.logspace(np.log10(3),np.log10(fmax/2),len(freqs)).astype(int)     
    
    # Compute
    conn = mnecon.spectral_connectivity_time(
        data,
        method      = con_types,
        freqs       = freqs, 
        faverage    = False,
        n_cycles    = freqcycles,
        mode        = 'cwt_morlet',
        sfreq       = srate,
        fmin        = fmin,
        fmax        = fmax,
        decim       = decim, 
        n_jobs      = -1
    )    
    
    # Compile into a dataframe        
    for con_no in range(len(con_types)):
        con_dataall     = conn[con_no].get_data().transpose([1,0,2])
                
        # Group into bands
        conn_bandwise = np.zeros(list(con_dataall.shape[:-1]) + [len(bands)])
        for band_no in range(len(bands)):
            bandindx = np.logical_and(
                freqs>=bands[band_no][0],freqs<=bands[band_no][1])
            conn_bandwise[:,:,band_no] = np.mean(con_dataall[:,:,bandindx],axis=-1)                
                        
        if chanlevel:                  
            
            # Graph theory measures of each channel
            df = compute_connectivitygraphtheory(
                conn_bandwise,chanlist,threshold_percentage=90,bands=bands)                        
            
        else:
            
            # Create square matrix version  
            conn_matrixbandwise = conn_bandwise.reshape([
                len(chanlist),len(chanlist),conn_bandwise.shape[-2],conn_bandwise.shape[-1]])  
            
            # Raw connectivity values for all valid channel pairs
            lower_tri_indices = np.tril_indices(len(chanlist), k=-1)
            conn_bandwise = conn_matrixbandwise[lower_tri_indices]
            n_conn,n_epoch,n_bands = conn_bandwise.shape
            
            chanpairnames = [f"{chanlist[i]}_{chanlist[j]}" 
                             for i, j in zip(lower_tri_indices[0], lower_tri_indices[1])]
            
            df = []
            for epoch_no in range(n_epoch):      
                temp_df1 = []
                for band_no in range(n_bands):
                    temp_df2 = pd.DataFrame([conn_bandwise[:,epoch_no,band_no]])
                    temp_df2.columns = [f'{con_types[con_no]}_{bands[band_no][-1]}_conn_{x}' for x in chanpairnames]  
                    
                    if len(temp_df1) == 0:                            
                        temp_df1 = temp_df2              
                    else:
                        temp_df1 = pd.concat([temp_df1,temp_df2],axis=1)
                
                temp_df1['Epoch_no']  = epoch_no+1
                if len(df) == 0:                            
                    df = temp_df1              
                else:
                    df = pd.concat([df,temp_df1],axis=0)
        
        return df


    
def compute_fritesbivariateconnectivity(
        data,srate,chanlist,
        con_types       = ['gcmi'],
        slwin_len       = .5, 
        slwin_step      = .2, 
        chanlevel       = True, 
        avgtime         = True, 
        bands           = [
            (1,4,'Delta'),(4,8,'Theta'),(6,10,'ThetaAlpha'),(8,12,'Alpha'), 
            (12,18,'Beta1'),(18,30,'Beta2'),(30,40,'Gamma1')
            ]
        ):
    
    # Prepare
    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data,axis=0),axis=0)        
    elif len(data.shape) == 2:
        if len(chanlist) == data.shape[0]: 
            data = np.expand_dims(data,axis=0)
        else:
            data = np.expand_dims(data,axis=1)
    n_epoch,n_chan,ntimes = data.shape
        
    for con_no in range(len(con_types)):
        if con_types[con_no] == 'gcmi':
            #%% Compute spectral connectivity using Frites
            from frites.estimator import GCMIEstimator
            from frites.conn import conn_dfc, define_windows
            import xarray as xr
            from mne.filter import filter_data
        
            # Prepare
            trials      = np.arange(n_epoch)
            times       = np.arange(ntimes)/srate            
            sl_win      = define_windows(times, slwin_len=slwin_len, slwin_step=slwin_step)[0]
            
            # Compute
            conn_bandwise = []
            for band_no in range(len(bands)):
                try:
                    # Band pass filter the data
                    banddata = filter_data(data, srate, bands[band_no][0], bands[band_no][1])
                            
                    # Convert to xarray
                    x = xr.DataArray(banddata, dims=('trials', 'space', 'times'),
                                     coords=(trials, chanlist, times))
                    
                    # Compute
                    dfc = conn_dfc(x, times='times', roi='space',win_sample=sl_win,
                                   estimator=GCMIEstimator(gpu=False))                    
                    con_dataall     = dfc.to_numpy().transpose([1,0,2])
                                        
                    
                    if len(conn_bandwise) == 0:
                        conn_bandwise    = np.zeros([len(bands)]+list(con_dataall.shape))
                    conn_bandwise[band_no] = con_dataall
                    
                except:
                    continue            
                    
            # Collapse across subepochs
            if avgtime:
                conn_bandwise = np.mean(conn_bandwise,axis=-1).transpose([1,2,0]) 
                                
                if chanlevel:                  
                    
                    # Graph theory measures of each channel
                    df = compute_connectivitygraphtheory(
                        conn_bandwise,chanlist,threshold_percentage=90,bands=bands)                        
                    
                else:
                    
                    # Raw connectivity values for all valid channel pairs
                    lower_tri_indices = np.tril_indices(len(chanlist), k=-1)
                    n_conn,n_epoch,n_bands = conn_bandwise.shape
                    
                    chanpairnames = [f"{chanlist[i]}_{chanlist[j]}" 
                                     for i, j in zip(lower_tri_indices[0], lower_tri_indices[1])]
                    
                    df = []
                    for epoch_no in range(n_epoch):      
                        temp_df1 = []
                        for band_no in range(n_bands):
                            temp_df2 = pd.DataFrame([conn_bandwise[:,epoch_no,band_no]])
                            temp_df2.columns = [f'{con_types[con_no]}_{bands[band_no][-1]}_conn_{x}' for x in chanpairnames]  
                            
                            if len(temp_df1) == 0:                            
                                temp_df1 = temp_df2              
                            else:
                                temp_df1 = pd.concat([temp_df1,temp_df2],axis=1)
                        
                        temp_df1['Epoch_no']  = epoch_no+1
                        if len(df) == 0:                            
                            df = temp_df1              
                        else:
                            df = pd.concat([df,temp_df1],axis=0)
                
            
            else:
                
                conn_bandwise = conn_bandwise.transpose([1,2,0,3]) 
                n_subepochs = conn_bandwise.shape[-1]
                df = []
                for subepoch_no in range(n_subepochs):                
                    if chanlevel:                  
                        
                        # Graph theory measures of each channel
                        temp_df0 = compute_connectivitygraphtheory(
                            conn_bandwise[:,:,:,subepoch_no],chanlist,threshold_percentage=90,bands=bands)                        
                        temp_df0['SubEpoch_no']  = subepoch_no+1
                        
                    else:
                        
                        # Raw connectivity values for all valid channel pairs
                        lower_tri_indices = np.tril_indices(len(chanlist), k=-1)
                        n_conn,n_epoch,n_bands = conn_bandwise[:,:,:,subepoch_no].shape
                        
                        chanpairnames = [f"{chanlist[i]}_{chanlist[j]}" 
                                         for i, j in zip(lower_tri_indices[0], lower_tri_indices[1])]
                        
                        temp_df0 = []
                        for epoch_no in range(n_epoch):      
                            temp_df1 = []
                            for band_no in range(n_bands):
                                temp_df2 = pd.DataFrame([conn_bandwise[:,epoch_no,band_no,subepoch_no]])
                                temp_df2.columns = [f'{con_types[con_no]}_{bands[band_no][-1]}_conn_{x}' for x in chanpairnames]  
                                
                                if len(temp_df1) == 0:                            
                                    temp_df1 = temp_df2              
                                else:
                                    temp_df1 = pd.concat([temp_df1,temp_df2],axis=1)
                            
                            temp_df1['Epoch_no']  = epoch_no+1
                            if len(temp_df0) == 0:                            
                                temp_df0 = temp_df1              
                            else:
                                temp_df0 = pd.concat([temp_df0,temp_df1],axis=0)
                        temp_df0['SubEpoch_no']  = subepoch_no+1
                    if len(df) == 0:                            
                        df = temp_df0              
                    else:
                        df = pd.concat([df,temp_df0],axis=0)
                    
            return df
                
     

    
def compute_cccbivariateconnectivity(
        data,srate,chanlist,
        con_types       = ['ccc'],
        slwin_len       = 0.4, # 100 points for SR of 250Hz
        slwin_curr      = 0.08,# 20 points for SR of 250Hz
        slwin_step      = 0.4, # 100 points for SR of 250Hz
        nbins           = 4,
        chanlevel       = True,
        bands           = [
            (1,4,'Delta'),(4,8,'Theta'),(6,10,'ThetaAlpha'),(8,12,'Alpha'), 
            (12,18,'Beta1'),(18,30,'Beta2'),(30,40,'Gamma1')
            ]
        ):
    
    # Prepare
    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data,axis=0),axis=0)        
    elif len(data.shape) == 2:
        if len(chanlist) == data.shape[0]: 
            data = np.expand_dims(data,axis=0)
        else:
            data = np.expand_dims(data,axis=1)
    n_epoch,n_chan,ntimes = data.shape
    
    
    lower_indices = np.tril_indices(n_chan, k=-1)  # Lower triangle (excluding diagonal)
    upper_indices = np.triu_indices(n_chan, k=1)   # Upper triangle (excluding diagonal)    
    con_indices   = (np.concatenate([lower_indices[0], upper_indices[0]]),
                     np.concatenate([lower_indices[1], upper_indices[1]]))
    
    n_convals = len(con_indices[0])
        
    for con_no in range(len(con_types)):
        if con_types[con_no] == 'ccc':
            
            #%% Compute spectral connectivity using CCC
            from ETC import CCC
            from mne.filter import filter_data
            
            # Compute
            conn_bandwise = np.zeros([n_convals,n_epoch,len(bands)])
            for band_no in range(len(bands)):
                try:
                    # Band pass filter the data
                    banddata = filter_data(data, srate, bands[band_no][0], bands[band_no][1])
                    
                    for epoch_no in tqdm(
                            range(n_epoch),
                            desc=f"Computing CCC | Band {band_no+1} | Epoch"):
                        for chanpair_no in range(n_convals):
                            conn_bandwise[chanpair_no,epoch_no,band_no] = CCC.compute(
                                seq_x=banddata[epoch_no,con_indices[0][chanpair_no],:],
                                seq_y=banddata[epoch_no,con_indices[1][chanpair_no],:],
                                LEN_past    =   int(slwin_len*srate),
                                ADD_meas    =   int(slwin_curr*srate),
                                STEP_size   =   int(slwin_step*srate),
                                n_partitions=   int(nbins))                                             
                    
                except:
                    continue            
                    

                                
            if chanlevel:                  
                
                # Graph theory measures of each channel
                df = compute_connectivitygraphtheory(
                    conn_bandwise,chanlist,threshold_percentage=90,bands=bands)                        
                
            else:
                
                # Raw connectivity values for all valid channel pairs
                n_conn,n_epoch,n_bands = conn_bandwise.shape
                
                chanpairnames = [f"{chanlist[i]}_{chanlist[j]}" 
                                 for i, j in zip(con_indices[0], con_indices[1])]
                
                df = []
                for epoch_no in range(n_epoch):      
                    temp_df1 = []
                    for band_no in range(n_bands):
                        temp_df2 = pd.DataFrame([conn_bandwise[:,epoch_no,band_no]])
                        temp_df2.columns = [f'{con_types[con_no]}_{bands[band_no][-1]}_conn_{x}' for x in chanpairnames]  
                        
                        if len(temp_df1) == 0:                            
                            temp_df1 = temp_df2              
                        else:
                            temp_df1 = pd.concat([temp_df1,temp_df2],axis=1)
                    
                    temp_df1['Epoch_no']  = epoch_no+1
                    if len(df) == 0:                            
                        df = temp_df1              
                    else:
                        df = pd.concat([df,temp_df1],axis=0)                
            return df            







#%%############################################################################
# Multivariate EEG connectivity


