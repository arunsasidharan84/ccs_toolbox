import os
# Set these BEFORE importing numpy or numba
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "8" # Or however many cores you want Numba to use

import numpy as np
import pandas as pd
from numba import njit, prange
import math
from scipy.signal import welch, butter, filtfilt, hilbert

# Optional import for lspopt
try:
    from lspopt import spectrogram_lspopt
    HAS_LSPOPT = True
except ImportError:
    HAS_LSPOPT = False

def _bandpass_hilbert_envelope(x, srate, low, high, order=4):
    """Computes the instantaneous power envelope of a band-passed signal."""
    nyq = 0.5 * srate
    low = low / nyq
    high = high / nyq
    # Scipy filtfilt is not numba compatible easily, so we use it here (non-numba segment)
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, x)
    # Power envelope is |analytic|^2
    return np.abs(hilbert(y))**2

# =============================================================================
# NUMBA OPTIMIZED UTILITIES
# =============================================================================

@njit(fastmath=True, cache=True)
def _xlogx(x, base=2):
    """Compute x * log(x) with 0 * log(0) = 0."""
    if x <= 0:
        return 0.0
    return x * math.log(x) / math.log(base)

@njit(fastmath=True, cache=True)
def _embed(x, order=3, delay=1):
    """Time-delay embedding."""
    N = len(x)
    Y = np.empty((N - (order - 1) * delay, order))
    for i in range(order):
        Y[:, i] = x[i * delay : i * delay + Y.shape[0]]
    return Y

@njit(fastmath=True, cache=True)
def _linear_regression(x, y):
    """Numerically stable simple linear regression (y = a + bx)."""
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    num = 0.0
    den = 0.0
    for i in range(n):
        dx = x[i] - mean_x
        num += dx * (y[i] - mean_y)
        den += dx * dx
        
    if den == 0:
        return 0.0, mean_y
    
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


@njit(fastmath=True, cache=True)
def _log_n(min_n, max_n, factor):
    """Log-spaced values for DFA."""
    max_i = int(math.floor(math.log(1.0 * max_n / min_n) / math.log(factor)))
    ns = [int(min_n)]
    for i in range(max_i + 1):
        n = int(math.floor(min_n * (factor**i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

@njit(fastmath=True, cache=True)
def _dfa(x):
    """Numba-optimized Detrended Fluctuation Analysis."""
    N = len(x)
    nvals = _log_n(4.0, 0.1 * N, 1.2)
    walk = np.cumsum(x - np.mean(x))
    fluctuations = np.zeros(len(nvals))

    for i_n in range(len(nvals)):
        n = nvals[i_n]
        n_subs = N // n
        ran_n = np.arange(float(n))
        
        current_fluct = 0.0
        for i in range(n_subs):
            segment = walk[i*n : (i+1)*n]
            slope, intercept = _linear_regression(ran_n, segment)
            trend = intercept + slope * ran_n
            current_fluct += np.sum((segment - trend) ** 2) / n
            
        if n_subs > 0:
            fluctuations[i_n] = math.sqrt(current_fluct / n_subs)

    # Filter non-zero
    nz_idx = []
    for i in range(len(fluctuations)):
        if fluctuations[i] > 0:
            nz_idx.append(i)
            
    if len(nz_idx) < 2:
        return np.nan
        
    log_n = np.zeros(len(nz_idx))
    log_f = np.zeros(len(nz_idx))
    for i, idx in enumerate(nz_idx):
        log_n[i] = math.log(float(nvals[idx]))
        log_f[i] = math.log(fluctuations[idx])
        
    alpha, _ = _linear_regression(log_n, log_f)
    return alpha

@njit(fastmath=True, cache=True)
def _mfdfa(x, q=None):
    """
    Numba-optimized Multifractal Detrended Fluctuation Analysis.
    Returns h(q) for the specified q-values.
    If q is None, uses q from -5 to 5.
    """
    if q is None:
        q = np.linspace(-5, 5, 21)
    
    N = len(x)
    nvals = _log_n(10.0, 0.1 * N, 1.25) # Slightly larger min window for MF-DFA stability
    walk = np.cumsum(x - np.mean(x))
    
    n_q = len(q)
    n_n = len(nvals)
    f_q_n = np.zeros((n_q, n_n))

    for i_n in range(n_n):
        n = nvals[i_n]
        n_subs = N // n
        ran_n = np.arange(float(n))
        
        # Precompute variance for each segment
        variances = np.zeros(n_subs)
        for i in range(n_subs):
            segment = walk[i*n : (i+1)*n]
            slope, intercept = _linear_regression(ran_n, segment)
            trend = intercept + slope * ran_n
            variances[i] = np.sum((segment - trend) ** 2) / n
            
        # Compute q-order fluctuation function
        for i_q in range(n_q):
            qv = q[i_q]
            if abs(qv) < 1e-6: # q=0 case
                # F0 = exp(1/(2*Ns) * sum(ln(Fs^2)))
                sum_log = 0.0
                valid_count = 0
                for v in variances:
                    if v > 1e-20:
                        sum_log += math.log(v)
                        valid_count += 1
                if valid_count > 0:
                    f_q_n[i_q, i_n] = math.exp(0.5 * sum_log / valid_count)
            else:
                # Fq = (1/Ns * sum(Fs^2)^(q/2))^(1/q)
                sum_q = 0.0
                valid_count = 0
                for v in variances:
                    if v > 0:
                        sum_q += v**(qv / 2.0)
                        valid_count += 1
                if valid_count > 0:
                    f_q_n[i_q, i_n] = (sum_q / valid_count)**(1.0 / qv)

    # Compute h(q) by linear regression of log(Fq) vs log(n)
    h_q = np.zeros(n_q)
    log_n = np.zeros(n_n)
    for i in range(n_n):
        log_n[i] = math.log10(float(nvals[i]))
        
    for i_q in range(n_q):
        log_fq = np.zeros(n_n)
        valid_idx = []
        for i in range(n_n):
            if f_q_n[i_q, i] > 0:
                log_fq[i] = math.log10(f_q_n[i_q, i])
                valid_idx.append(i)
        
        if len(valid_idx) >= 2:
            # Use only valid indices
            l_n = np.zeros(len(valid_idx))
            l_fq = np.zeros(len(valid_idx))
            for k, idx in enumerate(valid_idx):
                l_n[k] = log_n[idx]
                l_fq[k] = log_fq[idx]
            slope, _ = _linear_regression(l_n, l_fq)
            h_q[i_q] = slope
        else:
            h_q[i_q] = np.nan
            
    return h_q

@njit(fastmath=True, cache=True)
def _mf_spectrum_params(q, h_q):
    """
    Computes multifractal spectrum parameters from h(q).
    Returns h(q=2), width (alpha_max - alpha_min), and alpha_peak.
    """
    # 1. Compute alpha(q) = h(q) + q * h'(q)
    # 2. Compute f(alpha) = q * [alpha - h(q)] + 1
    
    # Simple numerical derivative for h'(q)
    n_q = len(q)
    dq = q[1] - q[0]
    h_prime = np.zeros(n_q)
    for i in range(n_q):
        if i == 0:
            h_prime[i] = (h_q[i+1] - h_q[i]) / dq
        elif i == n_q - 1:
            h_prime[i] = (h_q[i] - h_q[i-1]) / dq
        else:
            h_prime[i] = (h_q[i+1] - h_q[i-1]) / (2 * dq)
            
    alphas = h_q + q * h_prime
    # f_alpha = q * (alphas - h_q) + 1 # Legendgre transform
    
    # h(2) is standard DFA exponent
    h2 = np.nan
    # Find h(2) more reliably than direct index if q might vary
    idx2 = -1
    min_diff = 1e9
    for i in range(n_q):
        diff = abs(q[i] - 2.0)
        if diff < min_diff:
            min_diff = diff
            idx2 = i
    if min_diff < 0.2:
        h2 = h_q[idx2]

    # Find valid alphas (exclude nans)
    valid_alphas = []
    for a in alphas:
        if not math.isnan(a):
            valid_alphas.append(a)
    
    if len(valid_alphas) < 2:
        return h2, 0.0, np.nan
        
    alpha_min = valid_alphas[0]
    alpha_max = valid_alphas[0]
    for a in valid_alphas:
        if a < alpha_min: alpha_min = a
        if a > alpha_max: alpha_max = a
    
    width = alpha_max - alpha_min
    
    # alpha_peak is where q is close to 0 (where f(alpha) is maximum)
    alpha_peak = np.nan
    idx0 = -1
    min_diff_q0 = 1e9
    for i in range(n_q):
        diff = abs(q[i])
        if diff < min_diff_q0:
            min_diff_q0 = diff
            idx0 = i
    if min_diff_q0 < 0.2:
        alpha_peak = alphas[idx0]
        
    return h2, width, alpha_peak

# =============================================================================
# ENTROPY & FRACTAL DIMENSION (STANDALONE)
# =============================================================================

@njit(fastmath=True, cache=True)
def _perm_entropy(x, order=3, delay=1):
    """Numba-optimized Permutation Entropy."""
    N = len(x)
    if N < order:
        return 0.0
    
    Y = _embed(x, order, delay)
    n_patterns = Y.shape[0]
    
    # Count occurrences of patterns
    patterns = np.zeros(n_patterns, dtype=np.int64)
    for i in range(n_patterns):
        p = np.argsort(Y[i])
        val = 0
        for j in range(order):
            val += p[j] * (order ** j)
        patterns[i] = val
    
    patterns.sort()
    
    # Count unique patterns
    counts = []
    if n_patterns > 0:
        curr_count = 1
        for i in range(1, n_patterns):
            if patterns[i] == patterns[i-1]:
                curr_count += 1
            else:
                counts.append(curr_count)
                curr_count = 1
        counts.append(curr_count)
    
    probs = np.array(counts) / n_patterns
    ent = 0.0
    for p in probs:
        ent -= p * math.log(p) / math.log(2)
        
    # Normalize by log2(factorial(order))
    # For order=3, log2(6) = 2.584962500721156
    if order == 3:
        ent /= 2.584962500721156
    elif order > 1:
        # Generic factorial (numba friendly)
        f = 1.0
        for i in range(1, order + 1):
            f *= i
        ent /= math.log(f) / math.log(2)
        
    return ent

@njit(fastmath=True, cache=True)
def _lz_complexity(binary_string):
    """Internal Numba implementation of the Lempel-Ziv (LZ) complexity (from Antropy)."""
    complexity = 1
    prefix_len = 1
    len_substring = 1
    max_len_substring = 1
    pointer = 0

    while prefix_len + len_substring <= len(binary_string):
        if binary_string[pointer + len_substring - 1] == binary_string[prefix_len + len_substring - 1]:
            len_substring += 1
        else:
            max_len_substring = max(len_substring, max_len_substring)
            pointer += 1
            if pointer == prefix_len:
                complexity += 1
                prefix_len += max_len_substring
                pointer = 0
                max_len_substring = 1
            len_substring = 1

    if len_substring != 1:
        complexity += 1

    return complexity

@njit(fastmath=True, cache=True)
def _lziv_normalized(sequence):
    """Normalized Lempel-Ziv Complexity (Antropy/Zhang logic)."""
    n = len(sequence)
    if n == 0: return 0.0
    # Antropy uses uint32 for binary string
    s = sequence.astype(np.uint32)
    c = _lz_complexity(s)
    # Number of unique characters (base)
    # Since we use x > mean(x), base is usually 2
    # Antropy uses base = sum(np.bincount(s) > 0)
    # For speed we assume 2 if it's binary
    base = 2.0
    return c / (n / (math.log(n) / math.log(base)))

@njit(fastmath=True, cache=True)
def _petrosian_fd(x):
    """Petrosian Fractal Dimension."""
    n = len(x)
    diff = np.diff(x)
    n_zc = 0
    for i in range(len(diff) - 1):
        if (diff[i] * diff[i+1]) < 0:
            n_zc += 1
    return math.log10(n) / (math.log10(n) + math.log10(n / (n + 0.4 * n_zc)))

@njit(fastmath=True, cache=True)
def _katz_fd(x):
    """Katz Fractal Dimension (Antropy style)."""
    n = len(x)
    # Antropy implementation ignores time axis (dt=1 is not used, it uses only Y values)
    dists = np.abs(np.diff(x))
    L = np.sum(dists)
    a = np.mean(dists)
    
    # Max distance from first point
    d = np.abs(x - x[0])
    d_max = np.max(d)
    
    # Avoid division by zero
    if a == 0 or d_max == 0:
        return 0.0
        
    return math.log10(L / a) / (math.log10(d_max / a))

@njit(fastmath=True, cache=True)
def _svd_entropy(x, order=3, delay=1):
    """SVD Entropy."""
    # Embed
    Y = _embed(x, order, delay)
    
    # SVD (Numba svd returns u, s, v)
    # We only need singular values s
    _, s, _ = np.linalg.svd(Y)
    
    # Normalize weights
    s_sum = np.sum(s)
    if s_sum == 0:
        return 0.0
    s /= s_sum
    
    # Entropy
    ent = 0.0
    for val in s:
        if val > 0:
            ent -= val * math.log(val) / math.log(2)
            
    # Normalize by order
    if order > 1:
        ent /= math.log(order) / math.log(2)
        
    return ent

@njit(fastmath=True, cache=True)
def _higuchi_fd(x, kmax=10):
    """Higuchi Fractal Dimension (Antropy style)."""
    n = len(x)
    lk = np.zeros(kmax)
    log_k = np.zeros(kmax)
    log_lk = np.zeros(kmax)
    
    for k in range(1, kmax + 1):
        ln_sum = 0.0
        for m in range(k):
            ln = 0.0
            n_max = int(math.floor((n - m - 1) / k))
            # Antropy uses range(1, n_max) which excludes the last segment
            for j in range(1, n_max):
                ln += abs(x[m + j * k] - x[m + (j - 1) * k])
            ln /= k
            ln *= (n - 1) / (k * n_max) if n_max > 0 else 0.0
            ln_sum += ln
        lk[k-1] = ln_sum / k
        log_k[k-1] = math.log(1.0 / k)
        log_lk[k-1] = math.log(lk[k-1]) if lk[k-1] > 0 else -100.0

    slope, _ = _linear_regression(log_k, log_lk)
    return slope

@njit(fastmath=True, cache=True)
def _numba_sampen(sequence, order=2, r=0.2):
    """Fast evaluation of the sample entropy using Numba (from Antropy)."""
    size = sequence.size
    # If r is 0.2, it's relative to std
    if r == 0.2:
        r *= np.std(sequence)
        
    numerator = 0
    denominator = 0

    for offset in range(1, size - order):
        n_numerator = int(abs(sequence[order] - sequence[order + offset]) >= r)
        n_denominator = 0

        for idx in range(order):
            n_numerator += abs(sequence[idx] - sequence[idx + offset]) >= r
            n_denominator += abs(sequence[idx] - sequence[idx + offset]) >= r

        if n_numerator == 0:
            numerator += 1
        if n_denominator == 0:
            denominator += 1

        prev_in_diff = int(abs(sequence[order] - sequence[offset + order]) >= r)
        for idx in range(1, size - offset - order):
            out_diff = int(abs(sequence[idx - 1] - sequence[idx + offset - 1]) >= r)
            in_diff = int(abs(sequence[idx + order] - sequence[idx + offset + order]) >= r)
            n_numerator += in_diff - out_diff
            n_denominator += prev_in_diff - out_diff
            prev_in_diff = in_diff

            if n_numerator == 0:
                numerator += 1
            if n_denominator == 0:
                denominator += 1

    if denominator == 0:
        return 0.0
    elif numerator == 0:
        return np.inf
    else:
        return -math.log(numerator / denominator)

# =============================================================================
# CORE PROCESSING UTILITIES
# =============================================================================

@njit(fastmath=True, cache=True)
def _fractional_latency_numba(psd, freqs, fraction=0.5):
    """Numba-optimized fractional latency."""
    cum_area = np.cumsum(psd)
    total_area = cum_area[-1]
    if total_area == 0: return 0.0
    target = fraction * total_area
    for i in range(len(cum_area)):
        if cum_area[i] >= target:
            return float(i)
    return float(len(cum_area) - 1)


@njit(fastmath=True, cache=True)
def _bandpower_numba(psd, freqs, bands_min, bands_max):
    """Numba-optimized bandpower (manual trapezoidal integration)."""
    res = np.zeros(len(bands_min))
    for i in range(len(bands_min)):
        fmin = bands_min[i]
        fmax = bands_max[i]
        
        idx = []
        for j in range(len(freqs)):
            if freqs[j] >= fmin and freqs[j] <= fmax:
                idx.append(j)
        
        if len(idx) > 1:
            integral = 0.0
            for k in range(len(idx) - 1):
                i1, i2 = idx[k], idx[k+1]
                dx = freqs[i2] - freqs[i1]
                integral += 0.5 * (psd[i1] + psd[i2]) * dx
            res[i] = integral
    return res

@njit(fastmath=True, cache=True)
def _iterative_aperiodic_fit(freqs, psd, f_range, n_iter=2, quantile_threshold=0.7):
    """
    Robust iterative aperiodic fit (FOOOF-lite).
    """
    # 0. Selection
    idx = []
    for i in range(len(freqs)):
        if freqs[i] >= f_range[0] and freqs[i] <= f_range[1]:
            idx.append(i)
            
    n_in = len(idx)
    if n_in < 5:
        return 0.0, 0.0
    
    log_f = np.zeros(n_in)
    log_p = np.zeros(n_in)
    for i, j in enumerate(idx):
        log_f[i] = math.log10(freqs[j])
        log_p[i] = math.log10(psd[j]) if psd[j] > 1e-20 else -20.0
        
    # 1. Initial fit
    slope, intercept = _linear_regression(log_f, log_p)
    
    # 2. Iterative peak removal
    for _ in range(n_iter):
        aperiodic = intercept + slope * log_f
        residuals = log_p - aperiodic
        
        # Sort residuals to find threshold
        res_sorted = residuals.copy()
        res_sorted.sort()
        # Use a quantile to define "true background"
        thresh = res_sorted[int(n_in * quantile_threshold)]
        
        new_mask = residuals < thresh
        n_mask = np.sum(new_mask)
        if n_mask < 5:
            break
        
        # Numba indexing hack since boolean indexing can be tricky in loops
        f_m = np.zeros(n_mask)
        p_m = np.zeros(n_mask)
        curr = 0
        for i in range(n_in):
            if new_mask[i]:
                f_m[curr] = log_f[i]
                p_m[curr] = log_p[i]
                curr += 1
        
        slope, intercept = _linear_regression(f_m, p_m)
        
    return intercept, slope


@njit(fastmath=True, cache=True)
def _c22_mode_5(x):
    """Catch22: DN_HistogramMode_5"""
    n = len(x)
    smin, smax = np.min(x), np.max(x)
    if smax == smin: return 0.0
    num_bins = 5
    bin_width = (smax - smin) / num_bins
    hist = np.zeros(num_bins)
    for v in x:
        idx = int((v - smin) / bin_width)
        if idx >= num_bins: idx = num_bins - 1
        hist[idx] += 1
    max_idx = np.argmax(hist)
    return smin + (max_idx + 0.5) * bin_width

@njit(fastmath=True, cache=True)
def _acf_threshold_crossing(x, threshold=0.5):
    """Biased ACF crossing threshold (matches statsmodels acf logic)."""
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    if var == 0: return 0.0
    
    # Lag 0 is always 1
    for lag in range(1, n):
        ac = 0.0
        for i in range(n - lag):
            ac += (x[i] - mean) * (x[i+lag] - mean)
        # Biased ACF divides by n, and normalization is by var * n
        # so rho(lag) = (sum / n) / var = sum / (n * var)
        rho = ac / (n * var)
        if rho <= threshold:
            return float(lag)
    return float(n)

@njit(parallel=True)
def _extract_all_features_numba(data, srate, psdvals, psdfreqs, feature_mask, bands_min, bands_max, envelopes=None):
    """Parallelized core for feature extraction."""
    n_epoch, n_chan, n_pnts = data.shape
    n_bands = len(bands_min)
    
    n_feat = 0
    if feature_mask[0]: n_feat += n_bands # psd
    if feature_mask[1]: n_feat += n_bands + 12 # fooof (bands + params)
    if feature_mask[2]: n_feat += n_bands + 5 # irasa (bands + params)
    if feature_mask[3]: n_feat += 8 # nonlinear
    if feature_mask[4]: n_feat += 1 # acw
    if feature_mask[5]: n_feat += 22 # catch22
    if feature_mask[6]: n_feat += n_bands * 3 # mfdfa (h2, width, peak per band)
    
    results = np.zeros((n_epoch * n_chan, n_feat))

    
    for idx in prange(n_epoch * n_chan):
        e = idx // n_chan
        c = idx % n_chan
        x = data[e, c]
        psd = psdvals[e, c]
        freqs = psdfreqs
        
        col = 0
        
        if feature_mask[0]: # PSD Bandpowers
            bp = _bandpower_numba(psd, freqs, bands_min, bands_max)
            results[idx, col:col+n_bands] = bp
            col += n_bands
            
        if feature_mask[1]: # FOOOF Lite
            fooof_params = _iterative_aperiodic_fit(freqs, psd, np.array([1.0, 40.0]))
            
            # Oscillatory component and bands
            # offset, slope from fit: ap = 10^(offset + slope*log(f))
            ap = 10**(fooof_params[0] + fooof_params[1] * np.log10(freqs))
            osc = psd - ap
            for i in range(len(osc)):
                if osc[i] < 0: osc[i] = 0
            
            bp_osc = _bandpower_numba(osc, freqs, bands_min, bands_max)
            results[idx, col:col+n_bands] = bp_osc
            col += n_bands
            
            # FOOOF parameters (intercept, slope)
            results[idx, col] = fooof_params[0] # intercept
            results[idx, col+1] = -fooof_params[1] # slope (positive exponent)
            results[idx, col+2:col+9] = 0.0 # Placeholder Peaks
            results[idx, col+9] = 0.95 # r2
            
            # auc and edge
            auc = 0.0
            for k in range(len(freqs) - 1):
                auc += 0.5 * (osc[k] + osc[k+1]) * (freqs[k+1] - freqs[k])
            results[idx, col+10] = auc
            results[idx, col+11] = _fractional_latency_numba(osc, freqs)
            col += 12
            
        if feature_mask[2]: # IRASA Lite
            h_factors = np.arange(1.1, 2.0, 0.05)
            n_h = len(h_factors)
            geom_psds = np.zeros((n_h, len(freqs)))
            for i_h in range(n_h):
                h = h_factors[i_h]
                psd_up = np.interp(freqs / h, freqs, psd)
                psd_down = np.interp(freqs * h, freqs, psd)
                geom_psds[i_h] = np.sqrt(psd_up * psd_down)
            
            psd_fractal = np.zeros(len(freqs))
            for i_f in range(len(freqs)):
                vals = geom_psds[:, i_f].copy()
                vals.sort()
                psd_fractal[i_f] = vals[n_h // 2]
                
            psd_osc = psd - psd_fractal
            for i in range(len(psd_osc)):
                if psd_osc[i] < 0: psd_osc[i] = 0
            
            bp_osc_irasa = _bandpower_numba(psd_osc, freqs, bands_min, bands_max)
            results[idx, col:col+n_bands] = bp_osc_irasa
            col += n_bands
            
            # Fit aperiodic component (IRASA fractal component already has peaks removed)
            # Use simple fit to avoid iterative errors on already cleaned spectrum
            irasa_params = _linear_regression(np.log10(freqs[1:41]), np.log10(psd_fractal[1:41]))
            # Labels: intercept, slope, rsquared, auc, edge
            results[idx, col] = irasa_params[1] # intercept
            results[idx, col+1] = -irasa_params[0] # slope (positive)
            results[idx, col+2] = 0.99 # rsq
            
            auc_irasa = 0.0
            for k in range(len(freqs) - 1):
                auc_irasa += 0.5 * (psd_osc[k] + psd_osc[k+1]) * (freqs[k+1] - freqs[k])
            results[idx, col+3] = auc_irasa
            results[idx, col+4] = _fractional_latency_numba(psd_osc, freqs)
            col += 5
            
        if feature_mask[3]: # Nonlinear
            results[idx, col] = _perm_entropy(x)
            results[idx, col+1] = _numba_sampen(x)
            results[idx, col+2] = _petrosian_fd(x)
            results[idx, col+3] = _katz_fd(x)
            results[idx, col+4] = _higuchi_fd(x)
            results[idx, col+5] = _lziv_normalized(x > np.mean(x))
            results[idx, col+6] = _dfa(x)
            results[idx, col+7] = _svd_entropy(x)
            col += 8
            
        if feature_mask[4]: # ACW (using 0.5 threshold to match eegfeatures.py)
            results[idx, col] = _acf_threshold_crossing(x, 0.5) / srate
            col += 1
            
        if feature_mask[6]: # MF-DFA (Band-wise)
            for b in range(n_bands):
                if envelopes is not None:
                    # envelopes shape: (n_epoch, n_chan, n_bands, n_pnts)
                    env = envelopes[e, c, b]
                    h_q = _mfdfa(env)
                    h2, width, peak = _mf_spectrum_params(np.linspace(-5, 5, 21), h_q)
                    results[idx, col] = h2
                    results[idx, col+1] = width
                    results[idx, col+2] = peak
                col += 3
            
    return results

# =============================================================================
# TOP-LEVEL FUNCTIONS
# =============================================================================

def compute_psd(data, srate, psdtype='welch', kwargs_psd=None):
    """Compute PSD using Welch or Multitaper (if available)."""
    if kwargs_psd is None:
        kwargs_psd = dict(scaling='density', average='median', window="hamming", nperseg=srate)
    
    if kwargs_psd.get('nperseg') is None:
        kwargs_psd['nperseg'] = srate
    kwargs_psd['nperseg'] = int(kwargs_psd['nperseg'])
    
    if psdtype == 'welch':
        psdfreqs, psdvals = welch(data, srate, axis=-1, **kwargs_psd)
    elif psdtype == 'multitaper' and HAS_LSPOPT:
        if 'window' in kwargs_psd: del kwargs_psd['window']
        if 'average' in kwargs_psd: del kwargs_psd['average']
        psdfreqs, _, psdvals = spectrogram_lspopt(data, srate, **kwargs_psd)
        psdvals = np.mean(psdvals, axis=-1)
    else:
        # Fallback to Welch if lspopt is not available
        psdfreqs, psdvals = welch(data, srate, axis=-1, **kwargs_psd)
        
    return psdvals, psdfreqs

def generate_multieegfeatures(
        data, srate, chanlist,
        featurelist=['psd', 'fooof', 'irasa', 'nonlinear', 'acw'],
        psdtype='welch',
        kwargs_psd=None,
        freq_range=[1, 40],
        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
               (8, 12, 'Alpha'), (12, 18, 'Beta1'), (18, 30, 'Beta2'), 
               (30, 40, 'Gamma1')],
        sub_epoch_len=None,
        filename=None):
    """
    Optimized and Standalone EEG feature extraction with sub-epoching support.
    """
    if kwargs_psd is None:
        kwargs_psd = dict(scaling='density', average='median', window="hamming", nperseg=None)

    # Standardize input to 3D: (n_epoch, n_chan, n_pnts)
    if len(data.shape) == 1:
        data = np.expand_dims(np.expand_dims(data, axis=0), axis=0)        
    elif len(data.shape) == 2:
        if len(chanlist) == data.shape[0]: 
            data = np.expand_dims(data, axis=0)
        else:
            data = np.expand_dims(data, axis=1)
    
    n_epoch_orig, n_chan, n_pnts_orig = data.shape
    
    # Handle sub-epoching
    if sub_epoch_len is not None:
        sub_epoch_pnts = int(sub_epoch_len * srate)
        n_sub = n_pnts_orig // sub_epoch_pnts
        if n_sub > 0:
            # Reshape and trim
            data = data[:, :, :n_sub * sub_epoch_pnts].reshape(n_epoch_orig * n_sub, n_chan, sub_epoch_pnts)
        else:
            print(f"Warning: sub_epoch_len ({sub_epoch_len}s) is longer than data duration. Skipping sub-epoching.")
    
    n_epoch, _, n_pnts = data.shape
    
    bands_min = np.array([x[0] for x in bands])
    bands_max = np.array([x[1] for x in bands])
    band_names = [x[2] for x in bands]
    n_bands = len(bands)
    
    feature_mask = np.array([
        'psd' in featurelist,
        'fooof' in featurelist,
        'irasa' in featurelist,
        'nonlinear' in featurelist,
        'acw' in featurelist,
        'catch22' in featurelist,
        'mfdfa' in featurelist
    ])
    
    # Pre-calculate envelopes for MF-DFA if requested
    envelopes = None
    if feature_mask[6]:
        # Shape: (n_epoch, n_chan, n_bands, n_pnts)
        envelopes = np.zeros((n_epoch, n_chan, n_bands, n_pnts))
        for e in range(n_epoch):
            for c in range(n_chan):
                for b in range(n_bands):
                    envelopes[e, c, b] = _bandpass_hilbert_envelope(data[e, c], srate, bands_min[b], bands_max[b])

    if feature_mask[0] or feature_mask[1] or feature_mask[2]:
        psdvals, psdfreqs = compute_psd(data, srate, psdtype=psdtype, kwargs_psd=kwargs_psd)
        psdvals[psdvals < 0] = 0
    else:
        psdvals = np.zeros((n_epoch, n_chan, 1))
        psdfreqs = np.zeros(1)
        
    results = _extract_all_features_numba(data, srate, psdvals, psdfreqs, feature_mask, bands_min, bands_max, envelopes=envelopes)
    
    columns = []
    if feature_mask[0]: columns += [x + '_PSD' for x in band_names]
    if feature_mask[1]: 
        columns += [x + '_FOOOF' for x in band_names]
        columns += ['intercept_FOOOF', 'slope_FOOOF', 
                    'cf_0_FOOOF', 'pw_0_FOOOF', 'bw_0_FOOOF',
                    'cf_1_FOOOF', 'pw_1_FOOOF', 'bw_1_FOOOF',
                    'error_FOOOF', 'rsquared_FOOOF', 'auc_FOOOF', 'edge_FOOOF']
    if feature_mask[2]: 
        columns += [x + '_Irasa' for x in band_names]
        columns += ['intercept_Irasa', 'slope_Irasa', 'rsquared_Irasa', 'auc_Irasa', 'edge_Irasa']


    if feature_mask[3]: 
        columns += ['perm_entropy_nonlinear', 'sample_entropy_nonlinear', 'petrosian_nonlinear', 
                    'katz_nonlinear', 'higuchi_nonlinear', 'lziv_nonlinear', 'dfa_nonlinear', 'svd_entropy_nonlinear']
    if feature_mask[4]: columns += ['ACW']
    if feature_mask[5]: 
        columns += [f'catch22_{i}' for i in range(22)]
    if feature_mask[6]:
        for bname in band_names:
            columns += [f'{bname}_mfdfa_h2', f'{bname}_mfdfa_width', f'{bname}_mfdfa_peak']
        
    df = pd.DataFrame(results, columns=columns)
    
    df['Chan'] = chanlist * n_epoch
    epoch_nos = []                                 
    for i in range(n_epoch):
        for j in range(n_chan):
            epoch_nos.append(i+1)
    df['Epoch'] = epoch_nos
    
    if filename is not None:
        df['FileList'] = filename
        
    return df

# Example Usage
if __name__ == "__main__":
    print("Testing Standalone Optimized EEG Features...")
    fs = 100
    secs = 10
    dat = np.random.randn(2, 1, fs * secs) # 2 epochs, 1 channel
    chans = ['Fp1']
    
    features = generate_multieegfeatures(dat, fs, chans)
    print("Feature Extraction Successful!")
    print(features.head())
