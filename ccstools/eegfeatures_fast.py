import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*fooof.*")
warnings.filterwarnings("ignore", message=".*specparam.*")

import sys, os
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

import os

# Threading limits for BLAS/LAPACK — only set if not already configured by the
# environment (e.g. conda, MNE, or a parent process).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# NUMBA_NUM_THREADS is intentionally NOT set here.
# Numba raises RuntimeError if this value is changed after its thread pool is
# initialised — which happens as soon as any parallel-decorated function is
# compiled. MNE and other libraries in the same session may already have
# initialised the pool. We leave the value entirely to Numba's own default
# (all available cores) or whatever the user/environment has set upstream.

import numpy as np
import pandas as pd
from numba import njit, prange
import math
from scipy.signal import welch, butter, filtfilt, hilbert
def _make_progress_printer(n_epoch, desc):
    """
    Console-safe progress bar that works in Spyder, terminals, and most IPython
    consoles without relying on notebook widgets.

    Falls back to a no-op updater if tqdm is unavailable.
    """
    try:
        from tqdm import tqdm  # use standard console tqdm, not tqdm.auto
        bar = tqdm(
            total=n_epoch,
            desc=desc,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.2,   # avoids excessive redraw overhead
            unit="epoch"
        )

        last_n = [0]

        def update(e):
            target = e + 1
            delta = target - last_n[0]
            if delta > 0:
                bar.update(delta)
                last_n[0] = target
            if target >= n_epoch:
                bar.close()

        return update

    except Exception:
        # Safe fallback for any console if tqdm is not installed
        import sys
        import time

        t0 = [time.time()]
        last_n = [0]
        bar_width = 30

        def update(e):
            target = e + 1
            if target <= last_n[0]:
                return
            last_n[0] = target

            elapsed = time.time() - t0[0]
            frac = target / max(n_epoch, 1)
            filled = int(bar_width * frac)
            bar_str = '█' * filled + '░' * (bar_width - filled)
            rate = target / elapsed if elapsed > 0 else 0.0
            eta = (n_epoch - target) / rate if rate > 0 else 0.0

            sys.stdout.write(
                f'\r{desc}: [{bar_str}] {target}/{n_epoch} '
                f'{elapsed:.0f}s<{eta:.0f}s {rate:.2f} epoch/s'
            )
            sys.stdout.flush()

            if target >= n_epoch:
                sys.stdout.write('\n')
                sys.stdout.flush()

        return update

# Optional import for lspopt (multitaper PSD)
try:
    from lspopt import spectrogram_lspopt
    HAS_LSPOPT = True
except ImportError:
    HAS_LSPOPT = False

# =============================================================================
# NON-NUMBA UTILITIES
# =============================================================================

def _bandpass_hilbert_envelope(x, srate, low, high, order=4):
    """Computes the instantaneous power envelope of a band-passed signal."""
    nyq = 0.5 * srate
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    y = filtfilt(b, a, x)
    return np.abs(hilbert(y)) ** 2


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
        kw = {k: v for k, v in kwargs_psd.items() if k not in ('window', 'average')}
        psdfreqs, _, psdvals = spectrogram_lspopt(data, srate, **kw)
        psdvals = np.mean(psdvals, axis=-1)
    else:
        psdfreqs, psdvals = welch(data, srate, axis=-1, **kwargs_psd)
    return psdvals, psdfreqs

# =============================================================================
# NUMBA OPTIMIZED UTILITIES
# =============================================================================

@njit(fastmath=True, cache=True)
def _embed(x, order=3, delay=1):
    """Time-delay embedding."""
    N = len(x)
    Y = np.empty((N - (order - 1) * delay, order))
    for i in range(order):
        Y[:, i] = x[i * delay: i * delay + Y.shape[0]]
    return Y


@njit(fastmath=True, cache=True)
def _linear_regression(x, y):
    """
    Numerically stable simple linear regression (y = slope*x + intercept).
    Uses an explicit loop — safer than vectorised np.sum under fastmath=True.
    """
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
    """Log-spaced integer values for DFA/MFDFA window sizes."""
    max_i = int(math.floor(math.log(1.0 * max_n / min_n) / math.log(factor)))
    ns = [int(min_n)]
    for i in range(max_i + 1):
        n = int(math.floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)


@njit(fastmath=True, cache=True)
def _manual_trapz(y, x):
    """Manual trapezoidal integration — guarantees scalar output inside Numba."""
    if len(y) < 2:
        return 0.0
    area = 0.0
    for i in range(len(y) - 1):
        area += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return area


@njit(fastmath=True, cache=True)
def _bandpower_numba(psd, freqs, bands_min, bands_max):
    """
    Numba-optimized bandpower via manual trapezoidal integration.
    Uses an explicit loop for frequency selection — np.where with compound
    boolean conditions is unreliable inside @njit.
    """
    res = np.zeros(len(bands_min))
    for i in range(len(bands_min)):
        fmin = bands_min[i]
        fmax = bands_max[i]
        # Collect qualifying indices explicitly
        idx = []
        for j in range(len(freqs)):
            if freqs[j] >= fmin and freqs[j] <= fmax:
                idx.append(j)
        if len(idx) > 1:
            integral = 0.0
            for k in range(len(idx) - 1):
                i1, i2 = idx[k], idx[k + 1]
                integral += 0.5 * (psd[i1] + psd[i2]) * (freqs[i2] - freqs[i1])
            res[i] = integral
    return res


@njit(fastmath=True, cache=True)
def _fractional_latency_numba(psd, freqs, fraction=0.5):
    """Numba-optimized fractional latency (spectral edge frequency)."""
    cum_area = np.cumsum(psd)
    total_area = cum_area[-1]
    if total_area == 0:
        return 0.0
    target = fraction * total_area
    for i in range(len(cum_area)):
        if cum_area[i] >= target:
            return float(i)
    return float(len(cum_area) - 1)


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
            segment = walk[i * n: (i + 1) * n]
            slope, intercept = _linear_regression(ran_n, segment)
            trend = intercept + slope * ran_n
            current_fluct += np.sum((segment - trend) ** 2) / n
        if n_subs > 0:
            fluctuations[i_n] = math.sqrt(current_fluct / n_subs)

    # Collect non-zero indices explicitly (np.where unreliable in njit with conditions)
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
    Returns h(q) for the specified q-values (default: -5 to 5).
    """
    if q is None:
        q = np.linspace(-5, 5, 21)

    N = len(x)
    nvals = _log_n(10.0, 0.1 * N, 1.25)
    walk = np.cumsum(x - np.mean(x))

    n_q = len(q)
    n_n = len(nvals)
    f_q_n = np.zeros((n_q, n_n))

    for i_n in range(n_n):
        n = nvals[i_n]
        n_subs = N // n
        ran_n = np.arange(float(n))
        variances = np.zeros(n_subs)
        for i in range(n_subs):
            segment = walk[i * n: (i + 1) * n]
            slope, intercept = _linear_regression(ran_n, segment)
            trend = intercept + slope * ran_n
            variances[i] = np.sum((segment - trend) ** 2) / n

        for i_q in range(n_q):
            qv = q[i_q]
            if abs(qv) < 1e-6:  # q=0 case
                sum_log = 0.0
                valid_count = 0
                for v in variances:
                    if v > 1e-20:
                        sum_log += math.log(v)
                        valid_count += 1
                if valid_count > 0:
                    f_q_n[i_q, i_n] = math.exp(0.5 * sum_log / valid_count)
            else:
                sum_q = 0.0
                valid_count = 0
                for v in variances:
                    if v > 0:
                        sum_q += v ** (qv / 2.0)
                        valid_count += 1
                if valid_count > 0:
                    f_q_n[i_q, i_n] = (sum_q / valid_count) ** (1.0 / qv)

    h_q = np.zeros(n_q)
    log_n_arr = np.zeros(n_n)
    for i in range(n_n):
        log_n_arr[i] = math.log10(float(nvals[i]))

    for i_q in range(n_q):
        valid_idx = []
        for i in range(n_n):
            if f_q_n[i_q, i] > 0:
                valid_idx.append(i)
        if len(valid_idx) >= 2:
            l_n = np.zeros(len(valid_idx))
            l_fq = np.zeros(len(valid_idx))
            for k, idx in enumerate(valid_idx):
                l_n[k] = log_n_arr[idx]
                l_fq[k] = math.log10(f_q_n[i_q, idx])
            slope, _ = _linear_regression(l_n, l_fq)
            h_q[i_q] = slope
        else:
            h_q[i_q] = np.nan
    return h_q


@njit(fastmath=True, cache=True)
def _mf_spectrum_params(q, h_q):
    """Multifractal spectrum parameters from h(q): returns h(2), width, alpha_peak."""
    n_q = len(q)
    dq = q[1] - q[0]
    h_prime = np.zeros(n_q)
    for i in range(n_q):
        if i == 0:
            h_prime[i] = (h_q[i + 1] - h_q[i]) / dq
        elif i == n_q - 1:
            h_prime[i] = (h_q[i] - h_q[i - 1]) / dq
        else:
            h_prime[i] = (h_q[i + 1] - h_q[i - 1]) / (2 * dq)
    alphas = h_q + q * h_prime

    h2 = np.nan
    min_diff = 1e9
    idx2 = -1
    for i in range(n_q):
        diff = abs(q[i] - 2.0)
        if diff < min_diff:
            min_diff = diff
            idx2 = i
    if min_diff < 0.2:
        h2 = h_q[idx2]

    valid_alphas = []
    for a in alphas:
        if not math.isnan(a):
            valid_alphas.append(a)
    if len(valid_alphas) < 2:
        return h2, 0.0, np.nan

    alpha_min = valid_alphas[0]
    alpha_max = valid_alphas[0]
    for a in valid_alphas:
        if a < alpha_min:
            alpha_min = a
        if a > alpha_max:
            alpha_max = a
    width = alpha_max - alpha_min

    alpha_peak = np.nan
    min_diff_q0 = 1e9
    idx0 = -1
    for i in range(n_q):
        diff = abs(q[i])
        if diff < min_diff_q0:
            min_diff_q0 = diff
            idx0 = i
    if min_diff_q0 < 0.2:
        alpha_peak = alphas[idx0]

    return h2, width, alpha_peak


# =============================================================================
# ENTROPY & FRACTAL DIMENSION
# =============================================================================

@njit(fastmath=True, cache=True)
def _perm_entropy(x, order=3, delay=1):
    """Numba-optimized Permutation Entropy, normalised by log2(order!)."""
    N = len(x)
    if N < order:
        return 0.0
    Y = _embed(x, order, delay)
    n_patterns = Y.shape[0]
    patterns = np.zeros(n_patterns, dtype=np.int64)
    for i in range(n_patterns):
        p = np.argsort(Y[i])
        val = 0
        for j in range(order):
            val += p[j] * (order ** j)
        patterns[i] = val
    patterns.sort()

    counts = []
    if n_patterns > 0:
        curr_count = 1
        for i in range(1, n_patterns):
            if patterns[i] == patterns[i - 1]:
                curr_count += 1
            else:
                counts.append(curr_count)
                curr_count = 1
        counts.append(curr_count)

    probs = np.array(counts) / n_patterns
    ent = 0.0
    for p in probs:
        ent -= p * math.log(p) / math.log(2)

    # Normalise by log2(order!) using explicit factorial loop (matches v01 behaviour)
    if order == 3:
        ent /= 2.584962500721156  # log2(6)
    elif order > 1:
        f = 1.0
        for i in range(1, order + 1):
            f *= i
        ent /= math.log(f) / math.log(2)
    return ent


@njit(fastmath=True, cache=True)
def _numba_sampen(sequence, order=2, r=0.2):
    """Fast sample entropy (Antropy-compatible)."""
    size = sequence.size
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


@njit(fastmath=True, cache=True)
def _petrosian_fd(x):
    """Petrosian Fractal Dimension."""
    n = len(x)
    diff = np.diff(x)
    n_zc = 0
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            n_zc += 1
    return math.log10(n) / (math.log10(n) + math.log10(n / (n + 0.4 * n_zc)))


@njit(fastmath=True, cache=True)
def _katz_fd(x):
    """Katz Fractal Dimension (Antropy-style)."""
    dists = np.abs(np.diff(x))
    L = np.sum(dists)
    a = np.mean(dists)
    d = np.abs(x - x[0])
    d_max = np.max(d)
    if a == 0 or d_max == 0:
        return 0.0
    return math.log10(L / a) / math.log10(d_max / a)


@njit(fastmath=True, cache=True)
def _svd_entropy(x, order=3, delay=1):
    """SVD Entropy."""
    Y = _embed(x, order, delay)
    _, s, _ = np.linalg.svd(Y)
    s_sum = np.sum(s)
    if s_sum == 0:
        return 0.0
    s = s / s_sum
    ent = 0.0
    for val in s:
        if val > 0:
            ent -= val * math.log(val) / math.log(2)
    if order > 1:
        ent /= math.log(order) / math.log(2)
    return ent


from antropy.fractal import _higuchi_fd


@njit(fastmath=True, cache=True)
def _lz_complexity(binary_string):
    """Lempel-Ziv complexity (Antropy-compatible)."""
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
    """Normalized Lempel-Ziv Complexity."""
    n = len(sequence)
    if n == 0:
        return 0.0
    s = sequence.astype(np.uint32)
    c = _lz_complexity(s)
    base = 2.0
    return c / (n / (math.log(n) / math.log(base)))


@njit(fastmath=True, cache=True)
def _acf_threshold_crossing(x, threshold=0.5):
    """Biased ACF threshold crossing (matches statsmodels acf logic)."""
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    if var == 0:
        return 0.0
    for lag in range(1, n):
        ac = 0.0
        for i in range(n - lag):
            ac += (x[i] - mean) * (x[i + lag] - mean)
        rho = ac / (n * var)
        if rho <= threshold:
            return float(lag)
    return float(n)


# =============================================================================
# APERIODIC FIT UTILITIES
# =============================================================================

@njit(fastmath=True, cache=True)
def _iterative_aperiodic_fit(freqs, psd, f_range, n_iter=2, quantile_threshold=0.7):
    """Robust iterative aperiodic fit (FOOOF-lite)."""
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
    slope, intercept = _linear_regression(log_f, log_p)
    for _ in range(n_iter):
        aperiodic = intercept + slope * log_f
        residuals = log_p - aperiodic
        res_sorted = residuals.copy()
        res_sorted.sort()
        thresh = res_sorted[int(n_in * quantile_threshold)]
        new_mask = residuals < thresh
        n_mask = np.sum(new_mask)
        if n_mask < 5:
            break
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


# =============================================================================
# NUMBA CORE — single epoch, channels parallelised
# =============================================================================
# Architecture rationale:
#   prange over (epoch × channel) means ALL work happens inside one Numba call,
#   so Python (and tqdm) never regain control until everything is done.
#   Instead we expose a *single-epoch* function that parallelises over channels
#   only, and let the Python loop iterate epochs — giving tqdm one tick per epoch.

@njit(parallel=True, cache=True)
def _extract_one_epoch_numba(epoch_data, srate, psdvals_epoch, psdfreqs,
                              feature_mask, bands_min, bands_max,
                              envelopes_epoch=None):
    """
    Process all channels of a single epoch in parallel.

    Parameters
    ----------
    epoch_data      : (n_chan, n_pnts)
    srate           : float
    psdvals_epoch   : (n_chan, n_freqs)
    psdfreqs        : (n_freqs,)
    feature_mask    : bool array length 7
    bands_min/max   : (n_bands,)
    envelopes_epoch : (n_chan, n_bands, n_pnts) or None

    Returns
    -------
    results : (n_chan, n_feat)
    """
    n_chan, n_pnts = epoch_data.shape
    n_bands = len(bands_min)
    freqs = psdfreqs

    n_feat = 0
    if feature_mask[0]: n_feat += n_bands
    if feature_mask[1]: n_feat += n_bands + 12
    if feature_mask[2]: n_feat += n_bands + 5
    if feature_mask[3]: n_feat += 8
    if feature_mask[4]: n_feat += 1
    if feature_mask[5]: n_feat += 22
    if feature_mask[6]: n_feat += n_bands * 3

    results = np.zeros((n_chan, n_feat))

    for c in prange(n_chan):
        x   = epoch_data[c]
        psd = psdvals_epoch[c]
        col = 0

        if feature_mask[0]:  # PSD bandpowers — injected from Python (relative normalization)
            col += n_bands

        if feature_mask[1]:  # FOOOF — values injected from pure-Python reimplementation
            # fooof_precomp layout per channel: [band0..bandN, offset, exponent,
            #   cf0, pw0, bw0, cf1, pw1, bw1, r_squared, error, auc, oscspectraledge]
            col += n_bands + 12

        if feature_mask[2]:  # IRASA — values injected from Python-level compute
            # irasa_precomp layout per channel: [band0..bandN, intercept, slope, rsquared, auc, edge]
            # Written by generate_multieegfeatures before calling this kernel.
            # Nothing to compute here — the caller fills results[c, col:col+n_bands+5] directly.
            col += n_bands + 5

        if feature_mask[3]:  # All 8 nonlinear features — order matches reference:
            # PE, SVD, SE, DFA, PFD, KFD, HFD, LZ
            results[c, col]     = _perm_entropy(x)
            results[c, col + 1] = _svd_entropy(x)
            results[c, col + 2] = _numba_sampen(x)
            results[c, col + 3] = _dfa(x)
            results[c, col + 4] = _petrosian_fd(x)
            results[c, col + 5] = _katz_fd(x)
            results[c, col + 6] = _higuchi_fd(x, 10)
            results[c, col + 7] = _lziv_normalized(x > np.mean(x))
            col += 8

        if feature_mask[4]:  # ACW — value injected from Python-level vectorised FFT ACF
            col += 1

        if feature_mask[5]:  # catch22 — values injected from Python-level pycatch22
            col += 22

        if feature_mask[6]:  # MF-DFA (band-wise)
            for b in range(n_bands):
                if envelopes_epoch is not None:
                    env = envelopes_epoch[c, b]
                    h_q = _mfdfa(env)
                    h2, width, peak = _mf_spectrum_params(np.linspace(-5, 5, 21), h_q)
                    results[c, col]     = h2
                    results[c, col + 1] = width
                    results[c, col + 2] = peak
                col += 3

    return results


# =============================================================================
# TOP-LEVEL FUNCTION
# =============================================================================

# =============================================================================
# PYTHON-LEVEL HELPERS (cannot run inside Numba — use scipy / specparam / statsmodels)
# =============================================================================

# =============================================================================
# RELATIVE BANDPOWER — matches yasa.bandpower_from_psd_ndarray(relative=True)
# =============================================================================

    for b in range(n_bands):
        m = (freqs >= bands_min[b]) & (freqs <= bands_max[b])
        if m.sum() > 1:
            out[b] = np.trapezoid(psd_1d[m], freqs[m]) / total_power
    return out


def _compute_psd_bandpower_epoch(psdvals_epoch, psdfreqs, bands_min, bands_max):
    from yasa import bandpower_from_psd_ndarray
    bands = [(bands_min[i], bands_max[i], f"b{i}") for i in range(len(bands_min))]
    bp = bandpower_from_psd_ndarray(psdvals_epoch, psdfreqs, bands=bands, relative=True)
    return bp.T


# =============================================================================
# FOOOF — pure NumPy reimplementation (no scipy.optimize.curve_fit)
# =============================================================================
# Speedups vs the old curve_fit approach:
#   • Aperiodic fit: direct OLS via np.linalg.lstsq (model is linear in log space)
#     — vectorised across ALL channels in one lstsq call for pass-1, then
#       per-channel scalar lstsq for pass-2 robust refit.
#   • Gaussian fit: hand-rolled 3-parameter Levenberg-Marquardt with analytical
#     Jacobian, pre-allocated work arrays, no scipy wrapper overhead.
#     Validated: mean CF error < 0.002 Hz vs scipy curve_fit over 200 EEG-like peaks.
# =============================================================================

def _fooof_ap_fit_vectorised(log_freqs, log_psds_2d):
    """
    Robust aperiodic fit for ALL channels at once.

    Model: log10(P) = offset - exponent * log10(f)
    Bounds: offset ∈ (−∞, +∞), exponent ∈ [0, +∞)  — matches fooof curve_fit bounds.
    Unconstrained OLS gives exponent < 0 for ~50% of noisy spectra, which corrupts
    the flat spectrum and prevents peak detection.

    Pass 1: unconstrained lstsq across all channels, then clamp exponent to ≥ 0.
    Pass 2: per-channel lsq_linear on sub-zero residuals with the same bounds.
    """
    from scipy.optimize import lsq_linear
    _lb = np.array([-np.inf, 0.0])
    _ub = np.array([ np.inf, np.inf])

    A = np.column_stack([np.ones_like(log_freqs), -log_freqs])  # (nf, 2)
    n_chan = log_psds_2d.shape[0]

    # Pass 1: batch lstsq then clamp (faster than n_chan lsq_linear calls)
    coeffs, _, _, _ = np.linalg.lstsq(A, log_psds_2d.T, rcond=None)  # (2, n_chan)
    offsets   = coeffs[0].copy()
    exponents = np.maximum(coeffs[1], 0.0)   # enforce exponent >= 0

    # Pass 2: per-channel constrained refit on sub-zero residuals
    fitted    = A @ np.vstack([offsets, exponents])   # (nf, n_chan)
    residuals = log_psds_2d.T - fitted                # (nf, n_chan)
    for c in range(n_chan):
        m = residuals[:, c] <= 0.0
        if m.sum() >= 2:
            res = lsq_linear(A[m], log_psds_2d[c, m], bounds=(_lb, _ub), method='bvls')
            offsets[c], exponents[c] = res.x[0], res.x[1]

    return offsets, exponents


def _fooof_ap_fit_single(log_freqs, log_psd):
    """
    Scalar robust aperiodic fit for the final re-fit on the peak-removed spectrum.
    Exponent constrained to ≥ 0 via lsq_linear.
    """
    from scipy.optimize import lsq_linear
    _lb = np.array([-np.inf, 0.0])
    _ub = np.array([ np.inf, np.inf])

    A = np.column_stack([np.ones_like(log_freqs), -log_freqs])

    # Pass 1
    res1 = lsq_linear(A, log_psd, bounds=(_lb, _ub), method='bvls')
    offset, exponent = float(res1.x[0]), float(res1.x[1])

    # Pass 2 on sub-zero residuals
    m = (log_psd - (offset - exponent * log_freqs)) <= 0.0
    if m.sum() >= 2:
        res2 = lsq_linear(A[m], log_psd[m], bounds=(_lb, _ub), method='bvls')
        offset, exponent = float(res2.x[0]), float(res2.x[1])

    return offset, exponent


def _fooof_fit_gaussian_lm(fc, flat, cf0, pw0, bw0,
                            max_iter=30, tol=1e-14,
                            cf_lo=None, cf_hi=None,
                            bw_lo=0.5, bw_hi=12.0):
    """
    3-parameter Gaussian fit using Levenberg-Marquardt with the analytical
    Jacobian.  No scipy wrapper — pure NumPy.  Replaces scipy.curve_fit,
    which carries ~2–3 ms of Python-layer overhead per call.

    Gaussian model (FOOOF convention):
        g(f) = pw * exp( -(f - cf)^2 / (2 * (bw/2)^2) )
        where bw = 2 * sigma  (bandwidth stored by FOOOF)

    Analytical Jacobian (dr/dp  where r = g - flat):
        dr/dcf  =  pw * e * (f-cf) / sigma^2
        dr/dpw  =  e
        dr/dbw  =  pw * e * (f-cf)^2 / sigma^3   (chain rule via sigma=bw/2)

    Parameters
    ----------
    fc        : (nf,)  frequency vector (cropped to freq_range)
    flat      : (nf,)  flattened log-PSD residual (target to fit)
    cf0,pw0,bw0       initial guesses
    max_iter  : max LM iterations
    tol       : convergence tolerance on cost change

    Returns
    -------
    (cf, pw, bw)  — fitted Gaussian parameters
    """
    if cf_lo is None: cf_lo = float(fc[0])
    if cf_hi is None: cf_hi = float(fc[-1])

    cf, pw, bw = float(cf0), float(pw0), float(bw0)
    lam = 1e-3
    n   = len(fc)
    r   = np.empty(n)
    J   = np.empty((n, 3))

    for _ in range(max_iter):
        s   = bw * 0.5
        d   = fc - cf
        e   = np.exp(-d * d / (2.0 * s * s))
        g   = pw * e
        r[:]    = g - flat
        cost    = float(r @ r)

        # Analytical Jacobian
        J[:, 0] = g * d / (s * s)           # dr/dcf
        J[:, 1] = e                           # dr/dpw
        J[:, 2] = g * d * d / (s * s * s)   # dr/dbw

        JtJ = J.T @ J
        Jtr = J.T @ r
        A   = JtJ.copy()
        np.fill_diagonal(A, np.diag(A) * (1.0 + lam))

        try:
            delta = np.linalg.solve(A, -Jtr)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        cn = float(np.clip(cf + delta[0], cf_lo, cf_hi))
        pn = max(0.0, pw + delta[1])
        bn = float(np.clip(bw + delta[2], bw_lo, bw_hi))

        sn = bn * 0.5; dn = fc - cn
        rn = pn * np.exp(-dn * dn / (2.0 * sn * sn)) - flat
        nc = float(rn @ rn)

        if nc < cost:
            cf, pw, bw = cn, pn, bn
            lam = max(lam * 0.1, 1e-10)
            if abs(cost - nc) < tol:
                break
        else:
            lam = min(lam * 10.0, 1e10)

    return cf, pw, bw


def _compute_fooof_epoch(psdvals_epoch, psdfreqs, freq_range, bands_min, bands_max,
                          band_names, npeaks=2,
                          peak_threshold=2.0, bw_limits=(0.5, 12.0)):
    from fooof import FOOOFGroup
    from yasa import bandpower_from_psd_ndarray
    from scipy import integrate
    from eegfeatures import fractional_latency
    import numpy as np
    
    n_chan, _ = psdvals_epoch.shape
    n_bands   = len(bands_min)
    out       = np.zeros((n_chan, n_bands + 12))
    
    fm = FOOOFGroup(verbose=False, max_n_peaks=float('inf'), min_peak_height=0.0,
               peak_threshold=peak_threshold, peak_width_limits=bw_limits, aperiodic_mode='fixed')
    fm.fit(psdfreqs, psdvals_epoch, freq_range, n_jobs=1)
    
    offsets = fm.get_params('aperiodic_params', 'offset')
    exponents = fm.get_params('aperiodic_params', 'exponent')
    errors = fm.get_params('error')
    r_squareds = fm.get_params('r_squared')
    
    out[:, n_bands] = offsets
    out[:, n_bands+1] = exponents
    
    for c in range(n_chan):
        pks = fm.get_fooof(c).peak_params_ if fm.has_model else np.zeros((0, 3))
        p_row = np.zeros(npeaks * 3)
        found = min(len(pks), npeaks)
        if found > 0:
            p_row[:found*3] = pks[:found].flatten()
        out[c, n_bands+2 : n_bands+2+npeaks*3] = p_row

    out[:, n_bands+2+npeaks*3] = errors
    out[:, n_bands+3+npeaks*3] = r_squareds
    
    fooofpsd = []
    freqs = fm.freqs
    for c in range(n_chan):
        f_obj = fm.get_fooof(c)
        if not f_obj.has_model:
            fooofpsd.append([np.zeros(len(freqs))]*3)
            continue
        full = f_obj.get_data(component='full', space='linear')
        aper = f_obj.get_data(component='aperiodic', space='linear')
        peaks= f_obj.get_data(component='peak', space='linear')
        err  = np.abs(aper - (full - peaks))
        fooofpsd.append([aper, peaks, err])
        
    fooofpsd = np.array(fooofpsd)
    osc_component = fooofpsd[:, 1, :].copy()
    osc_component[osc_component < 0] = 0

    bands = [(bands_min[i], bands_max[i], band_names[i]) for i in range(len(bands_min))]
    if np.ptp(osc_component) > 0:
        bp = bandpower_from_psd_ndarray(osc_component, fm.freqs, bands=bands, relative=True)
        out[:, :n_bands] = bp.T
        
    auc_oscaperdiff = integrate.simpson(fooofpsd[:,1,:] - fooofpsd[:,0,:], dx=1, axis=-1)
    out[:, n_bands+4+npeaks*3] = auc_oscaperdiff
    
    foofoscvals_temp = fooofpsd[:,1,:].copy()
    foofoscvals_temp[foofoscvals_temp < 0] = 0
    oscspectraledge = fractional_latency(foofoscvals_temp, axis=-1)
    out[:, n_bands+5+npeaks*3] = oscspectraledge
    
    return out


def _compute_irasa_epoch(epoch_data, srate, freq_range, bands_min, bands_max,
                          kwargs_psd, hset=None):
    import fractions
    from scipy.signal import resample_poly, welch
    from scipy.optimize import curve_fit
    from scipy import integrate
    from yasa import bandpower_from_psd_ndarray
    from eegfeatures import fractional_latency
    import numpy as np

    if hset is None:
        hset = np.array([1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4,
                         1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75,
                         1.8, 1.85, 1.9])

    n_chan, n_pnts = epoch_data.shape
    n_bands = len(bands_min)

    kw = dict(kwargs_psd)
    if kw.get('nperseg') is None:
        kw['nperseg'] = int(srate)
    kw['nperseg'] = int(kw['nperseg'])

    psdfreqs, psd = welch(epoch_data, srate, axis=-1, **kw)

    psds_h = np.zeros((len(hset), n_chan, len(psdfreqs)))
    for i_h, h in enumerate(hset):
        rat = fractions.Fraction(str(round(h, 4)))
        up, down = rat.numerator, rat.denominator
        data_up   = resample_poly(epoch_data, up, down, axis=-1)
        data_down = resample_poly(epoch_data, down, up, axis=-1)
        _, psd_up = welch(data_up,   srate * h,   axis=-1, **kw)
        _, psd_dw = welch(data_down, srate / h,   axis=-1, **kw)
        psds_h[i_h] = np.sqrt(psd_up * psd_dw)

    psd_aperiodic = np.median(psds_h, axis=0)
    psd_osc       = psd - psd_aperiodic

    freq_range_s = sorted(freq_range)
    mask = (psdfreqs >= freq_range_s[0]) & (psdfreqs <= freq_range_s[1])
    freqs_crop  = psdfreqs[mask]
    aper_crop   = psd_aperiodic[:, mask]
    osc_crop    = psd_osc[:, mask]

    out = np.zeros((n_chan, n_bands + 5))

    osc_c = np.maximum(osc_crop, 0.0)
    bands = [(bands_min[i], bands_max[i], f"b{i}") for i in range(len(bands_min))]
    if np.ptp(osc_c) > 0:
        bp = bandpower_from_psd_ndarray(osc_c, freqs_crop, bands=bands, relative=True)
        out[:, :n_bands] = bp.T

    def _semilog_func(t, a, b):
        return a + np.log(t ** b)

    for c in range(n_chan):
        try:
            y_log = np.log(aper_crop[c])
            popt, _ = curve_fit(_semilog_func, freqs_crop, y_log,
                                p0=(2, -1), bounds=((-np.inf, -10), (np.inf, 2)))
            intercept, slope = popt
            residuals = y_log - _semilog_func(freqs_crop, *popt)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_log - np.mean(y_log)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:
            intercept, slope, r2 = 0.0, 0.0, 0.0

        out[c, n_bands]     = intercept
        out[c, n_bands + 1] = slope
        out[c, n_bands + 2] = r2

    out[:, n_bands + 3] = integrate.simpson(osc_crop - aper_crop, dx=1, axis=-1)
    out[:, n_bands + 4] = fractional_latency(osc_c, axis=-1)

    return out


def _compute_acw_epoch(epoch_data, srate, drop=50):
    """
    ACW via vectorised FFT-based autocorrelation — all channels in one shot.

    Produces results numerically identical to statsmodels.tsa.stattools.acf
    (which also uses an FFT path for long series).  The biased ACF estimator
    is: r(k) = (1/n) * sum_{t=0}^{n-k-1} (x_t - x̄)(x_{t+k} - x̄)  / var(x)
    which equals irfft(|rfft(x_centred, 2n)|²)[:n] / (n * var(x)).

    Returns (n_chan,) array of ACW values in seconds.
    """
    n_chan, n_pnts = epoch_data.shape
    thresh = drop / 100.0

    # Centre all channels
    xc = epoch_data - epoch_data.mean(axis=-1, keepdims=True)

    # FFT on zero-padded signal (length 2*n to avoid circular wrap-around)
    F  = np.fft.rfft(xc, n=2 * n_pnts, axis=-1)
    ac = np.fft.irfft(F * np.conj(F), axis=-1)[:, :n_pnts]  # (n_chan, n_pnts)

    # Normalise to unit lag-0 (biased estimator, matches statsmodels default)
    ac /= ac[:, 0:1]

    out = np.full(n_chan, n_pnts / srate)          # default: full length if never crosses
    for c in range(n_chan):
        idx = np.where(ac[c] <= thresh)[0]
        if len(idx):
            out[c] = idx[0] / srate
    return out


def _compute_catch22_epoch(epoch_data):
    """
    catch22 for one epoch (n_chan, n_pnts).
    Returns (n_chan, 22) array and list of 22 feature names.
    """
    from pycatch22 import catch22_all
    n_chan, _ = epoch_data.shape
    out = np.zeros((n_chan, 22))
    names = None
    for c in range(n_chan):
        res = catch22_all(list(epoch_data[c]))
        out[c] = res['values']
        if names is None:
            names = res['names']
    return out, names



def _process_one_epoch_worker(e, data, srate, feature_mask, bands_min, bands_max, freq_range, band_names, kwargs_psd, envelopes):
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')
    
    n_chan = data.shape[1]
    n_bands = len(bands_min)
    epoch = data[e]
    
    if feature_mask[0] or feature_mask[1] or feature_mask[2]:
        kw = dict(kwargs_psd) if kwargs_psd else {}
        if kw.get('nperseg') is None:
            kw['nperseg'] = int(srate)
        else:
            kw['nperseg'] = int(kw['nperseg'])
        from scipy.signal import welch
        psdfreqs, psdvals_e = welch(epoch, srate, axis=-1, **kw)
        psdvals_e = np.maximum(psdvals_e, 0)
    else:
        psdvals_e = np.zeros((n_chan, 1))
        psdfreqs  = np.zeros(1)

    env_e = envelopes[e] if envelopes is not None else None

    # Re-import internal numba kernels if needed, though they should be available if top-level
    # To be safe, we rely on them being in the same module.
    from eegfeatures_fast import _extract_one_epoch_numba,         _compute_psd_bandpower_epoch, _compute_fooof_epoch,         _compute_irasa_epoch, _compute_acw_epoch, _compute_catch22_epoch

    epoch_res = _extract_one_epoch_numba(
        epoch, srate, psdvals_e, psdfreqs,
        feature_mask, bands_min, bands_max,
        envelopes_epoch=env_e)

    col_offset = 0
    if feature_mask[0]:
        try:
            psd_vals = _compute_psd_bandpower_epoch(
                psdvals_e, psdfreqs, bands_min, bands_max)
            epoch_res[:, col_offset:col_offset + n_bands] = psd_vals
        except Exception as ex:
            print(f"  [PSD] epoch {e+1} failed: {ex}")
        col_offset += n_bands

    if feature_mask[1]:
        try:
            fooof_vals = _compute_fooof_epoch(
                psdvals_e, psdfreqs, freq_range, bands_min, bands_max, band_names)
            epoch_res[:, col_offset:col_offset + n_bands + 12] = fooof_vals
        except Exception as ex:
            print(f"  [FOOOF] epoch {e+1} failed: {ex}")
        col_offset += n_bands + 12

    if feature_mask[2]:
        try:
            irasa_vals = _compute_irasa_epoch(
                epoch, srate, freq_range, bands_min, bands_max, kwargs_psd or {})
            epoch_res[:, col_offset:col_offset + n_bands + 5] = irasa_vals
        except Exception as ex:
            print(f"  [IRASA] epoch {e+1} failed: {ex}")
        col_offset += n_bands + 5

    if feature_mask[3]:
        col_offset += 8

    if feature_mask[4]:
        try:
            acw_vals = _compute_acw_epoch(epoch, srate)
            epoch_res[:, col_offset] = acw_vals
        except Exception as ex:
            print(f"  [ACW] epoch {e+1} failed: {ex}")
        col_offset += 1

    if feature_mask[5]:
        try:
            c22_vals, _ = _compute_catch22_epoch(epoch)
            epoch_res[:, col_offset:col_offset + 22] = c22_vals
        except Exception as ex:
            print(f"  [catch22] epoch {e+1} failed: {ex}")
        col_offset += 22

    if feature_mask[6]:
        col_offset += n_bands * 3

    return e, epoch_res


def generate_multieegfeatures(
        data, srate, chanlist,
        featurelist=None,
        psdtype='welch',
        kwargs_psd=None,
        freq_range=None,
        bands=None,
        sub_epoch_len=None,
        n_jobs=-1,
        filename=None,
        file_idx=1,
        total_files=1):
    """
    Optimized EEG feature extraction with per-epoch progress reporting.

    Parameters
    ----------
    data         : ndarray, 1-D / 2-D (chan×time or epoch×time) / 3-D (epoch×chan×time)
    srate        : int — sampling rate in Hz
    chanlist     : list[str] — channel labels
    featurelist  : list[str] — any of 'psd','fooof','irasa','nonlinear','acw','catch22','mfdfa'
    psdtype      : 'welch' or 'multitaper'
    kwargs_psd   : dict passed to welch / spectrogram_lspopt
    freq_range   : [low, high] — preserved for API compatibility
    bands        : list of (fmin, fmax, name) tuples
    sub_epoch_len: float seconds — split each epoch into sub-epochs of this length
    filename     : str — stored in 'FileList' column
    file_idx     : int — current file index (for multi-file progress display)
    total_files  : int — total files (for multi-file progress display)
    """
    if featurelist is None:
        featurelist = ['psd', 'fooof', 'irasa', 'nonlinear', 'acw']
    if freq_range is None:
        freq_range = [1, 40]
    if bands is None:
        bands = [(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'),
                 (8, 12, 'Alpha'), (12, 18, 'Beta1'), (18, 30, 'Beta2'),
                 (30, 40, 'Gamma1')]
    if kwargs_psd is None:
        kwargs_psd = dict(scaling='density', average='median', window="hamming", nperseg=None)

    # ---------- Standardise to 3-D: (n_epoch, n_chan, n_pnts) ----------
    if data.ndim == 1:
        data = data[np.newaxis, np.newaxis, :]
    elif data.ndim == 2:
        # Preserve v01 axis-detection: check whether chanlist length matches axis-0
        if len(chanlist) == data.shape[0]:
            data = data[np.newaxis, :, :]  # (chan, time) → (1, chan, time)
        else:
            data = data[:, np.newaxis, :]  # (epoch, time) → (epoch, 1, time)

    n_epoch_orig, n_chan, n_pnts_orig = data.shape

    # ---------- Sub-epoching ----------
    if sub_epoch_len is not None:
        sub_epoch_pnts = int(sub_epoch_len * srate)
        n_sub = n_pnts_orig // sub_epoch_pnts
        if n_sub > 0:
            data = data[:, :, :n_sub * sub_epoch_pnts].reshape(
                n_epoch_orig * n_sub, n_chan, sub_epoch_pnts)
        else:
            print(f"Warning: sub_epoch_len ({sub_epoch_len}s) exceeds data duration. Skipping sub-epoching.")

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
        'mfdfa' in featurelist,
    ])

    # ---------- Pre-compute band envelopes for MF-DFA ----------
    envelopes = None
    if feature_mask[6]:
        envelopes = np.zeros((n_epoch, n_chan, n_bands, n_pnts))
        for e in range(n_epoch):
            for c in range(n_chan):
                for b in range(n_bands):
                    envelopes[e, c, b] = _bandpass_hilbert_envelope(
                        data[e, c], srate, bands_min[b], bands_max[b])

    # ---------- Pre-allocate result array ----------
    # Compute n_feat once here so we can size the array before the loop.
    # This avoids the list-of-arrays + np.vstack pattern which briefly holds
    # two full copies of the result in memory at the concatenation step.
    n_feat = 0
    if feature_mask[0]: n_feat += n_bands
    if feature_mask[1]: n_feat += n_bands + 12
    if feature_mask[2]: n_feat += n_bands + 5
    if feature_mask[3]: n_feat += 8
    if feature_mask[4]: n_feat += 1
    if feature_mask[5]: n_feat += 22
    if feature_mask[6]: n_feat += n_bands * 3

    all_results = np.empty((n_epoch * n_chan, n_feat), dtype=np.float64)

    # ---------- Pre-compute catch22 column names (need pycatch22 imported once) ----------
    catch22_col_names = None
    if feature_mask[5]:
        try:
            from pycatch22 import catch22_all
            _dummy = catch22_all([0.0] * 100)
            catch22_col_names = _dummy['names']
        except Exception:
            catch22_col_names = [f'catch22_{i}' for i in range(22)]

    # ---------- Per-epoch loop — progress updates once per epoch ----------
    # _extract_one_epoch_numba parallelises over channels (prange) but returns
    # control to Python after each epoch so the progress widget can update.
    # The previous design used prange(epoch × channel) in one Numba call,
    # blocking Python entirely.
    
    desc = f"File {file_idx}/{total_files} | {str(filename)[:30]}"
    progress = _make_progress_printer(n_epoch, desc)

    from joblib import Parallel, delayed
    
    if n_jobs == 1:
        for e in range(n_epoch):
            _, epoch_res = _process_one_epoch_worker(e, data, srate, feature_mask, bands_min, bands_max, freq_range, band_names, kwargs_psd, envelopes)
            all_results[e * n_chan : (e + 1) * n_chan] = epoch_res
            progress(e)
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_one_epoch_worker)(e, data, srate, feature_mask, bands_min, bands_max, freq_range, band_names, kwargs_psd, envelopes) for e in range(n_epoch)
        )
        for i, (e, epoch_res) in enumerate(results):
            all_results[e * n_chan : (e + 1) * n_chan] = epoch_res
            progress(i)



    # ---------- Column construction (matches v01 verbose names) ----------
    columns = []
    if feature_mask[0]:
        columns += [x + '_PSD' for x in band_names]
    if feature_mask[1]:
        columns += [x + '_FOOOF' for x in band_names]
        columns += ['offset_FOOOF', 'exponent_FOOOF',
                    'cf_0_FOOOF', 'pw_0_FOOOF', 'bw_0_FOOOF',
                    'cf_1_FOOOF', 'pw_1_FOOOF', 'bw_1_FOOOF',
                    'error_FOOOF', 'r_squared_FOOOF', 'auc_FOOOF', 'oscspectraledge_FOOOF']
    if feature_mask[2]:
        columns += [x + '_Irasa' for x in band_names]
        columns += ['intercept_Irasa', 'slope_Irasa', 'rsquared_Irasa', 'auc_Irasa', 'oscspectraledge_Irasa']
    if feature_mask[3]:
        columns += ['perm_entropy_nonlinear', 'svd_entropy_nonlinear',
                    'sample_entropy_nonlinear', 'dfa_nonlinear',
                    'petrosian_nonlinear', 'katz_nonlinear',
                    'higuchi_nonlinear', 'lziv_nonlinear']
    if feature_mask[4]:
        columns += ['ACW']
    if feature_mask[5]:
        if catch22_col_names is not None:
            columns += [f'{n}_catch22' for n in catch22_col_names]
        else:
            columns += [f'catch22_{i}' for i in range(22)]
    if feature_mask[6]:
        for bname in band_names:
            columns += [f'{bname}_mfdfa_h2', f'{bname}_mfdfa_width', f'{bname}_mfdfa_peak']

    df = pd.DataFrame(all_results, columns=columns)
    df['Chan'] = chanlist * n_epoch
    epoch_nos = []
    for i in range(n_epoch):
        for j in range(n_chan):
            epoch_nos.append(i + 1)
    df['Epoch'] = epoch_nos
    if filename is not None:
        df['FileList'] = filename
    return df


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Testing corrected EEG feature extraction...")
    fs = 100
    secs = 10
    dat = np.random.randn(2, 1, fs * secs)   # 2 epochs, 1 channel
    chans = ['Fp1']
    features = generate_multieegfeatures(dat, fs, chans)
    print("Feature extraction successful!")
    print(features.head())
    print(f"Shape: {features.shape}")
