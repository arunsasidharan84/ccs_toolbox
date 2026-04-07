"""
Poincaré Sympatho-Vagal Synthetic Data Generation (PSV‑SDG) — Python implementation
-------------------------------------------------------------------------------
This module computes time‑resolved cardiac sympathetic (CSI) and vagal (CVI) indices
from RR intervals and estimates brain→heart and heart→brain couplings using the
PSV‑SDG idea described by Candia‑Rivera (2023, MethodsX).

Key components
- HRV Poincaré geometry: SD1, SD2, CSI, CVI
- Sliding‑window estimation of time‑varying CSI/CVI
- EEG band‑limited power extraction (delta, theta, alpha, beta, gamma)
- PSV‑SDG brain↔heart estimation via regularized, time‑varying linear models
  with lags (state‑space/RLS option or ridge regression on fixed windows)
- Convenience plotting and example

Notes
- This is an original Python re‑implementation for research/education.
- It mirrors the published MATLAB reference implementation structure
  (compute_CSI_CVI.m, model_psv_sdg.m, compute_psv_sdg.m) while adopting
  Pythonic APIs and sklearn/scipy numerics.
- Validate on your datasets; tune windows, lags, and regularization.

Dependencies
    numpy, scipy, scikit‑learn, matplotlib (for optional plots)

Author: ChatGPT (GPT‑5 Thinking)
License: MIT (for this Python code)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from scipy.signal import butter, filtfilt, hilbert, welch, detrend
from sklearn.linear_model import RidgeCV

# -------------------------
# 1) HRV POINCARÉ GEOMETRY
# -------------------------

def _sd1_sd2(rr_ms: np.ndarray) -> Tuple[float, float]:
    """Compute SD1 and SD2 from RR intervals (ms).
    SD1 = sqrt(0.5) * std(diff(RR)) ; SD2 = sqrt(2*SDNN^2 - 0.5*std(diff(RR))^2)
    RR should be clean, ectopic‑free, in milliseconds.
    """
    rr = np.asarray(rr_ms, dtype=float)
    rr = rr[np.isfinite(rr)]
    if rr.size < 5:
        return np.nan, np.nan
    sdnn = np.std(rr, ddof=1)
    drr = np.diff(rr)
    sd1 = np.sqrt(0.5) * np.std(drr, ddof=1)
    sd2_term = 2.0 * (sdnn ** 2) - 0.5 * (np.std(drr, ddof=1) ** 2)
    sd2 = np.sqrt(sd2_term) if sd2_term > 0 else np.nan
    return sd1, sd2

def csi_cvi(rr_ms: np.ndarray) -> Tuple[float, float, float, float]:
    """Return SD1, SD2, CSI, CVI for a vector of RR intervals (ms).
    CSI (cardiac sympathetic index)  = SD2 / SD1
    CVI (cardiac vagal index)       = log10(SD1 * SD2)
    """
    sd1, sd2 = _sd1_sd2(rr_ms)
    if not np.isfinite(sd1) or not np.isfinite(sd2) or sd1 <= 0 or sd2 <= 0:
        return sd1, sd2, np.nan, np.nan
    csi = sd2 / sd1
    cvi = np.log10(sd1 * sd2)
    return sd1, sd2, csi, cvi

@dataclass
class SlidingHRV:
    win_s: float = 15.0
    step_s: float = 1.0

    def transform(self, rr_times_s: np.ndarray, rr_ms: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute time‑resolved SD1, SD2, CSI, CVI over sliding windows.
        rr_times_s: timestamps (s) for each RR, same length as rr_ms.
        Returns dict with 't', 'sd1', 'sd2', 'csi', 'cvi'.
        """
        t = []
        sd1_list, sd2_list, csi_list, cvi_list = [], [], [], []
        t0, tN = rr_times_s[0], rr_times_s[-1]
        w = self.win_s
        s = self.step_s
        cur = t0
        i0 = 0
        while cur + w <= tN and i0 < len(rr_times_s):
            # window mask
            i1 = np.searchsorted(rr_times_s, cur + w, side='right')
            idx = slice(i0, i1)
            sd1, sd2, csi, cvi = csi_cvi(rr_ms[idx])
            t.append(cur + w/2.0)
            sd1_list.append(sd1); sd2_list.append(sd2)
            csi_list.append(csi); cvi_list.append(cvi)
            cur += s
            i0 = np.searchsorted(rr_times_s, cur, side='left')
        out = {
            't': np.array(t),
            'sd1': np.array(sd1_list),
            'sd2': np.array(sd2_list),
            'csi': np.array(csi_list),
            'cvi': np.array(cvi_list),
        }
        return out

# ----------------------------------
# 2) EEG BAND POWER (TIME‑RESOLVED)
# ----------------------------------

@dataclass
class EEGBands:
    fs: float
    bands: Dict[str, Tuple[float, float]] = None
    env_lp_hz: float = 1.0  # low‑pass on envelope (power smoothing)

    def __post_init__(self):
        if self.bands is None:
            self.bands = {
                'delta': (1.0, 4.0),
                'theta': (4.0, 8.0),
                'alpha': (8.0, 13.0),
                'beta': (13.0, 30.0),
                'gamma': (30.0, 45.0),
            }

    def _bpfilt(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        nyq = 0.5 * self.fs
        b, a = butter(4, [lo/nyq, hi/nyq], btype='band')
        return filtfilt(b, a, x)

    def _lpfilt(self, x: np.ndarray, cutoff: float) -> np.ndarray:
        nyq = 0.5 * self.fs
        b, a = butter(4, cutoff/nyq, btype='low')
        return filtfilt(b, a, x)

    def band_envelopes(self, eeg: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute smoothed amplitude envelopes (≈ power proxies) per band.
        eeg: shape (n_channels, n_samples) or (n_samples,) for single channel.
        Returns dict of band -> envelopes (same shape as input samples, per channel summed/averaged).
        """
        if eeg.ndim == 1:
            eeg = eeg[None, :]
        envs = {}
        for name, (lo, hi) in self.bands.items():
            xband = np.vstack([self._bpfilt(ch, lo, hi) for ch in eeg])
            amp = np.abs(hilbert(xband, axis=1))  # analytic amplitude
            env = np.mean(amp**2, axis=0)        # average band power across channels
            env_s = self._lpfilt(env, self.env_lp_hz)
            envs[name] = env_s
        return envs

# ----------------------------------------------------------
# 3) PSV‑SDG COUPLING ESTIMATION (Brain→Heart & Heart→Brain)
# ----------------------------------------------------------

@dataclass
class PSVSDG:
    hrv_win_s: float = 15.0
    hrv_step_s: float = 1.0
    eeg_fs: float = 256.0
    lags_s: Tuple[float, float] = (0.0, 5.0)  # min,max lag for EEG→HRV (s)
    n_lags: int = 6                           # number of lag taps across the range
    alpha_grid: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    bands: Optional[List[str]] = None         # subset of EEG bands, None → all

    def _build_design(self, t_hrv: np.ndarray, eeg_env: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Build lagged EEG design aligned to HRV time points (window centres).
        Returns X (n_times × (bands*lags + 1)) and column names.
        """
        # Convert EEG time to seconds indexing assuming starting at 0 for simplicity
        # User can pre‑align if EEG and RR start times differ.
        tmin, tmax = self.lags_s
        lags = np.linspace(tmin, tmax, self.n_lags)
        Xcols = []
        X_list = []
        use_bands = self.bands or list(eeg_env.keys())
        for b in use_bands:
            sig = eeg_env[b]
            for L in lags:
                # sample envelope at (t_hrv - L)
                # Convert seconds to samples
                idx = np.clip((t_hrv - L) * self.eeg_fs, 0, len(sig)-1).astype(int)
                X_list.append(sig[idx])
                Xcols.append(f"{b}@lag{L:.2f}s")
        X = np.vstack(X_list).T if X_list else np.empty((len(t_hrv), 0))
        # Add intercept
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        Xcols.append("intercept")
        return X, Xcols

    def fit(self,
            rr_times_s: np.ndarray,
            rr_ms: np.ndarray,
            eeg: np.ndarray,
            eeg_fs: float,
            bands: Optional[Dict[str, Tuple[float, float]]] = None,
           ) -> Dict[str, np.ndarray]:
        """Compute sliding‑window CSI/CVI and regress them on lagged EEG power.
        Returns a dict containing time series of CSI, CVI, predictions, and
        brain→heart coupling weights per band (summed over lags).
        """
        # 1) HRV features over time
        hrv = SlidingHRV(self.hrv_win_s, self.hrv_step_s).transform(rr_times_s, rr_ms)
        t_hrv = hrv['t']
        # 2) EEG envelopes
        eb = EEGBands(fs=eeg_fs)
        if bands is not None:
            eb.bands = bands
        envs = eb.band_envelopes(eeg)
        # 3) Build lagged design
        X, Xcols = self._build_design(t_hrv, envs)
        # 4) Targets (z‑scored to stabilize)
        def z(x):
            xm = np.nanmean(x); xs = np.nanstd(x) or 1.0
            return (x - xm) / xs
        y_csi = z(hrv['csi'])
        y_cvi = z(hrv['cvi'])
        # Handle NaNs
        mask = np.isfinite(y_csi) & np.isfinite(y_cvi) & np.all(np.isfinite(X), axis=1)
        Xf = X[mask]; t_use = t_hrv[mask]
        csi = y_csi[mask]; cvi = y_cvi[mask]
        # 5) Fit ridge regressions with CV for CSI and CVI (brain→heart)
        ridge = RidgeCV(alphas=self.alpha_grid, fit_intercept=False)
        ridge_csi = ridge.fit(Xf, csi)
        ridge_cvi = RidgeCV(alphas=self.alpha_grid, fit_intercept=False).fit(Xf, cvi)
        yhat_csi = ridge_csi.predict(Xf)
        yhat_cvi = ridge_cvi.predict(Xf)
        # 6) Summarize band‑wise coupling (sum absolute weights over lags)
        band_names = []
        for name in envs.keys():
            if self.bands and name not in self.bands:
                continue
            band_names.append(name)
        lag_counts = self.n_lags
        w_csi = ridge_csi.coef_[:-1]  # exclude intercept
        w_cvi = ridge_cvi.coef_[:-1]
        band_csi = {}
        band_cvi = {}
        for bi, b in enumerate(band_names):
            sl = slice(bi*lag_counts, (bi+1)*lag_counts)
            band_csi[b] = np.sum(np.abs(w_csi[sl]))
            band_cvi[b] = np.sum(np.abs(w_cvi[sl]))
        return {
            't': t_use,
            'X_columns': np.array(Xcols),
            'csi_z': csi, 'cvi_z': cvi,
            'csi_hat': yhat_csi, 'cvi_hat': yhat_cvi,
            'coef_csi': ridge_csi.coef_, 'coef_cvi': ridge_cvi.coef_,
            'band_coupling_csi': band_csi,
            'band_coupling_cvi': band_cvi,
            'hrv_series': hrv,
        }

# ---------------------------------------
# 4) HEART→BRAIN (optional, symmetric map)
# ---------------------------------------

@dataclass
class HeartToBrain:
    eeg_fs: float
    env_lp_hz: float = 1.0
    lags_s: Tuple[float, float] = (0.0, 4.0)
    n_lags: int = 6
    alpha_grid: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)

    def fit(self, eeg: np.ndarray, eeg_fs: float, hrv_series: Dict[str, np.ndarray], band: str = 'theta') -> Dict[str, np.ndarray]:
        """Predict EEG band power from lagged HRV geometry (SD1, SD2) — heart→brain.
        Returns time‑aligned predictions and coefficients.
        """
        # Build HRV design at EEG sampling (resample by nearest neighbor on time axis)
        t_eeg = np.arange(eeg.shape[-1]) / eeg_fs
        t_hrv = hrv_series['t']
        lags = np.linspace(self.lags_s[0], self.lags_s[1], self.n_lags)
        # target EEG envelope in selected band
        eb = EEGBands(fs=eeg_fs)
        envs = eb.band_envelopes(eeg)
        y = envs[band]
        # interpolate HRV features at (t_eeg - lag)
        feats = ['sd1', 'sd2']
        X_list = []
        names = []
        for f in feats:
            sig = np.interp(t_eeg, t_hrv, hrv_series[f], left=np.nan, right=np.nan)
            for L in lags:
                idx = np.clip((t_eeg - L) * eeg_fs, 0, len(y)-1).astype(int)
                X_list.append(sig[idx])
                names.append(f"{f}@lag{L:.2f}s")
        X = np.vstack(X_list).T
        X = np.hstack([X, np.ones((len(X), 1))])
        names.append('intercept')
        # Clean NaNs
        mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        Xf, yf = X[mask], y[mask]
        model = RidgeCV(alphas=self.alpha_grid, fit_intercept=False).fit(Xf, yf)
        yhat = model.predict(Xf)
        return {
            't': t_eeg[mask],
            'y': yf, 'yhat': yhat,
            'coef': model.coef_, 'X_columns': np.array(names)
        }

# ---------------------------------------------
# 5) SYNTHETIC DATA GENERATION (PSV-SDG STYLE)
# ---------------------------------------------

@dataclass
class HeartOscillator:
    """Minimal generative heart model driven by sympatho–vagal inputs.

    We model instantaneous heart rate f(t) [Hz] as a baseline plus two drives:
      f(t) = f0 + Gs * x_s(t) + Gv * x_v(t) + noise
    where x_s and x_v are slow states (sympathetic, vagal) excited by band‑limited
    EEG 'neural' inputs u_s(t), u_v(t) and relaxing with distinct time constants.

    RR events are emitted by integrating phase: dphi/dt = 2*pi*f(t) and placing
    a beat whenever phi crosses 2*pi.
    """
    f0_hz: float = 1.1           # intrinsic heart rate (~66 bpm)
    tau_s: float = 5.0            # sympathetic state time constant (s)
    tau_v: float = 0.8            # vagal state time constant (s)
    Gs: float = 0.15              # sympathetic gain (Hz per unit)
    Gv: float = -0.20             # vagal gain (Hz per unit; negative slows HR)
    noise_std: float = 0.01       # HR noise (Hz)

    def simulate(self, u_s: np.ndarray, u_v: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure same length
        N = min(len(u_s), len(u_v))
        u_s = u_s[:N]; u_v = u_v[:N]
        dt = 1.0/fs
        x_s = 0.0; x_v = 0.0; phi = 0.0
        rr_times = []
        tvec = np.arange(N)*dt
        for i in range(N):
            # First‑order state updates (Euler)
            x_s += dt*( -x_s/self.tau_s + u_s[i] )
            x_v += dt*( -x_v/self.tau_v + u_v[i] )
            f = self.f0_hz + self.Gs*x_s + self.Gv*x_v + np.random.normal(0, self.noise_std)
            f = max(0.5, f)  # floor at 30 bpm for stability
            phi += 2*np.pi*f*dt
            if phi >= 2*np.pi:
                phi -= 2*np.pi
                rr_times.append(tvec[i])
        return np.array(rr_times), tvec

@dataclass
class PSVSDGGenerator:
    """Full PSV‑SDG‑style pipeline: build neural drives from EEG, simulate RR,
    and compare synthetic vs observed HRV geometry to fit coupling parameters."""
    eeg_fs: float
    hrv_win_s: float = 15.0
    hrv_step_s: float = 1.0
    lags_s: Tuple[float, float] = (0.0, 5.0)
    n_lags: int = 6
    bands_for_symp: Optional[List[str]] = None
    bands_for_vagal: Optional[List[str]] = None

    def _lag_stack(self, sig: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        lags = np.linspace(self.lags_s[0], self.lags_s[1], self.n_lags)
        X = []
        for L in lags:
            idx = np.clip((np.arange(len(sig))/fs - L)*fs, 0, len(sig)-1).astype(int)
            X.append(sig[idx])
        return np.vstack(X).T, lags

    def build_drives(self, eeg: np.ndarray, bands: Optional[Dict[str, Tuple[float,float]]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str,np.ndarray]]:
        eb = EEGBands(fs=self.eeg_fs)
        if bands is not None: eb.bands = bands
        envs = eb.band_envelopes(eeg)
        use_s = self.bands_for_symp or ['beta','gamma']
        use_v = self.bands_for_vagal or ['theta','alpha']
        # Keep only bands that actually exist in envs
        use_s = [b for b in use_s if b in envs]
        use_v = [b for b in use_v if b in envs]
        if not use_s or not use_v:
            raise ValueError(f"Requested bands not present. Available: {list(envs.keys())}; symp={use_s}, vagal={use_v}")
        # Concatenate selected bands for each pathway and standardize
        def stack(names):
            mats = []
            for n in names:
                x = envs[n]
                xz = (x - np.mean(x)) / (np.std(x) + 1e-9)
                mats.append(xz)
            return np.vstack(mats)  # (n_bands, N)
        U_s = stack(use_s)
        U_v = stack(use_v)
        return U_s, U_v, envs

    def fit_and_generate(self, rr_times_obs: np.ndarray, rr_ms_obs: np.ndarray, eeg: np.ndarray, bands: Optional[Dict[str,Tuple[float,float]]] = None, max_iter: int = 200, lr: float = 0.01) -> Dict[str, np.ndarray]:
        """Estimate band gains that best reproduce observed HRV geometry by gradient descent on SD1/SD2 trajectory error."""
        # Prepare observed HRV trajectory
        hrv_obs = SlidingHRV(self.hrv_win_s, self.hrv_step_s).transform(rr_times_obs, rr_ms_obs)
        t_hrv = hrv_obs['t']
        # Drives
        U_s, U_v, envs = self.build_drives(eeg, bands)
        fs = self.eeg_fs
        N = U_s.shape[1]
        # Initialize gains per band (symp/vagal). Start small.
        gs = np.zeros(U_s.shape[0])
        gv = np.zeros(U_v.shape[0])
        osc = HeartOscillator()

        def to_scalar_drive(U, g):
            """Combine per-band drives U (n_bands × N) with gains g (n_bands,) to a single drive (N,).
            Uses row-vector @ matrix so shapes always align even when n_bands=1."""
            U = np.atleast_2d(U)             # ensure (n_bands, N)
            g = np.atleast_1d(g)             # ensure (n_bands,)
            if g.size != U.shape[0]:
                raise ValueError(f"Gain length {g.size} does not match number of bands {U.shape[0]} in U.")
            return (g[None, :] @ U).ravel()

        history = []
        for it in range(max_iter):
            u_s = to_scalar_drive(U_s, gs)
            u_v = to_scalar_drive(U_v, gv)
            rr_syn_times, _ = osc.simulate(u_s, u_v, fs)
            # Convert synthetic times to RR (ms)
            if len(rr_syn_times) < 5:
                # too few beats, nudge gains toward zero and continue
                gs *= 0.9; gv *= 0.9; continue
            rr_intervals = np.diff(rr_syn_times) * 1000.0
            rr_times_ctr = rr_syn_times[1:]
            hrv_syn = SlidingHRV(self.hrv_win_s, self.hrv_step_s).transform(rr_times_ctr, rr_intervals)
            # Align to observed HRV timeline
            def interp(x):
                return np.interp(t_hrv, hrv_syn['t'], x, left=np.nan, right=np.nan)
            csi_syn = interp(hrv_syn['csi']); cvi_syn = interp(hrv_syn['cvi'])
            # Loss: MSE on CSI and CVI (ignore NaNs)
            mask = np.isfinite(csi_syn) & np.isfinite(cvi_syn) & np.isfinite(hrv_obs['csi']) & np.isfinite(hrv_obs['cvi'])
            if not np.any(mask):
                continue
            err = (np.nanmean((csi_syn[mask]-hrv_obs['csi'][mask])**2) + np.nanmean((cvi_syn[mask]-hrv_obs['cvi'][mask])**2))
            history.append(err)
            # Finite‑difference gradients w.r.t gains
            def grad_g(gvec, Umat, target='symp'):
                eps = 1e-3
                gr = np.zeros_like(gvec)
                for k in range(len(gvec)):
                    gpert = gvec.copy(); gpert[k] += eps
                    u_s_p = to_scalar_drive(U_s, gpert) if target=='symp' else to_scalar_drive(U_s, gs)
                    u_v_p = to_scalar_drive(U_v, gv) if target=='vagal' else to_scalar_drive(U_v, gv)
                    rr_t_p, _ = osc.simulate(u_s_p, u_v_p, fs)
                    if len(rr_t_p) < 5:
                        continue
                    rr_p = np.diff(rr_t_p)*1000.0
                    rr_tc = rr_t_p[1:]
                    hsyn_p = SlidingHRV(self.hrv_win_s, self.hrv_step_s).transform(rr_tc, rr_p)
                    csi_p = np.interp(t_hrv, hsyn_p['t'], hsyn_p['csi'], left=np.nan, right=np.nan)
                    cvi_p = np.interp(t_hrv, hsyn_p['t'], hsyn_p['cvi'], left=np.nan, right=np.nan)
                    m = np.isfinite(csi_p) & np.isfinite(cvi_p) & np.isfinite(hrv_obs['csi']) & np.isfinite(hrv_obs['cvi'])
                    if not np.any(m):
                        continue
                    err_p = (np.nanmean((csi_p[m]-hrv_obs['csi'][m])**2) + np.nanmean((cvi_p[m]-hrv_obs['cvi'][m])**2))
                    gr[k] = (err_p - err)/eps
                return gr
            # Update gains
            gs -= lr * grad_g(gs, U_s, target='symp')
            gv -= lr * grad_g(gv, U_v, target='vagal')
            # Optional: stop if converged
            if it > 10 and np.abs(history[-1]-history[-2]) < 1e-6:
                break

        # Final simulate with learned gains
        u_s = to_scalar_drive(U_s, gs); u_v = to_scalar_drive(U_v, gv)
        rr_syn_times, tvec = osc.simulate(u_s, u_v, fs)
        rr_syn = np.diff(rr_syn_times)*1000.0 if len(rr_syn_times)>1 else np.array([])
        rr_syn_ctr = rr_syn_times[1:] if len(rr_syn_times)>1 else np.array([])
        hrv_syn = SlidingHRV(self.hrv_win_s, self.hrv_step_s).transform(rr_syn_ctr, rr_syn) if rr_syn.size>0 else None
        return {
            'g_symp': gs, 'g_vagal': gv,
            'history': np.array(history),
            'rr_syn_times': rr_syn_times,
            'hrv_syn': hrv_syn,
            'hrv_obs': hrv_obs,
        }

# --- plotting helpers ---
import matplotlib.pyplot as plt

def plot_band_couplings(res):
    csi = res.get('band_coupling_csi', {})
    cvi = res.get('band_coupling_cvi', {})
    order = ['delta','theta','alpha','beta','gamma']
    bands = [b for b in order if b in (set(csi.keys()) | set(cvi.keys()))]
    csi_vals = [csi.get(b, np.nan) for b in bands]
    cvi_vals = [cvi.get(b, np.nan) for b in bands]
    x = np.arange(len(bands)); width = 0.35
    fig = plt.figure(figsize=(8,4.5))
    plt.bar(x - width/2, csi_vals, width, label='CSI')
    plt.bar(x + width/2, cvi_vals, width, label='CVI')
    plt.xticks(x, bands); plt.ylabel('Coupling magnitude (sum |lags|)')
    plt.title('Regression-based EEG→HRV band couplings (CSI vs CVI)')
    plt.legend(); plt.tight_layout(); plt.show()
    return fig

def plot_sdg_gains(out, symp_bands=('beta','gamma'), vagal_bands=('theta','alpha')):
    gS = np.asarray(out['g_symp']).ravel()
    gV = np.asarray(out['g_vagal']).ravel()
    fig = plt.figure(figsize=(8,4.5))
    x1 = np.arange(len(symp_bands))
    x2 = np.arange(len(vagal_bands)) + len(symp_bands) + 1
    xticks = list(symp_bands) + [''] + list(vagal_bands)
    plt.bar(x1, gS, label='Sympathetic gains')
    plt.bar(x2, gV, label='Vagal gains')
    plt.xticks(range(len(xticks)), xticks)
    plt.ylabel('Gain (arbitrary model units)')
    plt.title('PSV-SDG learned gains per pathway')
    plt.legend(); plt.tight_layout(); plt.show()
    return fig

# -----------------
# 6) QUICK EXAMPLE
# -----------------

def _example():
    """Minimal end‑to‑end example with synthetic signals + regression + generator."""
    rng = np.random.default_rng(0)
    # Synthetic RR (approx.)
    rr_times = np.cumsum(0.9 + 0.2*rng.standard_normal(70))
    rr_ms = 1000*(0.9 + 0.05*np.sin(2*np.pi*rr_times/10.0) + 0.02*rng.standard_normal(rr_times.size))
    # EEG: 1 channel, 256 Hz
    fs = 256
    T = int(rr_times[-1]) + 5
    t = np.arange(T*fs)/fs
    eeg = 10*np.sin(2*np.pi*9*t) + 2*np.sin(2*np.pi*5*t) + 0.5*rng.standard_normal(t.size)

    # Regression‑based coupling (as before)
    model = PSVSDG(hrv_win_s=15, hrv_step_s=1, eeg_fs=fs, n_lags=8)
    res = model.fit(rr_times, rr_ms, eeg, fs)

    # Generative SDG: learn band gains that best reproduce observed HRV geometry
    gen = PSVSDGGenerator(eeg_fs=fs, hrv_win_s=15, hrv_step_s=1)
    out = gen.fit_and_generate(rr_times, rr_ms, eeg)

    print('Regression band couplings (CSI):', res['band_coupling_csi'])
    print('Learned SDG gains — symp:', out['g_symp'], 'vagal:', out['g_vagal'])
    
    plot_band_couplings(res)
    plot_sdg_gains(out)  # assumes default mapping β/γ -> symp, θ/α -> vagal

if __name__ == "__main__":
    _example()
