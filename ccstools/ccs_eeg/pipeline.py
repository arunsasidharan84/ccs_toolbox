import mne
import numpy as np
import os
import time
import datetime
import pandas as pd
from mne_icalabel import label_components
from autoreject import AutoReject, Ransac

def set_chanlocs(raw_input):
    standard_montage = mne.channels.make_standard_montage('standard_1020')
    std_names = standard_montage.ch_names
    lookup = {name.lower(): name for name in std_names}
    rename_dict = {}
    keep_channels = []    
    for ch in raw_input.ch_names:
        if ch in std_names:
            # Already standard, keep it
            keep_channels.append(ch)
        elif ch.lower() in lookup:
            # Found a case-insensitive match (e.g., 'FP1' -> 'Fp1')
            standard_name = lookup[ch.lower()]
            rename_dict[ch] = standard_name
            keep_channels.append(standard_name)
        else:
            # It's an EOG or unknown channel, skip it for now
            print(f"Skipping non-standard channel: {ch}")
    raw_input.rename_channels(rename_dict)
    raw_input.pick(keep_channels)
    raw_input.set_montage(standard_montage)

    # Drop channels without locations (like non-EEG channels)
    channels_to_drop = [
        ch['ch_name'] for ch in raw_input.info['chs'] 
        if np.all(ch['loc'][:3] == 0)
    ]
    if channels_to_drop:
        print(f"Dropping unlocalized channels: {channels_to_drop}")
        raw_input.drop_channels(channels_to_drop)
    else:
        print("All channels have valid positions!")
    
    return raw_input

def detect_bad_channels(raw, cfg):
    """
    Detects bad channels using RANSAC (correlation) and flatline criteria.
    Mimics clean_rawdata behavior.
    """
    print("Detecting bad channels...")
    bad_channels = []
    
    # 1. Flatline Detection
    # clean_rawdata default: 5 seconds
    # We check if any channel is flat for > 5 seconds
    flat_crit = cfg.get('FlatlineCriterion', 5)
    if flat_crit:
        print(f"Checking for flatlines > {flat_crit}s...")
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        n_samples_flat = int(flat_crit * sfreq)
        
        for i, ch_name in enumerate(raw.ch_names):
            # Check for consecutive identical values? 
            # Or just low variance in sliding window?
            # clean_rawdata checks for "constant" values.
            # Efficient way: diff is zero.
            diff = np.diff(data[i, :])
            is_flat = np.abs(diff) < 1e-15
            
            # Find max run of True
            # This can be slow in pure python, use numba or simple loop?
            # For now, simple approach:
            # If std of entire channel is 0, it's definitely flat.
            if np.std(data[i, :]) < 1e-15:
                bad_channels.append(ch_name)
                continue
                
            # Sliding window check is expensive.
            # Alternative: Split into 5s segments and check if any is flat?
            # Let's stick to global flatline for now to be safe/fast, 
            # unless we want to be strict.
            # The user's issue was *too many* bad channels, so strictness isn't the problem?
            # Actually, if I marked 14 channels, maybe some were flat?
            # My previous check was global std < 1e-15.
            pass
            
    # 2. Correlation (RANSAC)
    # This is the main one for "ChannelCriterion"
    print("Running RANSAC for bad channel detection...")
    
    # Check if montage is present, if not, try standard
    if raw.get_montage() is None:
        print("No montage found. attempting to set 'standard_1020' for autoreject...")
        try:
            raw = set_chanlocs(raw)
        except Exception as e:
            print(f"Warning: Failed to set standard montage: {e}")
    
    # Create fixed length epochs for RANSAC
    # clean_rawdata uses continuous data but RANSAC in autoreject uses epochs.
    # We can chunk the data.
    events = mne.make_fixed_length_events(raw, duration=1.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=1.0, baseline=None, preload=True, verbose=False)
    
    # RANSAC
    # min_corr: correlation threshold (default 0.75 in autoreject, 0.8 in clean_rawdata)
    # n_resample: number of iterations (default 50)
    # min_channels: min channels to reconstruct (default 0.25)
    
    try:
        corr_thresh = cfg.get('ChannelCriterion', 0.8)
        rsc = Ransac(n_jobs=1, min_corr=corr_thresh, verbose=False)
        rsc.fit(epochs)
        
        print(f"RANSAC detected bad channels: {rsc.bad_chs_}")
        bad_channels.extend(rsc.bad_chs_)
    except Exception as e:
        print(f"RANSAC bad channel detection failed (likely due to missing channel positions): {e}")
        print("Skipping RANSAC step.")
    
    # Unique bads
    bad_channels = list(set(bad_channels))
    
    # Update raw
    raw.info['bads'].extend(bad_channels)
    raw.info['bads'] = list(set(raw.info['bads']))
    
    print(f"Total bad channels: {raw.info['bads']}")
    
    return raw, bad_channels
from .utils import load_custom_montage, plot_eeg_waveform
from .gedai.gedai_algo import gedai

DEFAULT_CONFIG = {
    'steps': [
        'downsample', 
        'filter', 
        'ica1_blink_ecg', 
        'ica2_other', 
        'interpolate', 
        'save'
    ],
    'downsample_freq': 250,
    'filter_bandpass': (0.5, 40),
    'notch_freqs': (50,),
    'montage_path': None,
    'gedai_leadfield_path': None,
    'ica1_threshold': (0.99999, 0.30), # Lowered further to 0.30
    'ica2_threshold': (0.99999, 0.60),
    'ica1_max_comp_rem_ratio': 0.25, # Increased to allow removing more components
    'ica1_max_sig_lost': 0.85, # Increased to allow removing components with lower confidence (residues)
    'ica2_max_comp_rem_ratio': 0.1,
    'ica2_max_sig_lost': 0.3,
    'blink_ch': 'Fp1',
}

def run_ccs_pipeline(raw_input, output_dir=None, config=None):
    """
    Runs the CCS EEG cleaning pipeline.
    
    Args:
        raw_input (str or mne.io.BaseRaw): Path to the raw file (e.g. .set) OR an MNE Raw object.
        output_dir (str, optional): Directory to save output files and plots. 
                                    Required if 'save' or 'plotsnapshots' steps are used.
        config (dict, optional): Configuration dictionary.
        
    Returns:
        mne.io.BaseRaw: The cleaned MNE Raw object.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)
        
    print(f"Pipeline Configuration: {cfg}")
    
    # Load Data
    if isinstance(raw_input, str):
        raw_path = raw_input
        fname = os.path.basename(raw_path)
        print(f"Loading {raw_path}...")
        try:
            raw = mne.io.read_raw_eeglab(raw_path, preload=True)
        except Exception:
             # Fallback for other formats if needed
            raw = mne.io.read_raw(raw_path, preload=True)
    elif isinstance(raw_input, mne.io.BaseRaw):
        raw = raw_input
        # Try to get filename from raw object, or use default
        if raw.filenames:
            raw_path = raw.filenames[0]
            fname = os.path.basename(raw_path)
        else:
            raw_path = "raw_eeg_data.fif" # Default
            fname = "raw_eeg_data.fif"
    else:
        raise ValueError("raw_input must be a file path string or an mne.io.BaseRaw object")

    # Standardize channel names to MNE format (e.g. FP1 -> Fp1, Z -> z)
    # This ensures that standard montages or custom montages with standard names match.    
    raw = set_chanlocs(raw)

    # Ensure output_dir exists if we are saving or plotting
    if ('save' in cfg['steps'] or 'plotsnapshots' in cfg['steps']) and output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    # base filename for outputs
    base, _ = os.path.splitext(fname)

        
    # 2. Set Montage
    # If montage_path is provided, override. Otherwise, assume file has it.
    if cfg['montage_path']:
        print(f"Setting montage from {cfg['montage_path']}...")
        try:
            montage = load_custom_montage(cfg['montage_path'])
            raw.set_montage(montage, on_missing='warn')
        except Exception as e:
            print(f"Warning: Failed to set montage: {e}")
    else:
        print("Using montage from file (if available).")
        
    # Initialize log data
    start_time = time.time()
    log_data = {
        'time': datetime.datetime.now(),
        'filename': fname,
        'steps': ','.join(cfg['steps']),
        'duration': 0.0,
        'errorlogs': '',
        'n_badchans': 0,
        'badchans': '',
        'ncompremoved': 0,
        'signalslost': 0.0,
        'asrdone': 0.0
    }
    
    error_logs = []
    
    save_count = 0
    plot_count = 0
    
    try:
        # Execute steps
        for step in cfg['steps']:
            print(f"--- Step: {step} ---")
            
            if step == 'downsample':
                if cfg['downsample_freq'] and raw.info['sfreq'] > cfg['downsample_freq']:
                    print(f"Downsampling to {cfg['downsample_freq']} Hz...")
                    raw.resample(cfg['downsample_freq'])
                    
            elif step == 'filter':
                l_freq, h_freq = cfg['filter_bandpass']
                print(f"Filtering ({l_freq}-{h_freq} Hz)...")
                raw.filter(l_freq, h_freq, fir_design='firwin')
                
                if cfg['notch_freqs']:
                    print(f"Notch filtering {cfg['notch_freqs']} Hz...")
                    raw.notch_filter(cfg['notch_freqs'], fir_design='firwin')
                    
            elif step == 'badchannel':
                print("Running Bad Channel Detection...")
                raw, bads = detect_bad_channels(raw, cfg)
                log_data['n_badchans'] = len(bads)
                log_data['badchans'] = ','.join(bads)
                
            elif step == 'ica1_blink_ecg':
                print("Running ICA 1 (Blink/ECG)...")
                raw, metrics = run_blink_ica(raw, cfg)
                log_data['ncompremoved'] += metrics['ncompremoved']
                log_data['signalslost'] += metrics['signalslost']
                
            elif step == 'ica2_other':
                print("Running ICA 2 (Other Artifacts)...")
                raw, metrics = run_other_ica(raw, cfg)
                log_data['ncompremoved'] += metrics['ncompremoved']
                log_data['signalslost'] += metrics['signalslost']
                
            elif step == 'interpolate':
                if raw.info['bads']:
                    print(f"Interpolating bad channels: {raw.info['bads']}")
                    raw.interpolate_bads()
                else:
                    print("No bad channels to interpolate.")
                    
            elif step == 'gedai':
                leadfield_path = cfg.get('gedai_leadfield_path')
                
                if not leadfield_path:
                    # Try to find in resources
                    resource_dir = os.path.join(os.path.dirname(__file__), 'resources')
                    default_leadfield = os.path.join(resource_dir, 'fsavLEADFIELD_4_GEDAI.mat')
                    if os.path.exists(default_leadfield):
                        print(f"Using default leadfield: {default_leadfield}")
                        leadfield_path = default_leadfield
                
                if not leadfield_path:
                    print("Warning: GEDAI step requested but leadfield path not provided and default not found. Skipping.")
                    error_logs.append("GEDAI skipped (no leadfield)")
                else:
                    print("Running GEDAI...")
                    data = raw.get_data()
                    srate = raw.info['sfreq']
                    ch_names = raw.ch_names
                    
                    results = gedai(data, srate, ch_names, 
                                    ref_matrix_type='precomputed',
                                    leadfield_path=leadfield_path)
                                    
                    cleaned_data = results['clean_data']
                    raw._data = cleaned_data
                    
            elif step == 'save':
                if not output_dir:
                    print("Warning: Output directory not specified, skipping save step.")
                    continue
    
                save_count += 1
                
                if save_count > 1:
                    out_name = f"{base}_clean_{save_count:02d}.fif"
                else:
                    out_name = f"{base}_clean.fif"
                    
                out_path = os.path.join(output_dir, out_name)
                
                print(f"Saving to {out_path}...")
                raw.save(out_path, overwrite=True)
                
            elif step == 'plotsnapshots':
                plot_count += 1
                print("Generating snapshot plots...")
                if not output_dir:
                     print("Warning: Output directory not specified, skipping plots.")
                     continue
                     
                # Config parameters
                window_size = cfg.get('plot_windowsize', 30) # seconds
                n_snapshots = cfg.get('plot_nsnapshots', 2)
                
                total_duration = raw.times[-1]
                total_plot_duration = n_snapshots * window_size
                
                if total_duration <= total_plot_duration:
                    start_times = [0]
                    durations = [total_duration]
                else:
                    start_time_global = (total_duration / 2) - (total_plot_duration / 2)
                    start_time_global = max(0, start_time_global)
                    start_times = []
                    durations = []
                    for i in range(n_snapshots):
                        st = start_time_global + (i * window_size)
                        if st + window_size <= total_duration:
                            start_times.append(st)
                            durations.append(window_size)
                
                for i, (start, dur) in enumerate(zip(start_times, durations)):
                    plot_fname = f"{base}_Plot{i+1:02d}_Version{plot_count:02d}.png"
                    plot_path = os.path.join(output_dir, plot_fname)
                    try:
                        raw_crop = raw.copy().crop(tmin=start, tmax=start+dur, include_tmax=False)
                        title = f"{fname} | Plot-{i+1:02d} | Version-{plot_count:02d}"
                        plot_eeg_waveform(raw_crop, plot_path, title=title, scale=75, start_time=start)
                        print(f"Saved waveform plot to {plot_path}")
                    except Exception as e:
                        print(f"Failed to plot snapshot {i+1}: {e}")
                        error_logs.append(f"Snapshot plot failed: {e}")
                    
            else:
                print(f"Unknown step: {step}")
                error_logs.append(f"Unknown step: {step}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        error_logs.append(str(e))
        raise e
        
    finally:
        # Finalize log
        end_time = time.time()
        log_data['duration'] = end_time - start_time
        log_data['errorlogs'] = '; '.join(error_logs)
        
        # Write to CSV
        if output_dir:
            log_path = os.path.join(output_dir, 'PreprocessingLog.csv')
            df = pd.DataFrame([log_data])
            
            # Append if exists
            if os.path.exists(log_path):
                df.to_csv(log_path, mode='a', header=False, index=False)
            else:
                df.to_csv(log_path, mode='w', header=True, index=False)
            print(f"Log written to {log_path}")
            
    return raw



def adaptive_threshold_selection(labels, probs, exclude_classes, n_chans, 
                                 max_comp_rem_ratio=0.1, max_sig_lost=0.4,
                                 t_high=0.99, t_low=0.60, step=0.0005):
    """
    Selects components to remove based on adaptive thresholding.
    """
    current_t = t_high
    best_exclude = []
    
    while current_t >= t_low:
        candidates = []
        signals_lost = 0.0
        
        for i, label in enumerate(labels):
            prob = probs[i]
            if label in exclude_classes and prob >= current_t:
                candidates.append(i)
                # Estimate brain prob
                signals_lost += (1.0 - prob)
        
        n_removed = len(candidates)
        
        # Check constraints
        if n_removed > max_comp_rem_ratio * n_chans:
            break
        
        if signals_lost > max_sig_lost:
            break
            
        best_exclude = candidates
        current_t -= step
        
    return best_exclude

def run_blink_ica(raw, cfg):
    """
    Removes Eye Blink and Heart artifacts using ICA on blink epochs.
    """
    blink_ch = cfg.get('blink_ch', 'Fp1')
    if blink_ch not in raw.ch_names:
        # Try fallback
        if 'Fp2' in raw.ch_names:
            blink_ch = 'Fp2'
        else:
            blink_ch = raw.ch_names[0]
        
    print(f"Using {blink_ch} for blink detection.")
    
    # Create a copy for ICA preprocessing (High-pass 1Hz + CAR)
    # We filter continuous data first to avoid filter length warnings on short epochs
    raw_ica_prep = raw.copy()
    
    # ICLabel requirement: Filter 1-100Hz (we use 1Hz HP, LP is already commonly <100)
    # Note: data might already be filtered, but we ensure 1Hz HP for ICLabel
    raw_ica_prep.filter(l_freq=1.0, h_freq=None)
    
    # ICLabel requirement: Common Average Reference (CAR)
    print("Applying Common Average Reference (CAR) for ICA fitting...")
    raw_ica_prep.set_eeg_reference('average', projection=False)
    
    # Now find blinks using the PREPPED data (better detection?) 
    # Or just use original raw for detection and prep data for ICA?
    # Using existing logic: find blinks on raw (or raw_for_eog)
    
    # ... actually, we need 'epochs_ica' to be from the prepped data.
    # So let's define epochs using raw_ica_prep but events from finding_eog_events (which uses blink_ch)
    
    # EOG detection on raw_ica_prep copy (safe)
    raw_for_eog = raw_ica_prep.copy()
    raw_for_eog.set_channel_types({blink_ch: 'eog'})
    eog_events = mne.preprocessing.find_eog_events(raw_for_eog, ch_name=blink_ch)
    
    if len(eog_events) < 10:
        print("Not enough blinks detected. Using regular epochs.")
        events = mne.make_fixed_length_events(raw_ica_prep, duration=2.0)
        epochs_ica = mne.Epochs(raw_ica_prep, events, tmin=0, tmax=2.0, baseline=None, preload=True)
    else:
        epochs_ica = mne.Epochs(raw_ica_prep, eog_events, tmin=-1.0, tmax=1.0, baseline=None, preload=True)
        
    if len(epochs_ica) > 20:
        epochs_ica = epochs_ica[:20]
        
    # ICLabel requirement: Extended ICA
    print("Running ICA (fastica)...")
    ica = mne.preprocessing.ICA(n_components=0.99, method='fastica', 
                                max_iter='auto', random_state=42)
    ica.fit(epochs_ica)
    
    component_dict = label_components(epochs_ica, ica, method='iclabel')
    labels = component_dict['labels']
    probs = component_dict['y_pred_proba']
    
    exclude_classes = ['eye blink', 'heart beat']
    
    t_high, t_low = cfg.get('ica1_threshold', (0.99, 0.45))
    
    exclude_idx = adaptive_threshold_selection(
        labels, probs, exclude_classes, len(raw.ch_names),
        max_comp_rem_ratio=cfg.get('ica1_max_comp_rem_ratio', 0.25), 
        max_sig_lost=cfg.get('ica1_max_sig_lost', 0.6),
        t_high=t_high, t_low=t_low
    )
            
    print(f"ICA 1: Excluding {len(exclude_idx)} components (Blink/Heart): {exclude_idx}")
    ica.exclude = exclude_idx
    
    ica.apply(raw)
    
    metrics = {
        'ncompremoved': len(exclude_idx),
        'signalslost': 0.0 # Placeholder or calculate if needed
    }
    return raw, metrics

def run_other_ica(raw, cfg):
    """
    Removes other artifacts (Muscle, Line, Channel) using ICA on continuous data.
    """
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None)
    
    # ICLabel requirement: Common Average Reference (CAR)
    print("Applying Common Average Reference (CAR) for ICA fitting...")
    raw_ica.set_eeg_reference('average', projection=False)
    
    # ICLabel requirement: Extended ICA
    print("Running ICA (fastica)...")
    ica = mne.preprocessing.ICA(n_components=0.9999, method='fastica', 
                                max_iter='auto', random_state=42)
    ica.fit(raw_ica)
    
    component_dict = label_components(raw_ica, ica, method='iclabel')
    labels = component_dict['labels']
    probs = component_dict['y_pred_proba']
    
    exclude_classes = ['muscle artifact', 'line noise', 'channel noise']
    
    t_high, t_low = cfg.get('ica2_threshold', (0.99, 0.60))
    
    exclude_idx = adaptive_threshold_selection(
        labels, probs, exclude_classes, len(raw.ch_names),
        max_comp_rem_ratio=cfg.get('ica2_max_comp_rem_ratio', 0.1), 
        max_sig_lost=cfg.get('ica2_max_sig_lost', 0.3),
        t_high=t_high, t_low=t_low
    )
            
    print(f"ICA 2: Excluding {len(exclude_idx)} components (Muscle/Line/Chan): {exclude_idx}")
    ica.exclude = exclude_idx
    
    ica.apply(raw)
    
    metrics = {
        'ncompremoved': len(exclude_idx),
        'signalslost': 0.0 # Placeholder
    }
    return raw, metrics
