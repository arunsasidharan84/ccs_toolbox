import numpy as np
import scipy.io
import mne
import os
import matplotlib
matplotlib.use('Agg') # Silence plot windows
import matplotlib.pyplot as plt

def load_custom_montage(mat_file_path):
    """
    Loads a custom montage from a .mat file (CCSNIMHANS format).
    
    Args:
        mat_file_path (str): Path to the .mat file (e.g., chanloc_10-05_346.mat).
        
    Returns:
        mne.channels.DigMontage: The MNE montage object.
    """
    if not os.path.exists(mat_file_path):
        # Check in local resources folder
        resource_dir = os.path.join(os.path.dirname(__file__), 'resources', 'resource_headplot')
        local_path = os.path.join(resource_dir, os.path.basename(mat_file_path))
        if os.path.exists(local_path):
            mat_file_path = local_path
        else:
            raise FileNotFoundError(f"Montage file not found: {mat_file_path} (checked local resources too)")
        
    try:
        mat = scipy.io.loadmat(mat_file_path, simplify_cells=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load .mat file: {e}")
        
    # Check for 'defaultChanLocs'
    if 'defaultChanLocs' not in mat:
        # Try to find any key that looks like it
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if len(keys) == 1:
            data = mat[keys[0]]
        else:
            raise ValueError(f"Could not identify channel data in {mat_file_path}. Keys: {keys}")
    else:
        data = mat['defaultChanLocs']
        
    # data should be a list of structs (dicts in simplify_cells=True)
    # or a structured array
    
    ch_names = []
    pos = {}
    
    # Handle if data is a single dict (one channel?) or list
    if isinstance(data, dict):
        data = [data] # Wrap in list
        
    for ch in data:
        # Check if fields exist
        if 'labels' not in ch:
            continue
            
        label = ch['labels']
        if not isinstance(label, str):
            continue
            
        # Coordinates
        # MATLAB X, Y, Z usually in generic units (often normalized or mm)
        # MNE expects meters.
        # We need to check the scale.
        # Usually EEGLAB coords are on a unit sphere or similar.
        
        try:
            x = ch['X']
            y = ch['Y']
            z = ch['Z']
            
            if x is None or np.isnan(x):
                continue
                
            # Assume coordinates are provided.
            # If they are empty, skip.
            
            # Store
            ch_names.append(label)
            pos[label] = np.array([x, y, z])
            
        except KeyError:
            continue
            
    if not ch_names:
        raise ValueError("No valid channel locations found in file.")
        
    # Create montage
    # We don't know the coordinate frame for sure, but usually 'head'.
    # MNE's make_dig_montage takes ch_pos dictionary.
    
    # Check scale: if values are > 10, likely mm. If < 1, likely m.
    # If around 1 (unit sphere), we might need to project or treat as m?
    # Usually head radius is ~0.09m.
    # If values are like 0.8, 0.5, they are likely unit sphere.
    # If values are like 80, 50, they are mm.
    
    # Let's inspect one value
    first_pos = pos[ch_names[0]]
    norm = np.linalg.norm(first_pos)
    
    if norm > 10: # mm
        scale = 1e-3
    elif norm > 0.5 and norm < 2: # Unit sphere?
        # If unit sphere, we might want to scale to head size?
        # MNE usually handles this if we map to a standard sphere.
        # But let's assume meters if it's small.
        scale = 1.0 # Or maybe it is already meters?
        # Actually if it's unit sphere, MNE might treat it as 1m radius head which is huge.
        # But for plotting it doesn't matter much.
        pass
    else:
        scale = 1.0
        
    # Apply scale
    for k in pos:
        pos[k] = pos[k] * scale
        
    # Coordinate Frame Correction
    # EEGLAB .mat files usually have X=Anterior, Y=Left, Z=Superior (or similar variations)
    # MNE expects X=Right, Y=Anterior, Z=Superior (RAS)
    # We detected that Fp1 (Left Frontal) had large +X (Anterior) and small +Y (Left) in the file.
    # To map to MNE:
    # MNE_X (Right) = -File_Y (Left) (Since Left is +, -Left is Right)
    # MNE_Y (Anterior) = File_X (Anterior)
    # MNE_Z (Superior) = File_Z (Superior)
    
    new_pos = {}
    for k, v in pos.items():
        # v is [x, y, z] in file frame
        file_x, file_y, file_z = v
        mne_x = -file_y
        mne_y = file_x
        mne_z = file_z
        new_pos[k] = np.array([mne_x, mne_y, mne_z])
        
    montage = mne.channels.make_dig_montage(ch_pos=new_pos, coord_frame='head')
    return montage

def plot_eeg_waveform(raw, output_path, title=None, scale=None, start_time=0):
    """
    Plots EEG data as multi-line waveform snapshot, similar to accs_plotEEGwaveform.m.
    Data is converted to uV for plotting.
    
    Args:
        raw (mne.io.Raw): MNE Raw object.
        output_path (str): Path to save the plot.
        title (str): Plot title.
        scale (float): Scaling factor in uV. If None, auto-scales based on data range.
        start_time (float): Start time offset for x-axis.
    """
    data = raw.get_data() * 1e6 # Convert to uV
    times = raw.times + start_time
    ch_names = raw.ch_names
    n_chans, n_samples = data.shape
    
    # Calculate scale
    if scale is None:
        eeg_min = np.min(data)
        eeg_max = np.max(data)
        scale = abs(eeg_max - eeg_min)
        # If scale is 0 (flatline), set default
        if scale == 0:
            scale = 100.0 # Default 100 uV
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    yticks = []
    yticklabels = []
    
    # Plot loop
    for i, chan_name in enumerate(ch_names):
        y_offset = (n_chans - i) * scale
        
        # Plot data + offset
        ax.plot(times, data[i, :] + y_offset, linewidth=0.5, color='b')
        
        yticks.append(y_offset)
        yticklabels.append(chan_name)
        
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time (s)')
    ax.set_xlim([times[0], times[-1]])
    
    # Y limits
    ax.set_ylim([0, (n_chans + 1) * scale])
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"EEG Waveforms (Scale: {scale:.1f} uV)")
        
    # Grid
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
