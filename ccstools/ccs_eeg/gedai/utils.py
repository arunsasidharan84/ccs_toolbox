import numpy as np
import scipy.io
import pywt
import os

def load_leadfield(filepath):
    """
    Loads the leadfield matrix from a .mat file.
    
    Args:
        filepath (str): Path to the .mat file.
        
    Returns:
        dict: A dictionary containing the leadfield data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Leadfield file not found: {filepath}")
        
    try:
        from pymatreader import read_mat
        mat = read_mat(filepath)
        
        # pymatreader returns a dict
        if 'leadfield4GEDAI' in mat:
            return mat['leadfield4GEDAI']
        elif 'L' in mat:
            return mat['L']
        else:
            # Check keys
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if len(keys) == 1:
                return mat[keys[0]]
            else:
                raise ValueError(f"Could not identify leadfield matrix in {filepath}. Keys: {keys}")
                
    except ImportError:
        # Fallback to h5py manual loading if pymatreader not available (but it should be)
        raise ImportError("pymatreader is required to load this leadfield file.")
    except Exception as e:
        raise RuntimeError(f"Failed to load leadfield file: {e}")

def get_leadfield_cov(leadfield_data, channel_labels):
    """
    Extracts the covariance matrix for the given channels from the leadfield data.
    
    Args:
        leadfield_data (dict): The loaded leadfield data.
        channel_labels (list of str): List of channel labels in the EEG data.
        
    Returns:
        numpy.ndarray: The covariance matrix (n_chans x n_chans).
    """
    # Normalize labels for comparison
    eeg_labels = [l.lower() for l in channel_labels]
    
    # Extract template labels
    # The structure might vary, assuming 'electrodes' -> 'Name' based on MATLAB code
    # template_electrode_labels = {L.leadfield4GEDAI.electrodes.Name};
    
    try:
        template_labels = leadfield_data['electrodes']['Name']
        if isinstance(template_labels, str): # Handle single electrode case unlikely but possible
             template_labels = [template_labels]
        template_labels = [l.lower() for l in template_labels]
    except KeyError:
        raise ValueError("Could not find electrode names in leadfield data.")

    # Find indices
    chan_indices = []
    for label in eeg_labels:
        try:
            idx = template_labels.index(label)
            chan_indices.append(idx)
        except ValueError:
             raise ValueError(f"Electrode label '{label}' not found in leadfield template.")
    
    # Extract covariance matrix
    # refCOV = L.leadfield4GEDAI.gram_matrix_avref(chanidx,chanidx);
    try:
        gram_matrix = leadfield_data['gram_matrix_avref']
        ref_cov = gram_matrix[np.ix_(chan_indices, chan_indices)]
        return ref_cov
    except KeyError:
        raise ValueError("Could not find 'gram_matrix_avref' in leadfield data.")

def modwt_mra(data, wavelet='haar', level=3):
    """
    Performs Maximal Overlap Discrete Wavelet Transform (MODWT) Multi-Resolution Analysis.
    Mimics MATLAB's modwtmra.
    
    Args:
        data (numpy.ndarray): Input signal (1D array).
        wavelet (str): Wavelet name (e.g., 'haar').
        level (int): Decomposition level.
        
    Returns:
        numpy.ndarray: Array of shape (level + 1, len(data)) containing the MRA components.
                       Rows 0 to level-1 are Details (D1, D2, ...), last row is Approximation (A_level).
                       Note: MATLAB modwtmra usually returns D1...Dn, An.
    """
    # PyWavelets swt returns [(cA_n, cD_n), ..., (cA_1, cD_1)]
    # We need to reconstruct the time-domain signals for each detail and the final approximation.
    
    # Pad data to be divisible by 2^level if necessary (SWT requires this)
    # However, MATLAB's modwt is circular/reflection. Pywt swt handles this if we use 'periodization' or similar?
    # Actually pywt.swt requires length divisible by 2^level.
    
    original_len = len(data)
    pad_len = 0
    divisor = 2**level
    if original_len % divisor != 0:
        pad_len = divisor - (original_len % divisor)
        # Pad with zeros or edge values. MATLAB modwt defaults to reflection ('sym') usually?
        # Let's use zero padding for simplicity or edge.
        data_padded = np.pad(data, (0, pad_len), 'edge')
    else:
        data_padded = data
        
    coeffs = pywt.swt(data_padded, wavelet, level=level, start_level=0)
    # coeffs is [(cA_n, cD_n), (cA_n-1, cD_n-1), ..., (cA_1, cD_1)]
    
    mra_coeffs = []
    
    # Reconstruct Details (D1, D2, ..., Dn)
    # To reconstruct Dj, we take cDj and set all other coeffs to zero, then iswt.
    # cDj is at index level - j in the coeffs list (since it's reversed).
    # coeffs[0] is level n, coeffs[-1] is level 1.
    
    for j in range(1, level + 1):
        # Create a zeroed copy of coeffs
        zero_coeffs = []
        for i in range(level):
            # Shape of coeffs at each level
            shape = coeffs[i][0].shape
            zero_coeffs.append((np.zeros(shape), np.zeros(shape)))
            
        # Set the specific detail coefficient
        # Detail j corresponds to index level - j in the list
        # e.g. level=3. j=1 (D1). index = 2. coeffs[2] is (cA1, cD1).
        idx = level - j
        # We want to keep cD_j.
        # The tuple is (cA, cD). We keep cD.
        current_cD = coeffs[idx][1]
        zero_coeffs[idx] = (np.zeros_like(current_cD), current_cD)
        
        # Inverse SWT
        rec = pywt.iswt(zero_coeffs, wavelet)
        mra_coeffs.append(rec[:original_len])
        
    # Reconstruct Approximation (An)
    # Keep only cA_n (which is at index 0 in coeffs list)
    zero_coeffs = []
    for i in range(level):
        shape = coeffs[i][0].shape
        zero_coeffs.append((np.zeros(shape), np.zeros(shape)))
        
    current_cA = coeffs[0][0] # cA_n
    zero_coeffs[0] = (current_cA, np.zeros_like(current_cA))
    
    rec = pywt.iswt(zero_coeffs, wavelet)
    mra_coeffs.append(rec[:original_len])
    
    # Return as array (bands x samples)
    # Order: D1, D2, ..., Dn, An
    return np.array(mra_coeffs)

def subspace_angles(A, B):
    """
    Calculates the principal angles between two subspaces.
    
    Args:
        A (numpy.ndarray): Orthonormal basis for first subspace.
        B (numpy.ndarray): Orthonormal basis for second subspace.
        
    Returns:
        numpy.ndarray: Principal angles in radians.
    """
    # Ensure double precision
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    
    # SVD of A.T @ B
    # svd returns s as 1D array of singular values
    _, S, _ = np.linalg.svd(A.T @ B)
    
    # Clip to valid range for arccos
    S = np.clip(S, -1.0, 1.0)
    
    angles = np.arccos(S)
    return np.sort(angles)

def create_cosine_weights(n_chans, n_samples, fullshift=True):
    """
    Creates cosine weights for overlap-add reconstruction.
    
    Args:
        n_chans (int): Number of channels.
        n_samples (int): Number of samples in the epoch.
        fullshift (bool): True for even (full shift), False for odd.
        
    Returns:
        numpy.ndarray: Weights matrix (n_chans x n_samples).
    """
    weights = np.zeros((n_chans, n_samples))
    
    if fullshift:
        # 1-based index u in MATLAB: 1 to n_samples
        u = np.arange(1, n_samples + 1)
        w = 0.5 - 0.5 * np.cos(2 * u * np.pi / n_samples)
    else:
        u = np.arange(1, n_samples) # 1 to n_samples-1
        w_vals = 0.5 - 0.5 * np.cos(2 * u * np.pi / (n_samples - 1))
        w = np.zeros(n_samples)
        w[:n_samples-1] = w_vals
        
    for i in range(n_chans):
        weights[i, :] = w
        
    return weights
