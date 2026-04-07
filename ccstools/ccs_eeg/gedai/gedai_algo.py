import numpy as np
import scipy.linalg
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed
from .utils import modwt_mra, create_cosine_weights, subspace_angles, get_leadfield_cov

def gedai(eeg_data, srate, chan_labels, ref_matrix_type='precomputed', 
          artifact_threshold_type='auto', epoch_size=1.0, 
          parallel=True, visualize_artifacts=False, leadfield_path=None):
    """
    Main GEDAI function to denoise EEG data.
    
    Args:
        eeg_data (numpy.ndarray): EEG data (channels x samples).
        srate (float): Sampling rate in Hz.
        chan_labels (list): List of channel labels.
        ref_matrix_type (str or numpy.ndarray): 'precomputed', 'interpolated' (not fully supported yet), or custom matrix.
        artifact_threshold_type (str): 'auto', 'auto+', 'auto-', or float.
        epoch_size (float): Epoch size in seconds.
        parallel (bool): Whether to use parallel processing.
        visualize_artifacts (bool): Not implemented in Python version.
        leadfield_path (str): Path to the leadfield .mat file (required for 'precomputed').
        
    Returns:
        dict: Dictionary containing cleaned data, artifacts, and scores.
    """
    
    # Ensure even number of samples per epoch
    samples_per_epoch = int(epoch_size * srate)
    if samples_per_epoch % 2 != 0:
        # Adjust epoch size to be even
        samples_per_epoch += 1 # Make it even (if it was odd, +1 makes it even)
        epoch_size = samples_per_epoch / srate
        
    # Create Reference Covariance Matrix
    if isinstance(ref_matrix_type, str):
        if ref_matrix_type == 'precomputed':
            if leadfield_path is None:
                raise ValueError("leadfield_path must be provided for 'precomputed' reference.")
            from .utils import load_leadfield
            L = load_leadfield(leadfield_path)
            ref_cov = get_leadfield_cov(L, chan_labels)
        elif ref_matrix_type == 'interpolated':
            raise NotImplementedError("Interpolated leadfield not yet implemented.")
        else:
            raise ValueError(f"Unknown ref_matrix_type: {ref_matrix_type}")
    else:
        ref_cov = ref_matrix_type
        
    # Pre-processing: Average Reference
    # MATLAB: GEDAI_nonRankDeficientAveRef
    # Subtract average including implicit reference (assumed 0 for unipolar)
    # Actually MATLAB code: sum(EEG.data,1)/(EEG.nbchan+1)
    n_chans = eeg_data.shape[0]
    avg_pot = np.sum(eeg_data, axis=0) / (n_chans + 1)
    eeg_data = eeg_data - avg_pot
    
    # First pass: Broadband denoising
    print("Artifact threshold detection... please wait")
    broadband_data, _, broadband_sensai, broadband_thresh = gedai_per_band(
        eeg_data, srate, epoch_size, ref_cov, 
        artifact_threshold_type, 'parabolic', parallel
    )
    
    sensai_score_per_band = [broadband_sensai]
    artifact_threshold_per_band = [broadband_thresh]
    
    # Second pass: Wavelet decomposition
    # MATLAB: modwt with 'haar', level 3
    # We use our modwt_mra which returns [D1, D2, D3, A3]
    # MATLAB modwt returns [W1; W2; ...; VJ] where Wj are details and VJ is approx.
    # Our modwt_mra returns [D1, D2, ..., Dn, An]
    
    # Transpose for PyWavelets if needed, but our utils handles 1D.
    # We need to apply this to each channel.
    # MATLAB: wpt_EEG = modwt(unfiltered_data, wavelet_type, number_of_wavelet_bands);
    # unfiltered_data is samples x channels (transposed).
    # modwt in MATLAB operates on columns.
    # So we get bands x samples x channels.
    
    unfiltered_data = broadband_data # channels x samples
    n_levels = 3
    
    # Perform MODWT per channel
    # Result: bands x channels x samples
    # We can use a loop or parallelize this too if slow
    coeffs_per_chan = []
    for ch in range(n_chans):
        coeffs = modwt_mra(unfiltered_data[ch, :], 'haar', n_levels)
        coeffs_per_chan.append(coeffs)
        
    # Stack: channels x bands x samples -> bands x channels x samples
    wpt_eeg = np.stack(coeffs_per_chan, axis=1)
    
    n_bands = wpt_eeg.shape[0]
    # Exclude lowest bands based on frequency
    # MATLAB: lowest_wavelet_bands_to_exclude = ceil(600/EEGin.srate);
    # This logic seems specific to their setup. 600 samples?
    # If srate=500, ceil(1.2) = 2.
    # If srate=250, ceil(2.4) = 3.
    # This likely refers to excluding low frequency bands (Approximation).
    lowest_bands_to_exclude = int(np.ceil(600 / srate))
    num_bands_to_process = n_bands - lowest_bands_to_exclude
    
    wavelet_band_filtered_data = np.zeros((num_bands_to_process, n_chans, wpt_eeg.shape[2]))
    
    # Denoise each band
    if parallel:
        results = Parallel(n_jobs=-1, backend='threading')(
            delayed(gedai_per_band)(
                wpt_eeg[f, :, :], srate, epoch_size, ref_cov, 
                artifact_threshold_type, 'parabolic', False
            ) for f in range(num_bands_to_process)
        )
        
        for f, res in enumerate(results):
            cleaned, _, sensai, thresh = res
            n_samples_cleaned = cleaned.shape[1]
            wavelet_band_filtered_data[f, :, :n_samples_cleaned] = cleaned
            sensai_score_per_band.append(sensai)
            artifact_threshold_per_band.append(thresh)
            
    else:
        for f in range(num_bands_to_process):
            cleaned, _, sensai, thresh = gedai_per_band(
                wpt_eeg[f, :, :], srate, epoch_size, ref_cov, 
                artifact_threshold_type, 'parabolic', False
            )
            n_samples_cleaned = cleaned.shape[1]
            wavelet_band_filtered_data[f, :, :n_samples_cleaned] = cleaned
            sensai_score_per_band.append(sensai)
            artifact_threshold_per_band.append(thresh)
            
    # Finalization
    # Reconstruct EEG
    # Sum processed bands
    cleaned_sum = np.sum(wavelet_band_filtered_data, axis=0)
    
    # Add back excluded bands (unprocessed)
    if lowest_bands_to_exclude > 0:
        excluded_sum = np.sum(wpt_eeg[num_bands_to_process:, :, :], axis=0)
        final_cleaned_data = cleaned_sum + excluded_sum
    else:
        final_cleaned_data = cleaned_sum
        
    # Pad to original length if needed
    original_len = eeg_data.shape[1]
    current_len = final_cleaned_data.shape[1]
    if current_len < original_len:
        # Pad with original data (or zeros, but original data is better to avoid drop)
        # Actually, if we processed broadband, we might have removed artifacts.
        # If we paste original data, we paste artifacts back.
        # But it's just 1 sample.
        # Let's pad with zeros for safety or repeat last sample.
        padding = np.zeros((n_chans, original_len - current_len))
        final_cleaned_data = np.hstack((final_cleaned_data, padding))
    elif current_len > original_len:
        final_cleaned_data = final_cleaned_data[:, :original_len]
    
    artifacts_data = eeg_data - final_cleaned_data
    
    # Final SENSAI score
    sensai_score = sensai_basic(final_cleaned_data, artifacts_data, srate, epoch_size, ref_cov, 1.0)
    
    return {
        'clean_data': final_cleaned_data,
        'artifacts': artifacts_data,
        'sensai_score': sensai_score,
        'sensai_score_per_band': sensai_score_per_band,
        'thresholds': artifact_threshold_per_band
    }

def gedai_per_band(data, srate, epoch_size, ref_cov, threshold_type, opt_type, parallel):
    """
    Process a single band of data.
    """
    n_chans, n_pnts = data.shape
    epoch_samples = int(srate * epoch_size)
    
    # Crop to whole epochs
    n_epochs = n_pnts // epoch_samples
    data = data[:, :n_epochs * epoch_samples]
    
    # Stream 1
    epoched_1 = data.reshape(n_chans, epoch_samples, n_epochs, order='F')
    
    # Stream 2 (shifted)
    shift = epoch_samples // 2
    data_2 = data[:, shift:-shift]
    epoched_2 = data_2.reshape(n_chans, epoch_samples, n_epochs - 1, order='F')
    
    # Covariance matrices
    # MATLAB: cov(EEGdata_epoched(:,:,epo)') -> cov of (samples x chans) -> chans x chans
    cov_1 = np.zeros((n_epochs, n_chans, n_chans))
    for i in range(n_epochs):
        cov_1[i] = np.cov(epoched_1[:, :, i])
        
    cov_2 = np.zeros((n_epochs - 1, n_chans, n_chans))
    for i in range(n_epochs - 1):
        cov_2[i] = np.cov(epoched_2[:, :, i])
        
    # GEVD
    reg_lambda = 0.05
    # refCOV_reg = (1-lambda)*ref + lambda*mean(eig(ref))*eye
    eig_ref = np.linalg.eigvalsh(ref_cov) # symmetric
    ref_cov_reg = (1 - reg_lambda) * ref_cov + reg_lambda * np.mean(eig_ref) * np.eye(n_chans)
    
    # Eigendecomposition
    # scipy.linalg.eig(a, b) solves a v = w b v
    # MATLAB: eig(COV, refCOV, 'chol')
    # We need generalized eigenvalues.
    
    def compute_gevd(cov_mat, ref_mat):
        # returns vals, vecs
        # scipy eigh returns (vals, vecs) for symmetric/hermitian matrices
        # vals are guaranteed real and sorted ascending
        # We use eigh because cov_mat and ref_mat are symmetric covariance matrices
        vals, vecs = scipy.linalg.eigh(cov_mat, ref_mat)
        # Sort by eigenvalue descending
        idx = np.argsort(vals)[::-1]
        return vals[idx], vecs[:, idx]

    evals_1 = np.zeros((n_epochs, n_chans)) # Store diagonals
    evecs_1 = np.zeros((n_epochs, n_chans, n_chans))
    
    for i in range(n_epochs):
        v, e = compute_gevd(cov_1[i], ref_cov_reg)
        evals_1[i] = np.real(v)
        evecs_1[i] = e
        
    evals_2 = np.zeros((n_epochs - 1, n_chans))
    evecs_2 = np.zeros((n_epochs - 1, n_chans, n_chans))
    
    for i in range(n_epochs - 1):
        v, e = compute_gevd(cov_2[i], ref_cov_reg)
        evals_2[i] = np.real(v)
        evecs_2[i] = e
        
    # Threshold determination
    if isinstance(threshold_type, str) and threshold_type.startswith('auto'):
        if threshold_type == 'auto+': noise_mult = 1
        elif threshold_type == 'auto': noise_mult = 3
        else: noise_mult = 6 # auto-
        
        min_t, max_t = 0, 12
        
        if opt_type == 'parabolic':
            # Minimize negative SENSAI score
            def objective(t):
                # We only need sensai score
                _, _, score = sensai(epoched_1, srate, epoch_size, t, ref_cov, evals_1, evecs_1, noise_mult)
                return -score
            
            res = minimize_scalar(objective, bounds=(min_t, max_t), method='bounded')
            optimal_threshold = res.x
        else:
            # Grid search not implemented for brevity, fallback to parabolic or default
            optimal_threshold = 6.0
            
        artifact_threshold = optimal_threshold
    else:
        artifact_threshold = float(threshold_type)
        
    # Clean EEG
    cleaned_1, artifacts_1, _ = clean_eeg(epoched_1, srate, epoch_size, artifact_threshold, ref_cov, evals_1, evecs_1)
    cleaned_2, artifacts_2, _ = clean_eeg(epoched_2, srate, epoch_size, artifact_threshold, ref_cov, evals_2, evecs_2)
    
    # Combine streams
    cosine_weights = create_cosine_weights(n_chans, epoch_samples, True)
    
    # Apply weights to stream 2
    # cleaned_2 is (n_chans, total_samples)
    # But clean_eeg returns continuous data
    
    # We need to apply weights to the continuous data of stream 2
    # Stream 2 length: (n_epochs-1)*epoch_samples
    
    # Weights are per epoch?
    # MATLAB:
    # cleaned_data_2(:, 1:shifting) = ...
    # cleaned_data_2(:, sample_end+1:end) = ...
    # Wait, MATLAB applies weights to the ENDS of the continuous stream 2?
    # "Apply weights to the second (shifted) stream"
    # It seems it applies weights to the transition regions?
    
    # Actually, MATLAB code:
    # cosine_weights = create_cosine_weights(..., 1);
    # cleaned_data_2(:, 1:shifting) = cleaned_data_2(:, 1:shifting) .* cosine_weights(:, 1:shifting);
    # ...
    
    # The cosine weights are size (chans, epoch_samples).
    # 1:shifting is the first half of the weights.
    
    len_2 = cleaned_2.shape[1]
    sample_end = len_2 - shift
    
    cleaned_2[:, :shift] *= cosine_weights[:, :shift]
    cleaned_2[:, sample_end:] *= cosine_weights[:, shift:]
    
    artifacts_2[:, :shift] *= cosine_weights[:, :shift]
    artifacts_2[:, sample_end:] *= cosine_weights[:, shift:]
    
    # Combine
    cleaned_data = cleaned_1.copy()
    artifacts_data = artifacts_1.copy()
    
    # Add stream 2 to the middle of stream 1
    # Stream 1 has n_epochs. Stream 2 has n_epochs-1.
    # Stream 2 starts at shift and ends at end-shift of Stream 1.
    
    cleaned_data[:, shift:shift+len_2] += cleaned_2
    artifacts_data[:, shift:shift+len_2] += artifacts_2
    
    # Final score
    _, _, final_score = sensai(epoched_1, srate, epoch_size, artifact_threshold, ref_cov, evals_1, evecs_1, 1)
    
    return cleaned_data, artifacts_data, final_score, artifact_threshold

def clean_eeg(epoched_data, srate, epoch_size, threshold, ref_cov, evals, evecs):
    """
    Removes artifacts based on GEVD components.
    """
    n_chans, n_samples, n_epochs = epoched_data.shape
    
    # Calculate global threshold using PIT
    # evals is (n_epochs, n_chans)
    # Flatten and take log
    # MATLAB: log(magnitudes(magnitudes > 0)) + 100
    flat_evals = np.abs(evals.flatten())
    valid_evals = flat_evals[flat_evals > 0]
    log_evals = np.log(valid_evals) + 100
    
    # T1 calculation
    t1 = 1.0 * (105 - threshold) / 100
    
    # PIT fitting
    # ECDF
    sorted_data = np.sort(log_evals)
    y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    # Unique values for interpolation
    unique_x, idx = np.unique(sorted_data, return_index=True)
    unique_y = y_vals[idx]
    
    # Interpolate to find transformed data
    # transformed_data = interp1(unique_x, unique_y, original_data, ...)
    # We can just use the y_vals corresponding to the data rank
    # But let's follow logic: find outliers where PIT > 0.95
    
    # We need to find the value in sorted_data where y_vals > 0.95
    cutoff_idx = np.searchsorted(unique_y, 0.95)
    if cutoff_idx < len(unique_x):
        cutoff_val = unique_x[cutoff_idx]
        # Outliers are values > cutoff_val
        outliers = sorted_data[sorted_data > cutoff_val]
        if len(outliers) > 0:
            threshold_val = t1 * np.min(outliers)
        else:
            threshold_val = t1 * np.max(sorted_data) # Fallback
    else:
        threshold_val = t1 * np.max(sorted_data)
        
    # Cleaning loop
    cleaned_epoched = np.zeros_like(epoched_data)
    artifacts_epoched = np.zeros_like(epoched_data)
    cosine_weights = create_cosine_weights(n_chans, n_samples, True)
    half_epoch = n_samples // 2
    
    for i in range(n_epochs):
        # Evecs for this epoch: (n_chans, n_chans)
        # Evals for this epoch: (n_chans,)
        # Note: evecs are columns.
        
        W = evecs[i].copy() # Spatial filter matrix
        
        # Zero out components below threshold
        # MATLAB: if abs(Eval(j,j,i)) < exp(Treshold1 - 100) -> component = 0
        # Wait, MATLAB code:
        # if abs(Eval(j,j,i)) < exp(Treshold1 - 100)
        #     component_spatial_filter(:,j) = 0;
        # end
        # This keeps components with LARGE eigenvalues?
        # Usually artifacts have LARGE eigenvalues in GEVD if ref is signal.
        # But here ref is Leadfield (Signal).
        # So Signal has LARGE eigenvalues. Noise has SMALL eigenvalues.
        # So we remove components with SMALL eigenvalues?
        # "if val < threshold, set to 0". Yes.
        
        current_evals = np.abs(evals[i])
        threshold_exp = np.exp(threshold_val - 100)
        
        # Identify components to keep/remove
        # We want to REMOVE artifacts.
        # If ref is Signal, then High Eigenvalues = Signal. Low Eigenvalues = Noise.
        # So we want to keep High Eigenvalues.
        # MATLAB sets column to 0 if value < threshold.
        # So it REMOVES Low Eigenvalues.
        # Correct.
        
        mask = current_evals < threshold_exp
        W[:, mask] = 0
        
        # Reconstruct Artifacts?
        # MATLAB:
        # artifacts_timecourses = component_spatial_filter' * EEGdata_epoched(:,:,i);
        # Signal_to_remove = Evec(:,:,i)' \ artifacts_timecourses;
        # artifacts(:, :, i) = Signal_to_remove;
        # cleaned_epoch = EEGdata_epoched(:,:,i) - Signal_to_remove;
        
        # Wait, if W has 0s for noise components, then W' * data projects only SIGNAL components?
        # If W has 0s for NOISE (small vals), then we are keeping SIGNAL.
        # So `artifacts_timecourses` contains SIGNAL sources?
        # And `Signal_to_remove` reconstructs SIGNAL?
        # Then `cleaned = data - Signal_to_remove` would be NOISE?
        
        # Let's re-read MATLAB carefully.
        # if abs(Eval) < exp(...) -> set col to 0.
        # So W contains only High Eigenvectors (Signal).
        # artifacts_timecourses = W' * data -> Signal Sources.
        # Signal_to_remove = inv(Evec') * artifacts_timecourses -> Reconstructed Signal.
        # cleaned = data - Signal_to_remove -> Noise?
        
        # Variable naming in MATLAB: `Signal_to_remove`.
        # Maybe the logic is inverted?
        # If refCOV is Leadfield (Signal), then Signal is in High Eigenvalues.
        # If we zero out Low Eigenvalues, we keep Signal.
        # So we reconstruct Signal.
        # Then we subtract Signal from Data.
        # Result is Noise.
        # But the function returns `cleaned_data`.
        # If `cleaned_data = Noise`, that's wrong.
        
        # Let's check `GEDAI.m`:
        # EEGclean.data = squeeze(sum(wavelet_band_filtered_data, 1));
        # EEGartifacts.data = EEGin.data - EEGclean.data;
        
        # If `clean_EEG` returns `cleaned_data` which is actually Noise, then `EEGclean` would be Noise.
        # And `EEGartifacts` would be Signal.
        # That contradicts the names.
        
        # Let's check `clean_EEG.m` again.
        # artifacts(:, :, i) = Signal_to_remove;
        # cleaned_epoch = EEGdata_epoched(:,:,i) - Signal_to_remove;
        
        # If `Signal_to_remove` is artifacts, then `cleaned_epoch` is clean.
        # So `Signal_to_remove` must be the Artifacts.
        # For `Signal_to_remove` to be Artifacts, W must contain Artifact components.
        # Artifact components would be those we want to REMOVE.
        # If we zero out columns where val < threshold, we KEEP columns where val > threshold.
        # So we keep High Eigenvalues.
        # If High Eigenvalues = Artifacts, then refCOV must be Noise?
        # But refCOV is Leadfield (Signal).
        
        # Maybe I am misunderstanding GEVD.
        # eig(R, S). Maximize v'Rv / v'Sv.
        # R = Data Covariance. S = Reference (Signal) Covariance.
        # Maximize Power(Data) / Power(Signal).
        # High eigenvalue = High Data Power relative to Signal Power -> Noise/Artifact?
        # Low eigenvalue = Low Data Power relative to Signal Power -> Signal?
        
        # MATLAB: eig(COV, refCOV).
        # COV is data. refCOV is signal model.
        # Large lambda means direction has much more variance in Data than in Signal Model.
        # Since Data = Signal + Noise, and Signal Model ~ Signal.
        # Large lambda -> Noise.
        # Small lambda (near 1) -> Signal.
        
        # So High Eigenvalues = Artifacts.
        # We zero out SMALL eigenvalues (Signal).
        # So W keeps LARGE eigenvalues (Artifacts).
        # So `Signal_to_remove` is Artifacts.
        # Correct.
        
        # Logic:
        # val < threshold -> set to 0. (Remove Signal components from the filter)
        # Keep val > threshold. (Keep Artifact components)
        # Reconstruct from kept components -> Artifacts.
        # Subtract from data -> Clean Data.
        
        # Implementation details:
        # Signal_to_remove = Evec' \ artifacts_timecourses
        # Evec' \ X is equivalent to inv(Evec') * X = inv(Evec)' * X.
        # Since Evec is not necessarily orthogonal in GEVD, we use solve.
        
        sources = W.T @ epoched_data[:, :, i]
        # sources shape: (n_chans, n_samples)
        
        # Reconstruct
        # We need to project back.
        # In PCA: X_hat = W * S.
        # In GEVD: X = V * S?
        # MATLAB: Evec(:,:,i)' \ artifacts_timecourses
        # Let A = Evec'. X = Signal_to_remove. B = artifacts_timecourses.
        # A * X = B -> X = A \ B.
        
        # So we solve Evec.T * X = sources
        recon_artifacts = scipy.linalg.solve(evecs[i].T, sources)
        
        artifacts_epoched[:, :, i] = recon_artifacts
        cleaned_epoch = epoched_data[:, :, i] - recon_artifacts
        
        # Windowing
        if i == 0:
            cleaned_epoch[:, half_epoch:] *= cosine_weights[:, half_epoch:]
        elif i == n_epochs - 1:
            cleaned_epoch[:, :half_epoch] *= cosine_weights[:, :half_epoch]
        else:
            cleaned_epoch *= cosine_weights
            
        cleaned_epoched[:, :, i] = cleaned_epoch
        
    # Reshape to continuous
    cleaned_data = cleaned_epoched.reshape(n_chans, -1, order='F')
    artifacts_data = artifacts_epoched.reshape(n_chans, -1, order='F')
    
    return cleaned_data, artifacts_data, threshold

def sensai(epoched_data, srate, epoch_size, threshold, ref_cov, evals, evecs, noise_mult):
    """
    Calculates SENSAI score.
    """
    # We need to run clean_eeg to get signal and noise
    # But clean_eeg takes continuous data? No, it takes epoched data in my implementation.
    # Wait, my clean_eeg takes epoched data.
    
    cleaned_cont, artifacts_cont, _ = clean_eeg(epoched_data, srate, epoch_size, threshold, ref_cov, evals, evecs)
    
    # SENSAI basic expects continuous data or we can adapt it
    # sensai_basic(signal_data, noise_data, ...)
    
    score = sensai_basic(cleaned_cont, artifacts_cont, srate, epoch_size, ref_cov, noise_mult)
    
    # We also need signal/noise subspace similarity if we want to match full output
    # But for optimization we just need score.
    
    return 0, 0, score # Placeholders for similarities

def sensai_basic(signal_data, noise_data, srate, epoch_size, ref_cov, noise_mult):
    """
    Computes SENSAI score based on subspace similarity.
    """
    n_chans = ref_cov.shape[0]
    epoch_samples = int(srate * epoch_size)
    
    # Regularize ref_cov
    reg_lambda = 0.05
    eig_ref = np.linalg.eigvalsh(ref_cov)
    ref_cov_reg = (1 - reg_lambda) * ref_cov + reg_lambda * np.mean(eig_ref) * np.eye(n_chans)
    
    # Template subspace (Top 3 PCs of Ref)
    _, evecs_ref = scipy.linalg.eigh(ref_cov_reg)
    # eigh returns ascending. We want top 3.
    evecs_template = evecs_ref[:, -3:]
    
    # Epoch data
    n_pnts = signal_data.shape[1]
    n_epochs = n_pnts // epoch_samples
    
    sig_epoched = signal_data[:, :n_epochs*epoch_samples].reshape(n_chans, epoch_samples, n_epochs, order='F')
    noise_epoched = noise_data[:, :n_epochs*epoch_samples].reshape(n_chans, epoch_samples, n_epochs, order='F')
    
    sig_sims = []
    noise_sims = []
    
    top_pcs = 3
    
    for i in range(n_epochs):
        # Signal Subspace
        cov_sig = np.cov(sig_epoched[:, :, i])
        _, evecs_sig = scipy.linalg.eigh(cov_sig)
        evecs_sig = evecs_sig[:, -top_pcs:]
        
        angles_sig = subspace_angles(evecs_sig, evecs_template)
        sig_sims.append(np.prod(np.cos(angles_sig)))
        
        # Noise Subspace
        cov_noise = np.cov(noise_epoched[:, :, i])
        _, evecs_noise = scipy.linalg.eigh(cov_noise)
        evecs_noise = evecs_noise[:, -top_pcs:]
        
        angles_noise = subspace_angles(evecs_noise, evecs_template)
        noise_sims.append(np.prod(np.cos(angles_noise)))
        
    avg_sig_sim = 100 * np.mean(sig_sims)
    avg_noise_sim = 100 * np.mean(noise_sims)
    
    score = avg_sig_sim - noise_mult * avg_noise_sim
    return score
