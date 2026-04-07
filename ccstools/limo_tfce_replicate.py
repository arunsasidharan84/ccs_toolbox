import numpy as np
import scipy.io as sio
from scipy.ndimage import label, binary_erosion

def limo_findcluster_python(onoff, adjacency, minnbchan=2):
    """
    Python implementation of LIMO's limo_findcluster logic.
    onoff: boolean array of shape (n_channels,)
    adjacency: symmetric adjacency matrix (n_channels, n_channels)
    minnbchan: minimum significant neighbors required to keep a sensor
    """
    n_chans = len(onoff)
    working_onoff = onoff.copy().astype(float)
    
    # LIMO minnbchan logic: iteratively remove nodes with < minnbchan neighbors
    if minnbchan > 0:
        nremoved = 1
        while nremoved > 0:
            # count neighbors: adjacency * onoff
            # adjacency usually includes diagonal as 0 in LIMO code logic but handles it
            nsigneighb = adjacency @ working_onoff
            remove = (working_onoff > 0) & (nsigneighb < minnbchan)
            nremoved = np.sum(remove)
            working_onoff[remove] = 0
            
    if np.sum(working_onoff) == 0:
        return np.zeros(n_chans, dtype=int), 0
        
    # Find connected components among remaining nodes
    # We can use a simple graph traversal or adjacency-based labeling
    labels = np.zeros(n_chans, dtype=int)
    num_clusters = 0
    visited = np.zeros(n_chans, dtype=bool)
    
    remaining_indices = np.where(working_onoff > 0)[0]
    for idx in remaining_indices:
        if not visited[idx]:
            num_clusters += 1
            # BFS to find all connected nodes
            queue = [idx]
            visited[idx] = True
            while queue:
                curr = queue.pop(0)
                labels[curr] = num_clusters
                # Find neighbors that are in working_onoff and not visited
                neighbors = np.where((adjacency[curr] > 0) & (working_onoff > 0) & (~visited))[0]
                for n in neighbors:
                    visited[n] = True
                    queue.append(n)
                    
    return labels, num_clusters

def limo_tfce_python(data, adjacency, E=0.5, H=2, dh=0.1, minnbchan=1):
    """
    Python implementation of LIMO's limo_tfce logic.
    data: (n_channels,)
    adjacency: (n_channels, n_channels)
    minnbchan: minimum significant neighbors required (default=1 for sparse montages)
    """
    # Match LIMO's increment logic exactly
    def compute_side(d):
        if np.max(d) == 0:
            return np.zeros_like(d)
            
        data_range = np.max(d) - np.min(d)
        if data_range > 1:
            precision = round(data_range / dh)
            if precision > 200:
                increment = data_range / 200.0
            else:
                increment = data_range / float(precision)
        else:
            increment = data_range * dh
            
        if increment == 0:
            return np.zeros_like(d)
            
        tfce_acc = np.zeros_like(d)
        # LIMO starts from min(d) and goes to max(d)
        thresholds = np.arange(np.min(d), np.max(d) + increment, increment)
        
        for h in thresholds:
            onoff = d > h
            if not np.any(onoff): continue
            
            labels, num = limo_findcluster_python(onoff, adjacency, minnbchan)
            
            extent_map = np.zeros_like(d)
            for c in range(1, num + 1):
                mask = labels == c
                extent = np.sum(mask)
                extent_map[mask] = extent
                
            tfce_acc += (extent_map**E) * (h**H) * increment
        return tfce_acc

    # LIMO doesn't always split if data is mostly positive or negative?
    # Actually, LIMO type 2 uses min(data) > 0 check (line 352)
    if np.min(data) > 0:
        return compute_side(data)
    else:
        # LIMO splits into positive and negative peaks
        pos_data = np.where(data > 0, data, 0)
        neg_data = np.where(data < 0, np.abs(data), 0)
        return compute_side(pos_data) + compute_side(neg_data)
