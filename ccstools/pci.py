'''
DISCLAIMER: Implements the PCI algorithm from
Casali, Adenauer G, Olivia Gosseries, Mario Rosanova, Mélanie Boly, Simone Sarasso, 
Karina R Casali, Silvia Casarotto, et al. “A Theoretically Based Index of Consciousness 
Independent of Sensory Processing and Behavior.” Science Translational Medicine 5, 
no. 198 (August 2013): 198ra105-198ra105. doi:10.1126/scitranslmed.3006294.

Original authors: 2016 Leonardo Barbosa (leonardo.barbosa@usp.br) &
					   Thierry Nieus (thierrynieus@gmail.com)
Modified author: Mar 2020; Arun Sasidharan, CCS, NIMHANS

Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
OTHER DEALINGS IN THE SOFTWARE.
'''

import numpy as np
from bitarray import bitarray
from scipy.stats import trim_mean,iqr

def calculate_pci_lower(D):
    '''
        Computes a lower bound of the PCI for the binary matrix D. 
        
        The PCI is computed on the ordered matrix D (i.e. the channels are ranked based on the evoked activity).  
        A speed up in the calculation is achieved removing the zero-rows (see Casali's MATLAB code).
    '''
    global ct 
    Irank=np.sum(D,axis=1).argsort()
    S=D[Irank,:].sum(axis=1)
    Izero=np.where(S==0)[0]
    number0removed = D.shape[1]*len(Izero)
    Dnew = D[Irank,:][Izero[-1]+1:,:]
    a = lz_complexity_2D(Dnew)

    nValues=float(D.shape[0]*D.shape[1])
    p1=np.sum(D)/nValues
    p0=1-p1

    N=np.log2(nValues)/nValues
    H=-p1*np.log2(p1)-p0*np.log2(p0)
    PCI=N*np.array(ct)/H
    return PCI

def calculate(D):
    return lz_complexity_2D(D) / pci_norm_factor(D)

def pci_norm_factor(D):

    L = D.shape[0] * D.shape[1]
    p1 = sum(1.0 * (D.flatten() == 1)) / L
    p0 = 1 - p1
    H = -p1 * np.log2(p1) -p0 * np.log2(p0)

    S = (L * H) / np.log2(L)

    return S

def lz_complexity_2D(D):
    #global ct   # time dependent complexity 
    if len(D.shape) != 2:
        raise Exception('data has to be 2D!')

    # initialize
    (L1, L2) = D.shape
    c=1; r=1; q=1; k=1; i=1
    stop = False

    # convert each column to a sequence of bits
    bits = [None] * L2
    for y in range(0,L2):
        bits[y] = bitarray(D[:,y].tolist())

    # action to perform every time it reaches the end of the column
    def end_of_column(r, c, i, q, k, stop):
        r += 1
        if r > L2:
            c += 1
            stop = True
        else:
            i = 0
            q = r - 1
            k = 1
        return r, c, i, q, k, stop

    ct=[]

    # main loop
    while not stop:

        if q == r:
            a = i+k-1
        else:
            a=L1

        # binary search
        #d = bits[r-1][i:i+k]
        #e = bits[q-1][0:a]
        #found = not not e.search(d, 1)
        found = not not bits[q-1][0:a].search(bits[r-1][i:i+k], 1)

        ## rowlling window (3x slower)
        #d = D[i:i+k,r-1]
        #e = D[0:a,q-1]
        #found = np.all(_rolling_window(e, len(d)) == d, axis=1).any()

        if found:
            k += 1
            if i+k > L1:
                (r, c, i, q, k, stop) = end_of_column(r, c, i, q, k, stop)
                ct.append(c)     
        else:
            q -= 1
            if q < 1:
                c += 1
                i = i + k
                if i + 1 > L1:
                    (r, c, i, q, k, stop) = end_of_column(r, c, i, q, k, stop)
                    ct.append(c)      
                else:
                    q = r
                    k = 1
    return np.array(ct)

def extract(epochdata,epochtimes):
    ''' 
    extract and preprocess relevant data from epochs
	epoch should be as nTR x nCH x nTIME (num trials x num channels x num time steps)
    '''
    global nCH,nTIME,nTR                
    nTR, nCH, nTIME = epochdata.shape
    N0 = np.where(epochtimes < 0)[0][-1]         # index of last negative time point
    erp = trim_mean(epochdata, 0.1, axis=0)
    basecorr = trim_mean(erp[:,:N0],0.1, axis=1) # mean baseline of prestimulus activity       
    Norm = iqr(erp[:,:N0],axis=1)/1.35           # standard deviation of baseline of prestimulus activity       
    erp -= np.kron(np.ones((1,nTIME)),basecorr.reshape((nCH,1)))  # subtract baseline
    NUM = np.kron(np.ones((1,N0)),basecorr.reshape((nCH,1)))
    DEN = np.kron(np.ones((1,N0)),Norm.reshape((nCH,1)))
    return N0, Norm, erp, NUM, DEN

def bootstrap_old(epochdata,Norm,N0,NUM,DEN,Nboot,alpha):
    ''' 
    non-parametric bootstrap procedure:
        - arguments:    
            epochs = number of trials (nTR) x number of channels (nCH) x number of time steps (nTIME)
        - note: 
            all arguments come the "extract" procedure
    '''
    nTR, nCH, nTIME = epochdata.shape 
    epochdata 		= epochdata.transpose((1,2,0))
    randontrialsT   = np.random.randint(0,nTR,nTR)
    Bootstraps      = np.zeros((Nboot,N0))     
    for per in range(Nboot):
        ET = np.zeros((nCH,N0))              
        for j in range(nTR): 
            randomsampT = np.random.randint(0,N0,N0)    
            ET 			+= epochdata[:,randomsampT,randontrialsT[j]]        
        ET 					= (ET/nTR-NUM)/DEN # computes a Z-value 
        Bootstraps[per,:] 	= np.max(np.abs(ET),axis=0)       # maximum statistics in space
    # computes threshold for binarization depending on alpha value 
    Bootstraps      = np.sort(np.reshape(Bootstraps,(Nboot*N0)))
    calpha          = 1-alpha 
    calpha_index    = int(np.round(calpha*Nboot*N0))-1               
    TT              = Norm*Bootstraps[calpha_index]                    # computes threshold based on alpha set before 
    Threshold       = np.kron(np.ones((1,nTIME)),TT.reshape((nCH,1)))  # set the same threshold for each time point 
    return Threshold

def bootstrap_threshold(baselinedata,Nboot=500,alpha=0.01):
    ''' 
    non-parametric bootstrap based threshold:
        - arguments:    
            baselinedata = number of trials (nTR) x number of channels (nCH) x number of time steps (nTIME)
        - note: 
            all arguments come the "extract" procedure
    '''
    nTR, nCH, nTIME = baselinedata.shape     
    baselineerp     = trim_mean(baselinedata, 0.1, axis=0)
    baselinemean    = trim_mean(baselineerp,0.1, axis=1) # mean of baseline activity       
    baselinestd     = iqr(baselineerp,axis=1)/1.35       # standard deviation of baseline activity       
    baselineerp    -= np.kron(np.ones((1,nTIME)),baselinemean.reshape((nCH,1)))  # subtract baseline
    NUM = np.kron(np.ones((1,nTIME)),baselinemean.reshape((nCH,1)))
    DEN = np.kron(np.ones((1,nTIME)),baselinestd.reshape((nCH,1)))
    
    baselinedata    = baselinedata.transpose((1,2,0))
    randontrialsT   = np.random.randint(0,nTR,nTR)
    Bootstraps      = np.zeros((Nboot,nTIME))     
    for per in range(Nboot):
        ET = np.zeros((nCH,nTIME))              
        for j in range(nTR): 
            randomsampT = np.random.randint(0,nTIME,nTIME)    
            ET 		   += baselinedata[:,randomsampT,randontrialsT[j]]        
        ET 					= (ET/nTR-NUM)/DEN # computes a Z-value 
        Bootstraps[per,:] 	= np.max(np.abs(ET),axis=0)     # maximum statistics in space
    # computes threshold for binarization depending on alpha value 
    Bootstraps      = np.sort(np.reshape(Bootstraps,(Nboot*nTIME)))
    calpha          = 1-alpha 
    calpha_index    = int(np.round(calpha*Nboot*nTIME))-1               
    Threshold       = baselinestd*Bootstraps[calpha_index]  # computes threshold based on alpha set before 
    return Threshold
	
	
def smooth(y, box_pts):
	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth
	
	
def compute_erp_pci(epochdata,epochtimes,Threshold,poststim):
	# extract data from epochs
    #N0, Norm, erp, NUM, DEN = extract(epochdata,epochtimes)
    nTR, nCH, nTIME = epochdata.shape
    
    # determine significant activity matrix
    erp         = trim_mean(epochdata, 0.1, axis=0)
    Threshold   = np.kron(np.ones((1,nTIME)),Threshold.reshape((nCH,1)))  # set the same threshold for each time point 
    sigbins     = np.array(np.abs(erp)>Threshold,dtype=int) 
    
    # rank the activity matrix - use mergesort that yields same results of Matlab
    Irank       = np.argsort(np.sum(sigbins,axis=1),kind='mergesort') 
    sigbinrank  = np.copy(sigbins)
    sigbinrank  = sigbins[Irank,:]
	
	# compute PCI    
    N0          = np.where(epochtimes < 0)[0][-1]
    npoint_art  = 4 # number of data points of the artifact
    nind        = len(list(x for x in epochtimes if poststim[0] < x < poststim[1])) - npoint_art # number of data points to take
    ind_bin     = N0 + npoint_art + np.arange(nind)
    sigbinrank_ind  = sigbinrank[:,ind_bin]
    erp_complexity 	= lz_complexity_2D(sigbinrank_ind)
    bin_norm    	= pci_norm_factor(sigbinrank_ind)
    erp_complexity 	= erp_complexity/bin_norm
    erp_pci 		= erp_complexity[-1]
    return erp_pci, erp_complexity, ind_bin, sigbinrank 