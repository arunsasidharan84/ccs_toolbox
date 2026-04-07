
def compute(EEGdata,maxorder = 30,minstd = 0.5,maxiter = 100,srate = [],show = False):
    """
    To perform Empirical Mode Decomposition (EMD) on EEG data 
    
    Usage: [imfs]  = abrl_EMD(EEGdata)
           [imfs,imf_residualvar,imf_freq,imf_env]  = abrl_EMD(EEGdata,maxorder,minstd,maxiter,srate,show)
    
    Inputs:
          EEGdata  = [nchan x nsamples]
       
    Optional Inputs:
          maxorder  = maximum order for EMD (default is 30)
          minstd    = minimum standard deviation in signal between sifting 
                      iterations as a stopping critia (default is .5)
          maxiter   = maximum number of sifting iterations as a stopping critia
                      (default is 100)
          srate     = sampling rate (If provided will convert wave interval 
                      into frequency)
          show      = If true, will plot the sifting process 
                      [Useful for learning/Troubleshooting] 
    
    Outputs:
          imfs                = Intrinsic Mode Function (IMF) time series 
                                [nchan x nmodes x nsamples]
          imf_residualvar     = explained residual variance by each IMF
                                [nchan x nmodes]
          imf_freq            = interval between oscillations (in samples; in Hz if srate provided)
                                [nchan x nmodes x 3] (lower, median and upper limits per mode)
          imf_env             = instantaneous amplitude of each mode
                                [nchan x nmodes x nsamples]
    """
    
    ###########################################################################
    # DISCLAIMER: code is written based on Mike X Cohen's book titled 
    #   "Analyzing Neural Time Series Data"(MIT Press) Chapter 23
    #   Mike X Cohen (mikexcohen@gmail.com)
    # 
    ###########################################################################
    # Original author: 2014 Mike X Cohen (mikexcohen@gmail.com)
    # Modified author: Jul 2019; Arun Sasidharan, ABRL
    # 
    # Copyright (C) 2014 Mike X Cohen (mikexcohen@gmail.com)
    # Copyright (C) 2018 ABRL, Axxonet System Technologies Pvt Ltd., Bengaluru, India
    # 
    # Permission is hereby granted, free of charge, to any person obtaining a 
    # copy of this software and associated documentation files (the "Software"), 
    # to deal in the Software without restriction, including without limitation 
    # the rights to use, copy, modify, merge, publish, distribute, sublicense, 
    # and/or sell copies of the Software, and to permit persons to whom the 
    # Software is furnished to do so, subject to the following conditions:
    # 
    # The above copyright notice and this permission notice shall be included 
    # in all copies or substantial portions of the Software.
    # 
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
    # OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
    # THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR 
    # OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
    # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
    # OTHER DEALINGS IN THE SOFTWARE.
    ###########################################################################

    import numpy as np
#    import matplotlib
#    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import UnivariateSpline
    
    # <codecell>
    
    # Get details of EEG data
    [nchans,npnts]  = EEGdata.shape;
    time            = np.arange(npnts);
    
    # Initialise the variables
    imfs                = np.zeros((nchans,maxorder+1,npnts));
    imf_residualvar     = np.zeros((nchans,maxorder+1));
    imf_freq            = np.zeros((nchans,maxorder+1,3));
    imf_env             = np.zeros((nchans,maxorder+1,npnts));
    imforders           = [];
    
    for chan_no in range(nchans):
        imfsignal = np.squeeze(EEGdata[chan_no,:]);
        origsignal = imfsignal;
        imforder = 0;
        stop     = False;
        
        # <codecell> Loop over IMF order
        while not stop:           
    
            # Initializations
            standdev = 10;
            numiter  = 0;
            signal   = imfsignal;
            SD = [];
                 
            # "Sifting Process" for each IMF
            # means iteratively identifying peaks/troughs in the
            # signal, interpolating across peaks/troughs, and then recomputing
            # peaks/troughs in the interpolated signal. Sifting ends when
            # variance between interpolated signal and previous sifting
            # iteration is minimized 
            while standdev>minstd and numiter<maxiter:
                
                # identify local min/maxima
                localmin  = np.unique(np.hstack((0, np.flatnonzero(np.diff(np.sign(np.diff(signal)))>0), npnts-1)));
                localmax  = np.unique(np.hstack((0, np.flatnonzero(np.diff(np.sign(np.diff(signal)))<0), npnts-1)));
                
                # Check if there are enough local min/maxima detected for computation
                if len(localmin) < 4 or len(localmax) < 4:
                    stop = True;
                    break;  
                
                # Create envelopes by interpolating min/max points (Spline interpolation)              
                FL = UnivariateSpline(localmin, signal[localmin], s=0, k=3);
                env_lower = FL(time);
                
                FU = UnivariateSpline(localmax, signal[localmax], s=0, k=3);
                env_upper = FU(time);
                
                # Plot the sifting step (For TROUBLESHOOTING only)
                if show:
                    if numiter == 0 and chan_no == 0 and imforder == 0:
#                        if 'fig' in locals():
#                            plt.close(fig);
                        fig = plt.figure(1)
                        ax  = fig.add_subplot(1, 1, 1)  
                        plt.ion()
                        plt.show()
                        
                    fig.clf()
                    plt.plot(time, env_lower, 'b')
                    plt.plot(time, env_upper, 'r')
                    plt.plot(time, signal, 'k')
                    plt.title("Chan: %i || IMF No: %i || Sifting step: %i" % (chan_no+1,imforder+1,numiter+1));
                    plt.draw()
                    plt.pause(0.0005)
                
                                
                # Compute residual and standard deviation
                prevsig   = signal;
                signal    = signal - (env_lower+env_upper)/2;
                standdev  = np.sum([((np.square(prevsig-signal)) / (np.square(prevsig) + np.finfo(float).eps))]); # eps prevents NaN's

                SD.append(standdev);
                
                numiter = numiter+1;
                
            # Show the Sifting steps
#            if show:
#                plt.show()
            
            # Compute residual variance
            p               = np.polyfit(signal,origsignal,1);
            datafit         = p[0] * signal + p[1];
            dataresid       = origsignal - datafit;
            SSresid         = np.sum(np.square(dataresid));
            SStotal         = (len(origsignal)-1) * np.var(origsignal);
            imf_residualvar[chan_no,imforder]   = 100*(1 - SSresid/SStotal);            
            
            # imf is residual of signal and min/max average (already redefined as signal)
            imfs[chan_no,imforder,:] = signal;
            
            # Get imf frequency
            waveintervals = np.hstack((np.diff(localmin[1:-1]),np.diff(localmax[1:-1])));
            if waveintervals.size != 0: 
                imf_freq[chan_no,imforder,0] = np.quantile(waveintervals, 0.75);
                imf_freq[chan_no,imforder,1] = np.quantile(waveintervals, 0.50);
                imf_freq[chan_no,imforder,2] = np.quantile(waveintervals, 0.25);
            else:
                imf_freq[chan_no,imforder,0] = npnts;
                imf_freq[chan_no,imforder,1] = npnts;
                imf_freq[chan_no,imforder,2] = npnts;
            
            # Get imf instantaneous amplitude
            imf_env[chan_no,imforder,:] = env_upper;
            
            imforder = imforder+1;
            
            # Residual is new signal
            imfsignal = imfsignal-signal;
            
            # Stop when only few points are left
            if len(localmin) < 6 or imforder>maxorder:
                stop = True;
            
        
        # <codecell> Include the residual signal
        imfs[chan_no,imforder,:] = imfsignal; # Residual signal becomes the last IMF
        imforders.append(imforder);
        
        # Compute residual variance of residual signal
        try:
            p               = np.polyfit(imfsignal,origsignal,1);
            datafit         = p[0] * imfsignal + p[1];
            dataresid       = origsignal - datafit;
            SSresid         = np.sum(np.square(dataresid));
            SStotal         = (len(origsignal)-1) * np.var(origsignal);
            imf_residualvar[chan_no,imforder]   = 100*(1 - SSresid/SStotal);
        except:
            imf_residualvar[chan_no,imforder]   = 0;
        
        #print("Completed channel: ",chan_no+1,"\n")
        
    # <codecell> Clear up the variables
    if 'fig' in locals():
        plt.close(fig);
    
    if 'srate' in locals():
        imf_freq = srate / (imf_freq+np.finfo(float).eps);
        
    imforders = np.asarray(imforders);
    
    # Merge the latter imfs to form uniform size
    nimfs = np.min((maxorder,np.min(imforders)));
    
    imfs[:,nimfs,:] = np.sum(imfs[:,nimfs:None,:],axis = 1);
    imfs = np.delete(imfs,np.s_[nimfs+1:None],1);
        
    imf_residualvar[:,nimfs] = np.sum(imf_residualvar[:,nimfs:None],axis = 1);
    imf_residualvar = np.delete(imf_residualvar,np.s_[nimfs+1:None],1);
        
    imf_freq[:,nimfs,:] = np.sum(imf_freq[:,nimfs:None,:],axis = 1);
    imf_freq = np.delete(imf_freq,np.s_[nimfs+1:None],1);
        
    imf_env[:,nimfs,:] = np.sum(imf_env[:,nimfs:None,:],axis = 1);
    imf_env = np.delete(imf_env,np.s_[nimfs+1:None],1);
    
    
    return imfs, imf_residualvar, imf_freq, imf_env