
def importBessWaveform(filename):
    """
    To import EEG data exported as ASCII from BESS's Waveform Plot
    
    Usage: [EEGdata,srate,chanlist] = importBessWaveform(filename)
    
    Inputs:
          filename = Name of .text file exported from BESS's Waveform Plot
    
    Outputs:
          EEGdata  = [nchan x nsamples]
          srate    = sampling rate
          chanlist = string array of channel list
    """
    
    ###########################################################################
    # Original author: Aug 2018; Arun Sasidharan, ABRL
    #
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
    
    fileID = open(filename,"r");
    if fileID.mode == 'r':
        
        # Read the file as separate lines
        contents = fileID.readlines()
        
        nrows = len(contents);
        
        # Get the row numbers of key header levels
        headerindx      = [contents.index(i) for i in contents if "BESS Text Export" in i][0]
        recordingindx   = [contents.index(i) for i in contents if "SamplingRate" in i][0]
        
        # Extract sampling rate from the header
        srate = float(contents[recordingindx][contents[recordingindx].index("SamplingRate = ")+15:contents[recordingindx].index(", Sample")]);
                      
        # Extract the channel list (without the contents in brackets)
        chanlist = [];
        for row_no in range(headerindx+1,recordingindx):
            chanlist.append(contents[row_no][0:contents[row_no].index("(")])
        
        # Extract the channel values (tab-separated)
        EEGdata = [];
        for row_no in range(recordingindx+2,nrows):
            EEGdata.append(np.fromstring(contents[row_no], dtype=np.float, sep='\t'));
        EEGdata = np.transpose(np.asarray(EEGdata));
        nchans,npts = EEGdata.shape;
        
        # Extract the time range
        times = contents[headerindx+1][contents[headerindx+1].index("(")+1:contents[headerindx+1].index(")")-1];
        starttime,endtime = np.fromstring(times, dtype=np.float, sep='-')       
        times = [];
        times.append(float(starttime));
        for timepnt_no in range(npts-1):
            times.append(times[timepnt_no] + 1/srate)
        times = np.asarray(times);
        
        
    return EEGdata, srate, chanlist, times