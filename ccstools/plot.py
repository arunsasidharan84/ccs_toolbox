

def waveform(EEGdata,srate,chanlist = [],scale = [],fig_ID = [],
             times = [],cursor = 0,color = 'blue',linewidth = 1,alpha = 0.8):
    """
    To plot EEG data as multi-line waveform snapshot
    
    Usage:    waveform(EEGdata,srate,chanlist,scale,fig_ID,times,cursor)
         OR   waveform(EEGdata,srate)
    
    Inputs:
      EEGdata  = [nchan x nsamples]
      srate    = sampling rate     
    
    Optional Input
      chanlist = string array of channel list
      scale    = scalar value that determins the spread of the waveforms
                  [Default = Auto]
      fig_ID   = figure ID
      times    = vector with time values to plot
      cursor   = sample point at which a vertical cursor need to be drawn
                 0 -> no cursor [Default] 
      color    = colour of waveforms ["blue"] 
    """
    ###########################################################################
    # Original author: Oct 2018; Arun Sasidharan, ABRL
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
    import matplotlib.pyplot as plt

    
    #%% Check the inputs
    [nchans,npnts]  = EEGdata.shape

    if not chanlist:
        chanlist = np.array(range(1,nchans+1), dtype=str);
    
    if not scale:
        scale = abs(np.min(EEGdata) - np.max(EEGdata));
        
    if not fig_ID:
        fig, fig_ID    = plt.subplots(1,figsize=(14,8))
        # plt.figure(); # Open a new figure
        
    if not any(times):
        times = np.array(range(1,npnts+1))/srate;
        
    if len(color) == 1 or len(color) != nchans:
        color = [color[0]]*nchans
        
    
    #%% Add waveforms to difference y-levels [leave one level space at top and bottom for better visibility]

    chanlabel_pos       = np.zeros((1,nchans+2));
    chanlabel_pos[0,0]  = (nchans+1)*scale;
    for chan in range(nchans):
        yfactor = np.ones((1,npnts))*(nchans-chan)*scale;
        chanlabel_pos[0,chan+1] = (nchans-chan)*scale;
        fig_ID.plot(times[range(npnts)],np.transpose(EEGdata[chan,:] + yfactor),
                    color=color[chan],linewidth=linewidth,alpha=alpha);
        
    #%% Change the y-label to channel names or channel numbers   
    y_locs     =  np.transpose(chanlabel_pos);
    y_labels   =  np.hstack(('',chanlist,''));
    fig_ID.set_yticks(y_locs[:,0],y_labels);

    #%% Beautify the plot
    fig_ID.set_xlabel('Time (s)');
    fig_ID.set_xlim([np.min(times),np.max(times)]);
    fig_ID.set_ylim([np.min(chanlabel_pos),np.max(chanlabel_pos)]);
    fig_ID.set_title("SR: %iHz || Scaling: %f" % (srate,scale));
    
    #%% Add a vertical cursor if specified
    if cursor != 0:
        cursor_time = times[0,cursor-1]
        fig_ID.plot((cursor_time,cursor_time),
                    (np.min(chanlabel_pos),np.max(chanlabel_pos)),
                    '-.r',linewidth=linewidth)
		
		
		
def hypnoplot(hypnodata,hypn_srate = 1,startindx = 0,color = "Grey",timelim = [],ax=[]):
    """
    Purpose: Plots sleep stage list into hypnogram

    """
    ###########################################################################
    # Original author: 17 Feb 2020; by Dr Arun Sasidharan, CCS, NIMHANS
    # 
    # Copyright (C) 2020 CCS, NIMHANS, Bengaluru, India
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
    import matplotlib.pyplot as plt

    timefactor  = 60*60*hypn_srate
    
    # Re-order the values to flip the hypnogram to standard format
    hypnodata2 = 8 - hypnodata
    hypnodata2[hypnodata2 == 8] = 9
    hypnodata2[hypnodata2 == 4] = 8
    
    # Get the time points based on epoch length or sampling rate
    timepts = np.arange(startindx,len(hypnodata)+startindx)/timefactor
    
    # Plot the basic stages as a step graph (if sampling rate is small) else line graph 
    # (Step plots take longer time for lonh hypnodata)
    if not ax:
        ax = plt.gca()
    if hypn_srate > 1:
        ax.plot(timepts,hypnodata2,color = color,linewidth=1)
    else: 
        ax.step(timepts,hypnodata2,where = 'post',color = color,linewidth=1)
    
    # Keep only REM values and plot on top as a thicker line
    R_hypnodata                 = np.asfarray(hypnodata2)
    R_hypnodata[hypnodata2 < 8] = np.nan
    R_hypnodata[hypnodata2 > 8] = np.nan
    R_timepts                   = timepts.copy()
    R_timepts[hypnodata2 < 8]   = np.nan
    R_timepts[hypnodata2 > 8]   = np.nan
    ax.plot(R_timepts,R_hypnodata,color = color,linewidth=3)
    
    # Add aesthetics to the plot
    ax.set_yticks(np.arange(5,10))
    ax.set_yticks(np.arange(5,10),('N3','N2','N1','R','W'))
    if not timelim:
       		timelim = [timepts[0],np.ceil(timepts[-1])]
    ax.set_xlim(timelim[0],timelim[-1])
    ax.set_xlabel('Time (hours)')
    
    
def headmap(channel_values,chanlocs,
            plottype = 'filled',electrodetype = 'black',
            colorlims = 'absmax',fig_ID = [],chan2highlight = [],
            fillcolor = 'RdBu_r',filltype = 'cubic', contourcolor = 'white'):
    """
    Purpose: To generate headMap with colour-coded density map

    Inputs:
        channel_values 	= vector with channel values to plot
        chanlocs        = chanloc structure of EEGLAB

    Optional Inputs:
        plottype		= filling of plot
                         'filled' (Default)
                         'empty'
        electrodetype   = appearance of electrodes
                         'black' (Default)
                         'coloured'
                         'none'
        colorlims       = 'maxmin' OR 'absmax'
        chan2highlight  = vector of channel indices to highlight

    Outputs:
        figure_id 		= figure handle
        colorbar_id		= colorbar handle

    DISCLAIMER: The code is re-written from "topoplot" function of EEGLAB, with
                modifications and optimizations to allow plotting in Python.
    """
    ###########################################################################
    # Original author: Aug 1996; Colin Humphries & Scott Makeig, CNL / Salk Institute
    # Modified author: Apr 2017; Arun Sasidharan, ABRL
    # 
    # Copyright (C) 1996 Colin Humphries & Scott Makeig, CNL / Salk Institute, USA
    # Copyright (C) 2017 ABRL, Axxonet System Technologies Pvt Ltd., Bengaluru, India
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
    import matplotlib.pyplot as plt

    from scipy.interpolate import griddata
    from scipy.signal import convolve2d as conv2
    from matplotlib import patches 
    from scipy.ndimage.filters import generic_filter
    
    #%% Check the inputs        
    if not fig_ID:
        fig_ID = plt.figure() # Open a new figure
        
    #%% Default parameters
    rmax                = 0.50
    MAPLIMITS           = colorlims     # Options are 'absmax' OR 'maxmin'
    BACKCOLOR           = "white"       # Figure background
    GRID_SCALE          = 150           # grids size to plot map
    CIRCGRID            = 201           # number of angles to use in drawing circles
    CONTOURNUM          = 6             # number of contour levels to plot
    AXHEADFAC           = 1.3           # head to axes scaling factor
    CCOLOR              = contourcolor  # default contour color
    CWIDTH              = 0.25          # default contour linewidth
    BLANKINGRINGWIDTH   = 0.35          # width of the blanking ring 
    HEADRINGWIDTH       = 0.007         # width of the cartoon head ring
    HLINEWIDTH          = 2             # default linewidth for head, nose, ears
    HEADCOLOR           = [0,0,0]       # default head outline color (black)
    headrad             = 0.57          # default head outline radius
    EMARKER             = '.'           # mark electrode locations with small disks
    ECOLOR              = [0,0,0]       # default electrode color (black)
    EMARKERLINEWIDTH    = 1             # default edge linewidth for emarkers
    
    def pol2cart(degrees,magnitude):
        angle = np.pi/180*degrees   	# convert degrees to radians
        x = magnitude * np.cos(angle)
        y = magnitude * np.sin(angle)
        return x, y
    Theta           = np.concatenate(chanlocs.theta).astype(None)
    Radii           = np.concatenate(chanlocs.radius).astype(None)
    chanX,chanY     = pol2cart(Theta,Radii) # transform electrode locations from polar to cartesian coordinates
    x               = -chanY                # rotate cordinates
    y               = chanX                 # rotate cordinates

    #%% Squeeze electrode arc_lengths towards the vertex to plot all inside the head cartoon
    plotrad = min(1.0,max(Radii)*1.02)    # default: just outside the outermost electrode location
    plotrad = max(plotrad,0.5)            # default: plot out to the 0.5 head boundary
    if not plotrad:
        plotrad = 0.5
    squeezefac = rmax/plotrad
    
    if plottype == 'filled' and len(y)>1:
        x    = x*squeezefac
        y    = y*squeezefac 
        
        #%% Make sure outermost channel will be plotted just inside rmax
        xmin = min(-rmax,min(x)) 
        xmax = max(rmax,max(x))
        ymin = min(-rmax,min(y))
        ymax = max(rmax,max(y)) 
        
        #%% Spread data into a square grid
        xi = np.linspace(xmin,xmax,GRID_SCALE)   # x-axis description (row vector)
        yi = np.linspace(ymin,ymax,GRID_SCALE)   # y-axis description (row vector)
        
        zi = griddata((x, y), channel_values, (xi[None,:], yi[:,None]), method=filltype) # interpolate data
        zi[np.isnan(zi)] = 0
        zi = np.fliplr(np.flipud(zi))

        disk_r = 10
        disk_y,disk_x = np.ogrid[-disk_r: disk_r+1, -disk_r: disk_r+1]
        disk = disk_x**2 + disk_y**2 <= disk_r**2
        disk = disk.astype(float)
        Zi = conv2(zi, disk,mode='same')
        #Zi = generic_filter(zi, np.median, footprint = disk)
        #Zi = generic_filter(zi, np.mean, footprint = disk)
        #Zi = zi
        
        #%% Create 2-d grid of coordinates and function values, suitable for 3-d plotting
        Xi,Yi = np.meshgrid(xi,yi)

    	#%% Add a mask outside the plotting circle
        mask    = np.sqrt(Xi**2 + Yi**2) <= rmax # mask outside the plotting circle
        Zi[mask == 0]  = np.nan                  # mask non-plotting voxels with NaNs
    
    	#%% Set map limits
        if type(MAPLIMITS) == str and MAPLIMITS == 'absmax':
            amax = np.nanmax(np.nanmax(np.abs(Zi)))
            amin = -amax
        elif type(MAPLIMITS) == str and MAPLIMITS in ('maxmin','minmax'):
            amin = np.nanmin(np.nanmin(Zi))
            amax = np.nanmax(np.nanmax(Zi))
        elif type(MAPLIMITS) == list and type(MAPLIMITS[0]) in (int,float):
            amin = MAPLIMITS[0]
            amax = MAPLIMITS[1]
        else:
            print('unknown ''maplimits'' value.')
    
        # If only one value, then make sure colour map is spread around
        if amin == amax and (0 not in (amin,amax)):
            agap = 0.25*amin
            amin = np.nanmin((amin - agap),-np.spacing(1))
            amax = np.nanmax((amax + agap),np.spacing(1))
        
        #%% Plot the density map   
        ax = plt.gca()
        img_ID = plt.imshow(np.vstack([Xi[0,:],Yi[:,0],Zi]), 
                            cmap=fillcolor, extent=[xmin,xmax,ymin,ymax])
        plt.colorbar(fraction=0.05, pad=0.05, shrink=0.25)
        plt.clim(amin,amax)
        
        
        #%% Add countour lines
        chs = plt.contour(Xi,-Yi,Zi,CONTOURNUM,
                          colors=CCOLOR,linewidths=CWIDTH,alpha=0.8)
    
    #%% Plot filled ring to mask jagged grid boundary
    hin     = squeezefac*headrad*(1- HEADRINGWIDTH/2)   # inner head ring radius
    rwidth  = BLANKINGRINGWIDTH                         # width of blanking outer ring
    rin     =  rmax*(1-rwidth/2)                        # inner ring radius
    if hin>rin:
    	rin = hin                                       # dont blank inside the head ring
    
    circ    = np.linspace(0,2*np.pi,CIRCGRID)
    rx      = np.sin(circ) 
    ry      = np.cos(circ) 
    ringx   = np.concatenate([np.concatenate([rx,rx[:1]])*(rin+rwidth),np.concatenate([rx,rx[:1]])*rin])
    ringy   = np.concatenate([np.concatenate([ry,ry[:1]])*(rin+rwidth),np.concatenate([ry,ry[:1]])*rin])

    # Paint the border with background color   
    patch = patches.Circle((0,0), radius=rin, transform=ax.transData)
    img_ID.set_clip_path(patch)
    
    #%% Plot head ring
    headx = np.concatenate([rx,rx[:1]])*hin
    heady = np.concatenate([ry,ry[:1]])*hin
    ringh = plt.plot(headx,heady,color = HEADCOLOR,
                     linewidth = HLINEWIDTH)    
    
    #%% Plot ears and nose
    base  = rmax - 0.0046
    basex = 0.18*rmax	# nose width
    tip   = 1.15*rmax 
    tiphw = 0.04*rmax	# nose tip half width
    tipr  = 0.01*rmax	# round the nose tip
    q     = 0.04        # lengthen the ear
    EarX  = np.array([.497-.005, .510, .518, .5299, .5419, .54, .547, .532, .510, .489-.005]) # rmax = 0.5
    EarY  = np.array([q+.0555, q+.0775, q+.0783, q+.0746, q+.0555, -.0055, -.0932, -.1313, -.1384, -.1199])
    sf    = headrad/plotrad   # squeeze the model ears and nose by this factor
    # Plot nose
    plt.plot(np.array([basex,tiphw,0,-tiphw,-basex])*sf,
             np.array([base,tip-tipr,tip,tip-tipr,base])*sf,
          color = HEADCOLOR,linewidth = HLINEWIDTH)
    # Plot left ear 
    plt.plot(EarX*sf,EarY*sf,color = HEADCOLOR,linewidth = HLINEWIDTH)    
    # Plot right ear
    plt.plot(-EarX*sf,EarY*sf,color = HEADCOLOR,linewidth = HLINEWIDTH) 
    
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-0.525,0.525)
    ax.set_ylim(-0.505,0.535)
    
    #%% Add electrode markers
    if len(y)>=160:
        EMARKERSIZE = 3
    elif len(y)>=128:
        EMARKERSIZE = 3
    elif len(y)>=100:
        EMARKERSIZE = 3
    elif len(y)>=80:
        EMARKERSIZE = 4
    elif len(y)>=64:
        EMARKERSIZE = 5
    elif len(y)>=48:
        EMARKERSIZE = 6
    elif len(y)>=32: 
        EMARKERSIZE = 8
    elif len(y)>=20: 
        EMARKERSIZE = 12
    elif len(y)<20: 
        EMARKERSIZE = 16
        
    if len(y) > 1:        
        if electrodetype == 'coloured':
           EMARKERSIZE = EMARKERSIZE*1.5            
           plt.plot(-x(channel_values>0),y(channel_values>0),marker = EMARKER, 
                    color = 'red', markersize = EMARKERSIZE,
                    linewidth = EMARKERLINEWIDTH, linestyle = '')
           plt.plot(-x(channel_values<0),y(channel_values<0),marker = EMARKER, 
                    color = 'blue', markersize = EMARKERSIZE,
                    linewidth = EMARKERLINEWIDTH, linestyle = '') 
        
        elif electrodetype == 'black':
           plt.plot(-x,y,marker = EMARKER, 
                    color = ECOLOR, markersize = EMARKERSIZE*0.5,
                    linewidth = EMARKERLINEWIDTH, linestyle = '')                
                   

    if len(chan2highlight) > 1:
        if electrodetype == 'none':
            plt.plot(-x(chan2highlight),y(chan2highlight), 
                     marker = EMARKER, color = ECOLOR, markersize = EMARKERSIZE*2,
                     linewidth = EMARKERLINEWIDTH+1, linestyle = '')
        else:
            plt.plot(-x(chan2highlight),y(chan2highlight), 
                     marker = 'o', color = ECOLOR, markersize = EMARKERSIZE,
                     linewidth = EMARKERLINEWIDTH+1, linestyle = '')
    
    #fig_ID.patch.set_facecolor('xkcd:%s',BACKCOLOR)
