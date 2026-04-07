# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:12:53 2023

@author: Arun Sasidharan
"""
#%% Import Libraries
import numpy as np
import pandas as pd 
import time
import datetime

class xampl10:
    
    def __init__(self, channames=["ECG"], port=""):
        
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        
        if port == "":
            import platform
            if platform.system() == 'Windows':
                print('Running on Windows')
                # import ftd2xx as ftd
                import serial.tools.list_ports
            elif platform.system() == 'Linux':
                print('Running on Linux')
                import pylibftdi as ftdi
                print(ftdi.Driver().list_devices())
            else:
                print('Unknown operating system')
                time.sleep(5)
                exit()
                
            try:
                    
                if platform.system() == 'Windows':
                    ports = serial.tools.list_ports.comports()
                    for portx, descx, hwidx in sorted(ports):
                        if 'AXXBLE00002A' in format(hwidx):
                            port = format(portx)
                            print("{}: {} [{}]".format(portx, descx, hwidx))
                    
                elif platform.system() == 'Linux':
                    device_ftd = ftdi.Device()                
                    port = "COM"+str(device_ftd.getComPortNumber())   
                    print(device_ftd.getDeviceInfo())
    
            except Exception as e:
                print(f'Check if xAMP-L10 connected: {e}')
                raise Exception(e)
            
            # if port == "":
            #     device_ftd = ftd.open(0)
            #     port = "COM"+str(device_ftd.getComPortNumber())
            #     device_ftd.close()
            
        """Get Data streaming Board parameters"""
        BoardShim.enable_dev_board_logger()
        self.params              = BrainFlowInputParams()
        self.board_id            = BoardIds.EPIDOME_BOARD.value
        self.params.serial_port  = port
        self.sf                  = BoardShim.get_sampling_rate(self.board_id)
        self.channames           = channames
        self.chanindx            = np.array(range(len(channames))) 
        self.board               = BoardShim(self.board_id, self.params)
        
        # Connect for Data streaming
        if self.board.is_prepared():
            self.board.release_all_sessions()
        
        time.sleep(1)
        self.board.prepare_session()
        print('xAMP-L10 connected')   
    
    def read(self, duration=5, show=False):
        """Collect data stream"""
        
        from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
        
        self.duration = duration
        self.board.start_stream()
        print('xAMP-L10 streaming.....')
                    
        if show:
            from ccstools.plot import waveform
            import matplotlib.pyplot as plt
            
            ##############################
            # Plot            
            fig, axs = plt.subplots(1,figsize=(10,8))
            start_time = time.perf_counter()
            time.sleep(1)
            elapsedtime = 0
            plt.show()
            try:
                while elapsedtime < duration:    
                    
                    # Collect the data
                    newest_data = self.board.get_current_board_data(2*self.sf)[self.chanindx]*1e6
                    
                    # Get the current data index
                    newest_samplenum = self.board.get_board_data_count()
                    times = (np.arange(newest_data.shape[-1])-newest_data.shape[-1]+newest_samplenum)/self.sf;
                   
                    # Filter the data
                    DataFilter.detrend(newest_data[0], DetrendOperations.CONSTANT);
                    DataFilter.perform_bandstop(newest_data[0], self.sf, 48, 52, 2,FilterTypes.BUTTERWORTH.value, 0)
                    DataFilter.perform_bandpass(newest_data[0], self.sf, 0.5, 35, 2,FilterTypes.BUTTERWORTH.value, 0)
                    
                    elapsedtime = (time.perf_counter() - start_time)
                    axs.cla()
                    waveform(newest_data,self.sf,self.channames,times=times,fig_ID=axs,scale=100,color="black")
                    plt.title('Data collection %2.1f%%' %(100*elapsedtime/duration))
                    plt.draw()
                    plt.pause(0.5)
            except KeyboardInterrupt:
                print("STOPPED BY USER by pressing CTRL + C.")
            finally:                
                plt.close(fig)   
                # Stop Data streaming
                self.board.stop_stream() 
                print("Stopping Data streaming")
            ##############################
        else:
            time.sleep(duration)
            # Stop Data streaming
            self.board.stop_stream() 
            print("Stopping Data streaming")
        
        # Get the data
        channel_data = self.board.get_board_data()[self.chanindx]*1e6
        
        return (channel_data)
            
    def findheartrate(self, channel_data,show=False):
        """Compute heart rate"""
        import neurokit2 as nk
        from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
        
        # Filter the data    
        DataFilter.detrend(channel_data[0], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(channel_data[0], self.sf, 1.0, 30.0, 2,FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(channel_data[0], self.sf, 48.0, 52.0, 2,FilterTypes.BUTTERWORTH.value, 0)
        
        # channel_data[0] = nk.ecg_clean(channel_data[0], sampling_rate=self.sf)
        
        # Compute heart rate
        # rpeak_bool, info = nk.ecg_peaks(channel_data[0], sampling_rate=self.sf, correct_artifacts=True)
        # heartrate = np.mean(nk.signal_rate(rpeak_bool,self.sf))
        # rpeaks = info['ECG_R_Peaks']
        
        signals, info = nk.bio_process(channel_data[0], sampling_rate=self.sf)
        heartrate = signals.ECG_Rate.mean()
        rpeaks = info['ECG_R_Peaks']
        
        print('Heart Rate is %0.2f bpm' %(heartrate))
        
        #%%
        if show:
            import matplotlib.pyplot as plt
            
            times = np.arange(0, len(channel_data[0]))/self.sf  # Create a time vector (seconds)
    
            fig, axs = plt.subplots(1, 1, figsize=(13, 5), sharex=True)
    
            # Raw ECG signal and R peaks
            # --------------------------
            axs.plot(times, channel_data[0], color='#c44e52', linewidth=1, label='ECG signal')
            axs.scatter(times[rpeaks], channel_data[0][rpeaks], color='gray', edgecolor="k", alpha=.6, label="R peaks")
            axs.set_ylabel('ECG (mV)')
            axs.set_xlabel('Time (s)')
            axs.legend()
        
        return (heartrate,rpeaks)
    
    def findhrv(self, channel_data,show=False,filesave=False, filename='ECG'):
        """Compute heart rate variability"""
        
        import neurokit2 as nk

        
        signals, __ = nk.bio_process(channel_data[0], sampling_rate=self.sf)
        # hrv_all = nk.hrv(signals, sampling_rate=self.sf,show=show)
        hrv_freq = nk.hrv_frequency(signals, sampling_rate=self.sf, show=show, normalize=True,psd_method="multitapers")
        # hrv_nonlinear = nk.hrv_nonlinear(signals, sampling_rate=self.sf, show=show)
        # hrv = pd.concat([hrv_freq,hrv_nonlinear],axis=1) 
        hrv = pd.concat([hrv_freq],axis=1) 
        
        if filesave:
            # Save
            hrvfname = (filename +
                '_' + 
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") +
                '.csv')
            hrv.to_csv(hrvfname)
        
        return (hrv)
        
    def save2edf(self, channel_data, filename='ECG', markerdata=None):
        """Save the data as EDF"""

        import pyedflib
        
        # Set file name
        edffname = (filename +
            '_' + 
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") +
            '.edf')

        # Get channel info
        chanlist = self.channames
        # chanlist.append('Timesync') # First channel after EEG is Time sync
        # chanlist.append('Marker') # Second channel after EEG is Marker
        channelinfo = pyedflib.highlevel.make_signal_headers(
            chanlist, dimension='uV', sample_frequency=self.sf,
            physical_min=-4000.0, physical_max=4000.0)
        
        if markerdata is not None:
            # Create the Annotation in format [
            # [timepoint, duration, description], [...]]
            marker_timepnts  = np.where(markerdata > 0)[0] 
            annotations = []
            for j in range(len(marker_timepnts)):
                annotations.append([
                    marker_timepnts[j]/self.sf,
                    0,
                    str(int(markerdata[marker_timepnts[j]]))])
            header = {'annotations':annotations}
        else:
            header = None

        # Write as EDF
        if hasattr(self,'duration'):
            channel_data = channel_data[:,:int(self.duration*self.sf)]
        else:
            channel_data = channel_data
        pyedflib.highlevel.write_edf(
            edffname,
            signals=channel_data,
            signal_headers=channelinfo,
            header=header,
            file_type=1)
    
    def close(self):
        """Release the Board"""
        self.board.release_session()
        print('xAMP-L10 disconnected') 




class museS:
    
    def __init__(self, 
                 channames=['TP9', 'Fp1', 'Fp2', 'TP10'],
                 chanindx=[1,2,3,4]):
        
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds,BrainFlowPresets

        """Get Data streaming Board parameters"""
        BrainFlowPresets(0)
        BoardShim.enable_dev_board_logger()
        self.params              = BrainFlowInputParams()
        self.board_id            = BoardIds.MUSE_S_BOARD.value
        self.sf                  = BoardShim.get_sampling_rate(self.board_id)
        if channames == "":
            self.channames       = BoardShim.get_eeg_names(self.board_id)
        else:
            self.channames       = channames
        if chanindx == []:
            self.chanindx        = BoardShim.get_eeg_channels(self.board_id)
        else:
            self.chanindx        = np.array(chanindx)                 
        self.board               = BoardShim(self.board_id, self.params)
        
        # Connect for Data streaming
        if self.board.is_prepared():
            self.board.release_all_sessions()
        
        time.sleep(1)
        self.board.prepare_session()
        print('Muse-S connected')   
    
    def read(self, duration=5, show=False, plotchans=[], plotscale=100):
        """Collect data stream"""
        
        from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
        
        self.duration = duration
        self.board.start_stream()
        print('Muse-S streaming.....')
                    
        if show:
            from ccstools.plot import waveform
            import matplotlib.pyplot as plt
            
            ##############################
            # Plot            
            
            if len(plotchans) == 0:
                plotchans = self.chanindx-1
            else:
                plotchans = np.array(plotchans)
                
            fig, axs = plt.subplots(1,figsize=(10,8))
            start_time = time.perf_counter()
            time.sleep(1)
            elapsedtime = 0
            plt.show()
            try:
                while elapsedtime < duration:    
                    
                    # Collect the data
                    newest_data = self.board.get_current_board_data(2*self.sf)[plotchans]#*1e6
                    
                    # Get the current data index
                    newest_samplenum = self.board.get_board_data_count()
                    times = (np.arange(newest_data.shape[-1])-newest_data.shape[-1]+newest_samplenum)/self.sf;
                   
                    # Filter the data
                    for chan_no in range(len(plotchans)):                        
                        DataFilter.detrend(newest_data[chan_no], DetrendOperations.CONSTANT);
                        DataFilter.perform_bandstop(newest_data[chan_no], self.sf, 48, 52, 2,FilterTypes.BUTTERWORTH.value, 0)
                        DataFilter.perform_bandpass(newest_data[chan_no], self.sf, 0.5, 35, 2,FilterTypes.BUTTERWORTH.value, 0)
                        
                    elapsedtime = (time.perf_counter() - start_time)
                    axs.cla()
                    import pdb;pdb.set_trace()
                    waveform(newest_data,self.sf,np.array(self.channames)[plotchans],times=times,fig_ID=axs,scale=plotscale,color="black")
                    plt.title('Data collection %2.1f%%' %(100*elapsedtime/duration))
                    plt.draw()
                    plt.pause(0.5)
            except KeyboardInterrupt:
                print("STOPPED BY USER by pressing CTRL + C.")
            finally:                
                plt.close(fig)   
                # Stop Data streaming
                self.board.stop_stream() 
                print("Stopping Data streaming")
            ##############################
        else:
            time.sleep(duration)
            # Stop Data streaming
            self.board.stop_stream() 
            print("Stopping Data streaming")
        
        # Get the data
        channel_data = self.board.get_board_data()[self.chanindx]*1e6
        
        return (channel_data)
            
    def findheartrate(self, channel_data,show=False):
        """Compute heart rate"""
        import neurokit2 as nk
        from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
        
        # Filter the data    
        DataFilter.detrend(channel_data[0], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(channel_data[0], self.sf, 1.0, 30.0, 2,FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(channel_data[0], self.sf, 48.0, 52.0, 2,FilterTypes.BUTTERWORTH.value, 0)
        
        # channel_data[0] = nk.ecg_clean(channel_data[0], sampling_rate=self.sf)
        
        # Compute heart rate
        # rpeak_bool, info = nk.ecg_peaks(channel_data[0], sampling_rate=self.sf, correct_artifacts=True)
        # heartrate = np.mean(nk.signal_rate(rpeak_bool,self.sf))
        # rpeaks = info['ECG_R_Peaks']
        
        signals, info = nk.bio_process(channel_data[0], sampling_rate=self.sf)
        heartrate = signals.ECG_Rate.mean()
        rpeaks = info['ECG_R_Peaks']
        
        print('Heart Rate is %0.2f bpm' %(heartrate))
        
        #%%
        if show:
            import matplotlib.pyplot as plt
            
            times = np.arange(0, len(channel_data[0]))/self.sf  # Create a time vector (seconds)
    
            fig, axs = plt.subplots(1, 1, figsize=(13, 5), sharex=True)
    
            # Raw ECG signal and R peaks
            # --------------------------
            axs.plot(times, channel_data[0], color='#c44e52', linewidth=1, label='ECG signal')
            axs.scatter(times[rpeaks], channel_data[0][rpeaks], color='gray', edgecolor="k", alpha=.6, label="R peaks")
            axs.set_ylabel('ECG (mV)')
            axs.set_xlabel('Time (s)')
            axs.legend()
        
        return (heartrate,rpeaks)
    
    def findhrv(self, channel_data,show=False,filesave=False, filename='ECG'):
        """Compute heart rate variability"""
        
        import neurokit2 as nk

        
        signals, __ = nk.bio_process(channel_data[0], sampling_rate=self.sf)
        # hrv_all = nk.hrv(signals, sampling_rate=self.sf,show=show)
        hrv_freq = nk.hrv_frequency(signals, sampling_rate=self.sf, show=show, normalize=True,psd_method="multitapers")
        # hrv_nonlinear = nk.hrv_nonlinear(signals, sampling_rate=self.sf, show=show)
        # hrv = pd.concat([hrv_freq,hrv_nonlinear],axis=1) 
        hrv = pd.concat([hrv_freq],axis=1) 
        
        if filesave:
            # Save
            hrvfname = (filename +
                '_' + 
                datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") +
                '.csv')
            hrv.to_csv(hrvfname)
        
        return (hrv)
        
    def save2edf(self, channel_data, filename='ECG', markerdata=None):
        """Save the data as EDF"""

        import pyedflib
        
        # Set file name
        edffname = (filename +
            '_' + 
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") +
            '.edf')

        # Get channel info
        chanlist = self.channames
        # chanlist.append('Timesync') # First channel after EEG is Time sync
        # chanlist.append('Marker') # Second channel after EEG is Marker
        channelinfo = pyedflib.highlevel.make_signal_headers(
            chanlist, dimension='uV', sample_frequency=self.sf,
            physical_min=-4000.0, physical_max=4000.0)
        
        if markerdata is not None:
            # Create the Annotation in format [
            # [timepoint, duration, description], [...]]
            marker_timepnts  = np.where(markerdata > 0)[0] 
            annotations = []
            for j in range(len(marker_timepnts)):
                annotations.append([
                    marker_timepnts[j]/self.sf,
                    0,
                    str(int(markerdata[marker_timepnts[j]]))])
            header = {'annotations':annotations}
        else:
            header = None

        # Write as EDF
        if hasattr(self,'duration'):
            channel_data = channel_data[:,:int(self.duration*self.sf)]
        else:
            channel_data = channel_data
        pyedflib.highlevel.write_edf(
            edffname,
            signals=channel_data,
            signal_headers=channelinfo,
            header=header,
            file_type=1)
    
    def close(self):
        """Release the Board"""
        self.board.release_session()
        print('Muse-S disconnected') 