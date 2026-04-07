# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 06:33:10 2024

@author: Arun Sasidharan
"""

#%% Import Libraries
import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt



class kmsgtrf_model:
        
    def __init__(self,
            ntimepts = 6,n_clusters = 6,n_components = 6,
            n_sgtkappa = 6,n_estimators = 100
            ):
        """Initialise models"""
        
        from sklearn.preprocessing import LabelEncoder,QuantileTransformer
        from sklearn.cluster import KMeans
        from sgt import SGT
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        from imblearn.over_sampling import SMOTE 
        
        self.n_clusters     = n_clusters
        self.ntimepts       = ntimepts
        self.n_components   = n_components
        self.n_estimators   = n_estimators 
        self.n_sgtkappa     = n_sgtkappa
        
        # Initilise Scaler
        self.scaler = QuantileTransformer(output_distribution = "normal")

        # Initilise Label Encoder
        self.le = LabelEncoder()
                        
        # Initilise Upampler
        self.smote_fold = SMOTE(sampling_strategy = 'auto',random_state = 42)

        # Initilise Clustering Model
        self.kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 42)

        # Initilise Sequence Graph Transform (SGT)
        self.sgt = SGT(kappa = n_sgtkappa, lengthsensitive = False)

        # Initilise PCA
        self.pca = PCA(n_components = self.n_components)           

        # Initilise RandomForest Classifier
        self.sgtrf_model = RandomForestClassifier(n_estimators = self.n_estimators) 
                 
        
    def train(self,X=[],Y=[],Z=[],ini_scaler=True,ini_kmeans=True,
              ini_labelencoder=True,ini_rf=True):
        """Train on data"""
        
        self.datasize   = len(X)        
        
        # Scale the data
        if ini_scaler:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        # Code the labels
        if ini_labelencoder:            
            Y = self.le.fit_transform(Y) 
        else:
            Y = self.le.transform(Y) 
        
        # Balance the data
        # X, Y = self.smote_fold.fit_resample(X, Y)
        
        # Get Cluster timeseries
        if ini_kmeans:
            XX = self.kmeans.fit_predict(X)
        else:
            XX = self.kmeans.predict(X) 
        XX = np.expand_dims(XX,axis=-1)
        
        # Reshape into [samples X features X timepoints]
        XX = np.lib.stride_tricks.sliding_window_view(XX, self.ntimepts,0)
        YY = np.lib.stride_tricks.sliding_window_view(Y, self.ntimepts,0)
        if len(Z) == 0:
            validindx = (YY.max(axis=1) - YY.min(axis=1))==0
        else:
            ZZ = np.lib.stride_tricks.sliding_window_view(Z, self.ntimepts,0)
            validindx = (ZZ[:,-1] - ZZ[:,0])==(self.ntimepts-1)
        
        XX = XX[validindx]
        YY = YY[validindx,0]
        
        # Remove samples with nan
        YY = YY[~np.isnan(XX.mean(axis=-1).mean(axis=-1))]
        XX = XX[~np.isnan(XX.mean(axis=-1).mean(axis=-1))]
        
        # Embed Cluster timeseries into values using Sequence Graph Transform (SGT)
        XXX_all = self.sgt.fit_transform(corpus=pd.DataFrame({'id':0,'sequence':XX.tolist()}))
        XXX_all = XXX_all.to_numpy()
        XXX_all = XXX_all[:,1:]
        
        # Data reduction using PCA
        XXX_all = self.pca.fit_transform(XXX_all)
                
        # Train using RandomForest Classifier on the sequence embeded values 
        if ini_rf:
            self.sgtrf_model.fit(XXX_all, YY)
        else:
            self.sgtrf_model.predict(XXX_all)
    
    
    def predict(self,X=[],Y=[],ini_scaler=False,ini_kmeans=False,plot=True):
        """Predict on new data"""
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.cluster import KMeans
        
        # Scale the data
        if ini_scaler:
            self.scaler = QuantileTransformer(output_distribution="normal")
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        #%% Get Cluster timeseries
        if ini_kmeans:
            self.kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
            XX = self.kmeans.fit_predict(X)
        else:
            XX = self.kmeans.predict(X)
        XX = np.expand_dims(XX,axis=-1)
        
        # Reshape into [samples X features X timepoints]
        XX = np.lib.stride_tricks.sliding_window_view(XX, self.ntimepts,0)
        
        # Embed Cluster timeseries into values using Sequence Graph Transform (SGT)
        XXX_all = self.sgt.transform(corpus=pd.DataFrame({'id':0,'sequence':XX.tolist()}))
        XXX_all = XXX_all.to_numpy()
        XXX_all = XXX_all[:,1:]
        
        # Data reduction using PCA
        XXX_all = self.pca.transform(XXX_all)
        
        # Predict using RandomForest Classifier on the sequence embeded values
        indx = range(len(X)-self.ntimepts+1)
        yy_prob = self.sgtrf_model.predict_proba(XXX_all)[indx]*100
        yy_pred = self.sgtrf_model.predict(XXX_all)[indx]
        
        if plot:
            # Plot
            plt.figure(figsize=[15,6])
            
            ncondn  = len(self.le.classes_)
            if len(Y) != 0:
                nrows   = ncondn+2
            else:
                nrows   = ncondn+1
            ncols   = 1
            plot_no = 1
            for condn_no in range(ncondn):
                plt.subplot(nrows,ncols,plot_no)
                plt.plot(np.arange(0,len(yy_prob)),yy_prob[:,condn_no])
                plt.ylabel(self.le.inverse_transform([condn_no])[0] + ' (%)')
                plt.ylim([-1,101])
                plot_no = plot_no + 1
            plt.subplot(nrows,ncols,plot_no)
            plt.step(np.arange(0,len(yy_pred)),yy_pred)
            plt.yticks(np.arange(0,ncondn))
            plt.yticks(np.arange(0,ncondn),
                       self.le.inverse_transform(np.arange(0,ncondn)))
            plt.ylabel('Predicted')
            plot_no = plot_no + 1
            if len(Y) != 0:
                Y = self.le.transform(Y)
                plt.subplot(nrows,ncols,plot_no)
                plt.step(np.arange(0,len(X)),Y)
                plt.yticks(np.arange(0,ncondn))
                plt.yticks(np.arange(0,ncondn),
                           self.le.inverse_transform(np.arange(0,ncondn)))
                plt.ylabel('Actual')
            plt.xlabel('Epochs')
        
        return yy_prob,yy_pred
    
    
    



#%%  

class eeg_realtime:
        
    def __init__(self,
            Amptype = 'xAMPL10'
            ):
        """Start LSL Stream"""
        import subprocess
        import os
        
        #%% Start LSL
        if Amptype == 'xAMPL10':
            self.LSLstream_name   = 'BESS.XAmp'
            # LSL Streamer will be opened using the following code
            self.LSLstreamer_exe  = 'AxxStreamerLSL.exe'
            self.LSLstreamer_path = r'C:\BESS\LSL_Streamer_XAmpL10'          
            curDir = os.getcwd()
            os.chdir(self.LSLstreamer_path)
            # self.lsl = subprocess.Popen([os.path.join(self.LSLstreamer_path,self.LSLstreamer_exe)],
            #                  stdin=subprocess.PIPE,
            #                  stdout=subprocess.PIPE,
            #                  )  
            
            try:
                arg_start = f'''{os.path.join(self.LSLstreamer_path,self.LSLstreamer_exe)} 
                        /stream_at_start:1 
                        /protocolname:16_Channel_SR250'''
                # arg_stop = f'''{os.path.join(self.LSLstreamer_path,self.LSLstreamer_exe)} 
                #         /stream_at_start:1 
                #         /protocolname:16_Channel_SR250'''
                # self.lsl = subprocess.Popen(arg_start,stdin=None,stdout=None)
                self.lsl = subprocess.Popen(arg_start,stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except:
                self.lsl = subprocess.Popen([os.path.join(self.LSLstreamer_path,self.LSLstreamer_exe)],
                                 stdin=None,
                                 stdout=None,
                                 )  
            os.chdir(curDir)
        elif Amptype == 'NEUPHONY':
            self.LSLstream_name   = 'NEUPHONY'
            # LSL Streamer will need to be openned manually
    
    def start_stream(self,
            bufferduration=60*5,
            chanindx = [0],channames = ['Fpz']): 
        """Start Stream"""
        from mne_lsl.stream import StreamLSL as Stream
        import time
        from mne.filter import filter_data                
        
        # Initialise Amp
        self.eeg         = Stream(name=self.LSLstream_name,stype='EEG',
                             bufsize=bufferduration).connect()
        self.srate       = int(self.eeg.info["sfreq"])
        self.chanlist    = np.array(self.eeg.info["ch_names"])
        
        chanindx = np.array(chanindx)        
        if len(chanindx) == 0:                    
            chanindx = np.arange(len(self.chanlist))
            
        chan = self.chanlist[chanindx]
            
        print(channames)
        if len(channames) == 0:
            channames = self.chanlist[chanindx]
        else:
            channames = np.array(channames)
        self.chanlist = channames 
        
    def start_pairedtrial(self,
            condn,stim_types,stim_images,stim_markers,
            cue_dur=2,act_dur=2,imag_cue=True,epoch_dur=1,
            resources_path=[],featurelist = ['catch22'],chan = ['Fp1']): 
        """Start Trial"""
        import pygame
        from mne_lsl.stream import StreamLSL as Stream
        import time
        from ccstools.eegfeatures import generate_multieegfeatures
        from mne.filter import filter_data
        
        # Initialise Display
        pygame.init()
        screen      = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        [SCREEN_WIDTH,SCREEN_HEIGHT] = pygame.display.get_window_size()
        clock       = pygame.time.Clock()
        BG_COLOR    = pygame.Color('gray12')
        clock.tick(60)  # Limit the frame rate to 60 FPS.
        pygame.mouse.set_visible(False)
        
        # Initialise Amp
        eeg         = Stream(name=self.LSLstream_name,stype='EEG',
                             bufsize=max(cue_dur,act_dur)).connect()
        self.srate       = eeg.info["sfreq"]
        self.chanlist    = eeg.info["ch_names"]
        time.sleep(0.5)    
        
        # Start
        self.randseqs    = np.random.permutation(len(stim_types)) 
        nTrials     = len(self.randseqs)
        trialrun    = True
        trialquit   = False
        i = 0
        self.epoch1 = []
        self.epoch2 = []
        self.eegfeatures_df1 = pd.DataFrame()
        self.eegfeatures_df2 = pd.DataFrame()
        screen.fill(BG_COLOR)  
        text            = pygame.font.SysFont("Arial", 54)
        txt_stim        = text.render(
            f'STARTING TASK  [ .... {condn} .... ]', True, (0,255,255))
        txt_stim_pos    = txt_stim.get_rect(
            center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        screen.blit(txt_stim, txt_stim_pos)
        pygame.display.update()
        time.sleep(3)
        
        while trialrun: # run for length of trials   
            
            # Check if "ESC" was pressed
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # uses the small 'x' on the top right to close the window
                    pygame.quit()
        
                if event.type == pygame.KEYDOWN: # processes all the Keydown events
                    if event.key == pygame.K_ESCAPE: # processes the Escape event (K_A would process the event that the key 'A' is hit
                        trialrun    = False    
                        trialquit   = True
                        continue            
                
            try:        
                
                screen.fill(BG_COLOR)  
                if imag_cue:
                    text            = pygame.font.SysFont("Arial", 44)
                    txt_stim        = text.render(
                        stim_types[self.randseqs[i]], True, (0,255,255))
                    txt_stim_pos    = txt_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/5))
                    screen.blit(txt_stim, txt_stim_pos)
                    img_stim        = pygame.image.load(
                        f"{resources_path}\{stim_images[self.randseqs[i]]}.jpg").convert()
                    img_stim        = pygame.transform.scale(
                        img_stim, (SCREEN_WIDTH/3, SCREEN_HEIGHT/3))
                    img_stim_pos    = img_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))        
                    screen.blit(img_stim, img_stim_pos)
                    pygame.display.update() 
                    # eeg.board.insert_marker(stim_markers[self.randseqs[i]]) 
                    time.sleep(cue_dur)
                else:
                    text            = pygame.font.SysFont("Arial", 44)
                    txt_stim        = text.render(
                        '***', True, (0,255,255))
                    txt_stim_pos    = txt_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/5))
                    screen.blit(txt_stim, txt_stim_pos)
                    pygame.mixer.init()
                    pygame.mixer.music.load(
                        f"{resources_path}\{stim_types[self.randseqs[i]]}.wav")
                    pygame.mixer.music.set_volume(1)
                    time.sleep(0.1)
                    pygame.mixer.music.play(loops=0)
                    time.sleep(0.5)
                    pygame.mixer.music.stop()
                    time.sleep(cue_dur-0.6)                
                
                screen.fill(BG_COLOR) 
                pygame.display.update()
                data, ts = eeg.get_data(cue_dur,picks=chan)
                data = filter_data(np.float64(data),int(self.srate), 1, 35)
                data = np.lib.stride_tricks.sliding_window_view(data.T,epoch_dur*int(self.srate),0).T 
                self.epoch1.append(data)
                tempeegfeatures_df = generate_multieegfeatures(
                    data,self.srate,chan,
                    featurelist = featurelist,
                    psdtype='welch',
                    kwargs_psd=dict(scaling='density',average='median',window="hamming",
                                    nperseg = int(self.srate)),
                    freq_range=[1,40],
                    bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
                           (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
                           (30, 40, 'Gamma1')])
                tempeegfeatures_df['Condn'] = stim_types[self.randseqs[i]]
                self.eegfeatures_df1 = pd.concat([self.eegfeatures_df1,tempeegfeatures_df],axis=0)
                time.sleep(0.5)
                
                text            = pygame.font.SysFont("Arial", 74)
                txt_stim        = text.render(
                    '+', True, (0,255,255))
                txt_stim_pos    = txt_stim.get_rect(
                    center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                screen.blit(txt_stim, txt_stim_pos)
                pygame.display.update() 
                # eeg.board.insert_marker(stim_markers[self.randseqs[i]]+1) 
                if condn == 'LISTEN':
                    pygame.mixer.init()
                    pygame.mixer.music.load(
                        f"{resources_path}\{stim_types[self.randseqs[i]]}.wav")
                    pygame.mixer.music.set_volume(1)
                    time.sleep(0.1)
                    pygame.mixer.music.play(loops=0)
                    time.sleep(0.5)
                    pygame.mixer.music.stop()
                    time.sleep(act_dur-0.6)
                else:                
                    time.sleep(act_dur)
                
                screen.fill(BG_COLOR) 
                pygame.display.update()
                data, ts = eeg.get_data(act_dur,picks=chan)
                data = filter_data(np.float64(data),int(self.srate), 1, 35)
                data = np.lib.stride_tricks.sliding_window_view(data.T,epoch_dur*int(self.srate),0).T 
                self.epoch2.append(data)
                self.tempeegfeatures_df = generate_multieegfeatures(
                    data,self.srate,chan,
                    featurelist = featurelist,
                    psdtype='welch',
                    kwargs_psd=dict(scaling='density',average='median',window="hamming",
                                    nperseg = int(self.srate)),
                    freq_range=[1,40],
                    bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
                           (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
                           (30, 40, 'Gamma1')])
                tempeegfeatures_df['Condn'] = stim_types[self.randseqs[i]]
                self.eegfeatures_df2 = pd.concat([self.eegfeatures_df2,tempeegfeatures_df],axis=0)
                time.sleep(0.5)
                
                i = i + 1
                print(i)
                if i == nTrials:
                    trialrun = False                    
            
                pygame.display.update() 
                
                if trialquit:
                    screen.fill(BG_COLOR)  
                    text            = pygame.font.SysFont("Arial", 64)
                    txt_stim        = text.render(
                        'QUITTING', True, (0,255,255))
                    txt_stim_pos    = txt_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                    screen.blit(txt_stim, txt_stim_pos)
                    pygame.display.update()
                    time.sleep(3)
                    
            except:
                # if ecg.board.is_prepared():
                #     ecg.close()
                i = i + 0
                print(i)
                if i == nTrials:
                    trialrun = False
                if trialquit:
                    screen.fill(BG_COLOR)  
                    text            = pygame.font.SysFont("Arial", 64)
                    txt_stim        = text.render(
                        'QUITTING', True, (0,255,255))
                    txt_stim_pos    = txt_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                    screen.blit(txt_stim, txt_stim_pos)
                    pygame.display.update()
                    time.sleep(3)
                continue
        
        # End
        eeg.disconnect()
        
        screen.fill(BG_COLOR)  
        text            = pygame.font.SysFont("Arial", 64)
        txt_stim        = text.render(
            'THANK YOU', True, (0,255,255))
        txt_stim_pos    = txt_stim.get_rect(
            center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        screen.blit(txt_stim, txt_stim_pos)
        pygame.display.update()
        time.sleep(3)
            
        pygame.quit()
        
        self.epoch1 = np.array(self.epoch1)
        self.epoch1 = self.epoch1.reshape([-1,self.epoch1.shape[-2],self.epoch1.shape[-1]])
        self.epoch2 = np.array(self.epoch2)
        self.epoch2 = self.epoch2.reshape([-1,self.epoch2.shape[-2],self.epoch2.shape[-1]])
        
        
    def start_singletrial(self,
            condn,stim_types,stim_images,stim_markers,
            cue_dur=2,imag_cue=True,epoch_dur=1,
            resources_path=[],featurelist = ['catch22'],chan = []): 
        """Start Trial"""
        import pygame
        from mne_lsl.stream import StreamLSL as Stream
        import time
        from ccstools.eegfeatures import generate_multieegfeatures
        from mne.filter import filter_data
        
        # Initialise Display
        pygame.init()
        screen      = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        [SCREEN_WIDTH,SCREEN_HEIGHT] = pygame.display.get_window_size()
        clock       = pygame.time.Clock()
        BG_COLOR    = pygame.Color('gray12')
        clock.tick(60)  # Limit the frame rate to 60 FPS.
        pygame.mouse.set_visible(False)
        
        # Initialise Amp
        eeg         = Stream(name=self.LSLstream_name,stype='EEG',
                             bufsize=cue_dur).connect()
        self.srate       = eeg.info["sfreq"]
        self.chanlist    = eeg.info["ch_names"]
        time.sleep(0.5)    
        
        # Start
        self.randseqs    = np.random.permutation(len(stim_types)) 
        nTrials     = len(self.randseqs)
        trialrun    = True
        trialquit   = False
        i = 0
        self.epoch1 = []
        self.eegfeatures_df1 = pd.DataFrame()
        screen.fill(BG_COLOR)  
        text            = pygame.font.SysFont("Arial", 54)
        txt_stim        = text.render(
            f'STARTING TASK  [ .... {condn} .... ]', True, (0,255,255))
        txt_stim_pos    = txt_stim.get_rect(
            center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        screen.blit(txt_stim, txt_stim_pos)
        pygame.display.update()
        time.sleep(3)
        
        while trialrun: # run for length of trials   
            
            # Check if "ESC" was pressed
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # uses the small 'x' on the top right to close the window
                    pygame.quit()
        
                if event.type == pygame.KEYDOWN: # processes all the Keydown events
                    if event.key == pygame.K_ESCAPE: # processes the Escape event (K_A would process the event that the key 'A' is hit
                        trialrun    = False    
                        trialquit   = True
                        continue            
                
            try:        
                
                screen.fill(BG_COLOR)  
                if imag_cue:
                    text            = pygame.font.SysFont("Arial", 44)
                    txt_stim        = text.render(
                        stim_types[self.randseqs[i]], True, (0,255,255))
                    txt_stim_pos    = txt_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/5))
                    screen.blit(txt_stim, txt_stim_pos)
                    img_stim        = pygame.image.load(
                        f"{resources_path}\{stim_images[self.randseqs[i]]}.jpg").convert()
                    img_stim        = pygame.transform.scale(
                        img_stim, (SCREEN_WIDTH/3, SCREEN_HEIGHT/3))
                    img_stim_pos    = img_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))        
                    screen.blit(img_stim, img_stim_pos)
                    pygame.display.update() 
                    # eeg.board.insert_marker(stim_markers[self.randseqs[i]]) 
                    time.sleep(cue_dur)
                else:
                    text            = pygame.font.SysFont("Arial", 44)
                    txt_stim        = text.render(
                        '***', True, (0,255,255))
                    txt_stim_pos    = txt_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/5))
                    screen.blit(txt_stim, txt_stim_pos)
                    pygame.mixer.init()
                    pygame.mixer.music.load(
                        f"{resources_path}\{stim_types[self.randseqs[i]]}.wav")
                    pygame.mixer.music.set_volume(1)
                    time.sleep(0.1)
                    pygame.mixer.music.play(loops=0)
                    time.sleep(0.5)
                    pygame.mixer.music.stop()
                    time.sleep(cue_dur-0.6)                
                
                screen.fill(BG_COLOR) 
                pygame.display.update()
                data, ts = eeg.get_data(cue_dur,picks=chan)
                data = filter_data(np.float64(data),int(self.srate), 1, 35)
                data = np.lib.stride_tricks.sliding_window_view(data.T,epoch_dur*int(self.srate),0).T                
                self.epoch1.append(data)
                tempeegfeatures_df = generate_multieegfeatures(
                    data,self.srate,chan,
                    featurelist = featurelist,
                    psdtype='welch',
                    kwargs_psd=dict(scaling='density',average='median',window="hamming",
                                    nperseg = int(self.srate)),
                    freq_range=[1,40],
                    bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
                           (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
                           (30, 40, 'Gamma1')])
                tempeegfeatures_df['Condn']  = stim_types[self.randseqs[i]]
                self.eegfeatures_df1 = pd.concat([self.eegfeatures_df1,tempeegfeatures_df],axis=0)
                time.sleep(0.5)                               
                
                i = i + 1
                print(i)
                if i == nTrials:
                    trialrun = False                    
            
                pygame.display.update() 
                
                if trialquit:
                    screen.fill(BG_COLOR)  
                    text            = pygame.font.SysFont("Arial", 64)
                    txt_stim        = text.render(
                        'QUITTING', True, (0,255,255))
                    txt_stim_pos    = txt_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                    screen.blit(txt_stim, txt_stim_pos)
                    pygame.display.update()
                    time.sleep(3)
                    
            except:
                # if ecg.board.is_prepared():
                #     ecg.close()
                i = i + 0
                print(i)
                if i == nTrials:
                    trialrun = False
                if trialquit:
                    screen.fill(BG_COLOR)  
                    text            = pygame.font.SysFont("Arial", 64)
                    txt_stim        = text.render(
                        'QUITTING', True, (0,255,255))
                    txt_stim_pos    = txt_stim.get_rect(
                        center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
                    screen.blit(txt_stim, txt_stim_pos)
                    pygame.display.update()
                    time.sleep(3)
                continue
        
        # End
        eeg.disconnect()
        
        screen.fill(BG_COLOR)  
        text            = pygame.font.SysFont("Arial", 64)
        txt_stim        = text.render(
            'THANK YOU', True, (0,255,255))
        txt_stim_pos    = txt_stim.get_rect(
            center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        screen.blit(txt_stim, txt_stim_pos)
        pygame.display.update()
        time.sleep(3)
            
        pygame.quit()
        
        self.epoch1 = np.array(self.epoch1)
        self.epoch1 = self.epoch1.reshape([-1,self.epoch1.shape[-2],self.epoch1.shape[-1]])
        
    def feature_stream(self,
            duration=30,plot=True,plt_dur=10,plt_scale=100,
            featurelist=['acw'],feature2plot=['ACW'],featurelim=[0,0.5],featuresmooth=1,
            featurewindow=15,
            chanindx = [0],channames = ['Fpz']): 
        """Start Trial"""
        from ccstools.plot import waveform
        from mne_lsl.stream import StreamLSL as Stream
        import time
        from mne.filter import filter_data
        from ccstools.eegfeatures import generate_multieegfeatures
        from ccstools.sigproc import smooth
        
        
        
        # Initialise Amp
        eeg         = Stream(name=self.LSLstream_name,stype='EEG',
                             bufsize=duration*2).connect()
        self.srate       = int(eeg.info["sfreq"])
        self.chanlist    = np.array(eeg.info["ch_names"])
        time.sleep(0.5)  
                
        
        # Plot      
        if plot:
            fig, axs    = plt.subplots(2,figsize=(14,8))
            plt.show()
        
        start_time  = time.perf_counter()
        time.sleep(1)
        elapsedtime = 0
        
                            
        chanindx = np.array(chanindx)        
        if len(chanindx) == 0:                    
            chanindx = np.arange(len(self.chanlist))
            
        chan = self.chanlist[chanindx]
            
        print(channames)
        if len(channames) == 0:
            channames = self.chanlist[chanindx]
        else:
            channames = np.array(channames)
        self.chanlist = channames        
        
            
        self.eegfeatures_df1 = pd.DataFrame()
        self.featureval     = [0]*featurewindow
        self.featuretimes   = []
        epoch_no = 0
        try:
            while elapsedtime < duration:    
                
                # Collect the data
                # newest_data = self.board.get_current_board_data(2*self.sf)[self.chanindx]*1e6
                newest_data, ts = eeg.get_data(plt_dur)
                newest_data = newest_data[chanindx]
                
                # Get the current data index
                newest_samplenum = ts[-1]
                times = (np.arange(newest_data.shape[-1])-newest_data.shape[-1]+newest_samplenum)/self.srate;
               
                # Filter the data
                newest_data = newest_data.astype('float64')
                newest_data = filter_data(np.float64(newest_data),self.srate, 1, 35)
                
                elapsedtime = (time.perf_counter() - start_time)
                
                if len(featurelist) != 0:
                    tempeegfeatures_df = generate_multieegfeatures(
                        newest_data.copy().mean(axis=0,keepdims=True),self.srate,channames[0],
                        featurelist = featurelist,
                        psdtype='welch',
                        kwargs_psd=dict(scaling='density',average='median',window="hamming",
                                        nperseg = self.srate),
                        freq_range=[1,40],
                        bands=[(1, 4, 'Delta'), (4, 8, 'Theta'), (6, 10, 'ThetaAlpha'), 
                               (8, 12, 'Alpha'), (12, 18, 'Beta1'),(18, 30, 'Beta2'), 
                               (30, 40, 'Gamma1')])
                    self.eegfeatures_df1 = pd.concat([self.eegfeatures_df1,tempeegfeatures_df],axis=0)
                
                if plot:
                    axs[0].cla()
                    waveform(newest_data,self.srate,list(channames),times=times,fig_ID=axs[0],scale=plt_scale,color="black")
                
                    if len(feature2plot) != 0:
                        axs[1].cla()
                        # self.featureval.pop(0)
                        self.featureval.append(np.mean(tempeegfeatures_df[feature2plot].to_numpy()))
                        self.featuretimes.append(elapsedtime)
                        epoch_no = epoch_no + 1
                        axs[1].plot(smooth(self.featureval[epoch_no:],featuresmooth))
                        if len(featurelim) != 0:
                            axs[1].set_ylim(featurelim)
                    
                    plt.title('Data collection %2.1f%%' %(100*elapsedtime/duration))
                    plt.draw()
                    plt.pause(0.5)
        except KeyboardInterrupt:
            print("STOPPED BY USER by pressing CTRL + C.")
        finally:                
            if plot:
                plt.close(fig)   
            # Stop Data streaming
            # self.board.stop_stream() 
            print("Stopping Data streaming")
        
        if len(feature2plot) != 0:
            self.featureval = self.featureval[featurewindow:]
        
        #%% Extract data
        data, ts = eeg.get_data(elapsedtime,picks=chan)
        eeg.disconnect() # Disconnect the Amp  
        data = data.astype('float64')
        data = filter_data(np.float64(data),self.srate, 1, 35)
        
        self.streamdata = data
        
    def audstimerp_stream(self,
            stimlist=[],filepath_Stim=[],
            plt_dur=10,plt_scale=100,
            interstimdur=1000,interstimjit=20,
            epochdurlim=[-200,800],epochamplim=[-5,5],
            chanindx = [0],channames = ['Fpz']): 
        
        """Start Trial"""
        from ccstools.plot import waveform
        from mne_lsl.stream import StreamLSL as Stream
        import time
        from mne.filter import filter_data
        from ccstools.sigproc import smooth
        import pygame
        
        nTrials     = len(stimlist)
        duration    = (nTrials*(interstimdur+100))/1000 + 5        
        chanindx    = np.array(chanindx)
        
        stimtypes = np.unique(stimlist)
        
        interstimperiods = np.arange(
            interstimdur - (interstimdur*interstimjit/100),
            interstimdur + (interstimdur*interstimjit/100)
            )
        interstimperiods = interstimperiods[np.random.permutation(len(interstimperiods))] 
        
        pygame.init()
        
        # Initialise Amp
        eeg         = Stream(name=self.LSLstream_name,stype='EEG',
                             bufsize=duration*2).connect()
        self.srate       = int(eeg.info["sfreq"])
        self.chanlist    = np.array(eeg.info["ch_names"])
        time.sleep(0.5)  
        
        # Plot            
        fig, axs    = plt.subplots(2,figsize=(14,8))
        start_time  = time.perf_counter()
        laststimtime = start_time
        time.sleep(1)
        elapsedtime = 0
        plt.show()
                            
        if len(chanindx) == 0:                    
            chanindx = np.arange(len(self.chanlist))
            
        chan = self.chanlist[chanindx]
            
        if len(channames) == 0:
            channames = self.chanlist[chanindx]
        else:
            channames = np.array(channames)[chanindx]
        self.chanlist = channames        
        
        
        trialrun    = True
        trialquit   = False
        i           = 0
        self.epochs = []
        
        
        while trialrun: # run for length of trials   
            
            # Check if "ESC" was pressed
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # uses the small 'x' on the top right to close the window
                    pygame.quit()
        
                if event.type == pygame.KEYDOWN: # processes all the Keydown events
                    if event.key == pygame.K_ESCAPE: # processes the Escape event (K_A would process the event that the key 'A' is hit
                        trialrun    = False    
                        trialquit   = True
                        continue   
        
            try:
                
                interstimperiod = (time.perf_counter() - laststimtime)*1000                
                
                # Collect the data
                # newest_data = self.board.get_current_board_data(2*self.sf)[self.chanindx]*1e6
                newest_data, ts = eeg.get_data(plt_dur)
                newest_data = newest_data[chanindx]
                
                # Get the current data index
                newest_samplenum = ts[-1]
                times = (np.arange(newest_data.shape[-1])-newest_data.shape[-1]+newest_samplenum)/self.srate;
               
                # Filter the data
                newest_data = newest_data.astype('float64')
                newest_data = filter_data(np.float64(newest_data),self.srate, 1, 35)
                
                elapsedtime = (time.perf_counter() - start_time)
                                
                
                axs[0].cla()
                waveform(newest_data,self.srate,list(channames),times=times,fig_ID=axs[0],scale=plt_scale,color="black")
                
                if interstimperiod > interstimperiods[i]:
                    laststimtime = elapsedtime
                    
                    stimFile = f"{filepath_Stim}\{stimlist[i]}"
                    pygame.mixer.init()
                    pygame.mixer.music.set_volume(1)
                    pygame.mixer.music.load(stimFile)
                    pygame.mixer.music.play(loops=0) #  repeats indefinitely   #loops=-1 
                    
                    # Collect the data
                    time.sleep(epochdurlim[-1]/1000)
                    epoch_data, ts = eeg.get_data((epochdurlim[-1] - epochdurlim[0])/1000)
                    epoch_data = epoch_data[chanindx]
                    epoch_times = (np.arange(epoch_data.shape[-1])/self.srate)*1000
                    epoch_times = epoch_times + epochdurlim[0]
                   
                    # Filter the data
                    epoch_data = epoch_data.astype('float64')
                    epoch_data = filter_data(np.float64(epoch_data),self.srate, 1, 35)
                    self.epochs.append(epoch_data)
                    
                    axs[1].cla()                    
                    for stim in stimtypes:
                        axs[1].plot(epoch_times,smooth(np.mean(np.array(self.epochs)[stimlist[:i+1] == stim],axis=0)[0],3))
                    axs[1].set_xlim(epochdurlim)
                    axs[1].set_ylim(epochamplim)
                    axs[1].legend(stimtypes)
                    plt.xlabel('Time (ms)')
                    
                                        
                
                plt.suptitle('Data collection %2.1f%%' %(100*(i+1)/nTrials))
                plt.draw()
                plt.pause(0.05)
                
                i = i + 1
                print(i)
                if i == nTrials:
                    trialrun = False 
            except KeyboardInterrupt:
                print("STOPPED BY USER by pressing CTRL + C.")
            finally:                
                # plt.close(fig)   
                # Stop Data streaming
                # self.board.stop_stream() 
                print("Stopping Data streaming")
        
        
 
        
        #%% Extract data
        data, ts = eeg.get_data(elapsedtime,picks=chan)
        eeg.disconnect() # Disconnect the Amp  
        data = data.astype('float64')
        data = filter_data(np.float64(data),self.srate, 1, 35)
        
        self.streamdata = data
        
    def stop_lsl(self):
        self.lsl.terminate()
        # self.disconnect()
        # output = self.lsl.communicate(input='stream_at_start:0'.encode())[0]
        # return output