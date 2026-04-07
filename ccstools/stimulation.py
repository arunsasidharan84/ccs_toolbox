# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 09:37:52 2023

@author: Sruthi Kuriakose
@author: Arun Sasidharan
"""

#%% Import Libraries
import time
import platform
if platform.system() == 'Windows':
    print('Running on Windows')
    import ftd2xx as ftd
    # import serial.tools.list_ports
elif platform.system() == 'Linux':
    print('Running on Linux')
    import pylibftdi as ftdi
    print(ftdi.Driver().list_devices())
else:
    print('Unknown operating system')
    time.sleep(5)
    exit()

#%%
class axxStim:
    """
    The axxStim class is used for presenting taVNS or tACS stimulation and 
    includes functions for changing stimulation parameters
    parameters, starting and stopping the stimulation, and 
    closing the connection to the device.
    """
    def __init__(self, nmparams_dict):
        
        self.nmparams_dict  = nmparams_dict        

        self.wave_type      = self.nmparams_dict['wave_type']
        self.freq_input     = self.nmparams_dict['freq_input']
        self.current_input  = self.nmparams_dict['current_input']
        self.pulse_duration = self.nmparams_dict['pulse_duration']
        self.time_stim      = self.nmparams_dict['time_stim']
        self.tavns_string   = self.get_parameter_string()

        try:
                
            if platform.system() == 'Windows':
                searchaxxStim = True
                device_no  = 0
                while searchaxxStim:
                    device_tvns = ftd.open(device_no)
                    if not ('AXBLEXSTIM' in str(device_tvns.getDeviceInfo()['description']) or 'Dual RS232-HS' in str(device_tvns.getDeviceInfo()['description'])):
                        device_no = device_no + 1
                        device_tvns.close()
                    else:
                        searchaxxStim = False
                        
                # device_tvns = ftd.open(0)    # Open first FTDI device
                # ftd.getDeviceInfoDetail()
                print(device_tvns.getDeviceInfo())# print some dictionary entries for the device, e.g.
                # {'type': 6, 'id': 67330064, 'description': b'AXBLEXSTIM A', 'serial': b'AXXBLE00002A'}
                # If this fails, it is usually because the device is still using the VCP driver, or
                # the Python library can’t find the necessary FTDI files (ftd2xx.lib, and ftd2xx.dll or ftd2xx64.dll);
                #  they need to be somewhere on the executable PATH.
                # z=device_tvns.getComPortNumber()
                # device_tvns.close()
                # port = "COM"+str(z)
                # ser = serial.Serial(port, 921600)
                # x = ser.write('hello')
                # ser.close()
                # this sequence of steps need to be done for present configuration of tvns,
                # can be removed later
                # device_tvns = ftd.open(0)    # Open first FTDI device
                ftstatus = device_tvns.setBaudRate(921600)
                
            elif platform.system() == 'Linux':
                device_tvns=ftdi.Device()
                device_tvns.baudrate=921600
                
            self.device_tvns = device_tvns      

        except Exception as e:
            print(f'Check if tvns connected: {e}')
            # stop_flag_dict['stop'] = True
            raise Exception(e)
    
    
    def get_parameter_string(self):
        """
        The function takes in various parameters related to electrical stimulation and returns a formatted\
        string command for a device.
        
        :param wave_type: The type of waveform to be used for stimulation (sine, square, triangle, pulse)\
        and whether it is uniphasic or biphasic
        :param freq_input: Frequency input in Hz
        :param current_input: The input current in microamperes, with a maximum value of 2000uA
        :param pulse_width: The width of the pulse in microseconds
        :param time_stim: The duration of the stimulation in seconds
        :return: a string called "parametercommand" which is a concatenation of various parameters such as\
        wave type, frequency, current, pulse width, and duration in a particular format.
        """

        duration_dict = {30:'q',60:'1',90:'2'}
        wavetype_dict = {'sine-uniphasic':'10','sine-biphasic':'00',
                          'square-uniphasic':'11','square-biphasic':'01',
                          'triangle-uniphasic':'12','triangle-biphasic':'02',
                          'pulse-uniphasic':'13','pulse-biphasic':'03'
                        }
        if self.time_stim<30:
            duration = duration_dict[30]
        elif self.time_stim<60:
            duration = duration_dict[60]
        else:
            #TODO - multiples of 90
            duration = duration_dict[90]            
        
        wave_type_param = wavetype_dict[self.wave_type]
        
        current = f"{self.current_input:04d}"  #issue if input more than 4 chars - 4 is mimum width not maximum
        freq = f"{self.freq_input:06.2f}" 
        pulse = f"{self.pulse_duration:04d}" if wave_type_param=='13' or wave_type_param=='03' else '0000'
        parametercommand = f"hhd{duration}c1a{current}f{freq}t{wave_type_param}p{pulse}"
        # print(f"debug parametercommand {parametercommand}")
        return parametercommand
    
    def start_stim(self):
        """
        This function sends a command to start a stimulation device.
        """
        self.tavns_string   = self.get_parameter_string()
        parametercommand = self.tavns_string #= self.get_parameter_string(wave_type,freq_input,current_input,pulse_duration,time_stim)
        parameterbytes = parametercommand.encode('utf-8')
        self.device_tvns.write(parameterbytes) #21 bytes
        time.sleep(0.01)
        startcommand = "rx".encode('utf-8')
        self.device_tvns.write(startcommand) #start
        print(f'Started command {parameterbytes}')
        
    def change_stim_discomfort(self,decrease_rate=0.4):
        """
        This function decreases the current input of a TVNS device and updates the corresponding parameters.
        
        :param decrease_rate: The rate at which the current input will be decreased. It is a float value \
        between 0 and 1. For example, if decrease_rate is 0.4, the current input will be decreased by 40%
        """
        stopcommand = "ry".encode('utf-8')
        self.device_tvns.write(stopcommand)
        print('Changed command')
        print('debug current_input before: ',self.current_input)
        self.current_input = round(self.current_input*(1-decrease_rate))
        self.nmparams_dict['current_input'] = self.current_input
        print('debug current_input after: ',self.current_input)
        self.tavns_string = self.get_parameter_string(self.wave_type, self.freq_input,
                                 self.current_input, self.pulse_duration, self.time_stim)
        print(f'debug tavns string-changed: {self.tavns_string}')
        self.stop_flag_dict['discomfort'] = False
        
    def update_closed_loop(self,paramtochange,changerate=0.1):
        """
        This function updates parameters for a closed loop system and logs the changes.
        
        :param paramtochange: The parameter that needs to be updated (either 'current_input' or 'freq_input')
        :param changerate: The rate at which the parameter to be changed will be updated. It is set to 0.1 by default
        """
        stopcommand = "ry".encode('utf-8')
        self.device_tvns.write(stopcommand)
        print('Changed command')
        if paramtochange=='current_input':
            print('debug current_input before: ',self.current_input)
            self.current_input = round(self.current_input*(1-changerate))
            self.nmparams_dict['current_input'] = self.current_input
            self.logger.info(f'current_input updated: {self.current_input}')
        if paramtochange=='freq_input':
            print('debug freq_input before: ',self.freq_input)
            self.freq_input = round(self.freq_input*(1-changerate))
            self.nmparams_dict['freq_input'] = self.freq_input
            self.logger.info(f'freq_input updated: {self.freq_input}')
        self.tavns_string = self.get_parameter_string(self.wave_type, self.freq_input,
                                 self.current_input, self.pulse_duration, self.time_stim)
        self.logger.info(f'Parameters updated: {self.tavns_string}')
        self.stop_flag_dict['update_tvns']=False
        
        
    def stop_stim(self):
        """
        This function sends a stop command to a device called "device_tvns" and logs a message indicating
        that the command has been stopped.
        """
        stopcommand = "ry".encode('utf-8')
        self.device_tvns.write(stopcommand)
        print('Stopped command')
        
    def close_connection(self):
        """
        This function closes the connection to a device, flushing buffers and resetting the device if
        necessary based on the operating system.
        """
        if platform.system() == 'Windows':
            # refer os.path.abspath(ftd.__file__)
            self.device_tvns.purge()  #Clears the device’s read and write buffers. #TODO - linux check
            self.device_tvns.resetDevice()
        elif platform.system() == 'Linux':
            self.device_tvns.flush()
        self.device_tvns.close()
        print('Closed connection to device')
        # device_tvns.clrDtr()
        # del d
        