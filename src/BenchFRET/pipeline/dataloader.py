import os
import pickle
import numpy as np
import traceback
import pandas as pd
import json
import copy
from abc import ABC, abstractmethod
from BenchFRET.simulation.trace_generator import trace_generator

class DataLoader(ABC):
    
    @abstractmethod
    def get_data(self):
        return  

    @abstractmethod
    def get_parameters(self,data_dict):
        return

    @abstractmethod
    def get_labels(self,data_dict):
        return

class NewSim(DataLoader): # currently only 2-color simulation supported

    def __init__(self,
                 n_traces = 100,
                 n_frames = 500,
                 n_colors = 2,
                 n_states = 2,
                 trans_mat=None, # if none, generates random tmat, else uses given tmat
                 noise=(.01, 1.2), # controls bg poisson noise scale factor
                 gamma_noise_prob=.8,  # probability of gamma noise
                 reduce_memory = True, # if true, only returns DD, DA
                                       # if false, returns DD, DA, AA, E, E_true, label, noise_level, min_E_diff, trans_mean
                 mode = "state_mode", # state_mode, n_states_mode
                 parallel_asynchronous = False,
                 outdir = "simulated_datasets",
                 export_mode = "",
                 export_name = "trace_dataset",
                 min_state_diff=0.1, # minimum intesnity difference between FRET states
                 ):
        
        """
        Simulation scripts are inspired by the DeepLASI implementation of DeepFRET simulation scripts.
    
        n_traces: Number of traces
        n_timesteps: Number of frames per trace
        n_colors: Number of colors (1-color, 2-color or 3-color data possible)
        balance_classes: Balance classes based on minimum number of labeled frames
        reduce_memory: Include/exclude trace parameters beside countrates
        state_mode: Label dynamic traces according to state occupancy, used for training state classifiers
        n_states_model: Label each trace according to number of observed traces, used for number of states classifier
        parallel_asynchronous: parallel processing (faster)
        outdir: Output directory
        export_mode: export mode, more modes will be added over time):
        """
        self.n_traces = n_traces
        self.n_frames = n_frames
        self.n_colors = n_colors
        self.n_states = n_states
        self.trans_mat = trans_mat
        self.noise = noise
        self.gamma_noise_prob = gamma_noise_prob
        self.reduce_memory = reduce_memory
        self.mode = mode
        self.parallel_asynchronous = parallel_asynchronous
        self.outdir = outdir
        self.export_mode = export_mode
        self.export_name = export_name
        self.min_state_diff = min_state_diff

        self.generator = trace_generator(n_traces=int(self.n_traces),
                                        n_frames=self.n_frames,
                                        n_colors=self.n_colors,
                                        n_states=self.n_states,
                                        trans_mat=self.trans_mat,
                                        noise=self.noise,
                                        gamma_noise_prob=self.gamma_noise_prob,
                                        reduce_memory=self.reduce_memory,
                                        mode=self.mode,
                                        parallel_asynchronous=self.parallel_asynchronous,
                                        outdir=self.outdir,
                                        export_mode=self.export_mode,
                                        export_name=self.export_name,
                                        min_state_diff=self.min_state_diff)        
        
        self.training_data, self.training_labels = self.generator.generate_traces()

    def get_data(self):

        if self.reduce_memory:
            print("fetched data: DD, DA")
        else:
            print("fetched data: DD, DA, AA, E, E_true, label, noise_level, min_E_diff, trans_mean")
        print()    
        return np.array(self.training_data)

    def get_parameters(self,data_dict):
        pass

    def get_labels(self):

        return np.array(self.training_labels)


class SimLoader(DataLoader):

    def __init__(self,
                data_type='pickeldict', # pickledict, text_files, ebfret
                data_path=''): 
        self.data_type = data_type
        self.data_path = data_path
        assert os.path.exists(self.data_path)==True, "data_path does not exist"
        if self.data_type == 'pickeldict':
            with open(self.data_path, 'rb') as f:
                self.dataset = pickle.load(f)
        # add text_files, ebfret later

    def get_data(self):
        if self.data_type == 'pickeldict':
            data = self.dataset["data"]
            data = np.array(data)
            if data.shape[2] == 2:
                print(f'traces are of shape:{data.shape}')
                print(f'got {data.shape[0]} traces containing {data.shape[2]} features: DD, DA')
                print()
                return data
            else:
                print(f'trace are of shape:{data.shape}')
                print(f'got {data.shape[0]} traces containing {data.shape[2]} features: DD, DA, AA, E, E_true, label, noise_level, min_E_diff, trans_mean')
                data_lite = data[:,:,:2]
                print(f'extracted only DD, DA as output 1, the full data as output 2')
                print()
                return data_lite, data
        
        # elif self.data_type == 'text_files':
        #     pass

        # else:
        #     pass

    def get_parameters(self):

        if self.data_type == 'pickeldict':
            param_dict = self.dataset["simulation_parameters"]
            print(f'got parameters dictionary containing the following keys:{list(param_dict.keys())}')
            print()
        return param_dict
        
        # elif self.data_type == 'text_files':
        #     pass

        # else:
        #     pass
    
    def get_labels(self):

        if self.data_type == 'pickeldict':
            labels = np.array(self.dataset["labels"])
            print(f'labels are of shape:{labels.shape}')
            print(f'got labels for {labels.shape[0]} traces')
        print()
        return labels
        
        # elif self.data_type == 'text_files':
        #     pass

        # else:
        #     pass
