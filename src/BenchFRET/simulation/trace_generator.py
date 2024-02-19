import pickle
import traceback
import numpy as np
from BenchFRET.simulation import training_data_1color, training_data_2color
from time import time
import os
import shutil
import datetime
import random
import string

class trace_generator():
    
    def __init__(self,
                 n_traces = 100,
                 n_frames = 500,
                 n_colors = 2,
                 n_states = 2,
                 trans_mat=None, # if none, generates random tmat, else uses given tmat
                 noise=(.01, 1.2), # controls bg poisson noise scale factor
                 gamma_noise_prob=.8,  # probability of gamma noise
                 reduce_memory = True,
                 mode = "state_mode", # state_mode, n_states_mode
                 parallel_asynchronous = False,
                 outdir = "",
                 export_mode = "",
                 export_name = "trace_dataset",
                 min_state_diff=0.1, # minimum intesnity difference between FRET states
                 
                    d_lifetime=400, # realted to bleaching
                    a_lifetime=400, # realted to bleaching
                    blink_prob=0.1, # realted to bleaching
                    crosstalk=(0, 0.4), # correction factors
                    dir_exc=(0, .2), # correction factors
                    gamma=(.5, 1.5), # correction factors
                    aa_offset=(.2, 2), # random offset of value between .2 and 2
                    max_trans_prob=.2, # unused parameter 
                    aggregation_prob=0.15,
                    max_aggregate_size=20,
                    non_spFRET_E=-1, # E value assigned for bleanched traces
                    noise_tolerance=.3, # tolerence above which traces are identified as noisy
                    s_tolerance=(0.05, 0.95), # tolerance for bleanched traces 
                    randomize_prob=0.25, # prob for artefacts
                    bg_overestimate_prob=.1, #  prob for artefacts
                    falloff_lifetime=500, # related to aggregation
                    falloff_prob=0.1, # related to aggregation
                    balance_bleach=True, 
                    merge_state_labels=True,
                    omit_bleach=False,
                    quenching=False
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
        export_mode: export mode, more modes will be added over time
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
        
        self.d_lifetime = d_lifetime
        self.a_lifetime = a_lifetime
        self.blink_prob = blink_prob
        self.crosstalk = crosstalk
        self.dir_exc = dir_exc
        self.gamma = gamma
        self.aa_offset = aa_offset
        self.max_trans_prob = max_trans_prob
        self.aggregation_prob = aggregation_prob
        self.max_aggregate_size = max_aggregate_size
        self.non_spFRET_E = non_spFRET_E
        self.noise_tolerance = noise_tolerance
        self.s_tolerance = s_tolerance
        self.randomize_prob = randomize_prob
        self.bg_overestimate_prob = bg_overestimate_prob
        self.falloff_lifetime = falloff_lifetime
        self.falloff_prob = falloff_prob
        self.balance_bleach = balance_bleach
        self.merge_state_labels = merge_state_labels
        self.omit_bleach = omit_bleach
        self.quenching = quenching
        
        self.check_mode()
        self.check_outdir()

        assert n_colors in [1,2], "available colours: 1, 2"
        assert export_mode in ["","text_files", "pickledict","ebfret", "ebFRET_files"], "available export modes: '','text_files', 'pickledict', 'ebfret', 'ebFRET_files'"
        
    def check_outdir(self, overwrite=False, folder_name = "simulated_traces"):
    
        if os.path.exists(self.outdir) == False:
            self.outdir = os.getcwd()
        
        if folder_name != "":
            self.outdir = os.path.join(self.outdir, f"{self.export_mode}")
            
        # if overwrite and os.path.exists(self.outdir):
        #         shutil.rmtree(self.outdir)

        if os.path.exists(self.outdir) == False:
            os.mkdir(self.outdir)

        
    def check_mode(self):
        
        assert self.mode in ["state_mode", "n_states_mode"], "available modes: 'state_mode', 'n_states_mode'"
        
        if self.mode == "state_mode":
            self.state_mode = True
            self.n_states_mode = False
        else:
            self.state_mode = False
            self.n_states_mode = True
    
    def generate_single_colour_traces(self):
        
        traces, _ = training_data_1color.simulate_1color_traces(
            n_traces=int(self.n_traces),
            max_n_states=self.n_states,
            n_frames=self.n_frames,
            state_mode=self.state_mode,
            n_states_mode=self.n_states_mode,
            reduce_memory=self.reduce_memory,
            parallel_asynchronous=self.parallel_asynchronous
        )
        
        training_data = []
        training_labels = []
        
        for trace in traces:
            
            training_labels.append(trace["label"].values)
            
            if self.reduce_memory:
                training_data.append(trace[["DD"]].values)
            else:
                training_data.append(trace[["DD", "DA", 
                                            "E", "E_true", 
                                            "label", "_noise_level", 
                                            "_min_E_diff", "trans_mean"]].values)
                
        return training_data, training_labels
    
    def generate_two_colour_traces(self):
        
        traces, _ = training_data_2color.simulate_2color_traces(
            n_traces=int(self.n_traces),
            n_frames=self.n_frames,
            max_n_states=self.n_states,
            trans_mat=self.trans_mat,
            noise=self.noise,
            gamma_noise_prob=self.gamma_noise_prob,
            reduce_memory=self.reduce_memory,
            state_mode=self.state_mode,
            n_states_mode=self.n_states_mode,
            parallel_asynchronous=self.parallel_asynchronous,
            min_state_diff=self.min_state_diff,
            d_lifetime=self.d_lifetime,
            a_lifetime=self.a_lifetime,
            blink_prob=self.blink_prob,
            crosstalk=self.crosstalk,
            dir_exc=self.dir_exc,
            gamma=self.gamma,
            aa_offset=self.aa_offset,
            max_trans_prob=self.max_trans_prob,
            aggregation_prob=self.aggregation_prob,
            max_aggregate_size=self.max_aggregate_size,
            non_spFRET_E=self.non_spFRET_E,
            noise_tolerance=self.noise_tolerance,
            s_tolerance=self.s_tolerance,
            randomize_prob=self.randomize_prob,
            bg_overestimate_prob=self.bg_overestimate_prob,
            falloff_lifetime=self.falloff_lifetime,
            falloff_prob=self.falloff_prob,
            balance_bleach=self.balance_bleach,
            merge_state_labels=self.merge_state_labels,
            omit_bleach=self.omit_bleach,
            quenching=self.quenching
        )
        
        training_data = []
        training_labels = []
        
        for trace in traces:

            try:

                training_labels.append(trace["label"].values)

                if self.reduce_memory:

                    if self.state_mode or self.n_states_mode:
                        if self.n_colors == 2:
                            training_data.append(trace[["DD", "DA"]].values)
                        else:
                            training_data.append(trace["E"].values)
                    else:
                        training_data.append(trace[["DD", "DA", "AA"]].values)

                else:
                    training_data.append(trace[["DD", "DA",
                                                "AA", "E",
                                                "E_true", "label",
                                                "_noise_level",
                                                "_min_E_diff", "trans_mean"]].values)
            except:
                print(traceback.format_exc())
                pass
                
        return training_data, training_labels
        
    def export_traces(self, training_data, training_labels):

        date = datetime.datetime.now().strftime("%Y_%m_%d")
        
        if self.export_mode == "text_files":
            
            print(f"exporting txt files to: {self.outdir}")
            
            for index, (data, label) in enumerate(zip(training_data, training_labels)):
                
                label = np.expand_dims(label, 1)
            
                dat = np.hstack([data, label])
                
                file_path = os.path.join(self.outdir, f"{self.export_name}_{date}_{index}.csv")
                
                np.savetxt(file_path, dat, delimiter=",")

        # if self.export_mode == "ebFRET_files":

        #     print(f"exporting ebFRET files to: {self.outdir}")

        #     traces = np.array(training_data)
        #     ebFRET_traces = []
        #     for i in range(traces.shape[0]):
        #         ebFRET_traces.append(np.hstack([np.expand_dims([i] * traces.shape[1], 1), traces[i]]))
        #     ebFRET_traces = np.vstack(ebFRET_traces)
        #     ebFRET_traces32 = ebFRET_traces.astype(np.float32)
        #     trace_path = os.path.join(self.outdir, "simulated-K04-N350-raw-stacked.dat")
        #     label_path = os.path.join(self.outdir, "simulated_traces_labels.dat")
        #     np.savetxt(trace_path, ebFRET_traces32, delimiter=" ")
        #     np.savetxt(label_path, training_labels, delimiter=" ")

        if self.export_mode == "pickledict":

            trace_dictionary = {"data": [], "labels": [], "simulation_parameters": {}}

            trace_dictionary["simulation_parameters"]["n_traces"] = self.n_traces
            trace_dictionary["simulation_parameters"]["n_frames"] = self.n_frames
            trace_dictionary["simulation_parameters"]["n_colors"] = self.n_colors
            trace_dictionary["simulation_parameters"]["n_states"] = self.n_states
            trace_dictionary["simulation_parameters"]["trans_mat"] = self.trans_mat
            trace_dictionary["simulation_parameters"]["noise"] = self.noise
            trace_dictionary["simulation_parameters"]["gamma_noise_prob"] = self.gamma_noise_prob
            trace_dictionary["simulation_parameters"]["reduce_memory"] = self.reduce_memory
            trace_dictionary["simulation_parameters"]["mode"] = self.mode
            trace_dictionary["simulation_parameters"]["parallel_asynchronous"] = self.parallel_asynchronous
            trace_dictionary["simulation_parameters"]["min_state_diff"] = self.min_state_diff

            for index, (data, label) in enumerate(zip(training_data, training_labels)):

                trace_dictionary["data"].append(data)
                trace_dictionary["labels"].append(label)

            file_path = os.path.join(self.outdir, f"{self.export_name}.pkl")

            with open(file_path, 'wb') as handle:
                pickle.dump(trace_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"exporting pickled dictionary to: {file_path}")


        if self.export_mode == "ebfret":

            file_path = os.path.join(self.outdir, f"{self.export_name}_{date}_SMD.mat")

            ebfret_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))

            smd_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))

            smd_dict = {"attr": {"data_package": "DeepGapSeq"},
                        "columns": [],
                        "data": {"attr": [],
                                 "id": [],
                                 "index": [],
                                 "values": []},
                        "id": ebfret_id, "type": "DeepGapSeq-simulated", }

            trace_names = ['Donor', 'Acceptor']

            for data_index, (data, label) in enumerate(zip(training_data, training_labels)):

                smd_values = data.tolist()
                smd_index = np.expand_dims(np.arange(len(smd_values)), -1).tolist()

                lowerbound = np.min(smd_values)

                smd_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=32))
                smd_attr = {"file": os.path.basename(file_path), "lowerbound":lowerbound,"restart": 0, "crop_min": 0, "crop_max": 20}

                if smd_dict["columns"] == []:
                    columns = trace_names
                    columns = [column.strip() for column in columns]
                    smd_dict["columns"] = columns

                smd_dict["data"]["attr"].append(smd_attr)
                smd_dict["data"]["id"].append(smd_id)
                smd_dict["data"]["index"].append(smd_index)
                smd_dict["data"]["values"].append(smd_values)

            import mat4py
            mat4py.savemat(file_path, smd_dict)

            print(f"exporting ebFRET SMD file to: {file_path}")




    def generate_traces(self):
        
        print("Generating traces...")
        start = time()
        
        if self.n_colors == 1 and not self.state_mode:
            
            training_data, training_labels = self.generate_single_colour_traces()
                
        elif self.n_colors == 2 or (self.n_colors == 1 and self.state_mode):
            
            training_data, training_labels = self.generate_two_colour_traces()
            
        stop = time()
        duration = stop - start
        
        unique_labels = np.unique(np.concatenate(training_labels))
        
        print(f"Spent {duration:.1f} s to generate {len(training_data)} traces")
        print("Labels: ", unique_labels)
        
        self.export_traces(training_data, training_labels)
        
        return training_data, training_labels
        



