from BenchFRET.pipeline.dataloader import SimLoader
from BenchFRET.pipeline.dataloader import RealLoader_training_raw
from BenchFRET.DeepLASI.wrapper import DeepLasiWrapper
from BenchFRET.pipeline.analysis import get_performance
import os
from tqdm.notebook import tqdm 
import re
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import importlib
import importlib.resources as resources
import tensorflow as tf
import pickle

def DeepLASI_performance_batch(folder_path,n_colors=2,n_states=2,save_path=None,save_memory=False,retrained_model_name=None):

    deeplasi = DeepLasiWrapper()
    
    file_paths = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    if save_memory:
        df = pd.DataFrame(columns=['noise', 'trans_rate', 'average_accuracy','predicted_states','algined_labels'])
    else:
        df = pd.DataFrame(columns=['noise', 'trans_rate', 'average_accuracy','average_precision','average_recall','predicted_states','algined_labels'])

    for file in tqdm(file_paths, desc='Processing files'):
        noise, trans_rate = re.findall(r'\d+\.\d+', file)
        
        print(f'for this dataset, noise is {noise}, transition_rate is {trans_rate}')

        simloader = SimLoader(data_path=os.path.join(folder_path, file))
        data = simloader.get_data()
        labels = simloader.get_labels()
        
        if retrained_model_name is not None:
            model_directory = resources.files(importlib.import_module(f'BenchFRET.DeepLASI'))
            model_path = os.path.join(model_directory,'retrained_models', retrained_model_name)
            model = tf.keras.models.load_model(model_path)
            detected_states, confidences = deeplasi._predict_states(data, n_states=n_states, model=model)
        else:
            detected_states, confidences = deeplasi._predict_states(data, n_states=n_states)
        
        average_accuracy, accuracy, aligned_labels = get_performance(n_states, detected_states, labels) # average_accuracy is a number, accuracy is 1d list, aligned_labels is a list of 1d arrays
        
        if not save_memory:
            aligned_labels = np.array(aligned_labels)
            detected_states = np.array(detected_states)

            conf_mat_precision = aggregate_confusion_matrix(aligned_labels,detected_states,n_states,mode='precision')
            average_precision = 0
            for i in range(n_states):
                average_precision += np.count_nonzero(aligned_labels == i)*conf_mat_precision[i,i]
            average_precision = average_precision/aligned_labels.size # the weighted average precision

            conf_mat_recall = aggregate_confusion_matrix(aligned_labels,detected_states,n_states,mode='recall')
            average_recall = 0
            for i in range(n_states):
                average_recall += np.count_nonzero(aligned_labels == i)*conf_mat_recall[i,i]
            average_recall = average_recall/aligned_labels.size # the weighted average recall

            df.loc[len(df)] = [noise, trans_rate, average_accuracy, average_precision, average_recall, detected_states, aligned_labels]
        else:
            df.loc[len(df)] = [noise, trans_rate, average_accuracy, detected_states, aligned_labels]
        
    if save_path is not None:
        df.to_json(save_path, orient='split')        # remember this orient='split' parameter when loading!
    else:
        df.to_json(f'DeepLASI_prediction_{n_states}_states.json', orient='split')
   

def aggregate_confusion_matrix(true,predicted,num_labels,mode='count'):
    
    aggregate_confusion_matrix = np.zeros((num_labels,num_labels))
    labels = np.arange(num_labels)
    for k in range(len(true)):
        conf_mat = confusion_matrix(true[k], predicted[k], labels=labels)
        aggregate_confusion_matrix += conf_mat
    if mode == 'precision':
        for i in range(aggregate_confusion_matrix.shape[0]):
            aggregate_confusion_matrix[:, i] = aggregate_confusion_matrix[:, i] / np.sum(aggregate_confusion_matrix[:, i])
    if mode == 'recall':
        for i in range(aggregate_confusion_matrix.shape[0]):
            aggregate_confusion_matrix[i] = aggregate_confusion_matrix[i]/np.sum(aggregate_confusion_matrix[i])
    
    return aggregate_confusion_matrix

def prepare_real_dataset(folder_path,save_path=None):

    file_paths = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    df = pd.DataFrame(columns = ['traces','labels','n_states'])
    for file in file_paths:
        loader = RealLoader_training_raw(os.path.join(folder_path, file))
        traces = loader.get_data()
        labels,n_states = loader.get_labels()
        for i in range(len(traces)):
            df.loc[len(df)] = [traces[i],labels[i],n_states[i]]
    sub_dfs = df.groupby('n_states')
    sub_dfs = [sub_dfs.get_group(x) for x in sub_dfs.groups]
    for i in range(len(sub_dfs)):
        trace_dictionary = {'data':[],'labels':[],'n_states':[]}
        trace_dictionary['data'].append(sub_dfs[i]['traces'].values)
        trace_dictionary['labels'].append(sub_dfs[i]['labels'].values)
        trace_dictionary['n_states'].append(sub_dfs[i]['n_states'].values)
        if save_path is not None:
            with open(os.path.join(save_path,f'real_dataset_{sub_dfs[i]["n_states"].values[0]}_states.pkl'), 'wb') as f:
                pickle.dump(trace_dictionary,f,protocol=pickle.HIGHEST_PROTOCOL)
            print(f'prepared real_dataset_{sub_dfs[i]["n_states"].values[0]}_states.pkl')
        else:
            raise ValueError('Please specify the save path')        