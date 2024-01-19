import tensorflow as tf
import importlib
import importlib.resources as resources
import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import statistics
from DeepGapSeq.simulation.deepgapseq_trace_generator import trace_generator
import matplotlib.pyplot as plt


class DeepLasiWrapper():

    def __init__(self, parent=None, n_colors=2, n_states=2, model_type="states", traces=[]):
        self.AnalysisGUI = parent
        self.n_colors = n_colors
        self.n_states = n_states
        self.model_type = model_type
        self.traces = traces

        # These are the DeepLASI labels for 1 and 2 color traces
        # e.g. if a 1 color trace is labeled 5 in DeepLASI trace classifier model, it is a dynamic trace

        self.one_color_trace_labels = {5: "dynamic", 4: "static", 3: "noisy", 2: "artifact", 1: "aggregate", 0: "bleached"}

        self.two_color_trace_labels = {7: "dynamic", 6: "static", 5: "noisy", 4: "artifact", 3: "aggregate", 2: "all_bleached", 1: "acceptor_bleached", 0: "donor_bleached"}

        self.check_gpu()

    def print_notification(self, message):
        if self.AnalysisGUI is not None:
            self.AnalysisGUI.print_notification(message)
        else:
            print(message)

    def check_gpu(self):
        """Check if a GPU is available, if not, run on CPU"""

        devices = tf.config.list_physical_devices('GPU')

        if len(devices) == 0:
            self.print_notification("Tensorflow running on CPU")
            self.gpu = False
            self.device = None
        else:
            self.print_notification("Tensorflow running on GPU")
            self.gpu = True
            self.device = devices[0]

    def check_data_format(self, traces):
        """Check if the input data is in the correct format: a list of numpy arrays of shape (B,N,C)"""

        correct_format = True

        for index, arr in enumerate(traces):
            if not isinstance(arr, np.ndarray):  # Check if the element is a NumPy array
                correct_format = False
                break
            if len(arr.shape) == 1 and self.n_colors == 1:
                arr = np.expand_dims(arr, axis=-1)
                traces[index] = arr
                correct_format == True
            else:
                if arr.shape[-1] != self.n_colors:
                    correct_format = False
                    break

        if correct_format == False:
            self.print_notification("input data must be a list of numpy arrays of shape (B,N,C)")

        return correct_format, traces

    def initialise_models(self, n_colors=""):
        """Load the pretrained DeepLASI models, only loads models for the specified number of colours"""

        self.deeplasi_models = {"states": {}, "n_states": "", "trace": ""}

        model_directory = resources.files(importlib.import_module(f'DeepGapSeq.DeepLASI'))

        if n_colors.isdigit():
            self.n_colors = int(n_colors)

        self.print_notification(f"Loading DeepLASI pretrained {self.n_colors}colour models")

        for n_states in range(2, 5):
            state_model_name = "DeepLASI_{}color_{}state_classifier.h5".format(self.n_colors, n_states)

            state_model_path = os.path.join(model_directory, "models", state_model_name)

            state_model = tf.keras.models.load_model(state_model_path)

            self.deeplasi_models["states"][n_states] = state_model

        n_states_model_name = "DeepLASI_{}color_number_of_states_classifier.h5".format(self.n_colors)
        n_states_model_path = os.path.join(model_directory, "models", n_states_model_name)
        n_states_model = tf.keras.models.load_model(n_states_model_path)

        self.deeplasi_models["n_states"] = n_states_model

        if self.n_colors == 1:
            trace_model_name = "DeepLASI_1Color_trace_classifier.h5"
        elif self.n_colors == 2:
            trace_model_name = "DeepLASI_2color_nonALEX_trace_classifier.h5"
        elif self.n_colors == 3:
            trace_model_name = "DeepLASI_3Color_trace_classifier.h5"

        trace_model_path = os.path.join(model_directory, "models", trace_model_name)
        trace_model = tf.keras.models.load_model(trace_model_path)

        self.deeplasi_models["trace"] = trace_model

        return self.deeplasi_models

    def preprocess_data(self, traces):
        """Normalise the traces to the range [0,1]"""

        for index, trace in enumerate(traces):
            # Find the minimum and maximum values for each channel
            min_values = np.min(trace, axis=0)
            max_values = np.max(trace, axis=0)

            # Normalize each channel independently
            normalised_trace = (trace - min_values) / (max_values - min_values)

            traces[index] = normalised_trace

        return traces

    def arrays_to_tensor(self, list_of_arrays):
        """Convert a list of NumPy arrays to a RaggedTensor"""

        # Concatenate the NumPy arrays along the first axis to create a flat tensor
        flat_tensor = tf.concat(list_of_arrays, axis=0)

        # Create value_rowids to indicate which values belong to which arrays
        value_rowids = tf.concat([tf.fill((a.shape[0],), i) for i, a in enumerate(list_of_arrays)], axis=0)

        # Create the ragged tensor from the flat tensor and value_rowids
        ragged_tensor = tf.RaggedTensor.from_value_rowids(values=flat_tensor, value_rowids=value_rowids)

        return ragged_tensor

    def pad_traces(self, list_of_arrays, pad_value=0):
        # Find the maximum length among all arrays
        max_length = max(len(arr) for arr in list_of_arrays)

        # Create a list to store the padded arrays
        padded_arrays = []

        for arr in list_of_arrays:
            # Calculate the amount of padding needed
            pad_length = max_length - len(arr)

            # Pad the array with zeros (or any other specified pad_value)
            padded_arr = np.pad(arr, ((0, pad_length), (0, 0)), constant_values=pad_value)

            # Append the padded array to the list
            padded_arrays.append(padded_arr)

        return padded_arrays

    def _predict_trace_label(self, trace, verbose=True):
        """Predict the DeepLASI label of a single trace"""

        if self.n_colors == 1:
            trace_prediction_labels = self.one_color_trace_labels
        elif self.n_colors == 2:
            trace_prediction_labels = self.two_color_trace_labels

        model = self.deeplasi_models["trace"]

        output = model.predict(trace, verbose=verbose)

        if output.shape[0] == 1:
            pred = statistics.mode(np.argmax(output[0], axis=1))
            conf = statistics.mode(np.max(output[0], axis=1))

            prediction_list = [pred]
            prediction_label_list = [trace_prediction_labels[pred]]
            confidence_list = [conf]

        else:
            prediction_list = []
            prediction_label_list = []
            confidence_list = []

            for dat in output:
                pred = statistics.mode(np.argmax(dat, axis=1))
                conf = statistics.mode(np.max(dat, axis=1))
                label = trace_prediction_labels[pred]

                prediction_list.append(pred)
                confidence_list.append(conf)
                prediction_label_list.append(label)

        return prediction_list, confidence_list, prediction_label_list

    def _predict_n_states(self, trace, verbose=True):
        """given a trace, predict the number of states in the trace, can be 2, 3, or 4"""

        model = self.deeplasi_models["n_states"]

        output = model.predict(trace, verbose=verbose)

        if output.shape[0] == 1:
            pred = statistics.mode(np.argmax(output[0], axis=1))
            conf = statistics.mode(np.max(output[0], axis=1))

            prediction_list = [pred + 2]
            confidence_list = [conf]

        else:
            prediction_list = []
            confidence_list = []

            for opt in output:
                pred = statistics.mode(np.argmax(opt, axis=1))
                conf = statistics.mode(np.max(opt, axis=1))

                prediction_list.append(pred + 2)
                confidence_list.append(conf)

        return prediction_list, confidence_list

    def _predict_states(self, trace, n_states):
        """given a trace and a target number of states, predict the state trace [00011110011....]"""

        model = self.deeplasi_models["states"][n_states]

        output = model.predict(trace, verbose=None)

        if output.shape[0] == 1:
            prediction_list = [np.argmax(output[0], axis=1)]
            confidence_list = [np.max(output[0], axis=1)]

        else:
            prediction_list = []
            confidence_list = []

            for opt in output:
                pred = np.argmax(opt, axis=1)
                conf = np.max(opt, axis=1)

                prediction_list.append(pred)
                confidence_list.append(conf)

        return prediction_list, confidence_list

    def predict(self, traces=[], n_colors=None, n_states=None, deeplasi_mode="fast", progress_callback=None):
        """
        The main method for predicting DeepLASI states from traces

        If all the traces are the same length, they can be proccessed in parallel, this is "fast" mode.
        If the traces are different lengths, they must be processed one at a time, this is "slow" mode.

        """

        if len(traces) > 0:
            self.traces = traces
        if n_colors in [1, 2, 3]:
            self.n_colors = n_colors

        correct_format, traces = self.check_data_format(traces)

        states_predictions = []
        trace_labels = []

        if correct_format:
            traces = self.preprocess_data(traces)

            self.initialise_models()

            self.print_notification(f"Predicting DeepLASI states for {len(traces)} traces...")

            if deeplasi_mode.lower() == "fast":
                """Predict the states of all traces in parallel"""

                traces = tf.convert_to_tensor(traces, dtype=tf.float32)

                # predict whether a trace is dynamic or static
                trace_prediction, trace_confidence, trace_labels = self._predict_trace_label(traces, verbose=False)

                if progress_callback is not None:
                    progress_callback.emit(33)

                if n_states is None:
                    # predict the number of states in the trace
                    n_states_prediction, n_states_confidence = self._predict_n_states(traces, verbose=False)
                else:
                    # use the user defined number of states
                    n_states_prediction = [n_states] * len(traces)
                    n_states_confidence = [1] * len(traces)

                if progress_callback is not None:
                    progress_callback.emit(66)

                for index, (trace, label, n_states) in enumerate(zip(traces, trace_labels, n_states_prediction)):
                    if label == "dynamic" and n_states in [2, 3, 4]:
                        trace = tf.expand_dims(trace, axis=0)

                        # predict the states of the trace
                        state_prediction, state_confidence = self._predict_states(trace, n_states)

                        states_predictions.append(state_prediction[0])

                    else:
                        # if the trace is static or the number of states is not 2,3,4, predict all zeros for the states
                        state_prediction = np.zeros_like(trace[:, 0])
                        n_states_prediction[index] = "N/A"

                        states_predictions.append(state_prediction)

                    if progress_callback is not None:
                        progress = int(((index + 1) / len(traces)) * 33) + 67
                        progress_callback.emit(progress)

            else:
                """Predict the states of all traces one at a time"""

                for index, trace in enumerate(traces):
                    trace = tf.convert_to_tensor([trace], dtype=tf.float32)

                    # predict whether a trace is dynamic or static
                    trace_pred, trace_conf, trace_label = self._predict_trace_label(trace, verbose=False)

                    if trace_label[0] != "dynamic":
                        # if the trace is static, predict all zeros for the states
                        n_states_pred = "N/A"
                        state_prediction = [np.zeros_like(trace[0, :, 0])]

                    else:
                        if n_states is None:
                            # predict the number of states in the trace
                            n_states_pred, n_states_conf = self._predict_n_states(trace, verbose=False)
                        else:
                            # use the user defined number of states
                            n_states_pred = [n_states]
                            n_states_conf = [1]

                        if n_states_pred[0] not in [2, 3, 4]:
                            # DeepLASI only supports 2, 3, or 4 states
                            state_prediction = [np.zeros_like(trace[0, :, 0])]
                        else:
                            # predict the states of the trace
                            state_prediction, state_confidence = self._predict_states(trace, n_states_pred[0])

                    states_predictions.append(state_prediction[0])
                    trace_labels.append(trace_label[0])

                    if progress_callback is not None:
                        progress = int(((index + 1) / len(traces)) * 100)
                        progress_callback.emit(progress)

        self.deeplasi_states = states_predictions
        self.deeplasi_labels = trace_labels

        return self.deeplasi_states, self.deeplasi_labels
