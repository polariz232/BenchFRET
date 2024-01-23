import numpy as np
import traceback
import abc
import pomegranate as pg
import numpy as np
import hmmlearn.hmm as hmm
import itertools
from DeepGapSeq._utils_worker import Worker
from functools import partial
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


class _HMM_methods:

    def get_combo_box_items(self, combo_box):
        items = []
        for index in range(combo_box.count()):
            items.append(combo_box.itemText(index))
        return items


    def update_hmm_fit_mode(self):

        try:

            hmm_algorithm = self.fitting_window.hmm_algorithm.currentText()

            if hmm_algorithm == "Pomegranate":
                hmm_fit_modes = ["Baum-Welch", "Viterbi", "Auto"]
            elif hmm_algorithm == "HMM Learn":
                hmm_fit_modes = ["Diag", "Spherical", "Full", "Tied", "Auto"]

            self.fitting_window.hmm_fit_mode.blockSignals(True)
            self.fitting_window.hmm_fit_mode.clear()
            self.fitting_window.hmm_fit_mode.addItems(hmm_fit_modes)
            self.fitting_window.hmm_fit_mode.blockSignals(False)

        except:
            pass


    def populate_HMM_options(self):

        try:
            if self.data_dict != {}:

                dataset_names = list(self.data_dict.keys())

                self.fitting_window.hmm_fit_data.clear()

                # get plot_mode combo box items
                plot_mode_items = self.get_combo_box_items(self.plot_mode)

                if "Trace" in plot_mode_items:
                    self.fitting_window.hmm_fit_data.addItem("Trace")
                if "Donor" in plot_mode_items:
                    self.fitting_window.hmm_fit_data.addItem("Donor")
                if "Acceptor" in plot_mode_items:
                    self.fitting_window.hmm_fit_data.addItem("Acceptor")
                if set(["Donor", "Acceptor"]).issubset(set(plot_mode_items)):
                    self.fitting_window.hmm_fit_data.addItem("FRET")
                if "FRET Efficiency" in plot_mode_items:
                    self.fitting_window.hmm_fit_data.addItem("FRET Efficiency")
                if "DA" in plot_mode_items:
                    self.fitting_window.hmm_fit_data.addItem("DA")
                if "AA" in plot_mode_items:
                    self.fitting_window.hmm_fit_data.addItem("AA")
                if "DD" in plot_mode_items:
                    self.fitting_window.hmm_fit_data.addItem("DD")
                if "AD" in plot_mode_items:
                    self.fitting_window.hmm_fit_data.addItem("AD")
                if "ALEX Efficiency" in plot_mode_items:
                    self.fitting_window.hmm_fit_data.addItem("Alex Efficiency")

                self.fitting_window.hmm_fit_dataset.clear()
                self.fitting_window.hmm_fit_dataset.addItems(dataset_names)

                self.update_hmm_fit_mode()

        except:
            print(traceback.format_exc())
            pass

    def format_hmm_fit_data(self, fit_data, hmm_mode, crop_range=None):

        try:

            fit_data = np.array(fit_data)

            if hmm_mode == "Pomegranate":

                if len(fit_data.shape) != 2:
                    fit_data = np.expand_dims(fit_data, axis=1)

            elif hmm_mode == "HMM Learn":

                if len(fit_data.shape) != 2:
                    fit_data = np.expand_dims(fit_data, axis=1)

            if crop_range != None:
                if len(crop_range) == 2:
                    crop_range = sorted(crop_range)
                    if crop_range[0] < 0:
                        crop_range[0] = 0
                    if crop_range[1] > len(fit_data):
                        crop_range[1] = len(fit_data)

                    fit_data = fit_data[int(crop_range[0]):int(crop_range[1])]

        except:
            print(traceback.format_exc())
            pass

        return fit_data

    def get_hmm_fit_data(self, dataset_name=None, channel_name=None, hmm_mode = None,
            crop=None, concat_traces = None):

        try:

            if dataset_name is None:
                dataset_name = self.fitting_window.hmm_fit_dataset.currentText()
            if channel_name is None:
                channel_name = self.fitting_window.hmm_fit_data.currentText()
            if hmm_mode is None:
                hmm_mode = self.fitting_window.hmm_algorithm.currentText()
            if crop is None:
                crop = self.fitting_window.hmm_crop_plots.isChecked()

            fit_dataset = {}
            fit_dataset_lengths = []

            if dataset_name in self.data_dict.keys():

                for localisation_index, localisation_data in enumerate(self.data_dict[dataset_name]):

                    channel_keys = list(localisation_data.keys())

                    user_label = localisation_data["user_label"]
                    nucleotide_label = localisation_data["nucleotide_label"]

                    if crop == True:
                        crop_range = localisation_data["crop_range"]
                    else:
                        crop_range = None

                    if self.get_filter_status("hmm", user_label, nucleotide_label) == False:

                        if channel_name != "FRET":
                            if channel_name in channel_keys:

                                if localisation_index not in fit_dataset.keys():
                                    fit_dataset[localisation_index] = {}

                                fit_data = localisation_data[channel_name]

                                fit_data = self.format_hmm_fit_data(fit_data, hmm_mode, crop_range)

                                fit_dataset[localisation_index] = fit_data
                                fit_dataset_lengths.append(fit_data.shape[0])

                        else:
                            if "Donor" in channel_keys and "Acceptor" in channel_keys:

                                if localisation_index not in fit_dataset.keys():
                                    fit_dataset[localisation_index] = {}

                                donor_trace = localisation_data["Donor"]
                                acceptor_trace = localisation_data["Acceptor"]

                                fit_data = np.stack((donor_trace, acceptor_trace), axis=1)

                                fit_data = self.format_hmm_fit_data(fit_data, hmm_mode, crop_range)

                                fit_dataset[localisation_index] = fit_data
                                fit_dataset_lengths.append(fit_data.shape[0])

        except:
            print(traceback.format_exc())
            pass

        return fit_dataset

    def pg_fit_hmm(self, data, n_states, n_iter=1000, n_init=1,
            fit_mode="baum-welch"):

        try:
            # note that the data should be in the shape (trace,n_frames,features)

            best_score = float("-inf")
            best_model = None
            best_model_type = None
            best_n_states = None

            fit_mode = fit_mode.lower()

            if fit_mode == "auto":
                fit_mode_list = ["baum-welch", "viterbi"]
            else:
                fit_mode_list = [fit_mode]

            if type(n_states) == str:
                if n_states.lower() == "auto":
                    n_states = range(2, 5)
                else:
                    n_states = [int(n_states)]
            else:
                n_states = [int(n_states)]

            for mode in fit_mode_list:
                for state in n_states:

                    model = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution,
                        n_components=state,
                        max_iterations=n_iter,
                        X=data,
                        n_jobs=1,
                        algorithm=mode,
                        n_init=n_init,
                    )

                    score = model.log_probability(data)

                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_model_type = mode
                        best_n_states = state

        except:
            print(traceback.format_exc())
            model = None
            pass

        return best_model

    def pg_predict_hmm(self, model, data):

        try:

            if model is not None:

                predictions = model.predict(data)

            else:
                predictions = None

        except:
            print(traceback.format_exc())
            predictions = None
            pass

        return predictions

    def hmmlearn_fit_hmm(self, data, n_states, n_iter=1000, n_init=1,
    fit_mode="diag"):

        try:

            best_score = float("-inf")
            best_model = None
            best_model_type = None
            best_n_states = None

            fit_mode = fit_mode.lower()

            if fit_mode == "auto":
                fit_mode_list = ["spherical", "diag", "full", "tied"]
            else:
                fit_mode_list = [fit_mode]

            if type(n_states) == str:
                if n_states.lower() == "auto":
                    n_states = range(2, 5)
                else:
                    n_states = [int(n_states)]
            else:
                n_states = [int(n_states)]

            for mode in fit_mode_list:
                for state in n_states:
                    for i in range(n_init):

                        try:
                            model = hmm.GaussianHMM(
                                n_components=state,
                                n_iter=n_iter,
                                covariance_type=mode,
                                random_state=np.random.randint(0, 1000),
                                verbose=False,
                            ).fit(data)

                            score = model.score(data)

                            if score > best_score:
                                best_score = score
                                best_model = model
                                best_model_type = mode
                                best_n_states = state

                        except:
                            pass

        except:
            print(traceback.format_exc())
            model = None
            pass

        return best_model

    def hmmlearn_predict_hmm(self, model, data):

            try:

                if model is not None:

                    predictions = model.predict(data)

                else:
                    predictions = None

            except:
                print(traceback.format_exc())
                predictions = None
                pass

            return predictions

    def detect_hmm_states_finished(self):

        try:
            self.fitting_window.hmm_progressbar.setValue(0)
            self.fitting_window.hmm_detect_states.setEnabled(True)

            self.plot_traces(update_plot=True)

        except:
            print(traceback.format_exc())
            pass

    def count_transitions(self, data):

        # Count of transitions
        count = 0

        try:
            # Iterate through the list
            for i in range(1, len(data)):
                # Check if the current element is different from the previous one
                if data[i] != data[i - 1]:
                    count += 1

        except:
            print(traceback.format_exc())

        return count

    def post_process_hmm_predictions(self, predictions, min_length=2, max_transitions=100):

        try:

            array = np.array(predictions)

            # Find the indices where adjacent elements differ
            change_points = np.where(array[:-1] != array[1:])[0] + 1
            change_points = np.insert(change_points, 0, 0)

            if type(min_length) == str:
                if min_length.isdigit():
                    min_length = int(min_length)
                else:
                    min_length = 0
            else:
                min_length = int(min_length)

            new_predictions = predictions.copy()

            for index in range(len(change_points)-1):

                start_index = change_points[index]
                end_index = change_points[index+1]

                state_length = end_index - start_index

                if state_length <= min_length:

                    try:

                        previous_state = new_predictions[start_index-1]
                        next_state = new_predictions[end_index]

                        if previous_state == next_state:
                            new_predictions[int(start_index):int(end_index)] = int(previous_state)

                    except:
                        pass

            array = np.array(new_predictions)

            # Find the indices where adjacent elements differ
            change_points = np.where(array[:-1] != array[1:])[0] + 1
            change_points = np.insert(change_points, 0, 0)

            n_transitions = len(change_points)-1

            if type(max_transitions) == str:
                if max_transitions.isdigit():
                    max_transitions = int(max_transitions)
                else:
                    max_transitions = n_transitions
            else:
                max_transitions = int(max_transitions)

            if n_transitions > max_transitions:
                new_predictions = np.zeros_like(predictions).tolist()

        except:
            print(traceback.format_exc())
            new_predictions = predictions
            pass

        return new_predictions

    def detect_hmm_states(self):

        try:

            self.fitting_window.hmm_detect_states.setEnabled(False)

            worker = Worker(self._detect_hmm_states)
            worker.signals.progress.connect(partial(self.gui_progrssbar,name="hmm"))
            worker.signals.finished.connect(self.detect_hmm_states_finished)
            self.threadpool.start(worker)

        except:
            self.fitting_window.hmm_detect_states.setEnabled(True)
            self.fitting_window.hmm_progressbar.setValue(0)
            print(traceback.format_exc())
            pass

    def find_peaks_with_moving_average(self, data, window_size=5, threshold=0.2):
        """
        Find peaks in 1D data using a moving average filter.

        Parameters:
        data (numpy.array): 1D array of data points.
        window_size (int): Size of the moving average window.
        threshold (float): Threshold value to identify a peak.

        Returns:
        numpy.array: Indices of the peaks.
        """
        # Calculate the moving average using a convolution
        moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

        # Find where the data exceeds the moving average by the threshold
        peaks = np.where(data[window_size - 1:] > moving_avg + threshold)[0]
        # Adjust the indices because of the convolution 'valid' mode
        peaks += window_size - 1

        return peaks

    def _detect_hmm_states(self, progress_callback=None):

        try:

            dataset_name = self.fitting_window.hmm_fit_dataset.currentText()

            hmm_mode = self.fitting_window.hmm_algorithm.currentText()
            n_states = self.fitting_window.hmm_n_states.currentText()
            n_init = int(self.fitting_window.hmm_n_init.text())
            fit_mode = self.fitting_window.hmm_fit_mode.currentText()
            concat_traces = self.fitting_window.hmm_concat_traces.isChecked()
            n_iter = int(self.fitting_window.hmm_n_iterations.text())
            min_length = self.fitting_window.hmm_min_length.text()
            max_transitions = self.fitting_window.hmm_max_transitions.text()

            fit_dataset = self.get_hmm_fit_data()

            n_traces = len(fit_dataset.keys())

            hmm_models = {}

            self.print_notification("Fitting HMM Model(s)...")

            if concat_traces == False:

                for fit_index, (trace_index, fit_data) in enumerate(fit_dataset.items()):

                    if hmm_mode == "Pomegranate":
                        model = self.pg_fit_hmm(fit_data, n_states, n_iter, n_init, fit_mode)
                    elif hmm_mode == "HMM Learn":
                        model = self.hmmlearn_fit_hmm(fit_data, n_states, n_iter, n_init, fit_mode)

                    hmm_models[trace_index] = model

                    progress = int(100*(fit_index+1)/n_traces)

                    if progress_callback is not None:
                        progress_callback.emit(progress)

            else:

                fit_data = np.concatenate(list(fit_dataset.values()), axis=0)

                if hmm_mode == "Pomegranate":
                    model = self.pg_fit_hmm(fit_data, n_states)
                elif hmm_mode == "HMM Learn":
                    model = self.hmmlearn_fit_hmm(fit_data, n_states)

                if model != None:
                    for trace_index in fit_dataset.keys():
                        hmm_models[trace_index] = model

                if progress_callback is not None:
                    progress_callback.emit(100)

            self.print_notification("Predicting HMM states...")

            for fit_index, (trace_index, fit_data) in enumerate(fit_dataset.items()):

                localisation_data = self.data_dict[dataset_name][trace_index]

                model = hmm_models[trace_index]

                predictions = np.zeros_like(fit_data.shape[0]).tolist()

                if model != None:

                    if hmm_mode == "Pomegranate":
                        predictions = self.pg_predict_hmm(model, fit_data)
                    elif hmm_mode == "HMM Learn":
                        predictions = self.hmmlearn_predict_hmm(model, fit_data)

                    if predictions is not None:
                        if len(predictions) == len(fit_data):

                            pass

                            predictions = self.post_process_hmm_predictions(predictions,
                                min_length, max_transitions)

                localisation_data["states"] = predictions
                progress = int(100*(fit_index+1)/n_traces)

                if progress_callback is not None:
                    progress_callback.emit(progress)


            self.print_notification("Computing state means...")
            self.compute_state_means(dataset_name=dataset_name)

            self.print_notification("State detection complete.")

        except:
            self.fitting_window.hmm_detect_states.setEnabled(True)
            print(traceback.format_exc())
            pass


