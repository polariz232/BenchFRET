import abc
import pomegranate as pg
import numpy as np 
import hmmlearn.hmm as hmm
import itertools

def permute_array_values(arr, n):
    # Generate all permutations of the range of values
    value_permutations = itertools.permutations(range(n))

    # Initialize an empty list to store the permuted arrays
    permuted_arrays = []

    for perm in value_permutations:
        # Create a mapping from original values to permuted values
        mapping = np.array(perm)

        # Apply the mapping to the array
        permuted_array = mapping[arr.astype(int)]
        permuted_arrays.append(permuted_array)

    return permuted_arrays

class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def predict(self, data):
        pass

    @abc.abstractmethod
    def get_performance():
        pass

class HMM_pg(Model):
    def __init__(self, n_states=2,algorithem='baum-welch',stop_threshold=1E-4, max_iterations=100):
        self.n_states = n_states
        self.algorithem = algorithem
        self.model = None

    def fit(self,data):
        self.model = pg.HiddenMarkovModel.from_samples( #note that the data should be in the shape (trace,n_frames,features)
            pg.NormalDistribution,
            n_components=self.n_states,
            X=data,
            n_jobs=-1,
            algorithm=self.algorithem
            )
        return self.model

    def predict(self,data):
        return np.array(self.model.predict(data))
   
    def get_performance(self,predicted_states, labels, verbose=False, best_only=True): 
        score = []
        permutation_of_labels = permute_array_values(labels, self.n_states)
        for perm in permutation_of_labels:
            mark = 0
            for i in range(predicted_states.shape[0]):
                if predicted_states[i] == perm[i]:
                    mark += 1
            mark = mark/len(predicted_states)
            score.append(mark)
        if best_only:
            best = np.argmax(score)
            score = score[best]
            tmat = self.model.dense_transition_matrix()[best]
            aligned_labels = permutation_of_labels[best]    
            if verbose:
                print(f'correct estimation rate: {score}')
                print(f'fitted transition matrix:\n{tmat}')
            return aligned_labels, score, tmat
        if verbose:
            print(f'correct estimation rate: {score}')
            print(f'fitted transition matrix:\n{tmat}')    
        return score, tmat


                

class HMM_learn():
    def __init__(self,n_states=2):
        self.model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            min_covar=100,
            tol = 0.0001,
            n_iter=1000,
            init_params="stmc",  # auto init all params
            algorithm="viterbi")

    def train_several_times(self,data,lengths,n_times):
        for i in range(n_times):
            remodel = self.model
            remodel.fit(data,lengths)
            if remodel.score(data,lengths) > self.model.score(data,lengths):
                self.model = remodel
        
    def fit(self,data,lengths,several_times=False):
        data = data - np.mean(data)
        data = data / np.std(data)
        self.model = self.model.fit(data,lengths)
        if several_times:
            self.train_several_times(data,lengths,10)

    def predict(self,data,lengths=None):
        data = data - np.mean(data)
        data = data / np.std(data)
        return self.model.predict(data,lengths)

    def get_performance(self,predicted_states, labels):
        score = 0
        for i in range(predicted_states.shape[0]):
            if predicted_states[i] == labels[i]:
                score += 1
        return score/len(predicted_states)    
        
class Model_selector():
    pass

