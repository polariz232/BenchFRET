import numpy as np
from scipy import signal
import pandas as pd
import multiprocessing as mp
import pomegranate as pg
from tqdm import tqdm
import os
import warnings
import platform
warnings.filterwarnings("ignore")


def simulate_2color_traces(
        n_traces,
        FRET_states=None,
        max_n_states=4,
        min_state_diff=0.1,
        d_lifetime=400,
        a_lifetime=400,
        blink_prob=0.1,
        crosstalk=(0, 0.4),
        dir_exc=(0, .2),
        gamma=(.5, 1.5),
        aa_offset=(.2, 2),
        n_frames=500,
        static_prob=.5,
        trans_mat=None,
        max_trans_prob=.2,
        noise=(.01, 1.2),
        gamma_noise_prob=.8,  
        aggregation_prob=0.15,
        max_aggregate_size=20,
        non_spFRET_E=-1,
        noise_tolerance=.3,
        s_tolerance=(0.05, 0.95),
        randomize_prob=0.25,
        bg_overestimate_prob=.1,
        falloff_lifetime=500,
        falloff_prob=0.1,
        balance_bleach=True,
        parallel_asynchronous=True,
        reduce_memory=True,
        merge_state_labels=True,
        state_mode=False,
        n_states_mode=False,
        omit_bleach=False,
        quenching=False
):
    eps = 1e-16
    # local function name needs to be set global in order for multiprocessing to work, not pretty but works
    global single_2color_trace

    def _E(DD, DA):
        return np.round(DA / (DD + DA), 5)

    def _S(DD, DA, AA):
        return (DD + DA) / (DD + DA + AA)

    def _DD(E):
        return 1 - E

    def _DA(DD, E):
        return -(DD * E) / (E - 1)

    def _AA(E):
        return np.ones(len(E))

    def generate_state_means(min_diff_E, n_states):
        """Returns random values and retries if they are too closely spaced"""
        states = np.random.uniform(0.01, 0.99, n_states)
        diff = np.diff(sorted(states))
        while any(diff < min_diff_E):
            states = np.random.uniform(0.01, 0.99, n_states)
            diff = np.diff(sorted(states))
        return states

    def generate_fret_states(mode, states_E=None, max_tprob=0, tmat=None):
        if states_E is not None:
            if mode == "aggregate":
                n_states = 1
                states_E = np.random.uniform(0.01, .99)
            else:
                n_states = np.size(states_E)
                states_E = np.array(states_E)
        else:
            if mode == "dynamic":
                if state_mode:
                    n_states = max_n_states
                else:
                    n_states = np.random.randint(2, max_n_states + 1)
            else:  # static
                n_states = 1
            states_E = generate_state_means(min_state_diff, n_states)
        if type(states_E) == float:
            dists = [pg.NormalDistribution(states_E, eps)]
        else:
            try:
                dists = [pg.NormalDistribution(m, eps) for m in states_E]
            except TypeError:
                dists = [pg.NormalDistribution(states_E, eps)]
        seed_val = int.from_bytes(os.urandom(4), byteorder="little")
        np.random.seed(seed_val)
        # randomized transition probability matrix
        if mode == "dynamic":
            if tmat is None:
                trans_prob = 1 / np.random.randint(1, 100)
                tmat = np.random.uniform(trans_prob, trans_prob, [n_states, n_states])
                for i in range(n_states):
                    tmat[i, i] = 0
                    tmat[i, i] = 1 - np.sum(tmat[i, :])
            else:
                tmat = np.array(tmat)

            # calculate equilibrium probability of states
            tmat_temp = np.vstack((tmat, [np.ones(n_states)]))
            b = np.ones((n_states + 1, 1))
            b[:n_states] = 0
            p_eq = np.linalg.lstsq(tmat_temp, b)[0]
            p_eq += (1 - np.sum(p_eq)) / n_states
        else:
            tmat = np.array([[1.0]])
            p_eq = np.array([1.0])

        model = pg.HiddenMarkovModel.from_matrix(
            tmat, distributions=dists, starts=p_eq
        )
        model.bake()
        final_matrix = model.dense_transition_matrix()[:n_states, :n_states]
        E_true = np.array(model.sample(n=1, length=n_frames))
        E_true = np.round(np.squeeze(E_true), 5)
        return E_true, final_matrix

    def randomize(DD, DA, AA, classifications, label, bg_overestimated):
        channels = [DD, DA, AA]
        if bg_overestimated:  # overestimated background subtraction
            if np.random.uniform(0, 1) < .5:  # donor channel
                lower = np.min(channels[0]) + .1
                channels[0] -= np.random.uniform(lower, lower+.4)
            else:  # acceptor channel
                lower = np.min(channels[1]) + .1
                overest = np.random.uniform(lower, lower+.4)
                channels[1] -= overest
                channels[2] -= overest
            DD, DA, AA = channels
        else:
            for idx, _ in enumerate(channels):
                if np.random.uniform(0, 1) < .9:  # merge with random walks
                    rndwalk = np.cumsum(np.random.normal(loc=0, scale=1, size=n_frames))
                    rndwalk -= np.min(rndwalk)
                    rndwalk /= np.max(rndwalk)
                    rndwalk *= np.random.uniform(.3, 1)
                    rndwalk += np.random.uniform(-.2, .2)
                    rndwalk[channels[idx] == 0] = 0
                    channels[idx] += rndwalk
                if np.random.uniform(0, 1) < .15:  # uncorrelated dynamics
                    frames = np.linspace(0, 1, n_frames, endpoint=False)
                    for _ in range(np.random.randint(1, 3)):
                        pulsewave = signal.square(frames * np.pi * np.random.randint(1, 50)) * np.random.uniform(-0.15, .15)
                        pulsewave[channels[idx] == 0] = 0
                        channels[idx] += pulsewave
                if np.random.uniform(0, 1) < .2:  # mirror
                    channels[idx] = channels[idx][::-1]
            DD, DA, AA = [np.where(x < 0, 0, x) for x in channels]
        label.fill(classifications["randomized"])
        if not omit_bleach:
            S_true = (DD + DA) / (DD + DA + AA)
            label[S_true < np.min(s_tolerance)] = classifications["don bleached"]
            label[S_true > np.max(s_tolerance)] = classifications["acc bleached"]
            label[AA == 0] = classifications["acc bleached"]
            label[DD == 0] = classifications["don bleached"]
            label[(DD == 0) & (DA == 0) & (AA == 0)] = classifications["all bleached"]
        return DD, DA, AA, label

    def check_states(trace, n_states_mode):
        
        if n_states_mode:
            trace["label"][trace["label"] == 7] = 0
            trace["label"][trace["label"] == 8] = 1
            trace["label"][trace["label"] == 9] = 2
            trace["label"][trace["label"] == 10] = 3
        if merge_state_labels:
            trace["label"][trace["label"] > 7] = 7
        
        return trace

    def single_2color_trace():
        if state_mode:
            classifications = {
                "1-state": 0,
                "2-state": 1,
                "3-state": 2,
                "4-state": 3,
                "don bleached": 4,
                "acc bleached": 5,
                "all bleached": 6,
                "aggregate": 7,
                "randomized": 8,
                "noisy": 9,
            }
        else:
            classifications = {
                "don bleached": 0,
                "acc bleached": 1,
                "all bleached": 2,
                "aggregate": 3,
                "randomized": 4,
                "noisy": 5,
                "1-state": 6,
                "2-state": 7,
                "3-state": 8,
                "4-state": 9,
                "5-state": 10,
            }

        frames = np.arange(1, n_frames + 1, 1)

        if np.random.uniform(0, 1) < aggregation_prob and not any((state_mode, n_states_mode)):
            is_aggregated = True
            E_true, matrix = generate_fret_states(
                mode="aggregate",
                states_E=FRET_states,
                tmat=trans_mat,
            )
            if max_aggregate_size >= 2:
                aggregate_size = np.random.randint(2, max_aggregate_size + 1)
            else:
                raise ValueError("Can't have an aggregate of size less than 2")
            n_pairs = np.random.poisson(aggregate_size)
            if n_pairs == 0:
                n_pairs = 2
        else:
            is_aggregated = False
            n_pairs = 1
            if np.random.uniform(0, 1) < static_prob and not any((state_mode, n_states_mode)):
                is_dynamic = False
                E_true, matrix = generate_fret_states(
                    mode="static",
                    states_E=FRET_states,
                    tmat=trans_mat,
                )
            else:
                is_dynamic = True
                E_true, matrix = generate_fret_states(
                    mode="dynamic",
                    states_E=FRET_states,
                    tmat=trans_mat,
                    max_tprob=max_trans_prob,
                )

        DD_total, DA_total, AA_total = [], [], []
        first_bleach_all = []

        for j in range(n_pairs):
            if d_lifetime is not None and not any((state_mode, n_states_mode)):
                bleach_D = int(np.ceil(np.random.exponential(d_lifetime)))
            else:
                bleach_D = np.nan

            if a_lifetime is not None and not any((state_mode, n_states_mode)):
                bleach_A = int(np.ceil(np.random.exponential(a_lifetime)))
            else:
                bleach_A = np.nan

            if balance_bleach and not any((state_mode, n_states_mode)):
                if not is_aggregated:
                    # increase rate of non-bleached traces for later balancing
                    if np.random.uniform(0, 1) < .4:
                        bleach_D = np.nan
                        bleach_A = np.nan

            first_bleach = min((bleach_A, bleach_D))

            # keep track of multiple fluorophores for aggregates
            first_bleach_all.append(first_bleach)

            # initialize intensities
            DD = _DD(E_true)
            DA = _DA(DD, E_true)
            AA = _AA(E_true)

            # force bleaching step for small aggregates
            if j == 1 and n_pairs == 2:
                while (bleach_D > n_frames) and (bleach_A > n_frames):
                    bleach_D = int(np.ceil(np.random.exponential(d_lifetime)))
                    bleach_A = int(np.ceil(np.random.exponential(a_lifetime)))
                first_bleach = min((bleach_D, bleach_A))

            # If donor bleaches first
            if first_bleach is not np.nan:
                if first_bleach == bleach_D:
                    DD[bleach_D:] = 0
                    DA[bleach_D:] = 0

                elif first_bleach == bleach_A:
                    DD[bleach_A:bleach_D] = 1
                    if is_aggregated and n_pairs <= 2:
                        spike_len = np.min((np.random.randint(2, 10), bleach_D))
                        DD[bleach_A:bleach_A + spike_len] = 2

                if bleach_D is not np.nan:
                    DD[bleach_D:] = 0
                    DA[bleach_D:] = 0
                if bleach_A is not np.nan:
                    DA[bleach_A:] = 0
                    AA[bleach_A:] = 0

            # Append to total fluorophore intensity per channel
            DD_total.append(DD)
            DA_total.append(DA)
            AA_total.append(AA)

        DD, DA, AA = [np.sum(x, axis=0) for x in (DD_total, DA_total, AA_total)]

        # Initialize labels
        label = np.zeros(n_frames)
        label.fill(-1)

        if is_aggregated:
            if np.random.uniform(0, 1) < falloff_prob:
                if falloff_lifetime is not None:
                    falloff_frame = int(
                        np.ceil(np.random.exponential(falloff_lifetime))
                    )
                else:
                    falloff_frame = None
                DD[falloff_frame:] = 0
                DA[falloff_frame:] = 0
                AA[falloff_frame:] = 0

            # First bleaching for each channel
            bleach_DD_all = np.argmax(DD == 0)
            bleach_DA_all = np.argmax(DA == 0)
            bleach_AA_all = np.argmax(AA == 0)

            # Find first bleaching overall
            first_bleach_all = min(
                (bleach_DD_all, bleach_DA_all, bleach_AA_all)
            )
            if first_bleach_all == 0:
                first_bleach_all = np.nan
            label.fill(classifications["aggregate"])
        else:
            first_bleach_all = min(first_bleach_all) if not np.isnan(first_bleach_all).any() else None

        # Blinking
        if not any((is_aggregated, state_mode, n_states_mode)) and np.random.uniform(0, 1) < blink_prob:
            blink_start = np.random.randint(1, n_frames-2)
            blink_time = np.random.randint(2, n_frames/5)

            if np.random.uniform(0, 1) < .5:
                DD[blink_start: (blink_start + blink_time)] = 0
                DA[blink_start: (blink_start + blink_time)] = 0
            else:
                DA[blink_start: (blink_start + blink_time)] = 0
                AA[blink_start: (blink_start + blink_time)] = 0

        if not is_aggregated:
            if bleach_A is not np.nan:
                label[bleach_A:] = classifications["acc bleached"]
                E_true[bleach_A:] = non_spFRET_E

            if bleach_D is not np.nan:
                label[bleach_D:] = classifications["don bleached"]
                E_true[bleach_D:] = non_spFRET_E

        if not omit_bleach:
            label[DD == 0] = classifications["don bleached"]
            label[AA == 0] = classifications["acc bleached"]
            label[(DD == 0) & (DA == 0) & (AA == 0)] = classifications["all bleached"]

        if is_aggregated:
            bleach_bool = np.isin(label, [0, 1, 2])
            if np.any(bleach_bool):
                first_bleach_all = np.argmax(bleach_bool)
            else:
                first_bleach_all = None

        # Random artifacts
        is_randomized = False
        if np.random.uniform(0, 1) < randomize_prob and n_pairs <= 2 and not any((state_mode, n_states_mode)):
            bg_overestimated = np.random.uniform(0, 1) < bg_overestimate_prob
            DD, DA, AA, label = randomize(
                DD=DD,
                DA=DA,
                AA=AA,
                classifications=classifications,
                label=label,
                bg_overestimated=bg_overestimated
            )
            is_randomized = True

        ### Future release: Include more traces starting with dark frames ###
        # if not is_aggregated:
        #     if np.random.uniform(0, 1) < .2:
        #         DD = DD[::-1]
        #         DA = DA[::-1]
        #         AA = AA[::-1]
        #         E_true = E_true[::-1]
        #         label = label[::-1]

        D_active = np.nonzero(DD)[0]
        A_active = np.nonzero(AA)[0]
        all_active = np.intersect1d(D_active, A_active)

        E_unbleached_true = E_true[all_active]
        if quenching and not any((is_aggregated, is_randomized)):
            if np.random.uniform(0, 1) < .33:
                # simulate quenching events
                # pulsewave = signal.square(frames * np.pi * np.random.randint(1, 100)) * np.random.uniform(-0.2, .2)
                AA *= _DD(E_true)
        unique_states, unique_states_idx = np.unique(
            E_unbleached_true[E_unbleached_true != non_spFRET_E], return_index=True
        )

        AA *= np.random.uniform(aa_offset[0], aa_offset[1])
        fluct_probs = np.random.uniform(0, 1, 2)
        if fluct_probs[0] < .3:  # green
            fluct = np.sin(np.linspace(0, np.random.randint(10, 500), n_frames)) * np.random.uniform(.0, .02) + 1
            fluct[DD == 0] = 0
            fluct[DA == 0] = 0
            DD *= fluct
            DA *= fluct
        if fluct_probs[1] < .3:  # red
            fluct = np.sin(np.linspace(0, np.random.randint(10, 500), n_frames)) * np.random.uniform(.0, .02) + 1
            fluct[AA == 0] = 0
            AA *= fluct

        # correction factors
        # gamma
        g = np.random.uniform(gamma[0], gamma[1])
        DA *= g
        # crosstalk
        ct = np.random.uniform(crosstalk[0], crosstalk[1])
        I_ct = DD * ct / (1 + ct)
        DA += I_ct
        DD -= I_ct
        # direct excitation
        de = np.random.uniform(dir_exc[0], dir_exc[1])
        DA[A_active] += de * AA[A_active]

        # Add noise
        bg_count = np.random.randint(100, 1000)
        noise_scale_intensity = np.random.uniform(0, .5)
        noise_scale_gamma = np.random.uniform(.0, .5)
        bg_noise_scale = np.random.uniform(noise[0], noise[1])
        for idx, s in enumerate([DD, DA, AA]):
            bg_poisson = np.random.poisson(bg_count, n_frames).astype('float') - bg_count
            bg_poisson /= bg_poisson.max()
            bg_poisson *= bg_noise_scale
            gaussian_noise = 0 if np.random.uniform(0, 1) < .5 else np.random.normal(0, noise_scale_intensity, n_frames) * s
            gamma_noise = 0 if np.random.uniform(0, 1) < gamma_noise_prob else np.random.gamma(1, noise_scale_gamma, len(s))
            s += gamma_noise
            s -= np.mean(gamma_noise)
            s += bg_poisson + gaussian_noise

        E_obs = _E(DD, DA)
        E_obs_active = E_obs[all_active]
        if first_bleach_all is not None:
            E_obs[all_active] = np.nan

        # noise labeling
        is_noisy = False
        states_std_all = -1
        if not any((is_aggregated, is_randomized)):
            states_std = []
            for state in unique_states:
                state_std = np.nanstd(E_obs_active[E_unbleached_true == state])
                states_std.append(state_std)
            states_std_min = np.min(states_std)
            if states_std_min > noise_tolerance:
                label[
                    (label != classifications["don bleached"]) &
                    (label != classifications["acc bleached"]) &
                    (label != classifications["all bleached"])
                    ] = classifications["noisy"]
                is_noisy = True
                if state_mode or n_states_mode:
                    return

        if state_mode:
            k_states = len(unique_states)
            sorted_states = E_unbleached_true[np.sort(unique_states_idx)]
            for state in range(k_states):
                label[
                    (label != classifications["don bleached"]) &
                    (label != classifications["acc bleached"]) &
                    (label != classifications["all bleached"]) &
                    (E_true == sorted_states[state])
                    ] = classifications[f"{state+1}-state"]
        else:
            if not any((is_noisy, is_aggregated, is_randomized)):
                if n_states_mode & (len(unique_states) == 1):
                    return
                for i in range(5):
                    n_states = i + 1
                    if len(unique_states) == n_states:
                        label[
                            (label != classifications["don bleached"]) &
                            (label != classifications["acc bleached"]) &
                            (label != classifications["all bleached"])
                            ] = classifications[f"{n_states}-state"]

        try:
            if state_mode:
                ediff = np.min(np.diff(np.unique(E_unbleached_true)))
            else:
                if label[0] in [7, 8, 9]:
                    ediff = np.min(np.diff(np.unique(E_unbleached_true)))
                else:
                    ediff = -1
        except ValueError:
            ediff = -1

        if any((is_noisy, is_aggregated, is_randomized)):
            E_true.fill(-1)

        if any((is_noisy, is_randomized)):
            E_true[label ==
                   classifications["don bleached"] |
                   classifications["acc bleached"] |
                   classifications["all bleached"]] = -1

        if reduce_memory:
            trace = pd.DataFrame(
                {
                    "DD": DD,
                    "DA": DA,
                    "AA": AA,
                    "E": E_obs,
                    "label": label,
                }
            )
        else:
            trace = pd.DataFrame(
                {
                    "DD": DD,
                    "DA": DA,
                    "AA": AA,
                    "E": E_obs,
                    "E_true": E_true,
                    "label": label,
                    "_noise_level": np.array(states_std_all).repeat(n_frames),
                    "_min_E_diff": np.array(ediff).repeat(n_frames),
                    "trans_mean": np.array((matrix[0, 1]+matrix[1, 0])/2).repeat(n_frames),
                }
            )
        trace.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
        trace.fillna(method="pad", inplace=True)

        return trace, matrix
    
    
    traces_list = []
    matrix_list = []        

    if parallel_asynchronous:
        if platform.processor() == 'arm':  # m1 macs
            ctx = mp.get_context("fork")
            pool = ctx.Pool(mp.cpu_count())
        else:  # intel cpus
            pool = mp.Pool(mp.cpu_count())
        jobs = [pool.apply_async(single_2color_trace) for _ in range(n_traces)]
        pool.close()
        trace_index = 0
        for index, job in enumerate(tqdm(jobs)):
            trace,matrix = job.get()
            if type(trace) != type(None):
                trace["trace_index"] = trace_index
                traces = check_states(trace, n_states_mode)
                traces_list.append(trace)
                matrix_list.append(matrix)
                trace_index +=1
        pool.join()
    else:
        rng = np.random.default_rng()
        rng.random()
        trace_index = 0
        for _ in tqdm(range(n_traces)):
            try: 
                trace,matrix = single_2color_trace()
                trace["trace_index"] = trace_index
                traces = check_states(trace, n_states_mode)
                traces_list.append(trace)
                matrix_list.append(matrix)
                trace_index +=1
            except TypeError: 
                pass
            
    
    return traces_list, matrix_list
