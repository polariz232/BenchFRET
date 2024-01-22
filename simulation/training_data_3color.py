import numpy as np
from scipy import signal
import pandas as pd
import multiprocessing as mp
import pomegranate as pg
from tqdm import tqdm
import warnings
import os
import platform
warnings.filterwarnings("ignore")

# DEFAULT PARAMETERS #
# n_traces,
# states_E_bg=None,
# states_E_br=None,
# states_E_gr=None,
# max_n_states=4,
# min_state_diff=0.1,
# static_dyes="random",
# b_lifetime=400,
# g_lifetime=400,
# r_lifetime=400,
# blink_prob=0.2,
# crosstalk_bg=(.0, .6),
# crosstalk_br=(.0, .1),
# crosstalk_gr=(.0, .3),
# dir_exc_bg=(.0, .3),
# dir_exc_br=(.0, .1),
# dir_exc_gr=(.0, .3),
# gamma_bg=(.9, 2),
# gamma_gr=(.6, 1.4),
# bb_offset=(.7, 1),
# gg_offset=(1, 1),
# rr_offset=(.5, 3),
# n_frames=500,
# trans_mat=None,
# trans_mat_bg=None,
# trans_mat_br=None,
# max_trans_prob=.2,
# static_prob=.5,
# noise=(.0, 1),
# gamma_noise_prob=0.8,
# noise_tolerance=.25,
# aggregation_prob=0.1,
# max_aggregate_size=20,
# non_spFRET_E=-1,
# s_tolerance_bg=(0.1, 0.9),
# s_tolerance_br=(0.1, 0.9),
# s_tolerance_gr=(0.1, 0.9),
# randomize_prob=.25,
# bg_overestimate_prob=.15,
# quenching=True,
# falloff_lifetime=500,
# falloff_prob=0.1,
# balance_bleach=True,
# parallel_asynchronous=True,
# reduce_memory=True,
# state_mode=False,
# n_states_mode=False,
# train_BR=False,
# train_BG=False
#############################

def simulate_3color_traces(
    n_traces,
    states_E_bg=None,
    states_E_br=None,
    states_E_gr=None,
    max_n_states=4,
    min_state_diff=0.1,
    static_dyes="random",
    b_lifetime=400,
    g_lifetime=400,
    r_lifetime=400,
    blink_prob=0.2,
    crosstalk_bg=(.0, .6),
    crosstalk_br=(.0, .1),
    crosstalk_gr=(.0, .3),
    dir_exc_bg=(.0, .3),
    dir_exc_br=(.0, .1),
    dir_exc_gr=(.0, .3),
    gamma_bg=(.9, 2),
    gamma_gr=(.6, 1.4),
    bb_offset=(.7, 1),
    gg_offset=(1, 1),
    rr_offset=(.5, 3),
    n_frames=500,
    trans_mat=None,
    trans_mat_bg=None,
    trans_mat_br=None,
    max_trans_prob=.2,
    static_prob=.5,
    noise=(.0, 1),
    gamma_noise_prob=0.8,
    noise_tolerance=.25,
    aggregation_prob=0.1,
    max_aggregate_size=20,
    non_spFRET_E=-1,
    s_tolerance_bg=(0.1, 0.9),
    s_tolerance_br=(0.1, 0.9),
    s_tolerance_gr=(0.1, 0.9),
    randomize_prob=.25,
    bg_overestimate_prob=.15,
    quenching=True,
    falloff_lifetime=500,
    falloff_prob=0.1,
    balance_bleach=True,
    parallel_asynchronous=True,
    reduce_memory=True,
    state_mode=False,
    n_states_mode=False,
    train_BR=False,
    train_BG=False
):
    eps = 1e-16

    # local function name needs to be set global in order for multiprocessing to work
    global single_3color_trace

    def _E_obs_bg(BB, BG, BR):
        return np.round(BG / (BB + BG + BR), 5)

    def _E_obs_br(BB, BG, BR):
        return np.round(BR / (BB + BG + BR), 5)

    def _E_bg(BB, BG, E_gr):
        return BG / (BB * (1 - E_gr) + BG)

    def _E_br(BB, BG, BR, E_gr):
        return (BR - E_gr * (BG + BR)) / (BB + BR - E_gr * (BB + BG + BR))

    def _E_gr(GG, GR):
        return GR / (GG + GR)

    def _E_app_bg(E_true_bg, E_true_br):
        return np.round(E_true_bg * (1 - E_true_br) / (1 - E_true_bg * E_true_br), 5)

    def _E_app_br(E_true_bg, E_true_br):
        return np.round(E_true_br * (1 - E_true_bg) / (1 - E_true_bg * E_true_br), 5)

    def _S_bg(BB, BG, BR, GG, GR):
        return (BB + BG + BR) / (BB + BG + BR + GG + GR)

    def _S_br(BB, BG, BR, RR):
        return (BB + BG + BR) / (BB + BG + BR + RR)

    def _S_gr(GG, GR, RR):
        return (GG + GR) / (GG + GR + RR)

    def _BB(E_app_bg, E_app_br):
        return np.round(1 - (E_app_bg + E_app_br), 5)

    def _GG(E_gr):
        return np.round(1 - E_gr, 5)

    def _BG(E_app_bg, E_gr):
        return np.round(E_app_bg * (1 - E_gr), 5)  # * (1 - E_gr)

    def _BG_2color(BB, E_bg):
        return np.round(-(BB * E_bg) / (E_bg - 1), 5)

    def _BR(E_app_br, E_app_bg, E_gr):
        return np.round(E_app_br + E_app_bg * E_gr, 5)  # * E_gr

    def _BR_2color(BB, E_br):
        return np.round(-(BB * E_br) / (E_br - 1), 5)

    def _GR(GG, E_gr):
        return np.round(-(GG * E_gr) / (E_gr - 1), 5)

    def _RR(E_gr):
        return np.ones(len(E_gr))

    def _calc_ct(DD, ct):
        # DD[DD < 0] = 0
        return np.round(DD * ct / (1 + ct), 5)

    def _calc_de(AA, de):
        # AA[AA < 0] = 0
        return np.round(AA * de, 5)

    def generate_states_E(min_diff_E, n_states):
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
            if mode == "alldynamic":
                # two dyes moving == 4 states
                # three dyes moving == 9 states
                # Hence limit to 2 states per dye pair
                # otherwise traces are indistinguishable from random noise
                n_states = 2
            elif mode == "dynamic":
                if state_mode:
                    n_states = max_n_states
                else:
                    n_states = np.random.randint(2, max_n_states + 1)
            else:  # static or aggregate
                n_states = 1
            states_E = generate_states_E(min_state_diff, n_states)
        if np.size(states_E) == 1:
            dists = [pg.NormalDistribution(states_E, eps)]
        else:
            dists = [pg.NormalDistribution(m, eps) for m in states_E]
        seed_val = int.from_bytes(os.urandom(4), byteorder="little")
        np.random.seed(seed_val)
        # randomized transition probability matrix
        if mode == "dynamic" or mode == "alldynamic":
            if tmat is None:
                if state_mode:
                    trans_prob = 1/np.random.randint(1, 100)
                    tmat = np.random.uniform(trans_prob, trans_prob, [n_states, n_states])
                else:
                    tmat = np.random.uniform(.0, max_tprob, [n_states, n_states])
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
            p_eq += (1-np.sum(p_eq)) / n_states
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

    def rnd_scale_E(E_in, low_lim_E=None):
        n_states = len(np.unique(E_in))
        if low_lim_E is None:
            delta = np.random.uniform(0.1 * (n_states - 1), .98)
            lower = np.random.uniform(.01, .99 - delta)
        else:
            delta = np.random.uniform(0.1 * (n_states - 1), .99 - low_lim_E)
            lower = np.random.uniform(low_lim_E, .99 - delta)

        if np.random.uniform(0, 1) < 0.5:
            E_in = 1 - E_in

        E_out = ((E_in - min(E_in)) / (max(E_in) - min(E_in))) * delta + lower
        return E_out

    def calc_lowlim_E(E_bg=None, E_br=None, E_gr=None):
        # Alexa488 - Cy3B, Atto488 - Cy3B, Alexa488 - Atto565
        r0_bg = np.max([68.61, 67.32, 64.01])
        # Alexa488/Atto488 - Atto 643, Alexa488 - Atto 647N,
        r0_br = np.max([49.22, 50.33])
        # Alexa 555, Cy3, Cy3B, Atto565 - Atto643, Atto565 - Atto647N
        r0_gr = np.max([45.63, 46.72, 64.41, 69.79, 70.11])
        if E_gr is None:
            r_bg = (1 / E_bg - 1) ** (1 / 6) * r0_bg
            r_br = (1 / E_br - 1) ** (1 / 6) * r0_br
            E_out = (1 - .01) / (1 + ((r_bg + r_br) / r0_gr) ** 6) + .01
        elif E_br is None:
            r_bg = (1 / E_bg - 1) ** (1 / 6) * r0_bg
            r_gr = (1 / E_gr - 1) ** (1 / 6) * r0_gr
            E_out = (1 - .01) / (1 + ((r_bg + r_gr) / r0_br) ** 6) + .01
        else:
            r_br = (1 / E_br - 1) ** (1 / 6) * r0_br
            r_gr = (1 / E_gr - 1) ** (1 / 6) * r0_gr
            E_out = (1 - .01) / (1 + ((r_br + r_gr) / r0_bg) ** 6) + .01
        return np.min(E_out)

    def randomize(BB, BG, BR, GG, GR, RR, classifications, label, bg_overestimated):
        channels = [BB, BG, BR, GG, GR, RR]
        if bg_overestimated:  # overestimated background subtraction
            ch = np.random.choice(("B", "G", "R", "all"))
            if ch == "B":  # donor channel
                lower = np.min(channels[0]) + .1
                channels[0] -= np.random.uniform(lower, lower+.4)
            elif ch == "G":  # acceptor channel
                lower = np.min([channels[1], channels[3]]) + .1
                overest = np.random.uniform(lower, lower+.4)
                channels[1] -= overest
                channels[3] -= overest
            elif ch == "R":
                lower = np.min([channels[2], channels[4], channels[5]]) + .1
                overest = np.random.uniform(lower, lower + .4)
                channels[2] -= overest
                channels[4] -= overest
                channels[5] -= overest
            BB, BG, BR, GG, GR, RR = channels
        else:
            for idx, _ in enumerate(channels):
                if np.random.uniform(0, 1) < .8:
                    rndwalk = np.cumsum(np.random.normal(loc=0, scale=1, size=n_frames))
                    rndwalk -= np.min(rndwalk)
                    rndwalk /= np.max(rndwalk)
                    rndwalk *= np.random.uniform(.3, 1)
                    rndwalk += np.random.uniform(-.2, .2)
                    rndwalk[channels[idx] == 0] = 0
                    channels[idx] += rndwalk
                if np.random.uniform(0, 1) < 1/6:
                    frames = np.linspace(0, 1, n_frames, endpoint=False)
                    for _ in range(2):
                        pulsewave = signal.square(frames * np.pi * np.random.randint(1, 20)) * np.random.uniform(-0.2, .2)
                        pulsewave[channels[idx] == 0] = 0
                        channels[idx] += pulsewave
                if np.random.uniform(0, 1) < 1/6:
                    channels[idx] = channels[idx][::-1]

            BB, BG, BR, GG, GR, RR = [np.where(x < 0, 0, x) for x in channels]

        label.fill(classifications["randomized"])

        S_bg = (BB + BG + BR) / (BB + BG + BR + GG + GR)
        S_br = (BB + BG + BR) / (BB + BG + BR + RR)
        S_gr = (GG + GR) / (GG + GR + RR)
        label[(S_bg < np.min(s_tolerance_bg)) & (S_br < np.min(s_tolerance_br))] = classifications["blue bleached"]
        label[(S_bg > np.max(s_tolerance_bg)) & (S_gr < np.min(s_tolerance_gr))] = classifications["green bleached"]
        label[(S_br > np.max(s_tolerance_br)) & (S_gr > np.max(s_tolerance_gr))] = classifications["red bleached"]
        label[(S_br < np.min(s_tolerance_br)) & (S_gr < np.min(s_tolerance_gr))] = classifications["blue and green bleached"]
        label[(S_bg < np.min(s_tolerance_bg)) & (S_gr > np.max(s_tolerance_gr))] = classifications["blue and red bleached"]
        label[(S_bg > np.max(s_tolerance_bg)) & (S_br > np.max(s_tolerance_br))] = classifications["green and red bleached"]

        label[(BB == 0) & (BG == 0) & (BR == 0)] = classifications["blue bleached"]
        label[GG == 0] = classifications["green bleached"]
        label[RR == 0] = classifications["red bleached"]
        label[(BB == 0) & (BG == 0) & (BR == 0) & (GG == 0) & (GR == 0)] = classifications["blue and green bleached"]
        label[(BB == 0) & (BG == 0) & (BR == 0) & (RR == 0)] = classifications["blue and red bleached"]
        label[(GG == 0) & (GR == 0) & (RR == 0)] = classifications["green and red bleached"]
        label[(BB == 0) & (GG == 0) & (RR == 0) & (BG == 0) & (BR == 0) & (GR == 0)] = classifications["all bleached"]

        return BB, BG, BR, GG, GR, RR, label

    def single_3color_trace():
        if state_mode:
            classifications = {
                "1-state": 0,
                "2-state": 1,
                "3-state": 2,
                "4-state": 3,
                "5-state": 4,
                "6-state": 5,
                "7-state": 6,
                "8-state": 7,
                "9-state": 8,
                "blue bleached": 9,
                "green bleached": 10,
                "red bleached": 11,
                "blue and green bleached": 12,
                "blue and red bleached": 13,
                "green and red bleached": 14,
                "all bleached": 15,
                "aggregate": 16,
                "randomized": 17,
                "noisy": 18,
            }
        else:
            classifications = {
                "blue bleached": 0,
                "green bleached": 1,
                "red bleached": 2,
                "blue and green bleached": 3,
                "blue and red bleached": 4,
                "green and red bleached": 5,
                "all bleached": 6,
                "aggregate": 7,
                "randomized": 8,
                "noisy": 9,
                "1-state": 10,
                "2-state": 11,
                "3-state": 12,
                "4-state": 13,
                "5-state": 14,
                "6-state": 15,
                "7-state": 16,
                "8-state": 17,
                "9-state": 18
            }
        frames = np.arange(1, n_frames + 1, 1)
        E_true_bg = np.zeros(n_frames)
        E_true_br = np.zeros(n_frames)
        E_true_gr = np.zeros(n_frames)
        static_pair = []
        if np.random.uniform(0, 1) < aggregation_prob and not any((state_mode, n_states_mode)):
            is_aggregated = True
            agg_dye = np.random.choice(("blue", "green", "red"))
            if agg_dye == "blue":
                E_true_bg, matrix_bg = generate_fret_states(
                    mode="aggregate",
                    tmat=trans_mat,
                    states_E=states_E_bg,
                )
                E_true_br, matrix_br = generate_fret_states(
                    mode="aggregate",
                    tmat=trans_mat,
                    states_E=states_E_br,
                )
                E_true_gr.fill(np.round(np.random.uniform(calc_lowlim_E(E_true_bg, E_true_br), .99), 5))
            elif agg_dye == "green":
                E_true_bg, matrix_bg = generate_fret_states(
                    mode="aggregate",
                    tmat=trans_mat,
                    states_E=states_E_br,
                )
                E_true_gr, matrix_gr = generate_fret_states(
                    mode="aggregate",
                    tmat=trans_mat,
                    states_E=states_E_gr,
                )
                E_true_br.fill(np.round(np.random.uniform(calc_lowlim_E(E_true_bg, E_true_gr), .99), 5))
            else:
                E_true_br, matrix_br = generate_fret_states(
                    mode="aggregate",
                    tmat=trans_mat,
                    states_E=states_E_br,
                )
                E_true_gr, matrix_gr = generate_fret_states(
                    mode="aggregate",
                    tmat=trans_mat,
                    states_E=states_E_gr,
                )
                E_true_bg.fill(np.round(np.random.uniform(calc_lowlim_E(E_true_br, E_true_gr), .99), 5))

            if max_aggregate_size >= 2:
                aggregate_size = np.random.randint(2, max_aggregate_size)
            else:
                raise ValueError("Can't have an aggregate of size less than 2")
            np.random.seed()
            n_pairs = np.random.poisson(aggregate_size)
            if n_pairs == 0:
                n_pairs = 2
        else:
            is_aggregated = False
            n_pairs = 1
            # all static
            if np.random.uniform(0, 1) < static_prob and not any((state_mode, n_states_mode)):
                E_true_bg, matrix_bg = generate_fret_states(
                    mode="static",
                    tmat=trans_mat,
                    states_E=states_E_bg,
                )
                E_true_br, matrix_br = generate_fret_states(
                    mode="static",
                    tmat=trans_mat,
                    states_E=states_E_br,
                )
                if states_E_gr == "random":
                    E_true_gr.fill(np.random.uniform(calc_lowlim_E(E_true_bg, E_true_br), .99))
                else:
                    E_true_gr, matrix_gr = generate_fret_states(
                        mode="static",
                        tmat=trans_mat,
                        states_E=states_E_gr,
                    )
            # dynamic
            else:
                if static_dyes == "random":
                    if state_mode:
                        static_pair = ["BG", "BR", "GR"][np.nonzero(np.random.multinomial(
                            1, [1/3, 1/3, 1/3]))[0].astype("int")[0]]
                    else:
                        static_pair = ["BG", "BR", "GR", None][np.nonzero(np.random.multinomial(
                            1, [1/4, 1/4, 1/4, 1/4]))[0].astype("int")[0]]
                elif static_dyes is None:
                    static_pair = None
                else:
                    static_pair = static_dyes
                if static_pair == "BG":
                    E_true_bg, matrix_bg = generate_fret_states(
                        mode="static",
                        tmat=trans_mat,
                        states_E=states_E_bg,
                    )
                    while len(np.unique(E_true_br)) < 2:
                        E_true_br, matrix_br = generate_fret_states(
                            mode="dynamic",
                            tmat=trans_mat,
                            max_tprob=max_trans_prob,
                            states_E=states_E_br,
                        )
                    E_true_gr = rnd_scale_E(E_true_br, low_lim_E=calc_lowlim_E(E_true_bg, E_true_br))
                elif static_pair == "BR":
                    E_true_br, matrix_br = generate_fret_states(
                        mode="static",
                        tmat=trans_mat,
                        states_E=states_E_br,
                    )
                    while len(np.unique(E_true_bg)) < 2:
                        E_true_bg, matrix_bg = generate_fret_states(
                            mode="dynamic",
                            tmat=trans_mat,
                            max_tprob=max_trans_prob,
                            states_E=states_E_bg,
                        )
                    if states_E_gr is not None:
                        E_true_gr = 1 - E_true_bg
                    else:
                        E_true_gr = rnd_scale_E(E_true_bg, low_lim_E=calc_lowlim_E(E_true_bg, E_true_br))
                elif static_pair == "GR":
                    E_true_gr, matrix_gr = generate_fret_states(
                        mode="static",
                        tmat=trans_mat,
                        states_E=states_E_bg,
                    )
                    while len(np.unique(E_true_bg)) < 2:
                        E_true_bg, matrix_bg = generate_fret_states(
                            mode="dynamic",
                            tmat=trans_mat,
                            max_tprob=max_trans_prob,
                            states_E=states_E_bg,
                        )
                    E_true_br = rnd_scale_E(E_true_bg, low_lim_E=calc_lowlim_E(E_bg=E_true_bg, E_gr=E_true_gr))
                else:  # all dynamic
                    while len(np.unique(E_true_bg)) < 2:
                        E_true_bg, matrix_bg = generate_fret_states(
                            mode="alldynamic",
                            tmat=trans_mat_bg,
                            max_tprob=max_trans_prob,
                            states_E=states_E_bg,
                        )
                    while len(np.unique(E_true_br)) < 2:
                        E_true_br, matrix_br = generate_fret_states(
                            mode="alldynamic",
                            tmat=trans_mat_br,
                            max_tprob=max_trans_prob,
                            states_E=states_E_br,
                        )
                    if states_E_gr is None:
                        E_true_gr = (rnd_scale_E(E_true_bg, low_lim_E=calc_lowlim_E(E_true_bg, E_true_br))
                                     + rnd_scale_E(E_true_br, low_lim_E=calc_lowlim_E(E_true_bg, E_true_br))) / 2
                    else:
                        E_true_gr = (E_true_bg + E_true_br) / 2

        BB_total, BG_total, BR_total, GG_total, GR_total, RR_total = [], [], [], [], [], []

        for j in range(n_pairs):
            if b_lifetime is not None and not any((state_mode, n_states_mode)):
                bleach_B = int(np.ceil(np.random.exponential(b_lifetime)))
            else:
                bleach_B = np.nan

            if g_lifetime is not None and not any((state_mode, n_states_mode)):
                bleach_G = int(np.ceil(np.random.exponential(g_lifetime)))
            else:
                bleach_G = np.nan

            if r_lifetime is not None and not any((state_mode, n_states_mode)):
                bleach_R = int(np.ceil(np.random.exponential(r_lifetime)))
            else:
                bleach_R = np.nan

            if balance_bleach and not any((state_mode, n_states_mode)):
                if all(i is None for i in (states_E_bg, states_E_br, states_E_gr)):
                    dyelfnan = [1, 2, 3, "all", None][np.nonzero(np.random.multinomial(
                        1, [0, 0, 0, .15, .85]))[0].astype("int")[0]]
                    if dyelfnan == 1:
                        bleach_B = np.nan
                    elif dyelfnan == 2:
                        bleach_G = np.nan
                    elif dyelfnan == 3:
                        bleach_R = np.nan
                    elif dyelfnan == "all":
                        bleach_B = np.nan
                        bleach_G = np.nan
                        bleach_R = np.nan

            first_bleach = min((bleach_B, bleach_G, bleach_R))

            # Calculate apparent FRET
            if train_BR or train_BG:
                E_app_bg = _E_app_bg(E_true_bg, 0)
                E_app_br = _E_app_br(0, E_true_br)
            else:
                E_app_br = _E_app_br(E_true_bg, E_true_br)
                E_app_bg = _E_app_bg(E_true_bg, E_true_br)
            # In case apparent BG or BR FRET should be limited by the defined minimum state difference
            # try:
            #     if np.min(np.diff(np.unique(E_app_bg))) < min_state_diff or \
            #             np.min(np.diff(np.unique(E_true_gr))) < min_state_diff:
            #         return single_3color_trace()
            # except ValueError:
            #     return
            BB = _BB(E_app_bg, E_app_br)
            if train_BR or train_BG:
                BG = _BG(E_app_bg, 0)
                BR = _BR(E_app_br, 0, 0)
            else:
                BG = _BG(E_app_bg, E_true_gr)
                BR = _BR(E_app_br, E_app_bg, E_true_gr)
            GG = _GG(E_true_gr)
            GR = _GR(GG, E_true_gr)
            RR = _RR(E_true_gr)
            if quenching:
                if np.random.uniform(0, 1) < .2:
                    # simulate quenching events
                    pulsewave = signal.square(frames * np.pi * np.random.randint(1, 20)) * np.random.uniform(-0.2, .2)
                    RR += pulsewave
            # Make sure there is at least one bleaching step for n_pairs == 2 otherwise aggregate is likely to resemble
            # static molecule
            if j == 1 and n_pairs == 2:
                while (bleach_B > n_frames) and (bleach_G > n_frames) and (bleach_R > n_frames):
                    bleach_B = int(np.ceil(np.random.exponential(b_lifetime)))
                    bleach_G = int(np.ceil(np.random.exponential(g_lifetime)))
                    bleach_R = int(np.ceil(np.random.exponential(r_lifetime)))
                first_bleach = min((bleach_B, bleach_G, bleach_R))
            if np.random.uniform(0, 1) < 0.5 and j == 1 and n_pairs == 2:
                dyerm = np.random.choice(("B", "G", "R"))
                if dyerm == "B":
                    BB[:] = 0
                    while (bleach_G > n_frames) and (bleach_R > n_frames):
                        if np.random.uniform(0, 1) < 0.5:
                            bleach_G = int(np.ceil(np.random.exponential(g_lifetime)))
                            first_bleach = bleach_G
                        else:
                            bleach_R = int(np.ceil(np.random.exponential(r_lifetime)))
                            first_bleach = bleach_R
                elif dyerm == "G":
                    BG[:] = 0
                    GG[:] = 0
                    while (bleach_B > n_frames) and (bleach_R > n_frames):
                        if np.random.uniform(0, 1) < 0.5:
                            bleach_B = int(np.ceil(np.random.exponential(b_lifetime)))
                            first_bleach = bleach_B
                        else:
                            bleach_R = int(np.ceil(np.random.exponential(r_lifetime)))
                            first_bleach = bleach_R
                elif dyerm == "R":
                    BR[:] = 0
                    GR[:] = 0
                    RR[:] = 0
                    while (bleach_B > n_frames) and (bleach_G > n_frames):
                        if np.random.uniform(0, 1) < 0.5:
                            bleach_B = int(np.ceil(np.random.exponential(b_lifetime)))
                            first_bleach = bleach_B
                        else:
                            bleach_G = int(np.ceil(np.random.exponential(g_lifetime)))
                            first_bleach = bleach_G

            if first_bleach is not np.nan:
                if first_bleach == bleach_G:  # green dye bleaches first
                    second_bleach = min((bleach_B, bleach_R))
                    if second_bleach is not np.nan:
                        BB[bleach_G:second_bleach] = 1 - E_true_br[bleach_G:second_bleach]
                        BR[bleach_G:second_bleach] = \
                            _BR_2color(BB[bleach_G:second_bleach], E_true_br[bleach_G:second_bleach])
                        if np.nan not in (bleach_B, bleach_R) and bleach_B > bleach_R:
                            if quenching:
                                if np.random.uniform(0, 1) < .2:
                                    # emulate quenching
                                    BB[bleach_R:bleach_B] = 1 - E_true_br[bleach_R:bleach_B]
                                    BB[bleach_R:bleach_B] *= (1+np.random.uniform(1.2, 1.5))
                                else:
                                    BB[bleach_R:bleach_B] = 1
                            else:
                                BB[bleach_R:bleach_B] = 1
                    else:
                        BB[bleach_G:] = 1 - E_true_br[bleach_G:]
                    if is_aggregated and n_pairs <= 2:
                        if bleach_B is not np.nan:
                            spike_len = np.min((np.random.randint(2, 10), bleach_B))
                            BB[bleach_G:bleach_G + spike_len] = 2
                elif first_bleach == bleach_R:  # red dye bleaches first
                    second_bleach = min((bleach_B, bleach_G))
                    if second_bleach is not np.nan:
                        BB[bleach_R:second_bleach] = 1 - E_true_bg[bleach_R:second_bleach]
                        BG[bleach_R:second_bleach] = \
                            _BG_2color(BB[bleach_R:second_bleach], E_true_bg[bleach_R:second_bleach])
                        if np.nan not in (bleach_B, bleach_G) and bleach_B > bleach_G:
                            if quenching:
                                if np.random.uniform(0, 1) < .5:
                                    # emulate quenching
                                    BB[bleach_G:bleach_B] = 1 - E_true_bg[bleach_G:bleach_B]
                                    BB[bleach_G:bleach_B] *= (1 + np.random.uniform(1.2, 1.5))
                                else:
                                    BB[bleach_G:bleach_B] = 1
                            else:
                                BB[bleach_G:bleach_B] = 1
                    else:
                        BB[bleach_R:] = 1 - E_true_bg[bleach_R:]
                    if is_aggregated and n_pairs <= 2:
                        # Spikes for small aggregates
                        if bleach_B is not np.nan:
                            spike_len1 = np.min((np.random.randint(2, 10), bleach_B))
                            BB[bleach_R:bleach_R + spike_len1] = 2
                        if bleach_G is not np.nan:
                            spike_len2 = np.min((np.random.randint(2, 10), bleach_G))
                            GG[bleach_R:bleach_R + spike_len2] = 2

            if bleach_B is not np.nan:
                BB[bleach_B:] = 0
                BG[bleach_B:] = 0
                BR[bleach_B:] = 0
            if bleach_R is not np.nan:
                if quenching:
                    if np.random.uniform(0, 1) < .5:
                        GG[bleach_R:] = 1 - E_true_gr[bleach_R:]
                        GG[bleach_R:] *= (1 + np.random.uniform(1.2, 2))
                    else:
                        GG[bleach_R:] = 1
                else:
                    GG[bleach_R:] = 1
                BR[bleach_R:] = 0
                GR[bleach_R:] = 0
                RR[bleach_R:] = 0
            if bleach_G is not np.nan:
                BG[bleach_G:] = 0
                GG[bleach_G:] = 0
                GR[bleach_G:] = 0

            BB_total.append(BB)
            BG_total.append(BG)
            BR_total.append(BR)
            GG_total.append(GG)
            GR_total.append(GR)
            RR_total.append(RR)

        BB, BG, BR, GG, GR, RR = \
            [np.sum(x, axis=0) for x in (BB_total, BG_total, BR_total, GG_total, GR_total, RR_total)]

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
                BB[falloff_frame:] = 0
                BG[falloff_frame:] = 0
                BR[falloff_frame:] = 0
                GG[falloff_frame:] = 0
                GR[falloff_frame:] = 0
                RR[falloff_frame:] = 0

            label.fill(classifications["aggregate"])

        if not any((is_aggregated, state_mode, n_states_mode)) and np.random.uniform(0, 1) < blink_prob:
            blink_time = np.random.randint(2, 50)
            blink_start = np.random.randint(1, n_frames)

            blinkedDye = np.random.choice(("B", "G", "R"))
            b_active = np.nonzero(BB[blink_start:(blink_start + blink_time)])[0] + blink_start
            g_active = np.nonzero(GG[blink_start:(blink_start + blink_time)])[0] + blink_start
            r_active = np.nonzero(RR[blink_start:(blink_start + blink_time)])[0] + blink_start
            if blinkedDye == "B":
                BB[b_active] = 0
                BG[b_active] = 0
                BR[b_active] = 0
            elif blinkedDye == "G":
                br_active = np.intersect1d(b_active, r_active)
                BB[b_active] = 1
                BB[br_active] = 1 - E_true_br[br_active]
                BG[g_active] = 0
                BR[br_active] = _BR_2color(BB[br_active], E_true_br[br_active])
                GG[g_active] = 0
                GR[g_active] = 0
            else:
                bg_active = np.intersect1d(b_active, g_active)
                BB[b_active] = 1
                BB[bg_active] = 1 - E_true_bg[bg_active]
                BG[bg_active] = _BG_2color(BB[bg_active], E_true_bg[bg_active])
                BR[r_active] = 0
                GG[g_active] = 1
                GR[r_active] = 0
                RR[r_active] = 0

        if not is_aggregated:
            if bleach_B is not np.nan:
                label[bleach_B:] = classifications["blue bleached"]
                E_true_bg[bleach_B:] = non_spFRET_E
                E_true_br[bleach_B:] = non_spFRET_E

            if bleach_G is not np.nan:
                label[bleach_G:] = classifications["green bleached"]
                E_true_bg[bleach_G:] = non_spFRET_E
                E_true_gr[bleach_G:] = non_spFRET_E

            if bleach_R is not np.nan:
                label[bleach_R:] = classifications["red bleached"]
                E_true_br[bleach_R:] = non_spFRET_E
                E_true_gr[bleach_R:] = non_spFRET_E

        label[(BB == 0) & (GG != 0) & (RR != 0)] = classifications["blue bleached"]
        label[(GG == 0) & (BB != 0) & (RR != 0)] = classifications["green bleached"]
        label[(RR == 0) & (BB != 0) & (GG != 0)] = classifications["red bleached"]
        label[(BB == 0) & (GG == 0) & (RR != 0)] = classifications["blue and green bleached"]
        label[(BB == 0) & (RR == 0) & (GG != 0)] = classifications["blue and red bleached"]
        label[(GG == 0) & (RR == 0) & (BB != 0)] = classifications["green and red bleached"]
        label[(BB == 0) & (GG == 0) & (RR == 0)] = classifications["all bleached"]

        if is_aggregated:
            bleach_bool = np.isin(label, [0, 1, 2, 3, 4, 5, 6])
            if np.any(bleach_bool):
                first_bleach_all = np.argmax(bleach_bool)
            else:
                first_bleach_all = None

        # randomly disturb trace / include artefacts
        is_randomized = False
        if np.random.uniform(0, 1) < randomize_prob and n_pairs <= 2 and not any((state_mode, n_states_mode)):
            bg_overestimated = np.random.uniform(0, 1) < bg_overestimate_prob
            BB, BG, BR, GG, GR, RR, label = randomize(
                BB=BB,
                BG=BG,
                BR=BR,
                GG=GG,
                GR=GR,
                RR=RR,
                classifications=classifications,
                label=label,
                bg_overestimated=bg_overestimated,
            )
            is_randomized = True

        b_active = np.nonzero(BB)[0]
        g_active = np.nonzero(GG)[0]
        r_active = np.nonzero(RR)[0]
        gr_active = np.intersect1d(g_active, r_active)
        all_active = np.intersect1d(b_active, gr_active)

        E_active_true_bg = E_app_bg[all_active]
        E_active_true_br = E_app_br[all_active]
        E_active_true_gr = E_true_gr[all_active]

        # Existing true states, later used as reference to observations
        if static_pair is None:
            E_active_true_all = np.round(E_active_true_gr
                                         + E_true_bg[all_active]
                                         + E_true_br[all_active], 5)
            uni_states_all, uni_states_all_idx = np.unique(E_active_true_all[E_active_true_all > 0],
                                                           return_index=True)
        unique_states_bg, unique_states_bg_idx = np.unique(
            E_active_true_bg[E_active_true_bg != non_spFRET_E],
            return_index=True
        )
        unique_states_br, unique_states_br_idx = np.unique(
            E_active_true_br[E_active_true_br != non_spFRET_E],
            return_index=True
        )
        unique_states_gr, unique_states_gr_idx = np.unique(
            E_active_true_gr[E_active_true_gr != non_spFRET_E],
            return_index=True
        )

        blue_offset = np.random.uniform(bb_offset[0], bb_offset[1])
        green_offset = np.random.uniform(gg_offset[0], gg_offset[1])
        red_offset = np.random.uniform(rr_offset[0], rr_offset[1])
        # Include different excitation powers
        BB *= blue_offset
        BG *= blue_offset
        BR *= blue_offset
        GG *= green_offset
        GR *= green_offset
        RR *= red_offset

        # simulate intensity fluctuations not affecting FRET efficiency
        fluct_probs = np.random.uniform(0, 1, 3)
        if fluct_probs[0] < .5:  # blue
            fluct = np.sin(np.linspace(0, np.random.randint(10, 300), n_frames)) * np.random.uniform(.01, .1) + 1
            fluct[BB == 0] = 0
            fluct[BG == 0] = 0
            fluct[BR == 0] = 0
            BB *= fluct
            BG *= fluct
            BR *= fluct
        if fluct_probs[1] < .5:  # green
            fluct = np.sin(np.linspace(0, np.random.randint(10, 300), n_frames)) * np.random.uniform(.01, .1) + 1
            fluct[GG == 0] = 0
            fluct[GR == 0] = 0
            GG *= fluct
            GR *= fluct
        if fluct_probs[2] < .5:  # red
            fluct = np.sin(np.linspace(0, np.random.randint(10, 300), n_frames)) * np.random.uniform(.01, .1) + 1
            fluct[RR == 0] = 0
            RR *= fluct

        # correction factors
        if np.any(E_true_gr[gr_active] < 0):  # in case channels were flipped in randomization
            E_true_gr = _E_gr(GG, GR)
        # gamma
        g_bg = np.random.uniform(gamma_bg[0], gamma_bg[1])
        g_gr = np.random.uniform(gamma_gr[0], gamma_gr[1])
        g_br = g_bg * g_gr

        BG *= g_bg
        BR *= g_br
        GR *= g_gr

        # draw direct excitation
        de_bg = np.random.uniform(dir_exc_bg[0], dir_exc_bg[1])
        de_br = np.random.uniform(dir_exc_br[0], dir_exc_br[1])
        de_gr = np.random.uniform(dir_exc_gr[0], dir_exc_gr[1])

        # draw crosstalk
        beta_bg = np.random.uniform(crosstalk_bg[0], crosstalk_bg[1])
        beta_br = np.random.uniform(crosstalk_br[0], crosstalk_br[1])
        beta_gr = np.random.uniform(crosstalk_gr[0], crosstalk_gr[1])
        beta_bx = beta_bg + beta_br
        ct_bg_ratio = np.divide(beta_bg, beta_bx, out=np.zeros_like(beta_bg), where=beta_bx != 0)
        ct_br_ratio = np.divide(beta_br, beta_bx, out=np.zeros_like(beta_br), where=beta_bx != 0)
        ct_bx = _calc_ct(BB, beta_bx)
        ct_bg = ct_bg_ratio * ct_bx
        ct_br = ct_br_ratio * ct_bx
        ct_gr = _calc_ct(GG, beta_gr)
        # ct from BG to R, the way alpha BG is defined, crosstalk needs to be calculated before adding dir ex of G.
        # Crosstalk of dir ex G to R needs to be calculated and added separately to BR
        ct_gr_BG = _calc_ct(BG, beta_gr)

        BB -= (ct_bg + ct_br)
        GG -= ct_gr  # subtract photons before adding direct excitation
        BG -= ct_gr_BG
        # add direct excitation
        BG[g_active] += _calc_de(GG[g_active], de_bg)
        BR[r_active] += _calc_de(RR[r_active], de_br)
        GR[r_active] += _calc_de(RR[r_active], de_gr)

        # add crosstalk
        ct_bgr_alphaBG = _calc_de(GG, de_bg) * beta_gr  # ct from dir ex of G to R
        BG += ct_bg
        BR += ct_br
        BR += ct_gr_BG  # ct_gr contribution from BG without dir ex
        BR += ct_bgr_alphaBG  # ct_gr contribution from dir ex of G from BG

        BR[gr_active] += de_bg * GG[gr_active] / (1 - E_true_gr[gr_active]) * E_true_gr[gr_active]
        GR += ct_gr

        # Add poisson noise
        b_bg_count = np.random.randint(100, 1000)
        g_bg_count = np.random.randint(100, 1000) * green_offset
        r_bg_count = np.random.randint(100, 1000) * red_offset
        # noise_scale_intensity = np.random.uniform(noise[0], noise[1]/3)
        noise_scale_intensity = np.random.uniform(0, .1)
        noise_scale_gamma = np.random.uniform(noise[0], noise[1]/3)
        bg_noise_scale = np.random.uniform(noise[0], noise[1])
        gamma_noise = False
        if np.random.uniform(0, 1) < gamma_noise_prob:
            gamma_noise = True
        for i, s in enumerate([BB, BG, GG, BR, GR, RR]):
            if i == 0:
                bg_poisson = np.random.poisson(b_bg_count, n_frames).astype('float') - b_bg_count
            elif i == 1 | i == 2:
                bg_poisson = np.random.poisson(g_bg_count, n_frames).astype('float') - g_bg_count
            else:
                bg_poisson = np.random.poisson(r_bg_count, n_frames).astype('float') - r_bg_count
            bg_poisson /= bg_poisson.max()
            bg_poisson *= bg_noise_scale
            gaussian_noise = np.random.normal(0, noise_scale_intensity, n_frames) * s
            if gamma_noise is True:
                gamma_noise = np.random.gamma(1, noise_scale_gamma, len(s))
                s += gamma_noise
                s -= np.mean(gamma_noise)
            s += gaussian_noise + bg_poisson

        # Observed E and S
        E_obs_bg = _E_obs_bg(BB, BG, BR)
        E_obs_br = _E_obs_br(BB, BG, BR)
        E_obs_gr = _E_gr(GG, GR)

        # Observed states
        E_active_bg = E_obs_bg[all_active]
        E_active_br = E_obs_br[all_active]
        E_active_gr = E_obs_gr[all_active]

        # noise label
        is_noisy = False
        noise_all = []
        noiselvl_total = 0
        if not any((is_aggregated, is_randomized)):
            for state in unique_states_bg:
                noise_all.append(np.std(E_active_bg[E_active_true_bg == state]))
            for state in unique_states_br:
                noise_all.append(np.std(E_active_br[E_active_true_br == state]))
            for state in unique_states_gr:
                noise_all.append(np.std(E_active_gr[E_active_true_gr == state]))
            noiselvl_total = np.mean(noise_all)

            if noiselvl_total > noise_tolerance:
                label[
                    (label != classifications["blue bleached"]) &
                    (label != classifications["green bleached"]) &
                    (label != classifications["red bleached"]) &
                    (label != classifications["blue and green bleached"]) &
                    (label != classifications["blue and red bleached"]) &
                    (label != classifications["green and red bleached"]) &
                    (label != classifications["all bleached"])
                    ] = classifications["noisy"]
                is_noisy = True
                if any((state_mode, n_states_mode)):
                    return

        # State labeling for trace classification and number of states classification
        if not any((is_noisy, is_aggregated, is_randomized, state_mode)):
            if (len(unique_states_bg) == 1) \
                    & (len(unique_states_br) == 1) \
                    & (len(unique_states_gr) == 1):
                label[
                    (label != classifications["blue bleached"]) &
                    (label != classifications["green bleached"]) &
                    (label != classifications["red bleached"]) &
                    (label != classifications["blue and green bleached"]) &
                    (label != classifications["blue and red bleached"]) &
                    (label != classifications["green and red bleached"]) &
                    (label != classifications["all bleached"])
                    ] = classifications["1-state"]
            else:
                n_states = max([len(unique_states_bg), len(unique_states_br), len(unique_states_gr)])
                if (n_states == 1) & n_states_mode:
                    return
                label[
                    (label != classifications["blue bleached"]) &
                    (label != classifications["green bleached"]) &
                    (label != classifications["red bleached"]) &
                    (label != classifications["blue and green bleached"]) &
                    (label != classifications["blue and red bleached"]) &
                    (label != classifications["green and red bleached"]) &
                    (label != classifications["all bleached"])
                    ] = classifications[f"{n_states}-state"]

        if state_mode:
            if static_dyes is None:
                if train_BR:
                    E_true = E_app_br
                    obs_states = len(unique_states_br_idx)
                    sorted_states = E_active_true_br[np.sort(unique_states_br_idx)]
                elif train_BG:
                    E_true = E_app_bg
                    obs_states = len(unique_states_bg_idx)
                    sorted_states = E_active_true_bg[np.sort(unique_states_bg_idx)]
                else:
                    E_true = E_true_bg + E_true_br + E_true_gr
                    obs_states = max([len(uni_states_all)])
                    sorted_states = E_active_true_all[np.sort(uni_states_all_idx)]
                for state in range(obs_states):
                    label[E_true == sorted_states[state]] = classifications[f"{state + 1}-state"]
            else:
                obs_states_ch = np.argmax([len(unique_states_bg), len(unique_states_br), len(unique_states_gr)])
                unique_states_idx = [unique_states_bg_idx, unique_states_br_idx, unique_states_gr_idx][obs_states_ch]
                E_active_true = [E_active_true_bg, E_active_true_br, E_active_true_gr][obs_states_ch]
                E_true = [E_app_bg, E_app_br, E_true_gr][obs_states_ch]
                obs_states = max([len(unique_states_bg), len(unique_states_br), len(unique_states_gr)])
                if obs_states < 2:
                    return
                sorted_states = E_active_true[np.sort(unique_states_idx)]
                for state in range(obs_states):
                    label[E_true == sorted_states[state]] = classifications[f"{state + 1}-state"]

        try:
            if state_mode:
                ediff = np.min(np.diff(np.unique(E_true_gr)))
            else:
                if label[0] in [11, 12, 13, 14]:
                    ediff = np.min(np.diff(np.unique(E_true_gr)))
                else:
                    ediff = -1
        except ValueError:
            ediff = -1

        if reduce_memory:
            if state_mode:
                single_trace = pd.DataFrame(
                    {
                        "BB": BB,
                        "BG": BG,
                        "BR": BR,
                        "GG": GG,
                        "GR": GR,
                        "label": label,
                    }
                )
            else:
                single_trace = pd.DataFrame(
                    {
                        "BB": BB,
                        "BG": BG,
                        "BR": BR,
                        "GG": GG,
                        "GR": GR,
                        "RR": RR,
                        "label": label,
                    }
                )
        else:
            single_trace = pd.DataFrame(
                {
                    "BB": BB,
                    "BG": BG,
                    "BR": BR,
                    "GG": GG,
                    "GR": GR,
                    "RR": RR,
                    "E_bg": E_obs_bg,
                    "E_br": E_obs_br,
                    "E_gr": E_obs_gr,
                    "E_true_bg": E_app_bg,
                    "E_true_br": E_app_br,
                    "E_true_gr": E_true_gr,
                    "label": label,
                    "_noise_level": np.array(noiselvl_total).repeat(n_frames),
                    "_min_E_diff": np.array(ediff).repeat(n_frames),
                    "trans_mean": np.array((matrix_bg[0, 1] + matrix_bg[1, 0])/2).repeat(n_frames),
                    # "E": E_obs_gr,
                    # "E_true": E_true_gr
                }
            )
        single_trace.replace([np.inf, -np.inf, np.nan], -1, inplace=True)
        single_trace.fillna(method="pad", inplace=True)

        return single_trace

    all_traces = []
    if parallel_asynchronous:
        if platform.processor() == 'arm':  # m1 macs
            ctx = mp.get_context("fork")
            pool = ctx.Pool(mp.cpu_count())
        else:  # intel cpus
            pool = mp.Pool(mp.cpu_count())
        jobs = [pool.apply_async(single_3color_trace) for _ in range(n_traces)]
        pool.close()
        for job in tqdm(jobs):
            all_traces.append(job.get())
        pool.join()
    else:
        for _ in tqdm(range(n_traces)):
            trace = single_3color_trace()
            all_traces.append(trace)

    all_traces = (
        pd.concat(all_traces, ignore_index=True, copy=False, sort=False)
        if len(all_traces) > 1
        else all_traces[0]
    )
    if n_states_mode:
        all_traces["label"][all_traces["label"] == 11] = 0
        all_traces["label"][all_traces["label"] == 12] = 1
        all_traces["label"][all_traces["label"] == 13] = 2
        all_traces["label"][all_traces["label"] == 14] = 3
    else:
        all_traces["label"][all_traces["label"] >= 11] = 11
    return all_traces
