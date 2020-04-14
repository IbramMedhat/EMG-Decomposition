# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def decompose(raw_signal, moving_avg_win_size=20, diffTh=12.65):
    raw_signal = np.array(raw_signal)
    
    # Locate MUAPs
    timestamps = get_muaps_timestamps(raw_signal, moving_avg_win_size,
                                      verbose=True)

    # TODO Determine which MU produced the MUAP
    # templates = compare_muaps_templates(raw_signal, timestamps)
    
    
    # TODO Use detected MUAPs and their time to update template and 
    # firing statistics, the MUAP is used as the initial estimate of the MU
    # template
    
    # return timestamps, templates

def get_muaps_timestamps(sig, 
                         moving_avg_win_size=20,
                         verbose=True):
    def find_noise(sig):
        # Noise is any part of the signal that does contain MUAPs
        return sig[:120]
    
    def apply_moving_avg(sig, win_size=20):
        smoothed_signal = np.zeros(sig.shape)
        for i in range(win_size - 1, len(sig)):
            avg_value = 0
            for v_i in range(win_size):
                avg_value += sig[i-v_i]
            avg_value /= win_size
            smoothed_signal[i] = avg_value
        return smoothed_signal

    def define_above_thresholds(sig, threshold,
                                discard_margin=20):
        values_indices = set()
        step_of_first_observation = None
    
        v_i = 0
        while(v_i < sig.shape[0]):
            v = sig[v_i]
            if v >= threshold:
                if step_of_first_observation is None:
                    step_of_first_observation = v_i
                    v_i += discard_margin
                    continue
            else:
                step_of_first_observation = None
            if step_of_first_observation is not None:
                values_indices.add(step_of_first_observation)
            v_i += 1

        values_indices = list(values_indices)
        values_indices.sort()
        return np.array(values_indices)

    
    rectified_signal = np.abs(raw_signal)

    # MUAP is detected if avg of rectified EMG in a window of
    # length T samples exceeds threshold (Moving Average)
    moving_avg_signal = apply_moving_avg(rectified_signal,
                                         win_size=moving_avg_win_size)
    if verbose:
        plot(moving_avg_signal, title="moving_avg_signal")
    
    noise = find_noise(rectified_signal)
    threshold = 3*np.std(noise)
    timestamps = define_above_thresholds(moving_avg_signal,
                                         threshold,
                                         discard_margin=moving_avg_win_size)
    if verbose:
        plot(raw_signal, title="Marked MUAPs", markers=timestamps,
             begin=2200, end=3200)
        
    return timestamps

def compare_muaps_templates():
    # TODO STEP 2
    pass


def plot(sig, title = "Plot of CT signal", sampling_rate=1,
         xlabel="t", ylabel="x(t)", markers=None, save=False,
         begin=0, end=1000):
    if end == 0:
        end = sig.shape[0]
    draw_sig=sig[begin:end]
    begin_time = begin/sampling_rate
    end_time = (end-1)/sampling_rate
    t = np.linspace(begin_time, end_time , draw_sig.shape[0])
    if markers is not None:
        markers = markers[(markers >= begin) & (markers <= end)]
        markers_amp = np.ones(markers.shape)*np.max(draw_sig)
        plt.plot(markers, markers_amp, marker="*",
                 linestyle=' ', color='r', label='R-Waves')
    plt.plot(t, draw_sig)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([begin_time, end_time])  
    if save:
        plt.savefig(title + ".jpg", progressive=True)
    plt.show()

if __name__ == '__main__':
    raw_signal = pd.read_csv("Data.txt", header=None)[0]
    decompose(raw_signal)
 