# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def decompose(raw_signal, moving_avg_win_size=20, diffTh=1265000, verbose=False):

    raw_signal = np.array(raw_signal)
    
    # Locate MUAPs
    timestamps = get_muaps_timestamps(raw_signal, moving_avg_win_size,
                                      verbose=verbose)

    # Determine which MU produced the MUAP
    # Use detected MUAPs and their time to update template and 
    # firing statistics, the MUAP is used as the initial estimate of the MU
    # template
    
    templates = define_muaps_templates(raw_signal, timestamps,
                                        diffTh,
                                        moving_avg_win_size,
                                        verbose=verbose)
    
    return timestamps, templates

def get_muaps_timestamps(sig, 
                         moving_avg_win_size=20,
                         verbose=False):
    def find_noise(sig):
        # Noise is any part of the signal that does contain MUAPs
        return sig[:110]
    
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
        adding_margin = discard_margin/2
        v_i = 0
        while v_i < sig.shape[0] :
            v = sig[v_i]
            if v >= threshold:
                if step_of_first_observation is None:
                    step_of_first_observation = v_i 
                    # v_i += discard_margin
                    # print("\nverify a spike")
                    consecutive_aboves = []
                    v_i_tmp = v_i + 1
                    while v_i < sig.shape[0]:
                        if sig[v_i] < threshold:
                            break
                        consecutive_aboves.append(sig[v_i])
                        v_i += 1
                    consecutive_aboves = np.array(consecutive_aboves)
                    # print(consecutive_aboves)
                    # Using first max. value occurance shifted back with
                    # window half size
                    step_of_first_observation =\
                        np.argmax(consecutive_aboves) +\
                        step_of_first_observation -\
                        adding_margin
                        
                    if consecutive_aboves.shape[0] < discard_margin:
                        v_i = v_i_tmp      
                        step_of_first_observation = None

                        
                    # for v_2i in range(step_of_first_observation, v_i):
                    #     # print("@v_2i =  %s, v = %s" % (v_2i, sig[v_2i]) )
                    #     if (v_2i < sig.shape[0]) and (sig[v_2i] < threshold):
                    #         step_of_first_observation = None
                    #         v_i = v_2i + 1
                    #         break
                    # continue
            else:
                step_of_first_observation = None
            if step_of_first_observation is not None:
                values_indices.add(int(step_of_first_observation))
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
        plot(moving_avg_signal, title="moving_avg_signal",
             begin=0, end=-1)
    
    noise = find_noise(rectified_signal)
    threshold = 3*np.std(noise)
    
    print("threshold value = %s" % threshold)
    # threshold = 11.7
    timestamps = define_above_thresholds(moving_avg_signal,
                                         threshold,
                                         discard_margin=moving_avg_win_size)
    if verbose:
        plot(raw_signal, title="Marked MUAPs", markers=timestamps,
             begin=0, end=-1)
        
    return timestamps

def define_muaps_templates(raw_signal, timestamps,
                            diffTh, moving_avg_win_size,
                            verbose=False):    
    # m: muap; k: template
    
    def sqr_diff(m, k):
        return np.sum((m - k)**2)

    def update_template(m, k):
        return np.add(m, k) * 0.5
    
    templates = []
    ap_margin = int(moving_avg_win_size/2)
    
    k_i = timestamps[0]
    k = np.array(raw_signal[k_i - ap_margin:k_i + ap_margin])
    templates.append(k)
    for i in range(1, timestamps.shape[0]):
        m_i = timestamps[i]
        m = np.array(raw_signal[m_i - ap_margin:m_i + ap_margin])
        merged = False
        for j, k in enumerate(templates):  
            diff = sqr_diff(m, k)
            if (diff < diffTh):
                #  m is part of k_i
                # print("Merge spike %s with template %s" % (i,j))
                templates[j] = update_template(m, k)
                merged = True
                break
        if not merged:
            templates.append(m)
            print("Spike %s makes new template %s" % (i, len(templates) -1))

    
    templates = np.array(templates)
    
    if verbose:
        for i, template in enumerate(templates):
            plot(template, title="MUAP " + str(i))
        
    return templates
    
    

def plot(sig, title = "Plot of CT signal", sampling_rate=1,
         xlabel="t", ylabel="x(t)", markers=None, save=False,
         begin=0, end=1000):
    if end <= 0 or end > sig.shape[0]:
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
    moving_avg_win_size=20
    diffTh=1275000
    raw_signal = pd.read_csv("Data.txt", header=None)[0]
    raw_signal = raw_signal.reset_index()[0]
    timestamps, templates = decompose(raw_signal,
                                      moving_avg_win_size=moving_avg_win_size,
                                      diffTh=diffTh, verbose=True)
    print("%s timstamps are detected, reduced to %s templates"\
          % (timestamps.shape[0], templates.shape[0]))
 
    # TODO A figure showing from sample 30000 to sample 35000 of the EMG signal
    # with an “*” marking the detected MUAPs colored with different colors
    # depending on the MU each MUAP belongs to (Similar to slide 19). 
    # Name the figure “DetectedMUAP.jpg”

    
    # TODO A figure showing the waveform of each template of the detected MUs
    # (Similar to slide 20). Name the figure “Templates.jpg”