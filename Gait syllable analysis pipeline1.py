import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_prominences
from scipy import signal
import tkinter.messagebox as messagebox

def highpass_filter(sig, cutoff_freq, order):
    nyquist_freq = 0.5 * fs
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normalized_cutoff, btype='high', analog=False)
    filtered_sig = signal.lfilter(b, a, sig)
    return filtered_sig

def lowpass_filter(sig, cutoff_freq, order):
    nyquist_freq = 0.5 * fs
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_sig = signal.lfilter(b, a, sig)
    return filtered_sig

def bandpass_filter(sig, low_cutoff_freq, high_cutoff_freq, order):
    nyquist_freq = 0.5 * fs
    normalized_low = low_cutoff_freq / nyquist_freq
    normalized_high = high_cutoff_freq / nyquist_freq
    b, a = signal.butter(order, [normalized_low, normalized_high],
                         btype='band', analog=False)
    filtered_sig = signal.lfilter(b, a, sig)
    return filtered_sig

def detect_movement_bouts_to_valley(ast2_data, immobility_threshold=0.1, min_immobility_duration=4, 
                                   min_movement_duration=0.5, max_expand_time=6.0):
    """Detect movement bouts and expand to nearest valleys in the speed signal"""
    try:
        timestamps = ast2_data['data']['timestamps']
        speed = ast2_data['data']['speed']
        
        if len(timestamps) != len(speed):
            if len(timestamps) == len(speed) + 1:
                timestamps = timestamps[:len(speed)]
            else:
                timestamps = np.linspace(timestamps[0], timestamps[-1], len(speed))
        
        if len(timestamps) > 1:
            sample_interval = np.mean(np.diff(timestamps))
            sample_rate = 1.0 / sample_interval
        else:
            sample_rate = 20.0
        
        min_immobility_samples = int(min_immobility_duration * sample_rate)
        max_expand_samples = int(max_expand_time * sample_rate)
        
        # Find valleys in the speed signal
        valley_prominence = 0.01
        valleys, _ = find_peaks(-speed, distance=int(0.5*sample_rate), prominence=valley_prominence)
        
        immobility_mask = np.abs(speed) < immobility_threshold
        
        immobility_regions = find_contiguous_regions(immobility_mask, min_immobility_samples)
        
        movement_regions = []
        
        if not immobility_regions:
            movement_regions.append((0, len(timestamps) - 1))
        else:
            if immobility_regions[0][0] > 0:
                movement_regions.append((0, immobility_regions[0][0] - 1))
            
            for i in range(len(immobility_regions) - 1):
                start = immobility_regions[i][1] + 1
                end = immobility_regions[i + 1][0] - 1
                duration = (end - start + 1) * sample_interval
                if start <= end and duration >= min_movement_duration:
                    movement_regions.append((start, end))
            
            if immobility_regions[-1][1] < len(timestamps) - 1:
                movement_regions.append((immobility_regions[-1][1] + 1, len(timestamps) - 1))
        
        movement_bouts = []
        total_samples = len(speed)
        
        for movement_start, movement_end in movement_regions:
            prev_immobility = False
            for imm_start, imm_end in immobility_regions:
                if imm_end <= movement_start:
                    immobility_duration = (imm_end - imm_start) * sample_interval
                    if immobility_duration >= min_immobility_duration:
                        prev_immobility = True
                        break
            
            next_immobility = False
            remaining_time = timestamps[-1] - timestamps[movement_end]
            
            if remaining_time < min_immobility_duration:
                next_immobility = False
            else:
                for imm_start, imm_end in immobility_regions:
                    if imm_start >= movement_end:
                        immobility_duration = (imm_end - imm_start) * sample_interval
                        if immobility_duration >= min_immobility_duration:
                            next_immobility = True
                            break
            
            if prev_immobility and (next_immobility or remaining_time < min_immobility_duration):
                # Find nearest valley before movement start
                valleys_before = [v for v in valleys if v <= movement_start]
                
                if len(valleys_before) > 0:
                    # Find the closest valley before movement start
                    valley_idx_before = valleys_before[-1]
                    
                    # Ensure we don't expand too far back
                    expand_start = max(0, valley_idx_before)
                    if (movement_start - expand_start) * sample_interval <= max_expand_time:
                        onset_idx = expand_start
                    else:
                        onset_idx = max(0, movement_start - max_expand_samples)
                else:
                    onset_idx = max(0, movement_start - max_expand_samples)
                
                # Find nearest valley after movement end
                valleys_after = [v for v in valleys if v >= movement_end]
                
                if len(valleys_after) > 0:
                    # Find the closest valley after movement end
                    valley_idx_after = valleys_after[0]
                    
                    # Ensure we don't expand too far forward
                    expand_end = min(total_samples - 1, valley_idx_after)
                    if (expand_end - movement_end) * sample_interval <= max_expand_time:
                        offset_idx = expand_end
                    else:
                        offset_idx = min(total_samples - 1, movement_end + max_expand_samples)
                else:
                    offset_idx = min(total_samples - 1, movement_end + max_expand_samples)
                
                movement_bouts.append((
                    timestamps[onset_idx],
                    timestamps[offset_idx]
                ))
        
        return movement_bouts
        
    except Exception as e:
        messagebox.showerror("ERROR", f"Error in detect_movement_bouts_to_valley: {str(e)}")
        return []

def find_contiguous_regions(mask, min_samples):
    """Find contiguous regions in a boolean mask with minimum length"""
    regions = []
    in_region = False
    region_start = 0
    
    for i in range(len(mask)):
        if mask[i] and not in_region:
            in_region = True
            region_start = i
        elif not mask[i] and in_region:
            in_region = False
            region_end = i - 1
            if region_end - region_start + 1 >= min_samples:
                regions.append((region_start, region_end))
    
    if in_region and len(mask) - region_start >= min_samples:
        regions.append((region_start, len(mask) - 1))
    
    return regions

def interpolate_low_reliability(values, reliability, threshold=0.90):
    valid_mask = reliability >= threshold
    indices = np.arange(len(values))
    
    if np.sum(valid_mask) > 1:
        interp_func = interp1d(
            indices[valid_mask], 
            values[valid_mask], 
            kind='linear', 
            fill_value="extrapolate"
        )
        interpolated = interp_func(indices)
        return interpolated
    else:
        return values

fs = 30
threadmill_diameter = 22  # in cm
Coefficient = np.pi*threadmill_diameter*12/1024/63  # convert pixel to cm

dlc_result_path = r"D:\Expriment\Data\Acethylcholine\637_day22DLC_resnet101_CINJan4shuffle1_500000.csv"
dlc_result = pd.read_csv(dlc_result_path, header=[0, 1, 2], low_memory=False)
scorer = dlc_result.columns.levels[0][0]
foot_front_left2 = 'foot_front_left2'
foot_front_right2 = 'foot_front_right2'
foot_hint_left2 = 'foot_hint_left2'
foot_hint_right2 = 'foot_hint_right2'

ffl_x = dlc_result.loc[:, (scorer, foot_front_left2, 'x')].values.astype(float)*Coefficient
ffl_y = dlc_result.loc[:, (scorer, foot_front_left2, 'y')].values.astype(float)*Coefficient
ffl_r = dlc_result.loc[:, (scorer, foot_front_left2, 'likelihood')].values.astype(float)

ffr_x = dlc_result.loc[:, (scorer, foot_front_right2, 'x')].values.astype(float)*Coefficient
ffr_y = dlc_result.loc[:, (scorer, foot_front_right2, 'y')].values.astype(float)*Coefficient
ffr_r = dlc_result.loc[:, (scorer, foot_front_right2, 'likelihood')].values.astype(float)

fhl_x = dlc_result.loc[:, (scorer, foot_hint_left2, 'x')].values.astype(float)*Coefficient
fhl_y = dlc_result.loc[:, (scorer, foot_hint_left2, 'y')].values.astype(float)*Coefficient
fhl_r = dlc_result.loc[:, (scorer, foot_hint_left2, 'likelihood')].values.astype(float)

fhr_x = dlc_result.loc[:, (scorer, foot_hint_right2, 'x')].values.astype(float)*Coefficient
fhr_y = dlc_result.loc[:, (scorer, foot_hint_right2, 'y')].values.astype(float)*Coefficient
fhr_r = dlc_result.loc[:, (scorer, foot_hint_right2, 'likelihood')].values.astype(float)

time_series = list(np.arange(1, len(ffl_x)+1)/fs)

ffl_x_before = ffl_x.copy()
ffl_y_before = ffl_y.copy()
ffr_x_before = ffr_x.copy()
ffr_y_before = ffr_y.copy()
fhl_x_before = fhl_x.copy()
fhl_y_before = fhl_y.copy()
fhr_x_before = fhr_x.copy()
fhr_y_before = fhr_y.copy()

ffl_x = interpolate_low_reliability(ffl_x, ffl_r, 0.90)
ffl_y = interpolate_low_reliability(ffl_y, ffl_r, 0.90)
ffr_x = interpolate_low_reliability(ffr_x, ffr_r, 0.90)
ffr_y = interpolate_low_reliability(ffr_y, ffr_r, 0.90)
fhl_x = interpolate_low_reliability(fhl_x, fhl_r, 0.90)
fhl_y = interpolate_low_reliability(fhl_y, fhl_r, 0.90)
fhr_x = interpolate_low_reliability(fhr_x, fhr_r, 0.90)
fhr_y = interpolate_low_reliability(fhr_y, fhr_r, 0.90)

ffl_x_smoothed = pd.Series(ffl_x).rolling(window=7, center=True, min_periods=1).mean().values
ffl_y_smoothed = pd.Series(ffl_y).rolling(window=7, center=True, min_periods=1).mean().values
ffr_x_smoothed = pd.Series(ffr_x).rolling(window=7, center=True, min_periods=1).mean().values
ffr_y_smoothed = pd.Series(ffr_y).rolling(window=7, center=True, min_periods=1).mean().values
fhl_x_smoothed = pd.Series(fhl_x).rolling(window=7, center=True, min_periods=1).mean().values
fhl_y_smoothed = pd.Series(fhl_y).rolling(window=7, center=True, min_periods=1).mean().values
fhr_x_smoothed = pd.Series(fhr_x).rolling(window=7, center=True, min_periods=1).mean().values
fhr_y_smoothed = pd.Series(fhr_y).rolling(window=7, center=True, min_periods=1).mean().values

low_pass_ffl_x = lowpass_filter(ffl_x, cutoff_freq=3, order=8)
low_pass_ffl_y = lowpass_filter(ffl_y, cutoff_freq=3, order=8)
low_pass_ffr_x = lowpass_filter(ffr_x, cutoff_freq=3, order=8)
low_pass_ffr_y = lowpass_filter(ffr_y, cutoff_freq=3, order=8)
low_pass_fhl_x = lowpass_filter(fhl_x, cutoff_freq=3, order=8)
low_pass_fhl_y = lowpass_filter(fhl_y, cutoff_freq=3, order=8)
low_pass_fhr_x = lowpass_filter(fhr_x, cutoff_freq=3, order=8)
low_pass_fhr_y = lowpass_filter(fhr_y, cutoff_freq=3, order=8)

# plt.figure(figsize=(16, 9))

# plt.subplot(2, 2, 1)
# plt.scatter(low_pass_ffl_x, low_pass_ffl_y, c=time_series, cmap='viridis', 
#            alpha=0.6, s=1)
# plt.colorbar(label='Time (s)')
# plt.title('Left Front Foot X-Y Trajectory (Low-pass Filtered)')
# plt.xlabel('X Position (cm)')
# plt.ylabel('Y Position (cm)')
# plt.grid(False)

# plt.subplot(2, 2, 2)
# plt.scatter(low_pass_ffr_x, low_pass_ffr_y, c=time_series, cmap='viridis', 
#            alpha=0.6, s=1)
# plt.colorbar(label='Time (s)')
# plt.title('Right Front Foot X-Y Trajectory (Low-pass Filtered)')
# plt.xlabel('X Position (cm)')
# plt.ylabel('Y Position (cm)')
# plt.grid(False)

# plt.subplot(2, 2, 3)
# plt.scatter(low_pass_fhl_x, low_pass_fhl_y, c=time_series, cmap='viridis', 
#            alpha=0.6, s=1)
# plt.colorbar(label='Time (s)')
# plt.title('Left Hind Foot X-Y Trajectory (Low-pass Filtered)')
# plt.xlabel('X Position (cm)')
# plt.ylabel('Y Position (cm)')
# plt.grid(False)

# plt.subplot(2, 2, 4)
# plt.scatter(low_pass_fhr_x, low_pass_fhr_y, c=time_series, cmap='viridis', 
#            alpha=0.6, s=1)
# plt.colorbar(label='Time (s)')
# plt.title('Right Hind Foot X-Y Trajectory (Low-pass Filtered)')
# plt.xlabel('X Position (cm)')
# plt.ylabel('Y Position (cm)')
# plt.grid(False)

# plt.tight_layout()
# plt.show()

speed_ffl_x = abs(np.diff(ffl_x))

low_pass_speed_ffl_x = lowpass_filter(speed_ffl_x, cutoff_freq=0.12, order=8)

# valleys_low_pass_speed_ffl_x, _ = find_peaks(-low_pass_speed_ffl_x, distance=15, prominence=0.01)

speed_data = {"speed": low_pass_speed_ffl_x,
              "timestamps": time_series[:-1]
              }
data = {"data": speed_data}

bouts = detect_movement_bouts_to_valley(data, immobility_threshold=0.2, min_immobility_duration=5, min_movement_duration=2, max_expand_time=6)

f, t, Zxx = signal.stft(ffl_x, fs=fs, window='hann', nperseg=256)
plt.figure(figsize=(16, 9))
plt.subplot(2, 1, 1)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.axhline(y=3, color='r', linestyle='--', alpha=0.8, label='Cutoff Frequency (3 Hz)')
plt.ylim(0, 4)
plt.title('STFT Magnitude')
plt.ylabel('Frequency(Hz)')
plt.xlabel('Time(s)')
plt.colorbar(label='Magnitude', orientation='horizontal')

plt.subplot(2, 1, 2)
plt.plot(time_series, ffl_x, label='Before Low-pass Filter', color='g', alpha=0.7)
plt.plot(time_series, low_pass_ffl_x, label='After Low-pass Filter', color='r', alpha=0.7)
plt.xlim(time_series[0], time_series[-1])
plt.title("x distance before vs. after Low-pass Filter of Left Front Foot")
plt.xlabel("Time(s)")
plt.ylabel("X distance(cm)")

plt.legend()
plt.tight_layout()
plt.show()

f1, t1, Zxx1 = signal.stft(speed_ffl_x, fs=fs, window='hann', nperseg=256)

plt.figure(figsize=(16, 9))
plt.subplot(4, 1, 1)
plt.plot(time_series, ffl_x, label='x distance of left front foot', color='g', alpha=0.7)
plt.xlim(time_series[0], time_series[-1])
plt.title("x distance of left front foot")
plt.xlabel("time(s)")
plt.ylabel("X distance(cm)")
plt.grid(False)
plt.subplot(4, 1, 2)
plt.plot(time_series[:-1], speed_ffl_x, label='speed of left front foot', color='g', alpha=0.7)
plt.xlim(time_series[0], time_series[-1])
plt.title("speed of left front foot")
plt.xlabel("time(s)")
plt.ylabel("speed in x direction(cm/s)")
plt.grid(False)
plt.subplot(4, 1, 3)
plt.pcolormesh(t1, f1, np.abs(Zxx1), shading='gouraud')
plt.axhline(y=0.12, color='r', linestyle='--', alpha=0.8, label='cutoff frequency (0.12 hz)')
plt.ylim(0, 0.2)
plt.title('stft magnitude')
plt.ylabel('frequency(hz)')
plt.xlabel('time(s)')
# plt.colorbar(label='magnitude')
plt.subplot(4, 1, 4)
plt.plot(time_series[:-1], low_pass_speed_ffl_x, label='speed of left front foot (Low-pass filtered)', color='g', alpha=0.7)
# plt.plot(np.array(time_series)[valleys_low_pass_speed_ffl_x], low_pass_speed_ffl_x[valleys_low_pass_speed_ffl_x], "o", color='g', markersize=6)
for bout in bouts:
    start_time, end_time = bout
    plt.axvspan(start_time, end_time, color='red', alpha=0.3, label='Movement Bout' if bout == bouts[0] else "")
plt.xlim(time_series[0], time_series[-1])
plt.title("speed of left front foot (low-pass filtered)")
plt.xlabel("time(s)")
plt.ylabel("speed in x direction(cm/s)")
plt.grid(False)
plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 9))

plt.subplot(4, 1, 1)
plt.plot(time_series, ffl_x_before, label='Before Interpolation', color='k', alpha=0.7)
plt.plot(time_series, ffl_x, label='After Interpolation', color='r', alpha=0.7)
plt.plot(time_series, ffl_x_smoothed, label='After Smoothing', color='b', alpha=0.7)
plt.plot(time_series, low_pass_ffl_x, label='After Low-pass Filter', color='g', alpha=0.7)
plt.title("x distance of Left Front Foot Before vs. After Interpolation vs. Smooth")
plt.xlim(time_series[-1000], time_series[-1])
plt.xlabel("Time(s)")
plt.ylabel("X distance(cm)")
# plt.legend()
plt.grid(False)

plt.subplot(4, 1, 2)
plt.plot(time_series, ffr_x_before, label='Before Interpolation', color='k', alpha=0.7)
plt.plot(time_series, ffr_x, label='After Interpolation', color='r', alpha=0.7)
plt.plot(time_series, ffr_x_smoothed, label='After Smoothing', color='b', alpha=0.7)
plt.plot(time_series, low_pass_ffr_x, label='After Low-pass Filter', color='g', alpha=0.7)
plt.title("x distance of Right Front Foot Before vs. After Interpolation vs. Smooth")
plt.xlim(time_series[-1000], time_series[-1])
plt.xlabel("Time(s)")
plt.ylabel("X distance(cm)")
# plt.legend()
plt.grid(False)

plt.subplot(4, 1, 3)
plt.plot(time_series, fhl_x_before, label='Before Interpolation', color='k', alpha=0.7)
plt.plot(time_series, fhl_x, label='After Interpolation', color='r', alpha=0.7)
plt.plot(time_series, fhl_x_smoothed, label='After Smoothing', color='b', alpha=0.7)
plt.plot(time_series, low_pass_fhl_x, label='After Low-pass Filter', color='g', alpha=0.7)
plt.title("x distance of Left Hint Foot Before vs. After Interpolation vs. Smooth")
plt.xlim(time_series[-1000], time_series[-1])
plt.xlabel("Time(s)")
plt.ylabel("X distance(cm)")
# plt.legend()
plt.grid(False)

plt.subplot(4, 1, 4)
plt.plot(time_series, fhr_x_before, label='Before Interpolation', color='k', alpha=0.7)
plt.plot(time_series, fhr_x, label='After Interpolation', color='r', alpha=0.7)
plt.plot(time_series, fhr_x_smoothed, label='After Smoothing', color='b', alpha=0.7)
plt.plot(time_series, low_pass_fhr_x, label='After Low-pass Filter', color='g', alpha=0.7)
plt.title("x distance of Right Hint Foot Before vs. After Interpolation vs. Smooth")
plt.xlim(time_series[-1000], time_series[-1])
plt.xlabel("Time(s)")
plt.ylabel("X distance(cm)")
plt.legend()
plt.grid(False)

plt.tight_layout()
plt.show()

peak_prominences = 5*Coefficient

peaksffl_x, _ = find_peaks(ffl_x, distance=5, prominence=peak_prominences)
peaksffr_x, _ = find_peaks(ffr_x, distance=5, prominence=peak_prominences)
peaksfhl_x, _ = find_peaks(fhl_x, distance=5, prominence=peak_prominences)
peaksfhr_x, _ = find_peaks(fhr_x, distance=5, prominence=peak_prominences)

valleysffl_x, _ = find_peaks(-ffl_x, distance=5, prominence=peak_prominences)
valleysffr_x, _ = find_peaks(-ffr_x, distance=5, prominence=peak_prominences)
valleysfhl_x, _ = find_peaks(-fhl_x, distance=5, prominence=peak_prominences)
valleysfhr_x, _ = find_peaks(-fhr_x, distance=5, prominence=peak_prominences)

peaks_ffl_x_smoothed, _ = find_peaks(ffl_x_smoothed, distance=5, prominence=peak_prominences)
peaks_ffr_x_smoothed, _ = find_peaks(ffr_x_smoothed, distance=5, prominence=peak_prominences)
peaks_fhl_x_smoothed, _ = find_peaks(fhl_x_smoothed, distance=5, prominence=peak_prominences)
peaks_fhr_x_smoothed, _ = find_peaks(fhr_x_smoothed, distance=5, prominence=peak_prominences)

valleys_ffl_x_smoothed, _ = find_peaks(-ffl_x_smoothed, distance=5, prominence=peak_prominences)
valleys_ffr_x_smoothed, _ = find_peaks(-ffr_x_smoothed, distance=5, prominence=peak_prominences)
valleys_fhl_x_smoothed, _ = find_peaks(-fhl_x_smoothed, distance=5, prominence=peak_prominences)
valleys_fhr_x_smoothed, _ = find_peaks(-fhr_x_smoothed, distance=5, prominence=peak_prominences)

peaksffl_x_low_pass, _ = find_peaks(low_pass_ffl_x, distance=5, prominence=peak_prominences)
peaksffr_x_low_pass, _ = find_peaks(low_pass_ffr_x, distance=5, prominence=peak_prominences)
peaksfhl_x_low_pass, _ = find_peaks(low_pass_fhl_x, distance=5, prominence=peak_prominences)
peaksfhr_x_low_pass, _ = find_peaks(low_pass_fhr_x, distance=5, prominence=peak_prominences)

valleysffl_x_low_pass, _ = find_peaks(-low_pass_ffl_x, distance=5, prominence=peak_prominences)
valleysffr_x_low_pass, _ = find_peaks(-low_pass_ffr_x, distance=5, prominence=peak_prominences)
valleysfhl_x_low_pass, _ = find_peaks(-low_pass_fhl_x, distance=5, prominence=peak_prominences)
valleysfhr_x_low_pass, _ = find_peaks(-low_pass_fhr_x, distance=5, prominence=peak_prominences)

shift_values = 400*Coefficient

plt.figure(figsize=(16, 9))
# plt.subplot(4, 1, 1)
plt.plot(time_series, ffl_x+shift_values, label='Raw', color='k', alpha=0.7)
plt.plot(time_series, ffl_x_smoothed, label='Smoothed', color='b', alpha=0.7)
plt.plot(time_series, low_pass_ffl_x-shift_values, label='Low-pass Filtered', color='g', alpha=0.7)
plt.plot(np.array(time_series)[peaksffl_x], ffl_x[peaksffl_x]+shift_values, "x", label='Peaks', color='r')
plt.plot(np.array(time_series)[valleysffl_x], ffl_x[valleysffl_x]+shift_values, "o", label='Valleys', color='g')
plt.plot(np.array(time_series)[peaks_ffl_x_smoothed], ffl_x_smoothed[peaks_ffl_x_smoothed], "x", color='r')
plt.plot(np.array(time_series)[valleys_ffl_x_smoothed], ffl_x_smoothed[valleys_ffl_x_smoothed], "o", color='g')
plt.plot(np.array(time_series)[peaksffl_x_low_pass], low_pass_ffl_x[peaksffl_x_low_pass]-shift_values, "x", color='r')
plt.plot(np.array(time_series)[valleysffl_x_low_pass], low_pass_ffl_x[valleysffl_x_low_pass]-shift_values, "o", color='g')
plt.xlim(time_series[-1000], time_series[-1])
plt.title("Detected Peaks and Valleys in Left Front Foot x distance")
plt.xlabel("Time(s)")
plt.ylabel("X distance(cm)")
plt.legend()
plt.grid(False)

# plt.subplot(4, 1, 2)
# plt.plot(time_series, ffr_x+shift_values, label='x distance of Right front Foot', color='k', alpha=0.7)
# plt.plot(time_series, ffr_x_smoothed, label='x distance of Right front Foot (Smoothed)', color='b', alpha=0.7)
# plt.plot(time_series, low_pass_ffr_x-shift_values, label='x distance of Right front Foot (Low-pass Filtered)', color='g', alpha=0.7)
# plt.plot(np.array(time_series)[peaksffr_x], ffr_x[peaksffr_x]+shift_values, "x", label='Peaks', color='r')
# plt.plot(np.array(time_series)[valleysffr_x], ffr_x[valleysffr_x]+shift_values, "o", label='Valleys', color='g')
# plt.plot(np.array(time_series)[peaks_ffr_x_smoothed], ffr_x_smoothed[peaks_ffr_x_smoothed], "x", label='Peaks', color='r')
# plt.plot(np.array(time_series)[valleys_ffr_x_smoothed], ffr_x_smoothed[valleys_ffr_x_smoothed], "o", label='Valleys', color='g')
# plt.plot(np.array(time_series)[peaksffr_x_low_pass], low_pass_ffr_x[peaksffr_x_low_pass]-shift_values, "x", label='Peaks (Low-pass)', color='r')
# plt.plot(np.array(time_series)[valleysffr_x_low_pass], low_pass_ffr_x[valleysffr_x_low_pass]-shift_values, "o", label='Valleys (Low-pass)', color='g')
# plt.xlim(time_series[-1000], time_series[-1])
# plt.title("Detected Peaks and Valleys in Right Front Foot x distance")
# plt.xlabel("Time(s)")
# plt.ylabel("X distance(cm)")
# plt.legend()
# plt.grid(False)

# plt.subplot(4, 1, 3)
# plt.plot(time_series, fhl_x+shift_values, label='x distance of Left hint Foot', color='k', alpha=0.7)
# plt.plot(time_series, fhl_x_smoothed, label='x distance of Left hint Foot (Smoothed)', color='b', alpha=0.7)
# plt.plot(time_series, low_pass_fhl_x-shift_values, label='x distance of Left hint Foot (Low-pass Filtered)', color='g', alpha=0.7)
# plt.plot(np.array(time_series)[peaksfhl_x], fhl_x[peaksfhl_x]+shift_values, "x", label='Peaks', color='r')
# plt.plot(np.array(time_series)[valleysfhl_x], fhl_x[valleysfhl_x]+shift_values, "o", label='Valleys', color='g')
# plt.plot(np.array(time_series)[peaks_fhl_x_smoothed], fhl_x_smoothed[peaks_fhl_x_smoothed], "x", label='Peaks', color='r')
# plt.plot(np.array(time_series)[valleys_fhl_x_smoothed], fhl_x_smoothed[valleys_fhl_x_smoothed], "o", label='Valleys', color='g')
# plt.plot(np.array(time_series)[peaksfhl_x_low_pass], low_pass_fhl_x[peaksfhl_x_low_pass]-shift_values, "x", label='Peaks (Low-pass)', color='r')
# plt.plot(np.array(time_series)[valleysfhl_x_low_pass], low_pass_fhl_x[valleysfhl_x_low_pass]-shift_values, "o", label='Valleys (Low-pass)', color='g')
# plt.xlim(time_series[-1000], time_series[-1])
# plt.title("Detected Peaks and Valleys in Left Hint Foot x distance")
# plt.xlabel("Time(s)")
# plt.ylabel("X distance(cm)")
# plt.legend()
# plt.grid(False)

# plt.subplot(4, 1, 4)
# plt.plot(time_series, fhr_x+shift_values, label='x distance of Right hint Foot', color='k', alpha=0.7)
# plt.plot(time_series, fhr_x_smoothed, label='x distance of Right hint Foot (Smoothed)', color='b', alpha=0.7)
# plt.plot(time_series, low_pass_fhr_x-shift_values, label='x distance of Right hint Foot (Low-pass Filtered)', color='g', alpha=0.7)
# plt.plot(np.array(time_series)[peaksfhr_x], fhr_x[peaksfhr_x]+shift_values, "x", label='Peaks', color='r')
# plt.plot(np.array(time_series)[valleysfhr_x], fhr_x[valleysfhr_x]+shift_values, "o", label='Valleys', color='g')
# plt.plot(np.array(time_series)[peaks_fhr_x_smoothed], fhr_x_smoothed[peaks_fhr_x_smoothed], "x", label='Peaks', color='r')
# plt.plot(np.array(time_series)[valleys_fhr_x_smoothed], fhr_x_smoothed[valleys_fhr_x_smoothed], "o", label='Valleys', color='g')
# plt.plot(np.array(time_series)[peaksfhr_x_low_pass], low_pass_fhr_x[peaksfhr_x_low_pass]-shift_values, "x", label='Peaks (Low-pass)', color='r')
# plt.plot(np.array(time_series)[valleysfhr_x_low_pass], low_pass_fhr_x[valleysfhr_x_low_pass]-shift_values, "o", label='Valleys (Low-pass)', color='g')
# plt.xlim(time_series[-1000], time_series[-1])
# plt.title("Detected Peaks and Valleys in Right Hint Foot x distance")
# plt.xlabel("Time(s)")
# plt.ylabel("X distance(cm)")
# plt.legend()
# plt.grid(False)

# plt.tight_layout()
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(time_series, low_pass_ffl_x-shift_values, label='Left front Foot', color='k', alpha=0.7)
plt.plot(np.array(time_series)[peaksffl_x_low_pass], low_pass_ffl_x[peaksffl_x_low_pass]-shift_values, "x", color='r')
plt.plot(np.array(time_series)[valleysffl_x_low_pass], low_pass_ffl_x[valleysffl_x_low_pass]-shift_values, "o", color='g')

plt.plot(time_series, low_pass_ffr_x, label='Right front Foot', color='b', alpha=0.7)
plt.plot(np.array(time_series)[peaksffr_x_low_pass], low_pass_ffr_x[peaksffr_x_low_pass], "x", color='r')
plt.plot(np.array(time_series)[valleysffr_x_low_pass], low_pass_ffr_x[valleysffr_x_low_pass], "o", color='g')

plt.plot(time_series, low_pass_fhl_x, label='Left hint Foot', color='g', alpha=0.7)
plt.plot(np.array(time_series)[peaksfhl_x_low_pass], low_pass_fhl_x[peaksfhl_x_low_pass], "x", color='r')
plt.plot(np.array(time_series)[valleysfhl_x_low_pass], low_pass_fhl_x[valleysfhl_x_low_pass], "o", color='g')

plt.plot(time_series, low_pass_fhr_x+shift_values, label='Right hint Foot', color='r', alpha=0.7)
plt.plot(np.array(time_series)[peaksfhr_x_low_pass], low_pass_fhr_x[peaksfhr_x_low_pass]+shift_values, "x", label='Peaks', color='r')
plt.plot(np.array(time_series)[valleysfhr_x_low_pass], low_pass_fhr_x[valleysfhr_x_low_pass]+shift_values, "o", label='Valleys', color='g')

plt.xlim(time_series[-1000], time_series[-1])
plt.title("Detected Peaks and Valleys in Low-Pass Filted Four Feet x distance")
plt.xlabel("Time(s)")
plt.ylabel("X distance(cm)")

plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

def plot_bout_peaks_valleys(bout_index, bout, time_series, low_pass_ffl_x, low_pass_ffr_x, 
                           low_pass_fhl_x, low_pass_fhr_x, peaksffl_x_low_pass, valleysffl_x_low_pass,
                           peaksffr_x_low_pass, valleysffr_x_low_pass, peaksfhl_x_low_pass, 
                           valleysfhl_x_low_pass, peaksfhr_x_low_pass, valleysfhr_x_low_pass,
                           shift_values=400*Coefficient):
    start_time, end_time = bout
    
    bout_indices = np.where((np.array(time_series) >= start_time) & (np.array(time_series) <= end_time))[0]
    
    if len(bout_indices) == 0:
        print(f"Bout {bout_index}: No data points in time range {start_time}-{end_time}")
        return
    
    bout_peaks_ffl = [p for p in peaksffl_x_low_pass if time_series[p] >= start_time and time_series[p] <= end_time]
    bout_valleys_ffl = [v for v in valleysffl_x_low_pass if time_series[v] >= start_time and time_series[v] <= end_time]
    
    bout_peaks_ffr = [p for p in peaksffr_x_low_pass if time_series[p] >= start_time and time_series[p] <= end_time]
    bout_valleys_ffr = [v for v in valleysffr_x_low_pass if time_series[v] >= start_time and time_series[v] <= end_time]
    
    bout_peaks_fhl = [p for p in peaksfhl_x_low_pass if time_series[p] >= start_time and time_series[p] <= end_time]
    bout_valleys_fhl = [v for v in valleysfhl_x_low_pass if time_series[v] >= start_time and time_series[v] <= end_time]
    
    bout_peaks_fhr = [p for p in peaksfhr_x_low_pass if time_series[p] >= start_time and time_series[p] <= end_time]
    bout_valleys_fhr = [v for v in valleysfhr_x_low_pass if time_series[v] >= start_time and time_series[v] <= end_time]
    
    plt.figure(figsize=(16, 9))
    
    plt.plot(np.array(time_series)[bout_indices], low_pass_ffl_x[bout_indices] - shift_values, 
             label='Left Front Foot', color='k', alpha=0.7)
    plt.plot(np.array(time_series)[bout_peaks_ffl], low_pass_ffl_x[bout_peaks_ffl] - shift_values, 
             "x", color='r', markersize=8)
    plt.plot(np.array(time_series)[bout_valleys_ffl], low_pass_ffl_x[bout_valleys_ffl] - shift_values, 
             "o", color='g', markersize=6)
    
    plt.plot(np.array(time_series)[bout_indices], low_pass_ffr_x[bout_indices], 
             label='Right Front Foot', color='b', alpha=0.7)
    plt.plot(np.array(time_series)[bout_peaks_ffr], low_pass_ffr_x[bout_peaks_ffr], 
             "x", color='r', markersize=8)
    plt.plot(np.array(time_series)[bout_valleys_ffr], low_pass_ffr_x[bout_valleys_ffr], 
             "o", color='g', markersize=6)
    
    plt.plot(np.array(time_series)[bout_indices], low_pass_fhl_x[bout_indices], 
             label='Left Hind Foot', color='g', alpha=0.7)
    plt.plot(np.array(time_series)[bout_peaks_fhl], low_pass_fhl_x[bout_peaks_fhl], 
             "x", color='r', markersize=8)
    plt.plot(np.array(time_series)[bout_valleys_fhl], low_pass_fhl_x[bout_valleys_fhl], 
             "o", color='g', markersize=6)
    
    plt.plot(np.array(time_series)[bout_indices], low_pass_fhr_x[bout_indices] + shift_values, 
             label='Right Hind Foot', color='r', alpha=0.7)
    plt.plot(np.array(time_series)[bout_peaks_fhr], low_pass_fhr_x[bout_peaks_fhr] + shift_values, 
             "x", color='r', markersize=8, label='Peaks')
    plt.plot(np.array(time_series)[bout_valleys_fhr], low_pass_fhr_x[bout_valleys_fhr] + shift_values, 
             "o", color='g', markersize=6, label='Valleys')
    
    plt.title(f"Movement Bout {bout_index + 1}: Peaks and Valleys in Four Feet x distance\n"
              f"Time: {start_time:.2f}s - {end_time:.2f}s (Duration: {end_time-start_time:.2f}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("X distance (cm)")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    print(f"Bout {bout_index + 1}: {start_time:.2f}s - {end_time:.2f}s "
          f"(Duration: {end_time-start_time:.2f}s)")
    print(f"  Left Front Foot: {len(bout_peaks_ffl)} peaks, {len(bout_valleys_ffl)} valleys")
    print(f"  Right Front Foot: {len(bout_peaks_ffr)} peaks, {len(bout_valleys_ffr)} valleys")
    print(f"  Left Hind Foot: {len(bout_peaks_fhl)} peaks, {len(bout_valleys_fhl)} valleys")
    print(f"  Right Hind Foot: {len(bout_peaks_fhr)} peaks, {len(bout_valleys_fhr)} valleys")
    print()

print(f"Total movement bouts detected: {len(bouts)}")
# for i, bout in enumerate(bouts):
#     plot_bout_peaks_valleys(i, bout, time_series, low_pass_ffl_x, low_pass_ffr_x,
#                            low_pass_fhl_x, low_pass_fhr_x, peaksffl_x_low_pass, valleysffl_x_low_pass,
#                            peaksffr_x_low_pass, valleysffr_x_low_pass, peaksfhl_x_low_pass,
#                            valleysfhl_x_low_pass, peaksfhr_x_low_pass, valleysfhr_x_low_pass)

def analyze_gait_pattern(bout_peaks_valleys, time_series, foot_names=['LF', 'RF', 'LH', 'RH']):
    all_events = []
    
    for foot_idx, foot_name in enumerate(foot_names):
        peaks = bout_peaks_valleys[foot_idx]['peaks']
        valleys = bout_peaks_valleys[foot_idx]['valleys']
        
        for p in peaks:
            all_events.append({
                'time': time_series[p],
                'index': p,
                'foot': foot_name,
                'type': 'peak'
            })
        
        for v in valleys:
            all_events.append({
                'time': time_series[v],
                'index': v,
                'foot': foot_name,
                'type': 'valley'
            })
    
    all_events = sorted(all_events, key=lambda x: x['time'])
    
    gait_cycles = []
    
    for foot_idx, foot_name in enumerate(foot_names):
        peaks = bout_peaks_valleys[foot_idx]['peaks']
        
        for i in range(len(peaks) - 1):
            cycle_start = time_series[peaks[i]]
            cycle_end = time_series[peaks[i + 1]]
            
            cycle_events = [e for e in all_events 
                          if cycle_start <= e['time'] <= cycle_end]
            
            if len(cycle_events) > 0:
                gait_cycles.append({
                    'reference_foot': foot_name,
                    'start_time': cycle_start,
                    'end_time': cycle_end,
                    'duration': cycle_end - cycle_start,
                    'events': cycle_events
                })
    
    return gait_cycles, all_events

def calculate_phase_relationships(gait_cycles, foot_names=['LF', 'RF', 'LH', 'RH']):
    phase_data = {f'{f1}-{f2}': [] for f1 in foot_names for f2 in foot_names if f1 != f2}
    
    for cycle in gait_cycles:
        ref_foot = cycle['reference_foot']
        duration = cycle['duration']
        
        if duration == 0:
            continue
        
        ref_time = cycle['start_time']
        
        for event in cycle['events']:
            if event['foot'] != ref_foot and event['type'] == 'peak':
                phase = (event['time'] - ref_time) / duration
                key = f"{ref_foot}-{event['foot']}"
                if key in phase_data:
                    phase_data[key].append(phase)
    
    return phase_data

def plot_gait_cycle_analysis(bout_index, bout, time_series, low_pass_signals, 
                             peaks_list, valleys_list, shift_values,
                             foot_names=['LF', 'RF', 'LH', 'RH'],
                             colors=['black', 'blue', 'green', 'red']):
    start_time, end_time = bout
    bout_indices = np.where((np.array(time_series) >= start_time) & 
                           (np.array(time_series) <= end_time))[0]
    
    if len(bout_indices) == 0:
        return
    
    bout_peaks_valleys = []
    shifts = [-shift_values, 0, 0, shift_values]
    
    for i in range(4):
        bout_peaks = [p for p in peaks_list[i] 
                     if start_time <= time_series[p] <= end_time]
        bout_valleys = [v for v in valleys_list[i] 
                       if start_time <= time_series[v] <= end_time]
        bout_peaks_valleys.append({'peaks': bout_peaks, 'valleys': bout_valleys})
    
    gait_cycles, all_events = analyze_gait_pattern(bout_peaks_valleys, time_series, foot_names)
    # print(gait_cycles)
    phase_relationships = calculate_phase_relationships(gait_cycles, foot_names)
    # print(phase_relationships)
    
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 1, hspace=0.4)
    
    ax1 = fig.add_subplot(gs[0])
    for i in range(4):
        signal_data = low_pass_signals[i][bout_indices] + shifts[i]
        ax1.plot(np.array(time_series)[bout_indices], signal_data,
                label=foot_names[i], color=colors[i], alpha=0.7, linewidth=2)
        
        peaks = bout_peaks_valleys[i]['peaks']
        valleys = bout_peaks_valleys[i]['valleys']
        ax1.plot(np.array(time_series)[peaks], 
                low_pass_signals[i][peaks] + shifts[i],
                "x", color='r', markersize=8)
        ax1.plot(np.array(time_series)[valleys], 
                low_pass_signals[i][valleys] + shifts[i],
                "o", color='g', markersize=6)
    
    ax1.set_xlim(start_time, end_time)
    ax1.set_title(f'Bout {bout_index + 1}: Four Limbs Trajectories\n'
                 f'Time: {start_time:.2f}s - {end_time:.2f}s', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('X Distance (cm)', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(False)
    
    # ax2 = fig.add_subplot(gs[1])
    # foot_y_positions = {name: i for i, name in enumerate(foot_names)}
    
    # for event in all_events:
    #     y_pos = foot_y_positions[event['foot']]
    #     marker = '^' if event['type'] == 'peak' else 'v'
    #     color = 'red' if event['type'] == 'peak' else 'green'
    #     ax2.plot(event['time'], y_pos, marker, color=color, markersize=12, 
    #             markeredgecolor='black', markeredgewidth=1)
    
    # lf_peaks = bout_peaks_valleys[0]['peaks']
    # for i in range(len(lf_peaks) - 1):
    #     cycle_start = time_series[lf_peaks[i]]
    #     cycle_end = time_series[lf_peaks[i + 1]]
    #     ax2.axvspan(cycle_start, cycle_end, alpha=0.1, 
    #                color='blue' if i % 2 == 0 else 'yellow')
    #     ax2.axvline(cycle_start, color='gray', linestyle='--', alpha=0.5)
    
    # ax2.set_yticks(range(len(foot_names)))
    # ax2.set_yticklabels(foot_names)
    # ax2.set_xlabel('Time (s)', fontsize=11)
    # ax2.set_title('Gait Event Sequence\n(delta Peak, delta Valley)', fontsize=12, fontweight='bold')
    # ax2.grid(False)
    # ax2.set_xlim(start_time, end_time)
    
    ax3 = fig.add_subplot(gs[1])
    phase_pairs = [k for k in phase_relationships.keys() if len(phase_relationships[k]) > 0]
    # print(phase_pairs)
    
    if phase_pairs:
        x_pos = np.arange(len(phase_pairs))
        means = [np.mean(phase_relationships[k]) for k in phase_pairs]
        stds = [np.std(phase_relationships[k]) for k in phase_pairs]
        n = len(phase_relationships)
        sems = stds / np.sqrt(n)
        
        bars = ax3.bar(x_pos, means, yerr=sems, capsize=5, alpha=0.7,
                      color=['steelblue', 'coral', 'lightgreen', 'plum', 'gold', 'cyan'][:len(phase_pairs)])
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(phase_pairs, rotation=45, ha='right')
        ax3.set_ylabel('Phase (normalized to cycle duration)', fontsize=11)
        ax3.set_title('Inter-limb Phase Relationships', fontsize=12, fontweight='bold')
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Anti-phase (0.5)')
        ax3.legend()
        ax3.grid(False)
    

    print(f"Gait Cycle Statistics:\n\n")
    print(f"Total Cycles (LF reference): {len([c for c in gait_cycles if c['reference_foot'] == 'LF'])}\n")
    
    stats_text = ""
    if gait_cycles:
        durations = [c['duration'] for c in gait_cycles if c['reference_foot'] == 'LF']
        if durations:
            stats_text += f"Average Cycle Duration: {np.mean(durations):.3f} +/- {np.std(durations):.3f} s\n"
            stats_text += f"Frequency: {1/np.mean(durations):.2f} Hz\n\n"
    
    stats_text += "Peak-Valley Counts:\n"
    for i, name in enumerate(foot_names):
        peaks = bout_peaks_valleys[i]['peaks']
        valleys = bout_peaks_valleys[i]['valleys']
        stats_text += f"  {name}: {len(peaks)} peaks, {len(valleys)} valleys\n"
    
    stats_text += "\nPhase Relationships (mean +/- std):\n"
    for pair in phase_pairs[:6]:
        if len(phase_relationships[pair]) > 0:
            mean_phase = np.mean(phase_relationships[pair])
            std_phase = np.std(phase_relationships[pair])
            n = len(phase_relationships[pair])
            sem_phase = std_phase / np.sqrt(n)
            stats_text += f"  {pair}: {mean_phase:.3f} +/- {sem_phase:.3f}\n"
    
    print(stats_text)

    return fig, gait_cycles, phase_relationships


print(f"\n{'='*60}")
print(f"Gait Cycle Analysis for {len(bouts)} Movement Bouts")
print(f"{'='*60}\n")

for i, bout in enumerate(bouts):
    low_pass_signals = [low_pass_ffl_x, low_pass_ffr_x, low_pass_fhl_x, low_pass_fhr_x]
    peaks_list = [peaksffl_x_low_pass, peaksffr_x_low_pass, peaksfhl_x_low_pass, peaksfhr_x_low_pass]
    valleys_list = [valleysffl_x_low_pass, valleysffr_x_low_pass, valleysfhl_x_low_pass, valleysfhr_x_low_pass]
    
    fig, gait_cycles, phase_rels = plot_gait_cycle_analysis(
        i, bout, time_series, low_pass_signals, peaks_list, valleys_list,
        shift_values, foot_names=['LF', 'RF', 'LH', 'RH'],
        colors=['black', 'blue', 'green', 'red']
    )
    
    plt.show()
    
    print(f"Bout {i + 1} Analysis Complete")
    print(f"-" * 60)

def analyze_limb_trajectories_per_bout(bout_index, bout, time_series, 
                                      low_pass_x_signals, low_pass_y_signals,
                                      peaks_list, valleys_list,
                                      foot_names=['LF', 'RF', 'LH', 'RH'],
                                      colors=['black', 'blue', 'green', 'red']):
    start_time, end_time = bout
    bout_indices = np.where((np.array(time_series) >= start_time) & 
                           (np.array(time_series) <= end_time))[0]
    
    if len(bout_indices) == 0:
        return None, None
    
    bout_peaks_valleys = []
    for i in range(4):
        bout_peaks = [p for p in peaks_list[i] 
                     if start_time <= time_series[p] <= end_time]
        bout_valleys = [v for v in valleys_list[i] 
                       if start_time <= time_series[v] <= end_time]
        bout_peaks_valleys.append({
            'peaks': sorted(bout_peaks),
            'valleys': sorted(bout_valleys)
        })
    
    pvp_sequences = {name: [] for name in foot_names}
    
    for i, name in enumerate(foot_names):
        peaks = bout_peaks_valleys[i]['peaks']
        valleys = bout_peaks_valleys[i]['valleys']
        
        for j in range(len(peaks) - 1):
            current_peak = peaks[j]
            next_peak = peaks[j + 1]
            
            between_valleys = [v for v in valleys if current_peak < v < next_peak]
            
            if between_valleys:
                valley = between_valleys[0]
                
                sequence_indices = list(range(current_peak, next_peak + 1))
                
                if len(sequence_indices) > 1:
                    x_trajectory = low_pass_x_signals[i][sequence_indices]
                    y_trajectory = low_pass_y_signals[i][sequence_indices]
                    
                    if len(x_trajectory) > 10:
                        norm_length = 100
                        t_original = np.linspace(0, 1, len(x_trajectory))
                        t_norm = np.linspace(0, 1, norm_length)
                        
                        x_norm = np.interp(t_norm, t_original, x_trajectory)
                        y_norm = np.interp(t_norm, t_original, y_trajectory)
                        
                        pvp_sequences[name].append({
                            'x': x_norm,
                            'y': y_norm,
                            'start_idx': current_peak,
                            'valley_idx': valley,
                            'end_idx': next_peak,
                            'start_time': time_series[current_peak],
                            'valley_time': time_series[valley],
                            'end_time': time_series[next_peak]
                        })
    
    return pvp_sequences, bout_peaks_valleys

def plot_bout_trajectory_analysis(bout_index, bout, pvp_sequences, 
                                 foot_names=['LF', 'RF', 'LH', 'RH'],
                                 colors=['black', 'blue', 'green', 'red']):
    if not any(len(sequences) > 0 for sequences in pvp_sequences.values()):
        print(f"Bout {bout_index + 1}: No valid peak-valley-peak sequences found")
        return
    
    # fig = plt.figure(figsize=(9, 9))
    
    # for i, name in enumerate(foot_names):
    #     sequences = pvp_sequences[name]
    #     if len(sequences) > 0:
    #         all_x = np.array([seq['x'] for seq in sequences])
    #         all_y = np.array([seq['y'] for seq in sequences])
            
    #         mean_x = np.mean(all_x, axis=0)
    #         mean_y = np.mean(all_y, axis=0)
    #         std_x = np.std(all_x, axis=0)
    #         std_y = np.std(all_y, axis=0)
            
    #         plt.plot(mean_x, mean_y, color=colors[i], linewidth=2, 
    #                 label=f'{name} (n={len(sequences)})')
            
    #         plt.fill_betweenx(mean_y, mean_x - std_x, mean_x + std_x, 
    #                          color=colors[i], alpha=0.2)
    
    # plt.xlabel('X Position (cm)')
    # plt.ylabel('Y Position (cm)')
    # plt.title(f'Bout {bout_index + 1}: Average P-V-P Trajectories\nwith Standard Deviation')
    # plt.legend()
    # plt.grid(False)
    # # plt.axis('equal')
    # plt.show()
    
    fig = plt.figure(figsize=(9, 9))
    
    all_points = []
    for i, name in enumerate(foot_names):
        sequences = pvp_sequences[name]
        for seq in sequences:
            points = np.column_stack([seq['x'], seq['y']])
            all_points.extend(points)
    
    if all_points:
        all_points = np.array(all_points)
        
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        heatmap, xedges, yedges = np.histogram2d(all_points[:, 0], all_points[:, 1], 
                                               bins=50, density=True)
        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = plt.imshow(heatmap.T, extent=extent, origin='lower', 
                       cmap='hot', aspect='auto')
        # plt.colorbar(im, label='Density')
        
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.title(f'Bout {bout_index + 1}: Spatial Distribution Heatmap')
        # plt.axis('equal')

    plt.show()
    
    fig = plt.figure(figsize=(9, 9))
    
    for i, name in enumerate(foot_names):
        sequences = pvp_sequences[name]
        for seq in sequences:
            # plt.plot(seq['x'], seq['y'], color=colors[i], alpha=0.1, linewidth=0.5)
            plt.scatter(seq['x'], seq['y'],color=colors[i], alpha=0.1, s=1)
        
        if sequences:
            all_x = np.array([seq['x'] for seq in sequences])
            all_y = np.array([seq['y'] for seq in sequences])
            mean_x = np.mean(all_x, axis=0)
            mean_y = np.mean(all_y, axis=0)
            plt.plot(mean_x, mean_y, color=colors[i], linewidth=2, label=name)
    
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.title(f'Bout {bout_index + 1}: Individual P-V-P Trajectories')
    plt.legend()
    plt.grid(False)
    # plt.axis('equal')
    plt.show()
    
    stats_text = f"Bout {bout_index + 1} Statistics:\n\n"
    total_sequences = 0
    
    for name in foot_names:
        sequences = pvp_sequences[name]
        stats_text += f"{name}: {len(sequences)} sequences\n"
        total_sequences += len(sequences)
        
        if sequences:
            durations = [seq['end_time'] - seq['start_time'] for seq in sequences]
            stats_text += f"  Duration: {np.mean(durations):.3f} ± {np.std(durations):.3f} s\n"
            
            distances = []
            for seq in sequences:
                dx = np.diff(seq['x'])
                dy = np.diff(seq['y'])
                distance = np.sum(np.sqrt(dx**2 + dy**2))
                distances.append(distance)
            
            stats_text += f"  Distance: {np.mean(distances):.2f} ± {np.std(distances):.2f} cm\n\n"
    
    stats_text += f"Total sequences: {total_sequences}"
    
    print(stats_text)
    
    return fig

def plot_all_bouts_trajectory_analysis(all_pvp_sequences, 
                                      foot_names=['LF', 'RF', 'LH', 'RH'],
                                      colors=['black', 'blue', 'green', 'red']):
    combined_sequences = {name: [] for name in foot_names}
    
    for bout_data in all_pvp_sequences:
        for name in foot_names:
            if name in bout_data and bout_data[name]:
                combined_sequences[name].extend(bout_data[name])
    
    if not any(len(sequences) > 0 for sequences in combined_sequences.values()):
        print("No valid peak-valley-peak sequences found across all bouts")
        return
    
    fig = plt.figure(figsize=(9, 9))

    
    for i, name in enumerate(foot_names):
        sequences = combined_sequences[name]
        if len(sequences) > 0:
            all_x = np.array([seq['x'] for seq in sequences])
            all_y = np.array([seq['y'] for seq in sequences])
            
            mean_x = np.mean(all_x, axis=0)
            mean_y = np.mean(all_y, axis=0)
            std_x = np.std(all_x, axis=0)
            std_y = np.std(all_y, axis=0)
            
            plt.plot(mean_x, mean_y, color=colors[i], linewidth=2, 
                    label=f'{name} (n={len(sequences)})')
            
            plt.fill_betweenx(mean_y, mean_x - std_x, mean_x + std_x, 
                             color=colors[i], alpha=0.2)
    
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.title('All Bouts: Average P-V-P Trajectories\nwith Standard Deviation')
    plt.legend()
    plt.grid(False)
    # plt.axis('equal')
    plt.show()
    
    fig = plt.figure(figsize=(9, 9))
    
    all_points = []
    for i, name in enumerate(foot_names):
        sequences = combined_sequences[name]
        for seq in sequences:
            points = np.column_stack([seq['x'], seq['y']])
            all_points.extend(points)
    
    if all_points:
        all_points = np.array(all_points)
        
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        
        heatmap, xedges, yedges = np.histogram2d(all_points[:, 0], all_points[:, 1], 
                                               bins=50, density=True)
        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = plt.imshow(heatmap.T, extent=extent, origin='lower', 
                       cmap='hot', aspect='auto')
        # plt.colorbar(im, label='Density')
        
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.title('All Bouts: Spatial Distribution Heatmap')
        # plt.axis('equal')
    
    plt.show()

    fig = plt.figure(figsize=(9, 9))
    
    for i, name in enumerate(foot_names):
        sequences = combined_sequences[name]
        for seq in sequences:
            # plt.plot(seq['x'], seq['y'], color=colors[i], alpha=0.1, linewidth=0.5)
            plt.scatter(seq['x'], seq['y'],color=colors[i], alpha=0.1, s=1)
        
        if sequences:
            all_x = np.array([seq['x'] for seq in sequences])
            all_y = np.array([seq['y'] for seq in sequences])
            mean_x = np.mean(all_x, axis=0)
            mean_y = np.mean(all_y, axis=0)
            plt.plot(mean_x, mean_y, color=colors[i], linewidth=2, label=name)
    
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    plt.title('All Bouts: Individual P-V-P Trajectories')
    plt.legend()
    plt.grid(False)
    # plt.axis('equal')
    plt.show()
    
    stats_text = "All Bouts Combined Statistics:\n\n"
    total_sequences = 0
    
    for name in foot_names:
        sequences = combined_sequences[name]
        stats_text += f"{name}: {len(sequences)} sequences\n"
        total_sequences += len(sequences)
        
        if sequences:
            durations = [seq['end_time'] - seq['start_time'] for seq in sequences]
            stats_text += f"  Duration: {np.mean(durations):.3f} ± {np.std(durations):.3f} s\n"
            
            distances = []
            for seq in sequences:
                dx = np.diff(seq['x'])
                dy = np.diff(seq['y'])
                distance = np.sum(np.sqrt(dx**2 + dy**2))
                distances.append(distance)
            
            stats_text += f"  Distance: {np.mean(distances):.2f} ± {np.std(distances):.2f} cm\n\n"
    
    stats_text += f"Total sequences across all bouts: {total_sequences}\n"
    stats_text += f"Total bouts analyzed: {len(all_pvp_sequences)}"
    
    print(stats_text)
    
    return fig

print(f"\n{'='*60}")
print(f"Peak-Valley-Peak Trajectory Analysis for {len(bouts)} Movement Bouts")
print(f"{'='*60}\n")

low_pass_x_signals = [low_pass_ffl_x, low_pass_ffr_x, low_pass_fhl_x, low_pass_fhr_x]
low_pass_y_signals = [low_pass_ffl_y, low_pass_ffr_y, low_pass_fhl_y, low_pass_fhr_y]
peaks_list = [peaksffl_x_low_pass, peaksffr_x_low_pass, peaksfhl_x_low_pass, peaksfhr_x_low_pass]
valleys_list = [valleysffl_x_low_pass, valleysffr_x_low_pass, valleysfhl_x_low_pass, valleysfhr_x_low_pass]

# all_bouts_pvp_sequences = []

# for i, bout in enumerate(bouts):
#     print(f"Analyzing trajectories for Bout {i + 1}...")
    
#     pvp_sequences, bout_peaks_valleys = analyze_limb_trajectories_per_bout(
#         i, bout, time_series, low_pass_x_signals, low_pass_y_signals,
#         peaks_list, valleys_list
#     )
    
#     if pvp_sequences is not None:
#         all_bouts_pvp_sequences.append(pvp_sequences)
        
#         fig = plot_bout_trajectory_analysis(i, bout, pvp_sequences)
#         if fig is not None:
#             plt.show()
    
#     print(f"Bout {i + 1} trajectory analysis completed")
#     print("-" * 50)

# if all_bouts_pvp_sequences:
#     print("\nGenerating combined analysis for all bouts...")
#     fig_all = plot_all_bouts_trajectory_analysis(all_bouts_pvp_sequences)
#     if fig_all is not None:
#         plt.show()
#     print("Combined analysis completed!")
# else:
#     print("No valid trajectory data found for analysis.")