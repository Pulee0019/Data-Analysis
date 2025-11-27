import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
from logger import log_message

def plot_raw_data(animal_data=None):
    """Plot raw fiber data"""
    try:
        if animal_data:
            fiber_data = animal_data.get('fiber_data_trimmed')
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            channel_data = animal_data.get('channel_data', {})
        else:
            fiber_data = globals().get('fiber_data_trimmed')
            channels = globals().get('channels', {})
            active_channels = globals().get('active_channels', [])
            channel_data = globals().get('channel_data', {})
        
        if fiber_data is None or not active_channels:
            log_message("Please load and crop fiber data and select channels first", "WARNING")
            return
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        time_col = channels['time']
        time_data = fiber_data[time_col]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(active_channels)))
        for i, channel_num in enumerate(active_channels):
            if channel_num in channel_data:
                for wavelength, col_name in channel_data[channel_num].items():
                    if col_name and col_name in fiber_data.columns:
                        color = colors[i]
                        alpha = 1.0 if wavelength == "470" else 0.6
                        ax.plot(time_data, fiber_data[col_name], color=color, alpha=alpha, 
                               label=f'CH{channel_num} {wavelength}nm')
        
        ax.set_title("Raw Fiber Photometry Data")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Fluorescence")
        ax.legend()
        ax.grid(False)
        
        plt.show()
        
    except Exception as e:
        log_message(f"Failed to plot raw data: {str(e)}", "ERROR")

def smooth_data(animal_data=None, window_size=11, poly_order=3, target_signal="470"):
    """Apply smoothing to fiber data - supports combined wavelengths"""
    try:
        if animal_data is not None and not isinstance(animal_data, dict):
            log_message(f"Invalid animal_data type: {type(animal_data)}", "ERROR")
            return False
            
        if animal_data:
            fiber_data = animal_data.get('fiber_data_trimmed')
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            channel_data = animal_data.get('channel_data', {})
            
            if fiber_data is None:
                log_message("No fiber data available", "WARNING")
                return False
                
            if 'preprocessed_data' not in animal_data:
                animal_data['preprocessed_data'] = fiber_data.copy()
            preprocessed_data = animal_data['preprocessed_data']
        else:
            fiber_data = globals().get('fiber_data_trimmed')
            channels = globals().get('channels', {})
            active_channels = globals().get('active_channels', [])
            channel_data = globals().get('channel_data', {})
            
            if fiber_data is None:
                log_message("No fiber data available", "WARNING")
                return False
                
            if not hasattr(globals(), 'preprocessed_data') or globals().get('preprocessed_data') is None:
                globals()['preprocessed_data'] = fiber_data.copy()
            preprocessed_data = globals()['preprocessed_data']
        
        if not hasattr(preprocessed_data, 'columns'):
            log_message("Preprocessed data is not a valid DataFrame", "ERROR")
            return False
            
        if fiber_data is None or not active_channels:
            log_message("Please load and crop fiber data and select channels first", "WARNING")
            return False
        
        if not isinstance(active_channels, list):
            active_channels = [active_channels] if active_channels else []
        
        # Parse target signal
        target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
        
        for channel_num in active_channels:
            if channel_num in channel_data:
                # Process each wavelength separately
                for wavelength in target_wavelengths:
                    target_col = channel_data[channel_num].get(wavelength)
                    if target_col and target_col in preprocessed_data.columns:
                        smoothed_col = f"CH{channel_num}_{wavelength}_smoothed"
                        preprocessed_data[smoothed_col] = savgol_filter(
                            preprocessed_data[target_col], window_size, poly_order)
        
        if animal_data:
            animal_data['preprocessed_data'] = preprocessed_data
        else:
            globals()['preprocessed_data'] = preprocessed_data
        
        return True
        
    except Exception as e:
        log_message(f"Smoothing failed: {str(e)}", "ERROR")
        return False
    
def baseline_correction(animal_data=None, model_type="Polynomial", target_signal="470"):
    """Apply baseline correction to fiber data - supports combined wavelengths"""
    try:
        if animal_data:
            if 'preprocessed_data' not in animal_data:
                log_message("Please apply smoothing first", "WARNING")
                return False
            preprocessed_data = animal_data['preprocessed_data']
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            channel_data = animal_data.get('channel_data', {})
        else:
            if not hasattr(globals(), 'preprocessed_data') or globals().get('preprocessed_data') is None:
                log_message("Please apply smoothing first", "WARNING")
                return False
            preprocessed_data = globals()['preprocessed_data']
            channels = globals().get('channels', {})
            active_channels = globals().get('active_channels', [])
            channel_data = globals().get('channel_data', {})
        
        if not isinstance(active_channels, list):
            active_channels = [active_channels] if active_channels else []
        
        time_col = channels['time']
        time_data = preprocessed_data[time_col]
        
        # Parse target signal
        target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
        
        for channel_num in active_channels:
            if channel_num in channel_data:
                # Process each wavelength separately
                for wavelength in target_wavelengths:
                    target_col = channel_data[channel_num].get(wavelength)
                    if not target_col or target_col not in preprocessed_data.columns:
                        continue
                    
                    smoothed_col = f"CH{channel_num}_{wavelength}_smoothed"
                    if smoothed_col in preprocessed_data.columns:
                        signal_col = smoothed_col
                    else:
                        signal_col = target_col
                    
                    signal_data = preprocessed_data[signal_col].values
                    
                    if model_type.lower() == "exponential":
                        def exp_model(t, a, b, c):
                            return a * np.exp(-b * t) + c
                        
                        p0 = [
                            np.max(signal_data) - np.min(signal_data),
                            0.01,
                            np.min(signal_data)
                        ]
                        
                        try:
                            params, _ = curve_fit(exp_model, time_data, signal_data, p0=p0, maxfev=5000)
                            baseline_pred = exp_model(time_data, *params)
                        except Exception as e:
                            log_message(f"Exponential fit failed: {str(e)}, using polynomial instead", "INFO")
                            X = time_data.values.reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, signal_data)
                            baseline_pred = model.predict(X)
                    else:
                        X = time_data.values.reshape(-1, 1)
                        model = LinearRegression()
                        model.fit(X, signal_data)
                        baseline_pred = model.predict(X)
                    
                    baseline_corrected_col = f"CH{channel_num}_{wavelength}_baseline_corrected"
                    preprocessed_data[baseline_corrected_col] = signal_data - baseline_pred
                    preprocessed_data[f"CH{channel_num}_{wavelength}_baseline_pred"] = baseline_pred
        
        if animal_data:
            animal_data['preprocessed_data'] = preprocessed_data
        else:
            globals()['preprocessed_data'] = preprocessed_data
        
        return True
        
    except Exception as e:
        log_message(f"Baseline correction failed: {str(e)}", "ERROR")
        return False

def motion_correction(animal_data=None, target_signal="470", reference_signal="410"):
    """Apply motion correction to fiber data - supports combined wavelengths"""
    try:
        if animal_data:
            if 'preprocessed_data' not in animal_data:
                log_message("Please apply smoothing first", "WARNING")
                return False
            preprocessed_data = animal_data['preprocessed_data']
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            channel_data = animal_data.get('channel_data', {})
        else:
            if not hasattr(globals(), 'preprocessed_data') or globals().get('preprocessed_data') is None:
                log_message("Please apply smoothing first", "WARNING")
                return False
            preprocessed_data = globals()['preprocessed_data']
            channels = globals().get('channels', {})
            active_channels = globals().get('active_channels', [])
            channel_data = globals().get('channel_data', {})
        
        if not isinstance(active_channels, list):
            active_channels = [active_channels] if active_channels else []
        
        if reference_signal != "410":
            log_message("Motion correction requires 410nm as reference signal", "WARNING")
            return False
        
        # Parse target signal
        target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
        
        for channel_num in active_channels:
            if channel_num in channel_data:
                ref_col = channel_data[channel_num].get('410')
                if not ref_col or ref_col not in preprocessed_data.columns:
                    log_message(f"No 410nm data for channel CH{channel_num}", "INFO")
                    continue
                
                # Process each wavelength separately
                for wavelength in target_wavelengths:
                    target_col = channel_data[channel_num].get(wavelength)
                    if not target_col or target_col not in preprocessed_data.columns:
                        continue
                    
                    smoothed_col = f"CH{channel_num}_{wavelength}_smoothed"
                    if smoothed_col in preprocessed_data.columns:
                        signal_col = smoothed_col
                    else:
                        signal_col = target_col
                    
                    signal_data = preprocessed_data[signal_col]
                    ref_data = preprocessed_data[ref_col]
                    
                    X = ref_data.values.reshape(-1, 1)
                    y = signal_data.values
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    predicted_signal = model.predict(X)
                    
                    motion_corrected_col = f"CH{channel_num}_{wavelength}_motion_corrected"
                    preprocessed_data[motion_corrected_col] = signal_data - predicted_signal
                    preprocessed_data[f"CH{channel_num}_{wavelength}_fitted_ref"] = predicted_signal
        
        if animal_data:
            animal_data['preprocessed_data'] = preprocessed_data
        else:
            globals()['preprocessed_data'] = preprocessed_data
        
        return True
        
    except Exception as e:
        log_message(f"Motion correction failed: {str(e)}", "ERROR")
        return False

def apply_preprocessing(animal_data=None, target_signal="470", reference_signal="410", 
                       baseline_period=(0, 60), apply_smooth=False, window_size=11, 
                       poly_order=3, apply_baseline=False, baseline_model="Polynomial", 
                       apply_motion=False):
    """Apply all selected preprocessing steps"""
    try:
        apply_smooth = bool(apply_smooth)
        apply_baseline = bool(apply_baseline)
        apply_motion = bool(apply_motion)
        
        if apply_smooth:
            if not smooth_data(animal_data, window_size, poly_order, target_signal):
                return False
        
        if apply_baseline:
            if not baseline_correction(animal_data, baseline_model, target_signal):
                return False
        
        if apply_motion and reference_signal == "410":
            if not motion_correction(animal_data, target_signal, reference_signal):
                return False

        return True
        
    except Exception as e:
        log_message(f"Preprocessing failed: {str(e)}", "ERROR")
        return False

def calculate_dff(animal_data=None, target_signal="470", reference_signal="410", 
                          baseline_period=(0, 60), apply_baseline=False):
    """Calculate and plot ΔF/F - supports combined wavelengths"""
    try:
        if animal_data:
            if 'preprocessed_data' not in animal_data:
                log_message("Please preprocess data first", "WARNING")
                return
            preprocessed_data = animal_data['preprocessed_data']
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            channel_data = animal_data.get('channel_data', {})
        else:
            if not hasattr(globals(), 'preprocessed_data') or globals().get('preprocessed_data') is None:
                log_message("Please preprocess data first", "WARNING")
                return
            preprocessed_data = globals()['preprocessed_data']
            channels = globals().get('channels', {})
            active_channels = globals().get('active_channels', [])
            channel_data = globals().get('channel_data', {})
        
        if not active_channels:
            log_message("No active channels selected", "WARNING")
            return
            
        time_col = channels['time']
        time_data = preprocessed_data[time_col]
        
        baseline_mask = (time_data >= baseline_period[0]) & (time_data <= baseline_period[1])
        
        if not baseline_mask.any():
            log_message("No data in baseline period", "ERROR")
            return
        
        dff_data_dict = {}
        
        # Parse target signal
        target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
        
        for channel_num in active_channels:
            if channel_num in channel_data:
                # Calculate dFF for each wavelength separately
                for wavelength in target_wavelengths:
                    target_col = channel_data[channel_num].get(wavelength)
                    if not target_col or target_col not in preprocessed_data.columns:
                        continue
                    
                    smoothed_col = f"CH{channel_num}_{wavelength}_smoothed"
                    if smoothed_col in preprocessed_data.columns:
                        raw_col = smoothed_col
                    else:
                        raw_col = target_col
                    
                    raw_target = preprocessed_data[raw_col]
                    
                    if reference_signal == "410" and apply_baseline:
                        motion_corrected_col = f"CH{channel_num}_{wavelength}_motion_corrected"
                        if motion_corrected_col in preprocessed_data.columns:
                            dff_data = preprocessed_data[motion_corrected_col] / np.median(raw_target)
                        else:
                            log_message("Please apply motion correction first", "ERROR")
                            return
                    elif reference_signal == "410" and not apply_baseline:
                        fitted_ref_col = f"CH{channel_num}_{wavelength}_fitted_ref"
                        if fitted_ref_col in preprocessed_data.columns:
                            denominator = preprocessed_data[fitted_ref_col]
                            denominator = denominator.replace(0, np.finfo(float).eps)
                            dff_data = (raw_target - preprocessed_data[fitted_ref_col]) / denominator
                        else:
                            log_message("Please apply motion correction first", "ERROR")
                            return
                    elif reference_signal == "baseline" and apply_baseline:
                        baseline_pred_col = f"CH{channel_num}_{wavelength}_baseline_pred"
                        if baseline_pred_col in preprocessed_data.columns:
                            baseline_median = np.median(raw_target[baseline_mask])
                            if baseline_median == 0:
                                baseline_median = np.finfo(float).eps
                            dff_data = (raw_target - preprocessed_data[baseline_pred_col]) / baseline_median
                        else:
                            log_message("Please apply baseline correction first", "ERROR")
                            return
                    elif reference_signal == "baseline" and not apply_baseline:
                        baseline_median = np.median(raw_target[baseline_mask])
                        if baseline_median == 0:
                            baseline_median = np.finfo(float).eps
                        dff_data = (raw_target - baseline_median) / baseline_median
                    
                    dff_col = f"CH{channel_num}_{wavelength}_dff"
                    preprocessed_data[dff_col] = dff_data
                    
                    # Store with wavelength identifier
                    key = f"{channel_num}_{wavelength}"
                    dff_data_dict[key] = dff_data
        
        if animal_data:
            animal_data['preprocessed_data'] = preprocessed_data
            animal_data['dff_data'] = dff_data_dict
            animal_data['target_signal'] = target_signal  # Store target signal info
        else:
            globals()['preprocessed_data'] = preprocessed_data
            globals()['dff_data'] = dff_data_dict
        
    except Exception as e:
        log_message(f"ΔF/F calculation failed: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()

def calculate_zscore(animal_data=None, target_signal="470", reference_signal="410", 
                             baseline_period=(0, 60), apply_baseline=False):
    """Calculate and plot Z-score - supports combined wavelengths"""
    try:
        if animal_data:
            if 'preprocessed_data' not in animal_data or animal_data.get('preprocessed_data') is None:
                log_message("Please calculate ΔF/F first", "WARNING")
                return None
            preprocessed_data = animal_data['preprocessed_data']
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            channel_data = animal_data.get('channel_data', {})
        else:
            if not hasattr(globals(), 'preprocessed_data') or globals().get('preprocessed_data') is None:
                log_message("Please calculate ΔF/F first", "WARNING")
                return None
            preprocessed_data = globals()['preprocessed_data']
            channels = globals().get('channels', {})
            active_channels = globals().get('active_channels', [])
            channel_data = globals().get('channel_data', {})
        
        if not active_channels:
            log_message("No active channels selected", "WARNING")
            return None
        
        time_col = channels.get('time')
        if time_col is None or time_col not in preprocessed_data.columns:
            log_message("Time column not found in preprocessed data", "ERROR")
            return None
        
        time_data = preprocessed_data[time_col]
        
        baseline_mask = (time_data >= baseline_period[0]) & (time_data <= baseline_period[1])
        
        if not any(baseline_mask):
            log_message("No data in baseline period", "ERROR")
            return None
        
        zscore_data_dict = {}
        
        # Parse target signal
        target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
        
        for channel_num in active_channels:
            # Calculate z-score for each wavelength separately
            for wavelength in target_wavelengths:
                dff_col = f"CH{channel_num}_{wavelength}_dff"
                if dff_col not in preprocessed_data.columns:
                    continue
                
                dff_data = preprocessed_data[dff_col]
                baseline_dff = dff_data[baseline_mask]
                
                if len(baseline_dff) < 2:
                    continue
            
                mean_dff = np.mean(baseline_dff)
                std_dff = np.std(baseline_dff)
                
                if std_dff == 0:
                    zscore_data = np.zeros_like(dff_data)
                else:
                    zscore_data = (dff_data - mean_dff) / std_dff
                
                zscore_col = f"CH{channel_num}_{wavelength}_zscore"
                preprocessed_data[zscore_col] = zscore_data
                
                # Store with wavelength identifier
                key = f"{channel_num}_{wavelength}"
                zscore_data_dict[key] = zscore_data
        
        if animal_data:
            animal_data['preprocessed_data'] = preprocessed_data
            animal_data['zscore_data'] = zscore_data_dict
        else:
            globals()['preprocessed_data'] = preprocessed_data
            globals()['zscore_data'] = zscore_data_dict

        return zscore_data_dict
        
    except Exception as e:
        log_message(f"Z-score calculation failed: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return None