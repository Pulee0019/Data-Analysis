import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import os
import numpy as np
import pandas as pd
from datetime import datetime
from logger import log_message

class DrugAnalysis:
    def __init__(self, root, multi_animal_data=None, current_animal_index=0):
        self.root = root
        self.multi_animal_data = multi_animal_data
        self.current_animal_index = current_animal_index
        
        # Colors for different days
        self.day_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                          '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#95a5a6']
        
        # For multi-animal analysis
        self.analysis_window = None
        self.table_data = {}
        self.row_headers = {}
        self.col_headers = {}
        self.used_animals = set()
        self.num_rows = 1
        self.num_cols = 6
        
    def single_animal_drug_analysis(self):
        """Single animal drug analysis entry point"""
        if not self.multi_animal_data:
            log_message("No animal data available", "ERROR")
            return
        
        if self.current_animal_index >= len(self.multi_animal_data):
            log_message("No animal selected", "ERROR")
            return
        
        animal_data = self.multi_animal_data[self.current_animal_index]
        
        # Check for drug events
        if 'fiber_data' not in animal_data and 'fiber_data_trimmed' not in animal_data:
            log_message("No fiber data available", "ERROR")
            return
        
        fiber_data = animal_data.get('fiber_data_trimmed')
        if fiber_data is None or fiber_data.empty:
            fiber_data = animal_data.get('fiber_data')
        channels = animal_data.get('channels', {})
        events_col = channels.get('events')
        
        if events_col is None or events_col not in fiber_data.columns:
            log_message("Events column not found", "ERROR")
            return
        
        # Find Drug events (Event1)
        drug_events = fiber_data[fiber_data[events_col].str.contains('Event1', na=False)]
        
        if len(drug_events) == 0:
            log_message("No drug events (Event1) found in fiber data", "WARNING")
            return
        
        time_col = channels['time']
        drug_start_time = drug_events[time_col].iloc[0]
        
        animal_data['drug_start_time'] = drug_start_time
        
        log_message(f"Drug event detected: start at {drug_start_time:.2f}s", "INFO")
        
        # Show parameter window
        self._show_single_animal_param_window(animal_data)
    
    def _show_single_animal_param_window(self, animal_data):
        """Show parameter window for single animal analysis"""
        param_window = tk.Toplevel(self.root)
        param_window.title("Drug Analysis Parameters - Single Animal")
        param_window.geometry("450x450")
        param_window.configure(bg='#f8f8f8')
        param_window.transient(self.root)
        param_window.grab_set()
        
        # Title
        title_label = tk.Label(param_window, text="Drug Analysis Settings", 
                            font=("Microsoft YaHei", 12, "bold"), bg="#f8f8f8", fg="#2c3e50")
        title_label.pack(pady=15)
        
        # Event type selection
        event_frame = tk.LabelFrame(param_window, text="Analysis Type", 
                                   font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        event_frame.pack(fill=tk.X, padx=20, pady=10)
        
        event_type_var = tk.StringVar(value="drug_induced")
        tk.Radiobutton(event_frame, text="Drug Induced", variable=event_type_var, 
                      value="drug_induced", bg="#f8f8f8", 
                      font=("Microsoft YaHei", 9)).pack(anchor=tk.W, padx=10, pady=5)
        tk.Radiobutton(event_frame, text="Running Induced", variable=event_type_var, 
                      value="running_induced", bg="#f8f8f8",
                      font=("Microsoft YaHei", 9)).pack(anchor=tk.W, padx=10, pady=5)
        
        # Time window settings
        time_frame = tk.LabelFrame(param_window, text="Time Window Settings", 
                                  font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        time_frame.pack(fill=tk.X, padx=20, pady=10)
        
        pre_frame = tk.Frame(time_frame, bg="#f8f8f8")
        pre_frame.pack(pady=5)
        tk.Label(pre_frame, text="Pre Event Window (s):", bg="#f8f8f8", 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        pre_time_var = tk.StringVar(value="600")
        tk.Entry(pre_frame, textvariable=pre_time_var, width=10, 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=5)
        
        post_frame = tk.Frame(time_frame, bg="#f8f8f8")
        post_frame.pack(pady=5)
        tk.Label(post_frame, text="Post Event Window (s):", bg="#f8f8f8", 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        post_time_var = tk.StringVar(value="1200")
        tk.Entry(post_frame, textvariable=post_time_var, width=10, 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=5)
        
        # Running type selection (only for running_induced)
        running_frame = tk.LabelFrame(param_window, text="Running Analysis Settings", 
                                     font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        running_frame.pack(fill=tk.X, padx=20, pady=10)
        
        running_type_label = tk.Label(running_frame, text="Running Type:", bg="#f8f8f8", 
                                      font=("Microsoft YaHei", 8))
        running_type_label.pack(anchor=tk.W, padx=10, pady=(5,2))
        
        running_types = ['general_onsets', 'jerks', 'locomotion_onsets', 'reset_onsets',
                        'general_offsets', 'locomotion_offsets', 'reset_offsets']
        running_type_var = tk.StringVar(value=running_types[0])
        running_type_combo = ttk.Combobox(running_frame, textvariable=running_type_var,
                                         values=running_types, state="readonly",
                                         font=("Microsoft YaHei", 8))
        running_type_combo.pack(padx=10, pady=5, fill=tk.X)
        
        drug_name_label = tk.Label(running_frame, text="Drug Name:", bg="#f8f8f8", 
                                   font=("Microsoft YaHei", 8))
        drug_name_label.pack(anchor=tk.W, padx=10, pady=(5,2))
        
        drug_name_var = tk.StringVar(value="Drug")
        drug_name_entry = tk.Entry(running_frame, textvariable=drug_name_var,
                                   font=("Microsoft YaHei", 8))
        drug_name_entry.pack(padx=10, pady=5, fill=tk.X)
        
        # Initially hide running settings
        running_frame.pack_forget()
        
        def update_ui(*args):
            if event_type_var.get() == "running_induced":
                running_frame.pack(fill=tk.X, padx=20, pady=10)
            else:
                running_frame.pack_forget()
        
        event_type_var.trace('w', update_ui)
        
        # Export option
        export_frame = tk.LabelFrame(param_window, text="Export Options", 
                                    font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        export_frame.pack(fill=tk.X, padx=20, pady=10)
        
        export_var = tk.BooleanVar(value=False)
        tk.Checkbutton(export_frame, text="Export statistic results to CSV", 
                      variable=export_var, bg="#f8f8f8",
                      font=("Microsoft YaHei", 8)).pack(anchor=tk.W, padx=10, pady=5)
        
        def run_analysis():
            try:
                pre_time = float(pre_time_var.get())
                post_time = float(post_time_var.get())
                event_type = event_type_var.get()
                
                if pre_time <= 0 or post_time <= 0:
                    log_message("Time must be positive numbers", "WARNING")
                    return
                
                if event_type == "running_induced":
                    running_type = running_type_var.get()
                    drug_name = drug_name_var.get().strip()
                    if not drug_name:
                        log_message("Please enter drug name", "WARNING")
                        return
                    param_window.destroy()
                    self._analyze_running_induced_single(animal_data, pre_time, post_time,
                                                         running_type, drug_name, export_var.get())
                else:
                    param_window.destroy()
                    self._analyze_drug_induced_single(animal_data, pre_time, post_time,
                                                      export_var.get())
                    
            except ValueError:
                log_message("Please enter valid time values", "WARNING")
        
        # Buttons
        button_frame = tk.Frame(param_window, bg="#f8f8f8")
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="Start Analysis", command=run_analysis,
                 bg="#3498db", fg="white", font=("Microsoft YaHei", 9, "bold"),
                 relief=tk.FLAT, padx=15, pady=5).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Cancel", command=param_window.destroy,
                 bg="#95a5a6", fg="white", font=("Microsoft YaHei", 9),
                 relief=tk.FLAT, padx=15, pady=5).pack(side=tk.LEFT, padx=10)
    
    def _analyze_drug_induced_single(self, animal_data, pre_time, post_time, export_statistics):
        """Analyze drug-induced effects (single animal)"""
        animal_id = animal_data.get('animal_id', 'Unknown')
        drug_start_time = animal_data['drug_start_time']
        
        # Get fiber data
        channels = animal_data.get('channels', {})
        time_col = channels['time']
        preprocessed_data = animal_data.get('preprocessed_data')
        
        if preprocessed_data is None:
            log_message("No preprocessed fiber data available", "ERROR")
            return
        
        fiber_timestamps = preprocessed_data[time_col].values
        
        # Get dFF data (z-score will be calculated from dFF)
        dff_data = animal_data.get('dff_data', {})
        
        if not dff_data:
            log_message("No dFF data available. Please calculate dFF first.", "ERROR")
            return
        
        # Get target signal
        target_signal = animal_data.get('target_signal', '470')
        target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
        
        # Get active channels
        active_channels = animal_data.get('active_channels', [])
        
        # Extract data around drug event
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        
        # Calculate episodes
        all_dff_episodes = {}
        all_zscore_episodes = {}
        
        for wavelength in target_wavelengths:
            all_dff_episodes[wavelength] = []
            all_zscore_episodes[wavelength] = []
        
        for channel in active_channels:
            for wavelength in target_wavelengths:
                # dFF data
                dff_key = f"{channel}_{wavelength}"
                if dff_key in dff_data:
                    data = dff_data[dff_key]
                    if isinstance(data, pd.Series):
                        data = data.values
                    
                    start_idx = np.argmin(np.abs(fiber_timestamps - (drug_start_time - pre_time)))
                    end_idx = np.argmin(np.abs(fiber_timestamps - (drug_start_time + post_time)))
                    
                    if end_idx > start_idx:
                        episode_data = data[start_idx:end_idx]
                        episode_times = fiber_timestamps[start_idx:end_idx] - drug_start_time
                        
                        if len(episode_times) > 1:
                            interp_data = np.interp(time_array, episode_times, episode_data)
                            all_dff_episodes[wavelength].append(interp_data)
                            
                            # Calculate z-score using pre_time window as baseline
                            baseline_mask = (episode_times >= -pre_time) & (episode_times <= 0)
                            baseline_data = episode_data[baseline_mask]
                            
                            if len(baseline_data) > 0:
                                mean_baseline = np.nanmean(baseline_data)
                                std_baseline = np.nanstd(baseline_data)
                                
                                if std_baseline == 0:
                                    std_baseline = 1e-10
                                
                                # Calculate z-score for the entire episode
                                zscore_episode = (episode_data - mean_baseline) / std_baseline
                                interp_zscore = np.interp(time_array, episode_times, zscore_episode)
                                all_zscore_episodes[wavelength].append(interp_zscore)
        
        # Export statistics if requested
        if export_statistics:
            self._export_drug_induced_statistics(animal_id, pre_time, post_time,
                                                 active_channels, target_wavelengths,
                                                 all_dff_episodes, all_zscore_episodes, time_array)
        
        # Plot results
        self._plot_drug_induced_results(animal_id, time_array, all_dff_episodes, 
                                        all_zscore_episodes, target_wavelengths, active_channels)
    
    def _analyze_running_induced_single(self, animal_data, pre_time, post_time, 
                                       running_type, drug_name, export_statistics):
        """Analyze running-induced effects around drug administration (single animal)"""
        animal_id = animal_data.get('animal_id', 'Unknown')
        drug_start_time = animal_data['drug_start_time']
        
        # Check for running behavior data
        if 'treadmill_behaviors' not in animal_data:
            log_message("No running behavior data. Please run running analysis first.", "ERROR")
            return
        
        treadmill_behaviors = animal_data['treadmill_behaviors']
        if running_type not in treadmill_behaviors:
            log_message(f"No {running_type} data available", "ERROR")
            return
        
        events = treadmill_behaviors[running_type]
        if len(events) == 0:
            log_message(f"No {running_type} events found", "WARNING")
            return
        
        # Split events into pre-drug and post-drug
        pre_drug_events = [e for e in events if e < drug_start_time]
        post_drug_events = [e for e in events if e >= drug_start_time]
        
        log_message(f"Found {len(pre_drug_events)} pre-drug and {len(post_drug_events)} post-drug {running_type} events")
        
        # Get data
        channels = animal_data.get('channels', {})
        time_col = channels['time']
        preprocessed_data = animal_data.get('preprocessed_data')
        fiber_timestamps = preprocessed_data[time_col].values
        
        # Get running data
        ast2_data = animal_data.get('ast2_data_adjusted')
        running_timestamps = ast2_data['data']['timestamps']
        processed_data = animal_data.get('running_processed_data')
        running_speed = processed_data['filtered_speed'] if processed_data else ast2_data['data']['speed']
        
        # Get fiber data
        dff_data = animal_data.get('dff_data', {})
        target_signal = animal_data.get('target_signal', '470')
        target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
        active_channels = animal_data.get('active_channels', [])
        
        # Analyze pre-drug and post-drug separately
        pre_drug_results = self._calculate_running_episodes(
            pre_drug_events, running_timestamps, running_speed,
            fiber_timestamps, dff_data,
            active_channels, target_wavelengths,
            pre_time, post_time, running_type)
        
        post_drug_results = self._calculate_running_episodes(
            post_drug_events, running_timestamps, running_speed,
            fiber_timestamps, dff_data,
            active_channels, target_wavelengths,
            pre_time, post_time, running_type)
        
        # Export statistics if requested
        if export_statistics:
            self._export_running_induced_statistics(
                animal_id, running_type, drug_name, pre_time, post_time,
                pre_drug_results, post_drug_results, target_wavelengths, active_channels)
        
        # Plot results
        self._plot_running_induced_results(
            animal_id, running_type, drug_name,
            pre_drug_results, post_drug_results,
            target_wavelengths, active_channels)
    
    def _calculate_running_episodes(self, events, running_timestamps, running_speed,
                                   fiber_timestamps, dff_data,
                                   active_channels, target_wavelengths,
                                   pre_time, post_time, event_type):
        """Calculate episodes around running events"""
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        
        # Running episodes
        running_episodes = []
        for event in events:
            start_idx = np.argmin(np.abs(running_timestamps - (event - pre_time)))
            end_idx = np.argmin(np.abs(running_timestamps - (event + post_time)))
            
            if end_idx > start_idx:
                episode_data = running_speed[start_idx:end_idx]
                episode_times = running_timestamps[start_idx:end_idx] - event
                
                if len(episode_times) > 1:
                    interp_data = np.interp(time_array, episode_times, episode_data)
                    running_episodes.append(interp_data)
        
        # Fiber episodes
        dff_episodes = {}
        zscore_episodes = {}
        
        for wavelength in target_wavelengths:
            dff_episodes[wavelength] = []
            zscore_episodes[wavelength] = []
        
        # Get zscore baseline parameters
        zscore_baseline_type, zscore_baseline_window = self._get_zscore_baseline_params(event_type)
        
        for channel in active_channels:
            for wavelength in target_wavelengths:
                dff_key = f"{channel}_{wavelength}"
                if dff_key in dff_data:
                    data = dff_data[dff_key]
                    if isinstance(data, pd.Series):
                        data = data.values
                    
                    for event in events:
                        # Determine baseline window based on event type
                        if zscore_baseline_type == 'offset':
                            baseline_start = event
                            baseline_end = event + zscore_baseline_window
                        else:
                            baseline_start = event - zscore_baseline_window
                            baseline_end = event
                        
                        baseline_start_idx = np.argmin(np.abs(fiber_timestamps - baseline_start))
                        baseline_end_idx = np.argmin(np.abs(fiber_timestamps - baseline_end))
                        
                        if baseline_end_idx > baseline_start_idx:
                            baseline_data = data[baseline_start_idx:baseline_end_idx]
                            mean_dff = np.nanmean(baseline_data)
                            std_dff = np.nanstd(baseline_data)
                            
                            if std_dff == 0:
                                std_dff = 1e-10
                            
                            # Extract plotting window
                            start_idx = np.argmin(np.abs(fiber_timestamps - (event - pre_time)))
                            end_idx = np.argmin(np.abs(fiber_timestamps - (event + post_time)))
                            
                            if end_idx > start_idx:
                                episode_data = data[start_idx:end_idx]
                                episode_times = fiber_timestamps[start_idx:end_idx] - event
                                
                                if len(episode_times) > 1:
                                    # Calculate z-score using fixed baseline window
                                    zscore_episode = (episode_data - mean_dff) / std_dff
                                    interp_zscore = np.interp(time_array, episode_times, zscore_episode)
                                    zscore_episodes[wavelength].append(interp_zscore)
                                    
                                    # Also store dFF data
                                    interp_dff = np.interp(time_array, episode_times, episode_data)
                                    dff_episodes[wavelength].append(interp_dff)
        
        return {
            'time': time_array,
            'running': np.array(running_episodes) if running_episodes else np.array([]),
            'dff': dff_episodes,
            'zscore': zscore_episodes
        }
    
    def _get_zscore_baseline_params(self, event_type):
        """Get z-score baseline parameters"""
        if event_type in ['general_onsets', 'reset_onsets']:
            return ('onset', 0.5)
        elif event_type in ['locomotion_onsets', 'jerks']:
            return ('onset', 2.0)
        elif event_type in ['general_offsets', 'reset_offsets']:
            return ('offset', 0.5)
        elif event_type in ['locomotion_offsets']:
            return ('offset', 2.0)
        else:
            return ('onset', 0.5)
    
    def _plot_drug_induced_results(self, animal_id, time_array, dff_episodes, 
                                   zscore_episodes, target_wavelengths, channels):
        """Plot drug-induced analysis results"""
        result_window = tk.Toplevel(self.root)
        channel_label = "+".join([str(c) for c in channels])
        wavelength_label = "+".join(target_wavelengths)
        result_window.title(f"Drug Analysis - {animal_id} - CH{channel_label} - {wavelength_label}nm")
        result_window.state('zoomed')
        result_window.configure(bg='#f8f8f8')
        
        num_wavelengths = len(target_wavelengths)
        num_cols = 2 * num_wavelengths  # dFF + z-score for each wavelength
        
        fig = Figure(figsize=(4 * num_cols, 8), dpi=100)
        
        fiber_colors = ['#008000', "#FF0000", '#FFA500']
        plot_idx = 1
        
        # Row 1: Traces
        for wl_idx, wavelength in enumerate(target_wavelengths):
            color = fiber_colors[wl_idx % len(fiber_colors)]
            
            # dFF trace
            ax_dff = fig.add_subplot(2, num_cols, plot_idx)
            episodes = dff_episodes.get(wavelength, [])
            if episodes:
                episodes_array = np.array(episodes)
                mean_response = np.nanmean(episodes_array, axis=0)
                std_response = np.nanstd(episodes_array, axis=0)
                
                ax_dff.plot(time_array, mean_response, color, linewidth=2, label='Mean')
                ax_dff.fill_between(time_array, mean_response - std_response,
                                   mean_response + std_response, color=color, alpha=0.3)
                ax_dff.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Drug')
                ax_dff.set_xlim(time_array[0], time_array[-1])
                ax_dff.set_xlabel('Time (s)')
                ax_dff.set_ylabel('ΔF/F')
                ax_dff.set_title(f'Fiber ΔF/F {wavelength}nm - CH{channel_label}')
                ax_dff.legend()
                ax_dff.grid(False)
            plot_idx += 1
            
            # Z-score trace
            ax_zscore = fig.add_subplot(2, num_cols, plot_idx)
            episodes = zscore_episodes.get(wavelength, [])
            if episodes:
                episodes_array = np.array(episodes)
                mean_response = np.nanmean(episodes_array, axis=0)
                std_response = np.nanstd(episodes_array, axis=0)
                
                ax_zscore.plot(time_array, mean_response, color, linewidth=2, label='Mean')
                ax_zscore.fill_between(time_array, mean_response - std_response,
                                      mean_response + std_response, color=color, alpha=0.3)
                ax_zscore.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Drug')
                ax_zscore.set_xlim(time_array[0], time_array[-1])
                ax_zscore.set_xlabel('Time (s)')
                ax_zscore.set_ylabel('Z-score')
                ax_zscore.set_title(f'Fiber Z-score {wavelength}nm - CH{channel_label}')
                ax_zscore.legend()
                ax_zscore.grid(False)
            plot_idx += 1
        
        # Row 2: Heatmaps
        for wl_idx, wavelength in enumerate(target_wavelengths):
            # dFF heatmap
            ax_dff_heat = fig.add_subplot(2, num_cols, plot_idx)
            episodes = dff_episodes.get(wavelength, [])
            if episodes:
                episodes_array = np.array(episodes)
                if len(episodes_array) == 1:
                    episodes_array = np.vstack([episodes_array[0], episodes_array[0]])
                    im = ax_dff_heat.imshow(episodes_array, aspect='auto',
                                        extent=[time_array[0], time_array[-1], 2, 1],
                                        cmap='viridis', origin='lower')
                    ax_dff_heat.set_ylabel('Trial')
                else:
                    im = ax_dff_heat.imshow(episodes_array, aspect='auto',
                                        extent=[time_array[0], time_array[-1],
                                                len(episodes), 1],
                                        cmap='viridis', origin='lower')
                    ax_dff_heat.set_ylabel('Trial')
                
                ax_dff_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
                ax_dff_heat.set_xlabel('Time (s)')
                ax_dff_heat.set_title(f'Fiber ΔF/F Heatmap {wavelength}nm')
                
                if len(episodes_array) == 1:
                    
                    norm = colors.Normalize(vmin=episodes_array[0].min(), vmax=episodes_array[0].max())
                    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax_dff_heat, orientation='horizontal')
                    cbar.set_label('ΔF/F')
                else:
                    plt.colorbar(im, ax=ax_dff_heat, label='ΔF/F', orientation='horizontal')
            else:
                ax_dff_heat.text(0.5, 0.5, f'No dFF data for {wavelength}nm',
                                ha='center', va='center', transform=ax_dff_heat.transAxes,
                                fontsize=12, color='#666666')
                ax_dff_heat.set_title(f'Fiber ΔF/F Heatmap {wavelength}nm')
                ax_dff_heat.axis('off')
            plot_idx += 1
            
            # Z-score heatmap
            ax_zscore_heat = fig.add_subplot(2, num_cols, plot_idx)
            episodes = zscore_episodes.get(wavelength, [])
            if episodes:
                episodes_array = np.array(episodes)
                
                if len(episodes_array) == 1:
                    episodes_array = np.vstack([episodes_array[0], episodes_array[0]])
                    im = ax_zscore_heat.imshow(episodes_array, aspect='auto',
                                            extent=[time_array[0], time_array[-1], 2, 1],
                                            cmap='coolwarm', origin='lower')
                    ax_zscore_heat.set_ylabel('Trial')
                else:
                    im = ax_zscore_heat.imshow(episodes_array, aspect='auto',
                                            extent=[time_array[0], time_array[-1],
                                                    len(episodes), 1],
                                            cmap='coolwarm', origin='lower')
                    ax_zscore_heat.set_ylabel('Trial')
                
                ax_zscore_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
                ax_zscore_heat.set_xlabel('Time (s)')
                ax_zscore_heat.set_title(f'Fiber Z-score Heatmap {wavelength}nm')
                
                if len(episodes_array) == 1:
                    norm = colors.Normalize(vmin=episodes_array[0].min(), vmax=episodes_array[0].max())
                    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax_zscore_heat, orientation='horizontal')
                    cbar.set_label('Z-score')
                else:
                    plt.colorbar(im, ax=ax_zscore_heat, label='Z-score', orientation='horizontal')
            else:
                ax_zscore_heat.text(0.5, 0.5, f'No z-score data for {wavelength}nm',
                                ha='center', va='center', transform=ax_zscore_heat.transAxes,
                                fontsize=12, color='#666666')
                ax_zscore_heat.set_title(f'Fiber Z-score Heatmap {wavelength}nm')
                ax_zscore_heat.axis('off')
            plot_idx += 1
        
        fig.tight_layout()
        
        # Add canvas
        canvas_frame = tk.Frame(result_window, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        
        log_message(f"Drug-induced analysis plotted for {animal_id}")
    
    def _plot_running_induced_results(self, animal_id, running_type, drug_name,
                                     pre_drug_results, post_drug_results,
                                     target_wavelengths, channels):
        """Plot running-induced analysis results with pre/post drug comparison"""
        result_window = tk.Toplevel(self.root)
        channel_label = "+".join([str(c) for c in channels])
        wavelength_label = "+".join(target_wavelengths)
        result_window.title(f"Drug-Running Analysis - {animal_id} - {running_type}")
        result_window.state('zoomed')
        result_window.configure(bg='#f8f8f8')
        
        num_wavelengths = len(target_wavelengths)
        num_cols = 1 + 2 * num_wavelengths  # running + dFF + z-score for each wavelength
        
        fig = Figure(figsize=(4 * num_cols, 8), dpi=100)
        
        fiber_colors = ['#008000', "#FF0000", '#FFA500']
        plot_idx = 1
        
        time_array = pre_drug_results['time']
        
        # Row 1: Traces
        # Running trace
        ax_running = fig.add_subplot(2, num_cols, plot_idx)
        
        if len(pre_drug_results['running']) > 0:
            mean_pre = np.nanmean(pre_drug_results['running'], axis=0)
            std_pre = np.nanstd(pre_drug_results['running'], axis=0)
            ax_running.plot(time_array, mean_pre, '#3498db', linewidth=2, label=f'Pre {drug_name}')
            ax_running.fill_between(time_array, mean_pre - std_pre, mean_pre + std_pre,
                                   color='#3498db', alpha=0.3)
        
        if len(post_drug_results['running']) > 0:
            mean_post = np.nanmean(post_drug_results['running'], axis=0)
            std_post = np.nanstd(post_drug_results['running'], axis=0)
            ax_running.plot(time_array, mean_post, '#e74c3c', linewidth=2, label=f'Post {drug_name}')
            ax_running.fill_between(time_array, mean_post - std_post, mean_post + std_post,
                                   color='#e74c3c', alpha=0.3)
        
        ax_running.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Event')
        ax_running.set_xlim(time_array[0], time_array[-1])
        ax_running.set_xlabel('Time (s)')
        ax_running.set_ylabel('Speed (cm/s)')
        ax_running.set_title(f'Running Speed - {running_type}')
        ax_running.legend()
        ax_running.grid(False)
        plot_idx += 1
        
        # Fiber traces
        for wl_idx, wavelength in enumerate(target_wavelengths):
            color = fiber_colors[wl_idx % len(fiber_colors)]
            
            # dFF trace
            ax_dff = fig.add_subplot(2, num_cols, plot_idx)
            
            pre_episodes = pre_drug_results['dff'].get(wavelength, [])
            if pre_episodes:
                episodes_array = np.array(pre_episodes)
                mean_pre = np.nanmean(episodes_array, axis=0)
                std_pre = np.nanstd(episodes_array, axis=0)
                ax_dff.plot(time_array, mean_pre, '#3498db', linewidth=2, label=f'Pre {drug_name}')
                ax_dff.fill_between(time_array, mean_pre - std_pre, mean_pre + std_pre,
                                   color='#3498db', alpha=0.3)
            
            post_episodes = post_drug_results['dff'].get(wavelength, [])
            if post_episodes:
                episodes_array = np.array(post_episodes)
                mean_post = np.nanmean(episodes_array, axis=0)
                std_post = np.nanstd(episodes_array, axis=0)
                ax_dff.plot(time_array, mean_post, '#e74c3c', linewidth=2, label=f'Post {drug_name}')
                ax_dff.fill_between(time_array, mean_post - std_post, mean_post + std_post,
                                   color='#e74c3c', alpha=0.3)
            
            ax_dff.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Event')
            ax_dff.set_xlim(time_array[0], time_array[-1])
            ax_dff.set_xlabel('Time (s)')
            ax_dff.set_ylabel('ΔF/F')
            ax_dff.set_title(f'Fiber ΔF/F {wavelength}nm')
            ax_dff.legend()
            ax_dff.grid(False)
            plot_idx += 1
            
            # Z-score trace
            ax_zscore = fig.add_subplot(2, num_cols, plot_idx)
            
            pre_episodes = pre_drug_results['zscore'].get(wavelength, [])
            if pre_episodes:
                episodes_array = np.array(pre_episodes)
                mean_pre = np.nanmean(episodes_array, axis=0)
                std_pre = np.nanstd(episodes_array, axis=0)
                ax_zscore.plot(time_array, mean_pre, '#3498db', linewidth=2, label=f'Pre {drug_name}')
                ax_zscore.fill_between(time_array, mean_pre - std_pre, mean_pre + std_pre,
                                      color='#3498db', alpha=0.3)
            
            post_episodes = post_drug_results['zscore'].get(wavelength, [])
            if post_episodes:
                episodes_array = np.array(post_episodes)
                mean_post = np.nanmean(episodes_array, axis=0)
                std_post = np.nanstd(episodes_array, axis=0)
                ax_zscore.plot(time_array, mean_post, '#e74c3c', linewidth=2, label=f'Post {drug_name}')
                ax_zscore.fill_between(time_array, mean_post - std_post, mean_post + std_post,
                                      color='#e74c3c', alpha=0.3)
            
            ax_zscore.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Event')
            ax_zscore.set_xlim(time_array[0], time_array[-1])
            ax_zscore.set_xlabel('Time (s)')
            ax_zscore.set_ylabel('Z-score')
            ax_zscore.set_title(f'Fiber Z-score {wavelength}nm')
            ax_zscore.legend()
            ax_zscore.grid(False)
            plot_idx += 1
        
        # Row 2: Heatmaps
        # Running heatmap
        ax_running_heat = fig.add_subplot(2, num_cols, plot_idx)
        
        if len(pre_drug_results['running']) > 0 and len(post_drug_results['running']) > 0:
            combined_running = np.vstack([pre_drug_results['running'], post_drug_results['running']])
            n_pre = len(pre_drug_results['running'])
            
            im = ax_running_heat.imshow(combined_running, aspect='auto',
                                       extent=[time_array[0], time_array[-1],
                                              len(combined_running), 1],
                                       cmap='viridis', origin='lower')
            ax_running_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax_running_heat.axhline(y=n_pre, color='white', linestyle='-', linewidth=2)
            ax_running_heat.set_xlabel('Time (s)')
            ax_running_heat.set_ylabel('Trial')
            ax_running_heat.set_title('Running Speed Heatmap')
            plt.colorbar(im, ax=ax_running_heat, label='Speed (cm/s)', orientation='horizontal')
        plot_idx += 1
        
        # Fiber heatmaps
        for wl_idx, wavelength in enumerate(target_wavelengths):
            # dFF heatmap
            ax_dff_heat = fig.add_subplot(2, num_cols, plot_idx)
            
            pre_episodes = pre_drug_results['dff'].get(wavelength, [])
            post_episodes = post_drug_results['dff'].get(wavelength, [])
            
            if pre_episodes and post_episodes:
                combined_dff = np.vstack([np.array(pre_episodes), np.array(post_episodes)])
                n_pre = len(pre_episodes)
                
                im = ax_dff_heat.imshow(combined_dff, aspect='auto',
                                       extent=[time_array[0], time_array[-1],
                                              len(combined_dff), 1],
                                       cmap='viridis', origin='lower')
                ax_dff_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
                ax_dff_heat.axhline(y=n_pre, color='white', linestyle='-', linewidth=2)
                ax_dff_heat.set_xlabel('Time (s)')
                ax_dff_heat.set_ylabel('Trial')
                ax_dff_heat.set_title(f'Fiber ΔF/F Heatmap {wavelength}nm')
                plt.colorbar(im, ax=ax_dff_heat, label='ΔF/F', orientation='horizontal')
            plot_idx += 1
            
            # Z-score heatmap
            ax_zscore_heat = fig.add_subplot(2, num_cols, plot_idx)
            
            pre_episodes = pre_drug_results['zscore'].get(wavelength, [])
            post_episodes = post_drug_results['zscore'].get(wavelength, [])
            
            if pre_episodes and post_episodes:
                combined_zscore = np.vstack([np.array(pre_episodes), np.array(post_episodes)])
                n_pre = len(pre_episodes)
                
                im = ax_zscore_heat.imshow(combined_zscore, aspect='auto',
                                          extent=[time_array[0], time_array[-1],
                                                 len(combined_zscore), 1],
                                          cmap='coolwarm', origin='lower')
                ax_zscore_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
                ax_zscore_heat.axhline(y=n_pre, color='white', linestyle='-', linewidth=2)
                ax_zscore_heat.set_xlabel('Time (s)')
                ax_zscore_heat.set_ylabel('Trial')
                ax_zscore_heat.set_title(f'Fiber Z-score Heatmap {wavelength}nm')
                plt.colorbar(im, ax=ax_zscore_heat, label='Z-score', orientation='horizontal')
            plot_idx += 1
        
        fig.tight_layout()
        
        # Add canvas
        canvas_frame = tk.Frame(result_window, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        
        log_message(f"Running-induced drug analysis plotted for {animal_id}")
    
    def _export_drug_induced_statistics(self, animal_id, pre_time, post_time,
                                       channels, wavelengths, dff_episodes, zscore_episodes, time_array):
        """Export drug-induced statistics"""
        rows = []
        
        pre_mask = (time_array >= -pre_time) & (time_array <= 0)
        post_mask = (time_array >= 0) & (time_array <= post_time)
        
        for channel in channels:
            for wavelength in wavelengths:
                # dFF statistics
                episodes = dff_episodes.get(wavelength, [])
                for trial_idx, episode_data in enumerate(episodes):
                    pre_data = episode_data[pre_mask]
                    post_data = episode_data[post_mask]
                    
                    rows.append({
                        'animal_id': animal_id,
                        'analysis_type': 'drug_induced',
                        'channel': channel,
                        'wavelength': wavelength,
                        'trial': trial_idx + 1,
                        'pre_min': np.min(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_max': np.max(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_mean': np.mean(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_area': np.trapz(pre_data, time_array[pre_mask]) if len(pre_data) > 0 else np.nan,
                        'post_min': np.min(post_data) if len(post_data) > 0 else np.nan,
                        'post_max': np.max(post_data) if len(post_data) > 0 else np.nan,
                        'post_mean': np.mean(post_data) if len(post_data) > 0 else np.nan,
                        'post_area': np.trapz(post_data, time_array[post_mask]) if len(post_data) > 0 else np.nan,
                        'signal_type': 'fiber_dff'
                    })
                
                # Z-score statistics
                episodes = zscore_episodes.get(wavelength, [])
                for trial_idx, episode_data in enumerate(episodes):
                    pre_data = episode_data[pre_mask]
                    post_data = episode_data[post_mask]
                    
                    rows.append({
                        'animal_id': animal_id,
                        'analysis_type': 'drug_induced',
                        'channel': channel,
                        'wavelength': wavelength,
                        'trial': trial_idx + 1,
                        'pre_min': np.min(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_max': np.max(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_mean': np.mean(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_area': np.trapz(pre_data, time_array[pre_mask]) if len(pre_data) > 0 else np.nan,
                        'post_min': np.min(post_data) if len(post_data) > 0 else np.nan,
                        'post_max': np.max(post_data) if len(post_data) > 0 else np.nan,
                        'post_mean': np.mean(post_data) if len(post_data) > 0 else np.nan,
                        'post_area': np.trapz(post_data, time_array[post_mask]) if len(post_data) > 0 else np.nan,
                        'signal_type': 'fiber_zscore'
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            save_dir = filedialog.askdirectory(title='Select directory to save statistics')
            if save_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"drug_induced_statistics_{timestamp}.csv"
                save_path = os.path.join(save_dir, filename)
                df.to_csv(save_path, index=False)
                log_message(f"Statistics exported to {save_path}")
    
    def _export_running_induced_statistics(self, animal_id, running_type, drug_name,
                                          pre_time, post_time, pre_results, post_results,
                                          wavelengths, channels):
        """Export running-induced statistics"""
        rows = []
        time_array = pre_results['time']
        pre_mask = (time_array >= -pre_time) & (time_array <= 0)
        post_mask = (time_array >= 0) & (time_array <= post_time)
        
        # Pre-drug statistics
        for channel in channels:
            for wavelength in wavelengths:
                # dFF
                episodes = pre_results['dff'].get(wavelength, [])
                for trial_idx, episode_data in enumerate(episodes):
                    pre_data = episode_data[pre_mask]
                    post_data = episode_data[post_mask]
                    
                    rows.append({
                        'animal_id': animal_id,
                        'analysis_type': 'running_induced',
                        'running_type': running_type,
                        'drug_condition': f'pre_{drug_name}',
                        'channel': channel,
                        'wavelength': wavelength,
                        'trial': trial_idx + 1,
                        'pre_min': np.min(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_max': np.max(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_mean': np.mean(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_area': np.trapz(pre_data, time_array[pre_mask]) if len(pre_data) > 0 else np.nan,
                        'post_min': np.min(post_data) if len(post_data) > 0 else np.nan,
                        'post_max': np.max(post_data) if len(post_data) > 0 else np.nan,
                        'post_mean': np.mean(post_data) if len(post_data) > 0 else np.nan,
                        'post_area': np.trapz(post_data, time_array[post_mask]) if len(post_data) > 0 else np.nan,
                        'signal_type': 'fiber_dff'
                    })
        
        # Post-drug statistics (similar structure)
        for channel in channels:
            for wavelength in wavelengths:
                episodes = post_results['dff'].get(wavelength, [])
                for trial_idx, episode_data in enumerate(episodes):
                    pre_data = episode_data[pre_mask]
                    post_data = episode_data[post_mask]
                    
                    rows.append({
                        'animal_id': animal_id,
                        'analysis_type': 'running_induced',
                        'running_type': running_type,
                        'drug_condition': f'post_{drug_name}',
                        'channel': channel,
                        'wavelength': wavelength,
                        'trial': trial_idx + 1,
                        'pre_min': np.min(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_max': np.max(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_mean': np.mean(pre_data) if len(pre_data) > 0 else np.nan,
                        'pre_area': np.trapz(pre_data, time_array[pre_mask]) if len(pre_data) > 0 else np.nan,
                        'post_min': np.min(post_data) if len(post_data) > 0 else np.nan,
                        'post_max': np.max(post_data) if len(post_data) > 0 else np.nan,
                        'post_mean': np.mean(post_data) if len(post_data) > 0 else np.nan,
                        'post_area': np.trapz(post_data, time_array[post_mask]) if len(post_data) > 0 else np.nan,
                        'signal_type': 'fiber_dff'
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            save_dir = filedialog.askdirectory(title='Select directory to save statistics')
            if save_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"running_induced_drug_statistics_{timestamp}.csv"
                save_path = os.path.join(save_dir, filename)
                df.to_csv(save_path, index=False)
                log_message(f"Statistics exported to {save_path}")
    
    """"Multi-Animal Drug Analysis"""

    def multi_animal_drug_analysis(self):
        """Multi-animal drug analysis with table interface"""
        if not self.multi_animal_data:
            log_message("No animal data available", "ERROR")
            return
        
        # Create analysis window similar to AcrossdayAnalysis
        if self.analysis_window is None or not self.analysis_window.winfo_exists():
            self.create_analysis_window()
        
        self.analysis_window.title("Multi-Animal Drug Analysis Configuration")
        self.analysis_window.deiconify()
        self.analysis_window.lift()
    
    def create_analysis_window(self):
        """Create multi-animal analysis configuration window"""
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.geometry("670x400")
        self.analysis_window.transient(self.root)
        
        main_frame = tk.Frame(self.analysis_window, bg="#f8f8f8")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg="#ecf0f1", relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(control_frame, text="+ Add Row", command=self.add_row,
                 bg="#fefefe", fg="#000000", font=("Microsoft YaHei", 9, "bold"),
                 relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="- Remove Row", command=self.remove_row,
                 bg="#ffffff", fg="#000000", font=("Microsoft YaHei", 9, "bold"),
                 relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="+ Add Column", command=self.add_column,
                 bg="#ffffff", fg="#000000", font=("Microsoft YaHei", 9, "bold"),
                 relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="- Remove Column", command=self.remove_column,
                 bg="#ffffff", fg="#000000", font=("Microsoft YaHei", 9, "bold"),
                 relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Run Analysis", command=self.run_multi_animal_analysis,
                 bg="#ffffff", fg="#000000", font=("Microsoft YaHei", 9, "bold"),
                 relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        # Table container
        table_container = tk.Frame(main_frame, bg="#ffffff")
        table_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(table_container, bg="#ffffff")
        v_scrollbar = tk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(table_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.table_frame = tk.Frame(self.canvas, bg="#ffffff")
        
        self.canvas.create_window((0, 0), window=self.table_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        def configure_scroll_region(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        self.table_frame.bind("<Configure>", configure_scroll_region)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.initialize_table()
    
    def initialize_table(self):
        """Initialize table with default values"""
        for i in range(self.num_rows):
            self.row_headers[i] = f"Day{i+1}"
        for j in range(self.num_cols):
            self.col_headers[j] = f"Animal{j+1}"
        
        self.rebuild_table()
        
    def run_multi_animal_analysis(self):
        """Run multi-animal drug analysis"""
        if not self.table_data:
            log_message("No data in table", "WARNING")
            return
        
        # Show parameter window
        self._show_multi_animal_param_window()
    
    def _show_multi_animal_param_window(self):
        """Show parameter window for multi-animal drug analysis"""
        param_window = tk.Toplevel(self.analysis_window)
        param_window.title("Drug Analysis Parameters - Multi Animal")
        param_window.geometry("450x450")
        param_window.configure(bg='#f8f8f8')
        param_window.transient(self.analysis_window)
        param_window.grab_set()
        
        # Title
        title_label = tk.Label(param_window, text="Multi-Animal Drug Analysis Settings", 
                            font=("Microsoft YaHei", 12, "bold"), bg="#f8f8f8", fg="#2c3e50")
        title_label.pack(pady=15)
        
        # Event type selection
        event_frame = tk.LabelFrame(param_window, text="Analysis Type", 
                                font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        event_frame.pack(fill=tk.X, padx=20, pady=10)
        
        event_type_var = tk.StringVar(value="drug_induced")
        tk.Radiobutton(event_frame, text="Drug Induced", variable=event_type_var, 
                    value="drug_induced", bg="#f8f8f8", 
                    font=("Microsoft YaHei", 9)).pack(anchor=tk.W, padx=10, pady=5)
        tk.Radiobutton(event_frame, text="Running Induced", variable=event_type_var, 
                    value="running_induced", bg="#f8f8f8",
                    font=("Microsoft YaHei", 9)).pack(anchor=tk.W, padx=10, pady=5)
        
        # Time window settings
        time_frame = tk.LabelFrame(param_window, text="Time Window Settings", 
                                font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        time_frame.pack(fill=tk.X, padx=20, pady=10)
        
        pre_frame = tk.Frame(time_frame, bg="#f8f8f8")
        pre_frame.pack(pady=5)
        tk.Label(pre_frame, text="Pre Event Window (s):", bg="#f8f8f8", 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        pre_time_var = tk.StringVar(value="600")
        tk.Entry(pre_frame, textvariable=pre_time_var, width=10, 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=5)
        
        post_frame = tk.Frame(time_frame, bg="#f8f8f8")
        post_frame.pack(pady=5)
        tk.Label(post_frame, text="Post Event Window (s):", bg="#f8f8f8", 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        post_time_var = tk.StringVar(value="1200")
        tk.Entry(post_frame, textvariable=post_time_var, width=10, 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=5)
        
        # Running type selection (only for running_induced)
        running_frame = tk.LabelFrame(param_window, text="Running Analysis Settings", 
                                    font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        running_frame.pack(fill=tk.X, padx=20, pady=10)
        
        running_type_label = tk.Label(running_frame, text="Running Type:", bg="#f8f8f8", 
                                    font=("Microsoft YaHei", 8))
        running_type_label.pack(anchor=tk.W, padx=10, pady=(5,2))
        
        running_types = ['general_onsets', 'jerks', 'locomotion_onsets', 'reset_onsets',
                        'general_offsets', 'locomotion_offsets', 'reset_offsets']
        running_type_var = tk.StringVar(value=running_types[0])
        running_type_combo = ttk.Combobox(running_frame, textvariable=running_type_var,
                                        values=running_types, state="readonly",
                                        font=("Microsoft YaHei", 8))
        running_type_combo.pack(padx=10, pady=5, fill=tk.X)
        
        drug_name_label = tk.Label(running_frame, text="Drug Name:", bg="#f8f8f8", 
                                font=("Microsoft YaHei", 8))
        drug_name_label.pack(anchor=tk.W, padx=10, pady=(5,2))
        
        drug_name_var = tk.StringVar(value="Drug")
        drug_name_entry = tk.Entry(running_frame, textvariable=drug_name_var,
                                font=("Microsoft YaHei", 8))
        drug_name_entry.pack(padx=10, pady=5, fill=tk.X)
        
        # Initially hide running settings
        running_frame.pack_forget()
        
        def update_ui(*args):
            if event_type_var.get() == "running_induced":
                running_frame.pack(fill=tk.X, padx=20, pady=10)
            else:
                running_frame.pack_forget()
        
        event_type_var.trace('w', update_ui)
        
        # Export option
        export_frame = tk.LabelFrame(param_window, text="Export Options", 
                                    font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        export_frame.pack(fill=tk.X, padx=20, pady=10)
        
        export_var = tk.BooleanVar(value=False)
        tk.Checkbutton(export_frame, text="Export statistic results to CSV", 
                    variable=export_var, bg="#f8f8f8",
                    font=("Microsoft YaHei", 8)).pack(anchor=tk.W, padx=10, pady=5)
        
        def run_analysis():
            try:
                pre_time = float(pre_time_var.get())
                post_time = float(post_time_var.get())
                event_type = event_type_var.get()
                
                if pre_time <= 0 or post_time <= 0:
                    log_message("Time must be positive numbers", "WARNING")
                    return
                
                if event_type == "running_induced":
                    running_type = running_type_var.get()
                    drug_name = drug_name_var.get().strip()
                    if not drug_name:
                        log_message("Please enter drug name", "WARNING")
                        return
                    param_window.destroy()
                    self._run_multi_animal_running_induced(pre_time, post_time,
                                                        running_type, drug_name, export_var.get())
                else:
                    param_window.destroy()
                    self._run_multi_animal_drug_induced(pre_time, post_time, export_var.get())
                    
            except ValueError:
                log_message("Please enter valid time values", "WARNING")
        
        # Buttons
        button_frame = tk.Frame(param_window, bg="#f8f8f8")
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="Start Analysis", command=run_analysis,
                bg="#3498db", fg="white", font=("Microsoft YaHei", 9, "bold"),
                relief=tk.FLAT, padx=15, pady=5).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="Cancel", command=param_window.destroy,
                bg="#95a5a6", fg="white", font=("Microsoft YaHei", 9),
                relief=tk.FLAT, padx=15, pady=5).pack(side=tk.LEFT, padx=10)

    def _run_multi_animal_drug_induced(self, pre_time, post_time, export_statistics):
        """Run drug-induced analysis for multiple animals grouped by day"""
        # Validate table data
        if not self.table_data:
            log_message("No data in the table to analyze", "WARNING")
            return
        
        # Group animals by day
        day_data = {}
        for i in range(self.num_rows):
            day_name = self.row_headers.get(i, f"Day{i+1}")
            day_animals = []
            
            for j in range(self.num_cols):
                if (i, j) in self.table_data:
                    animal_id = self.table_data[(i, j)]
                    # Find animal data
                    for animal_data in self.multi_animal_data:
                        if animal_data.get('animal_id') == animal_id:
                            day_animals.append(animal_data)
                            break
            
            if day_animals:
                day_data[day_name] = day_animals
        
        if not day_data:
            log_message("No valid data found for any day", "WARNING")
            return
        
        log_message(f"Starting multi-animal drug-induced analysis for {len(day_data)} days...")
        
        # Analyze each day
        results = {}
        all_statistics_rows = []
        
        for day_name, animals in day_data.items():
            log_message(f"Analyzing {day_name} with {len(animals)} animals...")
            day_result, day_stats = self._analyze_day_drug_induced(
                day_name, animals, pre_time, post_time, export_statistics)
            
            if day_result:
                results[day_name] = day_result
            if day_stats:
                all_statistics_rows.extend(day_stats)
        
        # Export statistics if requested
        if export_statistics and all_statistics_rows:
            self._export_multi_animal_drug_statistics(all_statistics_rows, "drug_induced")
        
        # Plot results
        if results:
            self._plot_multi_animal_drug_induced_results(results, pre_time, post_time)
            log_message("Multi-animal drug-induced analysis completed")
        else:
            log_message("No valid results for drug-induced analysis", "ERROR")

    def _run_multi_animal_running_induced(self, pre_time, post_time, running_type, 
                                        drug_name, export_statistics):
        """Run running-induced drug analysis for multiple animals grouped by day"""
        if not self.table_data:
            log_message("No data in the table to analyze", "WARNING")
            return
        
        # Group animals by day
        day_data = {}
        for i in range(self.num_rows):
            day_name = self.row_headers.get(i, f"Day{i+1}")
            day_animals = []
            
            for j in range(self.num_cols):
                if (i, j) in self.table_data:
                    animal_id = self.table_data[(i, j)]
                    for animal_data in self.multi_animal_data:
                        if animal_data.get('animal_id') == animal_id:
                            day_animals.append(animal_data)
                            break
            
            if day_animals:
                day_data[day_name] = day_animals
        
        if not day_data:
            log_message("No valid data found for any day", "WARNING")
            return
        
        log_message(f"Starting multi-animal running-induced drug analysis for {len(day_data)} days...")
        
        # Analyze each day
        results = {}
        all_statistics_rows = []
        
        for day_name, animals in day_data.items():
            log_message(f"Analyzing {day_name} with {len(animals)} animals...")
            day_result, day_stats = self._analyze_day_running_induced(
                day_name, animals, pre_time, post_time, running_type, drug_name, export_statistics)
            
            if day_result:
                results[day_name] = day_result
            if day_stats:
                all_statistics_rows.extend(day_stats)
        
        # Export statistics if requested
        if export_statistics and all_statistics_rows:
            self._export_multi_animal_drug_statistics(all_statistics_rows, "running_induced")
        
        # Plot results
        if results:
            self._plot_multi_animal_running_induced_results(results, running_type, 
                                                        drug_name, pre_time, post_time)
            log_message("Multi-animal running-induced drug analysis completed")
        else:
            log_message("No valid results for running-induced analysis", "ERROR")

    def _analyze_day_drug_induced(self, day_name, animals, pre_time, post_time, 
                                collect_statistics=False):
        """Analyze drug-induced effects for one day (multiple animals combined)"""
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        
        # Collect all wavelengths
        target_wavelengths = []
        for animal_data in animals:
            if 'target_signal' in animal_data:
                signal = animal_data['target_signal']
                wls = signal.split('+') if '+' in signal else [signal]
                target_wavelengths.extend(wls)
        target_wavelengths = sorted(list(set(target_wavelengths)))
        
        if not target_wavelengths:
            target_wavelengths = ['470']
        
        # Initialize storage
        all_dff_episodes = {wl: [] for wl in target_wavelengths}
        all_zscore_episodes = {wl: [] for wl in target_wavelengths}
        statistics_rows = []
        
        # Process each animal
        for animal_data in animals:
            try:
                animal_id = animal_data.get('animal_id', 'Unknown')
                
                fiber_data = animal_data.get('fiber_data_trimmed')
                if fiber_data is None or fiber_data.empty:
                    fiber_data = animal_data.get('fiber_data')
                channels = animal_data.get('channels', {})
                events_col = channels.get('events')
                
                if events_col is None or events_col not in fiber_data.columns:
                    log_message("Events column not found", "ERROR")
                    return
                
                # Find Drug events (Event1)
                drug_events = fiber_data[fiber_data[events_col].str.contains('Event1', na=False)]
                
                if len(drug_events) == 0:
                    log_message("No drug events (Event1) found in fiber data", "WARNING")
                    return
                
                time_col = channels['time']
                drug_start_time = drug_events[time_col].iloc[0]
                animal_data['drug_start_time'] = drug_start_time

                preprocessed_data = animal_data.get('preprocessed_data')
                
                if preprocessed_data is None:
                    continue
                
                fiber_timestamps = preprocessed_data[time_col].values
                dff_data = animal_data.get('dff_data', {})
                active_channels = animal_data.get('active_channels', [])
                
                # Extract episodes for each channel and wavelength
                for channel in active_channels:
                    for wavelength in target_wavelengths:
                        dff_key = f"{channel}_{wavelength}"
                        if dff_key in dff_data:
                            data = dff_data[dff_key]
                            if isinstance(data, pd.Series):
                                data = data.values
                            
                            start_idx = np.argmin(np.abs(fiber_timestamps - (drug_start_time - pre_time)))
                            end_idx = np.argmin(np.abs(fiber_timestamps - (drug_start_time + post_time)))
                            
                            if end_idx > start_idx:
                                episode_data = data[start_idx:end_idx]
                                episode_times = fiber_timestamps[start_idx:end_idx] - drug_start_time
                                
                                if len(episode_times) > 1:
                                    interp_data = np.interp(time_array, episode_times, episode_data)
                                    all_dff_episodes[wavelength].append(interp_data)
                                    
                                    # Calculate z-score using pre_time window as baseline
                                    baseline_mask = (episode_times >= -pre_time) & (episode_times <= 0)
                                    baseline_data = episode_data[baseline_mask]
                                    
                                    if len(baseline_data) > 0:
                                        mean_baseline = np.nanmean(baseline_data)
                                        std_baseline = np.nanstd(baseline_data)
                                        
                                        if std_baseline == 0:
                                            std_baseline = 1e-10
                                        
                                        # Calculate z-score for the entire episode
                                        zscore_episode = (episode_data - mean_baseline) / std_baseline
                                        interp_zscore = np.interp(time_array, episode_times, zscore_episode)
                                        all_zscore_episodes[wavelength].append(interp_zscore)
                                        
                                        # Collect statistics
                                        if collect_statistics:
                                            pre_mask = (time_array >= -pre_time) & (time_array <= 0)
                                            post_mask = (time_array >= 0) & (time_array <= post_time)
                                            
                                            pre_data = interp_data[pre_mask]
                                            post_data = interp_data[post_mask]
                                            
                                            statistics_rows.append({
                                                'day': day_name,
                                                'animal_id': animal_id,
                                                'analysis_type': 'drug_induced',
                                                'channel': channel,
                                                'wavelength': wavelength,
                                                'trial': 1,
                                                'pre_min': np.min(pre_data) if len(pre_data) > 0 else np.nan,
                                                'pre_max': np.max(pre_data) if len(pre_data) > 0 else np.nan,
                                                'pre_mean': np.mean(pre_data) if len(pre_data) > 0 else np.nan,
                                                'pre_area': np.trapz(pre_data, time_array[pre_mask]) if len(pre_data) > 0 else np.nan,
                                                'post_min': np.min(post_data) if len(post_data) > 0 else np.nan,
                                                'post_max': np.max(post_data) if len(post_data) > 0 else np.nan,
                                                'post_mean': np.mean(post_data) if len(post_data) > 0 else np.nan,
                                                'post_area': np.trapz(post_data, time_array[post_mask]) if len(post_data) > 0 else np.nan,
                                                'signal_type': 'fiber_dff'
                                            })
                                            
                                            pre_zscore_data = interp_zscore[pre_mask]
                                            post_zscore_data = interp_zscore[post_mask]
                                            
                                            statistics_rows.append({
                                                'day': day_name,
                                                'animal_id': animal_id,
                                                'analysis_type': 'drug_induced',
                                                'channel': channel,
                                                'wavelength': wavelength,
                                                'trial': 1,
                                                'pre_min': np.min(pre_zscore_data) if len(pre_zscore_data) > 0 else np.nan,
                                                'pre_max': np.max(pre_zscore_data) if len(pre_zscore_data) > 0 else np.nan,
                                                'pre_mean': np.mean(pre_zscore_data) if len(pre_zscore_data) > 0 else np.nan,
                                                'pre_area': np.trapz(pre_zscore_data, time_array[pre_mask]) if len(pre_zscore_data) > 0 else np.nan,
                                                'post_min': np.min(post_zscore_data) if len(post_zscore_data) > 0 else np.nan,
                                                'post_max': np.max(post_zscore_data) if len(post_zscore_data) > 0 else np.nan,
                                                'post_mean': np.mean(post_zscore_data) if len(post_zscore_data) > 0 else np.nan,
                                                'post_area': np.trapz(post_zscore_data, time_array[post_mask]) if len(post_zscore_data) > 0 else np.nan,
                                                'signal_type': 'fiber_zscore'
                                            })
            
            except Exception as e:
                log_message(f"Error analyzing {animal_data.get('animal_id', 'Unknown')}: {str(e)}", "ERROR")
                continue
        
        # Calculate results
        result = {
            'time': time_array,
            'dff': all_dff_episodes,
            'zscore': all_zscore_episodes,
            'target_wavelengths': target_wavelengths
        }
        
        return result, statistics_rows if collect_statistics else None

    def _analyze_day_running_induced(self, day_name, animals, pre_time, post_time, 
                                    running_type, drug_name, collect_statistics=False):
        """Analyze running-induced effects for one day (multiple animals, pre/post drug)"""
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        
        # Collect wavelengths
        target_wavelengths = []
        for animal_data in animals:
            if 'target_signal' in animal_data:
                signal = animal_data['target_signal']
                wls = signal.split('+') if '+' in signal else [signal]
                target_wavelengths.extend(wls)
        target_wavelengths = sorted(list(set(target_wavelengths)))
        
        if not target_wavelengths:
            target_wavelengths = ['470']
        
        # Initialize storage for pre and post drug
        pre_drug_running = []
        post_drug_running = []
        pre_drug_dff = {wl: [] for wl in target_wavelengths}
        post_drug_dff = {wl: [] for wl in target_wavelengths}
        pre_drug_zscore = {wl: [] for wl in target_wavelengths}
        post_drug_zscore = {wl: [] for wl in target_wavelengths}
        
        statistics_rows = []
        
        # Get zscore baseline parameters
        zscore_baseline_type, zscore_baseline_window = self._get_zscore_baseline_params(running_type)
        
        # Process each animal
        for animal_data in animals:
            try:
                fiber_data = animal_data.get('fiber_data_trimmed')
                if fiber_data is None or fiber_data.empty:
                    fiber_data = animal_data.get('fiber_data')
                channels = animal_data.get('channels', {})
                events_col = channels.get('events')
                
                if events_col is None or events_col not in fiber_data.columns:
                    log_message("Events column not found", "ERROR")
                    return
                
                # Find Drug events (Event1)
                drug_events = fiber_data[fiber_data[events_col].str.contains('Event1', na=False)]
                
                if len(drug_events) == 0:
                    log_message("No drug events (Event1) found in fiber data", "WARNING")
                    return
                
                time_col = channels['time']
                drug_start_time = drug_events[time_col].iloc[0]
                
                animal_data['drug_start_time'] = drug_start_time
                
                # Check for running behavior
                if 'treadmill_behaviors' not in animal_data:
                    continue
                
                treadmill_behaviors = animal_data['treadmill_behaviors']
                if running_type not in treadmill_behaviors:
                    continue
                
                events = treadmill_behaviors[running_type]
                pre_drug_events = [e for e in events if e < drug_start_time]
                post_drug_events = [e for e in events if e >= drug_start_time]
                
                preprocessed_data = animal_data.get('preprocessed_data')
                fiber_timestamps = preprocessed_data[time_col].values
                
                ast2_data = animal_data.get('ast2_data_adjusted')
                running_timestamps = ast2_data['data']['timestamps']
                processed_data = animal_data.get('running_processed_data')
                running_speed = processed_data['filtered_speed'] if processed_data else ast2_data['data']['speed']
                
                dff_data = animal_data.get('dff_data', {})
                active_channels = animal_data.get('active_channels', [])
                
                # Process pre-drug events
                for event in pre_drug_events:
                    # Running
                    start_idx = np.argmin(np.abs(running_timestamps - (event - pre_time)))
                    end_idx = np.argmin(np.abs(running_timestamps - (event + post_time)))
                    
                    if end_idx > start_idx:
                        episode_data = running_speed[start_idx:end_idx]
                        episode_times = running_timestamps[start_idx:end_idx] - event
                        
                        if len(episode_times) > 1:
                            interp_data = np.interp(time_array, episode_times, episode_data)
                            pre_drug_running.append(interp_data)
                
                # Process post-drug events (similar)
                for event in post_drug_events:
                    start_idx = np.argmin(np.abs(running_timestamps - (event - pre_time)))
                    end_idx = np.argmin(np.abs(running_timestamps - (event + post_time)))
                    
                    if end_idx > start_idx:
                        episode_data = running_speed[start_idx:end_idx]
                        episode_times = running_timestamps[start_idx:end_idx] - event
                        
                        if len(episode_times) > 1:
                            interp_data = np.interp(time_array, episode_times, episode_data)
                            post_drug_running.append(interp_data)
                
                # Process fiber data for pre and post drug events
                for channel in active_channels:
                    for wavelength in target_wavelengths:
                        dff_key = f"{channel}_{wavelength}"
                        if dff_key not in dff_data:
                            continue
                        
                        data = dff_data[dff_key]
                        if isinstance(data, pd.Series):
                            data = data.values
                        
                        # Process pre-drug events
                        for event in pre_drug_events:
                            # Determine baseline window based on event type
                            if zscore_baseline_type == 'offset':
                                baseline_start = event
                                baseline_end = event + zscore_baseline_window
                            else:
                                baseline_start = event - zscore_baseline_window
                                baseline_end = event
                            
                            baseline_start_idx = np.argmin(np.abs(fiber_timestamps - baseline_start))
                            baseline_end_idx = np.argmin(np.abs(fiber_timestamps - baseline_end))
                            
                            if baseline_end_idx > baseline_start_idx:
                                baseline_data = data[baseline_start_idx:baseline_end_idx]
                                mean_dff = np.nanmean(baseline_data)
                                std_dff = np.nanstd(baseline_data)
                                
                                if std_dff == 0:
                                    std_dff = 1e-10
                                
                                # Extract plotting window
                                start_idx = np.argmin(np.abs(fiber_timestamps - (event - pre_time)))
                                end_idx = np.argmin(np.abs(fiber_timestamps - (event + post_time)))
                                
                                if end_idx > start_idx:
                                    episode_data = data[start_idx:end_idx]
                                    episode_times = fiber_timestamps[start_idx:end_idx] - event
                                    
                                    if len(episode_times) > 1:
                                        # Calculate z-score using fixed baseline window
                                        zscore_episode = (episode_data - mean_dff) / std_dff
                                        interp_zscore = np.interp(time_array, episode_times, zscore_episode)
                                        pre_drug_zscore[wavelength].append(interp_zscore)
                                        
                                        # Also store dFF data
                                        interp_dff = np.interp(time_array, episode_times, episode_data)
                                        pre_drug_dff[wavelength].append(interp_dff)
                        
                        # Process post-drug events (similar)
                        for event in post_drug_events:
                            # Determine baseline window based on event type
                            if zscore_baseline_type == 'offset':
                                baseline_start = event
                                baseline_end = event + zscore_baseline_window
                            else:
                                baseline_start = event - zscore_baseline_window
                                baseline_end = event
                            
                            baseline_start_idx = np.argmin(np.abs(fiber_timestamps - baseline_start))
                            baseline_end_idx = np.argmin(np.abs(fiber_timestamps - baseline_end))
                            
                            if baseline_end_idx > baseline_start_idx:
                                baseline_data = data[baseline_start_idx:baseline_end_idx]
                                mean_dff = np.nanmean(baseline_data)
                                std_dff = np.nanstd(baseline_data)
                                
                                if std_dff == 0:
                                    std_dff = 1e-10
                                
                                # Extract plotting window
                                start_idx = np.argmin(np.abs(fiber_timestamps - (event - pre_time)))
                                end_idx = np.argmin(np.abs(fiber_timestamps - (event + post_time)))
                                
                                if end_idx > start_idx:
                                    episode_data = data[start_idx:end_idx]
                                    episode_times = fiber_timestamps[start_idx:end_idx] - event
                                    
                                    if len(episode_times) > 1:
                                        # Calculate z-score using fixed baseline window
                                        zscore_episode = (episode_data - mean_dff) / std_dff
                                        interp_zscore = np.interp(time_array, episode_times, zscore_episode)
                                        post_drug_zscore[wavelength].append(interp_zscore)
                                        
                                        # Also store dFF data
                                        interp_dff = np.interp(time_array, episode_times, episode_data)
                                        post_drug_dff[wavelength].append(interp_dff)
            except Exception as e:
                log_message(f"Error analyzing {animal_data.get('animal_id', 'Unknown')}: {str(e)}", "ERROR")
                continue
        
        # Combine results
        result = {
            'time': time_array,
            'pre_drug': {
                'running': np.array(pre_drug_running) if pre_drug_running else np.array([]),
                'dff': pre_drug_dff,
                'zscore': pre_drug_zscore
            },
            'post_drug': {
                'running': np.array(post_drug_running) if post_drug_running else np.array([]),
                'dff': post_drug_dff,
                'zscore': post_drug_zscore
            },
            'target_wavelengths': target_wavelengths
        }
        
        return result, statistics_rows if collect_statistics else None
    
    def _plot_multi_animal_drug_induced_results(self, results, pre_time, post_time):
        """Plot multi-animal drug-induced results with all days overlaid"""
        # Get wavelengths from first result
        target_wavelengths = []
        for day_name, data in results.items():
            if 'target_wavelengths' in data:
                target_wavelengths = data['target_wavelengths']
                break
        
        if not target_wavelengths:
            target_wavelengths = ['470']
        
        result_window = tk.Toplevel(self.root)
        wavelength_label = '+'.join(target_wavelengths)
        result_window.title(f"Multi-Animal Drug Analysis - All Days ({wavelength_label}nm)")
        result_window.state('zoomed')
        result_window.configure(bg='#f8f8f8')
        
        num_wavelengths = len(target_wavelengths)
        num_cols = 2 * num_wavelengths
        
        fig = Figure(figsize=(4 * num_cols, 8), dpi=100)
        
        fiber_colors = ['#008000', "#FF0000", '#FFA500']
        plot_idx = 1
        
        # Row 1: Traces
        for wl_idx, wavelength in enumerate(target_wavelengths):
            color = fiber_colors[wl_idx % len(fiber_colors)]
            
            # dFF trace
            ax_dff = fig.add_subplot(2, num_cols, plot_idx)
            for idx, (day_name, data) in enumerate(results.items()):
                day_color = self.day_colors[idx % len(self.day_colors)]
                episodes = data['dff'].get(wavelength, [])
                if episodes:
                    episodes_array = np.array(episodes)
                    mean_response = np.nanmean(episodes_array, axis=0)
                    sem_response = np.nanstd(episodes_array, axis=0) / np.sqrt(len(episodes))
                    
                    time_array = data['time']
                    ax_dff.plot(time_array, mean_response, color=day_color, linewidth=2, label=day_name)
                    ax_dff.fill_between(time_array, mean_response - sem_response,
                                    mean_response + sem_response, color=day_color, alpha=0.3)
            
            ax_dff.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Drug')
            ax_dff.set_xlim([time_array[0], time_array[-1]])
            ax_dff.set_xlabel('Time (s)')
            ax_dff.set_ylabel('ΔF/F')
            ax_dff.set_title(f'Fiber ΔF/F {wavelength}nm - All Days')
            ax_dff.legend()
            ax_dff.grid(False)
            plot_idx += 1
            
            # Z-score trace
            ax_zscore = fig.add_subplot(2, num_cols, plot_idx)
            for idx, (day_name, data) in enumerate(results.items()):
                day_color = self.day_colors[idx % len(self.day_colors)]
                episodes = data['zscore'].get(wavelength, [])
                if episodes:
                    episodes_array = np.array(episodes)
                    mean_response = np.nanmean(episodes_array, axis=0)
                    sem_response = np.nanstd(episodes_array, axis=0) / np.sqrt(len(episodes))
                    
                    time_array = data['time']
                    ax_zscore.plot(time_array, mean_response, color=day_color, linewidth=2, label=day_name)
                    ax_zscore.fill_between(time_array, mean_response - sem_response,
                                        mean_response + sem_response, color=day_color, alpha=0.3)
            
            ax_zscore.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Drug')
            ax_zscore.set_xlim([time_array[0], time_array[-1]])
            ax_zscore.set_xlabel('Time (s)')
            ax_zscore.set_ylabel('Z-score')
            ax_zscore.set_title(f'Fiber Z-score {wavelength}nm - All Days')
            ax_zscore.legend()
            ax_zscore.grid(False)
            plot_idx += 1
        
        # Row 2: Heatmaps
        for wl_idx, wavelength in enumerate(target_wavelengths):
            # dFF heatmap
            ax_dff_heat = fig.add_subplot(2, num_cols, plot_idx)
            all_episodes = []
            for day_name, data in results.items():
                episodes = data['dff'].get(wavelength, [])
                if episodes:
                    all_episodes.extend(episodes)
            
            if all_episodes:
                time_array = list(results.values())[0]['time']
                episodes_array = np.array(all_episodes)
                im = ax_dff_heat.imshow(episodes_array, aspect='auto',
                                    extent=[time_array[0], time_array[-1],
                                            len(episodes_array), 1],
                                    cmap='viridis', origin='lower')
                ax_dff_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
                ax_dff_heat.set_xlabel('Time (s)')
                ax_dff_heat.set_ylabel('Trial (All Days)')
                ax_dff_heat.set_title(f'Fiber ΔF/F Heatmap {wavelength}nm')
                plt.colorbar(im, ax=ax_dff_heat, label='ΔF/F', orientation='horizontal')
            plot_idx += 1
            
            # Z-score heatmap
            ax_zscore_heat = fig.add_subplot(2, num_cols, plot_idx)
            all_episodes = []
            for day_name, data in results.items():
                episodes = data['zscore'].get(wavelength, [])
                if episodes:
                    all_episodes.extend(episodes)
            
            if all_episodes:
                time_array = list(results.values())[0]['time']
                episodes_array = np.array(all_episodes)
                im = ax_zscore_heat.imshow(episodes_array, aspect='auto',
                                        extent=[time_array[0], time_array[-1],
                                                len(episodes_array), 1],
                                        cmap='coolwarm', origin='lower')
                ax_zscore_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
                ax_zscore_heat.set_xlabel('Time (s)')
                ax_zscore_heat.set_ylabel('Trial (All Days)')
                ax_zscore_heat.set_title(f'Fiber Z-score Heatmap {wavelength}nm')
                plt.colorbar(im, ax=ax_zscore_heat, label='Z-score', orientation='horizontal')
            plot_idx += 1
        
        fig.tight_layout()
        
        canvas_frame = tk.Frame(result_window, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        
        log_message(f"Multi-animal drug-induced results plotted for {len(results)} days")

    def _plot_multi_animal_running_induced_results(self, results, running_type, 
                                                drug_name, pre_time, post_time):
        """Plot multi-animal running-induced results with pre/post drug comparison"""
        target_wavelengths = []
        for day_name, data in results.items():
            if 'target_wavelengths' in data:
                target_wavelengths = data['target_wavelengths']
                break
        
        if not target_wavelengths:
            target_wavelengths = ['470']
        
        result_window = tk.Toplevel(self.root)
        wavelength_label = '+'.join(target_wavelengths)
        result_window.title(f"Multi-Animal Drug-Running Analysis - {running_type}")
        result_window.state('zoomed')
        result_window.configure(bg='#f8f8f8')
        
        num_wavelengths = len(target_wavelengths)
        num_cols = 1 + 2 * num_wavelengths
        
        fig = Figure(figsize=(4 * num_cols, 8), dpi=100)
        
        fiber_colors = ['#008000', "#FF0000", '#FFA500']
        plot_idx = 1
        
        time_array = list(results.values())[0]['time']
        
        # Row 1: Traces
        # Running trace
        ax_running = fig.add_subplot(2, num_cols, plot_idx)
        
        for idx, (day_name, data) in enumerate(results.items()):
            day_color = self.day_colors[idx % len(self.day_colors)]
            
            # Pre-drug
            if len(data['pre_drug']['running']) > 0:
                mean_pre = np.nanmean(data['pre_drug']['running'], axis=0)
                sem_pre = np.nanstd(data['pre_drug']['running'], axis=0) / np.sqrt(len(data['pre_drug']['running']))
                ax_running.plot(time_array, mean_pre, color=day_color, linewidth=2, 
                            linestyle='--', label=f'{day_name} Pre {drug_name}')
                ax_running.fill_between(time_array, mean_pre - sem_pre, mean_pre + sem_pre,
                                    color=day_color, alpha=0.2)
            
            # Post-drug
            if len(data['post_drug']['running']) > 0:
                mean_post = np.nanmean(data['post_drug']['running'], axis=0)
                sem_post = np.nanstd(data['post_drug']['running'], axis=0) / np.sqrt(len(data['post_drug']['running']))
                ax_running.plot(time_array, mean_post, color=day_color, linewidth=2, 
                            linestyle='-', label=f'{day_name} Post {drug_name}')
                ax_running.fill_between(time_array, mean_post - sem_post, mean_post + sem_post,
                                    color=day_color, alpha=0.3)
        
        ax_running.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Event')
        ax_running.set_xlim([time_array[0], time_array[-1]])
        ax_running.set_xlabel('Time (s)')
        ax_running.set_ylabel('Speed (cm/s)')
        ax_running.set_title(f'Running Speed - {running_type}')
        ax_running.legend(fontsize=8)
        ax_running.grid(False)
        plot_idx += 1
        
        # Fiber traces
        for wl_idx, wavelength in enumerate(target_wavelengths):
            color = fiber_colors[wl_idx % len(fiber_colors)]
            
            # dFF trace
            ax_dff = fig.add_subplot(2, num_cols, plot_idx)
            
            for idx, (day_name, data) in enumerate(results.items()):
                day_color = self.day_colors[idx % len(self.day_colors)]
                
                # Pre-drug
                pre_episodes = data['pre_drug']['dff'].get(wavelength, [])
                if pre_episodes:
                    episodes_array = np.array(pre_episodes)
                    mean_pre = np.nanmean(episodes_array, axis=0)
                    sem_pre = np.nanstd(episodes_array, axis=0) / np.sqrt(len(pre_episodes))
                    ax_dff.plot(time_array, mean_pre, color=day_color, linewidth=2, 
                            linestyle='--', label=f'{day_name} Pre {drug_name}')
                    ax_dff.fill_between(time_array, mean_pre - sem_pre, mean_pre + sem_pre,
                                    color=day_color, alpha=0.2)
                
                # Post-drug
                post_episodes = data['post_drug']['dff'].get(wavelength, [])
                if post_episodes:
                    episodes_array = np.array(post_episodes)
                    mean_post = np.nanmean(episodes_array, axis=0)
                    sem_post = np.nanstd(episodes_array, axis=0) / np.sqrt(len(post_episodes))
                    ax_dff.plot(time_array, mean_post, color=day_color, linewidth=2, 
                            linestyle='-', label=f'{day_name} Post {drug_name}')
                    ax_dff.fill_between(time_array, mean_post - sem_post, mean_post + sem_post,
                                    color=day_color, alpha=0.3)
            
            ax_dff.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Event')
            ax_dff.set_xlim([time_array[0], time_array[-1]])
            ax_dff.set_xlabel('Time (s)')
            ax_dff.set_ylabel('ΔF/F')
            ax_dff.set_title(f'Fiber ΔF/F {wavelength}nm')
            ax_dff.legend(fontsize=8)
            ax_dff.grid(False)
            plot_idx += 1
            
            # Z-score trace (similar structure)
            ax_zscore = fig.add_subplot(2, num_cols, plot_idx)
            
            for idx, (day_name, data) in enumerate(results.items()):
                day_color = self.day_colors[idx % len(self.day_colors)]
                
                pre_episodes = data['pre_drug']['zscore'].get(wavelength, [])
                if pre_episodes:
                    episodes_array = np.array(pre_episodes)
                    mean_pre = np.nanmean(episodes_array, axis=0)
                    sem_pre = np.nanstd(episodes_array, axis=0) / np.sqrt(len(pre_episodes))
                    ax_zscore.plot(time_array, mean_pre, color=day_color, linewidth=2, 
                                linestyle='--', label=f'{day_name} Pre {drug_name}')
                    ax_zscore.fill_between(time_array, mean_pre - sem_pre, mean_pre + sem_pre,
                                        color=day_color, alpha=0.2)
                
                post_episodes = data['post_drug']['zscore'].get(wavelength, [])
                if post_episodes:
                    episodes_array = np.array(post_episodes)
                    mean_post = np.nanmean(episodes_array, axis=0)
                    sem_post = np.nanstd(episodes_array, axis=0) / np.sqrt(len(post_episodes))
                    ax_zscore.plot(time_array, mean_post, color=day_color, linewidth=2, 
                                linestyle='-', label=f'{day_name} Post {drug_name}')
                    ax_zscore.fill_between(time_array, mean_post - sem_post, mean_post + sem_post,
                                        color=day_color, alpha=0.3)
            
            ax_zscore.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Event')
            ax_zscore.set_xlim([time_array[0], time_array[-1]])
            ax_zscore.set_xlabel('Time (s)')
            ax_zscore.set_ylabel('Z-score')
            ax_zscore.set_title(f'Fiber Z-score {wavelength}nm')
            ax_zscore.legend(fontsize=8)
            ax_zscore.grid(False)
            plot_idx += 1
        
        # Row 2: Heatmaps (combined all days)
        # Running heatmap
        ax_running_heat = fig.add_subplot(2, num_cols, plot_idx)
        all_pre_running = []
        all_post_running = []
        
        for day_name, data in results.items():
            if len(data['pre_drug']['running']) > 0:
                all_pre_running.extend(data['pre_drug']['running'])
            if len(data['post_drug']['running']) > 0:
                all_post_running.extend(data['post_drug']['running'])
        
        if all_pre_running and all_post_running:
            combined_running = np.vstack([np.array(all_pre_running), np.array(all_post_running)])
            n_pre = len(all_pre_running)
            
            im = ax_running_heat.imshow(combined_running, aspect='auto',
                                    extent=[time_array[0], time_array[-1],
                                            len(combined_running), 1],
                                    cmap='viridis', origin='lower')
            ax_running_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax_running_heat.axhline(y=n_pre, color='white', linestyle='-', linewidth=2)
            ax_running_heat.set_xlabel('Time (s)')
            ax_running_heat.set_ylabel('Trial')
            ax_running_heat.set_title('Running Speed Heatmap')
            plt.colorbar(im, ax=ax_running_heat, label='Speed (cm/s)', orientation='horizontal')
        plot_idx += 1
        
        # Fiber heatmaps
        for wl_idx, wavelength in enumerate(target_wavelengths):
            # dFF heatmap
            ax_dff_heat = fig.add_subplot(2, num_cols, plot_idx)
            all_pre_dff = []
            all_post_dff = []
            
            for day_name, data in results.items():
                pre_eps = data['pre_drug']['dff'].get(wavelength, [])
                post_eps = data['post_drug']['dff'].get(wavelength, [])
                if pre_eps:
                    all_pre_dff.extend(pre_eps)
                if post_eps:
                    all_post_dff.extend(post_eps)
            
            if all_pre_dff and all_post_dff:
                combined_dff = np.vstack([np.array(all_pre_dff), np.array(all_post_dff)])
                n_pre = len(all_pre_dff)
                
                im = ax_dff_heat.imshow(combined_dff, aspect='auto',
                                    extent=[time_array[0], time_array[-1],
                                            len(combined_dff), 1],
                                    cmap='viridis', origin='lower')
                ax_dff_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
                ax_dff_heat.axhline(y=n_pre, color='white', linestyle='-', linewidth=2)
                ax_dff_heat.set_xlabel('Time (s)')
                ax_dff_heat.set_ylabel('Trial')
                ax_dff_heat.set_title(f'Fiber ΔF/F Heatmap {wavelength}nm')
                plt.colorbar(im, ax=ax_dff_heat, label='ΔF/F', orientation='horizontal')
            plot_idx += 1
            
            # Z-score heatmap
            ax_zscore_heat = fig.add_subplot(2, num_cols, plot_idx)
            all_pre_zscore = []
            all_post_zscore = []
            
            for day_name, data in results.items():
                pre_eps = data['pre_drug']['zscore'].get(wavelength, [])
                post_eps = data['post_drug']['zscore'].get(wavelength, [])
                if pre_eps:
                    all_pre_zscore.extend(pre_eps)
                if post_eps:
                    all_post_zscore.extend(post_eps)
            
            if all_pre_zscore and all_post_zscore:
                combined_zscore = np.vstack([np.array(all_pre_zscore), np.array(all_post_zscore)])
                n_pre = len(all_pre_zscore)
                
                im = ax_zscore_heat.imshow(combined_zscore, aspect='auto',
                                        extent=[time_array[0], time_array[-1],
                                                len(combined_zscore), 1],
                                        cmap='coolwarm', origin='lower')
                ax_zscore_heat.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
                ax_zscore_heat.axhline(y=n_pre, color='white', linestyle='-', linewidth=2)
                ax_zscore_heat.set_xlabel('Time (s)')
                ax_zscore_heat.set_ylabel('Trial')
                ax_zscore_heat.set_title(f'Fiber Z-score Heatmap {wavelength}nm')
                plt.colorbar(im, ax=ax_zscore_heat, label='Z-score', orientation='horizontal')
            plot_idx += 1
        
        fig.tight_layout()
        
        canvas_frame = tk.Frame(result_window, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        
        log_message(f"Multi-animal running-induced drug results plotted for {len(results)} days")
        
    def _export_multi_animal_drug_statistics(self, statistics_rows, analysis_type):
        """Export multi-animal drug statistics"""
        if not statistics_rows:
            log_message("No statistics data to export", "WARNING")
            return
        
        df = pd.DataFrame(statistics_rows)
        
        save_dir = filedialog.askdirectory(title='Select directory to save statistics CSV')
        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_animal_drug_{analysis_type}_statistics_{timestamp}.csv"
            save_path = os.path.join(save_dir, filename)
            df.to_csv(save_path, index=False)
            log_message(f"Statistics exported to {save_path} ({len(df)} rows)")
            
    def add_row(self):
        """Add a new row to the table"""
        self.num_rows += 1
        self.row_headers[self.num_rows - 1] = f"Day{self.num_rows}"
        self.rebuild_table()
        log_message(f"Added row: Day{self.num_rows}")

    def remove_row(self):
        """Remove the last row"""
        if self.num_rows <= 1:
            log_message("Cannot remove the last row", "WARNING")
            return
        
        last_row = self.num_rows - 1
        for j in range(self.num_cols):
            if (last_row, j) in self.table_data:
                animal_id = self.table_data[(last_row, j)]
                self.used_animals.discard(animal_id)
                del self.table_data[(last_row, j)]
        
        del self.row_headers[last_row]
        self.num_rows -= 1
        self.rebuild_table()
        log_message(f"Removed row, now {self.num_rows} rows")

    def add_column(self):
        """Add a new column"""
        self.num_cols += 1
        self.col_headers[self.num_cols - 1] = f"Animal{self.num_cols}"
        self.rebuild_table()
        log_message(f"Added column: Animal{self.num_cols}")

    def remove_column(self):
        """Remove the last column"""
        if self.num_cols <= 1:
            log_message("Cannot remove the last column", "WARNING")
            return
        
        last_col = self.num_cols - 1
        for i in range(self.num_rows):
            if (i, last_col) in self.table_data:
                animal_id = self.table_data[(i, last_col)]
                self.used_animals.discard(animal_id)
                del self.table_data[(i, last_col)]
        
        del self.col_headers[last_col]
        self.num_cols -= 1
        self.rebuild_table()
        log_message(f"Removed column, now {self.num_cols} columns")

    def rebuild_table(self):
        """Rebuild the entire table"""
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # Corner cell
        corner = tk.Label(self.table_frame, text="", bg="#bdc3c7", 
                        relief=tk.RAISED, bd=2, width=12, height=2)
        corner.grid(row=0, column=0, sticky="nsew")
        
        # Column headers
        for j in range(self.num_cols):
            header_text = self.col_headers.get(j, f"Animal{j+1}")
            header = tk.Label(self.table_frame, text=header_text, 
                            bg="#ffffff", fg="#000000",
                            font=("Microsoft YaHei", 10, "bold"),
                            relief=tk.RAISED, bd=2, width=12, height=2)
            header.grid(row=0, column=j+1, sticky="nsew")
            header.bind("<Double-Button-1>", lambda e, col=j: self.rename_column(col))
        
        # Row headers and cells
        for i in range(self.num_rows):
            header_text = self.row_headers.get(i, f"Day{i+1}")
            header = tk.Label(self.table_frame, text=header_text,
                            bg="#ffffff", fg="#000000",
                            font=("Microsoft YaHei", 10, "bold"),
                            relief=tk.RAISED, bd=2, width=12, height=2)
            header.grid(row=i+1, column=0, sticky="nsew")
            header.bind("<Double-Button-1>", lambda e, row=i: self.rename_row(row))
            
            for j in range(self.num_cols):
                cell_value = self.table_data.get((i, j), "")
                cell = tk.Label(self.table_frame, text=cell_value,
                            bg="#ecf0f1", relief=tk.SUNKEN, bd=2,
                            width=15, height=3, anchor="center",
                            font=("Microsoft YaHei", 9))
                cell.grid(row=i+1, column=j+1, sticky="nsew", padx=1, pady=1)
                cell.bind("<Button-3>", lambda e, row=i, col=j: self.show_animal_selector(e, row, col))
        
        for i in range(self.num_rows + 1):
            self.table_frame.grid_rowconfigure(i, weight=1)
        for j in range(self.num_cols + 1):
            self.table_frame.grid_columnconfigure(j, weight=1)

    def rename_row(self, row_idx):
        """Rename a row header"""
        current_name = self.row_headers.get(row_idx, f"Day{row_idx+1}")
        
        dialog = tk.Toplevel(self.analysis_window)
        dialog.title("Rename Row")
        dialog.geometry("300x120")
        dialog.transient(self.analysis_window)
        dialog.grab_set()
        
        tk.Label(dialog, text="Enter new row name:", 
                font=("Microsoft YaHei", 10)).pack(pady=10)
        
        entry = tk.Entry(dialog, font=("Microsoft YaHei", 10), width=20)
        entry.insert(0, current_name)
        entry.pack(pady=5)
        entry.focus_set()
        entry.select_range(0, tk.END)
        
        def save_name():
            new_name = entry.get().strip()
            if new_name:
                self.row_headers[row_idx] = new_name
                self.rebuild_table()
                log_message(f"Renamed row {row_idx} to '{new_name}'")
            dialog.destroy()
        
        tk.Button(dialog, text="OK", command=save_name,
                bg="#FFFFFF", fg="#000000", font=("Microsoft YaHei", 9),
                padx=15, pady=5).pack(pady=10)
        
        entry.bind("<Return>", lambda e: save_name())

    def rename_column(self, col_idx):
        """Rename a column header"""
        current_name = self.col_headers.get(col_idx, f"Animal{col_idx+1}")
        
        dialog = tk.Toplevel(self.analysis_window)
        dialog.title("Rename Column")
        dialog.geometry("300x120")
        dialog.transient(self.analysis_window)
        dialog.grab_set()
        
        tk.Label(dialog, text="Enter new column name (ear tag):", 
                font=("Microsoft YaHei", 10)).pack(pady=10)
        
        entry = tk.Entry(dialog, font=("Microsoft YaHei", 10), width=20)
        entry.insert(0, current_name)
        entry.pack(pady=5)
        entry.focus_set()
        entry.select_range(0, tk.END)
        
        def save_name():
            new_name = entry.get().strip()
            if new_name:
                if new_name != current_name:
                    for i in range(self.num_rows):
                        if (i, col_idx) in self.table_data:
                            animal_id = self.table_data[(i, col_idx)]
                            self.used_animals.discard(animal_id)
                            del self.table_data[(i, col_idx)]
                
                self.col_headers[col_idx] = new_name
                self.rebuild_table()
                log_message(f"Renamed column {col_idx} to '{new_name}'")
            dialog.destroy()
        
        tk.Button(dialog, text="OK", command=save_name,
                bg="#FFFFFF", fg="#000000", font=("Microsoft YaHei", 9),
                padx=15, pady=5).pack(pady=10)
        
        entry.bind("<Return>", lambda e: save_name())

    def show_animal_selector(self, event, row, col):
        """Show animal selection menu"""
        col_header = self.col_headers.get(col, f"Animal{col+1}")
        is_custom_header = not col_header.startswith("Animal")
        
        available_animals = []
        for animal_data in self.multi_animal_data:
            animal_id = animal_data.get('animal_id', '')
            
            if is_custom_header:
                ear_tag = animal_id.split('-')[-1] if '-' in animal_id else ''
                if ear_tag == col_header and animal_id not in self.used_animals:
                    available_animals.append(animal_id)
            else:
                if animal_id not in self.used_animals:
                    available_animals.append(animal_id)
        
        if not available_animals:
            log_message("No available animals to select", "INFO")
            return
        
        menu = tk.Menu(self.analysis_window, tearoff=0)
        
        if (row, col) in self.table_data:
            menu.add_command(label="Clear", 
                        command=lambda: self.clear_cell(row, col))
            menu.add_separator()
        
        for animal_id in available_animals:
            menu.add_command(label=animal_id,
                        command=lambda aid=animal_id: self.select_animal(row, col, aid))
        
        menu.post(event.x_root, event.y_root)

    def select_animal(self, row, col, animal_id):
        """Select an animal for a cell"""
        if (row, col) in self.table_data:
            old_id = self.table_data[(row, col)]
            self.used_animals.discard(old_id)
        
        self.table_data[(row, col)] = animal_id
        self.used_animals.add(animal_id)
        
        self.rebuild_table()
        log_message(f"Selected {animal_id} for row {row}, col {col}")

    def clear_cell(self, row, col):
        """Clear a cell"""
        if (row, col) in self.table_data:
            animal_id = self.table_data[(row, col)]
            self.used_animals.discard(animal_id)
            del self.table_data[(row, col)]
            self.rebuild_table()
            log_message(f"Cleared cell at row {row}, col {col}")