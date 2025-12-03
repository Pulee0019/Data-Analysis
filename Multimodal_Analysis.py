import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import os
import numpy as np
import pandas as pd
from datetime import datetime
from logger import log_message

class MultimodalAnalysis:
    def __init__(self, root, multi_animal_data=None, current_animal_index=0, selected_bodyparts=None):
        self.root = root
        self.multi_animal_data = multi_animal_data
        self.current_animal_index = current_animal_index
        self.selected_bodyparts = selected_bodyparts if selected_bodyparts is not None else set()
        
        # Color configuration - consistent with TrajectoryPointCloudManager
        self.colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                      '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#95a5a6']
    
    def get_current_animal_data(self):
        """Get current selected animal data"""
        if self.multi_animal_data and self.current_animal_index < len(self.multi_animal_data):
            return self.multi_animal_data[self.current_animal_index]
        return None
    
    def running_event_activity_analysis(self, event_type):
        """
        Analyze specific running events (General Onsets, Jerks, etc.)
        event_type: string key in treadmill_behaviors (e.g., 'general_onsets', 'jerks', 'locomotion_initiations')
        """
        animal_data = self.get_current_animal_data()
        if not animal_data:
            log_message("No animal data available", "ERROR")
            return
            
        # Check required data
        required_data = ['treadmill_behaviors', 'dff_data', 'ast2_data_adjusted']
        missing_data = [data for data in required_data if data not in animal_data or animal_data[data] is None]
        if missing_data:
            log_message(f"Missing necessary data: {', '.join(missing_data)}", "ERROR")
            return
        
        # Format event name for display
        event_name_display = event_type.replace('_', ' ').title()
        
        # Create parameter setting window with UI styling
        param_window = tk.Toplevel(self.root)
        param_window.title(f"{event_name_display} Analysis Parameters")
        param_window.geometry("400x500")
        param_window.configure(bg='#f8f8f8')
        param_window.transient(self.root)
        param_window.grab_set()
        
        # Title
        title_label = tk.Label(param_window, text=f"{event_name_display} Analysis Settings", 
                            font=("Microsoft YaHei", 12, "bold"), bg="#f8f8f8", fg="#2c3e50")
        title_label.pack(pady=15)
        
        # Time window selection frame
        time_frame = tk.LabelFrame(param_window, text="Time Window Settings", 
                                font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        time_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Pre-time setting
        pre_frame = tk.Frame(time_frame, bg="#f8f8f8")
        pre_frame.pack(pady=5)
        tk.Label(pre_frame, text="Pre Time (s):", bg="#f8f8f8", 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        pre_time_var = tk.StringVar(value="30")
        pre_time_entry = tk.Entry(pre_frame, textvariable=pre_time_var, width=10, 
                                font=("Microsoft YaHei", 8))
        pre_time_entry.pack(side=tk.LEFT, padx=5)
        
        # Post-time setting
        post_frame = tk.Frame(time_frame, bg="#f8f8f8")
        post_frame.pack(pady=5)
        tk.Label(post_frame, text="Post Time (s):", bg="#f8f8f8", 
                font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        post_time_var = tk.StringVar(value="60")
        post_time_entry = tk.Entry(post_frame, textvariable=post_time_var, width=10, 
                                font=("Microsoft YaHei", 8))
        post_time_entry.pack(side=tk.LEFT, padx=5)
        
        # Channel selection frame
        channel_frame = tk.LabelFrame(param_window, text="Fiber Channel Selection", 
                                    font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        channel_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Get available channels
        available_channels = []
        if 'active_channels' in animal_data and animal_data['active_channels']:
            available_channels = [str(ch) for ch in animal_data['active_channels']]
        else:
            available_channels = ["1"]  # Default fallback
        
        # Create listbox for multi-selection with styling
        tk.Label(channel_frame, text="Select Channels:", bg="#f8f8f8", 
                font=("Microsoft YaHei", 8)).pack(anchor=tk.W, pady=(5,2))
        
        channel_listbox = tk.Listbox(channel_frame, selectmode=tk.MULTIPLE, height=4, 
                                    width=20, font=("Microsoft YaHei", 8))
        for channel in available_channels:
            channel_listbox.insert(tk.END, f"Channel {channel}")
        channel_listbox.pack(pady=5, padx=10, fill=tk.X)
        
        # Select first channel by default
        if available_channels:
            channel_listbox.selection_set(0)
        
        # Export option
        export_frame = tk.LabelFrame(param_window, text="Export Options", 
                                    font=("Microsoft YaHei", 9, "bold"), bg="#f8f8f8")
        export_frame.pack(fill=tk.X, padx=20, pady=10)
        
        export_var = tk.BooleanVar(value=False)
        export_check = tk.Checkbutton(export_frame, text="Export statistic results to CSV", 
                                    variable=export_var, bg="#f8f8f8",
                                    font=("Microsoft YaHei", 8))
        export_check.pack(anchor=tk.W, padx=10, pady=5)
        
        def run_analysis():
            try:
                pre_time = float(pre_time_var.get())
                post_time = float(post_time_var.get())
                
                # Get selected channels
                selected_indices = channel_listbox.curselection()
                if not selected_indices:
                    log_message("Please select at least one channel", "WARNING")
                    return
                
                selected_channels = [available_channels[i] for i in selected_indices]
                
                if pre_time <= 0 or post_time <= 0:
                    log_message("Time must be positive numbers", "WARNING")
                    return
                    
                param_window.destroy()
                self._plot_event_activity_analysis(animal_data, pre_time, post_time, 
                                                selected_channels, event_type, export_var.get())
                
            except ValueError:
                log_message("Please enter valid time values", "WARNING")
        
        # Action buttons frame
        button_frame = tk.Frame(param_window, bg="#f8f8f8")
        button_frame.pack(pady=15)
        
        run_btn = tk.Button(button_frame, text="Start Analysis", command=run_analysis,
                        bg="#3498db", fg="white", font=("Microsoft YaHei", 9, "bold"),
                        relief=tk.FLAT, padx=15, pady=5)
        run_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", 
                            command=param_window.destroy,
                            bg="#95a5a6", fg="white", font=("Microsoft YaHei", 9),
                            relief=tk.FLAT, padx=15, pady=5)
        cancel_btn.pack(side=tk.LEFT, padx=10)
    
    def _calculate_zscore_around_events(self, events, fiber_timestamps, dff_data, pre_time, post_time, target_signal):
        """Calculate z-score around events - supports combined wavelengths"""
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        all_zscore_episodes = {}
        
        # Parse target signal
        target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
        
        # Initialize storage for each wavelength
        for wavelength in target_wavelengths:
            all_zscore_episodes[wavelength] = []
        
        for event_time in events:
            baseline_start = event_time - pre_time
            baseline_end = event_time
            
            baseline_start_idx = np.argmin(np.abs(fiber_timestamps - baseline_start))
            baseline_end_idx = np.argmin(np.abs(fiber_timestamps - baseline_end))
            
            if baseline_end_idx > baseline_start_idx:
                for wavelength in target_wavelengths:
                    # Get data for this specific wavelength
                    if isinstance(dff_data, dict):
                        # Find matching key
                        wavelength_data = None
                        for key, data in dff_data.items():
                            if wavelength in key:
                                wavelength_data = data.values if isinstance(data, pd.Series) else data
                                break
                        if wavelength_data is None:
                            continue
                    else:
                        wavelength_data = dff_data
                    
                    baseline_data = wavelength_data[baseline_start_idx:baseline_end_idx]
                    mean_dff = np.nanmean(baseline_data)
                    std_dff = np.nanstd(baseline_data)
                    
                    if std_dff == 0:
                        std_dff = 1e-10
                    
                    start_idx = np.argmin(np.abs(fiber_timestamps - (event_time - pre_time)))
                    end_idx = np.argmin(np.abs(fiber_timestamps - (event_time + post_time)))
                    
                    if end_idx > start_idx:
                        episode_data = wavelength_data[start_idx:end_idx]
                        episode_times = fiber_timestamps[start_idx:end_idx] - event_time
                        
                        if len(episode_times) > 1:
                            zscore_episode = (episode_data - mean_dff) / std_dff
                            interp_data = np.interp(time_array, episode_times, zscore_episode)
                            all_zscore_episodes[wavelength].append(interp_data)
        
        return time_array, all_zscore_episodes
    
    def _plot_event_activity_analysis(self, animal_data, pre_time, post_time, selected_channels, event_type, export_statistics=False):
        """Plot analysis results for the selected event type - supports combined wavelengths"""
        # Get data
        animal_id = animal_data.get('animal_id', 'Unknown')
        treadmill_behaviors = animal_data['treadmill_behaviors']
        dff_data = animal_data['dff_data']
        ast2_data = animal_data['ast2_data_adjusted']
        target_signal = animal_data.get('target_signal', '470')
        
        events = treadmill_behaviors.get(event_type, [])
        running_timestamps = ast2_data['data']['timestamps']
        processed_data = animal_data.get('running_processed_data')
        running_speed = processed_data['filtered_speed'] if processed_data else ast2_data['data']['speed']
        
        event_name_display = event_type.replace('_', ' ').title()

        # Get fiber timestamps
        preprocessed_data = animal_data['preprocessed_data']
        channels = animal_data.get('channels', {})
        time_col = channels['time']
        fiber_timestamps = preprocessed_data[time_col]
        
        # Parse target signal
        target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
        
        # Process dff_data to organize by wavelength and channel
        channel_wavelength_dff_data = {}
        for channel in selected_channels:
            channel_wavelength_dff_data[channel] = {}
            for wavelength in target_wavelengths:
                key = f"{channel}_{wavelength}"
                if isinstance(dff_data, dict) and key in dff_data:
                    data = dff_data[key]
                    if isinstance(data, pd.Series):
                        data = data.values
                    channel_wavelength_dff_data[channel][wavelength] = data
        
        if not channel_wavelength_dff_data:
            log_message("No valid dFF data found", "ERROR")
            return
        
        if len(events) == 0:
            log_message(f"No {event_name_display} events found", "INFO")
            return
        
        # Calculate z-score episodes for each channel and wavelength
        all_channel_zscore_episodes = {}
        for channel in selected_channels:
            wavelength_dff_data = channel_wavelength_dff_data[channel]
            if not wavelength_dff_data:
                continue
                
            time_array, zscore_episodes_dict = self._calculate_zscore_around_events(
                events, fiber_timestamps, wavelength_dff_data, pre_time, post_time, target_signal)
            all_channel_zscore_episodes[channel] = zscore_episodes_dict
        
        # Export statistics if requested
        if export_statistics:
            self._export_event_statistics(animal_id, event_type, pre_time, post_time,
                                        selected_channels, target_wavelengths,
                                        events, running_timestamps, running_speed,
                                        all_channel_zscore_episodes, time_array)
        
        # Calculate average across channels for plotting
        avg_zscore_episodes_dict = {}
        for wavelength in target_wavelengths:
            all_episodes = []
            for channel in selected_channels:
                if channel in all_channel_zscore_episodes and wavelength in all_channel_zscore_episodes[channel]:
                    all_episodes.extend(all_channel_zscore_episodes[channel][wavelength])
            
            if all_episodes:
                avg_zscore_episodes_dict[wavelength] = all_episodes
        
        # Create result window
        result_window = tk.Toplevel(self.root)
        channel_label = "+".join(selected_channels)
        result_window.title(f"{event_name_display} Analysis - Animal {animal_id} - Channels {channel_label} - {target_signal} nm")
        result_window.state('zoomed')
        result_window.configure(bg='#f8f8f8')
        
        # Determine number of subplot rows needed
        num_wavelengths = len(target_wavelengths)
        num_cols = 1 + num_wavelengths  # 1 for running + each wavelength

        # Create matplotlib figure
        fig = Figure(figsize=(4 * num_cols, 8), dpi=100)

        plot_idx = 1

        # === Row 1: Traces (running + fiber z-score traces) ===
        # 1. Running trace
        ax1 = fig.add_subplot(2, num_cols, plot_idx)
        self._plot_running_around_events(ax1, events, running_timestamps, running_speed,
                                        pre_time, post_time, f"Running Speed Around {event_name_display}")
        plot_idx += 1

        # 2-N. Fiber z-score traces for each wavelength
        fiber_colors = ['#008000', "#FF0000", '#FFA500']
        for wl_idx, wavelength in enumerate(target_wavelengths):
            color = fiber_colors[wl_idx % len(fiber_colors)]
            ax_trace = fig.add_subplot(2, num_cols, plot_idx)
            zscore_episodes = avg_zscore_episodes_dict.get(wavelength, [])
            self._plot_fiber_zscore_around_events(
                ax_trace, time_array, zscore_episodes,
                f"Fiber Z-score {wavelength}nm - CH{channel_label}",
                color=color
            )
            plot_idx += 1

        # === Row 2: Heatmaps (running + fiber z-score heatmaps) ===
        # 1. Running heatmap
        ax2 = fig.add_subplot(2, num_cols, plot_idx)
        self._plot_running_heatmap(ax2, events, running_timestamps, running_speed,
                                pre_time, post_time, "Running Speed Heatmap")
        plot_idx += 1

        # 2-N. Fiber z-score heatmaps for each wavelength
        for wl_idx, wavelength in enumerate(target_wavelengths):
            ax_heatmap = fig.add_subplot(2, num_cols, plot_idx)
            self._plot_fiber_zscore_heatmap(
                ax_heatmap, avg_zscore_episodes_dict.get(wavelength, []), time_array,
                f"Fiber Z-score Heatmap {wavelength}nm - CH{channel_label}"
            )
            plot_idx += 1

        fig.tight_layout()

        # Add canvas to window
        canvas_frame = tk.Frame(result_window, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))
        self.toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        
        log_message(f"{event_name_display} analysis completed: {len(events)} events, "
                f"channels {channel_label}, wavelengths {target_signal}, time window [-{pre_time},{post_time}]s")
    
    def _export_event_statistics(self, animal_id, event_type, pre_time, post_time,
                           selected_channels, target_wavelengths,
                           events, running_timestamps, running_speed,
                           all_channel_zscore_episodes, time_array):
        """Export statistical results for event analysis"""
        # Prepare data for CSV
        rows = []
        
        # Process running data (not channel specific)
        for trial_idx, event_time in enumerate(events):
            # Find indices for time windows
            pre_mask = (time_array >= -pre_time) & (time_array <= 0)
            post_mask = (time_array >= 0) & (time_array <= post_time)
            
            # Get running speed for this event
            start_idx = np.argmin(np.abs(running_timestamps - (event_time - pre_time)))
            end_idx = np.argmin(np.abs(running_timestamps - (event_time + post_time)))
            
            if end_idx > start_idx:
                episode_data = running_speed[start_idx:end_idx]
                episode_times = running_timestamps[start_idx:end_idx] - event_time
                
                if len(episode_times) > 1:
                    interp_data = np.interp(time_array, episode_times, episode_data)
                    
                    # Calculate statistics for pre-window
                    pre_data = interp_data[pre_mask]
                    pre_min = np.min(pre_data) if len(pre_data) > 0 else np.nan
                    pre_max = np.max(pre_data) if len(pre_data) > 0 else np.nan
                    pre_mean = np.mean(pre_data) if len(pre_data) > 0 else np.nan
                    pre_area = np.trapz(pre_data, time_array[pre_mask]) if len(pre_data) > 0 else np.nan
                    
                    # Calculate statistics for post-window
                    post_data = interp_data[post_mask]
                    post_min = np.min(post_data) if len(post_data) > 0 else np.nan
                    post_max = np.max(post_data) if len(post_data) > 0 else np.nan
                    post_mean = np.mean(post_data) if len(post_data) > 0 else np.nan
                    post_area = np.trapz(post_data, time_array[post_mask]) if len(post_data) > 0 else np.nan
                    
                    rows.append({
                        'animal_id': animal_id,
                        'event_type': event_type,
                        'channel': 'N/A',
                        'wavelength': 'N/A',
                        'trial': trial_idx + 1,
                        'pre_min': pre_min,
                        'pre_max': pre_max,
                        'pre_mean': pre_mean,
                        'pre_area': pre_area,
                        'post_min': post_min,
                        'post_max': post_max,
                        'post_mean': post_mean,
                        'post_area': post_area,
                        'signal_type': 'running_speed'
                    })
        
        # Process fiber data for each channel and wavelength
        for channel in selected_channels:
            if channel not in all_channel_zscore_episodes:
                continue
                
            for wavelength in target_wavelengths:
                if wavelength not in all_channel_zscore_episodes[channel]:
                    continue
                    
                zscore_episodes = all_channel_zscore_episodes[channel][wavelength]
                
                for trial_idx, zscore_data in enumerate(zscore_episodes):
                    if trial_idx >= len(events):
                        break
                        
                    # Calculate statistics for pre-window
                    pre_mask = (time_array >= -pre_time) & (time_array <= 0)
                    pre_data = zscore_data[pre_mask]
                    pre_min = np.min(pre_data) if len(pre_data) > 0 else np.nan
                    pre_max = np.max(pre_data) if len(pre_data) > 0 else np.nan
                    pre_mean = np.mean(pre_data) if len(pre_data) > 0 else np.nan
                    pre_area = np.trapz(pre_data, time_array[pre_mask]) if len(pre_data) > 0 else np.nan
                    
                    # Calculate statistics for post-window
                    post_mask = (time_array >= 0) & (time_array <= post_time)
                    post_data = zscore_data[post_mask]
                    post_min = np.min(post_data) if len(post_data) > 0 else np.nan
                    post_max = np.max(post_data) if len(post_data) > 0 else np.nan
                    post_mean = np.mean(post_data) if len(post_data) > 0 else np.nan
                    post_area = np.trapz(post_data, time_array[post_mask]) if len(post_data) > 0 else np.nan
                    
                    rows.append({
                        'animal_id': animal_id,
                        'event_type': event_type,
                        'channel': channel,
                        'wavelength': wavelength,
                        'trial': trial_idx + 1,
                        'pre_min': pre_min,
                        'pre_max': pre_max,
                        'pre_mean': pre_mean,
                        'pre_area': pre_area,
                        'post_min': post_min,
                        'post_max': post_max,
                        'post_mean': post_mean,
                        'post_area': post_area,
                        'signal_type': 'fiber_zscore'
                    })
        
        # Create DataFrame and save to CSV
        if rows:
            df = pd.DataFrame(rows)

            save_dir = filedialog.askdirectory(title='Please select directory to save statistics CSV')
            
            # Save to CSV
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{event_type}_statistics_{timestamp}.csv"
            save_path = os.path.join(save_dir, filename)
            df.to_csv(save_path, index=False)
            
            log_message(f"Statistics exported to {save_path} ({len(df)} rows)")
        else:
            log_message("No data to export", "WARNING")
            
    def _plot_running_around_events(self, ax, events, timestamps, speed, pre_time, post_time, title):
        """Plot running speed around events (mean±std)"""
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        all_episodes = []
        
        for event in events:
            start_idx = np.argmin(np.abs(timestamps - (event - pre_time)))
            end_idx = np.argmin(np.abs(timestamps - (event + post_time)))
            
            if end_idx > start_idx:
                episode_data = speed[start_idx:end_idx]
                # Interpolate to standard time axis
                episode_times = timestamps[start_idx:end_idx] - event
                if len(episode_times) > 1:
                    interp_data = np.interp(time_array, episode_times, episode_data)
                    all_episodes.append(interp_data)
        
        if all_episodes:
            all_episodes = np.array(all_episodes)
            mean_response = np.nanmean(all_episodes, axis=0)
            std_response = np.nanstd(all_episodes, axis=0)
            
            ax.plot(time_array, mean_response, '#000000', linestyle='-', linewidth=2, label='Mean')
            ax.fill_between(time_array, mean_response - std_response, 
                          mean_response + std_response, color='#000000', alpha=0.3, label='Mean ± STD')
            ax.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Event')
            ax.axhline(y=0, color='#808080', linestyle='--', alpha=0.8, label='Baseline')
            
        ax.set_xlim(-pre_time, post_time)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Running Speed (cm/s)')
        ax.set_title(title)
        ax.legend()
        ax.grid(False)
    
    def _plot_fiber_zscore_around_events(self, ax, time_array, zscore_episodes, title, color="#008000"):
        """Plot fiber z-score around events (mean±std) - supports custom color"""
        if zscore_episodes:
            all_episodes = np.array(zscore_episodes)
            mean_response = np.nanmean(all_episodes, axis=0)
            std_response = np.nanstd(all_episodes, axis=0)
            
            ax.plot(time_array, mean_response, color, linestyle='-', linewidth=2, label='Mean')
            ax.fill_between(time_array, mean_response - std_response,
                        mean_response + std_response, color=color, alpha=0.3, label='Mean ± STD')
            ax.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Event')
            ax.axhline(y=0, color='#808080', linestyle='--', alpha=0.8, label='Baseline')
            
        ax.set_xlim(time_array[0], time_array[-1])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-score')
        ax.set_title(title)
        ax.legend()
        ax.grid(False)
    
    def _plot_running_heatmap(self, ax, events, timestamps, speed, pre_time, post_time, title):
        """Plot running speed heatmap"""
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        all_episodes = []
        
        for i, event in enumerate(events):
            start_idx = np.argmin(np.abs(timestamps - (event - pre_time)))
            end_idx = np.argmin(np.abs(timestamps - (event + post_time)))
            
            if end_idx > start_idx:
                episode_data = speed[start_idx:end_idx]
                episode_times = timestamps[start_idx:end_idx] - event
                if len(episode_times) > 1:
                    interp_data = np.interp(time_array, episode_times, episode_data)
                    all_episodes.append(interp_data)
        
        if all_episodes:
            all_episodes = np.array(all_episodes)
            im = ax.imshow(all_episodes, aspect='auto', extent=[-pre_time, post_time, len(events), 1], 
                         cmap='viridis', origin='lower')
            ax.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial Number')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='Running Speed (cm/s)', orientation='horizontal')
    
    def _plot_fiber_zscore_heatmap(self, ax, zscore_episodes, time_array, title):
        """Plot fiber z-score heatmap using pre-calculated z-score episodes"""
        if zscore_episodes:
            all_episodes = np.array(zscore_episodes)
            im = ax.imshow(all_episodes,
               aspect='auto',
               extent=[time_array[0], time_array[-1],
                       len(zscore_episodes), 1],
               cmap='coolwarm',
               origin='lower')
            ax.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial Number')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='Z-score', orientation='horizontal')
    
    def running_event_trajectory_analysis(self, period_type):
        """
        Analyze trajectories during specific running periods
        period_type: string key in treadmill_behaviors (e.g., 'movement_periods', 'rest_periods', 'continuous_locomotion_periods')
        """
        animal_data = self.get_current_animal_data()
        if not animal_data:
            log_message("No animal data available", "ERROR")
            return
            
        # Check necessary data
        if 'treadmill_behaviors' not in animal_data or animal_data['treadmill_behaviors'] is None:
            log_message("No running behavior data, please run running data analysis first", "ERROR")
            return
            
        if 'dlc_data' not in animal_data or animal_data['dlc_data'] is None:
            log_message("No DLC data", "ERROR")
            return
        
        periods = animal_data['treadmill_behaviors'].get(period_type, [])
        period_name_display = period_type.replace('_', ' ').title()

        if len(periods) == 0:
            log_message(f"No {period_name_display} found", "INFO")
            return
        
        # Use bodyparts selected in main UI
        if not self.selected_bodyparts:
            log_message("No bodyparts selected, please select bodyparts in the main UI", "WARNING")
            return
        
        selected_bodyparts = list(self.selected_bodyparts)
        self._plot_period_trajectories(animal_data, periods, selected_bodyparts, period_name_display)
    
    def _plot_period_trajectories(self, animal_data, periods, selected_bodyparts, period_name_display):
        """Plot trajectory point clouds during specified periods - All bodyparts on single plot"""
        dlc_data = animal_data['dlc_data']
        video_fps = animal_data.get('video_fps', 30)
        
        # Create result window with UI styling
        result_window = tk.Toplevel(self.root)
        result_window.title(f"{period_name_display} Trajectory Analysis")
        result_window.geometry("1200x900")
        result_window.configure(bg='#f8f8f8')

        # Create matplotlib figure with UI styling
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Set graph properties - UI style
        ax.set_title(f"{period_name_display} Trajectories", fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
        ax.set_xlabel("X Coordinate", fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_ylabel("Y Coordinate", fontsize=12, fontweight='bold', color='#2c3e50')
        ax.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')
        ax.set_facecolor('#ffffff')
        
        # Store all trajectory points for coordinate range calculation
        all_x_points = []
        all_y_points = []
        
        # Collect and plot all bodyparts on the same plot
        for i, bodypart in enumerate(selected_bodyparts):
            # Get color for this bodypart
            color = self.colors[i % len(self.colors)]
            
            # Collect trajectory points for this bodypart across all periods
            bodypart_x = []
            bodypart_y = []
            
            for j, (start_time, end_time) in enumerate(periods):
                # Convert time to frame indices
                start_frame = int(start_time * video_fps)
                end_frame = int(end_time * video_fps)
                
                # Ensure frame indices are within valid range
                start_frame = max(0, min(start_frame, len(dlc_data[bodypart]['x']) - 1))
                end_frame = max(0, min(end_frame, len(dlc_data[bodypart]['x']) - 1))
                
                if end_frame > start_frame:
                    x_data = dlc_data[bodypart]['x'][start_frame:end_frame]
                    y_data = dlc_data[bodypart]['y'][start_frame:end_frame]
                    
                    # Filter out invalid points (low likelihood or abnormal coordinates)
                    likelihood = dlc_data[bodypart]['likelihood'][start_frame:end_frame]
                    valid_mask = (likelihood > 0.8) & (x_data > 0) & (y_data > 0)
                    
                    valid_x = x_data[valid_mask]
                    valid_y = y_data[valid_mask]
                    
                    if len(valid_x) > 0:
                        # 2x downsampling for performance
                        step = 2
                        x_sampled = valid_x[::step]
                        y_sampled = valid_y[::step]
                        
                        # Plot trajectory lines - light gray thin lines
                        ax.plot(x_sampled, y_sampled, color='lightgray', linewidth=0.5, alpha=0.3)
                        
                        # Plot point cloud - using bodypart-specific colors
                        ax.scatter(x_sampled, y_sampled, s=8, alpha=0.7, 
                                 color=color, edgecolors='white', linewidth=0.3,
                                 label=f'{bodypart} ({len(x_sampled)} points)' if j == 0 else "")
                        
                        bodypart_x.extend(x_sampled)
                        bodypart_y.extend(y_sampled)
            
            # Collect all points for overall range calculation
            if bodypart_x and bodypart_y:
                all_x_points.extend(bodypart_x)
                all_y_points.extend(bodypart_y)
        
        # Set coordinate range for the plot
        if all_x_points and all_y_points:
            margin_x = (max(all_x_points) - min(all_x_points)) * 0.1
            margin_y = (max(all_y_points) - min(all_y_points)) * 0.1
            ax.set_xlim(min(all_x_points) - margin_x, max(all_x_points) + margin_x)
            ax.set_ylim(min(all_y_points) - margin_y, max(all_y_points) + margin_y)
        
        # Set axis style
        ax.tick_params(colors='#2c3e50', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1)
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        
        # Add canvas to window
        canvas_frame = tk.Frame(result_window, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))

        self.toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        
        log_message(f"{period_name_display} trajectory analysis completed: {len(periods)} periods, {len(selected_bodyparts)} bodyparts")

class AcrossdayAnalysis:
    def __init__(self, root, multi_animal_data=None):
        self.root = root
        self.multi_animal_data = multi_animal_data or []
        self.analysis_window = None
        self.table_data = {}  # {(row, col): animal_id}
        self.row_headers = {}  # {row_idx: header_text}
        self.col_headers = {}  # {col_idx: header_text}
        self.used_animals = set()  # Track used animal_ids
        self.num_rows = 1
        self.num_cols = 6
        self.target_event_type = None # Stores the current analysis type
        
        # Colors for different days
        self.day_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                          '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#95a5a6']
        
    def show_config_window(self, event_type):
        """
        Show configuration window for specific event type
        event_type: 'general_onsets', 'jerks', 'locomotion_initiations', 'locomotion_terminations'
        """
        self.target_event_type = event_type
        event_name_display = event_type.replace('_', ' ').title()
        
        # Create or focus window
        if self.analysis_window is None or not self.analysis_window.winfo_exists():
            self.create_analysis_window()
        
        self.analysis_window.title(f"Acrossday Analysis Configuration - {event_name_display}")
        self.analysis_window.deiconify()
        self.analysis_window.lift()

    def create_analysis_window(self):
        """Create the acrossday analysis configuration window"""
        self.analysis_window = tk.Toplevel(self.root)
        # Title will be set in show_config_window
        self.analysis_window.geometry("670x400")
        self.analysis_window.transient(self.root)
        
        # Main container
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
        
        tk.Button(control_frame, text="Run Analysis", command=self.run_analysis,
                 bg="#ffffff", fg="#000000", font=("Microsoft YaHei", 9, "bold"),
                 relief=tk.FLAT, padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        # Table container with scrollbars
        table_container = tk.Frame(main_frame, bg="#ffffff")
        table_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for scrolling
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
        
        # Initialize table
        self.initialize_table()
        self.update_scroll_region()
        
    def update_scroll_region(self):
        """Update scroll region"""
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def initialize_table(self):
        """Initialize the table with default values"""
        # Initialize headers
        for i in range(self.num_rows):
            self.row_headers[i] = f"Day{i+1}"
        for j in range(self.num_cols):
            self.col_headers[j] = f"Animal{j+1}"
        
        self.rebuild_table()
    
    def rebuild_table(self):
        """Rebuild the entire table"""
        # Clear existing widgets
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # Create corner cell
        corner = tk.Label(self.table_frame, text="", bg="#bdc3c7", 
                         relief=tk.RAISED, bd=2, width=12, height=2)
        corner.grid(row=0, column=0, sticky="nsew")
        
        # Create column headers
        for j in range(self.num_cols):
            header_text = self.col_headers.get(j, f"Animal{j+1}")
            header = tk.Label(self.table_frame, text=header_text, 
                            bg="#ffffff", fg="#000000",
                            font=("Microsoft YaHei", 10, "bold"),
                            relief=tk.RAISED, bd=2, width=12, height=2)
            header.grid(row=0, column=j+1, sticky="nsew")
            header.bind("<Double-Button-1>", lambda e, col=j: self.rename_column(col))
        
        # Create row headers and cells
        for i in range(self.num_rows):
            # Row header
            header_text = self.row_headers.get(i, f"Day{i+1}")
            header = tk.Label(self.table_frame, text=header_text,
                            bg="#ffffff", fg="#000000",
                            font=("Microsoft YaHei", 10, "bold"),
                            relief=tk.RAISED, bd=2, width=12, height=2)
            header.grid(row=i+1, column=0, sticky="nsew")
            header.bind("<Double-Button-1>", lambda e, row=i: self.rename_row(row))
            
            # Data cells
            for j in range(self.num_cols):
                cell_value = self.table_data.get((i, j), "")
                cell = tk.Label(self.table_frame, text=cell_value,
                              bg="#ecf0f1", relief=tk.SUNKEN, bd=2,
                              width=15, height=3, anchor="center",
                              font=("Microsoft YaHei", 9))
                cell.grid(row=i+1, column=j+1, sticky="nsew", padx=1, pady=1)
                cell.bind("<Button-3>", lambda e, row=i, col=j: self.show_animal_selector(e, row, col))
        
        # Configure grid weights for resizing
        for i in range(self.num_rows + 1):
            self.table_frame.grid_rowconfigure(i, weight=1)
        for j in range(self.num_cols + 1):
            self.table_frame.grid_columnconfigure(j, weight=1)
    
    def add_row(self):
        """Add a new row to the table"""
        self.num_rows += 1
        self.row_headers[self.num_rows - 1] = f"Day{self.num_rows}"
        self.rebuild_table()
        self.update_scroll_region()
        log_message(f"Added row: Day{self.num_rows}")
    
    def remove_row(self):
        """Remove the last row from the table"""
        if self.num_rows <= 1:
            log_message("Cannot remove the last row", "WARNING")
            return
        
        # Remove data from last row
        last_row = self.num_rows - 1
        for j in range(self.num_cols):
            if (last_row, j) in self.table_data:
                animal_id = self.table_data[(last_row, j)]
                self.used_animals.discard(animal_id)
                del self.table_data[(last_row, j)]
        
        del self.row_headers[last_row]
        self.num_rows -= 1
        self.rebuild_table()
        self.update_scroll_region()
        log_message(f"Removed row, now {self.num_rows} rows")
    
    def add_column(self):
        """Add a new column to the table"""
        self.num_cols += 1
        self.col_headers[self.num_cols - 1] = f"Animal{self.num_cols}"
        self.rebuild_table()
        self.update_scroll_region()
        log_message(f"Added column: Animal{self.num_cols}")
    
    def remove_column(self):
        """Remove the last column from the table"""
        if self.num_cols <= 1:
            log_message("Cannot remove the last column", "WARNING")
            return
        
        # Remove data from last column
        last_col = self.num_cols - 1
        for i in range(self.num_rows):
            if (i, last_col) in self.table_data:
                animal_id = self.table_data[(i, last_col)]
                self.used_animals.discard(animal_id)
                del self.table_data[(i, last_col)]
        
        del self.col_headers[last_col]
        self.num_cols -= 1
        self.rebuild_table()
        self.update_scroll_region()
        log_message(f"Removed column, now {self.num_cols} columns")
    
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
        self.update_scroll_region()
    
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
                # Clear cells in this column if column name changed
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
        self.update_scroll_region()
    
    def show_animal_selector(self, event, row, col):
        """Show animal selection menu"""
        # Get available animals
        col_header = self.col_headers.get(col, f"Animal{col+1}")
        
        # Check if column header is renamed (not default Animal{N})
        is_custom_header = not col_header.startswith("Animal")
        
        available_animals = []
        for animal_data in self.multi_animal_data:
            animal_id = animal_data.get('animal_id', '')
            
            if is_custom_header:
                # Filter by ear tag
                ear_tag = animal_id.split('-')[-1] if '-' in animal_id else ''
                if ear_tag == col_header and animal_id not in self.used_animals:
                    available_animals.append(animal_id)
            else:
                # Show all unused animals
                if animal_id not in self.used_animals:
                    available_animals.append(animal_id)
        
        if not available_animals:
            log_message("No available animals to select", "INFO")
            return
        
        # Create selection menu
        menu = tk.Menu(self.analysis_window, tearoff=0)
        
        # Add clear option if cell has value
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
        # Remove old selection if exists
        if (row, col) in self.table_data:
            old_id = self.table_data[(row, col)]
            self.used_animals.discard(old_id)
        
        # Add new selection
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
    
    def run_analysis(self):
        """Run acrossday analysis based on target_event_type"""
        # Validate table data
        if not self.table_data:
            log_message("No data in the table to analyze", "WARNING")
            return
        
        if not self.target_event_type:
            log_message("No analysis type selected. Please reopen via menu.", "ERROR")
            return
            
        event_name_display = self.target_event_type.replace('_', ' ').title()
        
        param_window = tk.Toplevel(self.analysis_window)
        param_window.title(f"{event_name_display} Analysis Parameters")
        param_window.geometry("300x250")
        param_window.configure(bg='#f8f8f8')
        param_window.transient(self.analysis_window)
        param_window.grab_set()
        
        # Title
        title_label = tk.Label(param_window, text="Time Window Settings", 
                            font=("Microsoft YaHei", 12, "bold"), bg="#f8f8f8", fg="#2c3e50")
        title_label.pack(pady=15)
        
        # Pre-time setting
        pre_frame = tk.Frame(param_window, bg="#f8f8f8")
        pre_frame.pack(pady=8)
        tk.Label(pre_frame, text="Pre Time (s):", bg="#f8f8f8", 
                font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        pre_time_var = tk.StringVar(value="30")
        pre_time_entry = tk.Entry(pre_frame, textvariable=pre_time_var, width=10, 
                                font=("Microsoft YaHei", 9))
        pre_time_entry.pack(side=tk.LEFT, padx=5)
        
        # Post-time setting
        post_frame = tk.Frame(param_window, bg="#f8f8f8")
        post_frame.pack(pady=8)
        tk.Label(post_frame, text="Post Time (s):", bg="#f8f8f8", 
                font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        post_time_var = tk.StringVar(value="60")
        post_time_entry = tk.Entry(post_frame, textvariable=post_time_var, width=10, 
                                font=("Microsoft YaHei", 9))
        post_time_entry.pack(side=tk.LEFT, padx=5)
        
        # Export option
        export_frame = tk.Frame(param_window, bg="#f8f8f8")
        export_frame.pack(pady=10)
        export_var = tk.BooleanVar(value=False)
        export_check = tk.Checkbutton(export_frame, text="Export statistic results to CSV", 
                                    variable=export_var, bg="#f8f8f8",
                                    font=("Microsoft YaHei", 9))
        export_check.pack()
        
        def start_with_params():
            try:
                pre_time = float(pre_time_var.get())
                post_time = float(post_time_var.get())
                
                if pre_time <= 0 or post_time <= 0:
                    log_message("Time must be positive numbers", "WARNING")
                    return
                    
                param_window.destroy()
                self._run_analysis_with_params(pre_time, post_time, export_var.get())
                
            except ValueError:
                log_message("Please enter valid time values", "WARNING")
        
        # Buttons
        button_frame = tk.Frame(param_window, bg="#f8f8f8")
        button_frame.pack(pady=20)
        
        run_btn = tk.Button(button_frame, text="Start Analysis", command=start_with_params,
                        bg="#3498db", fg="white", font=("Microsoft YaHei", 9, "bold"),
                        relief=tk.FLAT, padx=15, pady=5)
        run_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", 
                            command=param_window.destroy,
                            bg="#95a5a6", fg="white", font=("Microsoft YaHei", 9),
                            relief=tk.FLAT, padx=15, pady=5)
        cancel_btn.pack(side=tk.LEFT, padx=10)

    def _run_analysis_with_params(self, pre_time, post_time, export_statistics=False):
        self.pre_time = pre_time
        self.post_time = post_time
        self.export_statistics = export_statistics
        
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
        
        event_name_display = self.target_event_type.replace('_', ' ').title()
        log_message(f"Starting acrossday {event_name_display} analysis for {len(day_data)} days (pre={pre_time}s, post={post_time}s)...")
        
        # Perform analysis for each day
        results = {}
        all_statistics_rows = []  # Collect all statistics for export
        
        for day_name, animals in day_data.items():
            log_message(f"Analyzing {day_name} with {len(animals)} animals...")
            day_result, day_stats = self.analyze_day(day_name, animals, pre_time, post_time, 
                                                self.target_event_type, export_statistics)
            if day_result:
                results[day_name] = day_result
            if day_stats:
                all_statistics_rows.extend(day_stats)
        
        # Export statistics if requested
        if export_statistics and all_statistics_rows:
            self._export_acrossday_statistics(all_statistics_rows)
        
        if results:
            self.results = results
            self.plot_results(results)
            log_message("Acrossday analysis completed successfully")
        else:
            log_message("Acrossday analysis failed, no valid results", "ERROR")

    def analyze_day(self, day_name, animals, pre_time, post_time, event_type, collect_statistics=False):
        """Analyze data for one day - supports combined wavelengths and dynamic event type"""
        all_running_episodes = []
        all_fiber_episodes = {}  # Organized by wavelength
        statistics_rows = []  # For collecting statistics
        
        # Detect wavelengths from first animal
        target_signal = None
        target_wavelengths = []
        for animal_data in animals:
            if 'target_signal' in animal_data:
                target_signal = animal_data['target_signal']
                target_wavelengths = target_signal.split('+') if '+' in target_signal else [target_signal]
                break
        
        if not target_wavelengths:
            target_wavelengths = ['470']  # Default fallback
        
        # Initialize storage for each wavelength
        for wavelength in target_wavelengths:
            all_fiber_episodes[wavelength] = []
        
        for animal_data in animals:
            try:
                animal_id = animal_data.get('animal_id', 'Unknown')
                
                # Check required data
                if 'treadmill_behaviors' not in animal_data:
                    log_message(f"Skipping {animal_id}: no running behavior data", "WARNING")
                    continue
                
                if 'dff_data' not in animal_data or 'ast2_data_adjusted' not in animal_data:
                    log_message(f"Skipping {animal_id}: missing necessary data", "WARNING")
                    continue
                
                # Get events based on selected type
                events = animal_data['treadmill_behaviors'].get(event_type, [])
                if not events:
                    log_message(f"No {event_type} events for {animal_id}", "INFO")
                    continue
                
                # Get data
                ast2_data = animal_data['ast2_data_adjusted']
                running_timestamps = ast2_data['data']['timestamps']
                processed_data = animal_data.get('running_processed_data')
                running_speed = processed_data['filtered_speed'] if processed_data else ast2_data['data']['speed']
                
                preprocessed_data = animal_data['preprocessed_data']
                channels = animal_data.get('channels', {})
                time_col = channels['time']
                fiber_timestamps = preprocessed_data[time_col].values
                
                # Get dff_data organized by wavelength
                dff_data = animal_data['dff_data']

                # Get available channels
                available_channels = []
                if 'active_channels' in animal_data and animal_data['active_channels']:
                    available_channels = [str(ch) for ch in animal_data['active_channels']]
                else:
                    available_channels = ["1"]  # Default fallback
                    
                # Create time arrays
                time_array_fiber = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
                time_array_running = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
                
                # Process running data for each event
                for trial_idx, event_time in enumerate(events):
                    # Running data
                    start_idx = np.argmin(np.abs(running_timestamps - (event_time - pre_time)))
                    end_idx = np.argmin(np.abs(running_timestamps - (event_time + post_time)))
                    
                    if end_idx > start_idx:
                        episode_data = running_speed[start_idx:end_idx]
                        episode_times = running_timestamps[start_idx:end_idx] - event_time
                        
                        if len(episode_times) > 1:
                            interp_data = np.interp(time_array_running, episode_times, episode_data)
                            all_running_episodes.append(interp_data)
                            
                            # Collect statistics if requested
                            if collect_statistics:
                                pre_mask = (time_array_running >= -pre_time) & (time_array_running <= 0)
                                post_mask = (time_array_running >= 0) & (time_array_running <= post_time)
                                
                                pre_data = interp_data[pre_mask]
                                post_data = interp_data[post_mask]
                                
                                statistics_rows.append({
                                    'day': day_name,
                                    'animal_id': animal_id,
                                    'event_type': event_type,
                                    'channel': 'N/A',
                                    'wavelength': 'N/A',
                                    'trial': trial_idx + 1,
                                    'pre_min': np.min(pre_data) if len(pre_data) > 0 else np.nan,
                                    'pre_max': np.max(pre_data) if len(pre_data) > 0 else np.nan,
                                    'pre_mean': np.mean(pre_data) if len(pre_data) > 0 else np.nan,
                                    'pre_area': np.trapz(pre_data, time_array_running[pre_mask]) if len(pre_data) > 0 else np.nan,
                                    'post_min': np.min(post_data) if len(post_data) > 0 else np.nan,
                                    'post_max': np.max(post_data) if len(post_data) > 0 else np.nan,
                                    'post_mean': np.mean(post_data) if len(post_data) > 0 else np.nan,
                                    'post_area': np.trapz(post_data, time_array_running[post_mask]) if len(post_data) > 0 else np.nan,
                                    'signal_type': 'running_speed'
                                })
                
                # Process fiber data for each channel and wavelength
                for channel in available_channels:
                    for wavelength in target_wavelengths:
                        key = f"{channel}_{wavelength}"
                        if isinstance(dff_data, dict) and key in dff_data:
                            data = dff_data[key]
                            if isinstance(data, pd.Series):
                                data = data.values
                            
                            channel_wavelength_episodes = []
                            
                            for trial_idx, event_time in enumerate(events):
                                baseline_start = event_time - pre_time
                                baseline_end = event_time
                                baseline_start_idx = np.argmin(np.abs(fiber_timestamps - baseline_start))
                                baseline_end_idx = np.argmin(np.abs(fiber_timestamps - baseline_end))
                                
                                if baseline_end_idx > baseline_start_idx:
                                    baseline_data = data[baseline_start_idx:baseline_end_idx]
                                    mean_dff = np.nanmean(baseline_data)
                                    std_dff = np.nanstd(baseline_data)
                                    
                                    if std_dff == 0:
                                        std_dff = 1e-10
                                    
                                    start_idx = np.argmin(np.abs(fiber_timestamps - (event_time - pre_time)))
                                    end_idx = np.argmin(np.abs(fiber_timestamps - (event_time + post_time)))
                                    
                                    if end_idx > start_idx:
                                        episode_data = data[start_idx:end_idx]
                                        episode_times = fiber_timestamps[start_idx:end_idx] - event_time
                                        
                                        if len(episode_times) > 1:
                                            zscore_episode = (episode_data - mean_dff) / std_dff
                                            interp_data = np.interp(time_array_fiber, episode_times, zscore_episode)
                                            channel_wavelength_episodes.append(interp_data)
                                            
                                            # Collect statistics if requested
                                            if collect_statistics:
                                                pre_mask = (time_array_fiber >= -pre_time) & (time_array_fiber <= 0)
                                                post_mask = (time_array_fiber >= 0) & (time_array_fiber <= post_time)
                                                
                                                pre_data = interp_data[pre_mask]
                                                post_data = interp_data[post_mask]
                                                
                                                statistics_rows.append({
                                                    'day': day_name,
                                                    'animal_id': animal_id,
                                                    'event_type': event_type,
                                                    'channel': channel,
                                                    'wavelength': wavelength,
                                                    'trial': trial_idx + 1,
                                                    'pre_min': np.min(pre_data) if len(pre_data) > 0 else np.nan,
                                                    'pre_max': np.max(pre_data) if len(pre_data) > 0 else np.nan,
                                                    'pre_mean': np.mean(pre_data) if len(pre_data) > 0 else np.nan,
                                                    'pre_area': np.trapz(pre_data, time_array_fiber[pre_mask]) if len(pre_data) > 0 else np.nan,
                                                    'post_min': np.min(post_data) if len(post_data) > 0 else np.nan,
                                                    'post_max': np.max(post_data) if len(post_data) > 0 else np.nan,
                                                    'post_mean': np.mean(post_data) if len(post_data) > 0 else np.nan,
                                                    'post_area': np.trapz(post_data, time_array_fiber[post_mask]) if len(post_data) > 0 else np.nan,
                                                    'signal_type': 'fiber_zscore'
                                                })
                            
                            if channel_wavelength_episodes:
                                all_fiber_episodes[wavelength].extend(channel_wavelength_episodes)
                
            except Exception as e:
                log_message(f"Error analyzing {animal_data.get('animal_id', 'Unknown')}: {str(e)}", "ERROR")
                continue
        
        if not all_running_episodes:
            log_message(f"No valid episodes for {day_name}", "WARNING")
            return None, statistics_rows if collect_statistics else None
        
        # Calculate mean and SEM for running
        running_episodes = np.array(all_running_episodes)
        
        # Calculate mean and SEM for each wavelength
        fiber_results = {}
        for wavelength, episodes in all_fiber_episodes.items():
            if episodes:
                episodes_array = np.array(episodes)
                fiber_results[wavelength] = {
                    'time': time_array_fiber,
                    'mean': np.nanmean(episodes_array, axis=0),
                    'sem': np.nanstd(episodes_array, axis=0) / np.sqrt(len(episodes)),
                    'episodes': episodes_array
                }
        
        result = {
            'running': {
                'time': time_array_running,
                'mean': np.nanmean(running_episodes, axis=0),
                'sem': np.nanstd(running_episodes, axis=0) / np.sqrt(len(running_episodes)),
                'episodes': running_episodes
            },
            'fiber': fiber_results,  # Now organized by wavelength
            'target_wavelengths': target_wavelengths
        }
        
        return result, statistics_rows if collect_statistics else None

    def _export_acrossday_statistics(self, statistics_rows):
        """Export acrossday statistics to CSV"""
        if not statistics_rows:
            log_message("No statistics data to export", "WARNING")
            return
        
        # Create DataFrame
        df = pd.DataFrame(statistics_rows)
        
        save_dir = filedialog.askdirectory(title='Please select directory to save statistics CSV')
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"acrossday_{self.target_event_type}_statistics_{timestamp}.csv"
        save_path = os.path.join(save_dir, filename)
        df.to_csv(save_path, index=False)
        
        log_message(f"Acrossday statistics exported to {save_path} ({len(df)} rows)")
        
    def plot_results(self, results):
        """Plot analysis results - All days in one window - supports multiple wavelengths"""
        # Get target wavelengths from first result
        target_wavelengths = []
        for day_name, data in results.items():
            if 'target_wavelengths' in data:
                target_wavelengths = data['target_wavelengths']
                break
        
        if not target_wavelengths:
            target_wavelengths = ['470']  # Fallback
        
        event_name_display = self.target_event_type.replace('_', ' ').title()
        
        # Create result window
        result_window = tk.Toplevel(self.root)
        wavelength_label = '+'.join(target_wavelengths)
        result_window.title(f"Acrossday {event_name_display} Results - All Days ({wavelength_label}nm)")
        result_window.state("zoomed")
        result_window.configure(bg="#ffffff")
        
        # Calculate number of rows needed: 1 running row + N wavelength rows
        num_wavelengths = len(target_wavelengths)
        num_cols = 1 + num_wavelengths  # 1 for running + each wavelength

        # Create figure with subplots: 2 rows, num_cols columns
        fig = Figure(figsize=(4 * num_cols, 10), dpi=100)

        plot_idx = 1

        # === Row 1: All traces ===

        # 1. Running trace
        ax_running_trace = fig.add_subplot(2, num_cols, plot_idx)
        for idx, (day_name, data) in enumerate(results.items()):
            color = self.day_colors[idx % len(self.day_colors)]
            running_data = data['running']
            ax_running_trace.plot(running_data['time'], running_data['mean'],
                                color=color, linewidth=2, label=day_name)
            ax_running_trace.fill_between(running_data['time'],
                                        running_data['mean'] - running_data['sem'],
                                        running_data['mean'] + running_data['sem'],
                                        color=color, alpha=0.3)
        ax_running_trace.axvline(x=0, color='#808080', linestyle='--', alpha=0.8)
        ax_running_trace.axhline(y=0, color='#808080', linestyle='--', alpha=0.8)
        ax_running_trace.set_xlabel('Time (s)')
        ax_running_trace.set_ylabel('Running Speed (cm/s)')
        ax_running_trace.set_xlim(running_data['time'][0], running_data['time'][-1])
        ax_running_trace.set_title('Running Speed - All Days')
        ax_running_trace.legend()
        ax_running_trace.grid(False)
        plot_idx += 1

        # 2-N. Fiber traces for each wavelength
        fiber_colors = ['#008000', "#FF0000", '#FFA500']
        for wl_idx, wavelength in enumerate(target_wavelengths):
            ax_fiber_trace = fig.add_subplot(2, num_cols, plot_idx)
            for idx, (day_name, data) in enumerate(results.items()):
                day_color = self.day_colors[idx % len(self.day_colors)]
                if 'fiber' in data and wavelength in data['fiber']:
                    fiber_data = data['fiber'][wavelength]
                    ax_fiber_trace.plot(fiber_data['time'], fiber_data['mean'],
                                        color=day_color, linewidth=2, label=day_name)
                    ax_fiber_trace.fill_between(fiber_data['time'],
                                                fiber_data['mean'] - fiber_data['sem'],
                                                fiber_data['mean'] + fiber_data['sem'],
                                                color=day_color, alpha=0.3)

            ax_fiber_trace.axvline(x=0, color='#808080', linestyle='--', alpha=0.8)
            ax_fiber_trace.axhline(y=0, color='#808080', linestyle='--', alpha=0.8)
            ax_fiber_trace.set_xlabel('Time (s)')
            ax_fiber_trace.set_ylabel('Z-score')
            ax_fiber_trace.set_xlim(fiber_data['time'][0], fiber_data['time'][-1])
            ax_fiber_trace.set_title(f'Fiber Z-score {wavelength}nm - All Days')
            ax_fiber_trace.legend()
            ax_fiber_trace.grid(False)
            plot_idx += 1

        # === Row 2: All heatmaps ===

        # 1. Running heatmap
        ax_running_heatmap = fig.add_subplot(2, num_cols, plot_idx)
        self.plot_combined_running_heatmap(results, ax_running_heatmap)
        plot_idx += 1

        # 2-N. Fiber heatmaps for each wavelength
        for wl_idx, wavelength in enumerate(target_wavelengths):
            ax_fiber_heatmap = fig.add_subplot(2, num_cols, plot_idx)
            self.plot_combined_fiber_heatmap_by_wavelength(results, ax_fiber_heatmap, wavelength)
            plot_idx += 1

        fig.tight_layout()
                
        # Add canvas
        canvas_frame = tk.Frame(result_window, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))
        self.toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        
        # Create individual day windows
        self.create_individual_day_windows(results)
        
        log_message(f"Results plotted for {len(results)} days with {len(target_wavelengths)} wavelength(s)")

    def plot_combined_running_heatmap(self, results, ax):
        """Plot combined running heatmap for all days"""
        all_running_episodes = []
        for day_name, data in results.items():
            running_episodes = data['running']['episodes']
            all_running_episodes.extend(running_episodes)
        
        if all_running_episodes:
            time_array = list(results.values())[0]['running']['time']
            im = ax.imshow(all_running_episodes, aspect='auto',
                        extent=[time_array[0], time_array[-1], 
                                len(all_running_episodes), 1],
                        cmap='viridis', origin='lower')
            ax.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial (All Days)')
            ax.set_title('Running Heatmap - All Days')
            plt.colorbar(im, ax=ax, label='Speed (cm/s)', orientation='horizontal')
        else:
            ax.text(0.5, 0.5, 'No running data available', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#666666')
            ax.set_title('Running Heatmap - All Days')
            ax.axis('off')

    def plot_combined_fiber_heatmap_by_wavelength(self, results, ax, wavelength):
        """Plot combined fiber heatmap for all days for a specific wavelength"""
        all_fiber_episodes = []
        
        for day_name, data in results.items():
            if 'fiber' in data and wavelength in data['fiber']:
                fiber_data = data['fiber'][wavelength]
                if 'episodes' in fiber_data:
                    episodes = fiber_data['episodes']
                    all_fiber_episodes.extend(episodes)
        
        if all_fiber_episodes:
            all_fiber_episodes = np.array(all_fiber_episodes)
            time_array = list(results.values())[0]['fiber'][wavelength]['time']
            
            im = ax.imshow(all_fiber_episodes, aspect='auto',
                        extent=[time_array[0], time_array[-1],
                                len(all_fiber_episodes), 1],
                        cmap='coolwarm', origin='lower')
            ax.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial (All Days)')
            ax.set_title(f'Fiber Heatmap {wavelength}nm - All Days')
            plt.colorbar(im, ax=ax, label='Z-score', orientation='horizontal')
        else:
            ax.text(0.5, 0.5, f'No fiber data available for {wavelength}nm', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#666666')
            ax.set_title(f'Fiber Heatmap {wavelength}nm - All Days')
            ax.axis('off')

    def create_individual_day_windows(self, results):
        """Create individual windows for each day"""
        for day_name, data in results.items():
            self.create_single_day_window(day_name, data)

    def create_single_day_window(self, day_name, data):
        """Create window for a single day - supports multiple wavelengths"""
        day_window = tk.Toplevel(self.root)
        
        # Get target wavelengths
        target_wavelengths = data.get('target_wavelengths', ['470'])
        wavelength_label = '+'.join(target_wavelengths)
        
        day_window.title(f"Acrossday Analysis - {day_name} ({wavelength_label}nm)")
        day_window.state("zoomed")
        day_window.configure(bg='#f8f8f8')
        
        num_wavelengths = len(target_wavelengths)
        num_cols = 1 + num_wavelengths          # 1 running + N wavelengths
        fig = Figure(figsize=(4 * num_cols, 8), dpi=100)

        day_idx = list(self.results.keys()).index(day_name) if hasattr(self, 'results') else 0
        day_color = self.day_colors[day_idx % len(self.day_colors)]
        fiber_colors = ['#008000', "#FF0000", '#FFA500']

        plot_idx = 1

        # 1. running trace
        ax_rt = fig.add_subplot(2, num_cols, plot_idx)
        running_data = data['running']
        ax_rt.plot(running_data['time'], running_data['mean'],
                color=day_color, linewidth=2)
        ax_rt.fill_between(running_data['time'],
                        running_data['mean'] - running_data['sem'],
                        running_data['mean'] + running_data['sem'],
                        color=day_color, alpha=0.3)
        ax_rt.axvline(0, color='#808080', ls='--', alpha=0.8)
        ax_rt.axhline(0, color='#808080', ls='--', alpha=0.8)
        ax_rt.set_xlabel('Time (s)')
        ax_rt.set_ylabel('Speed (cm/s)')
        ax_rt.set_xlim(running_data['time'][0], running_data['time'][-1])
        ax_rt.set_title(f'{day_name} - Running Trace')
        ax_rt.grid(False)
        plot_idx += 1

        # 2-N. fiber traces
        for wl_idx, wl in enumerate(target_wavelengths):
            ax_ft = fig.add_subplot(2, num_cols, plot_idx)
            if ('fiber' not in data) or (wl not in data['fiber']):
                ax_ft.text(0.5, 0.5, f'No data for {wl}nm',
                        ha='center', va='center', transform=ax_ft.transAxes,
                        fontsize=12, color='#666666')
                ax_ft.set_title(f'{day_name} - Fiber Trace {wl}nm')
                ax_ft.axis('off')
                plot_idx += 1
                continue

            fiber_data = data['fiber'][wl]
            color = fiber_colors[wl_idx % len(fiber_colors)]
            ax_ft.plot(fiber_data['time'], fiber_data['mean'],
                    color=color, linewidth=2)
            ax_ft.fill_between(fiber_data['time'],
                            fiber_data['mean'] - fiber_data['sem'],
                            fiber_data['mean'] + fiber_data['sem'],
                            color=color, alpha=0.3)
            ax_ft.axvline(0, color='#808080', ls='--', alpha=0.8)
            ax_ft.axhline(0, color='#808080', ls='--', alpha=0.8)
            ax_ft.set_xlabel('Time (s)')
            ax_ft.set_ylabel('Z-score')
            ax_ft.set_xlim(fiber_data['time'][0], fiber_data['time'][-1])
            ax_ft.set_title(f'{day_name} - Fiber Trace {wl}nm')
            ax_ft.grid(False)
            plot_idx += 1

        # 1. running heatmap
        ax_rh = fig.add_subplot(2, num_cols, plot_idx)
        running_episodes = data['running']['episodes']
        time_arr = data['running']['time']
        if len(running_episodes):
            im1 = ax_rh.imshow(running_episodes, aspect='auto',
                            extent=[time_arr[0], time_arr[-1],
                                    len(running_episodes), 1],
                            cmap='viridis', origin='lower')
            ax_rh.axvline(0, color='#FF0000', ls='--', alpha=0.8)
            ax_rh.set_xlabel('Time (s)')
            ax_rh.set_ylabel('Trial')
            ax_rh.set_title(f'{day_name} - Running Heatmap')
            fig.colorbar(im1, ax=ax_rh, label='Speed (cm/s)', orientation='horizontal')
        else:
            ax_rh.text(0.5, 0.5, 'No running episodes', ha='center', va='center',
                    transform=ax_rh.transAxes, fontsize=12, color='#666666')
            ax_rh.set_title(f'{day_name} - Running Heatmap')
            ax_rh.axis('off')
        plot_idx += 1

        # 2-N. fiber heatmaps
        for wl_idx, wl in enumerate(target_wavelengths):
            ax_fh = fig.add_subplot(2, num_cols, plot_idx)
            if ('fiber' not in data) or (wl not in data['fiber']):
                ax_fh.text(0.5, 0.5, f'No data for {wl}nm', ha='center', va='center',
                        transform=ax_fh.transAxes, fontsize=12, color='#666666')
                ax_fh.set_title(f'{day_name} - Fiber Heatmap {wl}nm')
                ax_fh.axis('off')
                plot_idx += 1
                continue

            fiber_data = data['fiber'][wl]
            fiber_episodes = fiber_data['episodes']
            time_arr = fiber_data['time']
            if len(fiber_episodes):
                im2 = ax_fh.imshow(fiber_episodes, aspect='auto',
                                extent=[time_arr[0], time_arr[-1],
                                        len(fiber_episodes), 1],
                                cmap='coolwarm', origin='lower')
                ax_fh.axvline(0, color='#FF0000', ls='--', alpha=0.8)
                ax_fh.set_xlabel('Time (s)')
                ax_fh.set_ylabel('Trial')
                ax_fh.set_title(f'{day_name} - Fiber Heatmap {wl}nm')
                fig.colorbar(im2, ax=ax_fh, label='Z-score', orientation='horizontal')
            else:
                ax_fh.text(0.5, 0.5, f'No fiber episodes for {wl}nm',
                        ha='center', va='center', transform=ax_fh.transAxes,
                        fontsize=12, color='#666666')
                ax_fh.set_title(f'{day_name} - Fiber Heatmap {wl}nm')
                ax_fh.axis('off')
            plot_idx += 1

        fig.tight_layout()
        
        # Add canvas
        canvas_frame = tk.Frame(day_window, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))
        self.toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        
        log_message(f"Individual day plot created for {day_name} with {len(target_wavelengths)} wavelength(s)")