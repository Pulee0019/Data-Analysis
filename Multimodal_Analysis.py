import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import numpy as np
import pandas as pd
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
    
    def general_onsets_analysis(self):
        """Analyze GENERAL ONSETS events"""
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
        
        # Create parameter setting window with UI styling
        param_window = tk.Toplevel(self.root)
        param_window.title("GENERAL ONSETS Analysis Parameters")
        param_window.geometry("400x400")
        param_window.configure(bg='#f8f8f8')
        param_window.transient(self.root)
        param_window.grab_set()
        
        # Title
        title_label = tk.Label(param_window, text="⏱️ GENERAL ONSETS Analysis Settings", 
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
                self._plot_general_onsets_analysis(animal_data, pre_time, post_time, selected_channels)
                
            except ValueError:
                log_message("Please enter valid time values", "WARNING")
        
        # Action buttons frame
        button_frame = tk.Frame(param_window, bg="#f8f8f8")
        button_frame.pack(pady=15)
        
        run_btn = tk.Button(button_frame, text="🚀 Start Analysis", command=run_analysis,
                           bg="#3498db", fg="white", font=("Microsoft YaHei", 9, "bold"),
                           relief=tk.FLAT, padx=15, pady=5)
        run_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(button_frame, text="❌ Cancel", 
                              command=param_window.destroy,
                              bg="#95a5a6", fg="white", font=("Microsoft YaHei", 9),
                              relief=tk.FLAT, padx=15, pady=5)
        cancel_btn.pack(side=tk.LEFT, padx=10)
    
    def _calculate_zscore_around_onsets(self, onsets, fiber_timestamps, dff_data, pre_time, post_time):
        """Calculate z-score around onsets using dff data and pre-time as baseline period"""
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        all_zscore_episodes = []
        
        for onset in onsets:
            # Get baseline period for this onset (pre_time seconds before onset)
            baseline_start = onset - pre_time
            baseline_end = onset
            
            # Find indices for baseline period
            baseline_start_idx = np.argmin(np.abs(fiber_timestamps - baseline_start))
            baseline_end_idx = np.argmin(np.abs(fiber_timestamps - baseline_end))
            
            if baseline_end_idx > baseline_start_idx:
                # Calculate mean and std from baseline period
                baseline_data = dff_data[baseline_start_idx:baseline_end_idx]
                mean_dff = np.nanmean(baseline_data)
                std_dff = np.nanstd(baseline_data)
                
                # Avoid division by zero
                if std_dff == 0:
                    std_dff = 1e-10
                
                # Get data around onset for z-score calculation
                start_idx = np.argmin(np.abs(fiber_timestamps - (onset - pre_time)))
                end_idx = np.argmin(np.abs(fiber_timestamps - (onset + post_time)))
                
                if end_idx > start_idx:
                    episode_data = dff_data[start_idx:end_idx]
                    episode_times = fiber_timestamps[start_idx:end_idx] - onset
                    
                    if len(episode_times) > 1:
                        # Calculate z-score for this episode
                        zscore_episode = (episode_data - mean_dff) / std_dff
                        
                        # Interpolate to standard time axis
                        interp_data = np.interp(time_array, episode_times, zscore_episode)
                        all_zscore_episodes.append(interp_data)
        
        return time_array, all_zscore_episodes
    
    def _plot_general_onsets_analysis(self, animal_data, pre_time, post_time, selected_channels):
        """Plot GENERAL ONSETS analysis results"""
        # Get data
        treadmill_behaviors = animal_data['treadmill_behaviors']
        dff_data = animal_data['dff_data']
        ast2_data = animal_data['ast2_data_adjusted']
        
        general_onsets = treadmill_behaviors.get('general_onsets', [])
        running_timestamps = ast2_data['data']['timestamps']
        processed_data = animal_data.get('running_processed_data')
        running_speed = processed_data['filtered_speed']
        # running_speed = ast2_data['data']['speed']
        
        # Get fiber timestamps
        preprocessed_data = animal_data['preprocessed_data']
        channels = animal_data.get('channels', {})
        time_col = channels['time']
        fiber_timestamps = preprocessed_data[time_col]
        
        # **FIX 1: Properly handle dff_data format**
        combined_dff_data = None
        channel_label = ""
        
        # Check if dff_data is empty or None
        if dff_data is None or (isinstance(dff_data, pd.DataFrame) and dff_data.empty):
            log_message("No dFF data available", "ERROR")
            return
        
        # Handle different dff_data formats
        if isinstance(dff_data, pd.Series):
            # Single channel - use directly
            combined_dff_data = dff_data.values
            channel_label = "1"
        elif isinstance(dff_data, pd.DataFrame):
            # Multiple channels - calculate average of selected channels
            valid_columns = []
            for channel in selected_channels:
                col_name = f"CH{channel}_dff"
                if col_name in dff_data.columns:
                    valid_columns.append(col_name)
            
            if not valid_columns:
                log_message("No valid dFF columns found for selected channels", "ERROR")
                return
            
            if len(valid_columns) == 1:
                combined_dff_data = dff_data[valid_columns[0]].values
            else:
                combined_dff_data = dff_data[valid_columns].mean(axis=1).values
            
            channel_label = "+".join(selected_channels)
        elif isinstance(dff_data, dict):
            # Dictionary format - combine selected channels
            valid_channels = []
            for channel in selected_channels:
                if str(channel) in dff_data:
                    valid_channels.append(str(channel))
            
            if not valid_channels:
                log_message("No valid channels found in dFF data", "ERROR")
                return
            
            if len(valid_channels) == 1:
                combined_dff_data = dff_data[valid_channels[0]].values if isinstance(dff_data[valid_channels[0]], pd.Series) else dff_data[valid_channels[0]]
            else:
                # Average multiple channels
                channel_arrays = []
                for ch in valid_channels:
                    ch_data = dff_data[ch].values if isinstance(dff_data[ch], pd.Series) else dff_data[ch]
                    channel_arrays.append(ch_data)
                combined_dff_data = np.mean(channel_arrays, axis=0)
            
            channel_label = "+".join(valid_channels)
        else:
            log_message(f"Unsupported dFF data format: {type(dff_data)}", "ERROR")
            return
        
        if combined_dff_data is None:
            log_message("Failed to extract dFF data", "ERROR")
            return
        
        if len(general_onsets) == 0:
            log_message("No GENERAL ONSETS events found", "INFO")
            return
        
        # Calculate z-score episodes
        time_array, zscore_episodes = self._calculate_zscore_around_onsets(
            general_onsets, fiber_timestamps, combined_dff_data, pre_time, post_time)
        
        # Create result window with UI styling
        result_window = tk.Toplevel(self.root)
        result_window.title(f"GENERAL ONSETS Analysis - Channels {channel_label}")
        result_window.geometry("1300x900")
        result_window.configure(bg='#f8f8f8')
        
        # Main container
        main_container = tk.Frame(result_window, bg='#f8f8f8')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_container, 
                              text=f"GENERAL ONSETS Analysis - Channels {channel_label}",
                              font=("Microsoft YaHei", 14, "bold"), bg="#f8f8f8", fg="#2c3e50")
        title_label.pack(pady=(0, 15))
        
        # Info panel
        info_text = (f"Analysis Info: {len(general_onsets)} events, "
                    f"Time window: [-{pre_time}, {post_time}]s, "
                    f"Channels: {channel_label}")
        info_label = tk.Label(main_container, text=info_text,
                             font=("Microsoft YaHei", 9), bg="#f8f8f8", fg="#34495e")
        info_label.pack(pady=(0, 10))
        
        # Create matplotlib figure
        fig = Figure(figsize=(12, 8), dpi=100)
        
        # 1. Running mean±std plot
        ax1 = fig.add_subplot(221)
        self._plot_running_around_onsets(ax1, general_onsets, running_timestamps, running_speed, 
                                       pre_time, post_time, "Running Speed Around GENERAL ONSETS")
        
        # 2. Fiber z-score mean±std plot
        ax2 = fig.add_subplot(222)
        self._plot_fiber_zscore_around_onsets(ax2, time_array, zscore_episodes,
                                            f"Fiber Z-score Around GENERAL ONSETS (CH{channel_label})")
        
        # 3. Running heatmap
        ax3 = fig.add_subplot(223)
        self._plot_running_heatmap(ax3, general_onsets, running_timestamps, running_speed,
                                 pre_time, post_time, "Running Speed Heatmap")
        
        # 4. Fiber z-score heatmap
        ax4 = fig.add_subplot(224)
        self._plot_fiber_zscore_heatmap(ax4, zscore_episodes, time_array,
                                      f"Fiber Z-score Heatmap (CH{channel_label})")
        
        fig.tight_layout()
        
        # Add canvas to window
        canvas_frame = tk.Frame(main_container, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(main_container, bg='#f8f8f8')
        control_frame.pack(fill=tk.X, pady=15)
        
        # Save button
        save_btn = tk.Button(control_frame, text="💾 Save Figure", 
                           command=lambda: self._save_figure(fig),
                           bg="#27ae60", fg="white", font=("Microsoft YaHei", 10, "bold"),
                           relief=tk.FLAT, padx=20, pady=8)
        save_btn.pack(side=tk.RIGHT, padx=10)
        
        # Close button
        close_btn = tk.Button(control_frame, text="❌ Close", 
                            command=result_window.destroy,
                            bg="#e74c3c", fg="white", font=("Microsoft YaHei", 10),
                            relief=tk.FLAT, padx=20, pady=8)
        close_btn.pack(side=tk.RIGHT, padx=10)
        
        log_message(f"GENERAL ONSETS analysis completed: {len(general_onsets)} events, "
                   f"channels {channel_label}, time window [-{pre_time},{post_time}]s")
    
    def _plot_running_around_onsets(self, ax, onsets, timestamps, speed, pre_time, post_time, title):
        """Plot running speed around onsets (mean±std)"""
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        all_episodes = []
        
        for onset in onsets:
            start_idx = np.argmin(np.abs(timestamps - (onset - pre_time)))
            end_idx = np.argmin(np.abs(timestamps - (onset + post_time)))
            
            if end_idx > start_idx:
                episode_data = speed[start_idx:end_idx]
                # Interpolate to standard time axis
                episode_times = timestamps[start_idx:end_idx] - onset
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
            ax.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Onset')
            ax.axhline(y=0, color='#808080', linestyle='--', alpha=0.8, label='Baseline')
            
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Running Speed (cm/s)')
        ax.set_title(title)
        ax.legend()
        ax.grid(False)
    
    def _plot_fiber_zscore_around_onsets(self, ax, time_array, zscore_episodes, title):
        """Plot fiber z-score around onsets (mean±std) using pre-calculated z-score episodes"""
        if zscore_episodes:
            all_episodes = np.array(zscore_episodes)
            mean_response = np.nanmean(all_episodes, axis=0)
            std_response = np.nanstd(all_episodes, axis=0)
            
            ax.plot(time_array, mean_response, "#008000",linestyle='-', linewidth=2, label='Mean')
            ax.fill_between(time_array, mean_response - std_response,
                          mean_response + std_response, color='#008000', alpha=0.3, label='Mean ± STD')
            ax.axvline(x=0, color='#808080', linestyle='--', alpha=0.8, label='Onset')
            ax.axhline(y=0, color='#808080', linestyle='--', alpha=0.8, label='Baseline')
            
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-score')
        ax.set_title(title)
        ax.legend()
        ax.grid(False)
    
    def _plot_running_heatmap(self, ax, onsets, timestamps, speed, pre_time, post_time, title):
        """Plot running speed heatmap"""
        time_array = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
        all_episodes = []
        
        for i, onset in enumerate(onsets):
            start_idx = np.argmin(np.abs(timestamps - (onset - pre_time)))
            end_idx = np.argmin(np.abs(timestamps - (onset + post_time)))
            
            if end_idx > start_idx:
                episode_data = speed[start_idx:end_idx]
                episode_times = timestamps[start_idx:end_idx] - onset
                if len(episode_times) > 1:
                    interp_data = np.interp(time_array, episode_times, episode_data)
                    all_episodes.append(interp_data)
        
        if all_episodes:
            all_episodes = np.array(all_episodes)
            im = ax.imshow(all_episodes, aspect='auto', extent=[-pre_time, post_time, len(onsets), 1], 
                         cmap='viridis', origin='lower')
            ax.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial Number')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label='Running Speed (cm/s)')
    
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
            plt.colorbar(im, ax=ax, label='Z-score')
    
    def continuous_locomotion_analysis(self):
        """Analyze trajectories during CONTINUOUS LOCOMOTION PERIODS"""
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
        
        locomotion_periods = animal_data['treadmill_behaviors'].get('continuous_locomotion_periods', [])
        if len(locomotion_periods) == 0:
            log_message("No CONTINUOUS LOCOMOTION PERIODS found", "INFO")
            return
        
        # Use bodyparts selected in main UI
        if not self.selected_bodyparts:
            log_message("No bodyparts selected, please select bodyparts in the main UI", "WARNING")
            return
        
        selected_bodyparts = list(self.selected_bodyparts)
        self._plot_locomotion_trajectories(animal_data, locomotion_periods, selected_bodyparts)
    
    def _plot_locomotion_trajectories(self, animal_data, locomotion_periods, selected_bodyparts):
        """Plot trajectory point clouds during locomotion periods - All bodyparts on single plot"""
        dlc_data = animal_data['dlc_data']
        video_fps = animal_data.get('video_fps', 30)
        
        # Create result window with UI styling
        result_window = tk.Toplevel(self.root)
        result_window.title("CONTINUOUS LOCOMOTION Trajectory Analysis")
        result_window.geometry("1200x900")
        result_window.configure(bg='#f8f8f8')
        
        # Main container
        main_container = tk.Frame(result_window, bg='#f8f8f8')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title and info panel
        title_label = tk.Label(main_container, 
                              text="Continuous Locomotion Trajectory Analysis",
                              font=("Microsoft YaHei", 16, "bold"), bg="#f8f8f8", fg="#2c3e50")
        title_label.pack(pady=(0, 10))
        
        info_text = (f"Analysis Info: {len(locomotion_periods)} locomotion periods, "
                    f"{len(selected_bodyparts)} bodyparts, Video FPS: {video_fps}")
        info_label = tk.Label(main_container, text=info_text,
                             font=("Microsoft YaHei", 10), bg="#f8f8f8", fg="#34495e")
        info_label.pack(pady=(0, 15))
        
        # Create matplotlib figure with UI styling
        fig = Figure(figsize=(10, 8), dpi=100)
        ax = fig.add_subplot(111)
        
        # Set graph properties - UI style
        ax.set_title("🔍 Continuous Locomotion Trajectories", fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
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
            
            # Collect trajectory points for this bodypart across all locomotion periods
            bodypart_x = []
            bodypart_y = []
            
            for j, (start_time, end_time) in enumerate(locomotion_periods):
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
        canvas_frame = tk.Frame(main_container, bg='#f8f8f8')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(main_container, bg='#f8f8f8')
        control_frame.pack(fill=tk.X, pady=15)
        
        # Save button
        save_btn = tk.Button(control_frame, text="💾 Save Figure", 
                           command=lambda: self._save_figure(fig),
                           bg="#27ae60", fg="white", font=("Microsoft YaHei", 10, "bold"),
                           relief=tk.FLAT, padx=20, pady=8)
        save_btn.pack(side=tk.RIGHT, padx=10)
        
        # Close button
        close_btn = tk.Button(control_frame, text="❌ Close", 
                            command=result_window.destroy,
                            bg="#e74c3c", fg="white", font=("Microsoft YaHei", 10),
                            relief=tk.FLAT, padx=20, pady=8)
        close_btn.pack(side=tk.RIGHT, padx=10)
        
        log_message(f"Continuous locomotion trajectory analysis completed: {len(locomotion_periods)} periods, {len(selected_bodyparts)} bodyparts")

    def _save_figure(self, fig):
        """Save figure to file"""
        filename = filedialog.asksaveasfilename(
            title="Save Figure",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF File", "*.pdf"), ("SVG Image", "*.svg")]
        )
        if filename:
            try:
                fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                log_message(f"Figure saved: {filename}")
            except Exception as e:
                log_message(f"Failed to save figure: {str(e)}", "ERROR")

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
        
        # Colors for different days
        self.day_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                          '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#95a5a6']
        
        self.create_analysis_window()
    
    def create_analysis_window(self):
        """Create the acrossday analysis configuration window"""
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title("Acrossday Analysis Configuration")
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
        """Run acrossday analysis"""
        # Validate table data
        if not self.table_data:
            log_message("No data in the table to analyze", "WARNING")
            return
        
        param_window = tk.Toplevel(self.analysis_window)
        param_window.title("Acrossday Analysis Parameters")
        param_window.geometry("300x200")
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
        
        def start_with_params():
            try:
                pre_time = float(pre_time_var.get())
                post_time = float(post_time_var.get())
                
                if pre_time <= 0 or post_time <= 0:
                    log_message("Time must be positive numbers", "WARNING")
                    return
                    
                param_window.destroy()
                self._run_analysis_with_params(pre_time, post_time)
                
            except ValueError:
                log_message("Please enter valid time values", "WARNING")
        
        # Buttons
        button_frame = tk.Frame(param_window, bg="#f8f8f8")
        button_frame.pack(pady=20)
        
        run_btn = tk.Button(button_frame, text="🚀 Start Analysis", command=start_with_params,
                        bg="#3498db", fg="white", font=("Microsoft YaHei", 9, "bold"),
                        relief=tk.FLAT, padx=15, pady=5)
        run_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(button_frame, text="❌ Cancel", 
                            command=param_window.destroy,
                            bg="#95a5a6", fg="white", font=("Microsoft YaHei", 9),
                            relief=tk.FLAT, padx=15, pady=5)
        cancel_btn.pack(side=tk.LEFT, padx=10)

    def _run_analysis_with_params(self, pre_time, post_time):
        self.pre_time = pre_time
        self.post_time = post_time
        
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
        
        log_message(f"Starting acrossday analysis for {len(day_data)} days (pre={pre_time}s, post={post_time}s)...")
        
        # Perform analysis for each day
        results = {}
        for day_name, animals in day_data.items():
            log_message(f"Analyzing {day_name} with {len(animals)} animals...")
            day_result = self.analyze_day(day_name, animals, pre_time, post_time)
            if day_result:
                results[day_name] = day_result
        
        if results:
            self.results = results
            self.plot_results(results)
            log_message("Acrossday analysis completed successfully")
        else:
            log_message("Acrossday analysis failed, no valid results", "ERROR")

    def analyze_day(self, day_name, animals, pre_time, post_time):
        """Analyze data for one day with specified time windows"""
        all_running_episodes = []
        all_fiber_episodes = []
        
        for animal_data in animals:
            try:
                # Check required data
                if 'treadmill_behaviors' not in animal_data:
                    log_message(f"Skipping {animal_data.get('animal_id')}: no running behavior data", "WARNING")
                    continue
                
                if 'dff_data' not in animal_data or 'ast2_data_adjusted' not in animal_data:
                    log_message(f"Skipping {animal_data.get('animal_id')}: missing necessary data", "WARNING")
                    continue
                
                # Get general onsets
                general_onsets = animal_data['treadmill_behaviors'].get('general_onsets', [])
                if not general_onsets:
                    log_message(f"No general onsets for {animal_data.get('animal_id')}", "INFO")
                    continue
                
                # Get data
                ast2_data = animal_data['ast2_data_adjusted']
                running_timestamps = ast2_data['data']['timestamps']
                processed_data = animal_data.get('running_processed_data')
                running_speed = processed_data['filtered_speed']
                # running_speed = ast2_data['data']['speed']
                
                preprocessed_data = animal_data['preprocessed_data']
                channels = animal_data.get('channels', {})
                time_col = channels['time']
                fiber_timestamps = preprocessed_data[time_col].values
                
                # Handle dff_data format
                dff_data = animal_data['dff_data']
                combined_dff_data = None
                
                if isinstance(dff_data, pd.Series):
                    combined_dff_data = dff_data.values
                elif isinstance(dff_data, pd.DataFrame):
                    active_channels = animal_data.get('active_channels', [])
                    valid_columns = []
                    for channel in active_channels:
                        col_name = f"CH{channel}_dff"
                        if col_name in dff_data.columns:
                            valid_columns.append(col_name)
                    if valid_columns:
                        combined_dff_data = dff_data[valid_columns].mean(axis=1).values if len(valid_columns) > 1 else dff_data[valid_columns[0]].values
                    else:
                        log_message(f"No valid dFF columns for {animal_data.get('animal_id')}", "WARNING")
                        continue
                elif isinstance(dff_data, dict):
                    active_channels = animal_data.get('active_channels', [])
                    valid_channels = []
                    for channel in active_channels:
                        if str(channel) in dff_data:
                            valid_channels.append(str(channel))
                    
                    if not valid_channels:
                        log_message(f"No valid channels in dFF data for {animal_data.get('animal_id')}", "WARNING")
                        continue
                    
                    if len(valid_channels) == 1:
                        ch_data = dff_data[valid_channels[0]]
                        combined_dff_data = ch_data.values if isinstance(ch_data, pd.Series) else ch_data
                    else:
                        channel_arrays = []
                        for ch in valid_channels:
                            ch_data = dff_data[ch]
                            ch_array = ch_data.values if isinstance(ch_data, pd.Series) else ch_data
                            channel_arrays.append(ch_array)
                        combined_dff_data = np.mean(channel_arrays, axis=0)
                else:
                    log_message(f"Unsupported dFF format for {animal_data.get('animal_id')}", "WARNING")
                    continue
                
                if combined_dff_data is None:
                    log_message(f"Failed to extract dFF data for {animal_data.get('animal_id')}", "WARNING")
                    continue
                
                # Running episodes
                time_array_running = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
                for onset in general_onsets:
                    start_idx = np.argmin(np.abs(running_timestamps - (onset - pre_time)))
                    end_idx = np.argmin(np.abs(running_timestamps - (onset + post_time)))
                    
                    if end_idx > start_idx:
                        episode_data = running_speed[start_idx:end_idx]
                        episode_times = running_timestamps[start_idx:end_idx] - onset
                        
                        if len(episode_times) > 1:
                            interp_data = np.interp(time_array_running, episode_times, episode_data)
                            all_running_episodes.append(interp_data)
                
                # Fiber z-score episodes
                time_array_fiber = np.linspace(-pre_time, post_time, int((pre_time + post_time) * 10))
                for onset in general_onsets:
                    # Baseline period
                    baseline_start = onset - pre_time
                    baseline_end = onset
                    baseline_start_idx = np.argmin(np.abs(fiber_timestamps - baseline_start))
                    baseline_end_idx = np.argmin(np.abs(fiber_timestamps - baseline_end))
                    
                    if baseline_end_idx > baseline_start_idx:
                        baseline_data = combined_dff_data[baseline_start_idx:baseline_end_idx]
                        mean_dff = np.nanmean(baseline_data)
                        std_dff = np.nanstd(baseline_data)
                        
                        if std_dff == 0:
                            std_dff = 1e-10
                        
                        start_idx = np.argmin(np.abs(fiber_timestamps - (onset - pre_time)))
                        end_idx = np.argmin(np.abs(fiber_timestamps - (onset + post_time)))
                        
                        if end_idx > start_idx:
                            episode_data = combined_dff_data[start_idx:end_idx]
                            episode_times = fiber_timestamps[start_idx:end_idx] - onset
                            
                            if len(episode_times) > 1:
                                zscore_episode = (episode_data - mean_dff) / std_dff
                                interp_data = np.interp(time_array_fiber, episode_times, zscore_episode)
                                all_fiber_episodes.append(interp_data)
                
            except Exception as e:
                log_message(f"Error analyzing {animal_data.get('animal_id')}: {str(e)}", "ERROR")
                continue
        
        if not all_running_episodes or not all_fiber_episodes:
            log_message(f"No valid episodes for {day_name}", "WARNING")
            return None
        
        # Calculate mean and SEM
        running_episodes = np.array(all_running_episodes)
        fiber_episodes = np.array(all_fiber_episodes)
        
        return {
            'running': {
                'time': time_array_running,
                'mean': np.nanmean(running_episodes, axis=0),
                'sem': np.nanstd(running_episodes, axis=0) / np.sqrt(len(running_episodes)),
                'episodes': running_episodes
            },
            'fiber': {
                'time': time_array_fiber,
                'mean': np.nanmean(fiber_episodes, axis=0),
                'sem': np.nanstd(fiber_episodes, axis=0) / np.sqrt(len(fiber_episodes)),
                'episodes': fiber_episodes
            }
        }

    def plot_results(self, results):
        """Plot analysis results - All days in one window"""
        # Create result window
        result_window = tk.Toplevel(self.root)
        result_window.title("Acrossday Analysis Results - All Days")
        result_window.geometry("1600x1000")
        result_window.configure(bg="#ffffff")
        
        # Create figure with subplots
        fig = Figure(figsize=(16, 10), dpi=100)
        
        # 1. All days running curves (top left)
        ax1 = fig.add_subplot(221)
        for idx, (day_name, data) in enumerate(results.items()):
            color = self.day_colors[idx % len(self.day_colors)]
            running_data = data['running']
            ax1.plot(running_data['time'], running_data['mean'], 
                    color=color, linewidth=2, label=day_name)
            ax1.fill_between(running_data['time'], 
                            running_data['mean'] - running_data['sem'],
                            running_data['mean'] + running_data['sem'],
                            color=color, alpha=0.3)
        ax1.axvline(x=0, color='#808080', linestyle='--', alpha=0.8)
        ax1.axhline(y=0, color='#808080', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Running Speed (cm/s)')
        ax1.set_title('Running Speed - All Days')
        ax1.legend()
        ax1.grid(False)
        
        # 2. All days fiber curves (top right)
        ax2 = fig.add_subplot(222)
        for idx, (day_name, data) in enumerate(results.items()):
            color = self.day_colors[idx % len(self.day_colors)]
            fiber_data = data['fiber']
            ax2.plot(fiber_data['time'], fiber_data['mean'],
                    color=color, linewidth=2, label=day_name)
            ax2.fill_between(fiber_data['time'],
                            fiber_data['mean'] - fiber_data['sem'],
                            fiber_data['mean'] + fiber_data['sem'],
                            color=color, alpha=0.3)
        ax2.axvline(x=0, color='#808080', linestyle='--', alpha=0.8)
        ax2.axhline(y=0, color='#808080', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Z-score')
        ax2.set_title('Fiber Z-score - All Days')
        
        # 3. All days running heatmap (bottom left)
        ax3 = fig.add_subplot(223)
        self.plot_combined_running_heatmap(results, ax3)
        
        # 4. All days fiber heatmap (bottom right)
        ax4 = fig.add_subplot(224)
        self.plot_combined_fiber_heatmap(results, ax4)
        
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
        
        self.create_individual_day_windows(results)
        
        log_message(f"Results plotted for {len(results)} days")

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
            plt.colorbar(im, ax=ax, label='Speed (cm/s)')
        else:
            ax.text(0.5, 0.5, 'No running data available', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#666666')
            ax.set_title('Running Heatmap - All Days')
            ax.axis('off')

    def plot_combined_fiber_heatmap(self, results, ax):
        """Plot combined fiber heatmap for all days"""
        all_fiber_episodes = []
        for day_name, data in results.items():
            fiber_episodes = data['fiber']['episodes']
            all_fiber_episodes.extend(fiber_episodes)
        
        if all_fiber_episodes:
            time_array = list(results.values())[0]['fiber']['time']
            im = ax.imshow(all_fiber_episodes, aspect='auto',
                        extent=[time_array[0], time_array[-1],
                                len(all_fiber_episodes), 1],
                        cmap='coolwarm', origin='lower')
            ax.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Trial (All Days)')
            ax.set_title('Fiber Heatmap - All Days')
            plt.colorbar(im, ax=ax, label='Z-score')
        else:
            ax.text(0.5, 0.5, 'No fiber data available', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#666666')
            ax.set_title('Fiber Heatmap - All Days')
            ax.axis('off')

    def create_individual_day_windows(self, results):
        """Create individual windows for each day"""
        for day_name, data in results.items():
            self.create_single_day_window(day_name, data)

    def create_single_day_window(self, day_name, data):
        """Create window for a single day with 4 subplots"""
        day_window = tk.Toplevel(self.root)
        day_window.title(f"Acrossday Analysis - {day_name}")
        day_window.geometry("1400x900")
        day_window.configure(bg='#f8f8f8')
        
        # Create figure with 2x2 subplots
        fig = Figure(figsize=(14, 8), dpi=100)
        
        # Get color for this day
        day_idx = list(self.results.keys()).index(day_name) if hasattr(self, 'results') else 0
        color = self.day_colors[day_idx % len(self.day_colors)]
        
        # 1. Running trace (top left)
        ax1 = fig.add_subplot(221)
        running_data = data['running']
        ax1.plot(running_data['time'], running_data['mean'],
                color=color, linewidth=2)
        ax1.fill_between(running_data['time'],
                        running_data['mean'] - running_data['sem'],
                        running_data['mean'] + running_data['sem'],
                        color=color, alpha=0.3)
        ax1.axvline(x=0, color='#808080', linestyle='--', alpha=0.8)
        ax1.axhline(y=0, color='#808080', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speed (cm/s)')
        ax1.set_title(f'{day_name} - Running Trace')
        ax1.grid(False)
        
        # 2. Fiber trace (top right)
        ax2 = fig.add_subplot(222)
        fiber_data = data['fiber']
        ax2.plot(fiber_data['time'], fiber_data['mean'],
                color=color, linewidth=2)
        ax2.fill_between(fiber_data['time'],
                        fiber_data['mean'] - fiber_data['sem'],
                        fiber_data['mean'] + fiber_data['sem'],
                        color=color, alpha=0.3)
        ax2.axvline(x=0, color='#808080', linestyle='--', alpha=0.8)
        ax2.axhline(y=0, color='#808080', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Z-score')
        ax2.set_title(f'{day_name} - Fiber Trace')
        ax2.grid(False)
        
        # 3. Running heatmap (bottom left)
        ax3 = fig.add_subplot(223)
        running_episodes = data['running']['episodes']
        time_array = data['running']['time']
        
        if len(running_episodes) > 0:
            im1 = ax3.imshow(running_episodes, aspect='auto',
                            extent=[time_array[0], time_array[-1], 
                                len(running_episodes), 1],
                            cmap='viridis', origin='lower')
            ax3.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Trial')
            ax3.set_title(f'{day_name} - Running Heatmap')
            plt.colorbar(im1, ax=ax3, label='Speed (cm/s)')
        else:
            ax3.text(0.5, 0.5, 'No running episodes', 
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12, color='#666666')
            ax3.set_title(f'{day_name} - Running Heatmap')
            ax3.axis('off')
        
        # 4. Fiber heatmap (bottom right)
        ax4 = fig.add_subplot(224)
        fiber_episodes = data['fiber']['episodes']
        time_array = data['fiber']['time']
        
        if len(fiber_episodes) > 0:
            im2 = ax4.imshow(fiber_episodes, aspect='auto',
                            extent=[time_array[0], time_array[-1],
                                len(fiber_episodes), 1],
                            cmap='coolwarm', origin='lower')
            ax4.axvline(x=0, color="#FF0000", linestyle='--', alpha=0.8)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Trial')
            ax4.set_title(f'{day_name} - Fiber Heatmap')
            plt.colorbar(im2, ax=ax4, label='Z-score')
        else:
            ax4.text(0.5, 0.5, 'No fiber episodes', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, color='#666666')
            ax4.set_title(f'{day_name} - Fiber Heatmap')
            ax4.axis('off')
        
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
        
        log_message(f"Individual day plot created for {day_name}")