import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        running_speed = ast2_data['data']['speed']
        
        # Get fiber timestamps
        preprocessed_data = animal_data['preprocessed_data']
        channels = animal_data.get('channels', {})
        time_col = channels['time']
        fiber_timestamps = preprocessed_data[time_col]
        
        # Check if dff_data is a Series or DataFrame
        if isinstance(dff_data, pd.Series):
            # Single channel - use directly
            combined_dff_data = dff_data
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
                combined_dff_data = dff_data[valid_columns[0]]
            else:
                combined_dff_data = dff_data[valid_columns].mean(axis=1)
            
            channel_label = "+".join(selected_channels)
        else:
            log_message("Unsupported dFF data format", "ERROR")
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
                              text=f"📊 GENERAL ONSETS Analysis - Channels {channel_label}",
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
        
        # # Control panel
        # control_frame = tk.Frame(main_container, bg='#f8f8f8')
        # control_frame.pack(fill=tk.X, pady=10)
        
        # # Save button
        # save_btn = tk.Button(control_frame, text="💾 Save Figure", 
        #                    command=lambda: self._save_figure(fig),
        #                    bg="#27ae60", fg="white", font=("Microsoft YaHei", 10, "bold"),
        #                    relief=tk.FLAT, padx=20, pady=8)
        # save_btn.pack(side=tk.RIGHT, padx=10)
        
        # # Close button
        # close_btn = tk.Button(control_frame, text="❌ Close", 
        #                     command=result_window.destroy,
        #                     bg="#e74c3c", fg="white", font=("Microsoft YaHei", 10),
        #                     relief=tk.FLAT, padx=20, pady=8)
        # close_btn.pack(side=tk.RIGHT, padx=10)
        
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
        ax.set_title("📍 Continuous Locomotion Trajectories", fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
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