from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
import numpy as np
import threading
import traceback
import fnmatch
import signal
import json
import glob
import time
import sys
import os
import re

from Behavior_analysis import position_analysis, displacement_analysis, x_displacement_analysis
from Fiber_analysis import apply_preprocessing, calculate_and_plot_dff, calculate_and_plot_zscore
from Running_analysis import classify_treadmill_behavior, preprocess_running_data
from Multimodal_Analysis import MultimodalAnalysis
from logger import log_message, set_log_widget

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    log_message("OpenCV not installed, video export function unavailable. Please run 'pip install opencv-python' to install.", "WARNING")

# Global channel memory
CHANNEL_MEMORY_FILE = "channel_memory.json"
channel_memory = {}

def load_channel_memory():
    """Load channel memory from file"""
    global channel_memory
    if os.path.exists(CHANNEL_MEMORY_FILE):
        try:
            with open(CHANNEL_MEMORY_FILE, 'r') as f:
                channel_memory = json.load(f)
        except:
            channel_memory = {}

def save_channel_memory():
    """Save channel memory to file"""
    try:
        with open(CHANNEL_MEMORY_FILE, 'w') as f:
            json.dump(channel_memory, f)
    except:
        pass

load_channel_memory()

class BodypartVisualizationWindow:
    def __init__(self, parent_frame, data):
        self.parent_frame = parent_frame
        self.data = data
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.fps = 10
        self.play_thread = None
        
        self.window_width = 800
        self.window_height = 870
        self.is_minimized = False
        
        self.pan_start = None
        self.zoom_factor = 1.0
        self.original_xlim = None
        self.original_ylim = None
        self.is_panning = False
        
        if data:
            self.total_frames = min([len(bodypart_data['x']) for bodypart_data in data.values()])
        
        self.create_window()
        
    def create_window(self):
        self.window_frame = tk.Frame(self.parent_frame, bg="#f5f5f5", relief=tk.RAISED, bd=1)
        self.window_frame.place(x=0, y=0, width=self.window_width, height=self.window_height)
        
        self.window_frame.bind("<Button-1>", self.start_move)
        self.window_frame.bind("<B1-Motion>", self.do_move)
        
        title_frame = tk.Frame(self.window_frame, bg="#f5f5f5", height=25)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        title_frame.bind("<Button-1>", self.start_move)
        title_frame.bind("<B1-Motion>", self.do_move)
        
        title_label = tk.Label(title_frame, text="Bodyparts Position Visualization", bg="#f5f5f5", fg="#666666", 
                              font=("Microsoft YaHei", 10, "bold"))
        title_label.pack(side=tk.LEFT, padx=10, pady=3)
        title_label.bind("<Button-1>", self.start_move)
        title_label.bind("<B1-Motion>", self.do_move)
        
        # Window control buttons
        btn_frame = tk.Frame(title_frame, bg="#f5f5f5")
        btn_frame.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Minimize button
        minimize_btn = tk.Button(btn_frame, text="−", bg="#f5f5f5", fg="#999999", bd=0, 
                               font=("Arial", 8), width=2, height=1,
                               command=self.minimize_window, relief=tk.FLAT)
        minimize_btn.pack(side=tk.LEFT, padx=1)
        
        # Close button
        close_btn = tk.Button(btn_frame, text="×", bg="#f5f5f5", fg="#999999", bd=0, 
                             font=("Arial", 8), width=2, height=1,
                             command=self.close_window, relief=tk.FLAT)
        close_btn.pack(side=tk.LEFT, padx=1)

        reset_view_btn = tk.Button(btn_frame, text="🗘", bg="#f5f5f5", fg="#999999", bd=0,
                                 font=("Arial", 8), width=2, height=1,
                                 command=self.reset_view, relief=tk.FLAT)
        reset_view_btn.pack(side=tk.LEFT, padx=1)
        
        # Add window resize control point
        resize_frame = tk.Frame(self.window_frame, bg="#bdc3c7", width=15, height=15)
        resize_frame.place(relx=1.0, rely=1.0, anchor="se")
        resize_frame.bind("<Button-1>", self.start_resize)
        resize_frame.bind("<B1-Motion>", self.do_resize)
        resize_frame.config(cursor="size_nw_se")
        
        # Create matplotlib figure - with better color scheme
        self.fig = Figure(figsize=(7, 4.5), dpi=90, facecolor='#f8f9fa')
        self.ax = self.fig.add_subplot(111, facecolor='#ffffff')
        
        # Create canvas frame - add shadow effect
        canvas_frame = tk.Frame(self.window_frame, bg="#f5f5f5", relief=tk.SUNKEN, bd=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Control panel - modern design
        control_frame = tk.Frame(self.window_frame, bg="#ecf0f1", height=130, relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        # Progress bar - modern style (moved to top)
        progress_frame = tk.Frame(control_frame, bg="#ecf0f1")
        progress_frame.pack(fill=tk.X, pady=(8, 5))
        
        tk.Label(progress_frame, text="📊 Progress:", bg="#ecf0f1", 
                font=("Microsoft YaHei", 10, "bold"), fg="#2c3e50").pack(side=tk.LEFT, padx=10)
        self.progress_var = tk.DoubleVar()
        self.progress_scale = tk.Scale(progress_frame, from_=0, to=self.total_frames-1, 
                                      orient=tk.HORIZONTAL, variable=self.progress_var,
                                      command=self.on_progress_change, font=("Microsoft YaHei", 9),
                                      bg="#ecf0f1", fg="#2c3e50", highlightthickness=0,
                                      troughcolor="#bdc3c7", activebackground="#3498db",
                                      length=200)
        self.progress_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.frame_label = tk.Label(progress_frame, text=f"0/{self.total_frames}", bg="#ecf0f1", 
                                   width=12, font=("Microsoft YaHei", 10, "bold"), fg="#2c3e50")
        self.frame_label.pack(side=tk.RIGHT, padx=10)
        
        # Buttons and FPS control on the same line
        control_row_frame = tk.Frame(control_frame, bg="#ecf0f1")
        control_row_frame.pack(pady=(5, 8))
        
        # Play control buttons - modern button style
        btn_frame = tk.Frame(control_row_frame, bg="#ecf0f1")
        btn_frame.pack(side=tk.LEFT, padx=(10, 20))
        
        self.play_btn = tk.Button(btn_frame, text="▶ Play", command=self.toggle_play,
                                 bg="#27ae60", fg="white", font=("Microsoft YaHei", 10, "bold"),
                                 relief=tk.FLAT, padx=15, pady=5, cursor="hand2", width=8)
        self.play_btn.pack(side=tk.LEFT, padx=3)
        
        self.pause_btn = tk.Button(btn_frame, text="⏸ Pause", command=self.pause,
                                  bg="#f39c12", fg="white", font=("Microsoft YaHei", 10, "bold"),
                                  relief=tk.FLAT, padx=15, pady=5, cursor="hand2", width=8)
        self.pause_btn.pack(side=tk.LEFT, padx=3)
        
        self.reset_btn = tk.Button(btn_frame, text="🔄 Reset", command=self.reset,
                                  bg="#3498db", fg="white", font=("Microsoft YaHei", 10, "bold"),
                                  relief=tk.FLAT, padx=15, pady=5, cursor="hand2", width=8)
        self.reset_btn.pack(side=tk.LEFT, padx=3)
        
        # Video export button
        self.export_btn = tk.Button(btn_frame, text="🎬 Export Video", command=self.export_video,
                                   bg="#e74c3c", fg="white", font=("Microsoft YaHei", 10, "bold"),
                                   relief=tk.FLAT, padx=15, pady=5, cursor="hand2", width=10)
        self.export_btn.pack(side=tk.LEFT, padx=3)
        
        # FPS setting - modern style (same line as buttons)
        fps_frame = tk.Frame(control_row_frame, bg="#ecf0f1")
        fps_frame.pack(side=tk.RIGHT, padx=(20, 10))
        
        tk.Label(fps_frame, text="⚡ FPS:", bg="#ecf0f1", 
                font=("Microsoft YaHei", 10, "bold"), fg="#2c3e50").pack(side=tk.LEFT, padx=(0, 5))
        self.fps_var = tk.StringVar(value=str(self.fps))
        fps_spinbox = tk.Spinbox(fps_frame, from_=1, to=120, width=6, textvariable=self.fps_var,
                                command=self.update_fps, font=("Microsoft YaHei", 10), relief=tk.FLAT,
                                bg="white", fg="#2c3e50", buttonbackground="#bdc3c7")
        fps_spinbox.pack(side=tk.LEFT, padx=5)
        tk.Label(fps_frame, text="FPS", bg="#ecf0f1", 
                font=("Microsoft YaHei", 10, "bold"), fg="#2c3e50").pack(side=tk.LEFT, padx=(5, 0))
        
        # Show first frame
        self.update_plot()
        
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:
            self.pan_start = (event.xdata, event.ydata)
            self.is_panning = True
            
        elif event.button == 3:
            self.on_select(event)

    def on_release(self, event):
        if event.button == 1:
            self.pan_start = None
            self.is_panning = False

    def on_motion(self, event):
        if not self.is_panning or event.inaxes != self.ax or self.pan_start is None:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        dx = event.xdata - self.pan_start[0]
        dy = event.ydata - self.pan_start[1]
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        new_xlim = (xlim[0] - dx, xlim[1] - dx)
        new_ylim = (ylim[0] - dy, ylim[1] - dy)
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        self.canvas.draw_idle()
        
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            log_message("Scroll event outside axes, ignoring.", "WARNING")
            return
            
        try:
            current_time = getattr(self, '_last_scroll_time', 0)
            if time.time() - current_time < 0.05:
                return
            self._last_scroll_time = time.time()
            
            if event.xdata is None or event.ydata is None:
                return
                
            x, y = event.xdata, event.ydata
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            if not self._is_valid_range(xlim, ylim):
                self.reset_view()
                return
            
            zoom_factor = 1.1 if event.button == 'up' else 0.9
            
            new_xlim = (x - (x - xlim[0]) * zoom_factor, 
                    x + (xlim[1] - x) * zoom_factor)
            new_ylim = (y - (y - ylim[0]) * zoom_factor, 
                    y + (ylim[1] - y) * zoom_factor)
            
            if self._is_valid_range(new_xlim, new_ylim):
                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)
                self.canvas.draw_idle()
                
        except Exception as e:
            log_message(f"Scroll error: {str(e)}", "WARNING")
            self.reset_view()

    def _is_valid_range(self, xlim, ylim):
        try:
            return (all(np.isfinite(xlim)) and all(np.isfinite(ylim)) and
                    xlim[1] > xlim[0] and ylim[1] > ylim[0] and
                    abs(xlim[1] - xlim[0]) > 1e-10 and
                    abs(ylim[1] - ylim[0]) > 1e-10 and
                    abs(xlim[1] - xlim[0]) < 1e10 and
                    abs(ylim[1] - ylim[0]) < 1e10)
        except:
            return False

    def on_select(self, event):
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return
        
        min_distance = float('inf')
        closest_bodypart = None
        
        for i, (bodypart, data) in enumerate(self.data.items()):
            if self.current_frame < len(data['x']) and self.current_frame < len(data['y']):
                x = data['x'][self.current_frame]
                y = data['y'][self.current_frame]
                
                distance = ((click_x - x) ** 2 + (click_y - y) ** 2) ** 0.5
                
                click_threshold = 30
                
                if distance < click_threshold and distance < min_distance:
                    min_distance = distance
                    closest_bodypart = bodypart
        
        if closest_bodypart and closest_bodypart in bodypart_buttons:
            button = bodypart_buttons[closest_bodypart]
            toggle_bodypart(closest_bodypart, button)
        
    def reset_view(self):
        """Reset the view to original zoom and pan"""
        if self.original_xlim and self.original_ylim:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.zoom_factor = 1.0
            self.canvas.draw_idle()
            log_message("DLC view reset to original", "INFO")

    def update_fps(self):
        try:
            self.fps = int(self.fps_var.get())
            # Limit FPS range to avoid performance issues with high values
            if self.fps > 120:
                self.fps = 120
                self.fps_var.set("120")
            elif self.fps < 1:
                self.fps = 1
                self.fps_var.set("1")
        except ValueError:
            self.fps = 10
            self.fps_var.set("10")
    
    def toggle_play(self):
        if self.is_playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        if not self.is_playing and self.current_frame < self.total_frames - 1:
            self.is_playing = True
            self.play_btn.config(text="Playing...")
            self.schedule_next_frame()
    
    def pause(self):
        self.is_playing = False
        try:
            self.play_btn.config(text="▶ Play")
        except tk.TclError:
            pass
        if hasattr(self, 'after_id'):
            try:
                self.window_frame.after_cancel(self.after_id)
            except tk.TclError:
                pass
    
    def reset(self):
        self.pause()
        self.current_frame = 0
        self.progress_var.set(0)
        self.update_plot()
    
    def schedule_next_frame(self):
        if not self.window_frame.winfo_exists():
            return
            
        if self.is_playing and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.progress_var.set(self.current_frame)
            self.update_plot_optimized()
            
            delay_ms = max(1, int(1000.0 / self.fps))
            self.after_id = self.window_frame.after(delay_ms, self.schedule_next_frame)
        else:
            self.is_playing = False
            try:
                self.play_btn.config(text="▶ Play")
            except tk.TclError:
                pass
    
    def play_animation(self):
        """Keep original method as backup, but no longer used"""
        pass
    
    def on_progress_change(self, value):
        self.current_frame = int(float(value))
        self.update_plot()
    
    def update_plot(self):
        """Original update method, keep for compatibility"""
        self.update_plot_optimized()
    
    def update_plot_optimized(self):
        """Optimized plot update method, reduce unnecessary redraw operations"""
        # Only clear data points, keep axis settings
        if not hasattr(self, '_plot_initialized'):
            self._initialize_plot()
            self._plot_initialized = True
        
        # Clear previous scatter plots
        if hasattr(self, '_scatter_plots'):
            for scatter in self._scatter_plots:
                scatter.remove()
        
        self._scatter_plots = []
        
        # Use predefined colors
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                 '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#95a5a6']
        
        # Draw current frame data points
        for i, (bodypart, data) in enumerate(self.data.items()):
            if self.current_frame < len(data['x']) and self.current_frame < len(data['y']):
                x = data['x'][self.current_frame]
                y = data['y'][self.current_frame]
                
                # Draw point
                color = colors[i % len(colors)]
                scatter = self.ax.scatter(x, y, s=200, alpha=0.8, 
                                        color=color, edgecolors='black', linewidth=2)
                self._scatter_plots.append(scatter)
                
                # Add number inside circle
                text = self.ax.text(x, y, str(i+1), ha='center', va='center', 
                                   fontsize=8, fontweight='bold', color='black')
                self._scatter_plots.append(text)
        
        # Draw skeleton connections
        self._draw_skeleton()
        
        # Update title to show time or frame number
        if fps_conversion_enabled:
            current_time = frame_to_time(self.current_frame)
            total_time = frame_to_time(self.total_frames - 1)
            time_label = get_time_label()
            
            if time_unit_var and time_unit_var.get() == "Minutes":
                title_text = f"Bodyparts Position - {time_label}: {current_time:.2f}/{total_time:.2f}"
            else:
                title_text = f"Bodyparts Position - {time_label}: {current_time:.1f}/{total_time:.1f}"
        else:
            title_text = f"Bodyparts Position - Frame {self.current_frame + 1}/{self.total_frames}"
        
        self.ax.set_title(title_text, fontsize=12, fontweight='bold', color='#2c3e50', pad=15)
        
        # Update frame label
        if fps_conversion_enabled:
            current_time = frame_to_time(self.current_frame)
            total_time = frame_to_time(self.total_frames - 1)
            
            if time_unit_var and time_unit_var.get() == "Minutes":
                frame_text = f"{current_time:.2f}/{total_time:.2f}"
            else:
                frame_text = f"{current_time:.1f}/{total_time:.1f}"
        else:
            frame_text = f"{self.current_frame + 1}/{self.total_frames}"
        
        self.frame_label.config(text=frame_text)
        
        # Use blit for fast redraw (if supported)
        try:
            self.canvas.draw_idle()
        except:
            self.canvas.draw()
    
    def _draw_skeleton(self):
        """Draw skeleton connections"""
        global skeleton_connections
        
        if not skeleton_connections or not self.data:
            return
        
        # Draw skeleton connections
        for connection in skeleton_connections:
            bodypart1, bodypart2 = connection
            
            # Check if both bodyparts are in data
            if bodypart1 in self.data and bodypart2 in self.data:
                data1 = self.data[bodypart1]
                data2 = self.data[bodypart2]
                
                # Check if current frame is valid
                if (self.current_frame < len(data1['x']) and self.current_frame < len(data1['y']) and
                    self.current_frame < len(data2['x']) and self.current_frame < len(data2['y'])):
                    
                    x1, y1 = data1['x'][self.current_frame], data1['y'][self.current_frame]
                    x2, y2 = data2['x'][self.current_frame], data2['y'][self.current_frame]
                    
                    # Draw connection
                    line = self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.8)[0]
                    self._scatter_plots.append(line)
    
    def _initialize_plot(self):
        """Initialize static elements of the plot area"""
        self.ax.clear()
        
        # Set graph properties
        self.ax.set_xlabel("X Coordinate", fontsize=11, fontweight='bold', color='#2c3e50')
        self.ax.set_ylabel("Y Coordinate", fontsize=11, fontweight='bold', color='#2c3e50')
        
        # Set grid and background
        self.ax.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')
        self.ax.set_facecolor('#ffffff')
        
        # Set axis range and save original limits
        if self.data:
            all_x = []
            all_y = []
            for data in self.data.values():
                all_x.extend(data['x'])
                all_y.extend(data['y'])
            
            if all_x and all_y:
                margin_x = (max(all_x) - min(all_x)) * 0.1
                margin_y = (max(all_y) - min(all_y)) * 0.1
                self.original_xlim = (min(all_x) - margin_x, max(all_x) + margin_x)
                self.original_ylim = (min(all_y) - margin_y, max(all_y) + margin_y)
                self.ax.set_xlim(self.original_xlim)
                self.ax.set_ylim(self.original_ylim)
        
        # Set axis style
        self.ax.tick_params(colors='#2c3e50', labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1)
        
        self._scatter_plots = []
    
    def start_move(self, event):
        """Start dragging window"""
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.start_win_x = self.window_frame.winfo_x()
        self.start_win_y = self.window_frame.winfo_y()
    
    def do_move(self, event):
        """Drag window"""
        x = self.start_win_x + (event.x_root - self.start_x)
        y = self.start_win_y + (event.y_root - self.start_y)
        # Limit window from being dragged out of parent window
        parent_width = self.parent_frame.winfo_width()
        parent_height = self.parent_frame.winfo_height()
        window_width = self.window_frame.winfo_width()
        window_height = self.window_frame.winfo_height()
        
        x = max(0, min(x, parent_width - window_width))
        y = max(0, min(y, parent_height - window_height))
        
        self.window_frame.place(x=x, y=y)
    
    def minimize_window(self):
        """Minimize window"""
        if hasattr(self, 'is_minimized') and self.is_minimized:
            # Restore window
            self.window_frame.place(width=self.window_width, height=self.window_height)
            self.is_minimized = False
        else:
            # Save current window size
            self.window_width = self.window_frame.winfo_width()
            self.window_height = self.window_frame.winfo_height()
            # Minimize window
            self.window_frame.place(width=self.window_width, height=35)
            self.is_minimized = True
    
    def start_resize(self, event):
        """Start resizing window"""
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.start_width = self.window_frame.winfo_width()
        self.start_height = self.window_frame.winfo_height()
    
    def do_resize(self, event):
        """Resize window"""
        new_width = self.start_width + (event.x_root - self.start_x)
        new_height = self.start_height + (event.y_root - self.start_y)
        
        # Set minimum and maximum dimensions
        min_width, min_height = 400, 300
        max_width = self.parent_frame.winfo_width() - self.window_frame.winfo_x()
        max_height = self.parent_frame.winfo_height() - self.window_frame.winfo_y()
        
        new_width = max(min_width, min(new_width, max_width))
        new_height = max(min_height, min(new_height, max_height))
        
        self.window_frame.place(width=new_width, height=new_height)
        
        # Update matplotlib figure size
        if hasattr(self, 'fig'):
            self.fig.set_size_inches((new_width-100)/100, (new_height-200)/100)
            self.canvas.draw()
    
    def export_video(self):
        """Export animation as MP4 video file"""
        if not CV2_AVAILABLE:
            log_message("OpenCV not installed, cannot export video.\nPlease run 'pip install opencv-python' to install.", "ERROR")
            return
        
        # Select save path
        file_path = filedialog.asksaveasfilename(
            title="Save Video File",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi")]
        )
        
        if not file_path:
            return
        
        try:
            # Disable export button to prevent repeated clicks
            self.export_btn.config(state="disabled", text="Exporting...")
            
            # Create progress dialog
            progress_window = tk.Toplevel(self.window_frame)
            progress_window.title("Export Progress")
            progress_window.geometry("300x100")
            progress_window.resizable(False, False)
            progress_window.grab_set()  # Modal dialog
            
            progress_label = tk.Label(progress_window, text="Exporting video, please wait...")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, length=250, mode='determinate')
            progress_bar.pack(pady=10)
            progress_bar['maximum'] = self.total_frames
            
            # Execute export in separate thread
            def export_thread():
                try:
                    # Set video parameters
                    fps = min(self.fps, 30)  # Limit max FPS to 30
                    width, height = 800, 600
                    
                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                    
                    # Save current frame position
                    original_frame = self.current_frame
                    
                    # Generate video frame by frame
                    for frame_idx in range(self.total_frames):
                        self.current_frame = frame_idx
                        
                        # Create temporary figure for export
                        temp_fig = plt.figure(figsize=(10, 7.5), dpi=80)
                        temp_ax = temp_fig.add_subplot(111)
                        
                        # Draw current frame
                        self._draw_frame_for_export(temp_ax, frame_idx)
                        
                        # Convert matplotlib figure to numpy array
                        temp_fig.canvas.draw()
                        buf = np.frombuffer(temp_fig.canvas.tostring_rgb(), dtype=np.uint8)
                        buf = buf.reshape(temp_fig.canvas.get_width_height()[::-1] + (3,))
                        
                        # Resize image and convert color format
                        frame = cv2.resize(buf, (width, height))
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # Write video frame
                        out.write(frame)
                        
                        # Update progress bar
                        progress_bar['value'] = frame_idx + 1
                        progress_window.update()
                        
                        plt.close(temp_fig)
                    
                    # Release resources
                    out.release()
                    
                    # Restore original frame position
                    self.current_frame = original_frame
                    self.progress_var.set(self.current_frame)
                    self.update_plot()
                    
                    # Close progress dialog
                    progress_window.destroy()
                    
                    # Show success message
                    log_message(f"Video successfully exported to:\n{file_path}", "INFO")
                    
                except Exception as e:
                    progress_window.destroy()
                    log_message(f"Video export failed:\n{str(e)}", "ERROR")
                
                finally:
                    # Re-enable export button
                    self.export_btn.config(state="normal", text="🎬 Export Video")
            
            # Start export thread
            export_thread_obj = threading.Thread(target=export_thread)
            export_thread_obj.daemon = True
            export_thread_obj.start()
            
        except Exception as e:
            log_message(f"Export initialization failed:\n{str(e)}", "ERROR")
            self.export_btn.config(state="normal", text="🎬 Export Video")
    
    def _draw_frame_for_export(self, ax, frame_idx):
        """Draw single frame for video export"""
        ax.clear()
        
        # Set graph properties
        ax.set_title(f"Bodyparts Position - Frame {frame_idx + 1}/{self.total_frames}", 
                    fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
        ax.set_xlabel("X Coordinate", fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_ylabel("Y Coordinate", fontsize=12, fontweight='bold', color='#2c3e50')
        
        # Set grid and background
        ax.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')
        ax.set_facecolor('#ffffff')
        
        # Use predefined colors
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                 '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#95a5a6']
        
        # Draw skeleton connections (draw before points to avoid occlusion)
        global skeleton_connections
        if skeleton_connections and self.data:
            for connection in skeleton_connections:
                bodypart1, bodypart2 = connection
                
                # Check if both bodyparts are in data
                if bodypart1 in self.data and bodypart2 in self.data:
                    data1 = self.data[bodypart1]
                    data2 = self.data[bodypart2]
                    
                    # Check if current frame is valid
                    if (frame_idx < len(data1['x']) and frame_idx < len(data1['y']) and
                        frame_idx < len(data2['x']) and frame_idx < len(data2['y'])):
                        
                        x1, y1 = data1['x'][frame_idx], data1['y'][frame_idx]
                        x2, y2 = data2['x'][frame_idx], data2['y'][frame_idx]
                        
                        # Draw connection
                        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.8)
        
        # Draw current frame data points
        for i, (bodypart, data) in enumerate(self.data.items()):
            if frame_idx < len(data['x']) and frame_idx < len(data['y']):
                x = data['x'][frame_idx]
                y = data['y'][frame_idx]
                
                # Draw point
                color = colors[i % len(colors)]
                ax.scatter(x, y, s=120, alpha=0.8, 
                          color=color, edgecolors='black', linewidth=2)
                
                # Add number inside circle
                ax.text(x, y, str(i+1), ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white')
                
                # Add label
                ax.annotate(f"{i+1}. {bodypart}", (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=9, color=color, fontweight='bold')
        
        # Set axis range
        if self.data:
            all_x = []
            all_y = []
            for data in self.data.values():
                all_x.extend(data['x'])
                all_y.extend(data['y'])
            
            if all_x and all_y:
                margin_x = (max(all_x) - min(all_x)) * 0.1
                margin_y = (max(all_y) - min(all_y)) * 0.1
                ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
                ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
        
        # Set axis style
        ax.tick_params(colors='#2c3e50', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#bdc3c7')
            spine.set_linewidth(1)
    
    def close_window(self):
        global visualization_window, central_label
        
        self.pause()
        
        if hasattr(self, 'after_id'):
            try:
                self.window_frame.after_cancel(self.after_id)
            except:
                pass
        
        self.window_frame.destroy()
        visualization_window = None

        if 'central_label' in globals():
            try:
                if hasattr(central_label, 'winfo_exists') and central_label.winfo_exists():
                    central_label.pack(pady=20)
                else:
                    central_label = tk.Label(central_display_frame, text="Central Display Area\nBodyparts position visualization will be shown after reading CSV file", bg="#f8f8f8", fg="#666666")
                    central_label.pack(pady=20)
            except tk.TclError:
                central_label = tk.Label(central_display_frame, text="Central Display Area\nBodyparts position visualization will be shown after reading CSV file", bg="#f8f8f8", fg="#666666")
                central_label.pack(pady=20)

class FiberVisualizationWindow:
    def __init__(self, parent_frame, animal_data=None, target_signal="470"):
        self.parent_frame = parent_frame
        self.animal_data = animal_data
        self.is_minimized = False
        self.window_width = 615
        self.window_height = 470
        self.plot_type = "raw"
        self.target_signal = target_signal
        
        self.pan_start = None
        self.zoom_factor = 1.0
        self._plot_initialized = False
        self.original_xlim = None
        self.original_ylim = None
        self.is_panning = False

        self.running_analysis_results = {}
        self.current_analysis_type = None
        
        self.create_window()
        self.update_plot()
        self.create_preprocessing_controls()
        
    def create_window(self):
        self.window_frame = tk.Frame(self.parent_frame, bg="#f5f5f5", relief=tk.RAISED, bd=1)
        self.window_frame.place(x=800, y=0, width=self.window_width, height=self.window_height)
        
        self.window_frame.bind("<Button-1>", self.start_move)
        self.window_frame.bind("<B1-Motion>", self.do_move)
        
        title_frame = tk.Frame(self.window_frame, bg="#f5f5f5", height=25)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        title_frame.bind("<Button-1>", self.start_move)
        title_frame.bind("<B1-Motion>", self.do_move)
        
        title_label = tk.Label(title_frame, text="Fiber Photometry Data", bg="#f5f5f5", fg="#666666", 
                              font=("Microsoft YaHei", 9))
        title_label.pack(side=tk.LEFT, padx=10, pady=3)
        title_label.bind("<Button-1>", self.start_move)
        title_label.bind("<B1-Motion>", self.do_move)
        
        btn_frame = tk.Frame(title_frame, bg="#f5f5f5")
        btn_frame.pack(side=tk.RIGHT, padx=5, pady=2)
        
        minimize_btn = tk.Button(btn_frame, text="−", bg="#f5f5f5", fg="#999999", bd=0, 
                               font=("Arial", 8), width=2, height=1,
                               command=self.minimize_window, relief=tk.FLAT)
        minimize_btn.pack(side=tk.LEFT, padx=1)
        
        close_btn = tk.Button(btn_frame, text="×", bg="#f5f5f5", fg="#999999", bd=0, 
                             font=("Arial", 8), width=2, height=1,
                             command=self.close_window, relief=tk.FLAT)
        close_btn.pack(side=tk.LEFT, padx=1)

        reset_view_btn = tk.Button(btn_frame, text="🗘", bg="#f5f5f5", fg="#999999", bd=0,
                                 font=("Arial", 8), width=2, height=1,
                                 command=self.reset_view, relief=tk.FLAT)
        reset_view_btn.pack(side=tk.LEFT, padx=1)
        
        resize_frame = tk.Frame(self.window_frame, bg="#bdc3c7", width=15, height=15)
        resize_frame.place(relx=1.0, rely=1.0, anchor="se")
        resize_frame.bind("<Button-1>", self.start_resize)
        resize_frame.bind("<B1-Motion>", self.do_resize)
        resize_frame.config(cursor="size_nw_se")
        
        self.fig = Figure(figsize=(3, 1), dpi=90, facecolor='#f8f9fa')
        self.ax = self.fig.add_subplot(111, facecolor='#ffffff')
        
        canvas_frame = tk.Frame(self.window_frame, bg="#f5f5f5", relief=tk.SUNKEN, bd=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        for child in toolbar_frame.winfo_children():
            if isinstance(child, tk.Button):
                child.config(bg="#f5f5f5", fg="#666666", bd=0, padx=4, pady=2,
                            activebackground="#e0e0e0", activeforeground="#000000")

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        # self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
    
    def create_preprocessing_controls(self):
        control_frame = tk.Frame(self.window_frame, bg="#ecf0f1")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        button_width = 12
        button_height = 1

        tk.Button(control_frame, text="Raw", command=lambda: self.set_plot_type("raw"),
                bg="#3498db", fg="white", font=("Microsoft YaHei", 8), width=button_width, height=button_height).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(control_frame, text="Smoothed", command=lambda: self.set_plot_type("smoothed"),
                bg="#e67e22", fg="white", font=("Microsoft YaHei", 8), width=button_width, height=button_height).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(control_frame, text="Baseline Corr", command=lambda: self.set_plot_type("baseline_corrected"),
                bg="#2ecc71", fg="white", font=("Microsoft YaHei", 8), width=button_width, height=button_height).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(control_frame, text="Motion Corr", command=lambda: self.set_plot_type("motion_corrected"),
                bg="#9b59b6", fg="white", font=("Microsoft YaHei", 8), width=button_width, height=button_height).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(control_frame, text="ΔF/F", command=lambda: self.set_plot_type("dff"),
                bg="#f39c12", fg="white", font=("Microsoft YaHei", 8), width=button_width, height=button_height).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(control_frame, text="Z-Score", command=lambda: self.set_plot_type("zscore"),
                bg="#e74c3c", fg="white", font=("Microsoft YaHei", 8), width=button_width, height=button_height).pack(side=tk.LEFT, padx=2, pady=2)
        
    def on_click(self, event):
        """Handle mouse click event for panning"""
        if event.inaxes == self.ax and event.button == 1:
            self.pan_start = (event.xdata, event.ydata)
            self.is_panning = True
            
    def on_release(self, event):
        """Handle mouse release event"""
        if event.button == 1:
            self.pan_start = None
            self.is_panning = False
            
    def on_motion(self, event):
        """Handle mouse motion for panning"""
        if not self.is_panning or event.inaxes != self.ax or self.pan_start is None:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        dx = event.xdata - self.pan_start[0]
        dy = event.ydata - self.pan_start[1]
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        new_xlim = (xlim[0] - dx, xlim[1] - dx)
        new_ylim = (ylim[0] - dy, ylim[1] - dy)
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        self.canvas.draw_idle()
        
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            log_message("Scroll event outside axes, ignoring.", "WARNING")
            return
            
        try:
            current_time = getattr(self, '_last_scroll_time', 0)
            if time.time() - current_time < 0.05:
                return
            self._last_scroll_time = time.time()
            
            if event.xdata is None or event.ydata is None:
                return
                
            x, y = event.xdata, event.ydata
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            if not self._is_valid_range(xlim, ylim):
                self.reset_view()
                return
            
            zoom_factor = 1.1 if event.button == 'up' else 0.9
            
            new_xlim = (x - (x - xlim[0]) * zoom_factor, 
                    x + (xlim[1] - x) * zoom_factor)
            new_ylim = (y - (y - ylim[0]) * zoom_factor, 
                    y + (ylim[1] - y) * zoom_factor)
            
            if self._is_valid_range(new_xlim, new_ylim):
                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)
                self.canvas.draw_idle()
                
        except Exception as e:
            log_message(f"Scroll error: {str(e)}", "WARNING")
            self.reset_view()

    def _is_valid_range(self, xlim, ylim):
        try:
            return (all(np.isfinite(xlim)) and all(np.isfinite(ylim)) and
                    xlim[1] > xlim[0] and ylim[1] > ylim[0] and
                    abs(xlim[1] - xlim[0]) > 1e-10 and
                    abs(ylim[1] - ylim[0]) > 1e-10 and
                    abs(xlim[1] - xlim[0]) < 1e10 and
                    abs(ylim[1] - ylim[0]) < 1e10)
        except:
            return False
        
    def reset_view(self):
        """Reset the view to original zoom and pan"""
        if self.original_xlim and self.original_ylim:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.zoom_factor = 1.0
            self.canvas.draw_idle()
            log_message("Fiber view reset to original", "INFO")

    def update_plot(self):
        self.ax.clear()
        
        if self.animal_data:
            fiber_data = self.animal_data.get('fiber_data_trimmed') if 'fiber_data_trimmed' in self.animal_data else self.animal_data.get('fiber_data')
            channels = self.animal_data.get('channels', {})
            active_channels = self.animal_data.get('active_channels', [])
            channel_data = self.animal_data.get('channel_data', {})
            preprocessed_data = self.animal_data.get('preprocessed_data', {})
            dff_data = self.animal_data.get('dff_data', {})
            zscore_data = self.animal_data.get('zscore_data', {})
        else:
            fiber_data = globals().get('fiber_data_trimmed') or globals().get('fiber_data')
            channels = globals().get('channels', {})
            active_channels = globals().get('active_channels', [])
            channel_data = globals().get('channel_data', {})
            preprocessed_data = globals().get('preprocessed_data', {})
            dff_data = globals().get('dff_data', {})
            zscore_data = globals().get('zscore_data', {})
        
        if fiber_data is None or not active_channels:
            self.ax.text(0.5, 0.5, "No fiber data available\nPlease load fiber data first", 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
            self.ax.set_title("Fiber Photometry Data - No Data Available")
            self.canvas.draw()
            return
        
        time_col = channels.get('time')
        if time_col is None or time_col not in fiber_data.columns:
            self.ax.text(0.5, 0.5, f"Time column '{time_col}' not found in fiber data", 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=10)
            self.ax.set_title("Fiber Photometry Data - Error")
            self.canvas.draw()
            return
        
        time_data = fiber_data[time_col]
            
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        all_time_data = []
        all_value_data = []
        
        has_plotted_data = False

        if self.plot_type == "raw":
            has_data = False
            for i, channel_num in enumerate(active_channels):
                if channel_num in channel_data:
                    for wavelength, col_name in channel_data[channel_num].items():
                        if col_name and col_name in fiber_data.columns:
                            color = colors[i % len(colors)]
                            alpha = 1.0 if wavelength == "470" else 0.6
                            linewidth = 1.5 if wavelength == "470" else 1.0
                            self.ax.plot(time_data, fiber_data[col_name], color=color, alpha=alpha, 
                                    linewidth=linewidth, label=f'CH{channel_num} {wavelength}nm')
                            all_time_data.extend(time_data)
                            all_value_data.extend(fiber_data[col_name].values)
                            has_data = True
                            has_plotted_data = True
            
            if not has_data:
                self.ax.text(0.5, 0.5, "No raw data columns found\nCheck channel configuration", 
                            ha='center', va='center', transform=self.ax.transAxes, fontsize=10)
            
            self.ax.set_title("Fiber Photometry Data - Raw Signals", fontsize=12, fontweight='bold')
        
        elif self.plot_type == "smoothed":
            if self.animal_data:
                data_source = self.animal_data.get('preprocessed_data', pd.DataFrame())
            else:
                data_source = globals().get('preprocessed_data', pd.DataFrame())
                
            for i, channel_num in enumerate(active_channels):
                if channel_num in channel_data:
                    smoothed_cols = [
                        f"CH{channel_num}_470_smoothed",
                        f"CH{channel_num}_560_smoothed",
                        f"CH{channel_num}_{self.target_signal}_smoothed" if hasattr(self, 'target_signal') else None
                    ]
                    
                    for col in smoothed_cols:
                        if col and col in data_source.columns:
                            color = colors[i % len(colors)]
                            line = self.ax.plot(time_data, data_source[col], color=color,
                                        label=f'CH{channel_num} Smoothed')[0]
                            all_time_data.extend(time_data)
                            all_value_data.extend(data_source[col].values)
                            has_plotted_data = True
                            break
            self.ax.set_title("Fiber Photometry Data - Smoothed")
        
        elif self.plot_type == "baseline_corrected":
            if self.animal_data:
                data_source = self.animal_data.get('preprocessed_data', pd.DataFrame())
            else:
                data_source = globals().get('preprocessed_data', pd.DataFrame())
                
            for i, channel_num in enumerate(active_channels):
                baseline_col = f"CH{channel_num}_baseline_corrected"
                if baseline_col in data_source.columns:
                    color = colors[i % len(colors)]
                    line = self.ax.plot(time_data, data_source[baseline_col], color=color,
                                label=f'CH{channel_num} Baseline Corrected')[0]
                    all_time_data.extend(time_data)
                    all_value_data.extend(data_source[baseline_col].values)
                    has_plotted_data = True
            self.ax.set_title("Fiber Photometry Data - Baseline Corrected")
        
        elif self.plot_type == "motion_corrected":
            if self.animal_data:
                data_source = self.animal_data.get('preprocessed_data', pd.DataFrame())
            else:
                data_source = globals().get('preprocessed_data', pd.DataFrame())
                
            for i, channel_num in enumerate(active_channels):
                motion_col = f"CH{channel_num}_motion_corrected"
                if motion_col in data_source.columns:
                    color = colors[i % len(colors)]
                    line = self.ax.plot(time_data, data_source[motion_col], color=color,
                                label=f'CH{channel_num} Motion Corrected')[0]
                    all_time_data.extend(time_data)
                    all_value_data.extend(data_source[motion_col].values)
                    has_plotted_data = True
            self.ax.set_title("Fiber Photometry Data - Motion Corrected")
        
        elif self.plot_type == "dff":
            if self.animal_data and 'dff_data' in self.animal_data:
                dff_data = self.animal_data['dff_data']
                if hasattr(dff_data, 'empty') and dff_data.empty:
                    dff_data = {}
                elif not isinstance(dff_data, (dict, pd.Series)):
                    dff_data = {}
            else:
                dff_data = {}
                
            if dff_data is None or (hasattr(dff_data, 'empty') and dff_data.empty) or (isinstance(dff_data, dict) and not dff_data):
                if self.animal_data and 'preprocessed_data' in self.animal_data:
                    data_source = self.animal_data['preprocessed_data']
                else:
                    data_source = globals().get('preprocessed_data', pd.DataFrame())
                    
                for i, channel_num in enumerate(active_channels):
                    dff_col = f"CH{channel_num}_dff"
                    if dff_col in data_source.columns:
                        color = colors[i % len(colors)]
                        line = self.ax.plot(time_data, data_source[dff_col], color=color,
                                    label=f'CH{channel_num} ΔF/F')[0]
                        all_time_data.extend(time_data)
                        all_value_data.extend(data_source[dff_col].values)
                        has_plotted_data = True
            else:
                for i, channel_num in enumerate(active_channels):
                    if str(channel_num) in dff_data:
                        color = colors[i % len(colors)]
                        line = self.ax.plot(time_data, dff_data[str(channel_num)], color=color,
                                    label=f'CH{channel_num} ΔF/F')[0]
                        all_time_data.extend(time_data)
                        all_value_data.extend(dff_data[str(channel_num)].values)
                    else:
                        dff_col = f"CH{channel_num}_dff"
                        if self.animal_data and 'preprocessed_data' in self.animal_data and dff_col in self.animal_data['preprocessed_data'].columns:
                            color = colors[i % len(colors)]
                            line = self.ax.plot(time_data, self.animal_data['preprocessed_data'][dff_col], color=color,
                                        label=f'CH{channel_num} ΔF/F')[0]
                            all_time_data.extend(time_data)
                            all_value_data.extend(self.animal_data['preprocessed_data'][dff_col].values)
                            has_plotted_data = True
                        elif 'preprocessed_data' in globals() and dff_col in globals()['preprocessed_data'].columns:
                            color = colors[i % len(colors)]
                            line = self.ax.plot(time_data, globals()['preprocessed_data'][dff_col], color=color,
                                        label=f'CH{channel_num} ΔF/F')[0]
                            all_time_data.extend(time_data)
                            all_value_data.extend(globals()['preprocessed_data'][dff_col].values)
                            has_plotted_data = True
            self.ax.set_title("Fiber Photometry Data - ΔF/F")
        
        elif self.plot_type == "zscore":
            if self.animal_data and 'zscore_data' in self.animal_data:
                zscore_data = self.animal_data['zscore_data']
                if hasattr(zscore_data, 'empty') and zscore_data.empty:
                    zscore_data = {}
                elif not isinstance(zscore_data, (dict, pd.Series)):
                    zscore_data = {}
            else:
                zscore_data = {}
                
            if zscore_data is None or (hasattr(zscore_data, 'empty') and zscore_data.empty) or (isinstance(zscore_data, dict) and not zscore_data):
                if self.animal_data and 'preprocessed_data' in self.animal_data:
                    data_source = self.animal_data['preprocessed_data']
                else:
                    data_source = globals().get('preprocessed_data', pd.DataFrame())
                    
                for i, channel_num in enumerate(active_channels):
                    zscore_col = f"CH{channel_num}_zscore"
                    if zscore_col in data_source.columns:
                        color = colors[i % len(colors)]
                        line = self.ax.plot(time_data, data_source[zscore_col], color=color,
                                    label=f'CH{channel_num} Z-Score')[0]
                        all_time_data.extend(time_data)
                        all_value_data.extend(data_source[zscore_col].values)
                        has_plotted_data = True
            else:
                for i, channel_num in enumerate(active_channels):
                    if str(channel_num) in zscore_data:
                        color = colors[i % len(colors)]
                        line = self.ax.plot(time_data, zscore_data[str(channel_num)], color=color,
                                    label=f'CH{channel_num} Z-Score')[0]
                        all_time_data.extend(time_data)
                        all_value_data.extend(zscore_data[str(channel_num)].values)
                        has_plotted_data = True
                    else:
                        zscore_col = f"CH{channel_num}_zscore"
                        if self.animal_data and 'preprocessed_data' in self.animal_data and zscore_col in self.animal_data['preprocessed_data'].columns:
                            color = colors[i % len(colors)]
                            line = self.ax.plot(time_data, self.animal_data['preprocessed_data'][zscore_col], color=color,
                                        label=f'CH{channel_num} Z-Score')[0]
                            all_time_data.extend(time_data)
                            all_value_data.extend(self.animal_data['preprocessed_data'][zscore_col].values)
                            has_plotted_data = True
                        elif 'preprocessed_data' in globals() and zscore_col in globals()['preprocessed_data'].columns:
                            color = colors[i % len(colors)]
                            line = self.ax.plot(time_data, globals()['preprocessed_data'][zscore_col], color=color,
                                        label=f'CH{channel_num} Z-Score')[0]
                            all_time_data.extend(time_data)
                            all_value_data.extend(globals()['preprocessed_data'][zscore_col].values)
                            has_plotted_data = True
            self.ax.set_title("Fiber Photometry Data - Z-Score")
        
        if self.running_analysis_results and self.current_analysis_type:
            self._plot_running_analysis_markers(time_data)
        
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("ΔF/F" if self.plot_type == "dff" else "Z-Score" if self.plot_type == "zscore" else "Fluorescence")
        if has_plotted_data:
            self.ax.legend()
        self.ax.grid(False)
        
        if all_time_data and all_value_data:
            min_time = min(all_time_data)
            max_time = max(all_time_data)
            min_value = min(all_value_data)
            max_value = max(all_value_data)
            
            time_margin = (max_time - min_time) * 0.05
            value_margin = (max_value - min_value) * 0.1
            
            self.original_xlim = (min_time - time_margin, max_time + time_margin)
            self.original_ylim = (min_value - value_margin, max_value + value_margin)
            
            if not hasattr(self, '_plot_initialized') or not self._plot_initialized:
                self.ax.set_xlim(self.original_xlim)
                self.ax.set_ylim(self.original_ylim)
                self._plot_initialized = True
        
        self.canvas.draw()

    def _plot_running_analysis_markers(self, time_data):
        if not self.current_analysis_type or self.current_analysis_type not in self.running_analysis_results:
            return
        
        analysis_data = self.running_analysis_results[self.current_analysis_type]
        if not analysis_data:
            return
        
        analysis_styles = {
            'movement_periods': {'color': 'green', 'alpha': 0.3, 'label': 'Movement'},
            'rest_periods': {'color': 'red', 'alpha': 0.3, 'label': 'Rest'},
            'continuous_locomotion_periods': {'color': 'blue', 'alpha': 0.3, 'label': 'Locomotion'},
            'general_onsets': {'color': 'orange', 'alpha': 0.8, 'label': 'Onset'},
            'jerks': {'color': 'purple', 'alpha': 0.8, 'label': 'Jerks'},
            'locomotion_initiations': {'color': 'cyan', 'alpha': 0.8, 'label': 'Initiation'},
            'locomotion_terminations': {'color': 'magenta', 'alpha': 0.8, 'label': 'Termination'}
        }
        
        style = analysis_styles.get(self.current_analysis_type, {'color': 'gray', 'alpha': 0.5, 'label': 'Event'})
        
        if self.current_analysis_type in ['movement_periods', 'rest_periods', 'continuous_locomotion_periods']:
            for i, (start, end) in enumerate(analysis_data):
                self.ax.axvspan(start, end, alpha=style['alpha'], color=style['color'],
                              label=style['label'] if i == 0 else "")
        else:
            for i, event_time in enumerate(analysis_data):
                self.ax.axvline(x=event_time, color=style['color'], linestyle='--', 
                              alpha=style['alpha'], linewidth=2,
                              label=style['label'] if i == 0 else "")

    def update_running_analysis(self, analysis_type, analysis_data):
        self.current_analysis_type = analysis_type
        self.running_analysis_results[analysis_type] = analysis_data
        self.update_plot()

    # def clear_running_analysis(self):
    #     self.current_analysis_type = None
    #     self.running_analysis_results = {}
    #     self.update_plot()

    def set_plot_type(self, plot_type):
        self.plot_type = plot_type
        self.update_plot()

    def start_move(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.start_win_x = self.window_frame.winfo_x()
        self.start_win_y = self.window_frame.winfo_y()
    
    def do_move(self, event):
        x = self.start_win_x + (event.x_root - self.start_x)
        y = self.start_win_y + (event.y_root - self.start_y)
        parent_width = self.parent_frame.winfo_width()
        parent_height = self.parent_frame.winfo_height()
        window_width = self.window_frame.winfo_width()
        window_height = self.window_frame.winfo_height()
        
        x = max(0, min(x, parent_width - window_width))
        y = max(0, min(y, parent_height - window_height))
        
        self.window_frame.place(x=x, y=y)
    
    def minimize_window(self):
        if self.is_minimized:
            self.window_frame.place(width=self.window_width, height=self.window_height)
            self.is_minimized = False
        else:
            self.window_width = self.window_frame.winfo_width()
            self.window_height = self.window_frame.winfo_height()
            self.window_frame.place(width=self.window_width, height=35)
            self.is_minimized = True
    
    def start_resize(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.start_width = self.window_frame.winfo_width()
        self.start_height = self.window_frame.winfo_height()
    
    def do_resize(self, event):
        new_width = self.start_width + (event.x_root - self.start_x)
        new_height = self.start_height + (event.y_root - self.start_y)
        
        min_width, min_height = 400, 300
        max_width = self.parent_frame.winfo_width() - self.window_frame.winfo_x()
        max_height = self.parent_frame.winfo_height() - self.window_frame.winfo_y()
        
        new_width = max(min_width, min(new_width, max_width))
        new_height = max(min_height, min(new_height, max_height))
        
        self.window_frame.place(width=new_width, height=new_height)
        
        if hasattr(self, 'fig'):
            self.fig.set_size_inches((new_width-100)/100, (new_height-200)/100)
            self.canvas.draw()
    
    def close_window(self):
        global fiber_plot_window
        self.window_frame.destroy()
        fiber_plot_window = None

class RunningVisualizationWindow:
    def __init__(self, parent_frame, animal_data=None):
        self.parent_frame = parent_frame
        self.animal_data = animal_data
        self.is_minimized = False
        self.movement_bouts = []
        self.window_width = 615
        self.window_height = 400
        self.current_analysis_type = None
        self.analysis_data = None
        self.pan_start = None
        self.zoom_factor = 1.0
        self._plot_initialized = False
        self.original_xlim = None
        self.original_ylim = None
        self.is_panning = False
        
        self.create_window()
        self.update_plot()
        
    def create_window(self):
        self.window_frame = tk.Frame(self.parent_frame, bg="#f5f5f5", relief=tk.RAISED, bd=1)
        self.window_frame.place(x=800, y=470, width=self.window_width, height=self.window_height)
        
        self.window_frame.bind("<Button-1>", self.start_move)
        self.window_frame.bind("<B1-Motion>", self.do_move)
        
        title_frame = tk.Frame(self.window_frame, bg="#f5f5f5", height=25)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        title_frame.bind("<Button-1>", self.start_move)
        title_frame.bind("<B1-Motion>", self.do_move)
        
        title_label = tk.Label(title_frame, text="Threadmill Data", bg="#f5f5f5", fg="#666666", 
                              font=("Microsoft YaHei", 9))
        title_label.pack(side=tk.LEFT, padx=10, pady=3)
        title_label.bind("<Button-1>", self.start_move)
        title_label.bind("<B1-Motion>", self.do_move)
        
        btn_frame = tk.Frame(title_frame, bg="#f5f5f5")
        btn_frame.pack(side=tk.RIGHT, padx=5, pady=2)
        
        minimize_btn = tk.Button(btn_frame, text="−", bg="#f5f5f5", fg="#999999", bd=0, 
                               font=("Arial", 8), width=2, height=1,
                               command=self.minimize_window, relief=tk.FLAT)
        minimize_btn.pack(side=tk.LEFT, padx=1)
        
        close_btn = tk.Button(btn_frame, text="×", bg="#f5f5f5", fg="#999999", bd=0, 
                             font=("Arial", 8), width=2, height=1,
                             command=self.close_window, relief=tk.FLAT)
        close_btn.pack(side=tk.LEFT, padx=1)
        
        reset_view_btn = tk.Button(btn_frame, text="🗘", bg="#f5f5f5", fg="#999999", bd=0,
                                 font=("Arial", 8), width=2, height=1,
                                 command=self.reset_view, relief=tk.FLAT)
        reset_view_btn.pack(side=tk.LEFT, padx=1)
        
        resize_frame = tk.Frame(self.window_frame, bg="#bdc3c7", width=15, height=15)
        resize_frame.place(relx=1.0, rely=1.0, anchor="se")
        resize_frame.bind("<Button-1>", self.start_resize)
        resize_frame.bind("<B1-Motion>", self.do_resize)
        resize_frame.config(cursor="size_nw_se")
        
        self.fig = Figure(figsize=(3, 1), dpi=90, facecolor='#f8f9fa')
        self.ax = self.fig.add_subplot(111, facecolor='#ffffff')
        
        canvas_frame = tk.Frame(self.window_frame, bg="#f5f5f5", relief=tk.SUNKEN, bd=1)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        toolbar_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
        toolbar_frame.pack(fill=tk.X, padx=2, pady=(0,2))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        for child in toolbar_frame.winfo_children():
            if isinstance(child, tk.Button):
                child.config(bg="#f5f5f5", fg="#666666", bd=0, padx=4, pady=2,
                            activebackground="#e0e0e0", activeforeground="#000000")

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        # self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
    
    def on_click(self, event):
        """Handle mouse click event for panning"""
        if event.inaxes == self.ax and event.button == 1:
            self.pan_start = (event.xdata, event.ydata)
            self.is_panning = True
            
    def on_release(self, event):
        """Handle mouse release event"""
        if event.button == 1:
            self.pan_start = None
            self.is_panning = False
            
    def on_motion(self, event):
        """Handle mouse motion for panning"""
        if not self.is_panning or event.inaxes != self.ax or self.pan_start is None:
            return
            
        if event.xdata is None or event.ydata is None:
            return
            
        dx = event.xdata - self.pan_start[0]
        dy = event.ydata - self.pan_start[1]
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        new_xlim = (xlim[0] - dx, xlim[1] - dx)
        new_ylim = (ylim[0] - dy, ylim[1] - dy)
        
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        
        self.canvas.draw_idle()
        
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            log_message("Scroll event outside axes, ignoring.", "WARNING")
            return
            
        try:
            current_time = getattr(self, '_last_scroll_time', 0)
            if time.time() - current_time < 0.05:
                return
            self._last_scroll_time = time.time()
            
            if event.xdata is None or event.ydata is None:
                return
                
            x, y = event.xdata, event.ydata
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            if not self._is_valid_range(xlim, ylim):
                self.reset_view()
                return
            
            zoom_factor = 1.1 if event.button == 'up' else 0.9
            
            new_xlim = (x - (x - xlim[0]) * zoom_factor, 
                    x + (xlim[1] - x) * zoom_factor)
            new_ylim = (y - (y - ylim[0]) * zoom_factor, 
                    y + (ylim[1] - y) * zoom_factor)
            
            if self._is_valid_range(new_xlim, new_ylim):
                self.ax.set_xlim(new_xlim)
                self.ax.set_ylim(new_ylim)
                self.canvas.draw_idle()
                
        except Exception as e:
            log_message(f"Scroll error: {str(e)}", "WARNING")
            self.reset_view()

    def _is_valid_range(self, xlim, ylim):
        try:
            return (all(np.isfinite(xlim)) and all(np.isfinite(ylim)) and
                    xlim[1] > xlim[0] and ylim[1] > ylim[0] and
                    abs(xlim[1] - xlim[0]) > 1e-10 and
                    abs(ylim[1] - ylim[0]) > 1e-10 and
                    abs(xlim[1] - xlim[0]) < 1e10 and
                    abs(ylim[1] - ylim[0]) < 1e10)
        except:
            return False
        
    def reset_view(self):
        """Reset the view to original zoom and pan"""
        if self.original_xlim and self.original_ylim:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.zoom_factor = 1.0
            self.canvas.draw_idle()
            log_message("Running view reset to original", "INFO")
    
    def update_plot(self):
        self.ax.clear()
        
        if self.animal_data:
            ast2_data = self.animal_data.get('ast2_data_adjusted')
            processed_data = self.animal_data.get('running_processed_data')
            treadmill_behaviors = self.animal_data.get('treadmill_behaviors', {})
        else:
            ast2_data = globals().get('ast2_data_adjusted')
            processed_data = globals().get('running_processed_data')
            treadmill_behaviors = globals().get('treadmill_behaviors', {})
        
        if ast2_data is None:
            self.ax.text(0.5, 0.5, "No running data available", ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_title("Running Wheel Data - No Data")
        else:
            timestamps = ast2_data['data']['timestamps']
            
            if processed_data:
                speed = processed_data['filtered_speed']
                title_suffix = " (Filtered)"
            else:
                speed = ast2_data['data']['speed']
                title_suffix = ""
            
            # Plot speed data
            self.ax.plot(timestamps, speed, 'b-', label='Running Speed', linewidth=1, alpha=0.7)
            
            # Plot analysis data if available
            if self.current_analysis_type and self.analysis_data is not None:
                self.plot_analysis_data(timestamps, speed)
            
            # Set title based on current analysis
            if self.current_analysis_type:
                title = f"Running Data - {self.current_analysis_type.replace('_', ' ').title()}{title_suffix}"
            else:
                title = f"Running Wheel Data{title_suffix}"
                
            self.ax.set_title(title)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Speed (cm/s)")
            self.ax.legend()
            self.ax.grid(False)
            
            if len(speed) > 0 and len(timestamps) > 0:
                min_speed = min(speed)
                max_speed = max(speed)
                min_time = min(timestamps)
                max_time = max(timestamps)
                
                speed_margin = (max_speed - min_speed) * 0.1
                time_margin = (max_time - min_time) * 0.05
                
                if speed_margin < 0.1:
                    speed_margin = 0.1
                
                self.original_xlim = (min_time - time_margin, max_time + time_margin)
                self.original_ylim = (min_speed - speed_margin, max_speed + speed_margin)
                
                if not hasattr(self, '_plot_initialized') or not self._plot_initialized:
                    self.ax.set_xlim(self.original_xlim)
                    self.ax.set_ylim(self.original_ylim)
                    self._plot_initialized = True

        self.canvas.draw()
    
    def plot_analysis_data(self, timestamps, speed):
        """Plot different types of analysis data"""
        colors = {
            'movement_periods': 'green',
            'rest_periods': 'red', 
            'continuous_locomotion_periods': 'blue',
            'general_onsets': 'orange',
            'jerks': 'purple',
            'locomotion_initiations': 'cyan',
            'locomotion_terminations': 'magenta'
        }
        
        alpha_values = {
            'periods': 0.3,
            'events': 0.8
        }
        
        analysis_type = self.current_analysis_type
        data = self.analysis_data
        
        if not data:
            return
        
        color = colors.get(analysis_type, 'black')
        
        if analysis_type in ['movement_periods', 'rest_periods', 'continuous_locomotion_periods']:
            # Plot periods as shaded regions
            for i, (start, end) in enumerate(data):
                self.ax.axvspan(start, end, alpha=alpha_values['periods'], color=color,
                              label=analysis_type.replace('_', ' ').title() if i == 0 else "")
        else:
            # Plot events as vertical lines
            for i, event_time in enumerate(data):
                self.ax.axvline(x=event_time, color=color, linestyle='--', alpha=alpha_values['events'],
                              label=analysis_type.replace('_', ' ').title() if i == 0 else "")
    
    def start_move(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.start_win_x = self.window_frame.winfo_x()
        self.start_win_y = self.window_frame.winfo_y()
    
    def do_move(self, event):
        x = self.start_win_x + (event.x_root - self.start_x)
        y = self.start_win_y + (event.y_root - self.start_y)
        parent_width = self.parent_frame.winfo_width()
        parent_height = self.parent_frame.winfo_height()
        window_width = self.window_frame.winfo_width()
        window_height = self.window_frame.winfo_height()
        
        x = max(0, min(x, parent_width - window_width))
        y = max(0, min(y, parent_height - window_height))
        
        self.window_frame.place(x=x, y=y)
    
    def minimize_window(self):
        if self.is_minimized:
            self.window_frame.place(width=self.window_width, height=self.window_height)
            self.is_minimized = False
        else:
            self.window_width = self.window_frame.winfo_width()
            self.window_height = self.window_frame.winfo_height()
            self.window_frame.place(width=self.window_width, height=35)
            self.is_minimized = True
    
    def start_resize(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root
        self.start_width = self.window_frame.winfo_width()
        self.start_height = self.window_frame.winfo_height()
    
    def do_resize(self, event):
        new_width = self.start_width + (event.x_root - self.start_x)
        new_height = self.start_height + (event.y_root - self.start_y)
        
        min_width, min_height = 400, 300
        max_width = self.parent_frame.winfo_width() - self.window_frame.winfo_x()
        max_height = self.parent_frame.winfo_height() - self.window_frame.winfo_y()
        
        new_width = max(min_width, min(new_width, max_width))
        new_height = max(min_height, min(new_height, max_height))
        
        self.window_frame.place(width=new_width, height=new_height)
        
        if hasattr(self, 'fig'):
            self.fig.set_size_inches((new_width-100)/100, (new_height-200)/100)
            self.canvas.draw()
    
    def close_window(self):
        global running_plot_window
        self.window_frame.destroy()
        running_plot_window = None

def import_multi_animals():
    """Modified to support different experiment modes"""
    global selected_files, multi_animal_data, current_experiment_mode

    base_dir = filedialog.askdirectory(title="Select Free Moving/Behavioural directory")
    if not base_dir:
        return

    try:
        log_message("Scanning for multi-animal data...")

        date_dirs = glob.glob(os.path.join(base_dir, "20*"))
        for date_dir in date_dirs:
            if not os.path.isdir(date_dir):
                continue

            date_name = os.path.basename(date_dir)
            num_dirs = glob.glob(os.path.join(date_dir, "*"))
            for num_dir in num_dirs:
                if not os.path.isdir(num_dir):
                    continue
                ear_bar_dirs = glob.glob(os.path.join(num_dir, "*"))[0]
                ear_tag = os.path.basename(ear_bar_dirs)
                animal_id = f"{date_name}-{ear_tag}"

                files_found = {}
                patterns = {
                    'dlc': ['*dlc*.csv', '*deeplabcut*.csv'],
                    'fiber': ['fluorescence.csv', '*fiber*.csv'],
                    'ast2': ['*.ast2']
                }

                # Determine required files based on mode
                if current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2:
                    required_files = ['fiber', 'ast2']
                else:  # FIBER_AST2_DLC
                    required_files = ['dlc', 'fiber', 'ast2']

                for file_type, file_patterns in patterns.items():
                    # Skip DLC search if not needed
                    if file_type == 'dlc' and current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2:
                        continue
                    
                    found_file = None
                    for root_path, dirs, files in os.walk(num_dir):
                        for file in files:
                            file_lower = file.lower()
                            for pattern in file_patterns:
                                if fnmatch.fnmatch(file_lower, pattern.lower()):
                                    found_file = os.path.join(root_path, file)
                                    files_found[file_type] = found_file
                                    break
                            if found_file:
                                break
                        if found_file:
                            break

                animal_data = {
                    'animal_id': animal_id,
                    'files': files_found,
                    'processed': False,
                    'event_time_absolute': False,
                    'active_channels': [],
                    'experiment_mode': current_experiment_mode
                }

                if all(ft in files_found for ft in required_files):
                    if any(d['animal_id'] == animal_id for d in multi_animal_data):
                        log_message(f"Skip duplicate animal: {animal_id}")
                        continue

                    # Process DLC file (only if in full mode)
                    if current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC and 'dlc' in files_found:
                        try:
                            dlc_data = read_dlc_file(files_found['dlc'])
                            animal_data['dlc_data'] = dlc_data
                        except Exception as e:
                            log_message(f"Failed to load DLC for {animal_id}: {str(e)}", "ERROR")

                    # Process AST2 file
                    if 'ast2' in files_found:
                        try:
                            header, raw_data = h_AST2_readData(files_found['ast2'])
                            if running_channel < len(raw_data):
                                speed = h_AST2_raw2Speed(raw_data[running_channel], header, voltageRange=None)
                                ast2_data = {
                                    'header': header,
                                    'data': speed
                                }
                                animal_data['ast2_data'] = ast2_data
                            else:
                                log_message(f"Running channel {running_channel} out of range, using channel 2", "WARNING")
                                speed = h_AST2_raw2Speed(raw_data[2], header, voltageRange=None)
                                ast2_data = {
                                    'header': header,
                                    'data': speed
                                }
                                animal_data['ast2_data'] = ast2_data
                        except Exception as e:
                            log_message(f"Failed to load AST2 for {animal_id}: {str(e)}", "ERROR")

                    # Process fiber data
                    if 'fiber' in files_found:
                        try:
                            fiber_result = load_fiber_data(files_found['fiber'])
                            if fiber_result:
                                animal_data.update(fiber_result)
                        except Exception as e:
                            log_message(f"Failed to load fiber for {animal_id}: {str(e)}", "ERROR")

                    animal_data['processed'] = True
                    selected_files.append(animal_data)
                    file_listbox.insert(tk.END, f"{animal_id}")
                    multi_animal_data.append(animal_data)

        if not selected_files:
            log_message("No valid animal data found in the selected directory", "WARNING")
        else:
            mode_name = "Fiber+AST2" if current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2 else "Fiber+AST2+DLC"
            log_message(f"Found {len(selected_files)} animals in {mode_name} mode", "INFO")
            show_channel_selection_dialog()

    except Exception as e:
        log_message(f"Failed to import data: {str(e)}", "ERROR")

def import_single_animal():
    """Modified to support different experiment modes"""
    global selected_files, multi_animal_data, current_experiment_mode

    folder_path = filedialog.askdirectory(title="Select animal ear bar folder")
    if not folder_path:
        return

    try:
        folder_path = os.path.normpath(folder_path)
        parent_folder_path = os.path.dirname(folder_path)
        path_parts = folder_path.split(os.sep)
        if len(path_parts) < 4:
            log_message("Selected directory is not a valid animal data folder", "WARNING")
            return

        batch_name = path_parts[-3]
        ear_tag = path_parts[-1]
        animal_id = f"{batch_name}-{ear_tag}"

        patterns = {
            'dlc': ['*dlc*.csv', '*deeplabcut*.csv'],
            'fiber': ['fluorescence.csv', '*fiber*.csv'],
            'ast2': ['*.ast2']
        }

        # Determine required files based on mode
        if current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2:
            required_files = ['fiber', 'ast2']
        else:  # FIBER_AST2_DLC
            required_files = ['dlc', 'fiber', 'ast2']

        files_found = {}
        for file_type, file_patterns in patterns.items():
            # Skip DLC search if not needed
            if file_type == 'dlc' and current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2:
                continue
            
            found_file = None
            for root_path, dirs, files in os.walk(parent_folder_path):
                for file in files:
                    file_lower = file.lower()
                    for pattern in file_patterns:
                        if fnmatch.fnmatch(file_lower, pattern.lower()):
                            found_file = os.path.join(root_path, file)
                            files_found[file_type] = found_file
                            break
                    if found_file:
                        break
                if found_file:
                    break

        missing_files = [ft for ft in required_files if ft not in files_found]
        if missing_files:
            log_message(f"Required files missing: {', '.join(missing_files)}", "WARNING")
            return

        if any(d['animal_id'] == animal_id for d in multi_animal_data):
            log_message(f"Animal {animal_id} already exists, skipping duplicate import.", "INFO")
            return
        
        animal_data = {
            'animal_id': animal_id,
            'files': files_found,
            'processed': False,
            'event_time_absolute': False,
            'experiment_mode': current_experiment_mode
        }

        # Process DLC file (only if in full mode)
        if current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC and 'dlc' in files_found:
            try:
                dlc_data = read_dlc_file(files_found['dlc'])
                animal_data['dlc_data'] = dlc_data
                filename = os.path.basename(files_found['dlc'])
                match = re.search(r'cam(\d+)', filename)
                if match:
                    animal_data['cam_id'] = int(match.group(1))
            except Exception as e:
                log_message(f"Failed to load DLC for {animal_id}: {str(e)}", "ERROR")

        # Process AST2 file
        if 'ast2' in files_found:
            try:
                header, raw_data = h_AST2_readData(files_found['ast2'])
                if running_channel < len(raw_data):
                    speed = h_AST2_raw2Speed(raw_data[running_channel], header, voltageRange=None)
                    ast2_data = {
                        'header': header,
                        'data': speed
                    }
                    animal_data['ast2_data'] = ast2_data
                else:
                    log_message(f"Running channel {running_channel} out of range, using channel 2", "WARNING")
                    speed = h_AST2_raw2Speed(raw_data[2], header, voltageRange=None)
                    ast2_data = {
                        'header': header,
                        'data': speed
                    }
                    animal_data['ast2_data'] = ast2_data
            except Exception as e:
                log_message(f"Failed to load AST2 for {animal_id}: {str(e)}", "ERROR")

        # Process fiber data
        if 'fiber' in files_found:
            try:
                fiber_result = load_fiber_data(files_found['fiber'])
                if fiber_result:
                    animal_data.update(fiber_result)
            except Exception as e:
                log_message(f"Failed to load fiber for {animal_id}: {str(e)}", "ERROR")

        animal_data['processed'] = True
        selected_files.append(animal_data)
        multi_animal_data.append(animal_data)
        file_listbox.insert(tk.END, f"{animal_id} ")
        
        if 'fiber_data' in animal_data:
            show_channel_selection_dialog()
        
        mode_name = "Fiber+AST2" if current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2 else "Fiber+AST2+DLC"
        log_message(f"Successfully added animal: {animal_id} ({mode_name} mode)", "INFO")

    except Exception as e:
        log_message(f"Failed to add single animal: {str(e)}", "ERROR")

def clear_selected():
    selected_indices = file_listbox.curselection()
    for index in sorted(selected_indices, reverse=True):
        animal_id = file_listbox.get(index)
        file_listbox.delete(index)
        
        # Remove from selected_files and multi_animal_data
        if index < len(selected_files):
            animal_data = selected_files.pop(index)
            # Find and remove from multi_animal_data
            for i, data in enumerate(multi_animal_data):
                if data['animal_id'] == animal_data['animal_id']:
                    multi_animal_data.pop(i)
                    break

def clear_all():
    file_listbox.delete(0, tk.END)
    global selected_files
    selected_files = []
    global multi_animal_data
    multi_animal_data = []

def load_fiber_data(file_path=None):
    path = file_path
    try:
        fiber_data = pd.read_csv(path, skiprows=1, delimiter=',')
        fiber_data = fiber_data.loc[:, ~fiber_data.columns.str.contains('^Unnamed')]
        fiber_data.columns = fiber_data.columns.str.strip()

        time_col = None
        possible_time_columns = ['timestamp', 'timems', 'time', 'time(ms)']
        for col in fiber_data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in possible_time_columns):
                time_col = col
                break
        
        if not time_col:
            numeric_cols = fiber_data.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                time_col = numeric_cols[0]
        
        fiber_data[time_col] = fiber_data[time_col] / 1000
        
        events_col = None
        for col in fiber_data.columns:
            if 'event' in col.lower():
                events_col = col
                break
        
        channels = {'time': time_col, 'events': events_col}
        
        global channel_data
        channel_data = {}
        channel_pattern = re.compile(r'CH(\d+)-(\d+)', re.IGNORECASE)
        
        for col in fiber_data.columns:
            match = channel_pattern.match(col)
            if match:
                channel_num = int(match.group(1))
                wavelength = int(match.group(2))
                
                if channel_num not in channel_data:
                    channel_data[channel_num] = {'410': None, '415': None, '470': None, '560': None}
                
                if wavelength == 410 or wavelength == 415:
                    channel_data[channel_num]['410' if wavelength == 410 else '415'] = col
                elif wavelength == 470:
                    channel_data[channel_num]['470'] = col
                elif wavelength == 560:
                    channel_data[channel_num]['560'] = col
        
        log_message(f"Fiber data loaded, {len(channel_data)} channels detected", "INFO")
        
        return {
            'fiber_data': fiber_data,
            'channels': channels,
            'channel_data': channel_data
        }
    except Exception as e:
        log_message(f"Failed to load fiber data: {str(e)}", "ERROR")
        return None
    
def show_channel_selection_dialog():
    dialog = tk.Toplevel(root)
    dialog.title("Select Channels")
    dialog.geometry("290x150")
    dialog.transient(root)
    dialog.grab_set()
    
    main_frame = ttk.Frame(dialog, padding=5)
    main_frame.pack(fill="both", expand=True)
    
    canvas = tk.Canvas(main_frame)
    canvas.config(height=50, width=100)
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    global channel_vars
    channel_vars = {}
    
    if multi_animal_data:
        for animal_data in multi_animal_data:
            if 'channel_data' not in animal_data:
                continue
                
            animal_id = animal_data['animal_id']
            group = animal_data.get('group', '')
            label = f"{animal_id} ({group})" if group else animal_id
            
            animal_frame = ttk.LabelFrame(scrollable_frame, text=label)
            animal_frame.pack(fill="x", padx=5, pady=5, ipadx=5, ipady=5)
            
            channel_vars[animal_id] = {}
            
            saved_channels = channel_memory.get(animal_id, [])
            
            for channel_num in sorted(animal_data['channel_data'].keys()):
                default_value = channel_num in saved_channels or (not saved_channels and channel_num == 1)
                var = tk.BooleanVar(value=default_value)
                channel_vars[animal_id][channel_num] = var
                
                chk = ttk.Checkbutton(
                    animal_frame, 
                    text=f"Channel {channel_num}",
                    variable=var
                )
                chk.pack(anchor="w", padx=2, pady=2)
    else:
        if hasattr('channel_data') and channel_data:
            animal_frame = ttk.LabelFrame(scrollable_frame, text="Single Animal")
            animal_frame.pack(fill="x", padx=5, pady=5, ipadx=5, ipady=5)
            
            channel_vars['single'] = {}
            
            saved_channels = channel_memory.get('single', [])
            
            for channel_num in sorted(channel_data.keys()):
                default_value = channel_num in saved_channels or (not saved_channels and channel_num == 1)
                var = tk.BooleanVar(value=default_value)
                channel_vars['single'][channel_num] = var
                
                chk = ttk.Checkbutton(
                    animal_frame, 
                    text=f"Channel {channel_num}",
                    variable=var
                )
                chk.pack(anchor="w", padx=2, pady=2)
    
    btn_frame = ttk.Frame(dialog)
    btn_frame.pack(fill="x", pady=10)
    
    ttk.Button(btn_frame, text="Select All", command=lambda: toggle_all_channels(True)).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
    ttk.Button(btn_frame, text="Deselect All", command=lambda: toggle_all_channels(False)).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
    ttk.Button(btn_frame, text="Confirm", command=lambda: finalize_channel_selection(dialog)).grid(row=0, column=2, sticky="ew", padx=2, pady=2)

def toggle_all_channels(select):
    for animal_vars in channel_vars.values():
        for var in animal_vars.values():
            var.set(select)

def finalize_channel_selection(dialog):
    global channel_memory
    if multi_animal_data:
        for animal_data in multi_animal_data:
            animal_id = animal_data['animal_id']
            
            if animal_id in channel_vars:
                selected_channels = []
                for channel_num, var in channel_vars[animal_id].items():
                    if var.get():
                        selected_channels.append(channel_num)
                
                if not selected_channels:
                    log_message(f"Please select at least one channel for {animal_id}", "WARNING")
                    return
                    
                animal_data['active_channels'] = selected_channels
                
                channel_memory[animal_id] = selected_channels
                
                if 'fiber_data' not in animal_data or animal_data['fiber_data'] is None:
                    log_message(f"No fiber data available for {animal_id}, skipping alignment", "WARNING")
                    continue

                if animal_data['active_channels'] is None:
                    animal_data['active_channels'] = []

                alignment_success = align_data(animal_data)
                if not alignment_success:
                    log_message(f"Failed to align data for {animal_id}", "WARNING")
                    if 'fiber_data' in animal_data:
                        animal_data['fiber_data_trimmed'] = animal_data['fiber_data']
                
    else:
        selected_channels = []
        for channel_num, var in channel_vars['single'].items():
            if var.get():
                selected_channels.append(channel_num)
        
        if not selected_channels:
            log_message("Please select at least one channel", "WARNING")
            return
        
        global active_channels
        active_channels = selected_channels
        
        channel_memory['single'] = selected_channels
        
        if active_channels is None:
            active_channels = []
        
        alignment_success = align_data()
        if not alignment_success:
            log_message("Alignment failed, but continuing with available data", "WARNING")
            if hasattr(globals(), 'fiber_data'):
                globals()['fiber_data_trimmed'] = globals()['fiber_data']

    save_channel_memory()
    
    dialog.destroy()
    log_message("Selected channels for all animals")
    log_message(f"Processed {len(multi_animal_data)} animals")
    log_message(f"Imported {len(multi_animal_data)} animals")
    
    if multi_animal_data:
        global current_animal_index
        current_animal_index = 0
        main_visualization(multi_animal_data[current_animal_index])
        
        create_fiber_visualization(multi_animal_data[current_animal_index])
        if fiber_plot_window:
            fiber_plot_window.set_plot_type("raw")
            fiber_plot_window.update_plot()

def align_data(animal_data=None):
    """Modified align_data to support different experiment modes"""
    global current_experiment_mode
    
    try:
        # Determine which data to use
        if animal_data:
            fiber_data = animal_data.get('fiber_data')
            ast2_data = animal_data.get('ast2_data')
            channels = animal_data.get('channels', {})
            active_channels = animal_data.get('active_channels', [])
            experiment_mode = animal_data.get('experiment_mode', current_experiment_mode)
            dlc_data = animal_data.get('dlc_data') if experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC else None
        else:
            fiber_data = globals().get('fiber_data')
            ast2_data = globals().get('ast2_data')
            channels = globals().get('channels', {})
            active_channels = globals().get('active_channels', [])
            experiment_mode = current_experiment_mode
            dlc_data = globals().get('dlc_data') if experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC else None
        
        log_message(f"Alignment debug - Experiment mode: {experiment_mode}")
        log_message(f"Alignment debug - Fiber data: {fiber_data is not None}")
        log_message(f"Alignment debug - Channels: {channels}")
        log_message(f"Alignment debug - Active channels: {active_channels}")
        log_message(f"Alignment debug - DLC data: {dlc_data is not None}")
        log_message(f"Alignment debug - AST2 data: {ast2_data is not None}")

        # Check if we have the necessary data
        if fiber_data is None:
            log_message("Fiber data is None, cannot align", "ERROR")
            return False
            
        if not active_channels:
            log_message("No active channels selected, cannot align", "ERROR")
            return False
            
        if ast2_data is None:
            log_message("No running data available, cannot align", "ERROR")
            return False
        
        # DLC data is optional based on experiment mode
        if experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC and dlc_data is None:
            log_message("DLC mode selected but no DLC data available, cannot align", "ERROR")
            return False
        
        # Get events column from fiber data
        events_col = channels.get('events')
        if events_col is None or events_col not in fiber_data.columns:
            log_message("Events column not found in fiber data", "ERROR")
            return False
        
        time_col = channels['time']
        
        # Find Input2 events (running markers) - running start time
        input2_events = fiber_data[fiber_data[events_col].str.contains('Input2', na=False)]
        if len(input2_events) < 1:
            log_message("Could not find Input2 events for running start", "ERROR")
            return False
        
        # Get running start time (first Input2 event)
        running_start_time = input2_events[time_col].iloc[0]
        
        # Get fiber start time (first timestamp in fiber data)
        fiber_start_time = fiber_data[time_col].iloc[0]
        
        # Calculate running end time based on AST2 data
        running_timestamps = ast2_data['data']['timestamps']
        running_duration = running_timestamps[-1] - running_timestamps[0] if len(running_timestamps) > 1 else 0
        running_end_time = running_start_time + running_duration
        
        log_message(f"Running start time: {running_start_time:.2f}s")
        log_message(f"Running end time: {running_end_time:.2f}s")
        log_message(f"Running duration: {running_duration:.2f}s")
        log_message(f"Fiber start time: {fiber_start_time:.2f}s")
        
        # Adjust fiber data relative to running start time
        fiber_data_adjusted = fiber_data.copy()
        fiber_data_adjusted[time_col] = fiber_data_adjusted[time_col] - running_start_time
        
        # Trim fiber data to running duration
        fiber_data_trimmed = fiber_data_adjusted[
            (fiber_data_adjusted[time_col] >= 0) & 
            (fiber_data_adjusted[time_col] <= running_duration)].copy()
        
        if fiber_data_trimmed.empty:
            log_message("Trimmed fiber data is empty after alignment", "ERROR")
            return False
        
        # Initialize video-related variables
        video_start_time = None
        video_end_time = None
        video_total_frames = 0
        video_total_frames_trimmed = 0
        video_fps = 30  # Default
        video_start_offset = 0
        dlc_data_trimmed = None
        valid_video_frames = []
        
        # Process DLC data only in FIBER_AST2_DLC mode
        if experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC and dlc_data is not None:
            # Find Input4 events (video markers) - video start time
            input4_events = fiber_data[fiber_data[events_col].str.contains('Input4', na=False)]
            if len(input4_events) < 1:
                log_message("Could not find Input4 events for video start", "WARNING")
                log_message("Continuing without video alignment", "INFO")
            else:
                video_start_time = input4_events[time_col].iloc[0]
                
                # Calculate video parameters
                unique_bodyparts = list(dlc_data.keys())
                if unique_bodyparts:
                    video_total_frames = len(dlc_data[unique_bodyparts[0]]['x'])
                    video_duration = video_total_frames / video_fps
                    video_end_time = video_start_time + video_duration
                    
                    log_message(f"Video start time: {video_start_time:.2f}s")
                    log_message(f"Video total frames: {video_total_frames}")
                    log_message(f"Video duration: {video_duration:.2f}s")
                    log_message(f"Video end time: {video_end_time:.2f}s")
                    
                    # Trim video data to running duration
                    video_start_offset = video_start_time - running_start_time
                    video_end_offset = video_end_time - running_start_time
                    
                    # Only include video frames that are within the running period
                    valid_video_frames = []
                    for frame_idx in range(video_total_frames):
                        frame_time = video_start_offset + (frame_idx / video_fps)
                        if 0 <= frame_time <= running_duration:
                            valid_video_frames.append(frame_idx)
                    
                    # Create trimmed DLC data with only valid frames
                    dlc_data_trimmed = {}
                    for bodypart, data in dlc_data.items():
                        dlc_data_trimmed[bodypart] = {
                            'x': data['x'][valid_video_frames],
                            'y': data['y'][valid_video_frames],
                            'likelihood': data['likelihood'][valid_video_frames]
                        }
                    
                    video_total_frames_trimmed = len(valid_video_frames)
                    
                    log_message(f"Trimmed video frames: {video_total_frames_trimmed}/{video_total_frames}")
                    log_message(f"Video start offset: {video_start_offset:.2f}s")
                    log_message(f"Video end offset: {video_end_offset:.2f}s")
        
        # Adjust AST2 data relative to running start
        if ast2_data is not None:
            ast2_timestamps = ast2_data['data']['timestamps']
            ast2_timestamps_adjusted = ast2_timestamps - ast2_timestamps[0]  # AST2 timestamps are relative to running start
            
            # Trim AST2 data (should already be within running duration)
            valid_indices = (ast2_timestamps_adjusted >= 0) & (ast2_timestamps_adjusted <= running_duration)
            ast2_timestamps_trimmed = ast2_timestamps_adjusted[valid_indices]
            ast2_speed_trimmed = ast2_data['data']['speed'][valid_indices]
            
            # Create adjusted AST2 data
            ast2_data_adjusted = {
                'header': ast2_data['header'],
                'data': {
                    'timestamps': ast2_timestamps_trimmed,
                    'speed': ast2_speed_trimmed
                }
            }
            
            # Calculate sampling rate
            if len(ast2_timestamps_trimmed) > 1:
                ast2_sampling_rate = 1 / np.mean(np.diff(ast2_timestamps_trimmed))
            else:
                ast2_sampling_rate = 10  # Default assumption
        else:
            ast2_data_adjusted = None
            ast2_sampling_rate = None
        
        # Update the appropriate data structure
        if animal_data:
            animal_data.update({
                'fiber_data_adjusted': fiber_data_adjusted,
                'fiber_data_trimmed': fiber_data_trimmed,
                'ast2_data_adjusted': ast2_data_adjusted,
                'running_start_time': running_start_time,
                'running_end_time': running_end_time,
                'running_duration': running_duration,
                'fiber_sampling_rate': 10,  # Assuming fiber sampling rate is 10 Hz
                'ast2_sampling_rate': ast2_sampling_rate
            })
            
            # Add DLC-related data only in FIBER_AST2_DLC mode
            if experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC and dlc_data_trimmed is not None:
                animal_data.update({
                    'dlc_data_trimmed': dlc_data_trimmed,
                    'video_start_time': video_start_time,
                    'video_end_time': video_end_time,
                    'video_total_frames': video_total_frames_trimmed,
                    'video_fps': video_fps,
                    'video_start_offset': video_start_offset,
                    'valid_video_frames': valid_video_frames
                })
                # Replace original dlc_data with trimmed version for visualization
                animal_data['dlc_data'] = dlc_data_trimmed
        else:
            globals()['fiber_data_adjusted'] = fiber_data_adjusted
            globals()['fiber_data_trimmed'] = fiber_data_trimmed
            globals()['ast2_data_adjusted'] = ast2_data_adjusted
            globals()['running_start_time'] = running_start_time
            globals()['running_end_time'] = running_end_time
            globals()['running_duration'] = running_duration
            
            # Add DLC-related data only in FIBER_AST2_DLC mode
            if experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC and dlc_data_trimmed is not None:
                globals()['dlc_data'] = dlc_data_trimmed
                globals()['parsed_data'] = dlc_data_trimmed
                globals()['video_start_time'] = video_start_time
                globals()['video_end_time'] = video_end_time
                globals()['video_total_frames'] = video_total_frames_trimmed
                globals()['video_fps'] = video_fps
                globals()['video_start_offset'] = video_start_offset
        
        # Display alignment information
        info_message = f"Data aligned successfully (running data as reference)!\n"
        info_message += f"Experiment Mode: {experiment_mode}\n"
        info_message += f"Running start time: {running_start_time:.2f}s\n"
        info_message += f"Running end time: {running_end_time:.2f}s\n"
        info_message += f"Running duration: {running_duration:.2f}s\n"
        
        if experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC and video_start_time is not None:
            info_message += f"Video start time: {video_start_time:.2f}s\n"
            info_message += f"Video offset: {video_start_offset:.2f}s\n"
            info_message += f"Video frames (original/trimmed): {video_total_frames}/{video_total_frames_trimmed}\n"
            info_message += f"Video FPS: {video_fps}\n"
        
        info_message += f"Fiber sampling rate: 10 Hz\n"
        
        if ast2_sampling_rate is not None:
            info_message += f"Running wheel sampling rate: {ast2_sampling_rate:.2f} Hz"
        
        log_message(info_message, "INFO")
        log_message("Data aligned successfully using running data as reference")
        return True
        
    except Exception as e:
        log_message(f"Failed to align data: {str(e)}", "ERROR")
        log_message(f"Traceback: {traceback.format_exc()}", "ERROR")
        return False

def read_dlc_file(file_path):
    """Read dlc CSV file and parse bodyparts data"""
    if file_path:
        log_message(f"Selected: {file_path}")
        try:
            # Read CSV file, don't use first row as header
            df = pd.read_csv(file_path, header=None, low_memory=False)
            
            # Check if file has enough rows
            if len(df) < 4:
                log_message("CSV file doesn't have enough rows, at least 4 rows needed", "ERROR")
                return
            
            # Get bodyparts information from second row (index 1)
            if len(df) > 1:
                bodyparts_row = df.iloc[1].values
            else:
                log_message("Cannot find bodyparts information in second row", "ERROR")
                return
            
            # Find all unique bodyparts
            global unique_bodyparts
            unique_bodyparts = []
            for i in range(1, len(bodyparts_row), 3):  # Start from index 1, skip "bodyparts" title
                if i < len(bodyparts_row):
                    part = bodyparts_row[i]
                    if pd.notna(part) and str(part).strip() != '':
                        bodypart_name = str(part).strip()
                        if bodypart_name not in unique_bodyparts:
                            unique_bodyparts.append(bodypart_name)
            
            if not unique_bodyparts:
                log_message("No valid bodyparts information found", "ERROR")
                return
                
            log_message(f"Detected bodyparts: {unique_bodyparts}")
            log_message(f"CSV file total columns: {df.shape[1]}")
            
            # Create dictionary to store x, y, likelihood data for each bodypart
            bodypart_data = {}
            
            # Extract data starting from fourth row (index 3)
            data_start_row = 3
            if len(df) <= data_start_row:
                log_message("Not enough data rows, cannot extract data from fourth row", "ERROR")
                return
                
            data_rows = df.iloc[data_start_row:]
            
            # Check if there are enough columns
            expected_cols = len(unique_bodyparts) * 3
            if df.shape[1] < expected_cols:
                log_message(f"Not enough columns, expected {expected_cols}, got {df.shape[1]}", "ERROR")
                return
            
            # Create data vectors for each bodypart
            col_index = 1  # Start from second column, skip "bodyparts" title
            for bodypart in unique_bodyparts:
                try:
                    # Each bodypart occupies 3 columns: x, y, likelihood
                    if col_index + 2 < df.shape[1]:
                        x_data = data_rows.iloc[:, col_index].dropna().astype(float).values
                        y_data = data_rows.iloc[:, col_index + 1].dropna().astype(float).values
                        likelihood_data = data_rows.iloc[:, col_index + 2].dropna().astype(float).values
                        
                        bodypart_data[bodypart] = {
                            'x': x_data,
                            'y': y_data,
                            'likelihood': likelihood_data
                        }
                    else:
                        log_message("Bodypart '{bodypart}' column index out of range", "WARNING")
                        break
                except Exception as col_error:
                    log_message(f"Error processing bodypart '{bodypart}': {col_error}", "ERROR")
                    continue
                
                col_index += 3
            
            # Display parsing results
            result_info = f"File parsed successfully!\n"
            result_info += f"Found {len(unique_bodyparts)} bodyparts: {', '.join(unique_bodyparts)}\n"
            result_info += f"Data rows: {len(data_rows)}\n"
            
            for bodypart, data in bodypart_data.items():
                result_info += f"{bodypart}: x({len(data['x'])}), y({len(data['y'])}), likelihood({len(data['likelihood'])})"
            log_message(result_info, "INFO")
            
            # Print first few rows for verification
            log_message(f"Bodyparts found: {unique_bodyparts}")
            for bodypart, data in bodypart_data.items():
                log_message(f"\n{bodypart}:")
                log_message(f"  X (first 5): {data['x'][:5]}")
                log_message(f"  Y (first 5): {data['y'][:5]}")
                log_message(f"  Likelihood (first 5): {data['likelihood'][:5]}")
            
            # Store data as global variable for later use
            global parsed_data
            parsed_data = bodypart_data

            return parsed_data
        
        except Exception as e:
            log_message(f"Failed to read file: {e}", "ERROR")
            return None

def convert_num(s):
    s = s.strip()
    try:
        if '.' in s or 'e' in s or 'E' in s:
            return float(s)
        else:
            return int(s)
    except ValueError:
        return s

def h_AST2_readData(filename):
    header = {}
    
    with open(filename, 'rb') as fid:
        header_lines = []
        while True:
            line = fid.readline().decode('utf-8').strip()
            if line == 'header_end':
                break
            header_lines.append(line)
        
        for line in header_lines:
            match = re.match(r"header\.(\w+)\s*=\s*(.*);$", line)
            if not match:
                continue
            key = match.group(1)
            value_str = match.group(2).strip()
            
            if value_str.startswith("'") and value_str.endswith("'"):
                header[key] = value_str[1:-1]
            elif value_str.startswith('[') and value_str.endswith(']'):
                inner = value_str[1:-1].strip()
                if not inner:
                    header[key] = []
                else:
                    if ';' in inner:
                        rows = inner.split(';')
                        array = []
                        for row in rows:
                            row = row.strip()
                            if row:
                                elements = row.split()
                                array.append([convert_num(x) for x in elements])
                        header[key] = array
                    else:
                        elements = inner.split()
                        header[key] = [convert_num(x) for x in elements]
            else:
                header[key] = convert_num(value_str)
        
        binary_data = np.fromfile(fid, dtype=np.int16)
    
    if 'activeChIDs' in header and 'scale' in header:
        numOfCh = len(header['activeChIDs'])
        data = binary_data.reshape((numOfCh, -1), order='F') / header['scale']
    else:
        data = None
    
    # log_message(f"header:{header}")
    # log_message(f"data:{data}")
    return header, data

def h_AST2_raw2Speed(rawData, info, voltageRange=None):
    if voltageRange is None or len(voltageRange) == 0:
        voltageRange = h_calibrateVoltageRange(rawData)
    
    speedDownSampleFactor = 50
    
    rawDataLength = len(rawData)
    segmentLength = speedDownSampleFactor
    speedDataLength = rawDataLength // segmentLength
    
    if rawDataLength % segmentLength != 0:
        log_message(f"SpeedDataLength is not integer!  speedDataLength = {rawDataLength}, speedDownSampleFactor = {segmentLength}", "ERROR")
        rawData = rawData[:speedDataLength * segmentLength]
    
    t = ((np.arange(speedDataLength) + 0.5) * speedDownSampleFactor) / info['inputRate']
    time_segment = (np.arange(1, segmentLength + 1)) / info['inputRate']
    reshapedData = rawData.reshape(segmentLength, speedDataLength, order='F')
    speedData2 = h_computeSpeed2(time_segment, reshapedData, voltageRange)
    
    if invert_running:
        speedData2 = -speedData2
    
    speedData = {
        'timestamps': t,
        'speed': speedData2
    }
    
    return speedData

def h_calibrateVoltageRange(rawData):
    peakValue, peakPos = h_AST2_findPeaks(rawData)
    valleyValue, valleyPos = h_AST2_findPeaks(-rawData)
    valleyValue = [-x for x in valleyValue]
    
    if len(peakValue) > 0 and len(valleyValue) > 0:
        voltageRange = [np.mean(valleyValue), np.mean(peakValue)]
        if np.diff(voltageRange) > 3:
            log_message(f"Calibrated voltage range is {voltageRange}")
        else:
            log_message("Calibration error. Range too small")
            voltageRange = [0, 5]
    else:
        voltageRange = [0, 5]
        log_message("Calibration fail! Return default: [0 5].")
    
    return voltageRange

def h_AST2_findPeaks(data):
    transitionPos = np.where(np.abs(np.diff(data)) > 2)[0]
    
    transitionPos = transitionPos[(transitionPos > 50) & (transitionPos < len(data) - 50)]
    
    if len(transitionPos) >= 1:
        peakValue = np.zeros(len(transitionPos))
        peakPos = np.zeros(len(transitionPos))
        
        for i, pos in enumerate(transitionPos):
            segment = data[pos-50:pos+51]
            peakValue[i] = np.max(segment)
            peakPos[i] = pos - 50 + np.argmax(segment)
    else:
        return [], []
    
    avg = np.mean(data)
    maxData = np.max(data)
    thresh = avg + 0.8 * (maxData - avg)
    
    mask = peakValue > thresh
    peakValue = peakValue[mask]
    peakPos = peakPos[mask]
    
    return peakValue, peakPos

def h_computeSpeed2(time, data, voltageRange):
    deltaVoltage = voltageRange[1] - voltageRange[0]
    thresh = 3/5 * deltaVoltage
    
    diffData = np.diff(data, axis=0)
    I = np.abs(diffData) > thresh
    
    data = data.copy()
    for j in range(data.shape[1]):
        if np.any(I[:, j]):
            ind = np.where(I[:, j])[0]
            for i in ind:
                if diffData[i, j] < thresh:
                    data[i+1:, j] = data[i+1:, j] + deltaVoltage
                elif diffData[i, j] > thresh:
                    data[i+1:, j] = data[i+1:, j] - deltaVoltage
    
    dataInDegree = (data / deltaVoltage) * 360
    
    deltaDegree = np.mean(dataInDegree[-11:, :], axis=0) - np.mean(dataInDegree[:11, :], axis=0)
    
    I1 = deltaDegree > 200
    I2 = deltaDegree < -200
    deltaDegree[I1] = deltaDegree[I1] - 360
    deltaDegree[I2] = deltaDegree[I2] + 360
    
    duration = np.mean(time[-11:]) - np.mean(time[:11])
    speed = deltaDegree / duration
    
    diameter = threadmill_diameter
    speed2 = speed / 360 * diameter * np.pi
    
    return speed2

def create_visualization_window():
    """Create the bodyparts location visualization window"""
    global visualization_window, parsed_data, central_label
    
    if 'parsed_data' not in globals() or not parsed_data:
        log_message("Please load the behavior data file first", "WARNING")
        return
    
    # If the window already exists, close it first
    if visualization_window:
        visualization_window.close_window()
    
    # Hide the default label in the central display area
    if 'central_label' in globals():
        try:
            # Check if central_label still exists
            if hasattr(central_label, 'winfo_exists') and central_label.winfo_exists():
                central_label.pack_forget()
        except tk.TclError:
            # If central_label has been destroyed, create a new one
            central_label = tk.Label(central_display_frame, text="Central Display Area\nThe bodyparts location visualization window will be displayed after loading the CSV file", bg="#f8f8f8", fg="#666666")
    
    # Create a new visualization window
    visualization_window = BodypartVisualizationWindow(central_display_frame, parsed_data)

def create_trajectory_pointcloud():
    """Create the trajectory point cloud visualization window"""
    global parsed_data, selected_bodyparts, central_label, show_data_points_var
    
    if 'parsed_data' not in globals() or not parsed_data:
        log_message("Please load the behavior data file first", "WARNING")
        return
    
    if not selected_bodyparts:
        log_message("Please select the bodyparts to display the trajectory first", "WARNING")
        return
    
    # Hide the default label in the central display area
    if 'central_label' in globals():
        central_label.pack_forget()
    
    # Clear the central display area
    for widget in central_display_frame.winfo_children():
        widget.destroy()
    
    # Create a matplotlib figure
    fig = Figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111)
    
    # Set figure properties
    ax.set_title("🌟 Trajectory Point Cloud Visualization", fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
    ax.set_xlabel("X Coordinate", fontsize=12, fontweight='bold', color='#2c3e50')
    ax.set_ylabel("Y Coordinate", fontsize=12, fontweight='bold', color='#2c3e50')
    ax.grid(True, alpha=0.3, linestyle='--', color='#bdc3c7')
    ax.set_facecolor('#ffffff')
    
    # Use the same color configuration as the buttons
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
             '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#95a5a6']
    
    # Get the list of all bodyparts (in the same order as when the buttons were created)
    all_bodyparts = list(parsed_data.keys())
    
    # Plot the trajectory point cloud for the selected bodyparts
    for bodypart in selected_bodyparts:
        if bodypart in parsed_data:
            data = parsed_data[bodypart]
            x_data = data['x']
            y_data = data['y']
            
            # Downsample by a factor of 2
            step = 2
            x_sampled = x_data[::step]
            y_sampled = y_data[::step]
            
            # Get the corresponding color based on the bodypart's index in the original list (consistent with button colors)
            bodypart_index = all_bodyparts.index(bodypart)
            color = colors[bodypart_index % len(colors)]
            
            # Plot the trajectory lines
            ax.plot(x_sampled, y_sampled, color='lightgray', linewidth=0.5, alpha=0.3)

            # Plot the point cloud based on the checkbox state
            if show_data_points_var and show_data_points_var.get():
                # Plot the point cloud with 70% opacity
                ax.scatter(x_sampled, y_sampled, s=10, alpha=0.7, 
                          color=color, edgecolors='white', linewidth=0.5, 
                          label=f'{bodypart} ({len(x_sampled)} points)')
            else:
                # Only show the trajectory line, not the points, but still need to add a legend entry
                ax.plot([], [], color=color, linewidth=2, label=f'{bodypart} trajectory')
    
    # Set axis limits
    if selected_bodyparts:
        all_x = []
        all_y = []
        for bodypart in selected_bodyparts:
            if bodypart in parsed_data:
                all_x.extend(parsed_data[bodypart]['x'])
                all_y.extend(parsed_data[bodypart]['y'])
        
        if all_x and all_y:
            margin_x = (max(all_x) - min(all_x)) * 0.1
            margin_y = (max(all_y) - min(all_y)) * 0.1
            ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
            ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
    
    # Add legend
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Set axis style
    ax.tick_params(colors='#2c3e50', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(1)
    
    # Create canvas and add to central display area
    canvas = FigureCanvasTkAgg(fig, central_display_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Create control panel
    control_frame = tk.Frame(central_display_frame, bg='#f8f8f8', height=50)
    control_frame.pack(fill=tk.X, side=tk.BOTTOM)
    control_frame.pack_propagate(False)
    
    # Add close button
    close_btn = tk.Button(control_frame, text="❌ Close Point Cloud", 
                         command=lambda: close_pointcloud_window(),
                         bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
                         relief=tk.FLAT, padx=15, pady=5)
    close_btn.pack(side=tk.RIGHT, padx=10, pady=10)

    # Add show/hide data points checkbox
    if show_data_points_var is None:
        show_data_points_var = tk.BooleanVar(value=True)

    show_points_check = tk.Checkbutton(control_frame, text="Show Data Points", 
                                       variable=show_data_points_var, 
                                       command=create_trajectory_pointcloud,
                                       bg='#f8f8f8', font=('Arial', 10))
    show_points_check.pack(side=tk.RIGHT, padx=10)
    
    # Add information label
    info_label = tk.Label(control_frame, 
                         text=f"Displaying trajectory point cloud for {len(selected_bodyparts)} bodyparts",
                         bg='#f8f8f8', fg='#2c3e50', font=('Arial', 10))
    info_label.pack(side=tk.LEFT, padx=10, pady=10)
    
    log_message(f"Trajectory point cloud display completed - {len(selected_bodyparts)} bodyparts displayed")

def close_pointcloud_window():
    """Close the trajectory point cloud window and restore the visualization window"""
    global visualization_window, parsed_data
    
    # Clear the central display area
    for widget in central_display_frame.winfo_children():
        widget.destroy()
    
    # If data is available, restore the visualization window
    if 'parsed_data' in globals() and parsed_data:
        visualization_window = BodypartVisualizationWindow(central_display_frame, parsed_data)
        log_message("Bodyparts location visualization restored")
    else:
        # If no data, show the default label
        central_label = tk.Label(central_display_frame, text="Central Display Area\nThe bodyparts location visualization window will be displayed after loading the CSV file", bg="#f8f8f8", fg="#666666")
        central_label.pack(pady=20)
        log_message("Trajectory point cloud closed")

def toggle_bodypart(bodypart_name, button):
    """Toggle the state of the bodypart button"""
    global skeleton_building, skeleton_sequence
    
    if skeleton_building:
        # Special handling in skeleton building mode
        if bodypart_name in skeleton_sequence:
            # If already in the sequence, remove it and all subsequent connections
            index = skeleton_sequence.index(bodypart_name)
            skeleton_sequence = skeleton_sequence[:index]
            # Update button states
            for bp in bodypart_buttons:
                if bp in skeleton_sequence:
                    bodypart_buttons[bp].config(relief=tk.SUNKEN, bg="#e67e22")
                else:
                    bodypart_buttons[bp].config(relief=tk.RAISED)
        else:
            # Add to the skeleton sequence
            skeleton_sequence.append(bodypart_name)
            button.config(relief=tk.SUNKEN, bg="#e67e22")  # Orange indicates skeleton building
        
        log_message(f"Skeleton building sequence: {skeleton_sequence}"f"Skeleton building sequence: {skeleton_sequence}")
        
        # Enable confirm button if there are at least 2 points
        if len(skeleton_sequence) >= 2:
            confirm_skeleton_button.config(state=tk.NORMAL)
        else:
            confirm_skeleton_button.config(state=tk.DISABLED)
    else:
        # Normal selection mode
        if bodypart_name in selected_bodyparts:
            selected_bodyparts.remove(bodypart_name)
            button.config(relief=tk.RAISED)
        else:
            selected_bodyparts.add(bodypart_name)
            button.config(relief=tk.SUNKEN)
        log_message(f"Currently selected bodyparts: {list(selected_bodyparts)}")

def start_skeleton_building():
    """Start skeleton building mode"""
    global skeleton_building, skeleton_sequence
    
    skeleton_building = True
    skeleton_sequence = []
    
    # Reset all button states
    for bodypart, button in bodypart_buttons.items():
        button.config(relief=tk.RAISED)
    
    # Update button states
    add_skeleton_button.config(state=tk.DISABLED, text="Building...")
    confirm_skeleton_button.config(state=tk.DISABLED)
    
    # Clear current selections
    selected_bodyparts.clear()
    
    log_message("Skeleton building mode: Click bodyparts to create connections\n")
    log_message("Skeleton building mode started")

def confirm_skeleton():
    """Confirm skeleton building"""
    global skeleton_building, skeleton_sequence, skeleton_connections
    
    if len(skeleton_sequence) < 2:
        log_message("At least 2 bodyparts are required to build a skeleton", "WARNING")
        return
    
    # Create connections
    new_connections = []
    for i in range(len(skeleton_sequence) - 1):
        connection = (skeleton_sequence[i], skeleton_sequence[i + 1])
        new_connections.append(connection)
    
    skeleton_connections.extend(new_connections)
    
    # Exit skeleton building mode
    skeleton_building = False
    skeleton_sequence = []
    
    # Restore button states
    add_skeleton_button.config(state=tk.NORMAL, text="Add Skeleton")
    confirm_skeleton_button.config(state=tk.DISABLED)
    
    # Reset all button colors
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
             '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#95a5a6']
    
    for i, (bodypart, button) in enumerate(bodypart_buttons.items()):
        color = colors[i % len(colors)]
        button.config(bg=color, relief=tk.RAISED)
    
    # Update visualization window to display skeleton
    if visualization_window:
        visualization_window.update_plot_optimized()

    log_message(f"Skeleton connections: {skeleton_connections}")
    log_message(f"Skeleton building completed!\n{len(new_connections)} connections added")

def apply_fps_conversion():
    """Apply FPS conversion settings"""
    global fps_conversion_enabled, current_fps
    
    try:
        # Get FPS value
        fps_value = float(fps_var.get())
        if fps_value <= 0:
            log_message("FPS value must be greater than 0", "ERROR")
            return
        
        current_fps = fps_value
        fps_conversion_enabled = fps_conversion_var.get()
        
        # Update visualization window
        if visualization_window:
            visualization_window.update_plot_optimized()
        
        if fps_conversion_enabled:
            time_unit = time_unit_var.get()
            log_message(f"FPS conversion enabled: FPS={current_fps}, Time unit={time_unit}")
        else:
            log_message("FPS conversion disabled, displaying frame numbers")
            
    except ValueError:
        log_message("Please enter a valid FPS value", "ERROR")
    except Exception as e:
        log_message(f"Error applying FPS conversion: {str(e)}", "ERROR")

def frame_to_time(frame_index):
    """Convert frame index to time"""
    global fps_conversion_enabled, current_fps
    
    if not fps_conversion_enabled:
        return frame_index + 1  # Return frame number (starting from 1)
    
    # Calculate time in seconds
    time_seconds = frame_index / current_fps
    
    # Convert based on time unit
    time_unit = time_unit_var.get() if time_unit_var else "seconds"
    if time_unit == "minutes":
        return time_seconds / 60
    else:
        return time_seconds

def get_time_label():
    """Get time axis label"""
    global fps_conversion_enabled
    
    if not fps_conversion_enabled:
        return "Frame"
    
    time_unit = time_unit_var.get() if time_unit_var else "seconds"
    return f"Time({time_unit})"

def create_bodypart_buttons(bodyparts):
    """Create bodypart toggle buttons"""
    # Clear existing buttons
    for widget in left_frame.winfo_children():
        widget.destroy()
    
    bodypart_buttons.clear()
    selected_bodyparts.clear()
    
    # Define color configuration (consistent with visualization window)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
             '#1abc9c', '#e67e22', '#34495e', '#f1c40f', '#95a5a6']
    
    # Add title
    title_label = tk.Label(left_frame, text="🎯 Bodyparts:", font=("Microsoft YaHei", 12, "bold"), 
                          bg="#e0e0e0", fg="#2c3e50")
    title_label.pack(pady=(10, 5))
    
    # Create toggle button for each bodypart
    for i, bodypart in enumerate(bodyparts):
        color = colors[i % len(colors)]
        button_text = f"{i+1}. {bodypart}"  # Add numbering
        button = tk.Button(
            left_frame,
            text=button_text,
            width=15,
            relief=tk.RAISED,
            bg=color,
            fg="white",
            font=("Microsoft YaHei", 9, "bold"),
            activebackground=color,
            activeforeground="white",
            cursor="hand2",
            command=lambda bp=bodypart: toggle_bodypart(bp, bodypart_buttons[bp])
        )
        button.pack(pady=3, padx=8, fill=tk.X)
        bodypart_buttons[bodypart] = button
    
    # Add separator
    separator = tk.Frame(left_frame, height=2, bg="#bdc3c7")
    separator.pack(fill=tk.X, padx=10, pady=10)
    
    # Add skeleton function title
    skeleton_title = tk.Label(left_frame, text="🦴 Skeleton Building:", font=("Microsoft YaHei", 12, "bold"), 
                             bg="#e0e0e0", fg="#2c3e50")
    skeleton_title.pack(pady=(5, 5))
    
    # Add skeleton buttons
    add_skeleton_btn = tk.Button(
        left_frame,
        text="Add Skeleton",
        width=15,
        relief=tk.RAISED,
        bg="#3498db",
        fg="white",
        font=("Microsoft YaHei", 9, "bold"),
        activebackground="#2980b9",
        activeforeground="white",
        cursor="hand2",
        command=start_skeleton_building
    )
    add_skeleton_btn.pack(pady=3, padx=8, fill=tk.X)
    
    confirm_skeleton_btn = tk.Button(
        left_frame,
        text="Confirm",
        width=15,
        relief=tk.RAISED,
        bg="#27ae60",
        fg="white",
        font=("Microsoft YaHei", 9, "bold"),
        activebackground="#229954",
        activeforeground="white",
        cursor="hand2",
        command=confirm_skeleton,
        state=tk.DISABLED
    )
    confirm_skeleton_btn.pack(pady=3, padx=8, fill=tk.X)
    
    # Store buttons as global variables for later access
    global add_skeleton_button, confirm_skeleton_button
    add_skeleton_button = add_skeleton_btn
    confirm_skeleton_button = confirm_skeleton_btn
    
    # Add separator
    separator2 = tk.Frame(left_frame, height=2, bg="#bdc3c7")
    separator2.pack(fill=tk.X, padx=10, pady=10)
    
    # Add FPS conversion function title
    fps_title = tk.Label(left_frame, text="⏱️ FPS Conversion:", font=("Microsoft YaHei", 12, "bold"), 
                        bg="#e0e0e0", fg="#2c3e50")
    fps_title.pack(pady=(5, 5))
    
    # FPS setting frame
    fps_frame = tk.Frame(left_frame, bg="#e0e0e0")
    fps_frame.pack(pady=3, padx=8, fill=tk.X)
    
    fps_label = tk.Label(fps_frame, text="FPS:", font=("Microsoft YaHei", 9), 
                        bg="#e0e0e0", fg="#2c3e50")
    fps_label.pack(side=tk.LEFT)
    
    global fps_var
    fps_var = tk.StringVar(value="30")
    fps_entry = tk.Entry(fps_frame, textvariable=fps_var, width=8, 
                        font=("Microsoft YaHei", 9))
    fps_entry.pack(side=tk.RIGHT)
    
    time_unit_frame = tk.Frame(left_frame, bg="#e0e0e0")
    time_unit_frame.pack(pady=3, padx=8, fill=tk.X)
    
    time_unit_label = tk.Label(time_unit_frame, text="Time Unit:", font=("Microsoft YaHei", 9), 
                              bg="#e0e0e0", fg="#2c3e50")
    time_unit_label.pack(side=tk.LEFT)
    
    global time_unit_var
    time_unit_var = tk.StringVar(value="seconds")
    time_unit_combo = ttk.Combobox(time_unit_frame, textvariable=time_unit_var, 
                                  values=["seconds", "minutes"], width=6, state="readonly")
    time_unit_combo.pack(side=tk.RIGHT)
    
    # Enable FPS conversion checkbox
    global fps_conversion_var
    fps_conversion_var = tk.BooleanVar()
    fps_checkbox = tk.Checkbutton(left_frame, text="Enable FPS Conversion", 
                                 variable=fps_conversion_var,
                                 font=("Microsoft YaHei", 9),
                                 bg="#e0e0e0", fg="#2c3e50",
                                 activebackground="#e0e0e0",
                                 command=apply_fps_conversion)
    fps_checkbox.pack(pady=3, padx=8)
    
    # Apply button
    apply_fps_btn = tk.Button(
        left_frame,
        text="Apply Settings",
        width=15,
        relief=tk.RAISED,
        bg="#e67e22",
        fg="white",
        font=("Microsoft YaHei", 9, "bold"),
        activebackground="#d35400",
        activeforeground="white",
        cursor="hand2",
        command=apply_fps_conversion
    )
    apply_fps_btn.pack(pady=3, padx=8, fill=tk.X)

def main_visualization(animal_data=None):
    """Modified to handle different experiment modes"""
    global parsed_data, visualization_window, fiber_plot_window, running_plot_window
    global current_experiment_mode
    
    for widget in central_display_frame.winfo_children():
        widget.destroy()
    
    if animal_data is None:
        # Using global data
        if current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2_DLC:
            if not hasattr(globals(), 'parsed_data') or not parsed_data:
                log_message("No DLC data available for visualization", "WARNING")
                return
            
            create_bodypart_buttons(list(parsed_data.keys()))
            create_visualization_window()
        
        create_fiber_visualization()
        create_running_visualization()
    else:
        # Using animal_data
        animal_mode = animal_data.get('experiment_mode', EXPERIMENT_MODE_FIBER_AST2_DLC)
        
        if animal_mode == EXPERIMENT_MODE_FIBER_AST2_DLC:
            if 'dlc_data' not in animal_data or not animal_data['dlc_data']:
                log_message("No DLC data available for visualization", "WARNING")
            else:
                parsed_data = animal_data['dlc_data']
                create_bodypart_buttons(list(parsed_data.keys()))
                create_visualization_window()
        else:
            # Fiber+AST2 mode: show info message
            info_label = tk.Label(central_display_frame, 
                                 text="Fiber + AST2 Mode\n\nBodypart visualization not available\nSee fiber and running plots on the right",
                                 bg="#f8f8f8", fg="#666666",
                                 font=("Arial", 12))
            info_label.pack(pady=100)
        
        create_fiber_visualization(animal_data)
        create_running_visualization(animal_data)

def create_fiber_visualization(animal_data=None):
    global fiber_plot_window
    
    if fiber_plot_window:
        fiber_plot_window.close_window()
    
    target_signal = target_signal_var.get() if 'target_signal_var' in globals() else "470"
    
    if animal_data:
        if 'preprocessed_data' in animal_data:
            fiber_plot_window = FiberVisualizationWindow(central_display_frame, animal_data, target_signal)
        else:
            if 'fiber_data_trimmed' not in animal_data or animal_data['fiber_data_trimmed'] is None:
                if 'fiber_data' in animal_data and animal_data['fiber_data'] is not None:
                    log_message("Using fiber_data instead of fiber_data_trimmed", "INFO")
                    animal_data['fiber_data_trimmed'] = animal_data['fiber_data']
                else:
                    log_message("No fiber data available in animal_data", "WARNING")
                    return

            if 'channels' not in animal_data or not animal_data['channels']:
                log_message("No channels configuration in animal_data", "WARNING")
                return
                
            if 'active_channels' not in animal_data or not animal_data['active_channels']:
                log_message("No active channels in animal_data", "WARNING")
                return
                
            fiber_plot_window = FiberVisualizationWindow(central_display_frame, animal_data, target_signal)

def create_running_visualization(animal_data=None):
    global running_plot_window
    
    if running_plot_window:
        running_plot_window.close_window()
    
    running_plot_window = RunningVisualizationWindow(central_display_frame, animal_data)

def on_animal_select(event):
    global current_animal_index
    selection = file_listbox.curselection()
    if selection:
        current_animal_index = selection[0]
        if current_animal_index < len(multi_animal_data):
            main_visualization(multi_animal_data[current_animal_index])

def create_animal_list():
    for widget in right_frame.winfo_children():
        widget.destroy()

    multi_animal_frame = ttk.LabelFrame(right_frame, text="Multi Animal Analysis")
    multi_animal_frame.pack(fill=tk.X, padx=5, pady=5)
    
    list_frame = ttk.Frame(multi_animal_frame)
    list_frame.pack(fill=tk.X, expand=True, padx=5, pady=5)
    
    xscroll = tk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
    xscroll.pack(side=tk.BOTTOM, fill=tk.X)

    yscroll = tk.Scrollbar(list_frame)
    yscroll.pack(side=tk.RIGHT, fill=tk.Y)

    global file_listbox
    file_listbox = tk.Listbox(list_frame,
                              selectmode=tk.SINGLE,
                              xscrollcommand=xscroll.set,
                              yscrollcommand=yscroll.set,
                              height=5)
    file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    xscroll.config(command=file_listbox.xview)
    yscroll.config(command=file_listbox.yview)
    
    list_frame.columnconfigure(0, weight=1)
    list_frame.rowconfigure(0, weight=1)
    
    file_listbox.bind('<<ListboxSelect>>', on_animal_select)

    btn_frame = ttk.Frame(multi_animal_frame)
    btn_frame.pack(fill="both", padx=5, pady=5)

    style = ttk.Style()
    style.configure("Accent.TButton", 
                    font=("Microsoft YaHei", 10),
                    padding=(10, 5))

    clear_selected_btn = ttk.Button(btn_frame, 
                                text="Clear Selected", 
                                command=clear_selected,
                                style="Accent.TButton")
    clear_selected_btn.pack(fill="x", padx=2, pady=(2, 1))

    clear_all_btn = ttk.Button(btn_frame, 
                            text="Clear All", 
                            command=clear_all,
                            style="Accent.TButton")
    clear_all_btn.pack(fill="x", padx=2, pady=(1, 2))

def fiber_preprocessing():
    global preprocess_frame
    
    prep_window = tk.Toplevel(root)
    prep_window.title("Fiber Data Preprocessing")
    prep_window.geometry("320x600")
    prep_window.transient(root)
    prep_window.grab_set()
    
    status_label = tk.Label(prep_window, text="Fiber Data Preprocessing", font=("Arial", 12, "bold"))
    status_label.pack(pady=10)
    
    main_frame = ttk.Frame(prep_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    signal_frame = ttk.LabelFrame(main_frame, text="Signal Selection")
    signal_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(signal_frame, text="Target Signal:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    target_options = ["470", "560"]
    target_menu = ttk.OptionMenu(signal_frame, target_signal_var, "470", *target_options)
    target_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
    
    ttk.Label(signal_frame, text="Reference Signal:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ref_options = ["410", "baseline"]
    ref_menu = ttk.OptionMenu(signal_frame, reference_signal_var, "410", *ref_options)
    ref_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    
    global baseline_frame
    baseline_frame = ttk.LabelFrame(main_frame, text="Baseline Period")
    baseline_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Label(baseline_frame, text="Start (s):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    ttk.Entry(baseline_frame, textvariable=baseline_start, width=5).grid(row=0, column=1, padx=5, pady=5)
    
    ttk.Label(baseline_frame, text="End (s):").grid(row=0, column=2, sticky="w", padx=5, pady=5)
    ttk.Entry(baseline_frame, textvariable=baseline_end, width=5).grid(row=0, column=3, padx=5, pady=5)
    
    if reference_signal_var.get() != "baseline":
            baseline_frame.pack_forget()

    reference_signal_var.trace_add("write", update_baseline_ui)

    global smooth_frame
    smooth_frame = ttk.LabelFrame(main_frame, text="Smoothing")
    smooth_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Checkbutton(smooth_frame, text="Apply Smoothing", variable=apply_smooth,
                    command=lambda: toggle_widgets(smooth_frame, apply_smooth.get(), 1)).grid(row=0, column=0, sticky="w")
    
    ttk.Label(smooth_frame, text="Window Size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    ttk.Scale(smooth_frame, from_=3, to=101, orient=tk.HORIZONTAL, 
             variable=smooth_window, length=100).grid(row=1, column=1, padx=5, pady=5)
    ttk.Label(smooth_frame, textvariable=smooth_window).grid(row=1, column=2, padx=5, pady=5)
    
    ttk.Label(smooth_frame, text="Polynomial Order:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    ttk.Scale(smooth_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
             variable=smooth_order, length=100).grid(row=2, column=1, padx=5, pady=5)
    ttk.Label(smooth_frame, textvariable=smooth_order).grid(row=2, column=2, padx=5, pady=5)
    
    global baseline_corr_frame
    baseline_corr_frame = ttk.LabelFrame(main_frame, text="Baseline Correction")
    baseline_corr_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Checkbutton(baseline_corr_frame, text="Apply Baseline Correction", variable=apply_baseline, 
                    command=lambda: toggle_widgets(baseline_corr_frame, apply_baseline.get(), 1)).grid(row=0, column=0, sticky="w")

    ttk.Label(baseline_corr_frame, text="Baseline Model:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    model_options = ["Polynomial", "Exponential"]
    model_menu = ttk.OptionMenu(baseline_corr_frame, baseline_model, "Polynomial", *model_options)
    model_menu.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
    
    global motion_frame
    motion_frame = ttk.LabelFrame(main_frame, text="Motion Correction")
    motion_frame.pack(fill=tk.X, padx=5, pady=5)
    
    ttk.Checkbutton(motion_frame, text="Apply Motion Correction", variable=apply_motion).pack(anchor="w", padx=5, pady=5)
    
    global button_frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, padx=5, pady=10)
    
    ttk.Button(button_frame, text="Apply Preprocessing", 
              command=lambda: apply_preprocessing_wrapper()).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="Close", 
              command=prep_window.destroy).pack(side=tk.RIGHT, padx=5)
    

def update_baseline_ui(*args):
    global baseline_frame, smooth_frame, baseline_corr_frame, motion_frame, button_frame

    try:
        if reference_signal_var.get() == "baseline":
            if baseline_frame.winfo_exists():
                baseline_frame.pack(fill="x", padx=5, pady=5)
            if smooth_frame.winfo_exists():
                smooth_frame.pack_forget()
                smooth_frame.pack(fill=tk.X, padx=5, pady=5)
            if baseline_corr_frame.winfo_exists():
                baseline_corr_frame.pack_forget()
                baseline_corr_frame.pack(fill=tk.X, padx=5, pady=5)
            if motion_frame.winfo_exists():
                motion_frame.pack_forget()
                motion_frame.pack(fill=tk.X, padx=5, pady=5)
            if button_frame.winfo_exists():
                button_frame.pack_forget()
                button_frame.pack(fill=tk.X, padx=5, pady=10)
        else:
            if baseline_frame.winfo_exists():
                baseline_frame.pack_forget()
            if smooth_frame.winfo_exists():
                smooth_frame.pack_forget()
                smooth_frame.pack(fill=tk.X, padx=5, pady=5)
            if baseline_corr_frame.winfo_exists():
                baseline_corr_frame.pack_forget()
                baseline_corr_frame.pack(fill=tk.X, padx=5, pady=5)
            if motion_frame.winfo_exists():
                motion_frame.pack_forget()
                motion_frame.pack(fill=tk.X, padx=5, pady=5)
            if button_frame.winfo_exists():
                button_frame.pack_forget()
                button_frame.pack(fill=tk.X, padx=5, pady=10)
    except tk.TclError:
        pass

def toggle_widgets(parent_frame, show, index):
        children = parent_frame.winfo_children()
        if len(children) > index:
            if show:
                children[index].grid()
            else:
                children[index].grid_remove()

def apply_preprocessing_wrapper():
    try:
        if multi_animal_data:
            animal_data = multi_animal_data[current_animal_index]
        else:
            animal_data = None
        
        target_signal = str(target_signal_var.get())
        reference_signal = str(reference_signal_var.get())
        baseline_start_val = float(baseline_start.get())
        baseline_end_val = float(baseline_end.get())
        apply_smooth_val = bool(apply_smooth.get())
        window_size_val = int(smooth_window.get())
        poly_order_val = int(smooth_order.get())
        apply_baseline_val = bool(apply_baseline.get())
        baseline_model_val = str(baseline_model.get())
        apply_motion_val = bool(apply_motion.get())
        
        success = apply_preprocessing(
            animal_data, 
            target_signal,
            reference_signal,
            (baseline_start_val, baseline_end_val),
            apply_smooth_val,
            window_size_val,
            poly_order_val,
            apply_baseline_val,
            baseline_model_val,
            apply_motion_val
        )
        
        if not success:
            log_message("Preprocessing failed", "ERROR")
            return
        
        if fiber_plot_window:
            fiber_plot_window.animal_data = animal_data if animal_data else {
                'preprocessed_data': globals().get('preprocessed_data'),
                'channels': globals().get('channels'),
                'active_channels': globals().get('active_channels'),
                'channel_data': globals().get('channel_data')
            }
            fiber_plot_window.set_plot_type("preprocessed")
        
        log_message("Preprocessing applied successfully")
        
    except Exception as e:
        log_message(f"Preprocessing failed: {str(e)}", "ERROR")

def calculate_and_plot_dff_wrapper():
    try:
        if multi_animal_data:
            animal_data = multi_animal_data[current_animal_index]
            if 'preprocessed_data' not in animal_data or animal_data['preprocessed_data'] is None:
                log_message("Please apply preprocessing first", "WARNING")
                return
        else:
            animal_data = None
            if not hasattr(globals(), 'preprocessed_data') or globals().get('preprocessed_data') is None:
                log_message("Please apply preprocessing first", "WARNING")
                return
        
        target_signal = str(target_signal_var.get())
        reference_signal = str(reference_signal_var.get())
        baseline_start_val = float(baseline_start.get())
        baseline_end_val = float(baseline_end.get())
        apply_baseline_val = bool(apply_baseline.get())
        
        calculate_and_plot_dff(
            animal_data, 
            target_signal,
            reference_signal,
            (baseline_start_val, baseline_end_val),
            apply_baseline_val
        )
        
        if multi_animal_data:
            dff_col = f"CH{multi_animal_data[current_animal_index]['active_channels'][0]}_dff" if multi_animal_data[current_animal_index]['active_channels'] else None
            if dff_col and dff_col in multi_animal_data[current_animal_index]['preprocessed_data'].columns:
                multi_animal_data[current_animal_index]['dff_data'] = multi_animal_data[current_animal_index]['preprocessed_data'][dff_col]
        else:
            if hasattr(globals(), 'active_channels') and globals().get('active_channels'):
                dff_col = f"CH{globals().get('active_channels')[0]}_dff"
                if dff_col in globals().get('preprocessed_data', {}).columns:
                    globals()['dff_data'] = globals()['preprocessed_data'][dff_col]
        
        if fiber_plot_window:
            fiber_plot_window.set_plot_type("dff")
        
        log_message("ΔF/F calculated successfully")
    except Exception as e:
        log_message(f"ΔF/F calculation failed: {str(e)}", "ERROR")

def calculate_and_plot_zscore_wrapper():
    try:
        if multi_animal_data:
            animal_data = multi_animal_data[current_animal_index]
        else:
            animal_data = None
        
        zscore_data = calculate_and_plot_zscore(animal_data, 
                                 target_signal_var.get(),
                                 reference_signal_var.get(),
                                 (baseline_start.get(), baseline_end.get()),
                                 apply_baseline.get())
        
        if multi_animal_data:
            multi_animal_data[current_animal_index]['zscore_data'] = zscore_data
        else:
            globals()['zscore_data'] = zscore_data
        
        if fiber_plot_window:
            fiber_plot_window.set_plot_type("zscore")
        
        log_message("Z-score calculated successfully")
    except Exception as e:
        log_message(f"Z-score calculation failed: {str(e)}", "ERROR")

def running_data_analysis_wrapper(analysis_type):
    """Wrapper function for different types of running data analysis"""
    try:
        if multi_animal_data:
            animal_data = multi_animal_data[current_animal_index]
            ast2_data = animal_data.get('ast2_data_adjusted')
            processed_data = animal_data.get('running_processed_data')
        else:
            ast2_data = globals().get('ast2_data_adjusted')
            processed_data = globals().get('running_processed_data')
        
        if ast2_data is None:
            log_message("No running data available for analysis", "WARNING")
            return
        
        # Use filtered data if available, otherwise use original
        if processed_data:
            processed_ast2_data = {
                'header': ast2_data['header'] if ast2_data else {},
                'data': {
                    'timestamps': processed_data['timestamps'],
                    'speed': processed_data['filtered_speed']
                }
            }
            log_message(f"Using filtered running data for {analysis_type} analysis", "INFO")
            treadmill_behaviors = classify_treadmill_behavior(processed_ast2_data)
        else:
            log_message(f"Using original running data for {analysis_type} analysis", "INFO")
            treadmill_behaviors = classify_treadmill_behavior(ast2_data)
        
        # Store the full analysis results
        if multi_animal_data:
            multi_animal_data[current_animal_index]['treadmill_behaviors'] = treadmill_behaviors
        else:
            globals()['treadmill_behaviors'] = treadmill_behaviors
        
        # Get the specific analysis data based on type
        if analysis_type in treadmill_behaviors:
            analysis_data = treadmill_behaviors[analysis_type]
        else:
            log_message(f"Analysis type '{analysis_type}' not found in results", "ERROR")
            return
        
        # Update running plot with the specific analysis data
        if running_plot_window:
            running_plot_window.current_analysis_type = analysis_type
            running_plot_window.analysis_data = analysis_data
            running_plot_window.update_plot()
        
        if fiber_plot_window:
            fiber_plot_window.update_running_analysis(analysis_type, analysis_data)

        # Display results in a window
        display_running_analysis_results(analysis_type, analysis_data, treadmill_behaviors)
        
        log_message(f"Running {analysis_type} analysis completed")
        
    except Exception as e:
        log_message(f"Running {analysis_type} analysis failed: {str(e)}", "ERROR")

def display_running_analysis_results(analysis_type, analysis_data, treadmill_behaviors):
    """Display running analysis results in a new window"""
    result_window = tk.Toplevel(root)
    result_window.title(f"Running Analysis Results - {analysis_type.replace('_', ' ').title()}")
    result_window.geometry("600x400")
    
    # Create notebook for multiple tabs
    notebook = ttk.Notebook(result_window)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Results tab
    results_frame = ttk.Frame(notebook)
    notebook.add(results_frame, text="Results")
    
    # Create text widget with scrollbar
    text_frame = ttk.Frame(results_frame)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 9))
    scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
    text_widget.configure(yscrollcommand=scrollbar.set)
    
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Generate results text
    result_text = generate_analysis_text(analysis_type, analysis_data, treadmill_behaviors)
    text_widget.insert(tk.END, result_text)
    text_widget.config(state=tk.DISABLED)
    
    # Statistics tab
    stats_frame = ttk.Frame(notebook)
    notebook.add(stats_frame, text="Statistics")
    
    stats_text = tk.Text(stats_frame, wrap=tk.WORD, font=("Consolas", 9))
    stats_scrollbar = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=stats_text.yview)
    stats_text.configure(yscrollcommand=stats_scrollbar.set)
    
    stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    stats_text.insert(tk.END, generate_statistics_text(treadmill_behaviors))
    stats_text.config(state=tk.DISABLED)

def generate_analysis_text(analysis_type, analysis_data, treadmill_behaviors):
    """Generate formatted text for analysis results"""
    text = f"ANALYSIS TYPE: {analysis_type.replace('_', ' ').upper()}\n"
    text += "=" * 50 + "\n\n"
    
    if analysis_type in ['movement_periods', 'rest_periods', 'continuous_locomotion_periods']:
        # Period-based analyses
        if analysis_data:
            text += f"Found {len(analysis_data)} periods:\n\n"
            for i, (start, end) in enumerate(analysis_data):
                duration = end - start
                text += f"Period {i+1}:\n"
                text += f"  Start: {start:.2f}s\n"
                text += f"  End: {end:.2f}s\n"
                text += f"  Duration: {duration:.2f}s\n"
                text += f"  Mean Speed: {np.mean(treadmill_behaviors['smoothed_velocity'][int(start*10):int(end*10)]):.2f} cm/s\n\n"
        else:
            text += "No periods found.\n"
            
    elif analysis_type in ['general_onsets', 'jerks', 'locomotion_initiations', 'locomotion_terminations']:
        # Event-based analyses
        if analysis_data:
            text += f"Found {len(analysis_data)} events:\n\n"
            for i, event_time in enumerate(analysis_data):
                text += f"Event {i+1}: {event_time:.2f}s\n"
        else:
            text += "No events found.\n"
    
    return text

def generate_statistics_text(treadmill_behaviors):
    """Generate statistics text for all analysis types"""
    text = "OVERALL STATISTICS\n"
    text += "=" * 50 + "\n\n"
    
    # Basic data info
    velocity = treadmill_behaviors['smoothed_velocity']
    text += f"Total recording duration: {treadmill_behaviors['smoothed_velocity'].shape[0]/10:.1f}s\n"
    text += f"Mean velocity: {np.mean(velocity):.2f} cm/s\n"
    text += f"Max velocity: {np.max(velocity):.2f} cm/s\n"
    text += f"Velocity standard deviation: {np.std(velocity):.2f} cm/s\n\n"
    
    # Analysis-specific statistics
    for analysis_type in ['movement_periods', 'rest_periods', 'continuous_locomotion_periods']:
        data = treadmill_behaviors[analysis_type]
        if data:
            durations = [end - start for start, end in data]
            text += f"{analysis_type.replace('_', ' ').title()}:\n"
            text += f"  Count: {len(data)}\n"
            text += f"  Mean duration: {np.mean(durations):.2f}s\n"
            text += f"  Total duration: {np.sum(durations):.2f}s\n"
            text += f"  Percentage of recording: {np.sum(durations)/(velocity.shape[0]/10)*100:.1f}%\n\n"
    
    for analysis_type in ['general_onsets', 'jerks', 'locomotion_initiations', 'locomotion_terminations']:
        data = treadmill_behaviors[analysis_type]
        if data:
            text += f"{analysis_type.replace('_', ' ').title()}:\n"
            text += f"  Count: {len(data)}\n"
            if len(data) > 1:
                intervals = np.diff(data)
                text += f"  Mean interval: {np.mean(intervals):.2f}s\n"
                text += f"  Interval std: {np.std(intervals):.2f}s\n\n"
            else:
                text += "\n"
    
    return text

def setup_log_display():
    global log_text_widget
    for widget in bottom_display_frame.winfo_children():
        widget.destroy()
    
    log_text_widget = tk.Text(bottom_display_frame, wrap=tk.WORD, height=10, font=("Consolas", 9))
    log_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    set_log_widget(log_text_widget)
    log_message("The log system has been initialized. All messages will be displayed here.", "INFO")

def running_data_preprocess():
    """Running data preprocessing dialog"""
    global multi_animal_data, current_animal_index, running_channel, invert_running, threadmill_diameter
    
    # Get current animal data
    if not multi_animal_data or current_animal_index >= len(multi_animal_data):
        log_message("No animal data available for preprocessing", "WARNING")
        return
    
    animal_data = multi_animal_data[current_animal_index]
    ast2_data = animal_data.get('ast2_data_adjusted')
    
    if ast2_data is None:
        log_message("No running data available for preprocessing", "WARNING")
        return
    
    # Create preprocessing dialog
    prep_window = tk.Toplevel(root)
    prep_window.title("Running Data Preprocessing Settings")
    prep_window.geometry("500x800")
    prep_window.transient(root)
    prep_window.grab_set()
    
    # Main frame with scrollbar for adaptive height
    main_frame = ttk.Frame(prep_window, padding=15)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="🏃 Running Data Preprocessing", 
                           font=("Arial", 14, "bold"))
    title_label.pack(pady=(0, 15))
    
    # Basic Settings Frame
    basic_frame = ttk.LabelFrame(main_frame, text="📊 Basic Settings", padding=10)
    basic_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Running Channel Selection
    channel_frame = ttk.Frame(basic_frame)
    channel_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(channel_frame, text="Running Channel:").pack(side=tk.LEFT)
    
    # Get available channels from AST2 header
    available_channels = []
    if 'ast2_data' in animal_data and animal_data['ast2_data']:
        header = animal_data['ast2_data']['header']
        if 'activeChIDs' in header:
            available_channels = header['activeChIDs']
        else:
            # Fallback: try to determine from data shape
            if 'files' in animal_data and 'ast2' in animal_data['files']:
                try:
                    header, raw_data = h_AST2_readData(animal_data['files']['ast2'])
                    available_channels = list(range(len(raw_data)))
                except:
                    available_channels = [0, 1, 2, 3]  # Default fallback
    
    if not available_channels:
        available_channels = [0, 1, 2, 3]  # Default fallback
    
    channel_var = tk.StringVar(value=str(running_channel))
    channel_combo = ttk.Combobox(channel_frame, textvariable=channel_var, 
                                values=available_channels, state="readonly", width=10)
    channel_combo.pack(side=tk.RIGHT, padx=(10, 0))
    
    # Invert Running Checkbox
    invert_var = tk.BooleanVar(value=invert_running)
    invert_check = ttk.Checkbutton(basic_frame, text="Invert Running Values", 
                                  variable=invert_var)
    invert_check.pack(anchor="w", pady=5)
    
    # Threadmill Diameter
    diameter_frame = ttk.Frame(basic_frame)
    diameter_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(diameter_frame, text="Threadmill Diameter (cm):").pack(side=tk.LEFT)
    
    diameter_var = tk.StringVar(value=str(threadmill_diameter))
    diameter_entry = ttk.Entry(diameter_frame, textvariable=diameter_var, width=10)
    diameter_entry.pack(side=tk.RIGHT, padx=(10, 0))
    
    # Filter Configuration Frame
    filter_frame = ttk.LabelFrame(main_frame, text="🔧 Filter Configuration", padding=10)
    filter_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Smoothing Method Selection
    smooth_methods = [
        {"name": "No Smoothing", "type": "none", "params": []},
        {"name": "Moving Average", "type": "moving_average", 
         "params": [
             {"name": "window_size", "type": "int", "default": 20, "min": 3, "max": 51, "step": 2}
         ]},
        {"name": "Median Filter", "type": "median",
         "params": [
             {"name": "window_size", "type": "int", "default": 100, "min": 3, "max": 51, "step": 2}
         ]},
        {"name": "Savitzky-Golay", "type": "savitzky_golay",
         "params": [
             {"name": "window_size", "type": "int", "default": 100, "min": 5, "max": 51, "step": 2},
             {"name": "poly_order", "type": "int", "default": 3, "min": 1, "max": 5}
         ]},
        {"name": "Butterworth Low-pass", "type": "butterworth",
         "params": [
             {"name": "sampling_rate", "type": "float", "default": 10.0, "min": 1.0, "max": 100.0},
             {"name": "cutoff_freq", "type": "float", "default": 2.0, "min": 0.1, "max": 10.0},
             {"name": "filter_order", "type": "int", "default": 2, "min": 1, "max": 5}
         ]}
    ]
    
    # Method selection
    method_frame = ttk.Frame(filter_frame)
    method_frame.pack(fill=tk.X, pady=(0, 10))
    
    ttk.Label(method_frame, text="Smoothing Method:").pack(side=tk.LEFT)
    
    method_names = [method["name"] for method in smooth_methods]
    method_var = tk.StringVar(value=method_names[0])
    method_combo = ttk.Combobox(method_frame, textvariable=method_var, 
                               values=method_names, state="readonly", width=20)
    method_combo.pack(side=tk.RIGHT, padx=(10, 0))
    
    # Parameters frame (will be populated dynamically)
    params_frame = ttk.Frame(filter_frame)
    params_frame.pack(fill=tk.X, pady=5)
    
    # Store parameter variables
    param_vars = {}
    
    def update_parameters(*args):
        """Update parameter inputs based on selected method"""
        # Clear previous parameters
        for widget in params_frame.winfo_children():
            widget.destroy()
        
        param_vars.clear()
        
        # Find selected method
        selected_method_name = method_var.get()
        selected_method = None
        for method in smooth_methods:
            if method["name"] == selected_method_name:
                selected_method = method
                break
        
        if not selected_method or not selected_method["params"]:
            # No parameters needed
            ttk.Label(params_frame, text="No parameters needed for this method", 
                     foreground="gray").pack(pady=10)
            # Adjust window height
            return
        
        # Create parameter inputs
        ttk.Label(params_frame, text="Parameters:", font=("Arial", 9, "bold")).pack(anchor="w", pady=(0, 5))
        
        for param in selected_method["params"]:
            param_frame = ttk.Frame(params_frame)
            param_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(param_frame, text=f"{param['name'].replace('_', ' ').title()}:").pack(side=tk.LEFT)
            
            if param['type'] == 'int':
                var = tk.IntVar(value=param['default'])
                widget = ttk.Spinbox(param_frame, from_=param['min'], to=param['max'], 
                                   textvariable=var, width=8)
            else:  # float
                var = tk.DoubleVar(value=param['default'])
                widget = ttk.Spinbox(param_frame, from_=param['min'], to=param['max'], 
                                   increment=0.1, textvariable=var, width=8)
            
            widget.pack(side=tk.RIGHT, padx=(10, 0))
            param_vars[param['name']] = var
    
    # Initial parameter setup
    method_var.trace('w', update_parameters)
    update_parameters()
    
    # Preview Frame
    preview_frame = ttk.LabelFrame(main_frame, text="👁️ Preview", padding=10)
    preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
    
    # Create matplotlib figure for preview
    fig = Figure(figsize=(8, 3), dpi=80)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, preview_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def update_preview():
        """Update the preview plot with current settings"""
        ax.clear()
        
        # Get current filter settings
        selected_method_name = method_var.get()
        selected_method = None
        for method in smooth_methods:
            if method["name"] == selected_method_name:
                selected_method = method
                break
        
        filter_settings = []
        if selected_method and selected_method["type"] != "none":
            params = {}
            for param_name, var in param_vars.items():
                params[param_name] = var.get()
            
            filter_settings.append({
                'type': selected_method['type'],
                'params': params
            })
        
        # Apply preprocessing
        processed_data = preprocess_running_data(ast2_data, filter_settings)
        
        if processed_data:
            timestamps = processed_data['timestamps']
            original_speed = processed_data['original_speed']
            filtered_speed = processed_data['filtered_speed']
            
            invert_preview = invert_var.get()
            if invert_preview:
                original_speed = -original_speed
                filtered_speed = -filtered_speed
            
            # Plot original and filtered data
            ax.plot(timestamps, original_speed, 'b-', alpha=0.7, label='Original', linewidth=1)
            ax.plot(timestamps, filtered_speed, 'r-', label='Filtered', linewidth=1.5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Speed (cm/s)')
            ax.set_title('Running Speed: Original vs Filtered')
            ax.legend()
            ax.grid(False)
            
            canvas.draw()
    
    def apply_all_settings():
        """Apply all settings (channel, invert, diameter, and filters)"""
        global running_channel, invert_running, threadmill_diameter
        
        try:
            # Apply basic settings
            new_channel = int(channel_var.get())
            if new_channel < 0:
                raise ValueError("Channel must be non-negative")
            running_channel = new_channel
            
            invert_running = invert_var.get()
            
            new_diameter = float(diameter_var.get())
            if new_diameter <= 0:
                raise ValueError("Diameter must be positive")
            threadmill_diameter = new_diameter
            
            # Update AST2 data with new settings
            if 'files' in animal_data and 'ast2' in animal_data['files']:
                try:
                    header, raw_data = h_AST2_readData(animal_data['files']['ast2'])
                    if running_channel < len(raw_data):
                        speed = h_AST2_raw2Speed(raw_data[running_channel], header, voltageRange=None)
                        ast2_data_updated = {
                            'header': header,
                            'data': speed
                        }
                        animal_data['ast2_data'] = ast2_data_updated
                        
                        # Re-align data
                        align_data(animal_data)
                        
                        log_message(f"AST2 data updated with channel {running_channel}, diameter {threadmill_diameter} cm, invert: {invert_running}")
                    else:
                        log_message(f"Invalid channel: {running_channel}, max available: {len(raw_data)-1}", "ERROR")
                        return
                except Exception as e:
                    log_message(f"Failed to update AST2 data: {str(e)}", "ERROR")
                    return
            
            # Apply filter settings
            selected_method_name = method_var.get()
            selected_method = None
            for method in smooth_methods:
                if method["name"] == selected_method_name:
                    selected_method = method
                    break
            
            filter_settings = []
            if selected_method and selected_method["type"] != "none":
                params = {}
                for param_name, var in param_vars.items():
                    params[param_name] = var.get()
                
                filter_settings.append({
                    'type': selected_method['type'],
                    'params': params
                })
            
            # Apply preprocessing
            processed_data = preprocess_running_data(animal_data['ast2_data_adjusted'], filter_settings)
            
            if processed_data:
                # Store processed data
                animal_data['running_processed_data'] = processed_data
                
                # Update running visualization
                if running_plot_window:
                    running_plot_window.animal_data = animal_data
                    running_plot_window.update_plot()
                
                log_message(f"Running settings applied: Channel {running_channel}, Diameter {threadmill_diameter}cm, Invert {invert_running}, Filter: {selected_method_name}")
                prep_window.destroy()
            else:
                log_message("Failed to process running data", "ERROR")
                
        except ValueError as e:
            log_message(f"Invalid setting value: {str(e)}", "ERROR")
        except Exception as e:
            log_message(f"Error applying running settings: {str(e)}", "ERROR")
    
    # Button Frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)
    
    ttk.Button(button_frame, text="🔄 Update Preview", 
              command=update_preview).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(button_frame, text="✅ Apply All Settings", 
              command=apply_all_settings).pack(side=tk.LEFT, padx=5)
    
    ttk.Button(button_frame, text="❌ Cancel", 
              command=prep_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    # Initial preview
    update_preview()

def init_multimodal_analysis():
    """Initialize multimodal analyzer"""
    global multimodal_analyzer, multi_animal_data, current_animal_index, selected_bodyparts
    
    # Check if data is available
    if not multi_animal_data:
        log_message("Please import animal data first", "ERROR")
        return None
    
    if current_animal_index >= len(multi_animal_data):
        log_message("Please select an animal first", "ERROR")
        return None
    
    # Create analyzer instance with selected bodyparts
    multimodal_analyzer = MultimodalAnalysis(root, multi_animal_data, current_animal_index, selected_bodyparts)
    return multimodal_analyzer

def select_experiment_mode():
    """Open dialog to select experiment mode"""
    global current_experiment_mode
    
    mode_window = tk.Toplevel(root)
    mode_window.title("Select Experiment Mode")
    mode_window.geometry("400x400")
    mode_window.transient(root)
    mode_window.grab_set()
    
    # Title
    title_label = tk.Label(mode_window, text="🔬 Experiment Mode Selection", 
                          font=("Arial", 14, "bold"))
    title_label.pack(pady=20)
    
    # Description
    desc_label = tk.Label(mode_window, 
                         text="Select the type of data you want to analyze:",
                         font=("Arial", 10))
    desc_label.pack(pady=5)
    
    # Mode selection frame
    mode_frame = tk.Frame(mode_window)
    mode_frame.pack(pady=20)
    
    mode_var = tk.StringVar(value=current_experiment_mode)
    
    # Mode 1: Fiber + AST2
    mode1_radio = tk.Radiobutton(
        mode_frame,
        text="Fiber + AST2 (Running Only)",
        variable=mode_var,
        value=EXPERIMENT_MODE_FIBER_AST2,
        font=("Arial", 10),
        justify=tk.LEFT
    )
    mode1_radio.pack(anchor="w", pady=5)
    
    mode1_desc = tk.Label(mode_frame, 
                         text="  • Fiber photometry data\n  • Running wheel data (AST2)",
                         font=("Arial", 9), fg="gray", justify=tk.LEFT)
    mode1_desc.pack(anchor="w", padx=20)
    
    # Mode 2: Fiber + AST2 + DLC
    mode2_radio = tk.Radiobutton(
        mode_frame,
        text="Fiber + AST2 + DLC (Full Analysis)",
        variable=mode_var,
        value=EXPERIMENT_MODE_FIBER_AST2_DLC,
        font=("Arial", 10),
        justify=tk.LEFT
    )
    mode2_radio.pack(anchor="w", pady=(15, 5))
    
    mode2_desc = tk.Label(mode_frame, 
                         text="  • Fiber photometry data\n  • Running wheel data (AST2)\n  • DeepLabCut behavioral tracking",
                         font=("Arial", 9), fg="gray", justify=tk.LEFT)
    mode2_desc.pack(anchor="w", padx=20)
    
    def apply_mode():
        global current_experiment_mode
        new_mode = mode_var.get()
        
        # Check if there's existing data
        if multi_animal_data:
            response = tk.messagebox.askyesno(
                "Confirm Mode Change",
                "Changing experiment mode will clear all loaded data.\nDo you want to continue?"
            )
            if not response:
                return
            
            # Clear existing data
            clear_all()
        
        current_experiment_mode = new_mode
        
        # Update UI based on mode
        update_ui_for_mode()
        
        mode_name = "Fiber + AST2" if new_mode == EXPERIMENT_MODE_FIBER_AST2 else "Fiber + AST2 + DLC"
        log_message(f"Experiment mode set to: {mode_name}", "INFO")
        
        mode_window.destroy()
    
    # Button frame
    button_frame = tk.Frame(mode_window)
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Apply", command=apply_mode,
             bg="#27ae60", fg="white", font=("Arial", 10, "bold"),
             padx=20, pady=5).pack(side=tk.LEFT, padx=5)
    
    tk.Button(button_frame, text="Cancel", command=mode_window.destroy,
             bg="#95a5a6", fg="white", font=("Arial", 10, "bold"),
             padx=20, pady=5).pack(side=tk.LEFT, padx=5)

def update_ui_for_mode():
    """Update UI elements based on current experiment mode"""
    global current_experiment_mode
    
    # Update menu items
    if current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2:
        # Disable DLC-related menu items
        behaviour_analysis_menu.entryconfig("Position Analysis", state="disabled")
        behaviour_analysis_menu.entryconfig("Displacement Analysis", state="disabled")
        behaviour_analysis_menu.entryconfig("X Displacement Analysis", state="disabled")
        behaviour_analysis_menu.entryconfig("Trajectory Point Cloud", state="disabled")
        
        # Update multimodal menu
        multimodal_menu.entryconfig("LOCOMOTION Tratejory Analysis", state="disabled")
        
    else:  # FIBER_AST2_DLC mode
        # Enable all menu items
        behaviour_analysis_menu.entryconfig("Position Analysis", state="normal")
        behaviour_analysis_menu.entryconfig("Displacement Analysis", state="normal")
        behaviour_analysis_menu.entryconfig("X Displacement Analysis", state="normal")
        behaviour_analysis_menu.entryconfig("Trajectory Point Cloud", state="normal")
        
        multimodal_menu.entryconfig("LOCOMOTION Tratejory Analysis", state="normal")
    
    # Clear left panel if in Fiber+AST2 mode
    if current_experiment_mode == EXPERIMENT_MODE_FIBER_AST2:
        for widget in left_frame.winfo_children():
            widget.destroy()
        
        info_label = tk.Label(left_frame, 
                             text="Fiber + AST2 Mode\n\nBodypart tracking\nnot available",
                             bg="#e0e0e0", fg="#666666",
                             font=("Arial", 10))
        info_label.pack(pady=50)

def save_path_setting():
    print(1)

def export_now_result():
    print(1)

def on_closing():
    log_message("Main window closed, exiting the program...", "INFO")
    root.quit()
    root.destroy()
    os.kill(os.getpid(), signal.SIGTERM)

root = tk.Tk()
root.title("Behavior Syllable Analysis")
root.state('zoomed')

root.protocol("WM_DELETE_WINDOW", on_closing)

# Experiment mode settings
EXPERIMENT_MODE_FIBER_AST2 = "fiber+ast2"
EXPERIMENT_MODE_FIBER_AST2_DLC = "fiber+ast2+dlc"
current_experiment_mode = EXPERIMENT_MODE_FIBER_AST2_DLC  # Default mode

target_signal_var = tk.StringVar(value="470")
reference_signal_var = tk.StringVar(value="410")
baseline_start = tk.DoubleVar(value=0)
baseline_end = tk.DoubleVar(value=60)
apply_smooth = tk.BooleanVar(value=False)
smooth_window = tk.IntVar(value=11)
smooth_order = tk.IntVar(value=5)
apply_baseline = tk.BooleanVar(value=False)
baseline_model = tk.StringVar(value="Polynomial")
apply_motion = tk.BooleanVar(value=False)
preprocess_frame = None
multimodal_analyzer = None

current_animal_index = 0
fiber_plot_window = None
running_plot_window = None
bodypart_buttons = {}
selected_bodyparts = set()
visualization_window = None

skeleton_connections = []
skeleton_building = False
skeleton_sequence = []

fps_var = None
time_unit_var = None
fps_conversion_var = None
fps_conversion_enabled = False
current_fps = 30

show_data_points_var = None

running_channel = 2
invert_running = False
threadmill_diameter = 22

log_text_widget = None

main_container = tk.Frame(root)
main_container.pack(fill=tk.BOTH, expand=True)

left_frame = tk.Frame(main_container, width=200, bg="#e0e0e0")
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 0), pady=5)
left_frame.pack_propagate(False)

middle_frame = tk.Frame(main_container)
middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)

central_display_frame = tk.Frame(middle_frame, bg="#f8f8f8", relief=tk.SUNKEN, bd=1)
central_display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

bottom_display_frame = tk.Frame(middle_frame, bg="#f0f0f0", relief=tk.SUNKEN, bd=1, height=170)
bottom_display_frame.pack(fill=tk.X, pady=(0, 0))
bottom_display_frame.pack_propagate(False)

right_frame = tk.Frame(main_container, width=200, bg="#e8e8e8")
right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 5), pady=5)
right_frame.pack_propagate(False)

left_label = tk.Label(left_frame, text="Left Button Area", bg="#f8f8f8", fg="#666666")
left_label.pack(pady=20)

central_label = tk.Label(central_display_frame, text="Central Display Area", bg="#f8f8f8", fg="#666666")
central_label.pack(pady=20)

bottom_label = tk.Label(bottom_display_frame, text="Bottom Log Area", bg="#f0f0f0", fg="#666666")
bottom_label.pack(pady=10)

right_label = tk.Label(right_frame, text="Right List Area", bg="#e8e8e8", fg="#666666")
right_label.pack(pady=20)

menubar = tk.Menu(root)
root.config(menu=menubar)

file_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=file_menu)
import_animals_menu = tk.Menu(file_menu, tearoff=0)
file_menu.add_cascade(label="Import Animals", menu=import_animals_menu, state="normal")
import_animals_menu.add_command(label="Import Single Animal", command=import_single_animal)
import_animals_menu.add_command(label="Import Multiple Animals", command=import_multi_animals)
file_menu.add_command(label="Export", command=export_now_result)
file_menu.add_command(label="Exit", command=root.quit)

# Analysis menu
analysis_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Analysis", menu=analysis_menu)
analysis_menu.add_command(label="Running Data Preprocess", command=running_data_preprocess)
running_analysis_menu = tk.Menu(analysis_menu, tearoff=0)
analysis_menu.add_cascade(label="Running Data Analysis", menu=running_analysis_menu, state="normal")
running_analysis_menu.add_command(label="Movement Periods", 
                                 command=lambda: running_data_analysis_wrapper('movement_periods'))
running_analysis_menu.add_command(label="Rest Periods", 
                                 command=lambda: running_data_analysis_wrapper('rest_periods'))
running_analysis_menu.add_command(label="General Onsets", 
                                 command=lambda: running_data_analysis_wrapper('general_onsets'))
running_analysis_menu.add_command(label="Jerks", 
                                 command=lambda: running_data_analysis_wrapper('jerks'))
running_analysis_menu.add_command(label="Locomotion Initiations", 
                                 command=lambda: running_data_analysis_wrapper('locomotion_initiations'))
running_analysis_menu.add_command(label="Continuous Locomotion Periods", 
                                 command=lambda: running_data_analysis_wrapper('continuous_locomotion_periods'))
running_analysis_menu.add_command(label="Locomotion Terminations", 
                                 command=lambda: running_data_analysis_wrapper('locomotion_terminations'))
analysis_menu.add_separator()
analysis_menu.add_command(label="Fiber Data Preprocessing", command=fiber_preprocessing)
fiber_analysis_menu = tk.Menu(analysis_menu, tearoff=0)
analysis_menu.add_cascade(label="Fiber Data Analysis", menu=fiber_analysis_menu, state="normal")
fiber_analysis_menu.add_command(label="Calculate ΔF/F", command=lambda: calculate_and_plot_dff_wrapper())
fiber_analysis_menu.add_command(label="Calculate Z-Score", command=lambda: calculate_and_plot_zscore_wrapper())
analysis_menu.add_separator()
behaviour_analysis_menu = tk.Menu(analysis_menu, tearoff=0)
analysis_menu.add_cascade(label="Behavior Analysis", menu=behaviour_analysis_menu, state="normal")
behaviour_analysis_menu.add_command(label="Position Analysis", command=lambda: position_analysis(parsed_data, selected_bodyparts, root))
behaviour_analysis_menu.add_command(label="Displacement Analysis", command=lambda: displacement_analysis(parsed_data, selected_bodyparts, root))
behaviour_analysis_menu.add_command(label="X Displacement Analysis", command=lambda: x_displacement_analysis(parsed_data, selected_bodyparts, root))
behaviour_analysis_menu.add_command(label="Trajectory Point Cloud", command=create_trajectory_pointcloud)

multimodal_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Multimodal Analysis", menu=multimodal_menu)
multimodal_menu.add_command(label="GENERAL ONSETS Fiber Anlysis", 
                           command=lambda: init_multimodal_analysis().general_onsets_analysis())
multimodal_menu.add_command(label="LOCOMOTION Tratejory Analysis", 
                           command=lambda: init_multimodal_analysis().continuous_locomotion_analysis())

setting_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Settings", menu=setting_menu)
setting_menu.add_command(label="Experiment Type", command=select_experiment_mode)

if __name__ == "__main__":
    selected_files = []
    multi_animal_data = []
    create_animal_list()
    setup_log_display()
    root.mainloop()