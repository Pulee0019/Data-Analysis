import numpy as np
from logger import log_message

def classify_treadmill_behavior(ast2_data, 
                               movement_velocity_threshold=2.2,
                               movement_acceleration_threshold=40,
                               rest_velocity_threshold=0.2,
                               jerk_velocity_threshold=2.2,
                               locomotion_initiation_velocity_threshold=4.5,
                               continuous_locomotion_velocity_threshold=4.5,
                               continuous_locomotion_min_duration=3.0,
                               termination_velocity_threshold=2.2,
                               termination_duration=1.0,
                               smoothing_window=0.05,  # 50ms
                               classification_window=1.0,  # ±1s window
                               general_onset_velocity_threshold=0.2,
                               general_onset_peak_velocity_threshold=1.0,
                               pre_movement_quiet_period=0.5):
    """
    Classify treadmill behavior based on the provided criteria
    
    Returns a dictionary containing:
    - movement_periods: periods with elevated velocity and acceleration
    - rest_periods: periods with no movement
    - general_onsets: movement onsets from rest
    - jerks: rapidly terminated movement onsets
    - locomotion_initiations: movement onsets leading to sustained locomotion
    - continuous_locomotion_periods: sustained locomotion bouts
    - locomotion_terminations: transitions from locomotion to rest
    """
    try:
        timestamps = ast2_data['data']['timestamps']
        velocity = ast2_data['data']['speed']  # Assuming speed is velocity in cm/s
        
        if len(timestamps) <= 1:
            return create_empty_results()
        
        # Calculate sample rate
        sample_interval = np.mean(np.diff(timestamps))
        sample_rate = 1.0 / sample_interval
        
        # Smooth velocity with 50ms sliding window
        # smoothed_velocity = smooth_velocity(velocity, smoothing_window, sample_rate)
        smoothed_velocity = velocity
        
        # Calculate acceleration from smoothed velocity
        acceleration = calculate_acceleration(smoothed_velocity, timestamps)
        
        # Classify movement and rest periods
        movement_periods, rest_periods = classify_movement_rest_periods(
            timestamps, smoothed_velocity, acceleration, 
            movement_velocity_threshold, movement_acceleration_threshold,
            rest_velocity_threshold, classification_window, sample_rate
        )
        
        # Detect general movement onsets
        general_onsets = detect_general_onsets(
            timestamps, smoothed_velocity, general_onset_velocity_threshold,
            general_onset_peak_velocity_threshold, pre_movement_quiet_period
        )
        
        # Classify movement onsets into jerks and locomotion initiations
        jerks, locomotion_initiations = classify_movement_onsets(
            general_onsets, timestamps, smoothed_velocity, jerk_velocity_threshold,
            locomotion_initiation_velocity_threshold
        )
        
        # Detect continuous locomotion periods
        continuous_locomotion_periods = detect_continuous_locomotion(
            timestamps, smoothed_velocity, continuous_locomotion_velocity_threshold,
            continuous_locomotion_min_duration
        )

        continuous_locomotion_periods_for_termination = detect_continuous_locomotion(
            timestamps, smoothed_velocity, termination_velocity_threshold,
            continuous_locomotion_min_duration
        )
        
        # Detect locomotion terminations
        locomotion_terminations = detect_locomotion_terminations(
            timestamps, smoothed_velocity, continuous_locomotion_periods_for_termination,
            termination_velocity_threshold, termination_duration
        )
        
        return {
            'movement_periods': movement_periods,
            'rest_periods': rest_periods,
            'general_onsets': general_onsets,
            'jerks': jerks,
            'locomotion_initiations': locomotion_initiations,
            'continuous_locomotion_periods': continuous_locomotion_periods_for_termination,
            'locomotion_terminations': locomotion_terminations,
            'smoothed_velocity': smoothed_velocity,
            'acceleration': acceleration
        }
        
    except Exception as e:
        log_message(f"Error in classify_treadmill_behavior: {str(e)}", "ERROR")
        return create_empty_results()

def smooth_velocity(velocity, window_size, sample_rate):
    """Smooth velocity with a sliding window"""
    window_samples = int(window_size * sample_rate)
    if window_samples % 2 == 0:
        window_samples += 1  # Make odd for symmetric window
    
    if len(velocity) < window_samples:
        return velocity
    
    # Use uniform filter for smoothing
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(velocity, size=window_samples, mode='nearest')

def calculate_acceleration(velocity, timestamps):
    """Calculate acceleration from velocity"""
    if len(velocity) < 2:
        return np.zeros_like(velocity)
    
    # Calculate acceleration as difference in consecutive velocity measures
    acceleration = np.diff(velocity) / np.diff(timestamps)
    
    # Pad to match original length (forward difference)
    acceleration = np.concatenate([acceleration, [acceleration[-1]]])
    
    return acceleration

def classify_movement_rest_periods(timestamps, velocity, acceleration, 
                                  movement_vel_thresh, movement_acc_thresh,
                                  rest_vel_thresh, window_size, sample_rate):
    """Classify movement and rest periods"""
    window_samples = int(window_size * sample_rate)
    movement_mask = np.zeros(len(velocity), dtype=bool)
    rest_mask = np.zeros(len(velocity), dtype=bool)
    
    for i in range(len(velocity)):
        # Get window around current point
        start_idx = max(0, i - window_samples)
        end_idx = min(len(velocity), i + window_samples + 1)
        
        window_vel = velocity[start_idx:end_idx]
        window_acc = acceleration[start_idx:end_idx]
        
        # Movement period: velocity > threshold AND acceleration > threshold in window
        # movement_condition = (np.any(window_vel > movement_vel_thresh) and 
        #                     np.any(np.abs(window_acc) > movement_acc_thresh))
        
        movement_condition = np.any(window_vel > movement_vel_thresh)
                              
        # Rest period: no velocity > threshold in window
        rest_condition = np.all(window_vel <= rest_vel_thresh)
        
        movement_mask[i] = movement_condition
        rest_mask[i] = rest_condition
    
    # Find contiguous regions
    movement_periods = find_contiguous_regions(movement_mask, min_samples=1)
    rest_periods = find_contiguous_regions(rest_mask, min_samples=1)
    
    # Convert to timestamps
    movement_periods = [(timestamps[start], timestamps[end]) for start, end in movement_periods]
    rest_periods = [(timestamps[start], timestamps[end]) for start, end in rest_periods]
    
    return movement_periods, rest_periods

def detect_general_onsets(timestamps, velocity, onset_threshold, peak_threshold, quiet_period):
    """Detect general movement onsets from rest"""
    onsets = []
    sample_interval = np.mean(np.diff(timestamps))
    quiet_samples = int(quiet_period / sample_interval)
    post_onset_samples = int(1.0 / sample_interval)  # 1 second post-onset
    
    i = quiet_samples
    while i < len(velocity) - post_onset_samples:
        # Check if current point crosses threshold
        if velocity[i] >= onset_threshold and velocity[i-1] < onset_threshold:
            # Check pre-movement quiet period
            pre_movement_vel = velocity[i-quiet_samples:i]
            if np.all(pre_movement_vel <= onset_threshold):
                # Check peak velocity in post-onset period
                post_onset_vel = velocity[i:i+post_onset_samples]
                if np.max(post_onset_vel) >= peak_threshold:
                    onsets.append(timestamps[i])
                    i += post_onset_samples  # Skip ahead to avoid multiple detections
        i += 1
    
    return onsets

def classify_movement_onsets(onsets, timestamps, velocity, jerk_threshold, locomotion_threshold):
    """Classify movement onsets into jerks and locomotion initiations"""
    jerks = []
    locomotion_initiations = []
    sample_interval = np.mean(np.diff(timestamps))
    
    for onset_time in onsets:
        onset_idx = np.argmin(np.abs(timestamps - onset_time))
        
        # Window 1-2 seconds after onset for jerk classification
        start_1s = onset_idx + int(1.0 / sample_interval)
        end_2s = onset_idx + int(2.0 / sample_interval)
        
        if end_2s >= len(velocity):
            continue
            
        vel_1_2s = velocity[start_1s:end_2s]
        max_vel_1_2s = np.max(vel_1_2s)
        
        # Window 0.5-2 seconds after onset for locomotion classification
        start_0_5s = onset_idx + int(0.5 / sample_interval)
        vel_0_5_2s = velocity[start_0_5s:end_2s]
        mean_vel_0_5_2s = np.mean(vel_0_5_2s)
        
        if max_vel_1_2s < jerk_threshold:
            jerks.append(onset_time)
        elif mean_vel_0_5_2s > locomotion_threshold:
            locomotion_initiations.append(onset_time)
    
    return jerks, locomotion_initiations

def detect_continuous_locomotion(timestamps, velocity, velocity_threshold, min_duration):
    """Detect continuous locomotion periods"""
    sample_interval = np.mean(np.diff(timestamps))
    min_samples = int(min_duration / sample_interval)
    
    # Create mask for velocity above threshold
    locomotion_mask = velocity > velocity_threshold
    
    # Find contiguous regions
    locomotion_regions = find_contiguous_regions(locomotion_mask, min_samples)
    
    # Convert to timestamps
    locomotion_periods = [(timestamps[start], timestamps[end]) for start, end in locomotion_regions]
    
    return locomotion_periods

def detect_locomotion_terminations(timestamps, velocity, locomotion_periods, 
                                 velocity_threshold, termination_duration):
    """Detect locomotion terminations"""
    terminations = []
    sample_interval = np.mean(np.diff(timestamps))
    termination_samples = int(termination_duration / sample_interval)
    
    for start_time, end_time in locomotion_periods:
        end_idx = np.argmin(np.abs(timestamps - end_time))
        
        # Check if velocity drops below threshold and stays below
        if end_idx + termination_samples < len(velocity):
            post_termination_vel = velocity[end_idx+1:end_idx+termination_samples+1]
            if np.all(post_termination_vel < velocity_threshold):
                terminations.append(end_time)
    
    return terminations

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

def create_empty_results():
    """Create empty results dictionary"""
    return {
        'movement_periods': [],
        'rest_periods': [],
        'general_onsets': [],
        'jerks': [],
        'locomotion_initiations': [],
        'continuous_locomotion_periods': [],
        'locomotion_terminations': [],
        'smoothed_velocity': np.array([]),
        'acceleration': np.array([])
    }

# Keep the original function for backward compatibility
def detect_movement_bouts(ast2_data, immobility_threshold=0.2, min_immobility_duration=4, 
                         min_movement_duration=0.5, buffer_time=0.1):
    """Detect movement bouts based on the provided criteria"""
    try:
        timestamps = ast2_data['data']['timestamps']
        speed = ast2_data['data']['speed']
        if len(timestamps) > 1:
            sample_interval = np.mean(np.diff(timestamps))
            sample_rate = 1.0 / sample_interval
        else:
            sample_rate = 20.0
        
        # log_message(f"sample_rate = {sample_rate}")
        # log_message(f"Num of sample = {len(speed)}")
        min_immobility_samples = int(min_immobility_duration * sample_rate)
        buffer_samples = int(buffer_time * sample_rate)
        
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
        data_end_time = timestamps[-1]
        
        for movement_start, movement_end in movement_regions:
            prev_immobility = False
            for imm_start, imm_end in immobility_regions:
                if imm_end <= movement_start and (movement_start - imm_end) * sample_interval <= buffer_time:
                    immobility_duration = (imm_end - imm_start) * sample_interval
                    if immobility_duration >= min_immobility_duration:
                        prev_immobility = True
                        break
            
            next_immobility = False
            remaining_time = data_end_time - timestamps[movement_end]
            
            if remaining_time < min_immobility_duration:
                next_immobility = True
            else:
                for imm_start, imm_end in immobility_regions:
                    if imm_start >= movement_end and (imm_start - movement_end) * sample_interval <= buffer_time:
                        immobility_duration = (imm_end - imm_start) * sample_interval
                        if immobility_duration >= min_immobility_duration:
                            next_immobility = True
                            break
            
            if prev_immobility and (next_immobility or remaining_time < min_immobility_duration):
                immobility_std = np.std(speed[immobility_mask])
                movement_std_threshold = 2 * immobility_std
                
                onset_idx = movement_start
                for i in range(max(0, movement_start - buffer_samples), movement_start):
                    if np.abs(speed[i]) > movement_std_threshold:
                        onset_idx = i
                        break
                
                offset_idx = movement_end
                search_end = min(total_samples - 1, movement_end + buffer_samples)
                for i in range(search_end, movement_end, -1):
                    if np.abs(speed[i]) > movement_std_threshold:
                        offset_idx = i
                        break
                
                movement_bouts.append((
                    timestamps[onset_idx],
                    timestamps[offset_idx]
                ))
        
        return movement_bouts
        
    except Exception as e:
        log_message(f"Error in detect_movement_bouts: {str(e)}", "ERROR")
        return []

def apply_running_filters(speed_data, timestamps, filter_type, **kwargs):
    """
    Apply various filters to running speed data
    
    Parameters:
    - speed_data: array of running speed values
    - timestamps: array of corresponding timestamps
    - filter_type: type of filter to apply ('moving_average', 'median', 'savitzky_golay', 'butterworth')
    - **kwargs: filter-specific parameters
    
    Returns:
    - filtered_speed: filtered speed data
    """
    import numpy as np
    from scipy.signal import savgol_filter, butter, filtfilt
    
    if filter_type == 'moving_average':
        window_size = kwargs.get('window_size', 5)
        if window_size % 2 == 0:
            window_size += 1  # Make window size odd
        
        # Apply moving average filter
        kernel = np.ones(window_size) / window_size
        filtered_speed = np.convolve(speed_data, kernel, mode='same')
        
    elif filter_type == 'median':
        window_size = kwargs.get('window_size', 5)
        if window_size % 2 == 0:
            window_size += 1  # Make window size odd
        
        # Apply median filter
        filtered_speed = []
        half_window = window_size // 2
        
        for i in range(len(speed_data)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(speed_data), i + half_window + 1)
            window_data = speed_data[start_idx:end_idx]
            filtered_speed.append(np.median(window_data))
        
        filtered_speed = np.array(filtered_speed)
        
    elif filter_type == 'savitzky_golay':
        window_size = kwargs.get('window_size', 11)
        poly_order = kwargs.get('poly_order', 3)
        
        if window_size % 2 == 0:
            window_size += 1  # Make window size odd
        
        # Apply Savitzky-Golay filter
        filtered_speed = savgol_filter(speed_data, window_size, poly_order)
        
    elif filter_type == 'butterworth':
        sampling_rate = kwargs.get('sampling_rate', 10)  # Hz
        cutoff_freq = kwargs.get('cutoff_freq', 2.0)  # Hz
        filter_order = kwargs.get('filter_order', 2)
        
        # Design Butterworth low-pass filter
        nyquist_freq = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist_freq
        b, a = butter(filter_order, normal_cutoff, btype='low', analog=False)
        
        # Apply zero-phase filtering
        filtered_speed = filtfilt(b, a, speed_data)
        
    else:
        # No filtering
        filtered_speed = speed_data.copy()
    
    return filtered_speed

def preprocess_running_data(ast2_data, filter_settings):
    """
    Preprocess running data with specified filters
    
    Parameters:
    - ast2_data: dictionary containing running data
    - filter_settings: dictionary with filter configuration
    
    Returns:
    - processed_data: dictionary with original and filtered data
    """
    if ast2_data is None:
        return None
    
    try:
        speed = ast2_data['data']['speed']
        timestamps = ast2_data['data']['timestamps']
        
        # Apply filters sequentially
        filtered_speed = speed.copy()
        filter_history = ["original"]
        
        for filter_config in filter_settings:
            filter_type = filter_config['type']
            if filter_type != 'none':
                filtered_speed = apply_running_filters(
                    filtered_speed, timestamps, 
                    filter_type, **filter_config.get('params', {})
                )
                filter_history.append(filter_type)
        
        # Create processed data structure
        processed_data = {
            'original_speed': speed,
            'filtered_speed': filtered_speed,
            'timestamps': timestamps,
            'filter_history': filter_history,
            'filter_settings': filter_settings
        }
        
        return processed_data
        
    except Exception as e:
        log_message(f"Error in running data preprocessing: {str(e)}", "ERROR")
        return None