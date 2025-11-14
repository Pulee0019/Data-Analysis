import matplotlib.pyplot as plt
import numpy as np
import re

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
        # log_message(f"SpeedDataLength is not integer!  speedDataLength = {rawDataLength}, speedDownSampleFactor = {segmentLength}", "ERROR")
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
            print(f"Calibrated voltage range is {voltageRange}")
            # log_message(f"Calibrated voltage range is {voltageRange}")
        else:
            # log_message("Calibration error. Range too small")
            voltageRange = [0, 5]
    else:
        voltageRange = [0, 5]
        # log_message("Calibration fail! Return default: [0 5].")
    
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

running_data = r"D:\Expriment\Data\Acethylcholine\filename_AST2_1.ast2"
threadmill_diameter = 22  # in cm
invert_running = True
header, raw_data = h_AST2_readData(running_data)
data = h_AST2_raw2Speed(raw_data[2], header, voltageRange=None)
timestamps = data['timestamps']
speed = data['speed']
window_size = 100
kernel = np.ones(window_size) / window_size
filtered_speed = np.convolve(speed, kernel, mode='same')

plt.figure(figsize=(16,9))
plt.subplot(2, 1, 1)
plt.plot(timestamps, speed, label='speed on running wheel', color='g', alpha=0.7)
plt.xlim(timestamps[0], timestamps[-1])
plt.title("speed on treadmill")
plt.xlabel("time(s)")
plt.ylabel("speed(cm/s)")
plt.grid(False)

plt.subplot(2, 1, 2)
plt.plot(timestamps, filtered_speed, label='speed on running wheel', color='g', alpha=0.7)
plt.xlim(timestamps[0], timestamps[-1])
plt.title("filtered speed on treadmill")
plt.xlabel("time(s)")
plt.ylabel("speed(cm/s)")
plt.grid(False)

plt.show()