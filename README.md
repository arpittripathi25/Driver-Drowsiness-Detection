# Driver Drowsiness Detection System

A real-time webcam-based system that detects driver drowsiness by monitoring eye movement using computer vision techniques.

## Project Overview

This system uses your laptop's webcam to monitor a driver's face and detect signs of drowsiness by analyzing eye closure patterns. It implements the **Eye Aspect Ratio (EAR)** algorithm to determine when a driver's eyes are closed for extended periods, indicating potential drowsiness.

## Features

### Core Features
- **Real-time face detection** using dlib's frontal face detector
- **68-point facial landmark detection** for precise eye tracking
- **Eye Aspect Ratio (EAR) calculation** for accurate drowsiness detection
- **Configurable thresholds** for sensitivity adjustment
- **Audio alarm system** that triggers when drowsiness is detected
- **Visual status display** with color-coded indicators

### Advanced Features
- **Yawning detection** using mouth aspect ratio
- **Live UI dashboard** showing:
  - Current EAR value
  - System status (AWAKE/DROWSY)
  - Face detection status
  - FPS counter
  - Frame counter
- **Eye contour visualization** for better monitoring
- **Clean, modular code structure** with detailed comments

## How Eye Aspect Ratio (EAR) Works

The Eye Aspect Ratio is a simple but effective metric for detecting eye closure:

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

Where p1-p6 are the 6 eye landmarks:

- **p1, p4**: Horizontal eye corner points
- **p2, p6**: Vertical eye points (top/bottom)
- **p3, p5**: Vertical eye points (top/bottom)

**Key characteristics:**
- EAR values are relatively constant when eyes are open (~0.25-0.35)
- EAR rapidly approaches zero when eyes close
- EAR is invariant to head pose and image scaling

## System Requirements

- **Python 3.10+**
- **Webcam** (built-in laptop camera or external USB camera)
- **Windows/Linux/macOS** (tested on Windows)

## Installation

### 1. Clone or Download the Project

```bash
# If using git (optional)
git clone <repository-url>
cd Driver-Drowsiness-Detection

# Or simply download and extract the project folder
```

### 2. Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Download Facial Landmark Model

This is the most important step - you need to download dlib's 68-point facial landmark predictor:

```bash
# Download the shape predictor model
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Extract the downloaded file
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Move the .dat file to your project directory
# The file should be in the same folder as main.py
```

**Alternative download options:**
- Direct download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- Use your browser to download and extract manually

### 4. Prepare Alarm Sound

The system includes an `alarm.wav` file placeholder. You can:
- Use the provided alarm.wav (if included)
- Replace it with your own alarm sound file
- Download a free alarm sound from the internet

## File Structure

```
Driver-Drowsiness-Detection/
|
|--- main.py                           # Main detection system
|--- requirements.txt                  # Python dependencies
|--- alarm.wav                         # Alarm sound file
|--- shape_predictor_68_face_landmarks.dat  # Facial landmark model
|--- README.md                         # This file
```

## Usage

### Basic Usage

```bash
# Run the system with default settings
python main.py
```

### Advanced Usage with Custom Parameters

```bash
# Custom EAR threshold and frame count
python main.py --ear-threshold 0.3 --consecutive-frames 15

# Custom files
python main.py --shape-predictor custom_model.dat --alarm custom_alarm.wav
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--shape-predictor` | `shape_predictor_68_face_landmarks.dat` | Path to facial landmark model |
| `--alarm` | `alarm.wav` | Path to alarm sound file |
| `--ear-threshold` | `0.25` | EAR threshold for eye closure |
| `--consecutive-frames` | `20` | Frames needed to trigger alarm |

### Controls

- **ESC**: Exit the application
- **Close window**: Exit the application

## System Operation

1. **Initialization**: The system loads the facial landmark detector and initializes the camera
2. **Face Detection**: Continuously scans for faces in the video feed
3. **Landmark Detection**: When a face is found, detects 68 facial landmarks
4. **EAR Calculation**: Computes Eye Aspect Ratio for both eyes
5. **Drowsiness Detection**: Monitors EAR values over consecutive frames
6. **Alarm Trigger**: Sounds alarm if eyes remain closed for threshold duration

## Visual Indicators

### Status Colors
- **Green**: "AWAKE" - Normal operation, eyes open
- **Red**: "DROWSY" - Drowsiness detected, alarm active

### UI Elements
- **EAR Value**: Real-time eye aspect ratio (0.0 - 0.4 typical range)
- **Face Status**: Shows if face is currently detected
- **FPS Counter**: System performance indicator
- **Frame Counter**: Progress toward drowsiness threshold
- **Eye Contours**: Green outlines around detected eyes

## Troubleshooting

### Common Issues

**1. "Shape predictor file not found"**
```bash
# Ensure the file is in the correct location
ls shape_predictor_68_face_landmarks.dat

# Download if missing
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

**2. "Could not open camera"**
- Check if camera is properly connected
- Ensure no other applications are using the camera
- Try restarting the system
- Check camera permissions

**3. "Could not play alarm sound"**
- Verify alarm.wav exists in project directory
- Check system audio settings
- Try a different alarm sound file

**4. Poor detection accuracy**
- Ensure good lighting conditions
- Position camera at eye level
- Avoid extreme head angles
- Adjust EAR threshold if needed

### Performance Optimization

If you experience lag:
1. Reduce video resolution in code (`width=800` in main.py)
2. Close other applications
3. Ensure good lighting for faster face detection
4. Use a better quality webcam

## VS Code Setup

### Recommended Extensions

```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.pylint",
        "ms-vscode.cpptools",
        "redhat.vscode-yaml"
    ]
}
```

### VS Code Settings

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true
}
```

### Virtual Environment Setup (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Technical Details

### Algorithm Flow

1. **Capture Frame**: Get video frame from webcam
2. **Face Detection**: Use dlib HOG detector to find faces
3. **Landmark Detection**: Apply shape predictor to get 68 points
4. **Eye Extraction**: Isolate left and right eye landmarks
5. **EAR Calculation**: Compute eye aspect ratios
6. **Threshold Check**: Compare against EAR threshold
7. **Frame Counting**: Track consecutive low-EAR frames
8. **Alarm Trigger**: Activate alarm if threshold exceeded

### Key Parameters

- **EAR Threshold**: 0.25 (typical range 0.2-0.3)
- **Consecutive Frames**: 20 (approximately 1 second at 20 FPS)
- **Yawn Threshold**: 0.6 (mouth aspect ratio)
- **Video Resolution**: 800px width (auto-scaled)

## Project Structure Explanation

### main.py Components

- **DrowsinessDetector Class**: Main detection logic
- **eye_aspect_ratio()**: EAR calculation function
- **mouth_aspect_ratio()**: Yawning detection function
- **detect_drowsiness()**: Main processing pipeline
- **draw_ui_overlay()**: Visual interface elements
- **play_alarm()**: Audio alert system

### Dependencies

- **OpenCV**: Video capture and image processing
- **dlib**: Face detection and landmark prediction
- **scipy**: Distance calculations for EAR
- **imutils**: Image processing utilities
- **playsound**: Audio alarm playback

## Academic References

This implementation is based on established computer vision research:

1. **"Real-Time Eye Blink Detection using Facial Landmarks"** - Soukupová & Cech
2. **"Drowsy Driver Detection System"** - Various academic implementations
3. **dlib Library Documentation** - Davis King

## Future Enhancements

Potential improvements for academic projects:

1. **Multi-person detection** for passenger monitoring
2. **Machine learning integration** for personalized thresholds
3. **Mobile app deployment** using TensorFlow Lite
4. **Cloud-based alert system** for fleet management
5. **Historical data logging** for pattern analysis

## License

This project is provided for educational purposes. Please ensure compliance with your institution's academic integrity policies.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify all dependencies are properly installed
3. Ensure the shape predictor file is correctly placed
4. Test camera functionality with other applications

---

**Author**: College OS Mini Project  
**Version**: 1.0  
**Last Updated**: 2024
