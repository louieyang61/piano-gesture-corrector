# Piano Hand Gesture Recognition System

A computer vision-based system to help piano players improve their hand posture and technique through real-time feedback.

## Overview

This project uses OpenCV and MediaPipe to track hand movements and provide feedback on proper piano playing posture. It analyzes:

- Finger curvature and angles
- Hand arch shape
- Overall hand positioning

The system provides real-time visual feedback directly on screen, helping pianists maintain proper technique during practice sessions.

## Features

- **Real-time hand landmark detection**: Identifies 21 key points on each hand
- **Angle and curve detection**: Measures the angles of finger joints and hand curvature
- **Customizable posture guidance**: Can be calibrated to individual hand sizes and piano techniques
- **Supports both hands**: Detects and analyzes both left and right hands simultaneously
- **Visual feedback**: Color-coded on-screen feedback about posture quality

## Requirements

- Python 3.7+
- Webcam
- Libraries listed in `requirements.txt`

## Installation

1. Clone this repository or download the files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the main program:

```bash
python piano_gesture_recognition.py
```

2. Position your hands in front of the webcam as if you're playing piano
3. The system will display real-time feedback on your hand posture
4. Press 'q' to exit the program

### Calibration

Before regular use, you may want to calibrate the system to your specific hand dimensions and preferred playing technique:

1. Uncomment the calibration line in the main code:
```python
# Uncomment to run calibration first
recognizer.calibrate()
```

2. Run the program and follow the on-screen instructions
3. Hold your hands in the ideal piano playing position for 3 seconds
4. The system will record and save your optimal finger angles as the reference

## Customization

You can modify the ideal angle ranges in the code to match specific piano techniques or to accommodate different hand types. The default values are set for a standard relaxed hand position used in most classical piano techniques.

## How It Works

1. **Image Capture**: The webcam captures video frames of your hands
2. **Hand Detection**: MediaPipe's hand tracking model identifies key landmarks on your hand
3. **Feature Extraction**: The system calculates:
   - Angles between finger joints
   - Curvature of the hand arch
   - Position of each finger
4. **Analysis**: Compares current measurements against ideal ranges
5. **Feedback**: Provides visual cues about what needs correction

## Extending the Project

You could extend this project by:

- Adding data logging to track progress over time
- Connecting to a piano via MIDI to correlate technique with actual playing
- Integrating with Arduino for haptic feedback (vibration when posture needs correction)
- Adding specific exercise modes for different piano techniques

## Troubleshooting

- **Poor hand detection**: Ensure good lighting and position your hands clearly in view of the camera
- **Inaccurate measurements**: Run the calibration process to customize the system to your hands
- **High CPU usage**: Lower your webcam resolution or reduce the processing frame rate

## License

This project is provided for educational purposes. Feel free to modify and extend it for your personal use. 