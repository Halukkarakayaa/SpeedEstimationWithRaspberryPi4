# Real-Time Vehicle Speed Estimation using Raspberry Pi 4 & YOLOv8

This repository contains the source code, implementation details, and test results for my **Graduation Project**. The project focuses on developing a Computer Vision-based system using **YOLOv8** to detect vehicles and estimate their real-time speed.

## Project Overview

The primary goal of this project is to create a dynamic speed estimation system capable of running on both constrained embedded systems (Raspberry Pi 4) and standard Personal Computers. By utilizing the YOLOv8 object detection model and mapping pixel distance to real-world measurements over time, the system accurately estimates the speed of moving vehicles.

During development, we encountered and overcame hardware constraints—specifically the frame-rate limitations of standard USB web cameras when connected to a Raspberry Pi. To address this, the architecture was modularized to support various input methods and execution environments.

## Key Features

* **YOLOv8 Integration:** Fast and accurate vehicle detection using the YOLOv8 Nano (`yolov8n.pt`) model.
* **Cross-Environment Execution:** Separated codebase for optimal performance on both PC and Raspberry Pi.
* **Dual-Mode Processing:**
  * **Live Feed:** Processes real-time frames from a connected USB camera.
  * **Offline Processing:** Analyzes pre-recorded source videos.
* **Hardware Optimization:** Adjusted tracking parameters to mitigate USB camera bottleneck issues on the Raspberry Pi.

## Repository Structure

Here is an overview of the files included in this repository:

```text
├──  PC Execution Scripts
│   ├── project_camera_pc.py   # Runs live speed estimation via PC Webcam
│   └── project_video_pc.py    # Processes pre-recorded videos on a PC
│
├──  Raspberry Pi Execution Scripts
│   ├── project_camera.py      # Runs live speed estimation via USB Camera on Pi
│   └── project_video.py       # Processes pre-recorded videos on the Raspberry Pi
│
├──  Configuration & Models
│   ├── yolov8n.pt             # Pre-trained YOLOv8 Nano model weights
│   └── coordinates.txt        # Coordinates for Region of Interest (ROI) and distance mapping
│
├──  Source Videos
│   ├── video1.mp4             
│   ├── video2.mp4             
│   └── video3.mp4             
│
└──  Result Videos
    ├── result_video1.mp4      # Processed output of video1.mp4
    ├── result_video2.mp4      # Processed output of video2.mp4
    └── result_video3.mp4      # Processed output of video3.mp4
```

##  Installation & Setup

### Prerequisites

* **Hardware:** Raspberry Pi 4 Model B (or a standard PC), USB Web Camera.
* **Software Environment:** Python 3.x

### 1. Clone the Repository

```bash
git clone https://github.com/Halukkarakayaa/SpeedEstimationWithRaspberryPi4.git
cd SpeedEstimationWithRaspberryPi4
```

### 2. Install Dependencies

It is recommended to use a virtual environment. Install the required libraries via pip:

```bash
pip install opencv-python numpy ultralytics
```
*(Note: The `ultralytics` package is required to run the YOLOv8 model).*

##  How to Run

Depending on your environment and desired input, run the corresponding script. 

**To run on a PC using a pre-recorded test video:**
```bash
python project_video_pc.py
```

**To run on a PC using a live USB camera feed:**
```bash
python project_camera_pc.py
```

**To run on a Raspberry Pi (Video / Camera):**
```bash
python project_video.py
# OR
python project_camera.py
```

##  Results and Output

The system draws bounding boxes around detected vehicles and displays their estimated speed directly on the frame. 

You can review the effectiveness of the algorithm by checking the provided `.mp4` files. Each source video has a corresponding processed result video showing the estimated speeds in action:
* `video1.mp4` ➡️ `result_video1.mp4`
* `video2.mp4` ➡️ `result_video2.mp4`
* `video3.mp4` ➡️ `result_video3.mp4`

Despite the initial frame-rate drops experienced with the USB camera on the Raspberry Pi, the offline video processing and PC implementations demonstrate the core algorithm's high accuracy.

##  Author

**Haluk Karakaya**
* [LinkedIn](https://linkedin.com/in/halukkarakayaa)
* [GitHub](https://github.com/Halukkarakayaa)
