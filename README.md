# Human Detection and Tracking Using YOLOv8 and OpenCV

## 📌 Project Overview

This project is a Python-based human detection and tracking system designed to detect people in video footage and track their movements in real time. It uses the YOLOv8 object detection model and OpenCV for video processing, while ByteTrack is used to maintain unique tracking IDs for detected individuals.

The project demonstrates how Artificial Intelligence and Computer Vision can be applied to surveillance and border area monitoring.

## 🎯 Objectives

- Detect humans in video footage using YOLOv8.
- Track detected individuals across video frames using ByteTrack.
- Display bounding boxes, tracking IDs, and detection confidence scores.
- Count detected individuals in video frames.
- Generate alert messages based on detection events.
- Explore the application of AI-powered video surveillance for border monitoring.

## 🛠️ Technologies Used

- **Python** — Main programming language.
- **YOLOv8n** — Lightweight object detection model.
- **OpenCV** — Video processing and visualization.
- **ByteTrack** — Multi-object tracking.
- **Jupyter Notebook** — Development and experimentation.

## ⚙️ Key Features

- Human detection in video footage.
- Persistent tracking IDs for detected individuals.
- Bounding boxes around detected people.
- Confidence scores for detections.
- Frame-by-frame detection and tracking.
- Detection counts and alert messages.

## 🚀 How It Works

1. A video file is loaded into the Python environment.
2. OpenCV processes the video frames.
3. YOLOv8 detects humans in each frame.
4. ByteTrack assigns and maintains tracking IDs.
5. Bounding boxes and detection information are displayed.
6. Detection counts and alert messages are generated.

## 💻 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

### 2. Install the required libraries

```bash
pip install ultralytics opencv-python jupyter
```

### 3. Run the project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the project's notebook and run the cells in sequence. Ensure the required video file and model dependencies are available.

## 🎓 Project Significance

This project explores the use of Artificial Intelligence and Computer Vision in security monitoring. The detection and tracking capabilities can serve as a foundation for more advanced surveillance systems, including border monitoring, restricted-area monitoring, and human movement analysis.

## ⚠️ Limitations

The current implementation is a simulation based on video footage. It does not directly control a physical drone or constitute a complete operational border surveillance system.

## 🔮 Future Improvements

- Develop a graphical user interface.
- Integrate live camera feeds.
- Improve detection and tracking performance.
- Add configurable alerts and event logging.
- Explore integration with a drone-based surveillance platform.

## 👨‍💻 Author

**Abdulrahman Idowu**
