# **SwimVision: Automated Swimmer Tracking and Activity Classification in Competitive Swimming**

## **Authors**
- Philip Paškov

---

## **Motivation and Goal of the Project**
Competitive swimming analysis is a complex and time-sensitive process often performed manually by coaches and analysts. This project leverages computer vision and machine learning techniques to automate swimmer tracking, activity classification, and speed estimation, enabling real-time and precise analysis.  

The primary goals of the project are:
- To enhance coaching efficiency by reducing manual workload.
- To provide swimmers with actionable insights to improve their performance.
- To streamline event management through automated performance analysis tools.

---

## **Guide to Repository Contents**

### **Main Files and Directories**
- **`models/`**: Contains pre-trained and trained model files.
  - **`yolo/`**: YOLO models for swimmer detection.
  - **`mobilenetv2/`**: MobileNetV2 model for activity classification.
  - **`yolov8_pose/`**: Keypoint detection model for pool mapping.
- **`src/`**: Source code for the project.
  - **`tracker.py`**: Code for swimmer tracking and classification.
  - **`pool_mapper.py`**: Logic for pool keypoint detection and mapping.
  - **`batches_main.py`**: Main script for processing videos in batches.
- **`notebooks/`**: Jupyter notebooks for exploratory data analysis and testing models.
- **`outputs/`**: Includes results such as annotated videos and 2D pool visualizations.
- **`README.md`**: This file, explaining the project, structure, and replication steps.

### **Poster Presentation**
The **poster** and all related visual materials are available in the `poster/` directory.  

---

## **Replication Guide**

### **Step 1: Setting Up the Environment**
1. Clone this repository:  
   ```bash
   git clone https://github.com/[username]/swimvision.git
   cd swimvision
   ```
2. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```


### **Step 2: Running the Pipeline**
To process a swimming video, use the main script:  
```bash
python batches_main.py
```

---

## **Code Explanations**

- **`batches_main.py`**: Main script for video processing, combining swimmer detection, classification, and pool mapping into a unified pipeline.  
- **`pool_mapper.py`**: Detects keypoints in the pool for homography and maps swimmer positions onto a 2D pool layout.  
- **`tracker.py`**: Tracks swimmers across frames, performs activity classification, and calculates speed.  
- **`utils/`**: Contains helper functions for video reading, annotation drawing, and more.

---

## **Additional Information**

Unfortunately the models are too big to upload in github.
