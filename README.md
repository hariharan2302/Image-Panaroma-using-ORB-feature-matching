# 🖼️ Computer Vision and Image Processing Mini Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Project Overview

This mini project implements two fundamental computer vision tasks as part of **CSE 573 Computer Vision and Image Processing** coursework:

1. **🔍 Face Detection** - Automated face detection using Haar Cascade classifiers
2. **🌄 Image Panorama Stitching** - Creating panoramic images from multiple overlapping photographs

## 🎯 Features

### Task 1: Face Detection
- ✅ Robust face detection using OpenCV's Haar Cascade classifiers
- ✅ Batch processing of multiple images
- ✅ JSON output format for detection results
- ✅ Configurable detection parameters for accuracy optimization
- ✅ F-beta score evaluation for performance metrics

### Task 2: Image Panorama Stitching
- ✅ Automatic feature detection using ORB (Oriented FAST and Rotated BRIEF)
- ✅ Intelligent overlap detection (minimum 20% overlap requirement)
- ✅ Homography computation for perspective transformation
- ✅ Multi-image stitching with irregular output shapes
- ✅ Robust matching algorithm with customizable thresholds

## 🏗️ Project Structure

```
📦 CVIP-Mini-Project
├── 📁 ComputeFBeta/
│   └── 📄 ComputeFBeta.py          # F-beta score computation
├── 📁 images_panaroma/
│   ├── 🖼️ t2_1.png                # Sample panorama images
│   ├── 🖼️ t2_2.png
│   ├── 🖼️ t2_3.png
│   └── 🖼️ t2_4.png
├── 📄 task1.py                     # Face detection implementation
├── 📄 Image_panorama.py            # Panorama stitching implementation
├── 📄 utils.py                     # Utility functions and helpers
├── 📄 pack_submission.sh           # Automated submission script
├── 📄 LICENSE                      # Project license
└── 📄 README.md                    # Project documentation
```

## 🔧 Technical Implementation

### Face Detection Algorithm

The face detection system employs a **Haar Cascade Classifier** approach:

```python
# Core detection pipeline
my_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
my_grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
face_detect = my_cascade_face.detectMultiScale(my_grey_img, 1.3, 3, minSize=(13, 13))
```

**Key Parameters:**
- **Scale Factor**: 1.3 (30% size reduction per scale)
- **Min Neighbors**: 3 (minimum detections required)
- **Min Size**: 13×13 pixels (smallest detectable face)

### Panorama Stitching Pipeline

The panorama creation follows a sophisticated multi-step process:

| Step | Process | Algorithm |
|------|---------|-----------|
| 1️⃣ | **Feature Detection** | ORB (Oriented FAST + Rotated BRIEF) |
| 2️⃣ | **Feature Matching** | Custom distance-based matching |
| 3️⃣ | **Overlap Analysis** | Geometric overlap computation (≥20%) |
| 4️⃣ | **Homography Estimation** | RANSAC-based robust estimation |
| 5️⃣ | **Image Warping** | Perspective transformation |
| 6️⃣ | **Blending** | Sequential image composition |

#### 🔍 Feature Detection Details

```python
# ORB feature detector configuration
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray_img, None)
```

**Why ORB?**
- ⚡ **Speed**: Faster than SIFT/SURF
- 🎯 **Accuracy**: Robust to rotation and scale changes
- 💰 **Cost**: Patent-free alternative to SIFT/SURF

#### 📐 Overlap Detection Algorithm

The system determines image overlap using geometric analysis:

```python
def compute_overlap_percentage(image1, image2, homography):
    # Transform corner points using homography
    pts2 = cv2.perspectiveTransform(pts1, homography)
    
    # Calculate overlap ratio
    overlap_percentage = min(area1, area2) / max(area1, area2) * 100
    return 1 if overlap_percentage >= 20 else 0
```

## 🚀 Usage Instructions

### Prerequisites

```bash
pip install opencv-python numpy matplotlib argparse
```

### Running Face Detection

```bash
# Validate on validation dataset
python task1.py --input_path data/validation_folder/images --output ./result_task1_val.json

# Process test dataset
python task1.py --input_path data/test_folder/images --output ./result_task1.json
```

### Running Panorama Stitching

```bash
python Image_panorama.py --input_path data/images_panaroma --output_overlap ./task2_overlap.txt --output_panaroma ./task2_result.png
```

### Automated Submission

```bash
chmod +x pack_submission.sh
./pack_submission.sh your_ubit_name
```

## 📊 Performance Metrics

### Face Detection Evaluation

The system uses **F-beta score** for performance evaluation:

```
F-β = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall)
```

Where:
- **Precision** = True Positives / (True Positives + False Positives)
- **Recall** = True Positives / (True Positives + False Negatives)
- **β = 1** for F1-score (equal weight to precision and recall)

### Panorama Quality Metrics

| Metric | Requirement | Implementation |
|--------|-------------|----------------|
| **Minimum Overlap** | ≥20% | Geometric area calculation |
| **Feature Matches** | ≥4 points | RANSAC homography estimation |
| **Transformation** | 2D Planar | Perspective transformation matrix |

## 🔬 Algorithm Analysis

### Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| **Face Detection** | O(n × m × s) | O(1) |
| **Feature Detection** | O(n × m) | O(k) |
| **Feature Matching** | O(k₁ × k₂) | O(k₁ + k₂) |
| **Homography** | O(n³) | O(n²) |

Where:
- n, m = image dimensions
- s = number of scales
- k = number of keypoints

### Robustness Features

- 🛡️ **RANSAC**: Outlier rejection in homography estimation
- 🔄 **Multi-scale**: Detection at various image scales
- 📏 **Adaptive Thresholding**: Dynamic parameter adjustment
- 🎯 **Minimum Size Filtering**: Noise reduction

## 🎓 Educational Context

This project demonstrates practical applications of:

- **Computer Vision Fundamentals**: Feature detection, matching, and geometric transformations
- **Machine Learning**: Cascade classifiers and pattern recognition
- **Image Processing**: Geometric transformations and image composition
- **Algorithm Design**: Robust estimation and optimization techniques

## 📁 Sample Results

### Face Detection Output Format
```json
{
    "image1.jpg": [
        [x, y, width, height],
        [x2, y2, width2, height2]
    ],
    "image2.jpg": [
        [x, y, width, height]
    ]
}
```

### Panorama Overlap Matrix
```
[[0, 1, 0, 0],
 [1, 0, 1, 0],
 [0, 1, 0, 1],
 [0, 0, 1, 0]]
```

## 🤝 Contributing

This project follows academic integrity guidelines. For educational purposes:

1. 🔍 Study the implementation approaches
2. 📚 Understand the underlying algorithms
3. 🧪 Experiment with parameter tuning
4. 📊 Analyze performance characteristics

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**This project is created for educational purposes as part of CSE 573 coursework.**
