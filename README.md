# ECG Signal Analysis, Health Insights Dashboard & Image Processing GUI

This is my project for my first semester in the International Biometrics and Intelligent Vision Master's programme.  
This project is a unified application for multi - modal medical data analysis. It enables healthcare professionals to analyze patient health data , which includes :
* Managing patient data within a database.
* Processing and extracting insights from medical images and videos.
* Analyzing biomedical signals , such as ECG , to extract heartbeat information .

📌 Project Overview  
This project consists of:

* ECG Signal Analysis (PyQt5) – A GUI for analyzing ECG signals, detecting R-peaks, and computing FFT and periodogram.  
* Health Insights Dashboard (Plotly Dash) – Interactive visualizations for health metrics (stress, smoking, alcohol, diabetes).  
* Image & Video Processing GUI (PyQt5 + OpenCV) – A tool for real-time image and video processing, supporting medical image preprocessing, segmentation, and analysis.

# 🚀 Features

## 1️⃣ ECG Signal Analysis

📡 Load and process ECG signals from .dat files.  
🎚️ Butterworth filtering for noise reduction done automatically.  
🔍 R-peak detection and heart rate variability (HRV) analysis.  
📊 Fourier Transform (FFT) and Periodogram analysis.  
📌 Save and download results in an excel or csv file.

## 2️⃣ Health Insights Dashboard (Plotly Dash)
📊 Dashboard displaying the overall metrics of the patient data  
📊 Interactive bar charts & pie charts for stress, smoking, alcohol, and diabetes.   

## 3️⃣ Image & Video Processing GUI (PyQt6 + OpenCV)
### 🖼️ Image Processing
📂 Load images (JPG, PNG, BMP).  
🌑 Convert to Grayscale.  
🎭 Thresholding (Binary segmentation).  
📉 Blurring (Adjustable Gaussian Blur).  
🚫 Denoising (Noise reduction).  
🔍 Edge Detection (Canny method).  
✨ Sharpening (Enhance details).  
🔄 Morphological Operations (Erosion, Dilation, Fill Holes, Clear Borders, Remove Small Objects).  
📐 Extract Region Properties (Surface Area, Perimeter, Eccentricity).  
🔄 Undo/Redo Operations.  
💾 Save Processed Image.  
📊 Export Region Properties (CSV, Excel, JSON).

### 🎥 Video Processing
🎞️ Load videos (MP4, AVI, MOV).  
🌑 Convert to Grayscale.  
🔍 Edge Detection.  
🎭 Thresholding.  
🌈 Channel Visualization (Red, Green, Blue).  
▶️ Play, Pause, Stop Video.  
💾 Save Processed Video.

🛠️ Technologies Used
* Python 3.8+ (backend)  
* PyQt5 (GUI framework)
* Plotly Dash (dashboard visualization)
* OpenCV (image & video processing)
* NumPy, SciPy, Pandas (data processing & signal analysis)
* WFDB (ECG signal reading)
* MySQL (database storage)


📦 ECG-Health-Image-Analysis  
│── 📂 data/                  # ECG & medical image datasets  
│── 📂 database/              # SQL scripts and patient database  
│── 📂 gui/                   # PyQt ECG and Image Processing UI  
│── 📂 plots/                 # Plotly Dash visualizations  
│── 📜 main.py                # Entry point for ECG Analysis GUI  
│── 📜 dashboard.py           # Plotly Dash dashboard  
│── 📜 image_processing.py     # Image Processing GUI  
│── 📜 requirements.txt       # Required dependencies  
│── 📜 README.md              # Project documentation  

