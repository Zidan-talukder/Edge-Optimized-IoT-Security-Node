# Edge-Optimized Multi-Modal Security Node (IoT Vision)

## Executive Summary
This project demonstrates a resource-optimized, multi-modal computer vision pipeline designed for deployment on edge devices (Raspberry Pi / Nvidia Jetson Nano). To overcome the thermal and computational limits of continuous edge inferencing, the system employs a multi-state architecture. 

It utilizes lightweight background subtraction as a "wake trigger," ensuring heavy Local Binary Pattern Histograms (LBPH) and Deep Learning models are only activated upon motion detection. Unidentified anomalies are offloaded to a Cloud REST API for advanced tagging, establishing a highly efficient Edge-to-Cloud IoT pipeline.

## System Architecture

The pipeline operates in three distinct computational states:

1. **Idle State (Ultra-Low Power):**
   * Uses continuous feed monitoring via OpenCV.
   * Runs lightweight Gaussian Blurring and Absolute Difference (`cv2.absdiff`) thresholding to detect contour area changes.
   * Conserves CPU/GPU cycles until significant motion is detected.

2. **Active State (Edge Inferencing):**
   * Triggered by motion. Wakes up the Haar Cascade classifier for face detection.
   * Passes detected ROI to an LBPH Face Recognizer to determine user authorization.
   * Analyzes emotional state using a CPU-optimized 68-Landmark predictor.
   * Simultaneously runs HSV-based spatial tracking (`cv2.moments`) to identify pre-flagged colored anomalies (e.g., restricted items).

3. **Fallback State (Cloud Offloading):**
   * If an "Unknown" individual or restricted anomaly is detected, the local device halts heavy processing.
   * Captures the frame, encodes the payload, and transmits it via HTTP Requests to the Imagga Cloud API.
   * Parses the returning JSON to log high-level object tags.

## Technology Stack
* **Core Language:** Python 3.x
* **Computer Vision:** OpenCV (Haar Cascades, LBPH, Background Subtraction, Contour Tracking)
* **Deep Learning:** `facial_emotion_recognition` (CPU-optimized mode)
* **Cloud & Networking:** HTTP `requests`, JSON parsing, `urllib` (for IP camera stream decoding)
* **Target Hardware:** Raspberry Pi, Nvidia Jetson Nano

## Key Engineering Highlights
* **Dynamic Data Pipeline:** Engineered automated scripts to capture, resize, and label training datasets dynamically for the LBPH model.
* **Algorithmic Efficiency:** Selected LBPH over CNNs for facial recognition specifically to ensure real-time latency on constrained hardware.
* **IoT Integration:** Capable of decoding byte streams from remote wireless IP cameras rather than relying solely on local USB peripherals.