# Edge-Deployed Smart Surveillance & Emotion Tracking System with Cloud-API Fallback

## Executive Summary
I built this intelligent security system to run smoothly on small, power-efficient devices like a Raspberry Pi or Jetson Nano. A common challenge with edge AI is that running complex models continuously causes devices to overheat and lag. 

To solve this, I designed a smart, multi-state system. It stays in a low-power "idle" mode, keeping a constant watch using simple motion detection. When movement is detected, the system wakes up to perform more intensive tasks locally, such as face recognition, emotion tracking, and object tracking. If the system spots an unknown person or a potential security concern, it automatically offloads the analysis to a Cloud API for deeper investigation.

## How the System Works
The pipeline operates in three clear states to balance performance and power:

* **Idle State (Ultra-Low Power):** The system monitors the camera feed using basic OpenCV motion detection. It uses Gaussian blurring to ignore minor noise and only triggers the "Active State" when it detects significant movement.
* **Active State (Edge Inferencing):** Once triggered, the system uses Haar Cascades to detect faces. It then runs an LBPH model to check if the person is authorized and uses a 68-landmark predictor to track their emotional state. It also performs color-based tracking to spot specific items, like a restricted bag or tool.
* **Fallback State (Cloud Offloading):** Edge devices have limits. If the system encounters an "Unknown" person, it stops trying to guess locally. Instead, it snaps a photo and sends it to the Imagga Cloud API to get a list of semantic tags (like "backpack" or "electronics"), saving us from having to run a massive database on a tiny computer.

## Hardware Acceleration (Intel Movidius NCS2)
A raw Raspberry Pi CPU struggles when running computer vision and deep learning models at the same time, often dropping to an unusable frame rate (below 5 FPS).

To fix this, I integrated the **Intel Neural Compute Stick 2 (NCS2)**. By routing the heavy math through the NCS2 using the OpenVINO toolkit, I offloaded the deep learning workload from the CPU to the stick’s dedicated VPU. This allows the system to run in real-time, keeping the Pi cool and responsive even during active tracking.

## Technology Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (Background subtraction, Face detection, Color tracking)
* **AI/ML:** LBPH (Face Recognition), 68-Landmark predictor (Emotion tracking)
* **Cloud:** Imagga API (for cloud-based object tagging)
* **Hardware:** Raspberry Pi / Jetson Nano + Intel Movidius NCS2 (VPU Accelerator)

## Engineering Highlights
* **Resource Optimization:** Developed a "wake-on-motion" architecture to ensure the device only consumes power when necessary.
* **Dynamic Dataset Pipeline:** Created automated scripts to handle face capturing, resizing, and dataset labeling, making it easy to add new users to the system.
* **Hybrid Deployment:** Combined local, high-speed edge processing with cloud-based fallback for the best of both worlds—speed and accuracy.
