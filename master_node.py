"""
Edge-Optimized Multi-Modal Security Node (IoT Vision)
Author: MD. Zidan Talukder
Description: Master orchestrator that triggers heavy DL/Cloud models ONLY when local motion is detected.
"""

import cv2
import time
import requests
import json
import imutils
# Assuming your M5 script is accessible locally
from facial_emotion_recognition import EmotionRecognition 

class EdgeSecurityNode:
    def __init__(self):
        print("[INFO] Initializing Ultra-Low Power Edge Node...")
        self.cam = cv2.VideoCapture(0) # Or IP Camera URL
        time.sleep(2.0)
        
        # Load Local Models (M2, M4, M5)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.emotion_recognizer = EmotionRecognition(device='cpu')
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        # In production, load a pre-trained .yml here instead of training on the fly
        
        # Cloud API details (M6)
        self.imagga_url = "https://api.imagga.com/v2/tags"
        self.imagga_headers = {
            'accept': "application/json",
            'authorization': "Basic YOUR_API_KEY_HERE"
        }

    def log_to_cloud(self, image_path):
        """Fallback: Sends unknown anomalies to Cloud for heavy tagging (M6)"""
        print(f"[NETWORK] Offloading frame to Cloud API...")
        # Note: Imagga requires an image URL or a multipart file upload. 
        # This is a placeholder for the HTTP logic you built in M6.
        pass

    def run(self):
        firstFrame = None
        print("[INFO] System Idle. Monitoring for motion (M1)...")

        while True:
            _, frame = self.cam.read()
            if frame is None: break
            
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)

            # Stage 1: Ultra-Lightweight Motion Trigger (M1)
            if firstFrame is None:
                firstFrame = blurred
                continue

            frameDelta = cv2.absdiff(firstFrame, blurred)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = any(cv2.contourArea(c) > 500 for c in cnts)

            # Stage 2: Heavy Inferencing (Triggered only on motion)
            if motion_detected:
                cv2.putText(frame, "ACTIVE: MOTION DETECTED", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Face & Emotion Detection (M2, M4, M5)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # (Insert LBPH prediction logic here)
                    # (Insert Emotion recognition logic here)
                    
                    # If person is Unknown, trigger Cloud API (M6)
                    # self.log_to_cloud("temp_snapshot.jpg")

            cv2.imshow("IoT Security Node", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = EdgeSecurityNode()
    node.run()