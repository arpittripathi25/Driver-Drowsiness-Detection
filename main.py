"""
Driver Drowsiness Detection System
==================================
A real-time webcam-based system that detects driver drowsiness by monitoring eye movement.

Author: College OS Mini Project
Dependencies: OpenCV, dlib, scipy, imutils, playsound
 cd Driver-Drowsiness-Detection
 python main.py

"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import imutils
from playsound import playsound
import threading
import time
import os
import argparse

class DrowsinessDetector:
    """
    Main class for detecting driver drowsiness using facial landmarks and Eye Aspect Ratio (EAR)
    """
    
    def __init__(self, shape_predictor_path="shape_predictor_68_face_landmarks.dat", 
                 alarm_sound="alarm.wav", ear_threshold=0.25, consecutive_frames=40):
        """
        Initialize the drowsiness detector
        
        Args:
            shape_predictor_path: Path to dlib's facial landmark predictor
            alarm_sound: Path to alarm sound file
            ear_threshold: EAR threshold below which eyes are considered closed
            consecutive_frames: Number of consecutive frames with low EAR to trigger alarm
        """
        # Initialize dlib's face detector and facial landmark predictor
        print("[INFO] Loading facial landmark predictor...")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        
        # Thresholds and counters
        self.EAR_THRESHOLD = ear_threshold
        self.CONSECUTIVE_FRAMES = consecutive_frames
        self.COUNTER = 0
        self.eye_closed_start = None
        self.ALARM_ON = False
        
        # Alarm sound path
        self.alarm_sound = alarm_sound
        
        # FPS calculation variables
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Status tracking
        self.status = "AWAKE"
        self.face_detected = False
        self.current_ear = 0.0

        self.no_face_counter = 0
        self.NO_FACE_LIMIT = 120
        
        # Yawning detection variables
        self.yawn_threshold = 0.6  # Mouth aspect ratio threshold for yawning
        self.yawn_counter = 0
        self.yawn_consecutive_frames = 15
        
        print("[INFO] System initialized successfully!")
    
    def eye_aspect_ratio(self, eye):
        """
        Compute the Eye Aspect Ratio (EAR) for a given eye
        
        Args:
            eye: Array of 6 (x, y) coordinates representing eye landmarks
            
        Returns:
            EAR value for the eye
        """
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])  # Vertical distance 1
        B = dist.euclidean(eye[2], eye[4])  # Vertical distance 2
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """
        Compute the Mouth Aspect Ratio (MAR) for yawning detection
        
        Args:
            mouth: Array of mouth landmark coordinates
            
        Returns:
            MAR value for the mouth
        """
        # Compute vertical distances
        A = dist.euclidean(mouth[1], mouth[7])   # Top to bottom
        B = dist.euclidean(mouth[2], mouth[6])   # Top to bottom
        C = dist.euclidean(mouth[3], mouth[5])   # Top to bottom
        
        # Compute horizontal distance
        D = dist.euclidean(mouth[0], mouth[4])   # Left to right
        
        # Compute mouth aspect ratio
        mar = (A + B + C) / (3.0 * D)
        
        return mar
    
    def get_facial_landmarks(self, gray, rect):
        """
        Get facial landmarks for a detected face
        
        Args:
            gray: Grayscale image
            rect: Face bounding rectangle
            
        Returns:
            numpy array of (x, y) coordinates for facial landmarks
        """
        shape = self.predictor(gray, rect)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        return shape
    
    def extract_eye_landmarks(self, landmarks):
        """
        Extract left and right eye landmarks from facial landmarks
        
        Args:
            landmarks: Array of all facial landmarks
            
        Returns:
            Tuple of (left_eye_landmarks, right_eye_landmarks)
        """
        # Left eye indices (68-point model)
        (l_start, l_end) = (42, 48)
        left_eye = landmarks[l_start:l_end]
        
        # Right eye indices
        (r_start, r_end) = (36, 42)
        right_eye = landmarks[r_start:r_end]
        
        return left_eye, right_eye
    
    def extract_mouth_landmarks(self, landmarks):
        """
        Extract mouth landmarks for yawning detection
        
        Args:
            landmarks: Array of all facial landmarks
            
        Returns:
            Array of mouth landmarks
        """
        # Mouth indices (outer mouth)
        (m_start, m_end) = (48, 68)
        mouth = landmarks[m_start:m_end]
        
        # For MAR calculation, we need specific points
        mouth_points = np.array([
            landmarks[60],  # Bottom lip center
            landmarks[61],  # Bottom lip left
            landmarks[62],  # Bottom lip right
            landmarks[63],  # Bottom lip right
            landmarks[64],  # Bottom lip right
            landmarks[65],  # Bottom lip right
            landmarks[66],  # Bottom lip right
            landmarks[67],  # Bottom lip right
        ])
        
        return landmarks[m_start:m_end]
    
    def play_alarm(self):
       while self.ALARM_ON:
        if os.path.exists(self.alarm_sound):
            try:
                playsound(self.alarm_sound)
            except Exception as e:
                print(f"[WARNING] Could not play alarm sound: {e}")
                break
    def calculate_fps(self):
        """
        Calculate and update FPS counter
        """
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_eye_contours(self, frame, left_eye, right_eye):
        """
        Draw contours around the eyes
        
        Args:
            frame: Input frame
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
        """
        # Draw contours for left eye
        left_eye_hull = cv2.convexHull(left_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        
        # Draw contours for right eye
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
    
    def draw_ui_overlay(self, frame):
        """
        Draw UI overlay with status information
        
        Args:
            frame: Input frame to draw on
        """
        # Create semi-transparent overlay for UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status text color based on drowsiness
        status_color = (0, 255, 0) if self.status == "AWAKE" else (0, 0, 255)
        
        # Draw status information
        # cv2.putText(frame, f"Status: {self.status}", (20, 35), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(
             frame,
           "Status: DROWSY" if self.COUNTER >= 150 else "Status: AWAKE",
            (20, 35),
             cv2.FONT_HERSHEY_SIMPLEX,
             0.7,
             (0, 0, 255) if self.COUNTER >= 150 else (0, 255, 0),
              2
          )
        
        cv2.putText(frame, f"EAR: {self.current_ear:.3f}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        face_status = "Detected" if self.face_detected else "Not Detected"
        face_color = (0, 255, 0) if self.face_detected else (0, 0, 255)
        cv2.putText(frame, f"Face: {face_status}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)
        
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Frame Counter: {self.COUNTER}/{self.CONSECUTIVE_FRAMES}", 
                   (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press ESC to exit", (frame.shape[1] - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def detect_drowsiness(self, frame):
        shape = None
        """
        Main drowsiness detection function
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with overlays
        """
        # Resize frame and convert to grayscale
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=50)
        
        # Detect faces in the grayscale frame
        rects = self.detector(gray, 0)
        
        # Reset face detection status
        self.face_detected = len(rects) > 0
        
        # Loop over the face detections
        for rect in rects:
            # Determine the facial landmarks for the face region
             shape = self.get_facial_landmarks(gray, rect)
            
            # Extract left and right eye coordinates
             left_eye, right_eye = self.extract_eye_landmarks(shape)
            
            # Extract mouth landmarks for yawning detection
             mouth_landmarks = self.extract_mouth_landmarks(shape)
            
            # Compute the eye aspect ratio for both eyes
             left_ear = self.eye_aspect_ratio(left_eye)
             right_ear = self.eye_aspect_ratio(right_eye)
            
            # Average the eye aspect ratio for both eyes
             ear = (left_ear + right_ear) / 2.0
             self.current_ear = ear
            
            # Compute mouth aspect ratio for yawning detection
             mar = self.mouth_aspect_ratio(mouth_landmarks)
            
            # Draw contours around eyes
             self.draw_eye_contours(frame, left_eye, right_eye)
            
            # Check for drowsiness
            
             if ear < self.EAR_THRESHOLD:

               self.COUNTER += 1

               if self.COUNTER >= 150:

                 self.status = "DROWSY"
                #  print(self.status)

                 if not self.ALARM_ON:
                        self.ALARM_ON = True
                        threading.Thread(target=self.play_alarm, daemon=True).start()

             else:
              self.COUNTER = 0
             self.ALARM_ON = False
             self.status = "AWAKE"
            
            # Check for yawning (optional enhancement)
        # shape = self.get_facial_landmarks(gray, rect)
        # mouth_landmarks = self.extract_mouth_landmarks(shape)
        # mar = self.mouth_aspect_ratio(mouth_landmarks)
             if mar > self.yawn_threshold:
                self.yawn_counter += 1
                if self.yawn_counter >= self.yawn_consecutive_frames:
                    cv2.putText(frame, "YAWNING!", (10, frame.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
             else:
                self.yawn_counter = 0
            
            # Visualize the facial landmarks
        # for (x, y) in shape:
        #         cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        if self.face_detected and shape is not None:
          for (x, y) in shape:
           cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        # If no face detected, reset counters
        # if not self.face_detected:
        #     self.COUNTER = 0
        #     self.ALARM_ON = False
        #     # self.status = "AWAKE"
        #     self.current_ear = 0.0
        if not self.face_detected:
            self.no_face_counter += 1

            self.COUNTER = 0
            self.current_ear = 0.0

            if self.no_face_counter >= self.NO_FACE_LIMIT:
             self.status = "NO FACE"

             if not self.ALARM_ON:
               self.ALARM_ON = True
             threading.Thread(target=self.play_alarm, daemon=True).start()

        else:
         self.no_face_counter = 0
         self.ALARM_ON = False
        
        # Calculate FPS
        self.calculate_fps()
        
        # Draw UI overlay
        self.draw_ui_overlay(frame)
        
        return frame
    
    def run(self):
        """
        Main function to run the drowsiness detection system
        """
        print("[INFO] Starting video stream...")
        
        # Start the video stream
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("[ERROR] Could not open camera!")
            return
        
        print("[INFO] Camera opened successfully. Press ESC to exit.")
        
        try:
            while True:
                # Read a frame from the video stream
                ret, frame = cap.read()
                
                if not ret:
                    print("[ERROR] Failed to capture frame!")
                    break
                
                # Detect drowsiness
                frame = self.detect_drowsiness(frame)
                
                # Show the output frame
                cv2.imshow("Driver Drowsiness Detection System", frame)
                
                # Check for ESC key press
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    print("[INFO] Exiting...")
                    break
                    
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user. Exiting...")
            
        finally:
            # Cleanup
            print("[INFO] Cleaning up resources...")
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] System shutdown complete.")

def main():
    """
    Main function to handle command line arguments and start the system
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument("--shape-predictor", type=str, 
                       default="shape_predictor_68_face_landmarks.dat",
                       help="Path to facial landmark predictor")
    parser.add_argument("--alarm", type=str, default="alarm.wav",
                       help="Path to alarm sound file")
    parser.add_argument("--ear-threshold", type=float, default=0.25,
                       help="Eye aspect ratio threshold")
    parser.add_argument("--consecutive-frames", type=int, default=20,
                       help="Number of consecutive frames for drowsiness detection")
    
    args = parser.parse_args()
    
    # Check if shape predictor file exists
    if not os.path.exists(args.shape_predictor):
        print(f"[ERROR] Shape predictor file not found: {args.shape_predictor}")
        print("Please download shape_predictor_68_face_landmarks.dat from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    # Create and run the drowsiness detector
    detector = DrowsinessDetector(
        shape_predictor_path=args.shape_predictor,
        alarm_sound=args.alarm,
        ear_threshold=args.ear_threshold,
        consecutive_frames=args.consecutive_frames
    )
    
    detector.run()

if __name__ == "__main__":
    main()
