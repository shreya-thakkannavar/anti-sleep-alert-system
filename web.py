import cv2
import dlib
import numpy as np
import pygame
from scipy.spatial import distance as dist

class DrowsinessDetector:
    def __init__(self):
        # Initialize face and landmark detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("C:/Users/Asus/Shreya/vscode/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Initialize alarm
        pygame.mixer.init()
        self.alarm_sound = pygame.mixer.Sound("C:/Users/Asus/Shreya/vscode/alarm.wav")
        
        # Drowsiness parameters
        self.EYE_AR_THRESH = 0.25  # Eye aspect ratio threshold
        self.EYE_AR_CONSEC_FRAMES = 48 # Number of consecutive frames for drowsiness
        self.HEAD_THRESH = 90 # Head tilt threshold in degrees
        
        # Tracking variables
        self.COUNTER = 0
        self.ALARM_ON = False
    
    def eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio to detect blinks."""
        # Vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_head_pose_angle(self, shape):
        """Estimate head tilt angle."""
        nose_bridge = shape[27:31]  # Nose bridge landmarks
        nose_tip = shape[30]  # Nose tip landmark
        
        # Calculate the average position of the nose bridge points
        nose_bridge_mean_x = np.mean([p[0] for p in nose_bridge])
        nose_bridge_mean_y = np.mean([p[1] for p in nose_bridge])
        
        # Calculate angle between the mean of the nose bridge and the nose tip
        angle = np.arctan2(
            nose_tip[1] - nose_bridge_mean_y,
            nose_tip[0] - nose_bridge_mean_x
        ) * 180 / np.pi  # Convert to degrees
    
        return abs(angle)

    
    def detect_drowsiness(self):
        """Main drowsiness detection loop."""
        while True:
            # Read frame from webcam
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray, 0)
            
            for face in faces:
                # Detect facial landmarks
                shape = self.predictor(gray, face)
                shape = np.array([(p.x, p.y) for p in shape.parts()])
                
                # Extract left and right eye regions
                left_eye = shape[42:48]
                right_eye = shape[36:42]
                
                # Calculate eye aspect ratios
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                
                # Average eye aspect ratio
                ear = (left_ear + right_ear) / 2.0
                
                # Check head angle
                head_angle = self.get_head_pose_angle(shape)
                
                # Drowsiness detection logic
                if ear < self.EYE_AR_THRESH or head_angle > self.HEAD_THRESH:
                    self.COUNTER += 1
                    
                    # If eyes closed for too long or head tilted
                    if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                        if not self.ALARM_ON:
                            self.ALARM_ON = True
                            self.alarm_sound.play(-1)  # Continuous play
                        
                        # Draw alert on frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.COUNTER = 0
                    if self.ALARM_ON:
                        self.ALARM_ON = False
                        self.alarm_sound.stop()
                
                # Visualize eye and head metrics
                cv2.putText(frame, f"EAR: {ear:.2f} (Thresh: {self.EYE_AR_THRESH})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Head Angle: {head_angle:.2f} (Thresh: {self.HEAD_THRESH})", (10, 80),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Drowsiness Detector", frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Start the drowsiness detection."""
        try:
            self.detect_drowsiness()
        except Exception as e:
            print(f"Error: {e}")

# Main execution
if __name__ == "__main__":
    # Prerequisites:
    # 1. Install required libraries:
    #    pip install opencv-python dlib numpy pygame scipy
    # 2. Download facial landmark predictor:
    #    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # 3. Prepare an alarm sound file (alarm.wav)
    
    detector = DrowsinessDetector() 
    detector.run()