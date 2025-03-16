import cv2
import mediapipe as mp
import numpy as np
import math
import time

class PianoHandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Detect both hands for piano playing
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        
        # Define key finger indices for piano-specific measurements
        # For MediaPipe, fingers are indexed: thumb (0-4), index (5-8), middle (9-12), ring (13-16), pinky (17-20)
        self.finger_tips = [4, 8, 12, 16, 20]  # Fingertips
        self.finger_pips = [3, 7, 11, 15, 19]  # Middle joints
        self.finger_mcps = [2, 6, 10, 14, 18]  # Base joints
        
        # Define ideal angles for piano playing posture (can be customized)
        # These are example values - should be calibrated for each player
        self.ideal_finger_curvature = {
            'thumb': (55, 65),  # (min, max) in degrees
            'index': (160, 175),
            'middle': (160, 175),
            'ring': (155, 170),
            'pinky': (150, 165)
        }
        
        # Font for display
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Metrics for tracking
        self.posture_history = []
        self.feedback_messages = []
        
        # Colors
        self.GREEN = (0, 255, 0)    # Good posture
        self.YELLOW = (0, 255, 255) # Warning
        self.RED = (0, 0, 255)      # Bad posture
        
    def calculate_angle(self, a, b, c):
        """
        Calculate the angle between three points (at point b)
        Points are expected as (x, y) tuples
        Returns angle in degrees
        """
        # Create vectors ba and bc
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        
        # Calculate dot product
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        
        # Calculate magnitudes
        magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        
        # Calculate angle using dot product formula
        # Adding small epsilon to avoid division by zero
        cosine_angle = dot_product / (magnitude_ba * magnitude_bc + 1e-6)
        # Clamp value to valid range for arccos
        cosine_angle = min(1.0, max(-1.0, cosine_angle))
        
        angle_rad = math.acos(cosine_angle)
        angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_hand_arch(self, landmarks, img_shape):
        """Calculate the arch of the hand across the MCP joints (knuckles)"""
        h, w, _ = img_shape
        mcp_points = []
        
        # Extract MCP joints (knuckles)
        for idx in self.finger_mcps:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            mcp_points.append((x, y))
        
        # Calculate arch height relative to a baseline
        if len(mcp_points) < 4:  # Need at least index to pinky MCPs
            return 0
            
        # Create a baseline from index to pinky MCP
        baseline_start = mcp_points[1]  # index MCP
        baseline_end = mcp_points[4]    # pinky MCP
        
        # Find maximum vertical distance from any MCP to the baseline
        max_distance = 0
        for point in mcp_points[1:4]:  # Check middle and ring knuckles
            # Calculate perpendicular distance to baseline
            # Using the line equation method
            line_length = math.sqrt((baseline_end[0] - baseline_start[0])**2 + 
                                   (baseline_end[1] - baseline_start[1])**2)
            if line_length == 0:
                continue
                
            distance = abs((baseline_end[1] - baseline_start[1]) * point[0] - 
                          (baseline_end[0] - baseline_start[0]) * point[1] + 
                          baseline_end[0] * baseline_start[1] - 
                          baseline_end[1] * baseline_start[0]) / line_length
            
            max_distance = max(max_distance, distance)
            
        return max_distance
    
    def analyze_finger_curvature(self, landmarks, img_shape):
        """
        Analyze curvature of each finger for piano playing
        Returns a dict of angles and feedback
        """
        h, w, _ = img_shape
        finger_angles = {}
        feedback = {}
        
        # Calculate angle for each finger (from MCP to PIP to tip)
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, name in enumerate(finger_names):
            # Get the three points for angle calculation
            mcp = (int(landmarks[self.finger_mcps[i]].x * w), 
                   int(landmarks[self.finger_mcps[i]].y * h))
            pip = (int(landmarks[self.finger_pips[i]].x * w), 
                   int(landmarks[self.finger_pips[i]].y * h))
            tip = (int(landmarks[self.finger_tips[i]].x * w), 
                   int(landmarks[self.finger_tips[i]].y * h))
            
            # Calculate angle
            angle = self.calculate_angle(mcp, pip, tip)
            finger_angles[name] = angle
            
            # Compare with ideal range
            ideal_min, ideal_max = self.ideal_finger_curvature[name]
            if angle < ideal_min:
                feedback[name] = (f"{name.capitalize()} too curved ({int(angle)}°)", self.YELLOW)
            elif angle > ideal_max:
                feedback[name] = (f"{name.capitalize()} too flat ({int(angle)}°)", self.YELLOW)
            else:
                feedback[name] = (f"{name.capitalize()} good ({int(angle)}°)", self.GREEN)
                
        return finger_angles, feedback
    
    def process_frame(self, frame):
        """
        Process a single video frame
        Returns the processed frame with annotations
        """
        # Flip the image horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)
        
        # Get image dimensions
        h, w, _ = frame.shape
        
        # If hands detected
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine if left or right hand
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
                # Calculate hand arch
                arch_height = self.calculate_hand_arch(hand_landmarks.landmark, frame.shape)
                
                # Analyze finger curvatures
                angles, feedback = self.analyze_finger_curvature(hand_landmarks.landmark, frame.shape)
                
                # Determine overall posture quality
                bad_positions = sum(1 for msg, color in feedback.values() if color != self.GREEN)
                overall_status = "GOOD" if bad_positions == 0 else "ADJUST"
                overall_color = self.GREEN if bad_positions == 0 else self.YELLOW
                
                # Position text based on which hand it is
                if handedness == "Right":
                    x_pos = w - 300  # Right side of screen
                else:
                    x_pos = 10       # Left side of screen
                
                # Display hand information (more concise)
                y_pos = 30
                cv2.putText(frame, f"{handedness}: {overall_status}", 
                           (x_pos, y_pos), self.font, 0.7, overall_color, 2)
                
                # Display arch information
                y_pos += 30
                arch_status = "Good"
                arch_color = self.GREEN
                if arch_height < 10:
                    arch_status = "Too flat"
                    arch_color = self.YELLOW
                elif arch_height > 40:
                    arch_status = "Too arched"
                    arch_color = self.YELLOW
                    
                cv2.putText(frame, f"Arch: {arch_status}", 
                           (x_pos, y_pos), self.font, 0.6, arch_color, 2)
                
                # Display finger feedback (more concise)
                for i, (finger, (message, color)) in enumerate(feedback.items()):
                    y_pos += 25
                    # Extract only the essential information
                    concise_msg = message.split(" ")[0]  # Just the finger name
                    if "too curved" in message.lower():
                        concise_msg += ": Curved"
                    elif "too flat" in message.lower():
                        concise_msg += ": Flat"
                    else:
                        concise_msg += ": Good"
                    
                    cv2.putText(frame, concise_msg, (x_pos, y_pos), self.font, 0.5, color, 2)
                
                # Store metrics for this frame
                self.posture_history.append({
                    'timestamp': time.time(),
                    'hand': handedness,
                    'angles': angles,
                    'arch': arch_height,
                    'quality': overall_status
                })
        else:
            # No hands detected
            cv2.putText(frame, "No hands detected", 
                       (w//2 - 100, 30), self.font, 0.7, self.RED, 2)
        
        return frame
    
    def run(self, camera_id=0):
        """
        Run the hand gesture recognizer using webcam input
        """
        cap = cv2.VideoCapture(camera_id)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam")
                break
                
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow("Piano Hand Gesture Recognition", processed_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def calibrate(self):
        """
        Run a calibration routine to set ideal angles for the specific user
        """
        print("Calibration mode: Position your hand in ideal piano playing position")
        print("Hold steady for 3 seconds to capture reference angles")
        
        cap = cv2.VideoCapture(0)
        calibration_frames = []
        calibration_start = None
        calibrating = False
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if not calibrating:
                    calibrating = True
                    calibration_start = time.time()
                    cv2.putText(frame, "Hold position...", (10, 30), self.font, 1, (0,255,0), 2)
                else:
                    elapsed = time.time() - calibration_start
                    cv2.putText(frame, f"Calibrating: {3-int(elapsed)}s", (10, 30), self.font, 1, (0,255,0), 2)
                    
                    # Store data for calibration
                    if elapsed < 3:
                        calibration_frames.append(results.multi_hand_landmarks[0])
                    else:
                        # Calculate average angles
                        self._process_calibration(calibration_frames, frame.shape)
                        cv2.putText(frame, "Calibration complete!", (10, 70), self.font, 1, (0,255,0), 2)
                        time.sleep(2)
                        break
            else:
                calibrating = False
                cv2.putText(frame, "Position hand for calibration", (10, 30), self.font, 1, (0,0,255), 2)
                
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def _process_calibration(self, calibration_frames, img_shape):
        """Process calibration data to set ideal angles"""
        h, w, _ = img_shape
        finger_angles = {name: [] for name in ['thumb', 'index', 'middle', 'ring', 'pinky']}
        
        # Calculate angles for each frame
        for landmarks in calibration_frames:
            for i, name in enumerate(['thumb', 'index', 'middle', 'ring', 'pinky']):
                mcp = (int(landmarks.landmark[self.finger_mcps[i]].x * w), 
                       int(landmarks.landmark[self.finger_mcps[i]].y * h))
                pip = (int(landmarks.landmark[self.finger_pips[i]].x * w), 
                       int(landmarks.landmark[self.finger_pips[i]].y * h))
                tip = (int(landmarks.landmark[self.finger_tips[i]].x * w), 
                       int(landmarks.landmark[self.finger_tips[i]].y * h))
                
                angle = self.calculate_angle(mcp, pip, tip)
                finger_angles[name].append(angle)
        
        # Calculate average angles and set as ideal with a small tolerance
        for name in finger_angles:
            if finger_angles[name]:
                avg_angle = sum(finger_angles[name]) / len(finger_angles[name])
                self.ideal_finger_curvature[name] = (avg_angle - 10, avg_angle + 10)
                
        print("Calibration complete. New ideal angles:")
        for name, (min_angle, max_angle) in self.ideal_finger_curvature.items():
            print(f"{name}: {min_angle:.1f}° - {max_angle:.1f}°")

if __name__ == "__main__":
    recognizer = PianoHandGestureRecognizer()
    
    # Uncomment to run calibration first
    recognizer.calibrate()
    
    # Run the main detection system
    recognizer.run() 