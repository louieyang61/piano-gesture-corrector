import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time

class PianoPostureDeepLearningClassifier:
    def __init__(self, model_path=None):
        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5)
        
        # Categories for classification
        self.categories = ['too_curved', 'good', 'too_flat']
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.model = None
            print("No model loaded. Use train_model() to create a new model.")
        
        # Font and colors for display
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.GREEN = (0, 255, 0)    # Good posture
        self.YELLOW = (0, 255, 255) # Too curved
        self.RED = (0, 0, 255)      # Too flat
        
    def preprocess_landmarks(self, landmarks):
        """
        Convert hand landmarks to a feature vector suitable for ML model input.
        Normalizes coordinates to be relative to the wrist position.
        Returns a 1D array of shape (42,) containing x,y coordinates of all landmarks
        """
        if landmarks is None:
            return None
            
        # Extract wrist position for normalization
        wrist_x = landmarks[0].x
        wrist_y = landmarks[0].y
        
        # Create feature vector: normalized x,y coordinates for each landmark
        features = []
        for lm in landmarks:
            # Normalize coordinates relative to wrist
            norm_x = lm.x - wrist_x
            norm_y = lm.y - wrist_y
            features.extend([norm_x, norm_y])
            
        return np.array(features)
    
    def collect_training_data(self, data_dir='hand_posture_data'):
        """
        Collect and save training data from webcam images.
        This function helps create a labeled dataset for training the classifier.
        """
        # Create directories if they don't exist
        for category in self.categories:
            os.makedirs(os.path.join(data_dir, category), exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        current_category_idx = 0
        current_category = self.categories[current_category_idx]
        sample_count = 0
        
        print(f"Starting data collection. Current category: {current_category}")
        print("Press 'c' to capture a sample")
        print("Press 'n' to switch to next category")
        print("Press 'q' to quit")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB and process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw instructions and current category
            cv2.putText(frame, f"Category: {current_category} | Samples: {sample_count}", 
                      (10, 30), self.font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture, 'n' for next category, 'q' to quit", 
                      (10, 60), self.font, 0.7, (0, 255, 0), 2)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
            
            cv2.imshow('Data Collection', frame)
            key = cv2.waitKey(1) & 0xFF
            
            # Capture current frame as a sample
            if key == ord('c') and results.multi_hand_landmarks:
                # Save feature vector
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    features = self.preprocess_landmarks(hand_landmarks.landmark)
                    sample_path = os.path.join(data_dir, current_category, 
                                              f"sample_{sample_count}_hand_{i}.pkl")
                    with open(sample_path, 'wb') as f:
                        pickle.dump(features, f)
                        
                    # Also save the image for visualization
                    img_path = os.path.join(data_dir, current_category, 
                                           f"sample_{sample_count}_hand_{i}.jpg")
                    cv2.imwrite(img_path, frame)
                    
                sample_count += 1
                print(f"Saved sample {sample_count} for {current_category}")
            
            # Switch to next category
            elif key == ord('n'):
                sample_count = 0
                current_category_idx = (current_category_idx + 1) % len(self.categories)
                current_category = self.categories[current_category_idx]
                print(f"Switched to category: {current_category}")
            
            # Quit data collection
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Data collection complete")
    
    def prepare_dataset(self, data_dir='hand_posture_data'):
        """
        Prepare dataset from collected samples for training
        """
        X = []  # Features
        y = []  # Labels
        
        # Load all samples
        for i, category in enumerate(self.categories):
            category_dir = os.path.join(data_dir, category)
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory {category_dir} does not exist")
                continue
                
            for filename in os.listdir(category_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(category_dir, filename)
                    try:
                        with open(file_path, 'rb') as f:
                            features = pickle.load(f)
                            X.append(features)
                            y.append(i)  # Use index as the label
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # If we have collected any data
        if len(X) > 0:
            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            
            return X_train, X_test, y_train, y_test
        else:
            print("No data found. Please run collect_training_data() first.")
            return None, None, None, None
    
    def build_model(self, input_shape):
        """
        Build a neural network model for hand posture classification
        """
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(len(self.categories), activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, data_dir='hand_posture_data', epochs=50, batch_size=32):
        """
        Train the model on collected data
        """
        # Prepare dataset
        X_train, X_test, y_train, y_test = self.prepare_dataset(data_dir)
        
        if X_train is None:
            return
            
        # Build model if not already loaded
        if self.model is None:
            self.model = self.build_model(X_train.shape[1])
        
        # Train the model
        print(f"Training model on {len(X_train)} samples...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate the model
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Predictions for classification report
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.categories))
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, 'training_history.png'))
        plt.show()
        
        # Save the model
        model_path = os.path.join(data_dir, 'piano_posture_model.h5')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return history
    
    def predict_posture(self, landmarks):
        """
        Predict the posture category from hand landmarks
        """
        if self.model is None:
            print("No model loaded. Cannot predict.")
            return None, None
            
        # Preprocess landmarks
        features = self.preprocess_landmarks(landmarks)
        features = features.reshape(1, -1)  # Reshape for prediction
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        return self.categories[predicted_class], confidence
    
    def process_frame(self, frame):
        """
        Process a video frame, detect hands, and classify posture
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
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
                
                # Determine handedness
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Position text based on which hand it is
                if handedness == "Right":
                    x_pos = w - 300  # Right side of screen
                else:
                    x_pos = 10       # Left side of screen
                
                # Predict posture if model is loaded
                if self.model is not None:
                    posture, confidence = self.predict_posture(hand_landmarks.landmark)
                    
                    # Determine text color based on posture category
                    if posture == 'good':
                        color = self.GREEN
                    elif posture == 'too_curved':
                        color = self.YELLOW
                    else:  # too_flat
                        color = self.RED
                    
                    # Display prediction with more concise text
                    posture_display = posture.replace('too_', '').replace('_', ' ').title()
                    cv2.putText(frame, f"{handedness}: {posture_display}", 
                               (x_pos, 30 + 30*idx), self.font, 0.7, color, 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                               (x_pos, 60 + 30*idx), self.font, 0.6, color, 2)
                else:
                    cv2.putText(frame, f"{handedness}: No model", 
                               (x_pos, 30 + 30*idx), self.font, 0.7, (0, 0, 255), 2)
        else:
            # No hands detected
            cv2.putText(frame, "No hands detected", 
                       (w//2 - 100, 30), self.font, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def run(self, camera_id=0):
        """
        Run the hand posture classifier using webcam input
        """
        cap = cv2.VideoCapture(camera_id)
        
        # Frame rate calculation variables
        prev_frame_time = 0
        new_frame_time = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam")
                break
                
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Calculate and display FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            cv2.putText(processed_frame, f"FPS: {int(fps)}", 
                       (processed_frame.shape[1]-120, 30), self.font, 0.7, (255,255,255), 2)
            
            # Display the frame
            cv2.imshow("Piano Hand Posture Classification (DL)", processed_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create classifier
    classifier = PianoPostureDeepLearningClassifier()
    
    # Choose mode based on command line argument or ask user
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("Select mode:")
        print("1. Collect training data")
        print("2. Train model")
        print("3. Run classifier")
        mode = input("Enter number (1-3): ")
    
    if mode == "1" or mode == "collect":
        classifier.collect_training_data()
    elif mode == "2" or mode == "train":
        classifier.train_model()
    else:  # Default to run mode
        # Try to load existing model first
        model_path = os.path.join('hand_posture_data', 'piano_posture_model.h5')
        if os.path.exists(model_path):
            classifier = PianoPostureDeepLearningClassifier(model_path)
        classifier.run() 