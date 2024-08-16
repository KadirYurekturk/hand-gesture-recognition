import cv2
import mediapipe as mp
import numpy as np

# Mediapipe model initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to calculate the speed of hand movement
def calculate_speed(coords1, coords2):
    return np.linalg.norm(np.array(coords2) - np.array(coords1))

# Capturing video from webcam
cap = cv2.VideoCapture(0)

# Using Mediapipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    previous_position = None
    movement_threshold = 0.1  # Threshold to consider the hand movement as waving
    waving_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        # Recoloring our current image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection using hands
        results = hands.process(image)

        # Re-coloring back to normal
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extracting hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                current_position = (wrist.x, wrist.y)

                if previous_position:
                    movement_speed = calculate_speed(previous_position, current_position)
                    
                    if movement_speed > movement_threshold:
                        waving_count += 1
                    else:
                        waving_count = 0

                    if waving_count >= 5:  # If hand waves continuously for a few frames
                        cv2.putText(image, 'Bye Bye!', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
                        cv2.imshow('Hand Gesture', image)
                        cv2.waitKey(2000)
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                previous_position = current_position

        # Show the video feed
        cv2.imshow('Hand Gesture', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()