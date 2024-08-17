import cv2
import mediapipe as mp
import numpy as np

# Mediapipe model initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to calculate the direction of hand movement
def calculate_direction(coords1, coords2):
    delta_x = coords2[0] - coords1[0]
    if delta_x > 0.02:  # Move to the right (threshold lowered for sensitivity)
        return "right"
    elif delta_x < -0.02:  # Move to the left (threshold lowered for sensitivity)
        return "left"
    else:
        return "none"

# Capturing video from webcam
cap = cv2.VideoCapture(0)

# Using Mediapipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    previous_position = None
    direction = None
    wave_count = 0
    frame_counter = 0
    wave_threshold = 5  # Lowered the threshold for faster wave detection

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
                    current_direction = calculate_direction(previous_position, current_position)
                    
                    if direction is None:
                        direction = current_direction
                    elif direction == "right" and current_direction == "left":
                        wave_count += 1
                        direction = None
                        frame_counter = 0
                    elif direction == "left" and current_direction == "right":
                        wave_count += 1
                        direction = None
                        frame_counter = 0
                    else:
                        frame_counter += 1

                    # Dalga hareketi algılandığında
                    if wave_count >= 1:
                        cv2.putText(image, 'Dalga Hareketi Algılandı!', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5, cv2.LINE_AA)
                        wave_count = 0  # Dalga sayacını sıfırla
                        frame_counter = 0  # Frame sayacını sıfırla

                    # Eğer frame sayacı threshold'u geçerse, sayacı sıfırla
                    if frame_counter > wave_threshold:
                        direction = None
                        wave_count = 0
                        frame_counter = 0

                previous_position = current_position

        # Show the video feed
        cv2.imshow('Hand Gesture', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
