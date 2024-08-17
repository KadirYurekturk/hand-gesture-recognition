import cv2
import mediapipe as mp
import numpy as np

# Mediapipe model initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize a blank canvas
canvas = np.ones((720, 1280, 3), dtype="uint8") * 255  # White background

# Function to calculate the direction of hand movement
def calculate_direction(coords1, coords2):
    delta_x = coords2[0] - coords1[0]
    if delta_x > 0.02:
        return "right"
    elif delta_x < -0.02:
        return "left"
    else:
        return "none"

# Capturing video from webcam
cap = cv2.VideoCapture(0)

# Using Mediapipe Hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    previous_position = None
    drawing = False  # To check if drawing mode is active

    while cap.isOpened():
        ret, frame = cap.read()

        # Flip the frame horizontally to create a mirror effect
        frame = cv2.flip(frame, 1)

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
                
                # Get the index finger tip position
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * 1280)  # Scale to canvas size
                y = int(index_finger_tip.y * 720)

                if previous_position:
                    if drawing:  # If drawing mode is active, draw on the canvas
                        cv2.line(canvas, previous_position, (x, y), (0, 0, 0), 4)  # Draw a black line

                previous_position = (x, y)

            # If thumb tip and index finger tip are close together, activate drawing mode
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_coords = (int(thumb_tip.x * 1280), int(thumb_tip.y * 720))
            index_coords = (int(index_finger_tip.x * 1280), int(index_finger_tip.y * 720))

            distance = np.linalg.norm(np.array(thumb_coords) - np.array(index_coords))
            if distance < 30:  # If thumb and index finger are close, start drawing
                drawing = True
            else:
                drawing = False

        # Show the video feed and the canvas
        cv2.imshow('Hand Gesture', image)
        cv2.imshow('Canvas', canvas)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
