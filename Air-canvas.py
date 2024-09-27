import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Giving different arrays to handle colour points of different colours
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]
kpoints = [deque(maxlen=1024)]  # Points for black color

# These indexes will be used to mark the points in particular arrays of specific colours
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0
black_index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

# Adding black to the list of colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 0, 0)]  # Black is added
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)  # Clear button
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)  # Blue
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)  # Green
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)  # Red
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)  # Yellow
paintWindow = cv2.rectangle(paintWindow, (40, 70), (140, 130), (0, 0, 0), 2)  # Black (new)

# Adding labels to the buttons
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLACK", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)  # Black label

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Variables to store the last known face position
last_face_center = None
last_face_radius = None

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw the color selection rectangles
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)  # Clear button
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)  # Blue
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)  # Green
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)  # Red
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)  # Yellow
    frame = cv2.rectangle(frame, (40, 70), (140, 130), (0, 0, 0), 2)  # Black (new)

    # Adding labels
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLACK", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Detect faces in the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw circles around detected faces
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  # Assuming one face for simplicity
        center = (x + w // 2, y + h // 2)  # Calculate center of the detected face
        radius = int(max(w, h) // 2)  # Use half of the width or height for the radius
        last_face_center = center
        last_face_radius = radius
        cv2.circle(frame, center, radius, (128, 128, 128), 2)  # Draw a gray circle around the face
    else:
        # If no face detected, draw the last known position
        if last_face_center and last_face_radius:
            cv2.circle(frame, last_face_center, last_face_radius, (128, 128, 128), 2)  # Draw previous circle

    # Get hand landmark prediction
    result = hands.process(framergb)

    # Post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)
                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        # Check if the thumb is close to the index finger for drawing
        if (thumb[1] - center[1] < 30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1
            kpoints.append(deque(maxlen=512))  # Adding deque for black
            black_index += 1

        elif center[1] <= 130:  # Check within the range of both rows of buttons
            # Check which color was selected
            if 40 <= center[0] <= 140:
                if center[1] <= 65:  # Clear Button
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    kpoints = [deque(maxlen=512)]  # Reset black points

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0
                    black_index = 0

                    paintWindow[67:, :, :] = 255
                else:  # Black Button
                    colorIndex = 4  # Black color
            elif 160 <= center[0] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # Green
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # Red
            elif 505 <= center[0] <= 600:
                colorIndex = 3  # Yellow
        else:
            # Add points to the respective color arrays
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
            elif colorIndex == 4:
                kpoints[black_index].appendleft(center)  # Black

    # Append the next deques when nothing is detected to avoid messing up
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1
        kpoints.append(deque(maxlen=512))
        black_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints, kpoints]  # Including black points
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('x'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
