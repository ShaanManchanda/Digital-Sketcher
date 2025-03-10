import cv2
import os
import numpy as np
import HandTracking as ht

# Load palette images from the Header folder
folder_path = "Header"
my_list = os.listdir(folder_path)
palette = []

for im_path in my_list:
    image = cv2.imread(os.path.join(folder_path, im_path))
    palette.append(image)

# Default header is set to the eraser image initially
header = palette[my_list.index('eraser.jpg')]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = ht.HandDetector(detection_confidence=0.85)

# Initialize canvas after the first frame is captured
canvas = None
pt_size = 60
thickness = 30
current_color = (0, 0, 0)
x_prev, y_prev = 0, 0

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Flip image for mirror effect
    img = cv2.flip(img, 1)

    # Initialize canvas on the first frame with current frame dimensions
    if canvas is None:
        h, w, _ = img.shape
        canvas = np.zeros((h, w, 3), np.uint8)
        # Resize header to span the full width with a fixed height (e.g., 130 pixels)
        header = cv2.resize(header, (w, 130))
    else:
        h, w, _ = img.shape
        # If frame dimensions change, update the canvas and header
        if canvas.shape[0] != h or canvas.shape[1] != w:
            canvas = cv2.resize(canvas, (w, h))
            header = cv2.resize(header, (w, 130))

    # Detect hands and landmarks
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, draw=False)

    if lm_list:
        # Get the index finger position
        x_index, y_index = lm_list[8][1:3]
        fingers = detector.fingers_up()

        # Selection mode: two fingers up
        if fingers[1] == 1 and fingers[2] == 1:
            x_prev, y_prev = 0, 0  # Reset drawing starting point
            cv2.circle(img, (x_index, y_index), 15, (255, 255, 255), cv2.FILLED)
            # Use the header area (first 130 pixels) for selection
            if y_index < 130:
                # Use relative positions based on current frame width
                if 0 < x_index < w // 8:
                    header = palette[my_list.index('eraser.jpg')]
                    current_color = (0, 0, 0)
                elif w // 8 < x_index < w // 4:
                    header = palette[my_list.index('white.jpg')]
                    current_color = (255, 255, 255)
                elif w // 4 < x_index < 3 * w // 8:
                    header = palette[my_list.index('purple.jpg')]
                    current_color = (235, 23, 94)
                elif 3 * w // 8 < x_index < w // 2:
                    header = palette[my_list.index('red.jpg')]
                    current_color = (22, 22, 255)
                elif w // 2 < x_index < 5 * w // 8:
                    header = palette[my_list.index('orange.jpg')]
                    current_color = (77, 145, 255)
                elif 5 * w // 8 < x_index < 3 * w // 4:
                    header = palette[my_list.index('yellow.jpg')]
                    current_color = (89, 222, 255)
                elif 3 * w // 4 < x_index < 7 * w // 8:
                    header = palette[my_list.index('green.jpg')]
                    current_color = (87, 217, 126)
                elif 7 * w // 8 < x_index < w:
                    header = palette[my_list.index('blue.jpg')]
                    current_color = (255, 182, 56)
                # Resize header to maintain overlay dimensions
                header = cv2.resize(header, (w, 130))

        # Drawing mode: only index finger up
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x_index, y_index), pt_size, current_color, cv2.FILLED)
            if x_prev == 0 and y_prev == 0:
                x_prev, y_prev = x_index, y_index
            # Adjust thickness and point size for eraser vs drawing
            if current_color == (0, 0, 0):
                thickness = 60
                pt_size = 30
            else:
                thickness = 10
                pt_size = 10

            # Draw on both the live feed and the canvas
            cv2.line(img, (x_prev, y_prev), (x_index, y_index), current_color, thickness)
            cv2.line(canvas, (x_prev, y_prev), (x_index, y_index), current_color, thickness)
            x_prev, y_prev = x_index, y_index

    # Prepare inverse image from the canvas for blending
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    # Ensure the inverse image matches the frame size
    if img_inv.shape[:2] != (h, w):
        img_inv = cv2.resize(img_inv, (w, h))

    # Combine the images using bitwise operations
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # Overlay the header (palette) on the top of the image
    img[0:130, 0:w] = header

    # Blend the canvas and the current frame for a smoother appearance
    img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    cv2.imshow("Whiteboard", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
