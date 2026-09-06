import cv2
import mediapipe as mp
import numpy as np
import time
import os

# setup
CAM_WIDTH, CAM_HEIGHT = 1280, 720
BRUSH_THICKNESS = 8
ERASER_THICKNESS = 60
SMOOTHING = 5  # higher smoothing = smoother but laggier line
PALETTE = [
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Purple", (255, 0, 255)),
    ("Eraser", (0, 0, 0))  # black as placeholder
]
BUTTON_WIDTH = CAM_WIDTH // len(PALETTE)
BUTTON_HEIGHT = 80
MAX_MISSED_FRAMES = 5

# mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky finger landmark IDs


def fingers_up(landmarks, handedness_label):
    """Returns a list of 5 booleans indicating which fingers are up."""
    fingers = []
    # for thumb: compare x-position
    if handedness_label == "Right":
        fingers.append(landmarks[FINGERTIP_IDS[0]][0] > landmarks[FINGERTIP_IDS[0] - 1][0])
    else:
        fingers.append(landmarks[FINGERTIP_IDS[0]][0] < landmarks[FINGERTIP_IDS[0] - 1][0])
    # for other fingers: compare y-position; fingertip higher than knuckle position = extended
    for fingertip_id in FINGERTIP_IDS[1:]:
        fingers.append(landmarks[fingertip_id][1] < landmarks[fingertip_id - 2][1])
    return fingers


def draw_palette(img, selected_colour):
    """Draws the colour buttons."""
    for i in range(len(PALETTE)):
        button = PALETTE[i]
        name, colour = button[0], button[1]
        x1 = i * BUTTON_WIDTH
        x2 = x1 + BUTTON_WIDTH
        if name == "Eraser":
            display_colour = (50, 50, 50)
        else:
            display_colour = colour

        cv2.rectangle(img, (x1, 0), (x2, BUTTON_HEIGHT), display_colour, -1)
        if name != "Eraser":
            if colour == selected_colour:
                cv2.rectangle(img, (x1, 0), (x2, BUTTON_HEIGHT), (255,255,255), 3)
    return img


def select_camera():
    """Allows the user to preview available cameras and choose one."""
    available_indexes = []
    for i in range(5):
        test_cap = cv2.VideoCapture(i)
        opened = test_cap.isOpened()
        if opened:
            frame_read, frame = test_cap.read()
            if frame_read:
                available_indexes.append(i)
        test_cap.release()

    if len(available_indexes) == 1:
        return cv2.VideoCapture(available_indexes[0])

    # if multiple cameras are found: let the user choose
    curr_position = 0
    chosen_index = None
    print("Use left/right arrows to switch between cameras, ENTER to select, Q to quit.")

    while chosen_index is None:
        curr_idx = available_indexes[curr_position]
        preview_cap = cv2.VideoCapture(curr_idx)

        # camera warm up
        for _ in range(5):
            preview_cap.read()

        switch_camera = False
        while not switch_camera:
            frame_read, frame = preview_cap.read()
            if not frame_read:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                f"Camera {curr_idx + 1} ({curr_position + 1} of {len(available_indexes)})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "LEFT/RIGHT to switch, ENTER to select, Q to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imshow("Choose your camera", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                cv2.destroyAllWindows()
                return preview_cap
            elif key == ord('q'):  # quit
                preview_cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == 2:  # LEFT
                curr_position = (curr_position - 1) % len(available_indexes)
                switch_camera = True
            elif key == 3:  # RIGHT
                curr_position = (curr_position + 1) % len(available_indexes)
                switch_camera = True

        preview_cap.release()


def main():
    # open camera
    cap = select_camera()
    if cap is None:
        print("No camera selected, exiting program.")
        return
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check that a camera is open and is not being used by another application.")
        return

    canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)
    curr_colour = PALETTE[0][1]
    is_eraser = False

    prev_x = 0  # tracks previous fingertip position
    prev_y = 0
    smooth_x = 0
    smooth_y = 0
    missed_frames = 0
    print("Whiteboard is running. Press Q to quit, C to clear, S to save.")

    while cap.isOpened():
        grabbed_frame, frame = cap.read()
        if not grabbed_frame:
            print("Failed to get frame from webcam.")
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # checking for a detected hand and extracting landmarks
        if results.multi_hand_landmarks and results.multi_handedness:
            missed_frames = 0
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = results.multi_handedness[0].classification[0].label
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append((int(lm.x * CAM_WIDTH), int(lm.y * CAM_HEIGHT)))
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            up = fingers_up(landmarks, handedness_label)
            index_x, index_y = landmarks[8]
            if smooth_x == 0 and smooth_y == 0:  # initial values, first frame with hand detected
                smooth_x = index_x
                smooth_y = index_y
            smooth_x = smooth_x + (index_x - smooth_x) // SMOOTHING  # filters out small instabilities in motion
            smooth_y = smooth_y + (index_y - smooth_y) // SMOOTHING

            # current action
            index_only = up[1] and not up[2] and not up[3] and not up[4]
            select_mode = up[1] and up[2] and not up[3] and not up[4]
            open_palm = all(up)

            if select_mode:
                prev_x, prev_y = 0, 0  # lift pen
                if smooth_y < BUTTON_HEIGHT:
                    idx = min(smooth_x // BUTTON_WIDTH, len(PALETTE) - 1)
                    name, colour = PALETTE[idx]
                    if name == "Eraser":
                        is_eraser = True
                    else:
                        is_eraser = False
                        curr_colour = colour
                cv2.circle(frame, (smooth_x, smooth_y), 15, (255, 255, 255), 3)

            elif index_only and not open_palm:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = smooth_x, smooth_y
                if is_eraser:
                    thickness = ERASER_THICKNESS
                    draw_colour = (0, 0, 0)
                else:
                    thickness = BRUSH_THICKNESS
                    draw_colour = curr_colour
                cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_colour, thickness)
                prev_x, prev_y = smooth_x, smooth_y
                if is_eraser:
                    cv2.circle(frame, (smooth_x, smooth_y), 10, (255, 255, 255), cv2.FILLED)
                else:
                    cv2.circle(frame, (smooth_x, smooth_y), 10, curr_colour, cv2.FILLED)

            else:  # lift pen
                prev_x, prev_y = 0, 0

        else:  # no hand detected
            missed_frames += 1
            if missed_frames > MAX_MISSED_FRAMES:
                prev_x, prev_y = 0, 0

        # merging canvas with live camera feed
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)  # converts canvas to grayscale for easier comparison
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask_inv)
        combined = cv2.add(frame_bg, canvas_fg)

        if is_eraser:
            combined = draw_palette(combined, None)
        else:
            combined = draw_palette(combined, curr_colour)

        cv2.imshow("Hand-Tracking Virtual Whiteboard", combined)

        # keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), np.uint8)
        elif key == ord('s'):
            output_path = os.path.join(os.getcwd(), "whiteboard_output.png")
            cv2.imwrite(output_path, canvas)
            print(f"Saved canvas to {output_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()