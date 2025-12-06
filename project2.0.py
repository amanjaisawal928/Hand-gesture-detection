import cv2
import mediapipe as mp
import webbrowser
import time
import os

# --- CONFIGURATION ---
# Which sites to open
WEBSITES = {
    1: "https://www.youtube.com",
    2: "https://www.google.com/chrome",
    3: "https://www.facebook.com",
    4: "https://www.instagram.com",
    5: "https://www.linkedin.com"
}

# Cooldown settings
COOLDOWN_DURATION = 3  
last_action_time = 0

# --- SETUP SYSTEM ---
mp_hands = mp.solutions.hands
# Simple setup: We only look for 1 hand
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

print("System Started. Show your hand.")

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue

    # 1. Flip the image (Mirror view) and convert color
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. Find Hands
    results = hands.process(img_rgb)
    
    # 3. If a hand is found...
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark # Short variable name for easier typing
        
        # Draw the skeleton
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- COUNT FINGERS (SIMPLE WAY) ---
        fingers = 0

        # THUMB: Check if Tip (4) is to the right of Joint (3)
        if lm[4].x > lm[3].x:
            fingers = fingers + 1

        # INDEX FINGER: Check if Tip (8) is higher than Joint (6)
        # Note: In images, "Higher" means a smaller Y value
        if lm[8].y < lm[6].y:
            fingers = fingers + 1

        # MIDDLE FINGER: Tip (12) vs Joint (10)
        if lm[12].y < lm[10].y:
            fingers = fingers + 1

        # RING FINGER: Tip (16) vs Joint (14)
        if lm[16].y < lm[14].y:
            fingers = fingers + 1

        # PINKY FINGER: Tip (20) vs Joint (18)
        if lm[20].y < lm[18].y:
            fingers = fingers + 1

        # --- DISPLAY & TRIGGER ---
        # Show the number on screen
        cv2.putText(img, f"Fingers: {fingers}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Check if we should open a website
        current_time = time.time()
        if fingers in WEBSITES:
            if (current_time - last_action_time) > COOLDOWN_DURATION:
                site = WEBSITES[fingers]
                print(f"Opening {site}")
                webbrowser.open(site)
                last_action_time = current_time

    # 4. Show the final image
    cv2.imshow("Hand Controller", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()