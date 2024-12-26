import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import streamlit as st
from PIL import Image

# Streamlit configuration
st.set_page_config(layout="wide")
col1, col2 = st.columns([4, 4])

with col1:
    run = st.checkbox('Process', value=True)
    FRAME_WINDOW = st.image([])

with col2:
    st.title("Response")
    output_text_area = st.text_area("AI Output", height=300)

# Initialise webcam and HandDetector
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(staticMode=False, maxHands=1, detectionCon=0.7)

# Initialise variables
prev_pos = None
canvas = None

# Define functions
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmList[8][0:2]
        if prev_pos is not None:
            cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (0, 255, 0), 5)
        prev_pos = current_pos
    elif fingers == [1, 0, 0, 0, 0]:  # Thumb up (clear canvas)
        canvas = np.zeros_like(canvas)
        prev_pos = None
    return prev_pos, canvas

while run:
    success, img = cap.read()
    if not success:
        st.write("Failed to access the webcam.")
        break

    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        prev_pos, canvas = draw(info, prev_pos, canvas)

    combined_img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(combined_img, channels="BGR")

    # Placeholder for AI output (mockup for now)
    output_text_area.text("AI output will appear here.")
