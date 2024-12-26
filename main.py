import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as generative_ai
from PIL import Image
import streamlit as stream

# Set the page configuration (must be the first Streamlit command)
stream.set_page_config(layout="wide")

left_col, right_col = stream.columns([4, 4])

with left_col:
    is_running = stream.checkbox('Process', value=True)
    webcam_view = stream.image([])

with right_col:
    stream.title("Response")
    response_display = stream.subheader("")

# Configure the AI model
generative_ai.configure(api_key="#Place your API key here")
ai_model = generative_ai.GenerativeModel('gemini-1.5-flash')

# Initialise the webcam to capture video
video_capture = cv2.VideoCapture(0)
video_capture.set(3, 1280)
video_capture.set(4, 720)

# Initialise the HandDetector class
hand_tracker = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def extract_hand_info(frame):
    hands_detected, processed_frame = hand_tracker.findHands(frame, draw=False, flipType=True)
    if hands_detected:
        primary_hand = hands_detected[0]
        landmark_list = primary_hand['lmList']
        raised_fingers = hand_tracker.fingersUp(primary_hand)
        print(raised_fingers)
        return raised_fingers, landmark_list
    else:
        return None

def draw_on_canvas(hand_info, previous_position, sketch):
    raised_fingers, landmarks = hand_info
    current_position = None
    if raised_fingers == [0, 1, 0, 0, 0]:
        current_position = landmarks[8][0:2]
        if previous_position is None:
            previous_position = current_position
        cv2.line(sketch, current_position, previous_position, (0, 255, 0), 15)
    elif raised_fingers == [1, 0, 0, 0, 0]:
        sketch = np.zeros_like(sketch)
    return current_position, sketch

def process_with_ai(ai_model, sketch_image):
    image_for_ai = Image.fromarray(sketch_image)
    ai_response = ai_model.generate_content(["Solve this math problem", image_for_ai])
    # Extracting text from the response and cleaning unnecessary characters
    ai_output_text = ai_response.candidates[0].content.parts[0].text
    return ai_output_text.replace('$\\boxed{', '').replace('}$', '')

last_position = None
drawing_canvas = None
result_text = ""

# Continuously get frames from the webcam
while True:
    frame_success, video_frame = video_capture.read()
    video_frame = cv2.flip(video_frame, 1)
    if drawing_canvas is None:
        drawing_canvas = np.zeros_like(video_frame)

    hand_info = extract_hand_info(video_frame)
    if hand_info:
        finger_positions, landmarks = hand_info
        last_position, drawing_canvas = draw_on_canvas(hand_info, last_position, drawing_canvas)
        if finger_positions == [1, 1, 1, 1, 1]:
            result_text = process_with_ai(ai_model, drawing_canvas)
            print(result_text)

    combined_frame = cv2.addWeighted(video_frame, 0.7, drawing_canvas, 0.3, 0)
    webcam_view.image(combined_frame, channels="BGR")

    if result_text:
        response_display.text(result_text)

    cv2.waitKey(1)
