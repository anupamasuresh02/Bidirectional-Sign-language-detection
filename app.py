import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import time
import google.generativeai as genai
import streamlit as st

# Load the model and class mapping
model = load_model('model_99.h5')
with open('index_to_class_mapping.json', 'r') as json_file:
    index_to_class = json.load(json_file)

# Function to predict the label from an image
def get_label_from_frame(frame, x_min, y_min, x_max, y_max):
    # Crop the ROI from the frame
    roi = frame[y_min:y_max, x_min:x_max]
    roi_resized = cv2.resize(roi, (150, 150))  # Resize to match the model's input size
    img_array = image.img_to_array(roi_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict the class
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = index_to_class[str(predicted_class[0])]
    return predicted_label

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(1)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=50, detectShadows=False)

# Parameters for motion sensitivity
stationary_threshold = 50
motion_pixel_low = 300
motion_pixel_high = 2000
stationary_counters = {}
predicted_labels = {}

margin = 30  # Margin to expand the ROI

# Variables to track motion status and display text
previous_motion_status = None
display_text = ""
collected_text = []  # List to store collected text
previous_label = None
last_print_time = time.time()  # Timer to track when to print the collected text

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAB2D8zDGPVR5tsGUIbka4CVL4rG1FXXNE"  # Replace with your Gemini API key
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-pro')  # Use the Gemini Pro model

# Function to generate output using Gemini model
def generate_gemini_output(text_list):
    # Combine the collected text into a single string
    combined_text = " ".join(text_list)
    
    # Define a prompt
    prompt = f"""The following are sign language gestures detected in the last 5 seconds: {combined_text}. 
    Some of the characters might be repeated. 
    By removing those repeated characters, summarize the meaning or context of these gestures.
    'space' and 'del' also present in the sign add a space when 'space' is included and delete the previous letter when 'del'
    is detected. 
    Give only that summarized output. Don't give anything other than that."""
    
    # Generate output using Gemini
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit app
st.title("Hand Gesture Detection with Gemini Output")

# Create a layout with a single column for the video feed
st.header("Video Feed")
video_placeholder = st.empty()  # Placeholder for the video feed

# Create two columns below the video feed for the collected text and Gemini output
col1, col2 = st.columns(2)  # Two columns for text outputs

with col1:
    st.header("Collected Text")
    sign_change_placeholder = st.empty()  # Placeholder for the sign change text

with col2:
    st.header("Gemini Output")
    text_placeholder = st.empty()  # Placeholder for the Gemini output

# Main processing loop
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,  # Corrected
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)

        # Apply background subtraction
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = fgbg.apply(gray_frame)
        blurred_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        _, thresh_mask = cv2.threshold(blurred_mask, 25, 255, cv2.THRESH_BINARY)

        # Process the frame with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Predict and display sign for every frame
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Calculate bounding box coordinates
                h, w, _ = frame.shape
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x, x_min) 
                    y_min = min(y, y_min) 
                    x_max = max(x, x_max) 
                    y_max = max(y, y_max)

                # Expand the ROI with margin
                x_min = max(x_min - margin, 0)
                y_min = max(y_min - margin, 0)
                x_max = min(x_max + margin, w)
                y_max = min(y_max + margin, h)

                # Predict the label for the ROI
                label = get_label_from_frame(frame, x_min, y_min, x_max, y_max)

                # Check motion status
                roi = thresh_mask[y_min:y_max, x_min:x_max]
                motion_pixels = cv2.countNonZero(roi)
                if motion_pixels < motion_pixel_low:
                    motion_status = "Stationary"
                    motion_color = (0, 255, 0)  # Green
                elif motion_pixels > motion_pixel_high:
                    motion_status = "Excessive Motion"
                    motion_color = (0, 0, 255)  # Red
                else:
                    motion_status = "Moving"
                    motion_color = (255, 255, 0)  # Yellow

                # Check for transition from Excessive Motion to Stationary
                if previous_motion_status == "Excessive Motion" or previous_motion_status == "Moving" and motion_status == "Stationary":
                    display_text = f"Sign changed to: {label}"
                    # Add label to collected_text based on conditions
                    if label != previous_label:  # Only add if the label is different from the previous one
                        if label == "space":  # Add a space if the label is 'space'
                            collected_text.append(" ")
                        elif label == "del":  # Delete the previous label if the label is 'del'
                            if collected_text:  # Ensure there is something to delete
                                collected_text.pop()
                        else:  # Add the label if it's not 'space' or 'del'
                            collected_text.append(label)
                        previous_label = label  # Update the previous label
                    combined_text = ''.join(collected_text)
                    sign_change_placeholder.text(combined_text)  # Display sign change in the smaller window

                # Update previous motion status
                previous_motion_status = motion_status

                # Draw the bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), motion_color, 2)
                cv2.putText(frame, f"Status: {motion_status} : {motion_pixels}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
                cv2.putText(frame, f"Label: {label}", (x_max + 10, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the additional text in the corner of the screen
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Check if 20 seconds have passed
        current_time = time.time()
        if current_time - last_print_time >= 20:
            if collected_text:  # If there is collected text
                # Generate output using Gemini model
                gemini_output = generate_gemini_output(collected_text)
                text_placeholder.text(gemini_output)  # Display Gemini output in the text placeholder
                collected_text = []  # Clear the collected text list
            last_print_time = current_time  # Reset the timer

        # Display the video feed in the Streamlit app
        frame = cv2.resize(frame, (1000, 750))  # Resize frame for better visibility
        video_placeholder.image(frame, channels="BGR")

        # Exit on 'q' key (not applicable in Streamlit, but kept for reference)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()