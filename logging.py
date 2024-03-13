import streamlit as st
import cv2
import numpy as np
import os
import datetime
import pywhatkit
import time
import pyautogui
import glob

# Set up file handling for log messages
def write_to_log(message):
    with open("logging/log.json", "a") as log_file:
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# File handling methods for display messages
def display_warning(message):
    print(f"Warning: {message}")
    write_to_log(f"Warning: {message}")

def display_error(message):
    print(f"Error: {message}")
    write_to_log(f"Error: {message}")

def display_success(message):
    print(f"Success: {message}")

# Load pre-trained MobileNet SSD model for object detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Function to perform object detection using MobileNet SSD
def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    return detections, (w, h)

# Modified initialize_tracker function
def initialize_tracker(frame, box):
    tracker = cv2.TrackerMIL_create()

    # Extract bounding box coordinates
    (x, y, w, h) = [int(v) for v in box]

    # Check if the bounding box coordinates and dimensions are valid
    if w > 0 and h > 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
        try:
            # Initialize the tracker with the bounding box as a positive sample
            tracker.init(frame, (x, y, w, h))
            write_to_log(f"Tracker initialized for bounding box: {box}")
            return tracker
        except Exception as e:
            display_error(f"Error initializing tracker: {e}")
            return None
    else:
        display_warning("Ignoring invalid bounding box for tracker initialization")
        return None

# Function to check for motion between frames
def has_motion(prev_frame, current_frame, threshold=100):
    # Convert frames to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Resize frames to the minimum dimension
    min_height = min(prev_frame_gray.shape[0], current_frame_gray.shape[0])
    min_width = min(prev_frame_gray.shape[1], current_frame_gray.shape[1])

    prev_frame_gray = cv2.resize(prev_frame_gray, (min_width, min_height))
    current_frame_gray = cv2.resize(current_frame_gray, (min_width, min_height))

    # Compute absolute difference
    diff_frame = cv2.absdiff(prev_frame_gray, current_frame_gray)

    # Apply threshold to identify significant differences
    _, thresholded_diff = cv2.threshold(diff_frame, threshold, 255, cv2.THRESH_BINARY)

    # Count non-zero pixels in the thresholded difference
    motion = np.count_nonzero(thresholded_diff) > threshold

    return motion

# Login Page
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "user" and password == "123":
            st.empty()
            display_success("Logged in successfully as User!")
            return "user"
        elif username == "admin" and password == "123":
            display_success("Logged in successfully as Admin!")
            st.empty()
            return "admin"
        else:
            display_error("Invalid username or password.")
    return None

# User Page for Object Tracking
def user_page():
    st.title("User Page - Object Tracking")

    # Create an array to store trackers and objects
    trackers = []

    # DroidCam stream URL for user (replace with your phone's IP address and port)
    droidcam_url = "http://192.168.254.98:8080/video"

    # Open video capture using DroidCam stream URL
    cap = cv2.VideoCapture(droidcam_url)

    # Check if the video capture is opened successfully
    if not cap.isOpened():
        display_error("Error: Unable to open the DroidCam stream. Please check the URL.")
        return

    display_success("DroidCam stream opened successfully.")

    # Initialize previous frame for motion detection
    _, prev_frame = cap.read()

    # Session state for motion detection
    motion_detected = False

    # List to store frames for export
    export_frames = []

    # Variables for tracking WhatsApp message count
    start_time = time.time()
    message_count = 0

    # Streamlit loop for object tracking
    while True:
        ret, frame = cap.read()

        if not ret:
            display_error("Error reading frame from video capture.")
            break

        # Resize the frame to reduce memory usage and maintain quality
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Adjust size as needed

        detections, (frame_width, frame_height) = detect_objects(frame)

        # Check for motion in the frame
        motion = has_motion(prev_frame, frame)

        if motion and not motion_detected:
            display_warning("Motion detected!")
            motion_detected = True
        elif not motion and motion_detected:
            motion_detected = False
            display_success("Motion stopped.")

        # Display frames only when motion is detected
        if motion:
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.2:  # Display a green box only if there is motion
                    box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = f"Object {i + 1}: {int(confidence * 100)}%"

                    # Draw bounding box and label
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Initialize a tracker for the detected object
                    if startY < endY and startX < endX:
                        tracker = initialize_tracker(frame, (startX, startY, endX - startX, endY - startY))
                        if tracker is not None:
                            trackers.append((tracker, box))
                    else:
                        display_warning(f"Ignoring an invalid region for object_{i + 1}.jpg")

            # Update trackers for each object
            new_trackers = []
            for tracker, obj in trackers:
                success, box = tracker.update(frame)

                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    new_trackers.append((tracker, box))
                else:
                    # If tracking fails, remove the tracker and object
                    display_warning(f"Tracker for object {obj} removed due to tracking failure")

            # Update the trackers list
            trackers = new_trackers

            # Display the resulting frame using Streamlit
            st.image(frame, channels="BGR", use_column_width=True, output_format="BGR")

            # Store frames for export
            export_frames.append(frame)

            # Increment message count if confidence > 85%
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.85:  # Change the confidence threshold as needed
                    message_count += 1
                    # Save the frame when a security breach is detected
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    cv2.imwrite(f"security_breach_{current_time}.jpg", frame)

        # Update the previous frame for the next iteration
        prev_frame = frame.copy()

        # Check if 1 minute has passed
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= 60:  # 60 seconds = 1 minute
            # Send WhatsApp message with total message count
            if message_count > 0:
                latest_frame = max(glob.glob("security_breach_*.jpg"), key=os.path.getctime)
                pywhatkit.sendwhats_image("+916380923193", latest_frame, f"Security Breach Detected! Total messages: {message_count}")
            # Reset variables for the next minute
            start_time = current_time
            message_count = 0

    # Release resources
    cap.release()
    display_success("Video capture released.")

    # Ask the user if they want to export frames
    export_option = st.selectbox("Do you want to export the displayed frames?", ("Yes", "No"))
    if export_option == "Yes":
        submit_button = st.button("Submit")
        if submit_button:
            # Create a folder to save frames
            export_folder = "exported_frames"
            if not os.path.exists(export_folder):
                os.makedirs(export_folder)

            # Save frames with current time as filenames
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for i, frame in enumerate(export_frames):
                filename = os.path.join(export_folder, f"frame_{current_time}_{i}.jpg")
                cv2.imwrite(filename, frame)

            display_success(f"Frames exported successfully to folder: {export_folder}")

# Admin Page for Live Video Display
def admin_page():
    st.title("Admin Page - Live Video Display")

    # DroidCam stream URL for admin (replace with your phone's IP address and port)
    droidcam_url = "http://192.168.254.98:8080/video"

    # Open video capture using DroidCam stream URL
    cap = cv2.VideoCapture(droidcam_url)

    # Check if the video capture is opened successfully
    if not cap.isOpened():
        display_error("Error: Unable to open the DroidCam stream. Please check the URL.")
        return

    display_success("DroidCam stream opened successfully.")

    # Streamlit loop for displaying live video
    while True:
        ret, frame = cap.read()

        if not ret:
            display_error("Error reading frame from video capture.")
            break

        # Resize the frame to reduce memory usage and maintain quality
        frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Adjust size as needed

        # Display the resulting frame using Streamlit
        st.sidebar.image(frame, channels="BGR", caption="DroidCam IP Camera Feed", width=300, output_format="BGR")

    # Release resources
    cap.release()
    display_success("Video capture released.")

# Main function to run the application
def main():
    user_type = login()
    if user_type == "user":
        st.empty()
        user_page()
    elif user_type == "admin":
        st.empty()
        admin_page()

if __name__ == "__main__":
    main()
