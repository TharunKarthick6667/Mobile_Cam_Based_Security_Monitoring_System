import cv2
import streamlit as st
import numpy as np

# Load pre-trained MobileNet SSD model for object detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Create an array to store trackers and objects
trackers = []

# Function to perform object detection using MobileNet SSD
def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    return detections, (w, h)

def initialize_tracker(frame, box):
    tracker = cv2.TrackerMIL_create()

    # Extract bounding box coordinates
    (x, y, w, h) = [int(v) for v in box]

    # Check if the bounding box coordinates and dimensions are valid
    if w > 0 and h > 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
        # Initialize the tracker with the bounding box as a positive sample
        tracker.init(frame, (x, y, w, h))

        return tracker
    else:
        st.warning("Ignoring invalid bounding box for tracker initialization")
        return None

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

# Streamlit App
def main():
    global trackers  # Declare trackers as a global variable

    st.title("Real-time Object Tracking with Streamlit")

    # DroidCam stream URL (replace with your phone's IP address and port)
    droidcam_url = "http://192.168.31.99:8080/video"

    # Open video capture using DroidCam stream URL
    cap = cv2.VideoCapture(droidcam_url)

    # Check if the video capture is opened successfully
    if not cap.isOpened():
        st.error("Error: Unable to open the DroidCam stream. Please check the URL.")
        return

    # Initialize previous frame for motion detection
    _, prev_frame = cap.read()

    # Session state for motion detection
    motion_detected = False

    # Streamlit loop
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame to reduce memory usage
        frame = cv2.resize(frame, (640, 480))  # Adjust the resolution as needed

        detections, (frame_width, frame_height) = detect_objects(frame)

        # Check for motion in the frame
        motion = has_motion(prev_frame, frame)

        if motion and not motion_detected:
            st.warning("Motion detected!")
            motion_detected = True
        elif not motion and motion_detected:
            motion_detected = False

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.2 and motion:  # Display a green box only if there is motion
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
                    st.warning(f"Ignoring an invalid region for object_{i + 1}.jpg")

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
                st.warning(f"Tracker for object {obj} removed due to tracking failure")

        # Update the trackers list
        trackers = new_trackers

        # Display the resulting frame using Streamlit
        st.image(frame, channels="BGR", use_column_width=True, output_format="BGR")

        # Update the previous frame for the next iteration
        prev_frame = frame.copy()

    # Release resources
    cap.release()

if __name__ == "__main__":
    main()
