import cv2
import streamlit as st
import numpy as np
import tempfile
import os

# Load pre-trained MobileNet SSD model for object detection
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

# Create an array to store trackers and objects
trackers = []
objects = []

# Function to perform object detection using MobileNet SSD
def detect_objects(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    return detections, (w, h)

# Function to initialize a tracker for a detected object
def initialize_tracker(frame, box):
    tracker = cv2.TrackerMIL_create()
    (x, y, w, h) = [int(v) for v in box]
    # Check if the bounding box coordinates are valid before initialization
    if w > 0 and h > 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0]:
        tracker.init(frame, tuple(box))
        return tracker
    else:
        st.warning("Ignoring invalid bounding box for tracker initialization")
        return None

# Streamlit App
def main():
    st.title("Real-time Object Tracking with Streamlit")

    # File upload widget for recorded video
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Create a temporary directory to store the uploaded video
        temp_dir = tempfile.TemporaryDirectory()
        video_path = os.path.join(temp_dir.name, "uploaded_video.mp4")

        # Save the uploaded video to the temporary directory
        with open(video_path, "wb") as video_file:
            video_file.write(uploaded_file.read())

        # Open video capture using the temporary video file path
        cap = cv2.VideoCapture(video_path)

        # Streamlit loop
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Resize the frame to reduce memory usage
            frame = cv2.resize(frame, (640, 480))  # Adjust the resolution as needed

            detections, (frame_width, frame_height) = detect_objects(frame)

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.2:  # Threshold for confidence
                    box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = f"Object {i + 1}: {int(confidence * 100)}%"

                    # Draw bounding box and label
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Initialize tracker for the detected object
                    if startY < endY and startX < endX:
                        tracker = initialize_tracker(frame, (startX, startY, endX - startX, endY - startY))
                        if tracker is not None:
                            trackers.append((tracker, box))
                    else:
                        st.warning(f"Ignoring invalid region for object_{i + 1}.jpg")

            # Update trackers for each object
            for tracker, obj in trackers:
                success, box = tracker.update(frame)

                if success:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    # If tracking fails, remove the tracker and object
                    trackers.remove((tracker, obj))

            # Display the resulting frame using Streamlit
            st.image(frame, channels="BGR", use_column_width=True)

        # Release resources
        cap.release()
        # Close and remove the temporary directory
        temp_dir.cleanup()

if __name__ == "__main__":
    main()
