import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import datetime
import os
import time

# Load the trained violence detection model
model = tf.keras.models.load_model(r'E:\DUK\SEM 3\mini project\results\model\cnn_violence2.h5')

# Load YOLO
net = cv2.dnn.readNetFromDarknet(r'E:\DUK\SEM 3\mini project\results\model\yolov4.cfg', 
                                 r'E:\DUK\SEM 3\mini project\results\model\yolov4.weights')
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers_indices = output_layers_indices.flatten()
output_layers = [layer_names[i - 1] for i in output_layers_indices]

# Load class names
with open(r'E:\DUK\SEM 3\mini project\results\model\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture(0)

# Variables for recording
recording = False
video_writer = None
recording_start_time = None
MIN_RECORDING_DURATION = 10  # Minimum duration in seconds

# Directory to save detected videos
output_directory = r'E:\DUK\SEM 3\mini project\violence detected videos'
os.makedirs(output_directory, exist_ok=True)

# Function to make predictions for violence detection
def make_prediction(frame):
    img = Image.fromarray(frame)
    img = img.resize((128, 128))
    img = np.array(img) / 255.0  # Normalize if model was trained with normalized data
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img)[0][0]  # Get prediction as scalar
    
    return res > 0.5, res  # Return both a boolean for detection and the probability score

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Make prediction for violence detection
    violence_detected, confidence = make_prediction(frame)
    
    if violence_detected:
        if not recording:
            # Start recording with a new filename and timestamp
            recording = True
            recording_start_time = time.time()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_directory, f"violence_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
            video_writer = cv2.VideoWriter(filename, fourcc, 20, (width, height))
            print(f"Recording started: {filename}")

        # Write the frame to the video file
        video_writer.write(frame)
        cv2.putText(frame, f"Violence Detected ({confidence:.2f})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # Check if recording time has reached the minimum duration
        if recording and (time.time() - recording_start_time) >= MIN_RECORDING_DURATION:
            # Stop recording if minimum duration has been reached
            recording = False
            video_writer.release()
            print("Recording stopped after minimum duration")

    # Display the resulting frame
    cv2.imshow('YOLO Violence Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
if recording:
    video_writer.release()
cv2.destroyAllWindows()
