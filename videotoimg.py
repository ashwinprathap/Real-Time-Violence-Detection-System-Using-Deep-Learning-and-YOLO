import cv2
import os

# Path to the folder containing videos
video_folder = 'E:\\DUK\\SEM 3\\mini project\\videodatas\\Violence'

# Path to save the extracted frames
output_folder = 'E:\\DUK\\SEM 3\\mini project\\demo\\violence'


# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through each video file in the video folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add other formats if needed
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        # Check if video is opened successfully
        if not cap.isOpened():
            print(f"Error opening video file {video_file}")
            continue
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save the frame as an image
            frame_filename = f"{video_file}_frame_{frame_count}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            
            frame_count += 1
        
        cap.release()
        print(f"Frames extracted and saved for {video_file}")

print("Extraction complete.")
