import cv2

def save_frame_from_video(video_path, frame_number, output_image_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Error: Unable to open video from {video_path}")
        return

    # Set the frame position to the desired frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    if ret:
        # Save the frame as a jpg image
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_number} saved as {output_image_path}")
    else:
        print(f"Error: Unable to retrieve frame {frame_number}")

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    video_path = 'videos/barbell.mp4'  # Path to the input video
    frame_number = 250  # Frame number you want to capture
    output_image_path = 'images/barbell_250.jpg'  # Path to save the extracted frame

    save_frame_from_video(video_path, frame_number, output_image_path)