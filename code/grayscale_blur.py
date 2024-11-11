import cv2
import numpy as np
import argparse
from ultralytics import YOLO

def isolate_person_in_frame(frame, model, score_threshold=0.3, person_class_id=0, background_color='black'):
    # Perform inference on the frame
    results = model.predict(source=frame, save=False, conf=score_threshold)

    # Check if there are any detections
    if results[0].masks is None:
        # If no detections, return the original frame
        return frame

    # Get the segmentation masks and classes
    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # Initialize variables to find the largest object
    largest_area = 0
    largest_mask = None

    # Find the largest object among the detections
    for mask, cls in zip(masks, classes):
        if cls == person_class_id:
            # Calculate the area of the mask
            area = np.sum(mask)
            if area > largest_area:
                largest_area = area
                largest_mask = mask

    # Check if we found a valid mask
    if largest_mask is None:
        # If no valid mask, return the original frame
        return frame

    # Ensure largest_mask is binary (0 or 255)
    largest_mask = (largest_mask * 255).astype(np.uint8)

    # Resize the mask to match the frame size if needed
    if largest_mask.shape[:2] != frame.shape[:2]:
        largest_mask = cv2.resize(largest_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a 3-channel alpha mask for the largest object
    alpha_mask = cv2.merge([largest_mask, largest_mask, largest_mask])

    # Determine the background based on user input
    if background_color == 'white':
        background = np.ones_like(frame) * 255  # All white background
    else:
        background = np.zeros_like(frame)  # All black background

    # Combine the isolated largest object with the background using the mask
    object_isolated = cv2.bitwise_and(frame, alpha_mask)  # Isolate the largest object
    background_with_color = cv2.bitwise_and(background, cv2.bitwise_not(alpha_mask))  # Background as selected color

    # Combine the object with the background
    output_frame = cv2.add(object_isolated, background_with_color)

    return output_frame

def save_cropped_blurred_and_grayscale(input_image_path, output_image_path, seg_model_path, detection_model_path, score_threshold=0.3, background_color='black'):
    # Load the YOLOv8 models
    seg_model = YOLO(seg_model_path)
    detection_model = YOLO(detection_model_path)

    # Open the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to open image from {input_image_path}")
        return

    # Isolate the person in the current image
    person_isolated_frame = isolate_person_in_frame(image, seg_model, score_threshold, background_color=background_color)

    # Pass the isolated frame to the detection model
    detection_results = detection_model.predict(source=person_isolated_frame, save=False, conf=score_threshold)

    # Iterate through detection results and process the ROI (bounding box)
    for result in detection_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates

            # Extract region of interest (ROI)
            roi = person_isolated_frame[y1:y2, x1:x2]

            # Save the cropped ROI before any processing
            cv2.imwrite(f"{output_image_path}_cropped_roi.jpg", roi)

            # Apply Gaussian smoothing to the ROI
            blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)

            # Save the Gaussian blurred image before grayscale
            cv2.imwrite(f"{output_image_path}_blurred_roi.jpg", blurred_roi)

            # Convert blurred ROI to grayscale
            gray_blurred_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2GRAY)

            # Save the grayscale + Gaussian blurred image
            cv2.imwrite(f"{output_image_path}_grayscale_blurred_roi.jpg", gray_blurred_roi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save Cropped, Blurred, and Grayscale ROI Images Without Canny Edge Detection")
    parser.add_argument("--seg_model", type=str, default='yolov8n-seg.pt', help="Path to the segmentation model")
    parser.add_argument("--detection_model", type=str, required=True, help="Path to the trained detection model")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output images")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Score threshold for predictions")
    parser.add_argument("--background_color", type=str, default='black', choices=['black', 'white'], help="Background color (black or white)")

    args = parser.parse_args()

    save_cropped_blurred_and_grayscale(args.input, args.output, args.seg_model, args.detection_model, args.score_threshold, args.background_color)