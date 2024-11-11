import cv2
import numpy as np
import argparse
import math
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

def compute_gradient(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian smoothing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Compute the gradients using Sobel operator
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    
    return grad_magnitude

def determine_hysteresis_thresholds(grad_magnitude):
    # Flatten the gradient magnitude to 1D array for easier calculation of thresholds
    flat_grad = grad_magnitude.flatten()
    
    # Calculate high threshold as, for example, the 90th percentile of gradient magnitudes
    high_threshold = np.percentile(flat_grad, 99)
    
    # Calculate low threshold as 40% of the high threshold
    low_threshold = 0.4 * high_threshold
    
    return low_threshold, high_threshold

def calculate_angle(x1, y1, x2, y2):
    # Calculate the difference in x and y coordinates between two points.
    adjacent = abs(x2 - x1)
    opposite = abs(y2 - y1)

    # Calculate the angle using the arctangent function.
    angle_radians = math.atan2(opposite, adjacent)

    # Return the angle in degrees
    return 90 - math.degrees(angle_radians)

def find_important_point(contours, orientation, is_neck):
    # Initialize min_x, min_y, max_x, max_y by iterating through all contour points
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Calculate min_x, min_y, max_x, max_y by iterating through contour points
    for contour in contours:
        for point in contour:
            x, y = point[0]
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    best_point = None
    best_importance = -float('inf')

    for contour in contours:
        for point in contour:
            x, y = point[0]

            # Normalize the x and y values
            normalized_x = (x - min_x) / (max_x - min_x) if max_x - min_x != 0 else 0
            normalized_y = (y - min_y) / (max_y - min_y) if max_y - min_y != 0 else 0

            # Calculate importance based on orientation and point type (neck or back)
            if orientation == 'left' and is_neck: # Top-right point
                importance = normalized_x + 2*(1 - normalized_y)
            elif orientation == 'right' and is_neck: # Top-left point
                importance = (1 - normalized_x) + 2*(1 - normalized_y)
            elif orientation == 'left' and not is_neck: # Bottom-right point
                importance = normalized_x + 2*normalized_y
            elif orientation == 'right' and not is_neck: # Bottom-left point
                importance = (1 - normalized_x) + 2*normalized_y

            # Update the best point based on importance
            if importance > best_importance:
                best_importance = importance
                best_point = (x, y)

    return best_point

def process_image(input_image_path, output_image_path, seg_model_path, detection_model_path, score_threshold=0.3, orientation='left', background_color='black'):
    # Load the YOLOv8 models
    seg_model = YOLO(seg_model_path)
    detection_model = YOLO(detection_model_path)

    # Open the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to open image from {input_image_path}")
        return

    frame_height, frame_width = image.shape[:2]

    # Isolate the person in the current image
    person_isolated_frame = isolate_person_in_frame(image, seg_model, score_threshold, background_color=background_color)

    # Pass the isolated frame to the detection model
    detection_results = detection_model.predict(source=person_isolated_frame, save=False, conf=score_threshold)

    # Apply Gaussian smoothing, grayscale, and Canny edge detection
    for result in detection_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)  # Get bounding box (used only for region of interest extraction)
            roi = person_isolated_frame[y1:y2, x1:x2]

            # Apply Gaussian smoothing with a kernel size of (5, 5)
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)

            # Compute gradient magnitude for the region of interest
            grad_magnitude = compute_gradient(roi)

            # Determine optimal hysteresis thresholds
            low_threshold, high_threshold = determine_hysteresis_thresholds(grad_magnitude)

            # Print thresholds for debugging
            print(f"Low threshold: {low_threshold}, High threshold: {high_threshold}")

            # Apply Canny edge detection using the scalar thresholds
            if low_threshold is not None and high_threshold is not None:
                edges = cv2.Canny(blurred, low_threshold, high_threshold)
            else:
                print("Error: Invalid threshold values!")

            # Apply Canny edge detection using dynamic thresholds
            # edges = cv2.Canny(blurred, low_threshold, high_threshold)

            # Find the coordinates of the contours
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Find the neck and back points using the new method
            neck_point = find_important_point(contours, orientation, is_neck=True)
            back_point = find_important_point(contours, orientation, is_neck=False)

            if neck_point and back_point and neck_point[1] < back_point[1] and \
            not(orientation == 'left' and neck_point[0] >= back_point[0]) and not (orientation == 'right' and neck_point[0] <= back_point[0]):
                angle = calculate_angle(neck_point[0], neck_point[1], back_point[0], back_point[1])

                # Convert edges to a 3-channel image
                edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

                # Place the processed ROI back into the frame
                person_isolated_frame[y1:y2, x1:x2] = edges_3channel

                # Draw lines and dots on a separate layer
                overlay = person_isolated_frame.copy()

                # Check if points are within frame dimensions
                if 0 <= neck_point[0] + x1 < frame_width and 0 <= neck_point[1] + y1 < frame_height:
                    # Draw circles at the neck and back points
                    cv2.circle(overlay, (neck_point[0] + x1, neck_point[1] + y1), 10, (0, 0, 255), -1)
                if 0 <= back_point[0] + x1 < frame_width and 0 <= back_point[1] + y1 < frame_height:
                    cv2.circle(overlay, (back_point[0] + x1, back_point[1] + y1), 10, (0, 0, 255), -1)

                # Draw a vertical line from the neck point to the height of the back point
                cv2.line(overlay, (neck_point[0] + x1, neck_point[1] + y1), (neck_point[0] + x1, back_point[1] + y1), (0, 0, 255), 2)

                # Link the neck and back points using a red line
                cv2.line(overlay, (neck_point[0] + x1, neck_point[1] + y1), (back_point[0] + x1, back_point[1] + y1), (0, 0, 255), 2)

                # Overlay the angle in the middle at the top of the frame
                cv2.putText(overlay, f"{round(angle, 2)} degrees", (frame_width // 2 - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 8)

                # Blend the overlay with the original frame
                person_isolated_frame = cv2.addWeighted(overlay, 1.0, person_isolated_frame, 0.5, 0)

    # Save the processed image
    cv2.imwrite(output_image_path, person_isolated_frame)
    print(f"Output image saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolate person in image using YOLOv8 and process with another YOLOv8 model")
    parser.add_argument("--seg_model", type=str, default='yolov8n-seg.pt', help="Path to the segmentation model")
    parser.add_argument("--detection_model", type=str, required=True, help="Path to the trained detection model")
    parser.add_argument("--score_threshold", type=float, default=0.3, help="Score threshold for predictions")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, required=True, help="Path to the output image")
    parser.add_argument("--orientation", type=str, default='left', choices=['left', 'right'], help="Orientation of the person (facing left or right)")
    parser.add_argument("--background_color", type=str, default='black', choices=['black', 'white'], help="Background color (black or white)")

    args = parser.parse_args()

    process_image(args.input, args.output, args.seg_model, args.detection_model, args.score_threshold, args.orientation, args.background_color)