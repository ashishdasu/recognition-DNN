# Ashish Dasu (adasu)
# CS5330 — Project 5: Recognition using Deep Networks
# Extension: real-time digit recognition from webcam feed.
# Captures video frames, isolates a region of interest, preprocesses it
# to match MNIST format, and displays the CNN's prediction live.

# import statements
import sys
import torch
import cv2
import numpy as np
from train_network import MyNetwork


# Preprocesses a cropped ROI from the webcam frame to match MNIST input format.
# Converts to grayscale, resizes to 28x28, inverts if needed (MNIST is white-on-black),
# normalizes with MNIST statistics, and returns a tensor ready for the model.
def preprocess_frame(roi):
    # convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi

    # resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # invert so digits are white on black (matching MNIST)
    inverted = 255 - resized

    # apply threshold to clean up noise
    _, threshed = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)

    # normalize to [0,1] then apply MNIST normalization
    normalized = threshed.astype(np.float32) / 255.0
    normalized = (normalized - 0.1307) / 0.3081

    # convert to tensor with batch and channel dimensions
    tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0)
    return tensor, threshed


# Minimum confidence threshold — predictions below this are suppressed to avoid
# showing spurious guesses on blank/noisy input (e.g. empty background reads as 8 at 27%).
CONFIDENCE_THRESHOLD = 60  # percent

# Draws the prediction overlay on the frame: ROI box, predicted digit, and confidence.
# Only shows the prediction when confidence exceeds the threshold; otherwise displays
# "No digit detected" to give clear feedback that the model is uncertain.
def draw_overlay(frame, roi_box, prediction, confidence, preprocessed):
    x, y, w, h = roi_box

    # draw ROI rectangle — green when confident, yellow when uncertain
    if confidence >= CONFIDENCE_THRESHOLD:
        box_color = (0, 255, 0)
        label = f'Prediction: {prediction} ({confidence:.0f}%)'
    else:
        box_color = (0, 200, 255)
        label = f'No digit detected ({confidence:.0f}%)'

    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    cv2.putText(frame, label, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, box_color, 2)

    # show the preprocessed 28x28 image in the corner (scaled up for visibility)
    preview = cv2.resize(preprocessed, (112, 112), interpolation=cv2.INTER_NEAREST)
    preview_color = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    frame[10:122, 10:122] = preview_color
    cv2.putText(frame, '28x28 input', (10, 136), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1)

    # instructions
    cv2.putText(frame, 'Hold a digit in the green box', (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, 'Press Q to quit', (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# main function — opens webcam, runs live digit recognition loop
def main(argv):
    # load trained model
    model = MyNetwork()
    model.load_state_dict(torch.load('mnist_model.pth', weights_only=True))
    model.eval()
    print('Loaded model from mnist_model.pth')

    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: could not open webcam')
        return

    print('Webcam opened. Hold a digit in front of the camera inside the green box.')
    print('Press S to save a screenshot, Q to quit.')

    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # define a centered square ROI (1/3 of frame height)
        roi_size = h // 3
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2
        roi_box = (roi_x, roi_y, roi_size, roi_size)

        # extract and preprocess the ROI
        roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
        tensor, preprocessed = preprocess_frame(roi)

        # classify
        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.exp(output)  # convert log-probs to probs
            confidence, prediction = probabilities.max(1)

        # draw results on frame
        draw_overlay(frame, roi_box, prediction.item(),
                     confidence.item() * 100, preprocessed)

        cv2.imshow('Live Digit Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_count += 1
            path = f'report_images/live_demo_{screenshot_count}.png'
            cv2.imwrite(path, frame)
            print(f'Screenshot saved to {path}')

    cap.release()
    cv2.destroyAllWindows()
    print('Done.')


if __name__ == "__main__":
    main(sys.argv)
