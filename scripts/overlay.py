import cv2
import numpy as np

# Load the ground truth (segmented mask) and the prediction (outline) images
ground_truth = cv2.imread('convert/original/roi.png', cv2.IMREAD_GRAYSCALE)
prediction = cv2.imread('convert/outline/unetpp.tiff', cv2.IMREAD_GRAYSCALE)

# Resize images if necessary (ensure both are the same size)
if ground_truth.shape != prediction.shape:
    prediction = cv2.resize(prediction, (ground_truth.shape[1], ground_truth.shape[0]))

# Create a color version of the ground truth and prediction for better visualization
ground_truth_color = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2BGR)
#prediction_color = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

# Create a colored version of the prediction
prediction_color = np.zeros_like(ground_truth_color)
prediction_color[prediction > 0] = [255, 255, 255]  # Neon blue color for prediction

# Combine the ground truth and prediction
overlay = cv2.addWeighted(ground_truth_color, 1, prediction_color, 1, 0)

# Save the result
cv2.imwrite('convert/original/unetpp.png', overlay)
