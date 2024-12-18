import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_low_pass_filter(image, kernel_size):
    """Apply Gaussian Blur as a low-pass filter."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_high_pass_filter(image, kernel_size, scale_factor=2):
    """Apply High-Pass filter and amplify edge intensity for better visibility."""
    low_pass = apply_low_pass_filter(image, kernel_size)
    high_pass = cv2.subtract(image, low_pass)
    high_pass = cv2.normalize(high_pass, None, 0, 255, cv2.NORM_MINMAX)  # Normalize
    high_pass = cv2.convertScaleAbs(high_pass, alpha=scale_factor)       # Amplify edges
    return high_pass

def hybrid_image(image1, image2, low_pass_size, high_pass_size):
    """Create a hybrid image by combining low-pass and high-pass filters."""
    # Resize image2 to match image1 dimensions
    image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

    # Apply filters
    low_pass = apply_low_pass_filter(gray1, low_pass_size)
    high_pass = apply_high_pass_filter(gray2, high_pass_size)

    # Combine both images
    hybrid = cv2.addWeighted(low_pass, 0.6, high_pass, 1.2, 0)
    return gray1, low_pass, gray2, high_pass, hybrid

# Load two input images
image1 = cv2.imread("../images/airballoon.jpg")  # Replace with actual path
image2 = cv2.imread("../images/houses.jpg")      # Replace with actual path

# Check if images are loaded
if image1 is None or image2 is None:
    print("Error: Could not load one or both images. Check the file paths and image files.")
    exit()

# Filter sizes
low_pass_kernel = 25
high_pass_kernel = 15

# Generate hybrid image and all intermediate results
original1, low_pass_img, original2, high_pass_img, hybrid_img = hybrid_image(
    image1, image2, low_pass_kernel, high_pass_kernel
)

# Display all five images
plt.figure(figsize=(15, 8))

plt.subplot(1, 5, 1)
plt.title("Original Image 1")
plt.imshow(original1, cmap='gray')

plt.subplot(1, 5, 2)
plt.title("Low-Pass Filter")
plt.imshow(low_pass_img, cmap='gray')

plt.subplot(1, 5, 3)
plt.title("Original Image 2")
plt.imshow(original2, cmap='gray')

plt.subplot(1, 5, 4)
plt.title("High-Pass Filter")
plt.imshow(high_pass_img, cmap='gray')

plt.subplot(1, 5, 5)
plt.title("Hybrid Image")
plt.imshow(hybrid_img, cmap='gray')

plt.tight_layout()
plt.show()

# Save the output images
cv2.imwrite("../images/low_pass_result.jpg", low_pass_img)
cv2.imwrite("../images/high_pass_result.jpg", high_pass_img)
cv2.imwrite("../images/hybrid_image_result.jpg", hybrid_img)
