import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to show images in a 2x4 grid
def show(image, n, m, i, title):
    plt.subplot(n, m, i)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')


# a. Capture an image and perform Grey Scaling
def capture_image():
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    if not cap.isOpened():
        print("Error: Camera not found!")
        return None

    print("Press 's' to capture the image...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture image.")
            break

        cv2.imshow('Camera', frame)  # Show live camera feed

        # Press 's' to save the captured frame
        if cv2.waitKey(1) & 0xFF == ord('s'):
            gray0image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print("Image captured!")
            break

    cap.release()
    cv2.destroyAllWindows()
    return gray0image


# b(i). Thresholding: Only black and white
def binary_threshold(image, threshold=128):
    _, thresh_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresh_image


# b(ii). 16 Grey Levels (Divide range into 8 regions)
def gray_levels_16(image):
    step = 256 // 16  # Divide 0-255 into 16 levels
    gray_16 = (image // step) * step
    return gray_16


# c. Apply Sobel filter and Canny Edge Detection
def sobel_and_canny(image):
    sobel = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)  # Sobel filter
    sobel = np.absolute(sobel)
    sobel = np.uint8(sobel)

    canny = cv2.Canny(image, 100, 200)  # Canny Edge Detector
    return sobel, canny


# d. Filter noise using Gaussian Blur
def gaussian_blur(image, kernel_size=5):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


# e. Sharpen the image using a sharpening kernel
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


# f. Convert RGB to BGR color channel
def rgb_to_bgr(image):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return bgr_image


# Main execution
if __name__ == "__main__":
    # Step 1: Capture and gray-scale an image
    gray_image = capture_image()
    if gray_image is None:
        exit()

    # Step 2: Perform all tasks and display images
    plt.figure(figsize=(10, 5))

    # Original Grey Scaled Image
    show(gray_image, 2, 4, 1, "Gray Scale")

    # Binary Thresholding
    binary_image = binary_threshold(gray_image)
    show(binary_image, 2, 4, 2, "Binary Thresholding")

    # 16 Grey Levels
    gray_16_image = gray_levels_16(gray_image)
    show(gray_16_image, 2, 4, 3, "16 Grey Levels")

    # Sobel and Canny Edge Detection
    sobel_image, canny_image = sobel_and_canny(gray_image)
    show(sobel_image, 2, 4, 4, "Sobel Filter")
    show(canny_image, 2, 4, 5, "Canny Edge")

    # Gaussian Blur
    blurred_image = gaussian_blur(gray_image)
    show(blurred_image, 2, 4, 6, "Gaussian Blur")

    # Sharpened Image
    sharpened_image = sharpen_image(blurred_image)
    show(sharpened_image, 2, 4, 7, "Sharpened Image")

    # RGB to BGR
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    bgr_image = rgb_to_bgr(rgb_image)
    show(bgr_image, 2, 4, 8, "RGB to BGR")

    # Show all the plots
    plt.tight_layout()
    plt.show()
