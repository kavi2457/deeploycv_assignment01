import cv2
import matplotlib.pyplot as plt

# Define the 'show' function
def show(name, n, m, i, title):
    plt.subplot(n, m, i)
    plt.imshow(name, cmap='gray')
    plt.title(title)
    plt.axis('off')

# Load images using cv2.imread()
image1 = cv2.imread('airballoon.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('houses.jpg', cv2.IMREAD_GRAYSCALE) 

# Check if images are loaded correctly
if image1 is None or image2 is None:
    print("Error: Check the image file paths!")
else:
    # Initialize the figure
    plt.figure(figsize=(8, 4))

    # Use the 'show' function to display images
    show(image1, 1, 2, 1, "Image 1")
    show(image2, 1, 2, 2, "Image 2")

    # Show the plot
    plt.tight_layout()
    plt.show()
