from PIL import Image
import numpy as np
from collections import Counter

def is_color_within_range(color, lower_bound, upper_bound):
    """Check if a color is within a specified RGB range."""
    return all(lower <= val <= upper for val, lower, upper in zip(color, lower_bound, upper_bound))

def get_dominant_color(section, red_bounds, white_bounds):
    """
    Find the dominant colors in the section and classify as red or white.
    """
    # Flatten the section into a list of RGB tuples
    pixels = section.reshape(-1, 3)

    # Filter pixels to keep only red or white candidates
    red_pixels = [tuple(pixel) for pixel in pixels if is_color_within_range(pixel, *red_bounds)]
    white_pixels = [tuple(pixel) for pixel in pixels if is_color_within_range(pixel, *white_bounds)]

    # Count occurrences of red and white
    red_count = len(red_pixels)
    white_count = len(white_pixels)

    # Return the dominant color
    if red_count > white_count:
        return "red"
    elif white_count > red_count:
        return "white"
    else:
        return "none"

def detect_flag(image_path):
    """
    Determine if an image is the flag of Indonesia or Poland.
    """
    try:
        # Load the image and resize
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((300, 200))  # Resize for consistency
        img_array = np.array(img_resized)

        # Focus on central region (ignore edges and background noise)
        height, width, _ = img_array.shape
        center_y1, center_y2 = int(height * 0.1), int(height * 0.9)
        center_x1, center_x2 = int(width * 0.1), int(width * 0.9)
        center_region = img_array[center_y1:center_y2, center_x1:center_x2]

        # Split into top and bottom halves
        height_center = center_region.shape[0]
        top_half = center_region[:height_center // 2, :, :]
        bottom_half = center_region[height_center // 2:, :, :]

        # Define flexible thresholds for red and white
        red_bounds = (np.array([100, 0, 0]), np.array([255, 120, 120]))
        white_bounds = (np.array([180, 180, 180]), np.array([255, 255, 255]))

        # Detect dominant colors in each half
        top_color = get_dominant_color(top_half, red_bounds, white_bounds)
        bottom_color = get_dominant_color(bottom_half, red_bounds, white_bounds)

        # Flag classification
        if top_color == "red" and bottom_color == "white":
            return "Indonesia Flag"
        elif top_color == "white" and bottom_color == "red":
            return "Poland Flag"
        else:
            return "Neither Indonesia nor Poland Flag"

    except Exception as e:
        return f"Error: {e}"

# Test the function
if __name__ == "__main__":
    image_path = input("Enter the path to your flag image: ")
    result = detect_flag(image_path)
    print(result)
