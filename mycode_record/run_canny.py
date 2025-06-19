import cv2
from PIL import Image

# Load your uploaded image
image = cv2.imread("dreambird.jpg")

# Generate edge map using Canny filter
edges = cv2.Canny(image, 100, 200)

# Save edge map as image for ControlNet
Image.fromarray(edges).save("canny_map.png")

