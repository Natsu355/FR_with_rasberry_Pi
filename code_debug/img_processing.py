from picamera2 import Picamera2
import cv2
import time

# Configure the camera
camera = Picamera2()
config = camera.create_preview_configuration()
config['size'] = (1920, 1080)  # Adjust resolution as needed
config['format'] = 'BGR888'
camera.configure(config)
camera.start_preview()

# Capture a frame
frame = camera.capture_frame()

# Convert the frame to OpenCV format
image = cv2.imdecode(frame, cv2.IMREAD_COLOR)

# Process the image using OpenCV


# Display the processed image
cv2.imshow("Processed Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()