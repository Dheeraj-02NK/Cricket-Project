import cv2
import numpy as np

# Define ball color (replace with your ball's color in HSV)
lower_red = (170, 50, 50)
upper_red = (180, 255, 255)

# Define track point list and color

track_color = (0, 255, 0)  # Green for track

def track_ball():
  # Open the video capture
  cap = cv2.VideoCapture(0)

  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame captured successfully
    if not ret:
      print("Error: Failed to capture frame from video stream.")
      break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the ball color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Erode and Dilate the mask to reduce noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)[-2]

    # Track ball logic
    if len(cnts) > 0:
      # Find the largest contour
      c = max(cnts, key=cv2.contourArea)

      # Find the center of the contour (ball)
      M = cv2.moments(c)
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])

      track_points = []

      # Update track points list
      if len(track_points) > 10:  # Limit track points to avoid clutter
        track_points.pop(0)  # Remove oldest point
      track_points.append((cX, cY))  # Add new point

      # Draw track points
      for point in track_points:
        cv2.circle(frame, point, 2, track_color, -1)

      # Draw a circle around the ball and center
      cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)  # Green contour
      cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)  # Red center

      # Display the frame with the tracked ball
      cv2.imshow("Ball Tracking", frame)

    else:
      # Clear track points if no ball found
      track_points = []

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  # Release capture and close all windows
  cap.release()
  cv2.destroyAllWindows()

# Call the function
track_ball()
