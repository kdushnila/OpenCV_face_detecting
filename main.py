import cv2

# Load the pre-trained Haar cascade model for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read an image from a file
image = cv2.imread('faces/sea.jpg')
cv2.imshow('original', image)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image with face detections
cv2.imshow('Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
