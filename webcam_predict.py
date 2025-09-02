# Import required libraries
import cv2              # OpenCV for accessing webcam and image processing
import numpy as np      # NumPy for numerical operations
import tensorflow as tf # TensorFlow for loading and using the trained model

# ------------------- Load Model and Labels -------------------

# Load the trained mango classification model from the file
model = tf.keras.models.load_model("mango_model.h5")

# Read the class labels from the text file and store them in a list
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]  # Remove newline characters

# Set the image size to match the training size of the model
IMG_SIZE = (224, 224)

# ------------------- Start Webcam -------------------

# Open the default webcam (0 is the default camera index)
cap = cv2.VideoCapture(0)

# ------------------- Main Loop -------------------
while True:
    # Read a single frame from the webcam
    ret, frame = cap.read()

    # If frame not captured properly, exit loop
    if not ret:
        break

    # ------------------- Image Preprocessing -------------------

    # Resize the frame to match the input size of the model
    img = cv2.resize(frame, IMG_SIZE)

    # Normalize pixel values (0-255 → 0-1) and add batch dimension
    img_array = np.expand_dims(img / 255.0, axis=0)

    # ------------------- Prediction -------------------

    # Get prediction probabilities for each class
    predictions = model.predict(img_array)

    # Get the index of the class with highest probability
    label_idx = np.argmax(predictions)

    # Get the corresponding label name
    label = labels[label_idx]

    # Get the confidence percentage for the predicted class
    confidence = predictions[0][label_idx] * 100

    # ------------------- Display Prediction -------------------

    # Put text on the webcam frame showing the label and confidence
    cv2.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame in a window
    cv2.imshow("Mango Classifier", frame)

    # If 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------- Cleanup -------------------

# Release the webcam resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()