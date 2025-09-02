# Import required libraries
import cv2
import numpy as np
import tensorflow as tf

# ------------------- Load Model and Labels -------------------

model = tf.keras.models.load_model("mango_model.h5")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

IMG_SIZE = (224, 224)

# ------------------- Load and Preprocess Image -------------------

image_path = "data/mango_dataset_resized/Chaunsa (Black)/IMG_20210705_091828.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or unable to load.")
    exit()

img_resized = cv2.resize(image, IMG_SIZE)
img_array = np.expand_dims(img_resized / 255.0, axis=0)

# ------------------- Prediction -------------------

predictions = model.predict(img_array)
label_idx = np.argmax(predictions)
label = labels[label_idx]
confidence = predictions[0][label_idx] * 100

# ------------------- Display Result -------------------

# Prepare text
text = f"{label} ({confidence:.2f}%)"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1
color = (0, 255, 0)

# Get image dimensions
img_height, img_width = image.shape[:2]

# Get text size
(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

# Calculate centered position
x = (img_width - text_width) // 2
y = (img_height + text_height) // 2

# Draw text
cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

# Show result
cv2.imshow("Mango Classification Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
