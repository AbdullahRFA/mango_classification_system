# Import TensorFlow for deep learning
import tensorflow as tf
# Import ImageDataGenerator for image preprocessing and augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Import json for saving label mappings (optional)
import json

# Path to resized mango dataset (each variety in its own folder)
DATA_DIR = "data/mango_dataset_resized"
# The target image size for the model (width, height)
IMG_SIZE = (224, 224)
# Number of images to process in each batch
BATCH_SIZE = 32
# Number of training passes over the dataset
EPOCHS = 30

# ---------------------------
# 1. DATA AUGMENTATION
# ---------------------------
# ImageDataGenerator applies real-time data augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values (0–255 → 0–1)
    validation_split=0.2,      # Split dataset: 80% training, 20% validation
    rotation_range=20,         # Randomly rotate images by up to 20 degrees
    width_shift_range=0.2,     # Randomly shift images horizontally by up to 20%
    height_shift_range=0.2,    # Randomly shift images vertically by up to 20%
    shear_range=0.2,           # Shear transformation for geometric distortion
    zoom_range=0.2,            # Randomly zoom into images
    horizontal_flip=True       # Randomly flip images horizontally
)

# ---------------------------
# 2. TRAINING DATA GENERATOR
# ---------------------------
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,                  # Root folder containing subfolders for each class
    target_size=IMG_SIZE,      # Resize images to 224x224 for MobileNetV2
    batch_size=BATCH_SIZE,     # Number of images per batch
    class_mode='categorical',  # Multi-class classification (one-hot encoding)
    subset='training'          # Use only the training subset
)

# ---------------------------
# 3. VALIDATION DATA GENERATOR
# ---------------------------
val_generator = train_datagen.flow_from_directory(
    DATA_DIR,                  # Same dataset directory
    target_size=IMG_SIZE,      # Same image size as training
    batch_size=BATCH_SIZE,     # Same batch size
    class_mode='categorical',  # Multi-class classification
    subset='validation'        # Use only the validation subset
)

# ---------------------------
# 4. SAVE CLASS LABELS TO FILE
# ---------------------------
# Reverse dictionary: index → class name
labels = {v: k for k, v in train_generator.class_indices.items()}
# Write class names to a text file for later use
with open("labels.txt", "w") as f:
    for idx in range(len(labels)):
        f.write(labels[idx] + "\n")

# ---------------------------
# 5. LOAD PRE-TRAINED MOBILENETV2
# ---------------------------
# MobileNetV2 is a lightweight CNN trained on ImageNet (transfer learning)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),   # Image size + RGB channels
    include_top=False,             # Do not include the final classification layer
    weights='imagenet'             # Load weights pre-trained on ImageNet
)
# Freeze the base model to keep pre-trained weights fixed
base_model.trainable = False

# ---------------------------
# 6. ADD CUSTOM CLASSIFICATION LAYERS
# ---------------------------
model = tf.keras.Sequential([
    base_model,                        # Base CNN model for feature extraction
    tf.keras.layers.GlobalAveragePooling2D(),  # Convert features to 1D
    tf.keras.layers.Dense(256, activation='relu'),  # Fully connected layer
    tf.keras.layers.Dropout(0.3),       # Dropout for regularization
    tf.keras.layers.Dense(len(labels), activation='softmax')  # Output layer
])

# ---------------------------
# 7. COMPILE THE MODEL
# ---------------------------
model.compile(
    optimizer='adam',                  # Adam optimizer
    loss='categorical_crossentropy',    # Loss function for multi-class classification
    metrics=['accuracy']                # Track accuracy during training
)

# ---------------------------
# 8. TRAIN THE MODEL
# ---------------------------
history = model.fit(
    train_generator,                    # Training data
    validation_data=val_generator,      # Validation data
    epochs=EPOCHS                        # Number of passes over dataset
)

# ---------------------------
# 9. SAVE THE TRAINED MODEL
# ---------------------------
model.save("mango_model.h5")  # Save model to file
print("✅ Model saved as mango_model.h5")
